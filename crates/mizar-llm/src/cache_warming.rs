//! Prompt Cache 续命的无状态经济测算与安全门禁工具函数。
//!
//! 将缓存 TTL 推导、阶梯计价、休眠防错峰截止线与期望净收益决策封装为纯函数，
//! 供上层 `mizar-cache-warming` 与会话浪费审计复用。

use crate::types::{
    CacheRetention, GenerateOptions, ModelCostRates, ModelInfo, ReasoningEffort, TokenUsage,
    resolve_cache_retention,
};

/// `idle` 阶段默认认为用户继续提问的先验概率。
pub const CACHE_WARMING_IDLE_CONTINUATION_PROBABILITY: f64 = 0.15;

/// 触发一次保活请求所要求的最低期望净收益（美元）。
pub const CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS: f64 = 0.05;

/// `streaming`（活跃生成中）允许最长持续保活上限（毫秒）：60 分钟。
pub const CACHE_WARMING_MAX_AGE_MS: u64 = 60 * 60 * 1000;

/// `idle`（回合结束后空闲）允许最长持续保活上限（毫秒）：30 分钟。
pub const CACHE_WARMING_MAX_IDLE_AGE_MS: u64 = 30 * 60 * 1000;

/// 预热调度所需的最小 TTL 安全缓冲（毫秒）：10 秒。
pub const CACHE_WARMING_MIN_TTL_MS: u64 = 10_000;

/// 缓存保活所处阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheWarmingPhase {
    Streaming,
    Idle,
}

/// 经济测算结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheWarmingDecision {
    Warm,
    Stop,
}

/// 单次缓存续命期望收益评估明细。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheWarmingEvaluation {
    pub decision: CacheWarmingDecision,
    pub warm_cost: f64,
    pub miss_cost: f64,
    pub expected_savings: f64,
}

fn effective_rates(rates: &ModelCostRates, prompt_tokens: u64) -> (f64, f64, f64, f64) {
    let matched_tier = rates
        .tiers
        .iter()
        .filter(|tier| prompt_tokens > tier.input_tokens_above)
        .max_by_key(|tier| tier.input_tokens_above);
    match matched_tier {
        Some(tier) => (tier.input, tier.output, tier.cache_read, tier.cache_write),
        None => (
            rates.input,
            rates.output,
            rates.cache_read,
            rates.cache_write,
        ),
    }
}

/// 按模型单价（含 `tiers` 阶梯定价，单位：美元 / 1M tokens）计算单次请求费用。
pub fn calculate_cost(rates: &ModelCostRates, usage: &TokenUsage) -> f64 {
    const PER_MILLION: f64 = 1_000_000.0;
    let prompt_tokens = usage.prompt_tokens();
    let (input_rate, output_rate, cache_read_rate, cache_write_rate) =
        effective_rates(rates, prompt_tokens);

    (usage.input_tokens as f64 * input_rate
        + usage.output_tokens as f64 * output_rate
        + usage.cache_read_tokens.unwrap_or(0) as f64 * cache_read_rate
        + usage.cache_write_tokens.unwrap_or(0) as f64 * cache_write_rate)
        / PER_MILLION
}

/// 根据模型声明的 `prompt_cache` 与有效 `CacheRetention` 解析服务端缓存 TTL（毫秒）。
pub fn get_prompt_cache_ttl_ms(
    model: &ModelInfo,
    retention: Option<CacheRetention>,
) -> Option<u64> {
    let cache = model.prompt_cache.as_ref()?;
    let secs = match resolve_cache_retention(retention) {
        CacheRetention::None => return None,
        CacheRetention::Short => cache.short?,
        CacheRetention::Long => cache.long?,
    };
    Some(secs.saturating_mul(1000))
}

/// 计算距离下次预热触发的延迟（毫秒）：取 `ttl * 0.9` 与 `ttl - 10s` 中的较小值。
pub fn get_cache_warming_delay_ms(ttl_ms: u64) -> Option<u64> {
    if ttl_ms <= CACHE_WARMING_MIN_TTL_MS {
        return None;
    }
    let scaled = (ttl_ms as f64 * 0.9).floor() as u64;
    Some(scaled.min(ttl_ms - CACHE_WARMING_MIN_TTL_MS))
}

/// 计算允许触发预热的硬截止偏移（位于 `delay_ms` 与 `ttl_ms` 中点，防止休眠唤醒后误打入已过期窗口）。
pub fn get_cache_warming_refresh_deadline_offset_ms(ttl_ms: u64, delay_ms: u64) -> u64 {
    delay_ms.saturating_add(ttl_ms.saturating_sub(delay_ms) / 2)
}

/// 判断给定请求在协议与思考参数约束下是否可安全以 `max_tokens=1` 原样重放预热。
///
/// 旧版 Anthropic budget-based thinking 在 `max_tokens=1` 时会被适配层改写预算导致缓存键失配，
/// 仅当开启 `force_adaptive_thinking` 或未启用思考时才可安全重放。
pub fn is_cache_warming_replayable(
    protocol: &str,
    model: &ModelInfo,
    options: &GenerateOptions,
) -> bool {
    if protocol == "anthropic-messages"
        && options
            .reasoning_effort
            .is_some_and(|effort| effort != ReasoningEffort::Off)
        && !model.force_adaptive_thinking
    {
        return false;
    }
    true
}

/// 测算一次续命保活的成本与期望净收益（满足 `expected_savings >= $0.05` 时放行）。
pub fn evaluate_cache_warming(
    model: &ModelInfo,
    prompt_tokens: u64,
    phase: CacheWarmingPhase,
) -> CacheWarmingEvaluation {
    let Some(rates) = model.cost.as_ref() else {
        return CacheWarmingEvaluation {
            decision: CacheWarmingDecision::Stop,
            warm_cost: 0.0,
            miss_cost: 0.0,
            expected_savings: 0.0,
        };
    };

    let warm_cost = calculate_cost(
        rates,
        &TokenUsage {
            output_tokens: 1,
            cache_read_tokens: Some(prompt_tokens),
            ..Default::default()
        },
    );

    let (_, _, _, cache_write_rate) = effective_rates(rates, prompt_tokens);
    let miss_full_cost = if cache_write_rate > 0.0 {
        calculate_cost(
            rates,
            &TokenUsage {
                cache_write_tokens: Some(prompt_tokens),
                ..Default::default()
            },
        )
    } else {
        calculate_cost(
            rates,
            &TokenUsage {
                input_tokens: prompt_tokens,
                ..Default::default()
            },
        )
    };
    let read_only_cost = calculate_cost(
        rates,
        &TokenUsage {
            cache_read_tokens: Some(prompt_tokens),
            ..Default::default()
        },
    );
    let miss_cost = miss_full_cost - read_only_cost;

    let probability = match phase {
        CacheWarmingPhase::Streaming => 1.0,
        CacheWarmingPhase::Idle => CACHE_WARMING_IDLE_CONTINUATION_PROBABILITY,
    };
    let expected_savings = probability * miss_cost - warm_cost;
    let decision = if expected_savings >= CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS {
        CacheWarmingDecision::Warm
    } else {
        CacheWarmingDecision::Stop
    };

    CacheWarmingEvaluation {
        decision,
        warm_cost,
        miss_cost,
        expected_savings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelCostTier, PromptCacheLifetime};

    fn sample_anthropic_model() -> ModelInfo {
        ModelInfo {
            id: "claude-sonnet".into(),
            name: "Claude Sonnet".into(),
            cost: Some(ModelCostRates {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
                tiers: vec![ModelCostTier {
                    input_tokens_above: 200_000,
                    input: 6.0,
                    output: 22.5,
                    cache_read: 0.6,
                    cache_write: 7.5,
                }],
            }),
            prompt_cache: Some(PromptCacheLifetime {
                short: Some(300),
                long: Some(3600),
            }),
            force_adaptive_thinking: false,
            ..Default::default()
        }
    }

    #[test]
    fn calculate_cost_selects_highest_matching_tier_by_prompt_tokens() {
        let mut rates = sample_anthropic_model().cost.unwrap();
        rates.tiers.push(ModelCostTier {
            input_tokens_above: 500_000,
            input: 9.0,
            output: 30.0,
            cache_read: 0.9,
            cache_write: 10.0,
        });

        // 恰好 200_000 不触发 > 200_000 阶梯
        let base_usage = TokenUsage {
            input_tokens: 100_000,
            cache_read_tokens: Some(100_000),
            output_tokens: 1_000,
            ..Default::default()
        };
        let base_cost = calculate_cost(&rates, &base_usage);
        // 0.1 * 3.0 + 0.1 * 0.3 + 0.001 * 15.0 = 0.345
        assert!((base_cost - 0.345).abs() < 1e-9, "{base_cost}");

        // prompt_tokens = 250_000 -> 命中 > 200_000 阶梯
        let tier1_usage = TokenUsage {
            input_tokens: 50_000,
            cache_read_tokens: Some(150_000),
            cache_write_tokens: Some(50_000),
            output_tokens: 2_000,
            ..Default::default()
        };
        let tier1_cost = calculate_cost(&rates, &tier1_usage);
        // 0.05 * 6.0 + 0.15 * 0.6 + 0.05 * 7.5 + 0.002 * 22.5 = 0.3 + 0.09 + 0.375 + 0.045 = 0.81
        assert!((tier1_cost - 0.81).abs() < 1e-9, "{tier1_cost}");

        // prompt_tokens = 600_000 -> 命中最高匹配阶梯 (> 500_000)
        let tier2_usage = TokenUsage {
            cache_read_tokens: Some(600_000),
            output_tokens: 1_000,
            ..Default::default()
        };
        let tier2_cost = calculate_cost(&rates, &tier2_usage);
        // 0.6 * 0.9 + 0.001 * 30.0 = 0.54 + 0.03 = 0.57
        assert!((tier2_cost - 0.57).abs() < 1e-9, "{tier2_cost}");
    }

    #[test]
    fn prompt_cache_ttl_delay_and_deadline_math() {
        let model = sample_anthropic_model();
        assert_eq!(
            get_prompt_cache_ttl_ms(&model, Some(CacheRetention::Short)),
            Some(300_000)
        );
        assert_eq!(
            get_prompt_cache_ttl_ms(&model, Some(CacheRetention::Long)),
            Some(3_600_000)
        );
        assert_eq!(
            get_prompt_cache_ttl_ms(&model, Some(CacheRetention::None)),
            None
        );

        // 300s TTL -> 270s 延迟，285s 截止偏移
        let delay = get_cache_warming_delay_ms(300_000).unwrap();
        assert_eq!(delay, 270_000);
        assert_eq!(
            get_cache_warming_refresh_deadline_offset_ms(300_000, delay),
            285_000
        );

        // 短 TTL（30s）优先保留 10s 缓冲（20s < 27s）
        assert_eq!(get_cache_warming_delay_ms(30_000), Some(20_000));
        // <= 10s 不安排预热
        assert_eq!(get_cache_warming_delay_ms(10_000), None);
        assert_eq!(get_cache_warming_delay_ms(5_000), None);
    }

    #[test]
    fn replayability_gate_blocks_anthropic_budget_thinking_unless_adaptive() {
        let mut model = sample_anthropic_model();
        let with_thinking = GenerateOptions {
            reasoning_effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        };
        let off_thinking = GenerateOptions {
            reasoning_effort: Some(ReasoningEffort::Off),
            ..Default::default()
        };
        let no_thinking = GenerateOptions::default();

        // Anthropic 非 adaptive + 开启思考 -> 拒绝重放
        assert!(!is_cache_warming_replayable(
            "anthropic-messages",
            &model,
            &with_thinking
        ));
        // 关闭或未设置思考 -> 允许重放
        assert!(is_cache_warming_replayable(
            "anthropic-messages",
            &model,
            &off_thinking
        ));
        assert!(is_cache_warming_replayable(
            "anthropic-messages",
            &model,
            &no_thinking
        ));
        // 其他协议 -> 允许重放
        assert!(is_cache_warming_replayable(
            "openai-responses",
            &model,
            &with_thinking
        ));

        // 强制 adaptive thinking -> 允许重放
        model.force_adaptive_thinking = true;
        assert!(is_cache_warming_replayable(
            "anthropic-messages",
            &model,
            &with_thinking
        ));
    }

    #[test]
    fn evaluate_cache_warming_enforces_five_cent_expected_savings_threshold() {
        let model = sample_anthropic_model();

        // 100k tokens: warm = 0.030015, miss = 0.345
        // Streaming (p=1.0): E = 0.314985 >= 0.05 -> Warm
        let streaming = evaluate_cache_warming(&model, 100_000, CacheWarmingPhase::Streaming);
        assert_eq!(streaming.decision, CacheWarmingDecision::Warm);
        assert!(streaming.expected_savings >= CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS);

        // Idle (p=0.15): E = 0.15*0.345 - 0.030015 = 0.021735 < 0.05 -> Stop
        let idle_small = evaluate_cache_warming(&model, 100_000, CacheWarmingPhase::Idle);
        assert_eq!(idle_small.decision, CacheWarmingDecision::Stop);
        assert!(idle_small.expected_savings < CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS);

        // 300k tokens (命中 > 200k 阶梯：read=0.6, write=7.5, output=22.5):
        // warm = 0.3*0.6 + 22.5e-6 = 0.1800225, miss = 0.3*(7.5-0.6) = 2.07
        // Idle (p=0.15): E = 0.15*2.07 - 0.1800225 = 0.1304775 >= 0.05 -> Warm
        let idle_large = evaluate_cache_warming(&model, 300_000, CacheWarmingPhase::Idle);
        assert_eq!(idle_large.decision, CacheWarmingDecision::Warm);
        assert!(idle_large.expected_savings >= CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS);

        // cache_write = 0 时（如 OpenAI/Gemini）回退使用 input 单价算 miss_cost
        let no_write_fee_model = ModelInfo {
            id: "gpt".into(),
            name: "GPT".into(),
            cost: Some(ModelCostRates {
                input: 2.0,
                output: 8.0,
                cache_read: 0.2,
                cache_write: 0.0,
                tiers: Vec::new(),
            }),
            ..Default::default()
        };
        let eval_fallback =
            evaluate_cache_warming(&no_write_fee_model, 100_000, CacheWarmingPhase::Streaming);
        // warm = 0.020008, miss = 0.2 - 0.02 = 0.18 -> E = 0.159992 >= 0.05 -> Warm
        assert_eq!(eval_fallback.decision, CacheWarmingDecision::Warm);
        assert!((eval_fallback.miss_cost - 0.18).abs() < 1e-9);

        // 无 cost 元数据时始终 Stop
        let no_cost = ModelInfo::default();
        assert_eq!(
            evaluate_cache_warming(&no_cost, 1_000_000, CacheWarmingPhase::Streaming).decision,
            CacheWarmingDecision::Stop
        );
    }
}
