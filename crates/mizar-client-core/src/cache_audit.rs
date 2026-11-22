//! Prompt Cache 续命实时状态 DTO 与 `Cache Re-billed` 浪费审计算法。
//!
//! 本模块保持纯同步与无运行时依赖，供 `mizar-cache-warming`、`mizar-client` 与 `mizar-tui-ui` 共用。

use mizar_llm_types::{CacheWarmingMode, ModelCostRates, TokenUsage};
use serde::{Deserialize, Serialize};

/// 判定为意外前缀缓存跌落的最小重计费 token 下限，规避分词边界或尾部少量 token 抖动噪音。
pub const MIN_CACHE_DROP_TOKENS: u64 = 128;

/// 单会话 Prompt Cache 后台保活的实时观测状态快照。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CacheWarmingSessionStatus {
    pub mode: CacheWarmingMode,
    pub active: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_warm_in_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_warm_at_ms: Option<u128>,
    pub warm_count: usize,
    pub total_warm_cost_usd: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_warm_cost_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_miss_cost_usd: Option<f64>,
}

/// 基于会话用量账本统计出的意外缓存失效（`Cache Re-billed`）与保活开销汇总。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CacheMissAudit {
    pub cache_rebilled_tokens: u64,
    pub cache_rebilled_cost_usd: f64,
    pub miss_events: usize,
    pub warm_events: usize,
    pub total_warm_cost_usd: f64,
}

/// 从持久化会话日志投影出的精简审计采样点。
#[derive(Debug, Clone, PartialEq)]
pub enum UsageAuditSample {
    Assistant {
        usage: TokenUsage,
        rates: Option<ModelCostRates>,
    },
    CacheWarm {
        usage: TokenUsage,
        rates: Option<ModelCostRates>,
    },
    Compaction,
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

/// 按模型单价（含 `tiers` 阶梯定价，单位：美元 / 1M tokens）计算单次用量费用。
pub fn calculate_usage_cost(rates: &ModelCostRates, usage: &TokenUsage) -> f64 {
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

/// 沿会话 `Assistant`、`Usage(kind="cache_warm")` 与 `Compaction` 样本检测意外前缀缓存失效。
///
/// - `Compaction` 主动改写历史前缀，作为合法边界重置缓存锚点，不计为 `Cache Re-billed`；
/// - `CacheWarm` 既累计保活次数与保活花费，也同步刷新前缀缓存锚点；
/// - 当非压缩跨步出现 `prev_cached > curr_cache_read` 且重计费 token 数达到阈值时，
///   累加被重新计费的 token 与超出 `cache_read` 单价的额外开销。
pub fn compute_cache_miss_audit(
    samples: &[UsageAuditSample],
    fallback_rates: Option<&ModelCostRates>,
) -> CacheMissAudit {
    const PER_MILLION: f64 = 1_000_000.0;

    let mut audit = CacheMissAudit::default();
    let mut anchor_cached_tokens: Option<u64> = None;

    for sample in samples {
        let (usage, sample_rates, is_warm) = match sample {
            UsageAuditSample::Compaction => {
                anchor_cached_tokens = None;
                continue;
            }
            UsageAuditSample::Assistant { usage, rates } => (usage, rates.as_ref(), false),
            UsageAuditSample::CacheWarm { usage, rates } => (usage, rates.as_ref(), true),
        };

        let eff_rates = sample_rates.or(fallback_rates);
        if is_warm {
            audit.warm_events += 1;
            if let Some(rates) = eff_rates {
                audit.total_warm_cost_usd += calculate_usage_cost(rates, usage);
            }
        }

        let curr_read = usage.cache_read_tokens.unwrap_or(0);
        let curr_write = usage.cache_write_tokens.unwrap_or(0);
        let curr_input = usage.input_tokens;

        if let Some(prev_cached) = anchor_cached_tokens {
            let dropped = prev_cached.saturating_sub(curr_read);
            let uncached_curr = curr_write.saturating_add(curr_input);
            let rebilled = dropped.min(uncached_curr);

            if rebilled >= MIN_CACHE_DROP_TOKENS {
                audit.miss_events += 1;
                audit.cache_rebilled_tokens = audit.cache_rebilled_tokens.saturating_add(rebilled);

                if let Some(rates) = eff_rates {
                    let (input_rate, _, cache_read_rate, cache_write_rate) =
                        effective_rates(rates, usage.prompt_tokens());
                    let write_rebilled = rebilled.min(curr_write);
                    let input_rebilled = rebilled.saturating_sub(write_rebilled);
                    let extra_cost = (write_rebilled as f64
                        * (cache_write_rate - cache_read_rate).max(0.0)
                        + input_rebilled as f64 * (input_rate - cache_read_rate).max(0.0))
                        / PER_MILLION;
                    audit.cache_rebilled_cost_usd += extra_cost;
                }
            }
        }

        let curr_cached = curr_read.saturating_add(curr_write);
        anchor_cached_tokens = (curr_cached > 0).then_some(curr_cached);
    }

    audit
}

#[cfg(test)]
mod tests {
    use super::*;
    use mizar_llm_types::ModelCostTier;

    fn anthropic_rates() -> ModelCostRates {
        ModelCostRates {
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
        }
    }

    #[test]
    fn compute_cache_miss_audit_detects_unexpected_drops_ignores_compaction_and_tracks_warm() {
        let rates = anthropic_rates();
        let samples = vec![
            // 1. 首次建立 100k 缓存
            UsageAuditSample::Assistant {
                usage: TokenUsage {
                    input_tokens: 500,
                    cache_write_tokens: Some(100_000),
                    output_tokens: 200,
                    ..Default::default()
                },
                rates: None,
            },
            // 2. 正常保活刷新（命中 100k）
            UsageAuditSample::CacheWarm {
                usage: TokenUsage {
                    cache_read_tokens: Some(100_000),
                    output_tokens: 1,
                    ..Default::default()
                },
                rates: None,
            },
            // 3. 正常继续对话（命中 100k，追加 2k 写入）-> 锚点升级为 102k
            UsageAuditSample::Assistant {
                usage: TokenUsage {
                    input_tokens: 100,
                    cache_read_tokens: Some(100_000),
                    cache_write_tokens: Some(2_000),
                    output_tokens: 300,
                    ..Default::default()
                },
                rates: None,
            },
            // 4. 发生 Compaction 边界：清空锚点，压缩后首次重写不应算 Cache Re-billed
            UsageAuditSample::Compaction,
            UsageAuditSample::Assistant {
                usage: TokenUsage {
                    input_tokens: 300,
                    cache_write_tokens: Some(40_000),
                    output_tokens: 250,
                    ..Default::default()
                },
                rates: None,
            },
            // 5. 缓存意外失效：上轮锚点 40k，本轮 cache_read=0 且重写 42k -> Re-billed 40k！
            UsageAuditSample::Assistant {
                usage: TokenUsage {
                    input_tokens: 400,
                    cache_read_tokens: Some(0),
                    cache_write_tokens: Some(42_000),
                    output_tokens: 180,
                    ..Default::default()
                },
                rates: None,
            },
        ];

        let audit = compute_cache_miss_audit(&samples, Some(&rates));
        assert_eq!(audit.warm_events, 1);
        assert_eq!(audit.miss_events, 1);
        assert_eq!(audit.cache_rebilled_tokens, 40_000);
        // 保活费用：0.1 * 0.3 + 1e-6 * 15.0 = 0.030015
        assert!(
            (audit.total_warm_cost_usd - 0.030015).abs() < 1e-9,
            "{}",
            audit.total_warm_cost_usd
        );
        // 额外 Re-billed 费用：40k * (3.75 - 0.30) / 1M = 0.138
        assert!(
            (audit.cache_rebilled_cost_usd - 0.138).abs() < 1e-9,
            "{}",
            audit.cache_rebilled_cost_usd
        );
    }

    #[test]
    fn compute_cache_miss_audit_detects_openai_input_rebill_and_uses_tiers() {
        let rates = anthropic_rates();
        let samples = vec![
            // 建立 >200k 的 OpenAI 风格缓存命中（无 cache_write_tokens）
            UsageAuditSample::Assistant {
                usage: TokenUsage {
                    input_tokens: 1_000,
                    cache_read_tokens: Some(250_000),
                    output_tokens: 100,
                    ..Default::default()
                },
                rates: Some(rates.clone()),
            },
            // 意外全部掉缓存，改为按 input 重计费（prompt_tokens = 252_000 命中 >200k 阶梯：input=6.0, cache_read=0.6）
            UsageAuditSample::Assistant {
                usage: TokenUsage {
                    input_tokens: 252_000,
                    cache_read_tokens: Some(0),
                    output_tokens: 100,
                    ..Default::default()
                },
                rates: Some(rates),
            },
        ];

        let audit = compute_cache_miss_audit(&samples, None);
        assert_eq!(audit.miss_events, 1);
        assert_eq!(audit.cache_rebilled_tokens, 250_000);
        // 250_000 * (6.0 - 0.6) / 1M = 1.35
        assert!(
            (audit.cache_rebilled_cost_usd - 1.35).abs() < 1e-9,
            "{}",
            audit.cache_rebilled_cost_usd
        );
    }

    #[test]
    fn status_and_audit_dtos_roundtrip_serde_defaults() {
        let status = CacheWarmingSessionStatus {
            mode: CacheWarmingMode::Streaming,
            active: true,
            next_warm_in_secs: Some(270),
            next_warm_at_ms: Some(1_700_000_000_000),
            warm_count: 2,
            total_warm_cost_usd: 0.06,
            last_stop_reason: None,
            last_warm_cost_usd: Some(0.03),
            last_miss_cost_usd: Some(0.35),
        };
        let encoded = serde_json::to_string(&status).unwrap();
        let decoded: CacheWarmingSessionStatus = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, status);
    }
}
