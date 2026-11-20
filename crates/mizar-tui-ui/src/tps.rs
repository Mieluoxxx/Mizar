//! 解码吞吐（tok/s）折叠账本。
//!
//! 对应 TS 原版 `dsh-TUI/src/channel/projection.ts` 的 TPS 账本
//! （`tpsTurn` / `tpsStep` / `tpsTurnDecodeMs` / `tpsTurnDecodeTokens`）
//! 与 `src/channel/usage.ts` 的脚本加权估算。
//!
//! 口径：一轮（turn）内跨步累加，`Σ(真实 output tokens) / Σ(首 token → 步结束)`。
//! 工具执行时间和每次请求的 TTFT 都不进分母；分子优先用 provider 上报的
//! `usage.output_tokens`，缺失时才回退到字符估算。

use mizar_llm_types::TokenUsage;

/// ASCII 约 4 字符 / token。
const CHARS_PER_TOKEN_ASCII: f64 = 4.0;
/// CJK 汉字、假名、谚文、注音、彝文、CJK 标点与全角形式约 1.4 字符 / token。
const CHARS_PER_TOKEN_CJK: f64 = 1.4;
/// 其余文字（重音拉丁、西里尔、希腊、阿拉伯、希伯来、泰、天城文、emoji）按 2 字符 / token。
const CHARS_PER_TOKEN_OTHER: f64 = 2.0;

/// 短于该跨度的投递不是可靠的解码样本。
pub const MIN_TPS_DECODE_MS: u128 = 500;

/// 估算一段文本的 token 数，**保留小数**。
///
/// 浮点累加才可加：逐 delta 取整会把若干个 2-3 字符的碎片各自抬成 1 token
/// （ASCII 因此虚高 4 倍），而旧的 `chars / 4` 又把 CJK 低估约 3 倍。
/// 取整只在读取时做一次。
pub fn estimate_tokens_fraction(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    if text.is_ascii() {
        return text.len() as f64 / CHARS_PER_TOKEN_ASCII;
    }
    let mut ascii = 0usize;
    let mut dense = 0usize;
    let mut other = 0usize;
    for ch in text.chars() {
        let cp = ch as u32;
        if cp < 0x80 {
            ascii += 1;
        } else if is_dense_script(cp) {
            dense += 1;
        } else {
            other += 1;
        }
    }
    ascii as f64 / CHARS_PER_TOKEN_ASCII
        + dense as f64 / CHARS_PER_TOKEN_CJK
        + other as f64 / CHARS_PER_TOKEN_OTHER
}

/// BPE 词表按每个码点一两个 token 打包的书写系统。emoji 刻意不算：
/// 它们不是东亚宽字符文本，而一个 ZWJ 组合是多个码点、可能被计多个 token。
fn is_dense_script(code_point: u32) -> bool {
    matches!(code_point,
        0x1100..=0x11ff      // Hangul Jamo
        | 0x2e80..=0x303f    // CJK 部首 … CJK 符号与标点
        | 0x3040..=0x33ff    // 假名、注音、谚文兼容字母、带圈/兼容 CJK
        | 0x3400..=0x4dbf    // CJK 统一汉字扩展 A
        | 0x4e00..=0x9fff    // CJK 统一汉字
        | 0xa000..=0xa4cf    // 彝文音节与部首
        | 0xa960..=0xa97f    // Hangul Jamo Extended-A
        | 0xac00..=0xd7af    // 谚文音节
        | 0xd7b0..=0xd7ff    // Hangul Jamo Extended-B
        | 0xf900..=0xfaff    // CJK 兼容汉字
        | 0xfe30..=0xfe4f    // CJK 兼容形式
        | 0xfe50..=0xfe6f    // 小写变体（CJK 标点变体）
        | 0xff00..=0xff60    // 全角形式
        | 0xff61..=0xffdc    // 半角假名、半角谚文、半角 CJK 标点
        | 0xffe0..=0xffe6    // 全角符号
        | 0x1f200..=0x1f2ff  // 带圈表意文字补充
        | 0x20000..=0x3fffd  // CJK 扩展 B–G
    )
}

/// 一次进行中的模型调用步：首个 token 的边界与流式估算量。
#[derive(Debug, Clone, Default)]
struct OpenStep {
    first_token_ms: Option<u128>,
    estimate: f64,
}

/// 一轮对话的解码吞吐折叠。
///
/// 所有时间都来自事件自带的时间戳，不用事件消费时刻：UI 主循环会把积压的
/// delta 一次性排空，用消费时刻会把一整批 delta 算作零耗时。
#[derive(Debug, Clone, Default)]
pub struct TpsLedger {
    rate: Option<f64>,
    /// 进入本轮前的读数：本轮没有可测量样本时恢复它，而不是留空或归零。
    rate_before_turn: Option<f64>,
    /// 已归入分母的步解码耗时（毫秒），跨步累加。
    decode_ms: u128,
    /// 已归入分子的输出 token 数，跨步累加。
    decode_tokens: f64,
    sampled: bool,
    open: Option<OpenStep>,
}

impl TpsLedger {
    /// 当前读数（tok/s）。没有可靠样本时为 `None`。
    pub fn rate(&self) -> Option<f64> {
        self.rate
    }

    /// 清空账本（切换会话时用）。
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 开启新一轮：跨步合计归零，先留住旧读数作为本轮无样本时的回退。
    pub fn on_turn_start(&mut self) {
        self.rate_before_turn = self.rate;
        self.decode_ms = 0;
        self.decode_tokens = 0.0;
        self.sampled = false;
        self.open = None;
    }

    /// 开启新的一步模型调用：解码跨度从本步的首个 token 重新起算，
    /// 但本轮的 `decode_ms` / `decode_tokens` 继续累加。
    pub fn on_step_start(&mut self) {
        self.open = Some(OpenStep::default());
    }

    /// 记录本步首个 token 的时间边界（同一步内重复到达只认第一次）。
    pub fn on_first_token(&mut self, ts_ms: u128) {
        if let Some(open) = &mut self.open {
            open.first_token_ms.get_or_insert(ts_ms);
        }
    }

    /// 喂入一个流式增量。首次增量自身也建立边界，以覆盖没有独立首 token 事件的后端。
    pub fn on_token_delta(&mut self, text: &str, ts_ms: u128) {
        let estimate = estimate_tokens_fraction(text);
        if estimate <= 0.0 {
            return;
        }
        let (elapsed, total_estimate) = {
            let Some(open) = &mut self.open else {
                return;
            };
            let first = *open.first_token_ms.get_or_insert(ts_ms);
            open.estimate += estimate;
            (ts_ms.saturating_sub(first), open.estimate)
        };
        if elapsed <= MIN_TPS_DECODE_MS {
            return;
        }
        self.publish(
            self.decode_ms + elapsed,
            self.decode_tokens + total_estimate.ceil(),
        );
    }

    /// 一步结束：落入本步的真实解码跨度，并用 provider 的 usage 定案。
    ///
    /// 只有同时具备首 token 与合法用量的步才贡献分子；有跨度但没有 token 数
    /// （隐藏生成、或后端不报用量且没有内容）时保留跨度而不采样零速率。
    pub fn on_step_end(&mut self, ts_ms: u128, usage: Option<TokenUsage>) {
        let Some(open) = self.open.take() else {
            return;
        };
        let Some(first) = open.first_token_ms else {
            return;
        };
        let elapsed = ts_ms.saturating_sub(first);
        if elapsed <= MIN_TPS_DECODE_MS {
            return;
        }
        let tokens = usage
            .map(|u| u.output_tokens as f64)
            .or_else(|| (open.estimate > 0.0).then(|| open.estimate.ceil()));
        self.decode_ms += elapsed;
        if let Some(tokens) = tokens {
            self.decode_tokens += tokens;
            self.sampled = true;
            self.publish(self.decode_ms, self.decode_tokens);
        }
    }

    /// 一轮结束：取整轮加权读数作为历史样本。
    ///
    /// 返回 `Some(rate)` 表示本轮有可测量样本；`None` 表示没有，
    /// 此时读数回退到进入本轮前的值。
    pub fn on_turn_end(&mut self) -> Option<f64> {
        self.open = None;
        if self.sampled && self.decode_ms > 0 {
            let rate = self.decode_tokens / (self.decode_ms as f64 / 1000.0);
            self.rate = Some(rate);
            return Some(rate);
        }
        self.rate = self.rate_before_turn;
        None
    }

    fn publish(&mut self, decode_ms: u128, tokens: f64) {
        if decode_ms == 0 {
            return;
        }
        let rate = tokens / (decode_ms as f64 / 1000.0);
        if rate.is_finite() {
            self.rate = Some(rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(output_tokens: u64) -> Option<TokenUsage> {
        Some(TokenUsage {
            output_tokens,
            ..TokenUsage::default()
        })
    }

    #[test]
    fn ascii_estimate_matches_the_legacy_rate() {
        assert_eq!(estimate_tokens_fraction(""), 0.0);
        assert_eq!(estimate_tokens_fraction("abcd"), 1.0);
        assert_eq!(estimate_tokens_fraction(&"x".repeat(400)), 100.0);
    }

    #[test]
    fn cjk_estimate_is_script_weighted_not_chars_over_four() {
        // 80 个汉字 ≈ 57 token；旧的 chars/4 只给 20。
        let dense = estimate_tokens_fraction(&"中".repeat(80));
        assert!((dense - 80.0 / 1.4).abs() < 1e-9, "{dense}");
        assert!(dense > 50.0, "CJK 被低估：{dense}");

        // emoji 走 OTHER 档，不按密集文字算。
        let emoji = estimate_tokens_fraction("🙂🙂");
        assert!((emoji - 1.0).abs() < 1e-9, "{emoji}");
    }

    #[test]
    fn short_deltas_are_not_rounded_up_individually() {
        // 40 个 1 字符的 delta：逐次取整会得到 40 token，真实值只有 10。
        let mut ledger = TpsLedger::default();
        ledger.on_turn_start();
        ledger.on_step_start();
        ledger.on_first_token(0);
        for i in 0..40u128 {
            ledger.on_token_delta("x", i * 50);
        }
        ledger.on_step_end(2_000, None);
        // 10 token / 2s = 5 tok/s；逐次取整会变成 40 / 2s = 20 tok/s。
        assert_eq!(ledger.rate(), Some(5.0));
    }

    #[test]
    fn multi_step_turn_folds_tokens_over_decode_span() {
        let mut ledger = TpsLedger::default();
        ledger.on_turn_start();

        // 第一步：100 token / 1s
        ledger.on_step_start();
        ledger.on_first_token(0);
        ledger.on_token_delta(&"x".repeat(400), 500);
        ledger.on_step_end(1_000, usage(100));
        assert_eq!(ledger.rate(), Some(100.0));

        // 工具执行 51 秒：不进入分母
        // 第二步：300 token / 3s
        ledger.on_step_start();
        ledger.on_first_token(52_000);
        ledger.on_token_delta(&"x".repeat(400), 52_500);
        ledger.on_step_end(55_000, usage(300));

        // (100 + 300) / (1s + 3s) = 100 tok/s，不是 (400 / 54s)
        assert_eq!(ledger.rate(), Some(100.0));
        assert_eq!(ledger.on_turn_end(), Some(100.0));
    }

    #[test]
    fn burst_of_deltas_keeps_the_elapsed_span() {
        let mut ledger = TpsLedger::default();
        ledger.on_turn_start();
        ledger.on_step_start();
        ledger.on_first_token(0);
        ledger.on_token_delta(&"x".repeat(400), 600);
        let baseline = ledger.rate().unwrap();

        // 同一时间戳上排空 50 个 delta：时间戳不变，估算量增长，速率随之变大，
        // 但跨度仍然来自事件时间戳而不是消费时刻。
        for _ in 0..50 {
            ledger.on_token_delta(&"x".repeat(400), 600);
        }
        assert!(ledger.rate().unwrap() > baseline);
        // 跨度未增长，所以速率仍是「该跨度内的 token 数」，不会凭空出现零耗时。
        ledger.on_step_end(1_600, usage(800));
        assert_eq!(ledger.rate(), Some(500.0));
    }

    #[test]
    fn short_spans_are_not_sampled() {
        let mut ledger = TpsLedger::default();
        ledger.on_turn_start();
        ledger.on_step_start();
        ledger.on_first_token(0);
        ledger.on_token_delta(&"x".repeat(400), 500);
        // 500ms 边界本身不算可靠样本
        ledger.on_step_end(500, usage(500));
        assert_eq!(ledger.rate(), None);
        assert_eq!(ledger.on_turn_end(), None);
    }

    #[test]
    fn a_turn_without_samples_restores_the_previous_reading() {
        let mut ledger = TpsLedger::default();
        ledger.on_turn_start();
        ledger.on_step_start();
        ledger.on_first_token(0);
        ledger.on_token_delta(&"x".repeat(400), 600);
        ledger.on_step_end(2_000, usage(100));
        assert_eq!(ledger.rate(), Some(50.0));
        ledger.on_turn_end();
        assert_eq!(ledger.rate(), Some(50.0));

        // 下一轮只有一次被取消的步：读数保留上一轮的 50，而不是清空。
        ledger.on_turn_start();
        ledger.on_step_start();
        ledger.on_step_end(5_000, None);
        assert_eq!(ledger.on_turn_end(), None);
        assert_eq!(ledger.rate(), Some(50.0));
    }

    #[test]
    fn missing_usage_falls_back_to_the_streamed_estimate() {
        let mut ledger = TpsLedger::default();
        ledger.on_turn_start();
        ledger.on_step_start();
        ledger.on_first_token(0);
        ledger.on_token_delta(&"x".repeat(400), 1_000);
        ledger.on_step_end(2_000, None);
        // 100 估算 token / 2s
        assert_eq!(ledger.rate(), Some(50.0));
    }

    #[test]
    fn empty_deltas_do_not_start_the_decode_clock() {
        let mut ledger = TpsLedger::default();
        ledger.on_turn_start();
        ledger.on_step_start();
        ledger.on_token_delta("", 0);
        ledger.on_step_end(1_000, usage(100));
        // 没有首 token 边界就没有跨度，也不采样。
        assert_eq!(ledger.rate(), None);
    }

    /// 后端只报用量、不发首 token 也不发内容的步（隐藏生成立即失败）：
    /// 既没有跨度也没有可测内容，不得污染已有读数。
    #[test]
    fn step_end_without_first_token_leaves_the_ledger_untouched() {
        let mut ledger = TpsLedger::default();
        ledger.on_turn_start();
        ledger.on_step_start();
        ledger.on_step_end(3_000, usage(500));
        assert_eq!(ledger.rate(), None);
        assert_eq!(ledger.on_turn_end(), None);
    }

    /// 取消的步可能在 TurnStart 之前就结束（切换会话时的残留事件）：
    /// 没有开启的步时不报错、也不产生样本。
    #[test]
    fn step_end_without_an_open_step_is_ignored() {
        let mut ledger = TpsLedger::default();
        ledger.on_step_end(1_000, usage(100));
        assert_eq!(ledger.rate(), None);
        ledger.on_turn_start();
        ledger.on_step_end(2_000, usage(100));
        assert_eq!(ledger.rate(), None);
        assert_eq!(ledger.on_turn_end(), None);
    }
}
