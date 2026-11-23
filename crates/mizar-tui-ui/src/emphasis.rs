//! `pi-emphasis` 闭环视觉语法引擎：
//!
//! 1. 生成交互式 TUI 期间注入的 `reply_style` 规范文本（由宿主挂载进 `mizar-system-prompt`）；
//! 2. 保持 `Session` 存储、剪贴板复制与导出文本的原始纯净 Markdown；
//! 3. 在渲染层将 `[[Done]]` / `[[Failed]]` / `[[Needs you]]` 等标记渲染为语义胶囊、
//!    为标题/引导词自动加图标，并为 `✓`/`✗`/`→`/`├──` 与锚定加粗词着色。

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PillTone {
    Success,
    Error,
    Warning,
    Danger,
    Mode,
    Owner,
    Neutral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmphasisConfig {
    pub enabled: bool,
    pub round_caps: bool,
    pub icons: bool,
    pub guidance: bool,
}

impl Default for EmphasisConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl EmphasisConfig {
    pub fn from_env() -> Self {
        let off = |k: &str| matches!(std::env::var(k).as_deref(), Ok("0" | "false" | "off"));
        Self {
            enabled: !off("MIZAR_TUI_EMPHASIS"),
            round_caps: !off("MIZAR_TUI_ROUND_CAPS") && !off("PI_EMPHASIS_ROUND_CAPS"),
            icons: !off("MIZAR_TUI_PILL_ICONS") && !off("PI_EMPHASIS_PILL_ICONS"),
            guidance: !off("MIZAR_TUI_EMPHASIS_GUIDANCE") && !off("PI_EMPHASIS_GUIDANCE"),
        }
    }
}

const ICON_CHECK: &str = "\u{f00c}";
const ICON_CROSS: &str = "\u{f00d}";
const ICON_CLOCK: &str = "\u{f017}";

const ERROR_CORE: &[&str] = &["Failed", "Blocked", "Broken", "Unsafe"];
const ERROR_ALSO: &[&str] = &["fail", "failing", "error", "rejected", "down"];

const WARNING_CORE: &[&str] = &[
    "Needs you",
    "Decision",
    "Unverified",
    "Partial",
    "Skipped",
    "In progress",
    "Pending",
    "At risk",
];
const WARNING_ALSO: &[&str] = &[
    "needs review",
    "waiting",
    "review",
    "todo",
    "stale",
    "draft",
];

const SUCCESS_CORE: &[&str] = &[
    "Done",
    "Passed",
    "Observed",
    "Fixed",
    "Live",
    "Recommended",
    "Keep",
];
const SUCCESS_ALSO: &[&str] = &[
    "pass",
    "ready",
    "merged",
    "safe",
    "verified",
    "shipped",
    "ok",
    "approved",
    "complete",
    "completed",
    "finished",
    "healthy",
];

const DANGER_CORE: &[&str] = &["Costs money", "Destructive", "Irreversible"];
const MODE_CORE: &[&str] = &[
    "Brainstorm",
    "Investigate",
    "Preview",
    "Implement",
    "Qualify",
];
const OWNER_CORE: &[&str] = &["Owned", "Shared", "Foreign"];

const SYNONYMS: &[(&str, &str)] = &[
    ("complete", "Done"),
    ("completed", "Done"),
    ("ok", "Done"),
    ("finished", "Done"),
    ("pass", "Passed"),
    ("fail", "Failed"),
    ("failing", "Failed"),
    ("error", "Failed"),
    ("waiting", "Pending"),
    ("needs review", "Review"),
];

const PILL_ICONS: &[(&str, &str)] = &[
    ("done", ICON_CHECK),
    ("passed", ICON_CHECK),
    ("pass", ICON_CHECK),
    ("complete", ICON_CHECK),
    ("completed", ICON_CHECK),
    ("ok", ICON_CHECK),
    ("verified", ICON_CHECK),
    ("ready", ICON_CHECK),
    ("observed", "\u{f06e}"),
    ("recommended", "\u{f005}"),
    ("fixed", "\u{f0ad}"),
    ("live", "\u{f135}"),
    ("shipped", "\u{f135}"),
    ("approved", "\u{f164}"),
    ("failed", ICON_CROSS),
    ("fail", ICON_CROSS),
    ("failing", ICON_CROSS),
    ("error", ICON_CROSS),
    ("rejected", ICON_CROSS),
    ("blocked", "\u{f05e}"),
    ("broken", "\u{f127}"),
    ("unsafe", "\u{f132}"),
    ("down", "\u{f063}"),
    ("needs you", "\u{f007}"),
    ("decision", "\u{f126}"),
    ("unverified", "\u{f128}"),
    ("at risk", "\u{f071}"),
    ("in progress", "\u{f110}"),
    ("pending", ICON_CLOCK),
    ("waiting", ICON_CLOCK),
    ("partial", "\u{f042}"),
    ("skipped", "\u{f04e}"),
    ("inferred", "\u{f0eb}"),
    ("new", "\u{f067}"),
    ("changed", "\u{f040}"),
    ("optional", "\u{f10c}"),
    ("costs money", "\u{f155}"),
    ("destructive", "\u{f1f8}"),
    ("irreversible", "\u{f023}"),
    ("review", "\u{f06e}"),
    ("stale", "\u{f1da}"),
    ("draft", "\u{f044}"),
    ("safe", "\u{f132}"),
    ("brainstorm", "\u{f0eb}"),
    ("investigate", "\u{f002}"),
    ("preview", "\u{f06e}"),
    ("implement", "\u{f0ad}"),
    ("qualify", "\u{f0c3}"),
    ("owned", "\u{f007}"),
    ("shared", "\u{f0c0}"),
    ("foreign", "\u{f023}"),
    ("keep", "\u{f08d}"),
    ("merged", "\u{f126}"),
    ("healthy", "\u{f004}"),
    ("todo", "\u{f0ae}"),
];

const LABEL_ICONS: &[(&str, &str)] = &[
    ("finding", "\u{f002}"),
    ("terminal", "\u{f120}"),
    ("folder", "\u{f07b}"),
    ("file", "\u{f15b}"),
    ("branch", "\u{f126}"),
    ("review", "\u{f06e}"),
    ("ideas", "\u{f0eb}"),
    ("cause", "\u{f0e7}"),
    ("fix", "\u{f0ad}"),
    ("risk", "\u{f071}"),
    ("limits", "\u{f071}"),
    ("next", "\u{f061}"),
    ("rollback", "\u{f0e2}"),
    ("undo", "\u{f0e2}"),
    ("options", "\u{f0ca}"),
    ("result", "\u{f11e}"),
    ("why", "\u{f059}"),
    ("how", "\u{f013}"),
    ("cost", "\u{f155}"),
    ("rule", "\u{f0e3}"),
    ("example", "\u{f121}"),
    ("checked", "\u{f00c}"),
    ("pills", "\u{f02b}"),
    ("colors", "\u{f1fc}"),
    ("icons", "\u{f005}"),
    ("scoreboard", "\u{f080}"),
    ("verdict", "\u{f0e3}"),
    ("effect", "\u{f140}"),
    ("evidence", "\u{f06e}"),
    ("plan", "\u{f0ae}"),
    ("status", "\u{f012}"),
    ("decision", "\u{f126}"),
];

pub const ACTION_ICONS: &[(&str, &str)] = &[
    ("check", "\u{f046}"),
    ("do", "\u{f0a4}"),
    ("next", "\u{f061}"),
    ("unblock", "\u{f09c}"),
    ("left", "\u{f141}"),
    ("run", "\u{f04b}"),
    ("guard", "\u{f132}"),
    ("undo", "\u{f0e2}"),
    ("see", "\u{f08e}"),
];

fn starts_with_pua_glyph(s: &str) -> Option<(char, usize)> {
    let trimmed = s.trim_start();
    let ch = trimmed.chars().next()?;
    let code = ch as u32;
    let is_pua = (0xE000..=0xF8FF).contains(&code) || (0xF0000..=0xFFFFD).contains(&code);
    if is_pua {
        let rest_start = s.len() - trimmed.len() + ch.len_utf8();
        Some((ch, rest_start))
    } else {
        None
    }
}

fn collapse_spaces(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedPill {
    pub icon: Option<String>,
    pub word: String,
    pub key: String,
    pub tone: PillTone,
}

fn known_words_sorted() -> Vec<&'static str> {
    let mut words: Vec<&'static str> = Vec::new();
    for list in [
        ERROR_CORE,
        ERROR_ALSO,
        WARNING_CORE,
        WARNING_ALSO,
        SUCCESS_CORE,
        SUCCESS_ALSO,
        DANGER_CORE,
        MODE_CORE,
        OWNER_CORE,
    ] {
        words.extend_from_slice(list);
    }
    for (k, _) in PILL_ICONS {
        words.push(k);
    }
    for (k, _) in SYNONYMS {
        words.push(k);
    }
    words.sort_by_key(|w| std::cmp::Reverse(w.len()));
    words
}

/// 解析 `[[label]]` 的图标、规范词与语义色调，与 `pi-emphasis` `splitLabel`/`toneOf` 一致。
pub fn parse_pill(raw_label: &str) -> ParsedPill {
    let own = starts_with_pua_glyph(raw_label);
    let body = match own {
        Some((_, rest_idx)) => collapse_spaces(raw_label[rest_idx..].trim()),
        None => collapse_spaces(raw_label.trim()),
    };
    let lower = body.to_ascii_lowercase();
    let known = known_words_sorted().into_iter().find(|k| {
        let kl = k.to_ascii_lowercase();
        lower == kl
            || (lower.starts_with(&kl)
                && lower
                    .as_bytes()
                    .get(kl.len())
                    .is_some_and(|b| *b == b' ' || *b == b':'))
    });

    let canonical = known.and_then(|k| {
        let kl = k.to_ascii_lowercase();
        SYNONYMS.iter().find(|(s, _)| *s == kl).map(|(_, c)| *c)
    });

    let key = canonical
        .map(str::to_ascii_lowercase)
        .or_else(|| known.map(str::to_ascii_lowercase))
        .unwrap_or_else(|| lower.clone());

    let word = match (canonical, known) {
        (Some(c), Some(k)) if body.len() >= k.len() => format!("{c}{}", &body[k.len()..]),
        _ => body,
    };

    let icon = own.map(|(ch, _)| ch.to_string()).or_else(|| {
        PILL_ICONS
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, ic)| (*ic).to_string())
    });

    let tone = tone_of_key(&key);
    ParsedPill {
        icon,
        word,
        key,
        tone,
    }
}

pub fn tone_of_key(key: &str) -> PillTone {
    let in_list = |list: &[&str]| list.iter().any(|w| w.eq_ignore_ascii_case(key));
    if in_list(DANGER_CORE) {
        PillTone::Danger
    } else if in_list(ERROR_CORE) || in_list(ERROR_ALSO) {
        PillTone::Error
    } else if in_list(WARNING_CORE) || in_list(WARNING_ALSO) {
        PillTone::Warning
    } else if in_list(SUCCESS_CORE) || in_list(SUCCESS_ALSO) {
        PillTone::Success
    } else if in_list(MODE_CORE) {
        PillTone::Mode
    } else if in_list(OWNER_CORE) {
        PillTone::Owner
    } else {
        PillTone::Neutral
    }
}

/// 将 `[[label]]` 渲染为带语义背景/圆角帽的 `Span` 列表（空格转为 `\u{00a0}` 防折行截断）。
pub fn render_pill_spans(
    raw_label: &str,
    theme: &Theme,
    cfg: &EmphasisConfig,
) -> Vec<Span<'static>> {
    if !cfg.enabled {
        return vec![Span::styled(
            format!("[[{raw_label}]]"),
            Style::default().fg(theme.text),
        )];
    }

    let parsed = parse_pill(raw_label);
    let (fg, bg) = match parsed.tone {
        PillTone::Success => theme.pill_success,
        PillTone::Error => theme.pill_error,
        PillTone::Warning => theme.pill_warning,
        PillTone::Danger => theme.pill_danger,
        PillTone::Mode => theme.pill_mode,
        PillTone::Owner => theme.pill_owner,
        PillTone::Neutral => theme.pill_neutral,
    };

    let inner = if cfg.icons
        && let Some(ic) = parsed.icon
        && !parsed.word.is_empty()
    {
        format!("{ic} {}", parsed.word)
    } else if parsed.word.is_empty() {
        raw_label.trim().to_string()
    } else {
        parsed.word
    };
    let protected = inner.replace(' ', "\u{00a0}");

    let has_panel_bg = !matches!(bg, Color::Reset);
    let label_style = if has_panel_bg {
        Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(fg)
            .add_modifier(Modifier::BOLD | Modifier::REVERSED)
    };

    if cfg.round_caps && has_panel_bg {
        let cap_style = Style::default().fg(bg);
        vec![
            Span::styled("\u{e0b6}", cap_style),
            Span::styled(protected, label_style),
            Span::styled("\u{e0b4}", cap_style),
        ]
    } else {
        vec![Span::styled(
            format!("\u{00a0}{protected}\u{00a0}"),
            label_style,
        )]
    }
}

pub fn is_mode_pill(raw_label: &str) -> bool {
    matches!(parse_pill(raw_label).tone, PillTone::Mode)
}

pub fn pill_matches_label(pill_raw: &str, label: &str) -> bool {
    let stem = label
        .trim()
        .trim_end_matches([':', '：'])
        .trim()
        .to_ascii_lowercase();
    if stem.is_empty() {
        return false;
    }
    let pill = parse_pill(pill_raw);
    stem == pill.key
        || stem == pill.word.to_ascii_lowercase()
        || stem == pill_raw.trim().to_ascii_lowercase()
}

/// 给已知标题或行首加粗标签补充图标（例如 `Finding:` -> ` Finding:`）。
pub fn decorate_label_with_icon(label: &str, cfg: &EmphasisConfig) -> String {
    if !cfg.enabled || !cfg.icons || starts_with_pua_glyph(label).is_some() {
        return label.to_string();
    }
    let trimmed = label.trim_end();
    let word = trimmed.trim_end_matches(':').trim().to_ascii_lowercase();
    let icon = LABEL_ICONS
        .iter()
        .find(|(k, _)| *k == word)
        .map(|(_, ic)| *ic)
        .or_else(|| {
            PILL_ICONS
                .iter()
                .find(|(k, _)| *k == word)
                .map(|(_, ic)| *ic)
        })
        .or_else(|| {
            if trimmed.ends_with(':') {
                ACTION_ICONS
                    .iter()
                    .find(|(k, _)| *k == word)
                    .map(|(_, ic)| *ic)
            } else {
                None
            }
        });
    match icon {
        Some(ic) => format!("{ic} {label}"),
        None => label.to_string(),
    }
}

/// 若加粗词是 `Check:` / `Do:` / `Next:` 等行动引导词，返回带图标文本。
pub fn decorate_action_lead_in(bold_inner: &str, cfg: &EmphasisConfig) -> String {
    if !cfg.enabled || !cfg.icons || starts_with_pua_glyph(bold_inner).is_some() {
        return bold_inner.to_string();
    }
    if let Some(stem) = bold_inner.strip_suffix(':') {
        let lower = stem.trim().to_ascii_lowercase();
        if let Some((_, ic)) = ACTION_ICONS.iter().find(|(k, _)| *k == lower) {
            return format!("{ic} {bold_inner}");
        }
    }
    bold_inner.to_string()
}

/// 根据同一份词库动态生成紧凑英文 `reply_style` 提示词段落（`<= 800B`），保证模型规范与渲染器永远同步。
pub fn reply_guidance_prompt() -> String {
    let pick = |list: &[&str], max: usize| -> String {
        list.iter()
            .take(max)
            .map(|w| format!("[[{w}]]"))
            .collect::<Vec<_>>()
            .join("/")
    };
    let cap_join = |targets: &[&str], pairs: &[(&str, &str)]| -> String {
        targets
            .iter()
            .filter(|target| pairs.iter().any(|(k, _)| k == *target))
            .map(|w| {
                let mut ch = w.chars();
                match ch.next() {
                    Some(first) => format!("{}{}", first.to_ascii_uppercase(), ch.as_str()),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join("/")
    };

    let key_labels = cap_join(
        &["finding", "cause", "fix", "risk", "result", "next"],
        LABEL_ICONS,
    );
    let actions = cap_join(&["check", "do", "next", "unblock", "run"], ACTION_ICONS);

    format!(
        "These rules shape replies in this terminal (auto-styled pills/icons; never type glyphs):\n\
         - During multi-step work, briefly tell the user what you are doing.\n\
         - Start with a mode pill ({mode}) or bold label ({labels}), never both (e.g. `**Investigate** [[Investigate]]`).\n\
         - Use pills sparingly for status only (not in paths/code/prose): {err}; {warn}; {ok}; {danger}. Follow error/warn/danger with a bold action ({actions}:).\n\
         - Bold 1-3 anchors per item; use plain `├──`/`└──`/`→`/`✓`/`✗`.",
        mode = pick(MODE_CORE, MODE_CORE.len()),
        labels = key_labels,
        err = pick(ERROR_CORE, 3),
        warn = pick(WARNING_CORE, 4),
        ok = pick(SUCCESS_CORE, 3),
        danger = pick(DANGER_CORE, 2),
        actions = actions,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_synonyms_and_details_in_pills() {
        let done = parse_pill("Completed");
        assert_eq!(done.word, "Done");
        assert_eq!(done.tone, PillTone::Success);
        assert_eq!(done.icon.as_deref(), Some(ICON_CHECK));

        let detail = parse_pill("Failed: 3 tests");
        assert_eq!(detail.word, "Failed: 3 tests");
        assert_eq!(detail.tone, PillTone::Error);
        assert_eq!(detail.icon.as_deref(), Some(ICON_CROSS));

        let danger = parse_pill("Costs money");
        assert_eq!(danger.tone, PillTone::Danger);
    }

    #[test]
    fn renders_non_breaking_spaces_and_caps() {
        let theme = Theme::mist_dark();
        let cfg = EmphasisConfig {
            enabled: true,
            round_caps: true,
            icons: true,
            guidance: true,
        };
        let spans = render_pill_spans("Needs you", &theme, &cfg);
        let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.starts_with('\u{e0b6}'));
        assert!(text.ends_with('\u{e0b4}'));
        assert!(text.contains("Needs\u{00a0}you"));

        let square = render_pill_spans(
            "In progress",
            &theme,
            &EmphasisConfig {
                round_caps: false,
                ..cfg
            },
        );
        let sq_text: String = square.iter().map(|s| s.content.as_ref()).collect();
        assert!(!sq_text.contains('\u{e0b6}'));
        assert!(sq_text.contains("In\u{00a0}progress"));
    }

    #[test]
    fn prompt_guidance_reflects_core_vocabulary() {
        let prompt = reply_guidance_prompt();
        assert!(
            prompt.len() <= 800,
            "reply_guidance_prompt must be <= 800 bytes, got {}:\n{prompt}",
            prompt.len()
        );
        assert!(
            !prompt
                .chars()
                .any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)),
            "reply_guidance_prompt must be pure English: {prompt}"
        );
        assert!(prompt.contains("[[Done]]"));
        assert!(prompt.contains("[[Needs you]]"));
        assert!(prompt.contains("[[Costs money]]"));
        assert!(prompt.contains("[[Failed]]"));
        assert!(prompt.contains("[[Investigate]]"));
        assert!(prompt.contains("Finding"));
        assert!(prompt.contains("Check"));
        assert!(!prompt.contains("Pills are required"));
        assert!(prompt.contains("Use pills sparingly"));
        assert!(prompt.contains("**Investigate** [[Investigate]]"));
        assert!(
            prompt.contains("During multi-step work, briefly tell the user what you are doing.")
        );

        // 双向契约校验：提示词中出现的每个 [[...]] 标记都必须被渲染器识别为非 Neutral 胶囊
        let mut rest = prompt.as_str();
        let mut pill_count = 0usize;
        while let Some(start) = rest.find("[[") {
            let after = &rest[start + 2..];
            let end = after.find("]]").expect("unclosed [[");
            let label = &after[..end];
            let parsed = parse_pill(label);
            assert_ne!(
                parsed.tone,
                PillTone::Neutral,
                "pill [[{label}]] in reply_guidance_prompt must match renderer vocabulary"
            );
            pill_count += 1;
            rest = &after[end + 2..];
        }
        assert!(pill_count >= 10, "expected core pills, got {pill_count}");
    }

    #[test]
    fn detects_mode_pills_and_label_matches() {
        assert!(is_mode_pill("Investigate"));
        assert!(!is_mode_pill("Done"));
        assert!(pill_matches_label("Investigate", "Investigate:"));
        assert!(pill_matches_label("Completed", "Done"));
        assert!(!pill_matches_label("Done", "Finding"));
    }
}
