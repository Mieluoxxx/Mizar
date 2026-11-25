//! 会话标题分类、解析与格式化（`MMDD｜TYPE｜Topic`）。

use chrono::{DateTime, Local};

const FULLWIDTH_PIPE: char = '｜';
const MAX_TOPIC_CHARS: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionTitleType {
    Fea,
    Des,
    Fix,
    Opt,
    Rel,
    Exp,
    Doc,
    Res,
}

impl SessionTitleType {
    pub fn as_code(self) -> &'static str {
        match self {
            Self::Fea => "FEA",
            Self::Des => "DES",
            Self::Fix => "FIX",
            Self::Opt => "OPT",
            Self::Rel => "REL",
            Self::Exp => "EXP",
            Self::Doc => "DOC",
            Self::Res => "RES",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        let token = raw
            .trim()
            .trim_matches(|ch: char| matches!(ch, '[' | ']' | '(' | ')' | '`' | '*' | '"' | '\''))
            .to_ascii_uppercase();
        match token.as_str() {
            "FEA" => Some(Self::Fea),
            "DES" => Some(Self::Des),
            "FIX" => Some(Self::Fix),
            "OPT" => Some(Self::Opt),
            "REL" => Some(Self::Rel),
            "EXP" => Some(Self::Exp),
            "DOC" => Some(Self::Doc),
            "RES" => Some(Self::Res),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedModelTitle {
    pub kind: SessionTitleType,
    pub topic: String,
}

fn strip_think_blocks(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + "<think>".len()..];
        if let Some(end) = after_start.find("</think>") {
            rest = &after_start[end + "</think>".len()..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out
}

fn looks_like_mmdd(raw: &str) -> bool {
    let trimmed = raw.trim();
    trimmed.len() == 4 && trimmed.bytes().all(|b| b.is_ascii_digit())
}

fn clean_topic(raw: &str) -> Option<String> {
    let stripped = raw
        .trim()
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | '`' | '“' | '”' | '‘' | '’'))
        .trim_start_matches(|ch: char| {
            ch.is_whitespace() || matches!(ch, '|' | '｜' | ':' | '：' | '-' | '—')
        })
        .trim();
    let collapsed = stripped.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return None;
    }
    let count = collapsed.chars().count();
    if count <= MAX_TOPIC_CHARS {
        Some(collapsed)
    } else {
        let mut head: String = collapsed.chars().take(MAX_TOPIC_CHARS - 1).collect();
        head.push('…');
        Some(head)
    }
}

/// 从模型原始输出解析 `TYPE｜Topic`（兼容半角 `|`、前缀 `MMDD|` 及 `TYPE: Topic`）。
pub fn parse_model_title_output(raw: &str) -> Option<ParsedModelTitle> {
    let without_think = strip_think_blocks(raw);
    let line = without_think
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("```") && !line.starts_with("---"))?
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | '`'));

    let normalized = line.replace(FULLWIDTH_PIPE, "|");
    let parts: Vec<&str> = normalized.split('|').collect();
    if parts.len() >= 3
        && looks_like_mmdd(parts[0])
        && let Some(kind) = SessionTitleType::parse(parts[1])
    {
        let topic = clean_topic(&parts[2..].join(" "))?;
        return Some(ParsedModelTitle { kind, topic });
    }
    if parts.len() >= 2
        && let Some(kind) = SessionTitleType::parse(parts[0])
    {
        let topic = clean_topic(&parts[1..].join(" "))?;
        return Some(ParsedModelTitle { kind, topic });
    }

    // 兼容 `TYPE: Topic` / `TYPE：Topic` / `TYPE - Topic`
    for sep in [':', '：', '-'] {
        if let Some((left, right)) = line.split_once(sep)
            && let Some(kind) = SessionTitleType::parse(left)
        {
            let topic = clean_topic(right)?;
            return Some(ParsedModelTitle { kind, topic });
        }
    }

    // 兼容精确大写前缀 `FEA Topic`
    if let Some((first, rest)) = line.split_once(char::is_whitespace)
        && first.len() == 3
        && first.bytes().all(|b| b.is_ascii_uppercase())
        && let Some(kind) = SessionTitleType::parse(first)
    {
        let topic = clean_topic(rest)?;
        return Some(ParsedModelTitle { kind, topic });
    }

    None
}

fn local_mmdd(ts_ms: u128) -> String {
    let dt = i64::try_from(ts_ms)
        .ok()
        .filter(|&ms| ms > 0)
        .and_then(DateTime::from_timestamp_millis)
        .map(|utc| utc.with_timezone(&Local))
        .unwrap_or_else(Local::now);
    dt.format("%m%d").to_string()
}

/// 按本地日期格式化为 `MMDD｜TYPE｜Topic`（分隔符为全角 `｜` U+FF5C）。
pub fn format_session_title(ts_ms: u128, parsed: &ParsedModelTitle) -> String {
    format!(
        "{}{FULLWIDTH_PIPE}{}{FULLWIDTH_PIPE}{}",
        local_mmdd(ts_ms),
        parsed.kind.as_code(),
        parsed.topic
    )
}

/// 将 `"provider/model"`（支持模型 id 本身含 `/`）或裸 `"model"` 拆分为 `(Option<provider>, model)`。
pub fn parse_model_ref(raw: &str) -> Option<(Option<String>, String)> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some((provider, model)) = trimmed.split_once('/') {
        let p = provider.trim();
        let m = model.trim();
        if !p.is_empty() && !m.is_empty() {
            return Some((Some(p.to_string()), m.to_string()));
        }
        if !m.is_empty() {
            return Some((None, m.to_string()));
        }
        return None;
    }
    Some((None, trimmed.to_string()))
}
