//! 会话标题生成的消息抽取（仅提取 User 消息）与 Prompt 构造。

use mizar_llm::Message;
use mizar_session::{SessionEntry, SessionEvent, derive_session_title};

const MAX_SINGLE_MSG_CHARS: usize = 400;
const MAX_TOTAL_USER_CHARS: usize = 2400;

fn clean_user_message_for_title(raw: &str) -> Option<String> {
    let cleaned_lines: Vec<String> = raw.lines().filter_map(derive_session_title).collect();
    if cleaned_lines.is_empty() {
        return None;
    }
    let joined = cleaned_lines.join(" ");
    let count = joined.chars().count();
    if count <= MAX_SINGLE_MSG_CHARS {
        Some(joined)
    } else {
        let mut head: String = joined.chars().take(MAX_SINGLE_MSG_CHARS - 1).collect();
        head.push('…');
        Some(head)
    }
}

/// 仅从会话日志中提取干净的 `User` 输入列表，排除助手回复、工具调用/结果及系统提示。
pub fn extract_user_messages(entries: &[SessionEntry]) -> Vec<String> {
    entries
        .iter()
        .filter_map(|entry| match &entry.event {
            SessionEvent::User { text, .. } => clean_user_message_for_title(text),
            _ => None,
        })
        .collect()
}

/// 构造标题生成模型请求的 `[System, User]` 消息列表。
pub fn build_title_messages(user_messages: &[String]) -> Vec<Message> {
    let mut selected: Vec<String> = Vec::new();
    let mut used_chars = 0usize;

    if let Some(first) = user_messages.first() {
        used_chars += first.chars().count();
        selected.push(format!("1. {first}"));
    }

    if user_messages.len() > 1 {
        let mut tail_items: Vec<(usize, &String)> = Vec::new();
        for (idx, msg) in user_messages.iter().enumerate().skip(1).rev() {
            let len = msg.chars().count();
            if used_chars + len > MAX_TOTAL_USER_CHARS && !tail_items.is_empty() {
                break;
            }
            used_chars += len;
            tail_items.push((idx + 1, msg));
            if used_chars >= MAX_TOTAL_USER_CHARS {
                break;
            }
        }
        tail_items.reverse();
        if let Some((first_tail_idx, _)) = tail_items.first()
            && *first_tail_idx > 2
        {
            selected.push("... (earlier turns omitted)".to_string());
        }
        for (idx, msg) in tail_items {
            selected.push(format!("{idx}. {msg}"));
        }
    }

    let conversation = selected.join("\n");
    let system_prompt = "\
You name coding-agent terminal sessions from the user's messages.
Classify the overall session goal into ONE 3-letter TYPE from:
- FEA: new feature or capability
- DES: design, architecture, UI/UX layout
- FIX: bug fix, debugging, error investigation
- OPT: refactor, optimization, performance, cleanup
- REL: release, CI/CD, packaging, dependency maintenance
- EXP: exploration, Q&A, code reading, prototyping
- DOC: documentation, comments, changelog
- RES: research, comparison, deep analysis

Rules:
1. Match the user's primary language (Chinese for Chinese requests, English for English requests).
2. Topic must be concrete and concise (roughly 4 to 16 Chinese chars or 2 to 6 English words).
3. Do NOT repeat CLI invocation noise, raw file paths, or UUIDs.
4. Output ONLY a single line in exact format: TYPE｜Topic
   Example (Chinese): FEA｜自动会话标题生成
   Example (English): OPT｜JSONL Session Persistence";

    let user_prompt = format!(
        "User messages in chronological order:\n<user-messages>\n{conversation}\n</user-messages>\n\nOutput ONLY `TYPE｜Topic`."
    );

    vec![
        Message::System {
            content: system_prompt.to_string(),
        },
        Message::user(user_prompt),
    ]
}
