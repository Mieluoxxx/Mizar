//! 无损情景回溯插件（综合 `dim` `Recall` + `maka` N-gram Recall + `obelisk` 防爆护栏）。
//!
//! - 注册单一常驻只读工具：`recall`（固定 `ToolExposure::Direct`，全程冻结工具声明集不动态改表）；
//! - 同时支持按 `query` 搜索摘要片段（穿透压缩与谱系边界）以及按 `seq`/`session_id` 分页拉取原文；
//! - 优先调用 SQLite 后端的 FTS5 1..=3 n-gram 索引，未挂载持久化时自动回退至内存扫描；
//! - 严格限定单次返回字符上限、跨会话作用域校验与 `is_current_session` 标记，防止回溯反向撑爆上下文。

use std::sync::Arc;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_llm::ContentBlock;
use mizar_session::{
    RecallSearchHit, RecallSearchQuery, SessionContextExt, SessionEntry, SessionEvent,
    SessionListQuery, SessionStore,
};
use mizar_tools::{
    ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolExposure, ToolOutput, ToolsContextExt,
};
use serde_json::{Value, json};

pub const RECALL_TOOL_NAME: &str = "recall";

const MAX_SEARCH_LIMIT: usize = 20;
const MAX_SEARCH_OUTPUT_BYTES: usize = 8_192;
const MAX_READ_RANGE_ENTRIES: u64 = 12;
const DEFAULT_READ_CHAR_LIMIT: usize = 6_000;
const MAX_READ_CHAR_LIMIT: usize = 8_192;
const MAX_READ_OUTPUT_BYTES: usize = 16_384;
const FALLBACK_SNIPPET_CHARS: usize = 240;

fn truncate_utf8_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[derive(Debug, Clone)]
pub struct RecallConfig {
    /// 是否启用情景回溯工具（默认 true）。
    pub enabled: bool,
    pub default_search_limit: usize,
    pub read_char_limit: usize,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_search_limit: 8,
            read_char_limit: DEFAULT_READ_CHAR_LIMIT,
        }
    }
}

fn render_entry_verbatim(entry: &SessionEntry) -> Option<String> {
    match &entry.event {
        SessionEvent::User { text, images } => {
            if images.is_empty() {
                Some(format!("[seq #{} | user]\n{}", entry.seq, text))
            } else {
                Some(format!(
                    "[seq #{} | user]\n[{} image(s)]\n{}",
                    entry.seq,
                    images.len(),
                    text
                ))
            }
        }
        SessionEvent::Assistant { blocks, .. } => {
            let mut parts = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text } if !text.trim().is_empty() => {
                        parts.push(text.clone());
                    }
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        parts.push(format!("[tool_call {name} id={id}] {arguments}"));
                    }
                    ContentBlock::Reasoning { .. } | ContentBlock::Text { .. } => {}
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(format!(
                    "[seq #{} | assistant]\n{}",
                    entry.seq,
                    parts.join("\n")
                ))
            }
        }
        SessionEvent::ToolResult {
            tool_call_id,
            name,
            content,
            is_error,
            ..
        } => {
            let status = if *is_error { "error" } else { "ok" };
            Some(format!(
                "[seq #{} | tool_result:{name} ({status}, id={tool_call_id})]\n{content}",
                entry.seq
            ))
        }
        SessionEvent::BashExecution {
            command,
            output,
            exit_code,
            cancelled,
            exclude_from_context,
        } => {
            let tag = if *exclude_from_context {
                "bash-local"
            } else {
                "bash"
            };
            let code_str = if *cancelled {
                "cancelled".to_string()
            } else {
                exit_code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "?".to_string())
            };
            Some(format!(
                "[seq #{} | {tag} exit={code_str}]\n$ {command}\n{output}",
                entry.seq
            ))
        }
        SessionEvent::Compaction {
            summary,
            first_kept_seq,
            ..
        } => Some(format!(
            "[seq #{} | compaction (kept_from=#{first_kept_seq})]\n{summary}",
            entry.seq
        )),
        SessionEvent::ContextNotice { .. }
        | SessionEvent::ToolResultPrune { .. }
        | SessionEvent::Selection { .. }
        | SessionEvent::Trace { .. } => None,
    }
}

fn build_fallback_snippet(text: &str, terms: &[String]) -> String {
    let collapsed: String = text
        .chars()
        .map(|c| {
            if c == '\n' || c == '\r' || c == '\t' {
                ' '
            } else {
                c
            }
        })
        .collect();
    let chars: Vec<char> = collapsed.chars().collect();
    if chars.len() <= FALLBACK_SNIPPET_CHARS {
        return collapsed;
    }
    let lower = collapsed.to_lowercase();
    let match_idx = terms
        .iter()
        .filter_map(|t| lower.find(t.as_str()).map(|b| lower[..b].chars().count()))
        .min()
        .unwrap_or(0);
    let start = match_idx.saturating_sub(FALLBACK_SNIPPET_CHARS / 4);
    let end = (start + FALLBACK_SNIPPET_CHARS).min(chars.len());
    let start = end.saturating_sub(FALLBACK_SNIPPET_CHARS);
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    let slice: String = chars[start..end].iter().collect();
    out.push_str(slice.trim());
    if end < chars.len() {
        out.push('…');
    }
    out
}

fn fallback_in_memory_search(
    store: &SessionStore,
    query: &RecallSearchQuery,
) -> Vec<RecallSearchHit> {
    let terms: Vec<String> = query
        .query
        .split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    if terms.is_empty() {
        return Vec::new();
    }

    let target_sessions: Vec<mizar_session::LoadedSession> = if !query.session_ids.is_empty() {
        query
            .session_ids
            .iter()
            .filter_map(|id| store.inspect_readonly(id))
            .collect()
    } else if let Ok(headers) = store.list_headers(&SessionListQuery {
        cwd: query.cwd.clone(),
        limit: Some(50),
    }) {
        headers
            .iter()
            .filter_map(|h| store.inspect_readonly(&h.id))
            .collect()
    } else {
        Vec::new()
    };

    let mut exact = Vec::new();
    let mut partial = Vec::new();

    for session in target_sessions {
        let header = session.header;
        for entry in &session.entries {
            let Some(rendered) = render_entry_verbatim(entry) else {
                continue;
            };
            let lower = rendered.to_lowercase();
            let matched = terms.iter().filter(|t| lower.contains(t.as_str())).count();
            if matched == 0 {
                continue;
            }
            let hit = RecallSearchHit {
                session_id: header.id.clone(),
                session_title: header.title.clone(),
                seq: entry.seq,
                ts: entry.ts,
                kind: entry.event.kind_name().to_string(),
                snippet: build_fallback_snippet(&rendered, &terms),
            };
            if matched == terms.len() {
                exact.push(hit);
            } else {
                partial.push(hit);
            }
        }
    }

    exact.sort_by(|a, b| b.ts.cmp(&a.ts).then_with(|| b.seq.cmp(&a.seq)));
    partial.sort_by(|a, b| b.ts.cmp(&a.ts).then_with(|| b.seq.cmp(&a.seq)));
    exact.extend(partial);
    exact.truncate(query.limit.clamp(1, MAX_SEARCH_LIMIT));
    exact
}

fn is_session_authorized(store: &SessionStore, caller_sid: Option<&str>, target_sid: &str) -> bool {
    let Some(caller_id) = caller_sid else {
        return true;
    };
    if caller_id == target_sid {
        return true;
    }
    let lineage = store.lineage_session_ids(caller_id);
    if lineage.iter().any(|id| id == target_sid) {
        return true;
    }
    let caller_cwd = store
        .inspect_readonly(caller_id)
        .and_then(|s| s.header.cwd)
        .or_else(|| store.default_cwd());
    let target_cwd = store
        .inspect_readonly(target_sid)
        .and_then(|s| s.header.cwd);
    match (caller_cwd, target_cwd) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

fn execute_recall_search(
    store: &SessionStore,
    caller_sid: Option<String>,
    query_str: String,
    scope: &str,
    limit: usize,
) -> Result<ToolOutput, BoxError> {
    let limit = limit.clamp(1, MAX_SEARCH_LIMIT);
    let (session_ids, cwd) = match scope {
        "current" => match &caller_sid {
            Some(sid) => (vec![sid.clone()], None),
            None => (Vec::new(), store.default_cwd()),
        },
        "workspace" => {
            let ws = caller_sid
                .as_deref()
                .and_then(|sid| store.inspect_readonly(sid))
                .and_then(|s| s.header.cwd)
                .or_else(|| store.default_cwd());
            (Vec::new(), ws)
        }
        _ => match &caller_sid {
            Some(sid) => (store.lineage_session_ids(sid), None),
            None => (Vec::new(), store.default_cwd()),
        },
    };

    let search_query = RecallSearchQuery {
        query: query_str.clone(),
        session_ids,
        cwd,
        limit,
    };

    // SQLite FTS5 返回 Ok(Some([])) 代表已完整检索索引且无匹配，严禁回退 50 会话冷扫。
    // 仅在持久化未挂载 (Ok(None)) 或检索出错 (Err(_)) 时走内存兜底扫描。
    let mut hits = match store.search_recall(&search_query) {
        Ok(Some(db_hits)) => db_hits,
        Ok(None) | Err(_) => fallback_in_memory_search(store, &search_query),
    };
    hits.truncate(limit);

    if hits.is_empty() {
        return Ok(ToolOutput::text(format!(
            "No preserved history matched query {:?} in scope '{scope}'.",
            query_str
        ))
        .with_structured(json!({
            "mode": "search",
            "query": query_str,
            "scope": scope,
            "hits": []
        })));
    }

    let footer = "Use recall(seq=..., session_id=...) to inspect full verbatim entries.";
    let mut formatted_entries = Vec::with_capacity(hits.len());
    let mut structured_hits = Vec::with_capacity(hits.len());
    let mut truncated_by_bytes = false;
    let max_header_bytes = format!(
        "Found {} preserved history hit(s) for {:?} (scope={scope}):",
        hits.len(),
        query_str
    )
    .len();
    let mut used_bytes = max_header_bytes + 1 + footer.len() + 1;

    for hit in &hits {
        let is_current = caller_sid.as_deref() == Some(hit.session_id.as_str());
        let origin = if is_current {
            "current".to_string()
        } else {
            format!("session:{}", hit.session_id)
        };
        let prefix = format!("- [{origin} seq #{} | {}] ", hit.seq, hit.kind);
        let mut snippet = hit.snippet.clone();
        let line_len = prefix.len() + snippet.len() + 1;

        if used_bytes + line_len > MAX_SEARCH_OUTPUT_BYTES {
            truncated_by_bytes = true;
            if formatted_entries.is_empty() {
                let avail = MAX_SEARCH_OUTPUT_BYTES.saturating_sub(used_bytes + prefix.len() + 4);
                snippet = format!("{}…", truncate_utf8_bytes(&snippet, avail));
            } else {
                break;
            }
        }

        let line = format!("{prefix}{snippet}");
        used_bytes += line.len() + 1;
        formatted_entries.push(line);
        structured_hits.push(json!({
            "sessionId": hit.session_id,
            "sessionTitle": hit.session_title,
            "isCurrentSession": is_current,
            "seq": hit.seq,
            "kind": hit.kind,
            "snippet": snippet,
        }));
    }

    let header_line = format!(
        "Found {} preserved history hit(s) for {:?} (scope={scope}):",
        formatted_entries.len(),
        query_str
    );
    let mut lines = Vec::with_capacity(formatted_entries.len() + 2);
    lines.push(header_line);
    lines.extend(formatted_entries);
    lines.push(footer.into());

    let meta = json!({
        "mode": "search",
        "query": query_str,
        "scope": scope,
        "truncated": truncated_by_bytes,
        "hits": structured_hits,
    });
    Ok(ToolOutput::text(lines.join("\n"))
        .with_structured(meta.clone())
        .with_details(meta))
}

async fn execute_recall_read(
    store: &SessionStore,
    caller_sid: Option<&str>,
    args: &Value,
    start_seq: u64,
    default_char_limit: usize,
) -> Result<ToolOutput, BoxError> {
    let requested_end = args
        .get("end_seq")
        .and_then(Value::as_u64)
        .unwrap_or(start_seq)
        .max(start_seq);
    let end_seq = requested_end.min(start_seq.saturating_add(MAX_READ_RANGE_ENTRIES - 1));
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map(|v| (v as usize).clamp(100, MAX_READ_CHAR_LIMIT))
        .unwrap_or_else(|| default_char_limit.clamp(100, MAX_READ_CHAR_LIMIT));
    let to_file = args
        .get("to_file")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let target_sid = args
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .or_else(|| caller_sid.map(ToString::to_string))
        .ok_or("Missing `session_id` and no active session.")?;

    if !is_session_authorized(store, caller_sid, &target_sid) {
        return Err(format!(
            "Access denied: session `{target_sid}` is outside current lineage or workspace."
        )
        .into());
    }

    let session = store
        .inspect_readonly(&target_sid)
        .ok_or_else(|| format!("Session `{target_sid}` not found."))?;

    let mut rendered_blocks = Vec::new();
    for entry in &session.entries {
        if entry.seq < start_seq || entry.seq > end_seq {
            continue;
        }
        if let Some(block) = render_entry_verbatim(entry) {
            rendered_blocks.push(block);
        }
    }

    if rendered_blocks.is_empty() {
        return Ok(ToolOutput::text(format!(
            "No readable surface entries found in session `{target_sid}` for seq {start_seq}..={end_seq}."
        ))
        .with_structured(json!({
            "mode": "read",
            "sessionId": target_sid,
            "startSeq": start_seq,
            "endSeq": end_seq,
            "found": false
        })));
    }

    let full_text = rendered_blocks.join("\n\n---\n\n");
    let total_chars = full_text.chars().count();

    if to_file {
        let safe_sid: String = target_sid
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .take(32)
            .collect();
        let out_path = std::env::temp_dir().join(format!(
            "mizar_recall_{safe_sid}_{start_seq}_{end_seq}_{}.txt",
            std::process::id()
        ));
        tokio::fs::write(&out_path, full_text.as_bytes()).await?;
        let meta = json!({
            "mode": "read",
            "sessionId": target_sid,
            "startSeq": start_seq,
            "endSeq": end_seq,
            "totalChars": total_chars,
            "filePath": out_path.display().to_string(),
        });
        return Ok(ToolOutput::text(format!(
            "Exported {total_chars} chars (session `{target_sid}`, seq {start_seq}..={end_seq}) to: {}\nUse `read` with offset/limit to inspect it without bloating context.",
            out_path.display()
        ))
        .with_structured(meta.clone())
        .with_details(meta));
    }

    if offset >= total_chars {
        return Ok(ToolOutput::text(format!(
            "Offset {offset} is beyond end of recalled text ({total_chars} chars total)."
        ))
        .with_structured(json!({
            "mode": "read",
            "sessionId": target_sid,
            "startSeq": start_seq,
            "endSeq": end_seq,
            "totalChars": total_chars,
            "offset": offset,
            "truncated": false
        })));
    }

    let mut slice: String = full_text.chars().skip(offset).take(limit).collect();
    let mut returned_chars = slice.chars().count();
    let has_more_chars = offset + returned_chars < total_chars;
    let footer_for = |next: usize| {
        format!(
            "\n\n[Showing chars {offset}..{next} of {total_chars}. Call recall(seq={start_seq}, end_seq={end_seq}, offset={next}) to continue, or pass to_file=true.]"
        )
    };
    let footer_overhead = if has_more_chars {
        footer_for(offset + returned_chars).len()
    } else {
        0
    };
    if slice.len() + footer_overhead > MAX_READ_OUTPUT_BYTES {
        let worst_footer_len = footer_for(total_chars).len();
        let max_slice_bytes = MAX_READ_OUTPUT_BYTES.saturating_sub(worst_footer_len);
        let capped = truncate_utf8_bytes(&slice, max_slice_bytes).to_string();
        returned_chars = capped.chars().count();
        slice = capped;
    }

    let next_offset = if offset + returned_chars < total_chars {
        Some(offset + returned_chars)
    } else {
        None
    };

    let mut out = slice;
    if let Some(next) = next_offset {
        out.push_str(&footer_for(next));
    }

    let meta = json!({
        "mode": "read",
        "sessionId": target_sid,
        "startSeq": start_seq,
        "endSeq": end_seq,
        "totalChars": total_chars,
        "offset": offset,
        "nextOffset": next_offset,
        "truncated": next_offset.is_some(),
    });
    Ok(ToolOutput::text(out)
        .with_structured(meta.clone())
        .with_details(meta))
}

fn build_recall_tool(
    store: Arc<SessionStore>,
    default_search_limit: usize,
    default_read_char_limit: usize,
) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        RECALL_TOOL_NAME,
        "Search preserved session history or read verbatim entries by seq across compaction boundaries, lineage, or workspace sessions. Pass `query` to search compact snippets, or `seq` (with optional `end_seq`, `session_id`, `offset`, `limit`, `to_file`) to read verbatim entries.",
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords, file paths, or symbols to search across preserved history. Omit when reading by seq."
                },
                "scope": {
                    "type": "string",
                    "enum": ["current", "lineage", "workspace"],
                    "description": "Search scope when using query: 'current', 'lineage' (default), or 'workspace'."
                },
                "seq": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Starting event seq number to read verbatim."
                },
                "end_seq": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional inclusive ending seq number when reading (max 12 entries per call)."
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID when reading (defaults to current session; restricted to current lineage or workspace)."
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Character offset into rendered entry text for pagination when reading (default 0)."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Max hits when searching (default 8, max 20), or max inline characters when reading (default 6000, min 100, max 8192)."
                },
                "to_file": {
                    "type": "boolean",
                    "description": "When reading, write full verbatim text to a temp file and return its path instead of inlining."
                }
            }
        }),
        move |args, call_ctx| {
            let store = Arc::clone(&store);
            async move {
                let query_opt = args
                    .get("query")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string);
                let has_seq = args.get("seq").is_some();
                let seq_opt = if has_seq {
                    let seq = args
                        .get("seq")
                        .and_then(Value::as_u64)
                        .filter(|&v| v >= 1)
                        .ok_or("Invalid `seq` parameter: must be an integer >= 1.")?;
                    Some(seq)
                } else {
                    None
                };

                match (query_opt, seq_opt) {
                    (Some(_), Some(_)) => {
                        Err("Provide either `query` to search or `seq` to read, not both.".into())
                    }
                    (None, Some(start_seq)) => {
                        execute_recall_read(
                            &store,
                            call_ctx.session_id(),
                            &args,
                            start_seq,
                            default_read_char_limit,
                        )
                        .await
                    }
                    (Some(query_str), None) => {
                        let scope = args
                            .get("scope")
                            .and_then(Value::as_str)
                            .unwrap_or("lineage");
                        let limit = args
                            .get("limit")
                            .and_then(Value::as_u64)
                            .map(|v| (v as usize).clamp(1, MAX_SEARCH_LIMIT))
                            .unwrap_or_else(|| default_search_limit.clamp(1, MAX_SEARCH_LIMIT));
                        execute_recall_search(
                            &store,
                            call_ctx.session_id().map(ToString::to_string),
                            query_str,
                            scope,
                            limit,
                        )
                    }
                    (None, None) => {
                        Err("Provide either `query` to search history or `seq` (>= 1) to read verbatim entries.".into())
                    }
                }
            }
        },
    )
    .with_exposure(ToolExposure::Direct)
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_annotations(ToolAnnotations {
        read_only: true,
        destructive: false,
        idempotent: true,
        open_world: false,
    })
    .with_prompt_snippet("Search or read preserved verbatim session history across compaction boundaries")
}

#[derive(Default, Clone, Copy)]
pub struct RecallPlugin;

impl Plugin for RecallPlugin {
    type Config = RecallConfig;

    fn name(&self) -> Option<&str> {
        Some("recall")
    }

    fn inject(&self) -> Inject {
        Inject::from(["sessions", "tools"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            if !config.enabled {
                return Ok(None);
            }
            let sessions = ctx.sessions()?;
            let tools = ctx.tools()?;

            ctx.effect("recall.register", move || {
                Ok::<_, mizar_tools::ToolError>(Some(tools.register(build_recall_tool(
                    sessions,
                    config.default_search_limit,
                    config.read_char_limit,
                ))?))
            })?;

            Ok(None)
        })
    }
}
