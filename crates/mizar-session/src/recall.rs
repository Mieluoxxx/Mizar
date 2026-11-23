//! 无损情景回溯插件（综合 `dim` `Recall` + `maka` N-gram Recall + `obelisk` 防爆护栏）。
//!
//! - 注册单一常驻只读工具：`recall`（固定 `ToolExposure::Direct`，全程冻结工具声明集不动态改表）；
//! - 同时支持按 `query` 搜索摘要片段（穿透压缩与谱系边界）以及按 `seq`/`session_id` 分页拉取原文；
//! - 优先调用 SQLite 后端的 FTS5 1..=3 n-gram 索引，未挂载持久化时自动回退至内存扫描；
//! - 严格限定单次返回字符上限、跨会话作用域校验与 `is_current_session` 标记，防止回溯反向撑爆上下文。

use std::sync::Arc;

use crate::{
    RecallSearchHit, RecallSearchQuery, SessionEntry, SessionEvent, SessionListQuery, SessionStore,
};
use mizar_tools::{
    ToolAnnotations, ToolDefinition, ToolError, ToolExecutionMode, ToolExposure, ToolOutput,
    ToolRuntime,
};
use mizar_types::hooks::{BoxError, HookGuard};
use mizar_types::llm::ContentBlock;
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
        | SessionEvent::Selection { .. }
        | SessionEvent::Usage { .. }
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

async fn fallback_in_memory_search(
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

    let target_sessions: Vec<crate::LoadedSession> = if !query.session_ids.is_empty() {
        let mut out = Vec::with_capacity(query.session_ids.len());
        for id in &query.session_ids {
            if let Some(sess) = store.inspect_readonly_async(id).await {
                out.push(sess);
            }
        }
        out
    } else if let Ok(headers) = store
        .list_headers_async(SessionListQuery {
            cwd: query.cwd.clone(),
            limit: Some(50),
            ..Default::default()
        })
        .await
    {
        let mut out = Vec::with_capacity(headers.len());
        for h in &headers {
            if let Some(sess) = store.inspect_readonly_async(&h.id).await {
                out.push(sess);
            }
        }
        out
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

async fn is_session_authorized(
    store: &SessionStore,
    caller_sid: Option<&str>,
    target_sid: &str,
) -> bool {
    let Some(caller_id) = caller_sid else {
        return true;
    };
    if caller_id == target_sid {
        return true;
    }
    let lineage = store.lineage_session_ids_async(caller_id).await;
    if lineage.iter().any(|id| id == target_sid) {
        return true;
    }
    let caller_cwd = match store
        .inspect_header_async(caller_id)
        .await
        .and_then(|h| h.cwd)
    {
        Some(cwd) => Some(cwd),
        None => store.default_cwd(),
    };
    let target_cwd = store
        .inspect_header_async(target_sid)
        .await
        .and_then(|h| h.cwd);
    match (caller_cwd, target_cwd) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

async fn execute_recall_search(
    store: Arc<SessionStore>,
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
            let ws = match caller_sid.as_deref() {
                Some(sid) => store.inspect_header_async(sid).await.and_then(|h| h.cwd),
                None => None,
            }
            .or_else(|| store.default_cwd());
            (Vec::new(), ws)
        }
        _ => match &caller_sid {
            Some(sid) => (store.lineage_session_ids_async(sid).await, None),
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
    let mut hits = match store.search_recall_async(search_query.clone()).await {
        Ok(Some(db_hits)) => db_hits,
        Ok(None) | Err(_) => fallback_in_memory_search(&store, &search_query).await,
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
    // 预留头部行（最多 ~96 字节）与末尾提示行及换行符，保证总输出 <= MAX_SEARCH_OUTPUT_BYTES
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
    store: Arc<SessionStore>,
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

    if !is_session_authorized(&store, caller_sid, &target_sid).await {
        return Err(format!(
            "Access denied: session `{target_sid}` is outside current lineage or workspace."
        )
        .into());
    }

    let session = store
        .inspect_readonly_async(&target_sid)
        .await
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
                            store,
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
                            store,
                            call_ctx.session_id().map(ToString::to_string),
                            query_str,
                            scope,
                            limit,
                        )
                        .await
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

/// 注册 `recall` 工具到 `ToolRuntime`；`config.enabled == false` 时返回 `Ok(None)`。
pub fn register_recall_tool(
    tools: &Arc<ToolRuntime>,
    sessions: Arc<SessionStore>,
    config: RecallConfig,
) -> Result<Option<HookGuard>, ToolError> {
    if !config.enabled {
        return Ok(None);
    }
    let guard = tools.register(build_recall_tool(
        sessions,
        config.default_search_limit,
        config.read_char_limit,
    ))?;
    Ok(Some(guard))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlite::SqliteSessionPersistence;
    use crate::{SessionCreateOptions, SessionStore};
    use mizar_tools::{ToolExecution, ToolRuntime};
    use mizar_types::hooks::BoxFuture;

    #[tokio::test]
    async fn static_recall_freezes_tool_list_across_compaction_and_reads_compacted_verbatim_text() {
        let backend = SqliteSessionPersistence::open_async(":memory:")
            .await
            .unwrap();
        let store = SessionStore::with_persistence(backend.clone());
        let tools = ToolRuntime::new();
        let _recall_guard =
            register_recall_tool(&tools, Arc::clone(&store), RecallConfig::default())
                .unwrap()
                .unwrap();

        let s = store.create_with(SessionCreateOptions {
            id: Some("rec-1".into()),
            cwd: Some("/ws/test".into()),
            ..Default::default()
        });

        // 静态常驻：压缩/剪枝前即已存在于 Declared 列表，全程工具列表冻结不变
        let declared_before: Vec<String> = tools
            .list_declared(Some(&s.id))
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(declared_before, vec![RECALL_TOOL_NAME.to_string()]);

        s.append(SessionEvent::user("请确认数据库连接池上限与回溯策略"));
        s.append(SessionEvent::ToolResult {
            tool_call_id: "tc-1".into(),
            name: "read".into(),
            content: "MAX_POOL_CONNECTIONS = 64 // 关键配置原文".into(),
            is_error: false,
        });

        // 发生 Compaction，Declared 工具列表完全不变（零 KV Cache 破坏）
        s.compact("前情已压缩", 0, Some(500));
        s.flush_async().await.unwrap();

        let declared_after: Vec<String> = tools
            .list_declared(Some(&s.id))
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(declared_after, declared_before);

        let search_res = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "query": "MAX_POOL_CONNECTIONS", "scope": "current" }),
                Some(s.id.clone()),
            ))
            .await;
        assert!(!search_res.is_error, "{}", search_res.content);
        assert!(
            search_res.content.contains("seq #2"),
            "{}",
            search_res.content
        );
        assert!(
            search_res.content.contains("recall(seq="),
            "{}",
            search_res.content
        );

        let read_res = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "seq": 2 }),
                Some(s.id.clone()),
            ))
            .await;
        assert!(!read_res.is_error, "{}", read_res.content);
        assert!(
            read_res
                .content
                .contains("MAX_POOL_CONNECTIONS = 64 // 关键配置原文"),
            "{}",
            read_res.content
        );

        // 校验参数互斥与缺参错误处理
        let both_err = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "query": "MAX_POOL", "seq": 2 }),
                Some(s.id.clone()),
            ))
            .await;
        assert!(both_err.is_error);

        let empty_err = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({}),
                Some(s.id.clone()),
            ))
            .await;
        assert!(empty_err.is_error);

        // 校验分页提示使用统一工具名 recall(...)
        s.append(SessionEvent::ToolResult {
            tool_call_id: "tc-2".into(),
            name: "read".into(),
            content: "X".repeat(350),
            is_error: false,
        });
        let tc2_seq = s.last_seq();
        let page_res = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "seq": tc2_seq, "limit": 100 }),
                Some(s.id.clone()),
            ))
            .await;
        assert!(!page_res.is_error, "{}", page_res.content);
        assert!(
            page_res
                .content
                .contains(&format!("Call recall(seq={tc2_seq}")),
            "{}",
            page_res.content
        );

        let _ = store.flush_async(None).await;
        let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;
    }

    #[tokio::test]
    async fn blocks_cross_workspace_session_read_but_allows_lineage() {
        let store = SessionStore::new();
        let tools = ToolRuntime::new();
        let _recall_guard =
            register_recall_tool(&tools, Arc::clone(&store), RecallConfig::default())
                .unwrap()
                .unwrap();

        let parent = store.create_with(SessionCreateOptions {
            id: Some("parent-s".into()),
            cwd: Some("/ws/a".into()),
            ..Default::default()
        });
        parent.append(SessionEvent::user("父会话的原始需求 alpha_secret"));
        let child = store
            .fork("parent-s", 1, SessionCreateOptions::default())
            .unwrap();

        let other = store.create_with(SessionCreateOptions {
            id: Some("other-ws".into()),
            cwd: Some("/ws/b".into()),
            ..Default::default()
        });
        other.append(SessionEvent::user("另一个工作区的机密内容"));

        // 跨工作区读取应被拒绝
        let denied = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "seq": 1, "session_id": "other-ws" }),
                Some(child.id.clone()),
            ))
            .await;
        assert!(denied.is_error);

        // 同谱系父会话读取应放行
        let allowed = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "seq": 1, "session_id": "parent-s" }),
                Some(child.id.clone()),
            ))
            .await;
        assert!(!allowed.is_error, "{}", allowed.content);
        assert!(allowed.content.contains("alpha_secret"));
    }

    #[tokio::test]
    async fn readonly_queries_do_not_hydrate_or_repair_cold_sessions_and_resume_keeps_static_recall()
     {
        let dir = std::env::temp_dir().join(format!(
            "mizar_recall_readonly_{}_{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("sessions.sqlite");

        // 1. 首轮进程创建一个含压缩历史与悬挂工具调用的会话并落盘
        {
            let backend1 = SqliteSessionPersistence::open_async(&db_path)
                .await
                .unwrap();
            let store1 = SessionStore::with_persistence(backend1.clone());
            let s1 = store1.create_with(SessionCreateOptions {
                id: Some("cold-arch".into()),
                cwd: Some("/ws/ro".into()),
                ..Default::default()
            });
            s1.append(SessionEvent::user("历史中的 secret_token_xyz"));
            s1.append(SessionEvent::Assistant {
                blocks: vec![ContentBlock::ToolCall {
                    id: "hang-call".into(),
                    name: "bash".into(),
                    arguments: "{}".into(),
                }],
                replay: None,
                usage: None,
            });
            s1.compact("压缩冷会话", 2, Some(100));
            s1.flush_async().await.unwrap();
            let _ = tokio::task::spawn_blocking(move || backend1.shutdown()).await;
        }

        // 2. 新进程挂载 recall 工具：从同工作区另一会话查询 cold-arch，不得水合也不得修补落盘
        let backend2 = SqliteSessionPersistence::open_async(&db_path)
            .await
            .unwrap();
        let store2 = SessionStore::with_persistence(backend2.clone());
        let tools2 = ToolRuntime::new();
        let _recall_guard =
            register_recall_tool(&tools2, Arc::clone(&store2), RecallConfig::default())
                .unwrap()
                .unwrap();
        let caller = store2.create_with(SessionCreateOptions {
            id: Some("caller-ro".into()),
            cwd: Some("/ws/ro".into()),
            ..Default::default()
        });

        let read_res = tools2
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "seq": 1, "session_id": "cold-arch" }),
                Some(caller.id.clone()),
            ))
            .await;
        assert!(!read_res.is_error, "{}", read_res.content);
        assert!(read_res.content.contains("secret_token_xyz"));
        assert!(!store2.contains_in_memory("cold-arch"));
        // 确认未向 cold-arch 追加第 4 条（TOOL_OUTCOME_UNKNOWN）条目
        let cold_ro = store2.inspect_readonly("cold-arch").unwrap();
        assert_eq!(cold_ro.entries.len(), 3);

        // 3. 静态 recall 在 resume 前后均已声明
        let resumed = store2.get("cold-arch").unwrap();
        assert_eq!(resumed.log().len(), 4);
        assert!(
            tools2
                .list_declared(Some("cold-arch"))
                .iter()
                .any(|t| t.name == RECALL_TOOL_NAME)
        );

        let _ = store2.flush_async(None).await;
        let _ = tokio::task::spawn_blocking(move || backend2.shutdown()).await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn empty_fts5_search_does_not_fallback_to_cold_scan() {
        use crate::{
            LoadedSession, PersistenceError, PersistenceResult, SessionHeader, SessionHeaderPatch,
            SessionPersistence,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct EmptyFtsBackend {
            list_calls: AtomicUsize,
            load_calls: AtomicUsize,
            search_calls: AtomicUsize,
        }

        impl SessionPersistence for EmptyFtsBackend {
            fn create_session(&self, _: &SessionHeader) -> PersistenceResult<()> {
                Ok(())
            }
            fn update_header(&self, _: &str, _: SessionHeaderPatch) -> PersistenceResult<()> {
                Ok(())
            }
            fn load_session(&self, _: &str) -> PersistenceResult<Option<LoadedSession>> {
                self.load_calls.fetch_add(1, Ordering::SeqCst);
                Ok(None)
            }
            fn load_header(&self, _: &str) -> PersistenceResult<Option<SessionHeader>> {
                Ok(None)
            }
            fn list_sessions(&self, _: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
                self.list_calls.fetch_add(1, Ordering::SeqCst);
                Ok(vec![SessionHeader::new("other-cold", 1)])
            }
            fn append_events(&self, _: &str, _: &[SessionEntry]) -> PersistenceResult<()> {
                Ok(())
            }
            fn flush(&self, _: Option<&str>) -> PersistenceResult<()> {
                Ok(())
            }
            fn flush_async(&self, _: Option<String>) -> BoxFuture<'static, PersistenceResult<()>> {
                Box::pin(async { Ok(()) })
            }
            fn fork_session(
                &self,
                from_id: &str,
                _: u64,
                _: &SessionHeader,
            ) -> PersistenceResult<LoadedSession> {
                Err(PersistenceError::NotFound(from_id.to_string()))
            }
            fn search_recall(
                &self,
                _: &RecallSearchQuery,
            ) -> PersistenceResult<Option<Vec<RecallSearchHit>>> {
                self.search_calls.fetch_add(1, Ordering::SeqCst);
                Ok(Some(Vec::new()))
            }
        }

        let backend = Arc::new(EmptyFtsBackend {
            list_calls: AtomicUsize::new(0),
            load_calls: AtomicUsize::new(0),
            search_calls: AtomicUsize::new(0),
        });

        let store = SessionStore::with_persistence(backend.clone());
        let tools = ToolRuntime::new();
        let _recall_guard =
            register_recall_tool(&tools, Arc::clone(&store), RecallConfig::default())
                .unwrap()
                .unwrap();
        let caller = store.create_with(SessionCreateOptions {
            id: Some("fts-empty-caller".into()),
            cwd: Some("/ws/empty-fts".into()),
            ..Default::default()
        });
        caller.append(SessionEvent::user("无关文本"));

        let res = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "query": "不存在的词条", "scope": "workspace" }),
                Some(caller.id.clone()),
            ))
            .await;
        assert!(!res.is_error, "{}", res.content);
        assert!(res.content.contains("No preserved history matched"));
        assert_eq!(backend.search_calls.load(Ordering::SeqCst), 1);
        assert_eq!(backend.list_calls.load(Ordering::SeqCst), 0);
        assert_eq!(backend.load_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn fallback_runs_only_when_persistence_absent_or_errors_and_enforces_caps() {
        use crate::{
            LoadedSession, PersistenceError, PersistenceResult, SessionHeader, SessionHeaderPatch,
            SessionPersistence,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct ErrorFtsBackend {
            search_calls: AtomicUsize,
        }

        impl SessionPersistence for ErrorFtsBackend {
            fn create_session(&self, _: &SessionHeader) -> PersistenceResult<()> {
                Ok(())
            }
            fn update_header(&self, _: &str, _: SessionHeaderPatch) -> PersistenceResult<()> {
                Ok(())
            }
            fn load_session(&self, _: &str) -> PersistenceResult<Option<LoadedSession>> {
                Ok(None)
            }
            fn list_sessions(&self, _: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
                Ok(Vec::new())
            }
            fn append_events(&self, _: &str, _: &[SessionEntry]) -> PersistenceResult<()> {
                Ok(())
            }
            fn flush(&self, _: Option<&str>) -> PersistenceResult<()> {
                Ok(())
            }
            fn flush_async(&self, _: Option<String>) -> BoxFuture<'static, PersistenceResult<()>> {
                Box::pin(async { Ok(()) })
            }
            fn fork_session(
                &self,
                from_id: &str,
                _: u64,
                _: &SessionHeader,
            ) -> PersistenceResult<LoadedSession> {
                Err(PersistenceError::NotFound(from_id.to_string()))
            }
            fn search_recall(
                &self,
                _: &RecallSearchQuery,
            ) -> PersistenceResult<Option<Vec<RecallSearchHit>>> {
                self.search_calls.fetch_add(1, Ordering::SeqCst);
                Err(PersistenceError::Backend("fts unavailable".into()))
            }
        }

        let backend = Arc::new(ErrorFtsBackend {
            search_calls: AtomicUsize::new(0),
        });
        let store = SessionStore::with_persistence(backend.clone());
        let tools = ToolRuntime::new();
        let _recall_guard =
            register_recall_tool(&tools, Arc::clone(&store), RecallConfig::default())
                .unwrap()
                .unwrap();
        let s = store.create_with(SessionCreateOptions {
            id: Some("cap-sess".into()),
            cwd: Some("/ws/caps".into()),
            ..Default::default()
        });
        for i in 1..=30 {
            s.append(SessionEvent::user(format!(
                "ERR_FALLBACK_KEY #{i} 中文占位符内容_{}",
                "语".repeat(120)
            )));
        }

        // 1. FTS 出错时回退内存扫描并强制条数上限 (<= 20) 与字节上限 (<= 8 KiB)
        let search_res = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "query": "ERR_FALLBACK_KEY", "scope": "current", "limit": 99 }),
                Some(s.id.clone()),
            ))
            .await;
        assert!(!search_res.is_error, "{}", search_res.content);
        assert_eq!(backend.search_calls.load(Ordering::SeqCst), 1);
        assert!(search_res.content.len() <= MAX_SEARCH_OUTPUT_BYTES);
        let hits = search_res
            .structured_content
            .as_ref()
            .and_then(|v| v.get("hits"))
            .and_then(Value::as_array)
            .unwrap();
        assert!(!hits.is_empty() && hits.len() <= MAX_SEARCH_LIMIT);

        // 2. 多字节中文读取超出 16 KiB 时，按 UTF-8 字节硬上限截断并保持 nextOffset 准确
        let big_seq = 31u64;
        s.append(SessionEvent::user("汉".repeat(7_000)));
        let page1 = tools
            .execute(ToolExecution::new(
                RECALL_TOOL_NAME,
                json!({ "seq": big_seq, "limit": 8192 }),
                Some(s.id.clone()),
            ))
            .await;
        assert!(!page1.is_error, "{}", page1.content);
        assert!(page1.content.len() <= MAX_READ_OUTPUT_BYTES);
        assert!(
            page1
                .structured_content
                .as_ref()
                .and_then(|v| v.get("nextOffset"))
                .and_then(Value::as_u64)
                .is_some()
        );
    }
}
