//! 无损情景回溯插件（综合 `dim` `Recall` + `maka` N-gram Recall + `obelisk` 防爆护栏）。
//!
//! - 注册两个只读工具：`recall_search` 与 `recall_read`；
//! - 默认暴露级别为 `ToolExposure::Deferred`，短会话零 Schema 上下文开销；
//! - 一旦会话发生首次 `ToolResultPrune` / `Compaction`，或处于 `parent_id` 分叉/换段谱系中，
//!   自动将该会话的 `recall_search` / `recall_read` 激活为模型可见；
//! - 优先调用 SQLite 后端的 FTS5 1..=3 n-gram 索引，未挂载持久化时自动回退至内存扫描；
//! - 严格限定单次返回字符上限、跨会话作用域校验与 `is_current_session` 标记，防止回溯反向撑爆上下文。

use std::collections::HashSet;
use std::sync::Arc;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_agent_loop::{AgentPreStepEvent, EVENT_AGENT_PRE_STEP};
use mizar_llm::ContentBlock;
use mizar_session::{
    EVENT_SESSION_CREATED, EVENT_SESSION_EVENT, EVENT_SESSION_LOADED, RecallSearchHit,
    RecallSearchQuery, Session, SessionContextExt, SessionCreatedPayload, SessionEntry,
    SessionEvent, SessionEventPayload, SessionListQuery, SessionLoadedPayload, SessionStore,
};
use mizar_tools::{
    ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolExposure, ToolOutput, ToolsContextExt,
};
use serde_json::{Value, json};

pub const RECALL_SEARCH_TOOL_NAME: &str = "recall_search";
pub const RECALL_READ_TOOL_NAME: &str = "recall_read";

const MAX_SEARCH_LIMIT: usize = 20;
const MAX_READ_RANGE_ENTRIES: u64 = 12;
const DEFAULT_READ_CHAR_LIMIT: usize = 6_000;
const MAX_READ_CHAR_LIMIT: usize = 8_192;
const FALLBACK_SNIPPET_CHARS: usize = 240;

#[derive(Debug, Clone)]
pub struct RecallConfig {
    /// 是否启用情景回溯工具（默认 true）。
    pub enabled: bool,
    /// 发生剪枝/压缩或存在父会话谱系时自动向当前会话激活 `recall_*`（默认 true）。
    pub auto_activate_on_compaction: bool,
    pub default_search_limit: usize,
    pub read_char_limit: usize,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_activate_on_compaction: true,
            default_search_limit: 8,
            read_char_limit: DEFAULT_READ_CHAR_LIMIT,
        }
    }
}

fn should_activate_recall(session: &Session) -> bool {
    let header = session.header();
    header.compacted_up_to_seq.is_some()
        || header.parent_id.is_some()
        || !session.pruned_map().is_empty()
}

fn render_entry_verbatim(entry: &SessionEntry) -> Option<String> {
    match &entry.event {
        SessionEvent::User { text, images } => {
            if images.is_empty() {
                Some(format!("[seq #{} | user]\n{}", entry.seq, text))
            } else {
                Some(format!(
                    "[seq #{} | user]\n[图片 {} 张]\n{}",
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

fn build_recall_search_tool(store: Arc<SessionStore>, default_limit: usize) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        RECALL_SEARCH_TOOL_NAME,
        "Search preserved session history across compaction boundaries, pruned tool outputs, parent/forked lineage, or other sessions in the same workspace. Returns compact snippets with session_id and seq handles for recall_read.",
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords, file paths, symbol names, or Chinese/English phrases to search for"
                },
                "scope": {
                    "type": "string",
                    "enum": ["current", "lineage", "workspace"],
                    "description": "Search scope: 'current' (only this session), 'lineage' (this session + parent/fork ancestors, default), or 'workspace' (all sessions in the same workspace)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of hits to return (default 8, max 20)"
                }
            },
            "required": ["query"]
        }),
        move |args, call_ctx| {
            let store = Arc::clone(&store);
            async move {
                let query_str = args
                    .get("query")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .ok_or("缺少 query 参数")?
                    .to_string();
                let scope = args
                    .get("scope")
                    .and_then(Value::as_str)
                    .unwrap_or("lineage");
                let limit = args
                    .get("limit")
                    .and_then(Value::as_u64)
                    .map(|v| (v as usize).clamp(1, MAX_SEARCH_LIMIT))
                    .unwrap_or_else(|| default_limit.clamp(1, MAX_SEARCH_LIMIT));

                let caller_sid = call_ctx.session_id().map(ToString::to_string);

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
                session_ids: session_ids.clone(),
                cwd: cwd.clone(),
                limit,
            };

            let mut hits = match store.search_recall(&search_query) {
                Ok(Some(db_hits)) => db_hits,
                Ok(None) => fallback_in_memory_search(&store, &search_query),
                Err(_) => fallback_in_memory_search(&store, &search_query),
            };

            // 若在 workspace 范围下但无 SQLite，补扫所有已列出的会话头（已在内存者）
            if hits.is_empty()
                && scope == "workspace"
                && cwd.is_some()
                && let Ok(headers) = store.list_headers(&SessionListQuery {
                    cwd: cwd.clone(),
                    limit: Some(50),
                })
            {
                let mut ids = HashSet::new();
                for h in headers {
                    ids.insert(h.id);
                }
                let q = RecallSearchQuery {
                    query: query_str.clone(),
                    session_ids: ids.into_iter().collect(),
                    cwd: None,
                    limit,
                };
                hits = fallback_in_memory_search(&store, &q);
            }

            if hits.is_empty() {
                return Ok(ToolOutput::text(format!(
                    "No preserved history matched query {:?} in scope '{scope}'.",
                    query_str
                ))
                .with_structured(json!({
                    "query": query_str,
                    "scope": scope,
                    "hits": []
                })));
            }

            let mut lines = Vec::with_capacity(hits.len() + 2);
            lines.push(format!(
                "Found {} preserved history hit(s) for {:?} (scope={scope}):",
                hits.len(),
                query_str
            ));
            let mut structured_hits = Vec::with_capacity(hits.len());
            for hit in &hits {
                let is_current = caller_sid.as_deref() == Some(hit.session_id.as_str());
                let origin = if is_current {
                    "current".to_string()
                } else {
                    format!("session:{}", hit.session_id)
                };
                lines.push(format!(
                    "- [{origin} seq #{} | {}] {}",
                    hit.seq, hit.kind, hit.snippet
                ));
                structured_hits.push(json!({
                    "sessionId": hit.session_id,
                    "sessionTitle": hit.session_title,
                    "isCurrentSession": is_current,
                    "seq": hit.seq,
                    "kind": hit.kind,
                    "snippet": hit.snippet,
                }));
            }
            lines.push("Use recall_read(seq=..., session_id=...) to inspect full verbatim entries.".into());

            let meta = json!({
                "query": query_str,
                "scope": scope,
                "hits": structured_hits,
            });
            Ok(ToolOutput::text(lines.join("\n"))
                .with_structured(meta.clone())
                .with_details(meta))
            }
        },
    )
    .with_exposure(ToolExposure::Deferred)
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_annotations(ToolAnnotations {
        read_only: true,
        destructive: false,
        idempotent: true,
        open_world: false,
    })
    .with_prompt_snippet("Search compacted or archived session history")
}

fn build_recall_read_tool(store: Arc<SessionStore>, default_char_limit: usize) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        RECALL_READ_TOOL_NAME,
        "Read preserved verbatim session log entries by seq number or range, including entries that were pruned or compacted out of the active context window. Supports character offset/limit pagination or exporting large entries to a temporary file.",
        json!({
            "type": "object",
            "properties": {
                "seq": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Starting event seq number to read"
                },
                "end_seq": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional inclusive ending seq number (at most 12 entries per call)"
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID (defaults to current session; must belong to current lineage or workspace)"
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Character offset into the rendered entry text for pagination (default 0)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 8192,
                    "description": "Maximum characters to return inline (default 6000, max 8192)"
                },
                "to_file": {
                    "type": "boolean",
                    "description": "If true, write full verbatim text to a temporary file and return its path instead of inlining it"
                }
            },
            "required": ["seq"]
        }),
        move |args, call_ctx| {
            let store = Arc::clone(&store);
            async move {
                let start_seq = args
                    .get("seq")
                    .and_then(Value::as_u64)
                    .filter(|&v| v >= 1)
                    .ok_or("缺少有效的 seq 参数（需 >= 1）")?;
                let requested_end = args
                    .get("end_seq")
                    .and_then(Value::as_u64)
                    .unwrap_or(start_seq)
                    .max(start_seq);
                let end_seq = requested_end.min(start_seq.saturating_add(MAX_READ_RANGE_ENTRIES - 1));
                let offset = args
                    .get("offset")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                let limit = args
                    .get("limit")
                    .and_then(Value::as_u64)
                    .map(|v| (v as usize).clamp(100, MAX_READ_CHAR_LIMIT))
                    .unwrap_or_else(|| default_char_limit.clamp(100, MAX_READ_CHAR_LIMIT));
                let to_file = args
                    .get("to_file")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);

                let caller_sid = call_ctx.session_id();
            let target_sid = args
                .get("session_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .or_else(|| caller_sid.map(ToString::to_string))
                .ok_or("缺少 session_id 参数且当前无活跃会话")?;

            if !is_session_authorized(&store, caller_sid, &target_sid) {
                return Err(format!("无权跨工作区访问会话 `{target_sid}` 的历史记录").into());
            }

            let session = store
                .inspect_readonly(&target_sid)
                .ok_or_else(|| format!("未找到会话 `{target_sid}`"))?;

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
                    .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                    .take(32)
                    .collect();
                let out_path = std::env::temp_dir().join(format!(
                    "mizar_recall_{safe_sid}_{start_seq}_{end_seq}_{}.txt",
                    std::process::id()
                ));
                tokio::fs::write(&out_path, full_text.as_bytes()).await?;
                let meta = json!({
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
                    "sessionId": target_sid,
                    "startSeq": start_seq,
                    "endSeq": end_seq,
                    "totalChars": total_chars,
                    "offset": offset,
                    "truncated": false
                })));
            }

            let slice: String = full_text.chars().skip(offset).take(limit).collect();
            let returned_chars = slice.chars().count();
            let next_offset = if offset + returned_chars < total_chars {
                Some(offset + returned_chars)
            } else {
                None
            };

            let mut out = slice;
            if let Some(next) = next_offset {
                out.push_str(&format!(
                    "\n\n[Showing chars {offset}..{} of {total_chars}. Call recall_read(seq={start_seq}, end_seq={end_seq}, offset={next}) to continue, or pass to_file=true.]",
                    offset + returned_chars
                ));
            }

            let meta = json!({
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
        },
    )
    .with_exposure(ToolExposure::Deferred)
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_annotations(ToolAnnotations {
        read_only: true,
        destructive: false,
        idempotent: true,
        open_world: false,
    })
    .with_prompt_snippet("Read verbatim historical session entries by seq")
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
            let mut cleanups = Vec::new();

            {
                let rt = Arc::clone(&tools);
                let store = Arc::clone(&sessions);
                let limit = config.default_search_limit;
                cleanups.push(ctx.effect("recall.register-search", move || {
                    Ok::<_, mizar_tools::ToolError>(Some(
                        rt.register(build_recall_search_tool(store, limit))?,
                    ))
                })?);
            }
            {
                let rt = Arc::clone(&tools);
                let store = Arc::clone(&sessions);
                let char_limit = config.read_char_limit;
                cleanups.push(ctx.effect("recall.register-read", move || {
                    Ok::<_, mizar_tools::ToolError>(Some(
                        rt.register(build_recall_read_tool(store, char_limit))?,
                    ))
                })?);
            }

            for session in sessions.list() {
                if should_activate_recall(&session) {
                    tools.activate_tools(
                        &session.id,
                        &[
                            RECALL_SEARCH_TOOL_NAME.to_string(),
                            RECALL_READ_TOOL_NAME.to_string(),
                        ],
                    );
                }
            }

            if config.auto_activate_on_compaction {
                let rt_created = Arc::clone(&tools);
                let ctx_created = ctx.clone();
                cleanups.push(ctx.effect("recall.on-session-created", move || {
                    let h = ctx_created.on(
                        EVENT_SESSION_CREATED,
                        move |ev: &SessionCreatedPayload| {
                            if should_activate_recall(&ev.session) {
                                rt_created.activate_tools(
                                    &ev.session.id,
                                    &[
                                        RECALL_SEARCH_TOOL_NAME.to_string(),
                                        RECALL_READ_TOOL_NAME.to_string(),
                                    ],
                                );
                            }
                        },
                    )?;
                    Ok::<_, cordis::CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?);

                let rt_loaded = Arc::clone(&tools);
                let ctx_loaded = ctx.clone();
                cleanups.push(ctx.effect("recall.on-session-loaded", move || {
                    let h =
                        ctx_loaded.on(EVENT_SESSION_LOADED, move |ev: &SessionLoadedPayload| {
                            if should_activate_recall(&ev.session) {
                                rt_loaded.activate_tools(
                                    &ev.session.id,
                                    &[
                                        RECALL_SEARCH_TOOL_NAME.to_string(),
                                        RECALL_READ_TOOL_NAME.to_string(),
                                    ],
                                );
                            }
                        })?;
                    Ok::<_, cordis::CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?);

                let rt_ev = Arc::clone(&tools);
                let ctx_ev = ctx.clone();
                cleanups.push(ctx.effect("recall.on-session-event", move || {
                    let h = ctx_ev.on(EVENT_SESSION_EVENT, move |ev: &SessionEventPayload| {
                        if matches!(
                            ev.entry.event,
                            SessionEvent::Compaction { .. } | SessionEvent::ToolResultPrune { .. }
                        ) {
                            rt_ev.activate_tools(
                                &ev.session.id,
                                &[
                                    RECALL_SEARCH_TOOL_NAME.to_string(),
                                    RECALL_READ_TOOL_NAME.to_string(),
                                ],
                            );
                        }
                    })?;
                    Ok::<_, cordis::CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?);

                let rt_step = Arc::clone(&tools);
                let ctx_step = ctx.clone();
                cleanups.push(ctx.effect("recall.on-pre-step", move || {
                    let h = ctx_step.on_serial_with::<AgentPreStepEvent, (), _, _>(
                        EVENT_AGENT_PRE_STEP,
                        move |ev: AgentPreStepEvent| {
                            let rt = Arc::clone(&rt_step);
                            async move {
                                if should_activate_recall(&ev.session) {
                                    rt.activate_tools(
                                        &ev.session.id,
                                        &[
                                            RECALL_SEARCH_TOOL_NAME.to_string(),
                                            RECALL_READ_TOOL_NAME.to_string(),
                                        ],
                                    );
                                }
                                None
                            }
                        },
                        cordis::EventOptions {
                            priority: 20,
                            ..Default::default()
                        },
                    )?;
                    Ok::<_, cordis::CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?);
            }

            Ok(Some(Cleanup::async_fn(move || async move {
                for h in cleanups {
                    h.dispose().await;
                }
            })))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cordis::Service;
    use mizar_session::{PrunedToolResult, SessionCreateOptions, SessionStore};
    use mizar_session_sqlite::{SqlitePersistenceConfig, SqlitePersistencePlugin};
    use mizar_tools::{ToolExecution, ToolRuntime};

    #[tokio::test]
    async fn auto_activates_on_compaction_and_reads_pruned_verbatim_text() {
        let app = Context::new();
        app.plugin(
            SqlitePersistencePlugin,
            SqlitePersistenceConfig::new(":memory:"),
        )
        .await
        .unwrap();
        app.plugin(SessionStore::plugin(), ()).await.unwrap();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app.plugin(RecallPlugin, RecallConfig::default())
            .await
            .unwrap();

        let store = app.sessions().unwrap();
        let tools = app.tools().unwrap();

        let s = store.create_with(SessionCreateOptions {
            id: Some("rec-1".into()),
            cwd: Some("/ws/test".into()),
            ..Default::default()
        });

        // 压缩/剪枝前 recall 工具默认延迟，不占活跃声明集
        assert!(
            !tools
                .list_declared(Some(&s.id))
                .iter()
                .any(|t| t.name == RECALL_SEARCH_TOOL_NAME)
        );

        s.append(SessionEvent::user("请确认数据库连接池上限与回溯策略"));
        s.append(SessionEvent::ToolResult {
            tool_call_id: "tc-1".into(),
            name: "read".into(),
            content: "MAX_POOL_CONNECTIONS = 64 // 关键配置原文".into(),
            is_error: false,
        });
        let prune_seq = s.append_prune(vec![PrunedToolResult {
            target_seq: 2,
            replacement: "[pruned; see seq #2]".into(),
            original_chars: 42,
        }]);
        assert!(prune_seq.is_some());

        // 发生剪枝后自动激活 recall_search 与 recall_read
        let declared_names: Vec<String> = tools
            .list_declared(Some(&s.id))
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(declared_names.contains(&RECALL_SEARCH_TOOL_NAME.to_string()));
        assert!(declared_names.contains(&RECALL_READ_TOOL_NAME.to_string()));

        // 再发生 Compaction，依然可穿透搜索并读回被剪枝+被压缩的原始文本
        s.compact("前情已压缩", 0, Some(500));
        s.flush_async().await.unwrap();

        let search_res = tools
            .execute(ToolExecution::new(
                RECALL_SEARCH_TOOL_NAME,
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

        let read_res = tools
            .execute(ToolExecution::new(
                RECALL_READ_TOOL_NAME,
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

        app.fiber().dispose().await;
    }

    #[tokio::test]
    async fn blocks_cross_workspace_session_read_but_allows_lineage() {
        let app = Context::new();
        app.plugin(SessionStore::plugin(), ()).await.unwrap();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app.plugin(RecallPlugin, RecallConfig::default())
            .await
            .unwrap();

        let store = app.sessions().unwrap();
        let tools = app.tools().unwrap();

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
                RECALL_READ_TOOL_NAME,
                json!({ "seq": 1, "session_id": "other-ws" }),
                Some(child.id.clone()),
            ))
            .await;
        assert!(denied.is_error);

        // 同谱系父会话读取应放行
        let allowed = tools
            .execute(ToolExecution::new(
                RECALL_READ_TOOL_NAME,
                json!({ "seq": 1, "session_id": "parent-s" }),
                Some(child.id.clone()),
            ))
            .await;
        assert!(!allowed.is_error, "{}", allowed.content);
        assert!(allowed.content.contains("alpha_secret"));

        app.fiber().dispose().await;
    }

    #[tokio::test]
    async fn readonly_queries_do_not_hydrate_or_repair_cold_sessions_and_resume_rearms_tools() {
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
            let app1 = Context::new();
            app1.plugin(
                SqlitePersistencePlugin,
                SqlitePersistenceConfig::new(&db_path),
            )
            .await
            .unwrap();
            app1.plugin(SessionStore::plugin(), ()).await.unwrap();
            let store1 = app1.sessions().unwrap();
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
            app1.fiber().dispose().await;
        }

        // 2. 新进程挂载 RecallPlugin：从同工作区另一会话查询 cold-arch，不得水合也不得修补落盘
        let app2 = Context::new();
        app2.plugin(
            SqlitePersistencePlugin,
            SqlitePersistenceConfig::new(&db_path),
        )
        .await
        .unwrap();
        app2.plugin(SessionStore::plugin(), ()).await.unwrap();
        app2.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app2.plugin(RecallPlugin, RecallConfig::default())
            .await
            .unwrap();

        let store2 = app2.sessions().unwrap();
        let tools2 = app2.tools().unwrap();
        let caller = store2.create_with(SessionCreateOptions {
            id: Some("caller-ro".into()),
            cwd: Some("/ws/ro".into()),
            ..Default::default()
        });

        let read_res = tools2
            .execute(ToolExecution::new(
                RECALL_READ_TOOL_NAME,
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

        // 3. 真正 resume (get) cold-arch 时，应在未运行任何 Turn 前立即激活 recall 工具
        let resumed = store2.get("cold-arch").unwrap();
        assert_eq!(resumed.log().len(), 4);
        assert!(
            tools2
                .list_declared(Some("cold-arch"))
                .iter()
                .any(|t| t.name == RECALL_SEARCH_TOOL_NAME)
        );

        app2.fiber().dispose().await;
        let _ = std::fs::remove_dir_all(&dir);
    }
}
