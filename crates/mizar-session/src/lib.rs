//! 会话日志（对应 `NamiDSH/packages/session`）。
//!
//! 会话 = 一条只追加的事件日志。发给模型的消息列表是从日志「推导」出来的。
//! 支持可选的持久化后端（`SessionPersistence`）、上下文压缩（`Compaction`）
//! 以及同源承载 `LogOnly` 审计/轨迹事件。

pub mod model;
pub mod persistence;
pub mod recall;
pub mod sqlite;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use mizar_types::hooks::{HookGuard, SyncEventBus};
use mizar_types::llm::{ContentBlock, Message, ReasoningEffort};
use uuid::Uuid;

pub use model::{
    CompactionDigest, EntryVisibility, LoadedSession, PinnedInput, RecallSearchHit,
    RecallSearchQuery, SESSION_FORMAT_VERSION, SessionCreateOptions, SessionEntry, SessionEvent,
    SessionHeader, SessionHeaderPatch, SessionListQuery, SessionOrigin, SessionSelection,
    SessionTraceEvent, TitlePatch, TitleSource, TitleSourcePatch, TraceSeverity,
    derive_session_title,
};
pub use persistence::{PersistenceError, PersistenceResult, SessionPersistence};
pub use recall::{RECALL_TOOL_NAME, RecallConfig, register_recall_tool};
pub use sqlite::{SqlitePersistenceConfig, SqliteSessionPersistence};

#[derive(Clone)]
pub struct SessionEventPayload {
    pub session: Arc<Session>,
    pub entry: SessionEntry,
}

#[derive(Clone)]
pub struct SessionCreatedPayload {
    pub session: Arc<Session>,
}

#[derive(Clone)]
pub struct SessionLoadedPayload {
    pub session: Arc<Session>,
}

#[derive(Clone)]
pub struct SessionTitleUpdatedPayload {
    pub session: Arc<Session>,
    pub title: Option<String>,
    pub title_source: Option<TitleSource>,
}

struct SessionShared {
    persistence: Mutex<Option<Arc<dyn SessionPersistence>>>,
    on_event: SyncEventBus<SessionEventPayload>,
    on_created: SyncEventBus<SessionCreatedPayload>,
    on_loaded: SyncEventBus<SessionLoadedPayload>,
    on_title_updated: SyncEventBus<SessionTitleUpdatedPayload>,
}

impl Default for SessionShared {
    fn default() -> Self {
        Self {
            persistence: Mutex::new(None),
            on_event: SyncEventBus::new(),
            on_created: SyncEventBus::new(),
            on_loaded: SyncEventBus::new(),
            on_title_updated: SyncEventBus::new(),
        }
    }
}

impl SessionShared {
    fn persistence(&self) -> Option<Arc<dyn SessionPersistence>> {
        self.persistence.lock().unwrap().clone()
    }
}

struct SessionInner {
    header: SessionHeader,
    log: Vec<SessionEntry>,
    min_last_seq: u64,
}

pub struct Session {
    pub id: String,
    inner: Mutex<SessionInner>,
    turn_counter: AtomicU64,
    shared: Arc<SessionShared>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn infer_turn_counter(entries: &[SessionEntry]) -> u64 {
    let mut max_turn = 0u64;
    let mut user_count = 0u64;
    for entry in entries {
        match &entry.event {
            SessionEvent::Trace {
                detail:
                    SessionTraceEvent::TurnStart { turn } | SessionTraceEvent::TurnEnd { turn, .. },
            } => {
                max_turn = max_turn.max(*turn);
            }
            SessionEvent::User { .. } => {
                user_count += 1;
            }
            _ => {}
        }
    }
    if max_turn > 0 { max_turn } else { user_count }
}

fn collect_visible_indices(log: &[SessionEntry]) -> Vec<usize> {
    let mut last_compaction: Option<(usize, u64, u64)> = None;
    for (idx, entry) in log.iter().enumerate() {
        if let SessionEvent::Compaction { first_kept_seq, .. } = &entry.event {
            last_compaction = Some((idx, entry.seq, *first_kept_seq));
        }
    }

    let Some((comp_idx, comp_seq, first_kept_seq)) = last_compaction else {
        return log
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.is_surface().then_some(i))
            .collect();
    };

    let mut indices = Vec::with_capacity(log.len().saturating_sub(comp_idx) + 16);
    indices.push(comp_idx);
    for (idx, entry) in log.iter().enumerate() {
        if entry.is_surface() && entry.seq >= first_kept_seq && entry.seq != comp_seq {
            if matches!(entry.event, SessionEvent::Compaction { .. }) {
                continue;
            }
            indices.push(idx);
        }
    }
    indices
}

/// 从会话事件流收集只追加的工具结果裁剪映射（`target_seq -> notice`）。
pub fn collect_pruned_map(log: &[SessionEntry]) -> HashMap<u64, String> {
    let mut map = HashMap::new();
    for entry in log {
        if let SessionEvent::Trace {
            detail: SessionTraceEvent::ToolResultPrune { target_seq, notice },
        } = &entry.event
        {
            map.insert(*target_seq, notice.clone());
        }
    }
    map
}

/// 将压缩检查点统一格式化为发给模型的软硬分离上下文块。
pub fn render_compaction_message(summary: &str, digest: &CompactionDigest) -> String {
    let mut out = String::new();
    out.push_str("<mizar-compaction>\n");
    if !digest.is_empty() {
        out.push_str("<mizar-digest>\n");
        if !digest.modified_files.is_empty() {
            out.push_str("Modified files (authoritative tool record):\n");
            for path in &digest.modified_files {
                out.push_str("- ");
                out.push_str(path);
                out.push('\n');
            }
        }
        if !digest.commands_run.is_empty() {
            out.push_str("Commands run (authoritative tool record):\n");
            for cmd in &digest.commands_run {
                out.push_str("- ");
                out.push_str(cmd);
                out.push('\n');
            }
        }
        if !digest.pinned_inputs.is_empty() {
            out.push_str("Pinned active-turn user input:\n");
            for pin in &digest.pinned_inputs {
                out.push_str(&format!("[input seq #{}]", pin.seq));
                if pin.truncated {
                    out.push_str(" (truncated; use recall for full text)");
                }
                out.push('\n');
                out.push_str(&pin.text);
                out.push('\n');
            }
        }
        out.push_str("</mizar-digest>\n");
    }
    out.push_str("<mizar-summary>\n");
    out.push_str(summary.trim());
    out.push_str("\n</mizar-summary>\n");
    out.push_str("Note: Earlier messages were compacted. Full verbatim history remains available via `recall`. If anything in the summary conflicts with later messages or current workspace files, trust the later messages and current files.\n");
    out.push_str("</mizar-compaction>");
    out
}

fn find_dangling_tool_calls(entries: &[SessionEntry]) -> Vec<(String, String)> {
    let visible = collect_visible_indices(entries);
    let mut pending: Vec<(String, String)> = Vec::new();

    for idx in visible {
        match &entries[idx].event {
            SessionEvent::User { .. } | SessionEvent::Compaction { .. } => {
                pending.clear();
            }
            SessionEvent::Assistant { blocks, .. } => {
                for b in blocks {
                    if let ContentBlock::ToolCall { id, name, .. } = b {
                        pending.push((id.clone(), name.clone()));
                    }
                }
            }
            SessionEvent::ToolResult { tool_call_id, .. } => {
                pending.retain(|(id, _)| id != tool_call_id);
            }
            SessionEvent::ContextNotice { .. }
            | SessionEvent::BashExecution { .. }
            | SessionEvent::Selection { .. }
            | SessionEvent::Usage { .. }
            | SessionEvent::Trace { .. } => {}
        }
    }

    pending
}

/// 检查从持久化或分叉截断加载的日志尾部是否残留未返回 `ToolResult` 的工具调用。
/// 若存在悬挂工具调用，则自动追加合成的 `TOOL_OUTCOME_UNKNOWN` 错误结果以保持对话协议合法。
pub fn repair_dangling_tool_calls(loaded: &mut LoadedSession) -> Vec<SessionEntry> {
    let pending = find_dangling_tool_calls(&loaded.entries);

    if pending.is_empty() {
        return Vec::new();
    }

    let entries_last = loaded.entries.last().map_or(0, |e| e.seq);
    let base_seq = if let Some(max_persisted) = loaded.max_persisted_seq {
        max_persisted.max(entries_last) + 1
    } else if loaded.skipped_events > 0 {
        entries_last.saturating_add(loaded.skipped_events as u64) + 1
    } else {
        entries_last + 1
    };
    let ts = now_millis();
    let mut repaired = Vec::with_capacity(pending.len());
    for (offset, (tool_call_id, name)) in pending.into_iter().enumerate() {
        let entry = SessionEntry {
            seq: base_seq + offset as u64,
            event: SessionEvent::ToolResult {
                tool_call_id,
                name,
                content: "[会话恢复：上次运行意外中断，工具结果未知 (TOOL_OUTCOME_UNKNOWN)]"
                    .to_string(),
                is_error: true,
            },
            ts,
        };
        loaded.header.event_count += 1;
        loaded.header.surface_count += 1;
        loaded.header.updated_at_ms = loaded.header.updated_at_ms.max(ts);
        loaded.entries.push(entry.clone());
        repaired.push(entry);
    }
    if let Some(last_repaired) = repaired.last() {
        loaded.max_persisted_seq = Some(
            loaded
                .max_persisted_seq
                .map_or(last_repaired.seq, |prev| prev.max(last_repaired.seq)),
        );
    }
    repaired
}

impl Session {
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        let header = SessionHeader::new(id.clone(), now_millis());
        Self::from_parts(Arc::new(SessionShared::default()), header, Vec::new())
    }

    pub fn from_loaded(loaded: LoadedSession) -> Self {
        Self::from_loaded_with_shared(Arc::new(SessionShared::default()), loaded)
    }

    fn from_loaded_with_shared(shared: Arc<SessionShared>, loaded: LoadedSession) -> Self {
        let entries_last = loaded.entries.last().map_or(0, |e| e.seq);
        let floor = if let Some(max_p) = loaded.max_persisted_seq {
            max_p.max(entries_last)
        } else if loaded.skipped_events > 0 {
            entries_last.saturating_add(loaded.skipped_events as u64)
        } else {
            entries_last
        };
        Self::from_parts_with_floor(shared, loaded.header, loaded.entries, floor)
    }

    fn from_parts(
        shared: Arc<SessionShared>,
        header: SessionHeader,
        log: Vec<SessionEntry>,
    ) -> Self {
        let floor = log.last().map_or(0, |e| e.seq);
        Self::from_parts_with_floor(shared, header, log, floor)
    }

    fn from_parts_with_floor(
        shared: Arc<SessionShared>,
        mut header: SessionHeader,
        log: Vec<SessionEntry>,
        min_last_seq: u64,
    ) -> Self {
        let id = header.id.clone();
        header.event_count = log.len() as u64;
        header.surface_count = log.iter().filter(|e| e.is_surface()).count() as u64;
        let max_compacted = log
            .iter()
            .filter_map(|e| match &e.event {
                SessionEvent::Compaction { first_kept_seq, .. } => Some(*first_kept_seq),
                _ => None,
            })
            .max();
        header.compacted_up_to_seq = match (header.compacted_up_to_seq, max_compacted) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let min_last_seq = min_last_seq.max(log.last().map_or(0, |e| e.seq));
        let turn_init = infer_turn_counter(&log);
        Self {
            id,
            inner: Mutex::new(SessionInner {
                header,
                log,
                min_last_seq,
            }),
            turn_counter: AtomicU64::new(turn_init),
            shared,
        }
    }

    pub fn header(&self) -> SessionHeader {
        self.inner.lock().unwrap().header.clone()
    }

    /// 分配会话内单调递增的轮次号（从 1 开始）。
    pub fn next_turn(&self) -> u64 {
        self.turn_counter.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// 追加一条事件（`Surface` 或 `LogOnly`），同步写入可选持久化队列，并广播 `on_event`。
    pub fn append(self: &Arc<Self>, event: SessionEvent) {
        let persistence = self.shared.persistence();
        let (entry, emitted_title) = {
            let mut inner = self.inner.lock().unwrap();
            let seq = inner
                .log
                .last()
                .map_or(0, |e| e.seq)
                .max(inner.min_last_seq)
                + 1;
            inner.min_last_seq = seq;
            let ts = now_millis();
            let entry = SessionEntry { seq, event, ts };

            inner.header.updated_at_ms = inner.header.updated_at_ms.max(ts);
            inner.header.event_count += 1;
            if entry.is_surface() {
                inner.header.surface_count += 1;
            }

            let mut title_patch: Option<SessionHeaderPatch> = None;
            let mut emitted_title: Option<(Option<String>, Option<TitleSource>)> = None;
            match &entry.event {
                SessionEvent::User { text, .. } if inner.header.title.is_none() => {
                    if let Some(title) = derive_session_title(text) {
                        inner.header.title = Some(title.clone());
                        inner.header.title_source = Some(TitleSource::Auto);
                        title_patch = Some(SessionHeaderPatch {
                            title: TitlePatch::Set(title.clone()),
                            title_source: TitleSourcePatch::Set(TitleSource::Auto),
                            updated_at_ms: Some(ts),
                            compacted_up_to_seq: None,
                        });
                        emitted_title = Some((Some(title), Some(TitleSource::Auto)));
                    }
                }
                SessionEvent::BashExecution { command, .. } if inner.header.title.is_none() => {
                    let cmd_title = format!("! {command}");
                    if let Some(title) = derive_session_title(&cmd_title) {
                        inner.header.title = Some(title.clone());
                        inner.header.title_source = Some(TitleSource::Auto);
                        title_patch = Some(SessionHeaderPatch {
                            title: TitlePatch::Set(title.clone()),
                            title_source: TitleSourcePatch::Set(TitleSource::Auto),
                            updated_at_ms: Some(ts),
                            compacted_up_to_seq: None,
                        });
                        emitted_title = Some((Some(title), Some(TitleSource::Auto)));
                    }
                }
                SessionEvent::Compaction { first_kept_seq, .. } => {
                    inner.header.compacted_up_to_seq = Some(
                        inner
                            .header
                            .compacted_up_to_seq
                            .map_or(*first_kept_seq, |prev| prev.max(*first_kept_seq)),
                    );
                    title_patch = Some(SessionHeaderPatch {
                        title: TitlePatch::Keep,
                        title_source: TitleSourcePatch::Keep,
                        updated_at_ms: Some(ts),
                        compacted_up_to_seq: inner.header.compacted_up_to_seq,
                    });
                }
                _ => {}
            }

            inner.log.push(entry.clone());

            if let Some(ref backend) = persistence {
                if let Err(err) = backend.append_events(&self.id, std::slice::from_ref(&entry)) {
                    tracing::warn!(
                        target: "mizar::session",
                        "持久化追加事件失败 (session={}): {err}",
                        self.id
                    );
                }
                if let Some(patch) = title_patch
                    && let Err(err) = backend.update_header(&self.id, patch)
                {
                    tracing::warn!(
                        target: "mizar::session",
                        "持久化更新标题失败 (session={}): {err}",
                        self.id
                    );
                }
            }

            (entry, emitted_title)
        };

        self.shared.on_event.emit(&SessionEventPayload {
            session: Arc::clone(self),
            entry,
        });
        if let Some((title, title_source)) = emitted_title {
            self.shared
                .on_title_updated
                .emit(&SessionTitleUpdatedPayload {
                    session: Arc::clone(self),
                    title,
                    title_source,
                });
        }
    }

    /// 幂等修复内存会话尾部因运行中断或超时残留的悬挂 `ToolCall`（通过正常 `append` 落盘与广播）。
    pub fn repair_dangling_tool_calls(self: &Arc<Self>) -> usize {
        let pending = {
            let inner = self.inner.lock().unwrap();
            find_dangling_tool_calls(&inner.log)
        };
        let repaired = pending.len();
        for (tool_call_id, name) in pending {
            self.append(SessionEvent::ToolResult {
                tool_call_id,
                name,
                content: "[会话恢复：上次运行意外中断或超时，工具结果未知 (TOOL_OUTCOME_UNKNOWN)]"
                    .to_string(),
                is_error: true,
            });
        }
        repaired
    }

    /// 设置会话标题（标记为人工设置 `TitleSource::Manual`）。
    /// 空标题返回 `false` 且不做修改。
    pub fn set_title(self: &Arc<Self>, title: impl Into<String>) -> bool {
        self.set_title_with_source(title, TitleSource::Manual)
    }

    /// 设置会话标题及来源（`Auto` / `Manual`）。
    /// 空标题或与当前完全一致时返回 `false`。
    pub fn set_title_with_source(
        self: &Arc<Self>,
        title: impl Into<String>,
        source: TitleSource,
    ) -> bool {
        let cleaned = title.into().replace(['\r', '\n'], " ").trim().to_string();
        if cleaned.is_empty() {
            return false;
        }
        let persistence = self.shared.persistence();
        let ts = now_millis();
        let changed = {
            let mut inner = self.inner.lock().unwrap();
            if inner.header.title.as_deref() == Some(cleaned.as_str())
                && inner.header.title_source == Some(source)
            {
                false
            } else {
                inner.header.title = Some(cleaned.clone());
                inner.header.title_source = Some(source);
                inner.header.updated_at_ms = inner.header.updated_at_ms.max(ts);
                if let Some(ref backend) = persistence
                    && let Err(err) = backend.update_header(
                        &self.id,
                        SessionHeaderPatch {
                            title: TitlePatch::Set(cleaned.clone()),
                            title_source: TitleSourcePatch::Set(source),
                            updated_at_ms: Some(ts),
                            compacted_up_to_seq: None,
                        },
                    )
                {
                    tracing::warn!(
                        target: "mizar::session",
                        "持久化更新标题失败 (session={}): {err}",
                        self.id
                    );
                }
                true
            }
        };
        if !changed {
            return false;
        }
        self.shared
            .on_title_updated
            .emit(&SessionTitleUpdatedPayload {
                session: Arc::clone(self),
                title: Some(cleaned),
                title_source: Some(source),
            });
        true
    }

    /// 原子 CAS 更新自动生成的标题，避免异步生成期间用户手动或并发改名被覆盖。
    ///
    /// - 返回 `Ok(applied)` 表示完成写入并广播更新事件（或已等于目标值）；
    /// - 返回 `Err(current_title)` 表示因所有权或预期旧值不匹配而跳过写入。
    pub fn compare_and_set_title_auto(
        self: &Arc<Self>,
        expected_title: Option<&str>,
        title: impl Into<String>,
        respect_ownership: bool,
    ) -> Result<String, Option<String>> {
        let cleaned = title.into().replace(['\r', '\n'], " ").trim().to_string();
        if cleaned.is_empty() {
            return Err(self.header().title);
        }
        let persistence = self.shared.persistence();
        let ts = now_millis();
        enum CasOutcome {
            Skipped(Option<String>),
            Unchanged,
            Updated,
        }
        let outcome = {
            let mut inner = self.inner.lock().unwrap();
            if (respect_ownership && inner.header.title_source == Some(TitleSource::Manual))
                || inner.header.title.as_deref() != expected_title
            {
                CasOutcome::Skipped(inner.header.title.clone())
            } else if inner.header.title.as_deref() == Some(cleaned.as_str())
                && inner.header.title_source == Some(TitleSource::Auto)
            {
                CasOutcome::Unchanged
            } else {
                inner.header.title = Some(cleaned.clone());
                inner.header.title_source = Some(TitleSource::Auto);
                inner.header.updated_at_ms = inner.header.updated_at_ms.max(ts);
                if let Some(ref backend) = persistence
                    && let Err(err) = backend.update_header(
                        &self.id,
                        SessionHeaderPatch {
                            title: TitlePatch::Set(cleaned.clone()),
                            title_source: TitleSourcePatch::Set(TitleSource::Auto),
                            updated_at_ms: Some(ts),
                            compacted_up_to_seq: None,
                        },
                    )
                {
                    tracing::warn!(
                        target: "mizar::session",
                        "持久化更新标题失败 (session={}): {err}",
                        self.id
                    );
                }
                CasOutcome::Updated
            }
        };
        match outcome {
            CasOutcome::Skipped(cur) => Err(cur),
            CasOutcome::Unchanged => Ok(cleaned),
            CasOutcome::Updated => {
                self.shared
                    .on_title_updated
                    .emit(&SessionTitleUpdatedPayload {
                        session: Arc::clone(self),
                        title: Some(cleaned.clone()),
                        title_source: Some(TitleSource::Auto),
                    });
                Ok(cleaned)
            }
        }
    }

    /// 清空会话标题与来源标记。
    pub fn clear_title(self: &Arc<Self>) -> bool {
        let persistence = self.shared.persistence();
        let ts = now_millis();
        let changed = {
            let mut inner = self.inner.lock().unwrap();
            if inner.header.title.is_none() && inner.header.title_source.is_none() {
                false
            } else {
                inner.header.title = None;
                inner.header.title_source = None;
                inner.header.updated_at_ms = inner.header.updated_at_ms.max(ts);
                if let Some(ref backend) = persistence
                    && let Err(err) = backend.update_header(
                        &self.id,
                        SessionHeaderPatch {
                            title: TitlePatch::Clear,
                            title_source: TitleSourcePatch::Clear,
                            updated_at_ms: Some(ts),
                            compacted_up_to_seq: None,
                        },
                    )
                {
                    tracing::warn!(
                        target: "mizar::session",
                        "持久化清空标题失败 (session={}): {err}",
                        self.id
                    );
                }
                true
            }
        };
        if !changed {
            return false;
        }
        self.shared
            .on_title_updated
            .emit(&SessionTitleUpdatedPayload {
                session: Arc::clone(self),
                title: None,
                title_source: None,
            });
        true
    }

    /// 便捷追加一条仅用于审计/轨迹回放的 `LogOnly` 边界事件。
    pub fn append_trace(self: &Arc<Self>, detail: SessionTraceEvent) {
        self.append(SessionEvent::Trace { detail });
    }

    /// 最新一条 `Selection` 快照（含 `LogOnly`）。
    pub fn latest_selection(&self) -> Option<SessionSelection> {
        let inner = self.inner.lock().unwrap();
        inner.log.iter().rev().find_map(|e| e.event.as_selection())
    }

    /// 记录当前模型与推理强度快照。与最新一条完全相同时不写，避免每轮都追加。
    /// 返回是否真的追加了一条。
    pub fn record_selection(
        self: &Arc<Self>,
        provider: Option<String>,
        model: Option<String>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> bool {
        let next = SessionSelection {
            provider,
            model,
            reasoning_effort,
        };
        if self.latest_selection().as_ref() == Some(&next) {
            return false;
        }
        self.append(SessionEvent::Selection {
            provider: next.provider,
            model: next.model,
            reasoning_effort: next.reasoning_effort,
        });
        true
    }

    /// 折叠较早的表面历史为一条 `Compaction` 摘要，保留最近 `keep_recent_surface` 条表面事件。
    /// 原始历史仍完整保存在日志与持久化数据库中，返回新建 `Compaction` 条目的 `seq`。
    pub fn compact(
        self: &Arc<Self>,
        summary: impl Into<String>,
        keep_recent_surface: usize,
        tokens_before: Option<u64>,
    ) -> u64 {
        let first_kept_seq = {
            let inner = self.inner.lock().unwrap();
            let surface_entries: Vec<&SessionEntry> = inner
                .log
                .iter()
                .filter(|e| e.is_surface() && !matches!(e.event, SessionEvent::Compaction { .. }))
                .collect();
            let next_seq = inner.log.last().map_or(1, |e| e.seq + 1);
            if keep_recent_surface == 0 || surface_entries.is_empty() {
                next_seq
            } else if keep_recent_surface >= surface_entries.len() {
                surface_entries[0].seq
            } else {
                let mut idx = surface_entries.len() - keep_recent_surface;
                // 避免把 Assistant(ToolCall) 和其后的 ToolResult 从中间切开
                while idx > 0
                    && matches!(surface_entries[idx].event, SessionEvent::ToolResult { .. })
                {
                    idx -= 1;
                }
                surface_entries[idx].seq
            }
        };

        self.append(SessionEvent::Compaction {
            summary: summary.into(),
            first_kept_seq,
            tokens_before,
            tokens_after: None,
            digest: CompactionDigest::default(),
        });
        let inner = self.inner.lock().unwrap();
        inner.log.last().map_or(first_kept_seq, |e| e.seq)
    }

    /// 以显式计算好的 `first_kept_seq` 与确定性硬事实 `digest` 追加一条 `Compaction` 检查点。
    pub fn compact_at(
        self: &Arc<Self>,
        summary: impl Into<String>,
        first_kept_seq: u64,
        tokens_before: Option<u64>,
        tokens_after: Option<u64>,
        digest: CompactionDigest,
    ) -> u64 {
        self.append(SessionEvent::Compaction {
            summary: summary.into(),
            first_kept_seq,
            tokens_before,
            tokens_after,
            digest,
        });
        let inner = self.inner.lock().unwrap();
        inner.log.last().map_or(first_kept_seq, |e| e.seq)
    }

    /// 返回最新一条事件序号（空会话返回 `0`）。
    pub fn last_seq(&self) -> u64 {
        self.inner.lock().unwrap().log.last().map_or(0, |e| e.seq)
    }

    /// 若会话发生过压缩，返回最新一条 `Compaction` 条目本身的 `seq` 与 `first_kept_seq`。
    pub fn latest_compaction_boundary(&self) -> Option<(u64, u64)> {
        let inner = self.inner.lock().unwrap();
        inner.log.iter().rev().find_map(|e| match &e.event {
            SessionEvent::Compaction { first_kept_seq, .. } => Some((e.seq, *first_kept_seq)),
            _ => None,
        })
    }

    pub fn flush(&self) -> PersistenceResult<()> {
        if let Some(backend) = self.shared.persistence() {
            backend.flush(Some(&self.id))
        } else {
            Ok(())
        }
    }

    pub async fn flush_async(&self) -> PersistenceResult<()> {
        if let Some(backend) = self.shared.persistence() {
            backend.flush_async(Some(self.id.clone())).await
        } else {
            Ok(())
        }
    }

    /// 返回对模型可见、参与对话展示的表面事件列表（不含 `LogOnly` 轨迹边界事件）。
    pub fn log(&self) -> Vec<SessionEntry> {
        self.inner
            .lock()
            .unwrap()
            .log
            .iter()
            .filter(|e| e.is_surface())
            .cloned()
            .collect()
    }

    /// 返回经过最新 `Compaction` 折叠后仍留在模型窗口内的表面事件。
    pub fn visible_log(&self) -> Vec<SessionEntry> {
        let inner = self.inner.lock().unwrap();
        collect_visible_indices(&inner.log)
            .into_iter()
            .map(|i| inner.log[i].clone())
            .collect()
    }

    /// 零克隆地借用当前可见表面事件切片进行读取检查。
    pub fn with_visible_entries<R>(&self, f: impl FnOnce(&[&SessionEntry]) -> R) -> R {
        let inner = self.inner.lock().unwrap();
        let refs: Vec<&SessionEntry> = collect_visible_indices(&inner.log)
            .into_iter()
            .map(|i| &inner.log[i])
            .collect();
        f(&refs)
    }

    /// 返回完整只追加事件流（含 `Surface` 与 `LogOnly` 轨迹边界事件）。
    pub fn all_entries(&self) -> Vec<SessionEntry> {
        self.inner.lock().unwrap().log.clone()
    }

    /// 零克隆地借用完整事件切片进行统计或轨迹投影。
    pub fn with_all_entries<R>(&self, f: impl FnOnce(&[SessionEntry]) -> R) -> R {
        let inner = self.inner.lock().unwrap();
        f(&inner.log)
    }

    /// 返回当前会话通过 `ToolResultPrune` 标记投影替换的工具结果映射（`target_seq -> notice`）。
    pub fn pruned_map(&self) -> HashMap<u64, String> {
        let inner = self.inner.lock().unwrap();
        collect_pruned_map(&inner.log)
    }

    /// 从日志推导出发给模型的消息列表（自动应用最新 `Compaction` 窗口与 `ToolResultPrune` 投影并跳过 `LogOnly` 事件）。
    pub fn derive_messages(&self) -> Vec<Message> {
        let inner = self.inner.lock().unwrap();
        let indices = collect_visible_indices(&inner.log);
        let pruned_map = collect_pruned_map(&inner.log);
        let mut messages = Vec::with_capacity(indices.len());
        for idx in indices {
            let entry = &inner.log[idx];
            match &entry.event {
                SessionEvent::User { text, images } => messages.push(Message::User {
                    content: text.clone(),
                    images: images.clone(),
                }),
                SessionEvent::ContextNotice { text, images, .. } => messages.push(Message::User {
                    content: text.clone(),
                    images: images.clone(),
                }),
                SessionEvent::Assistant { blocks, replay, .. } => {
                    messages.push(Message::Assistant {
                        blocks: blocks.clone(),
                        replay: replay.clone(),
                    });
                }
                SessionEvent::ToolResult {
                    tool_call_id,
                    content,
                    is_error,
                    ..
                } => {
                    let effective_content = pruned_map
                        .get(&entry.seq)
                        .cloned()
                        .unwrap_or_else(|| content.clone());
                    messages.push(Message::Tool {
                        content: effective_content,
                        tool_call_id: tool_call_id.clone(),
                        is_error: *is_error,
                    });
                }
                SessionEvent::Compaction {
                    summary, digest, ..
                } => {
                    messages.push(Message::user(render_compaction_message(summary, digest)));
                }
                SessionEvent::BashExecution {
                    command,
                    output,
                    exit_code,
                    cancelled,
                    exclude_from_context,
                } => {
                    if !*exclude_from_context {
                        let mut content = format!("Ran `{command}`\n");
                        if !output.is_empty() {
                            content.push_str("```\n");
                            content.push_str(output);
                            if !output.ends_with('\n') {
                                content.push('\n');
                            }
                            content.push_str("```");
                        } else {
                            content.push_str("(no output)");
                        }
                        if *cancelled {
                            content.push_str("\n\n(command cancelled)");
                        } else if let Some(code) = exit_code
                            && *code != 0
                        {
                            content.push_str(&format!("\n\nCommand exited with code {code}"));
                        }
                        messages.push(Message::user(content));
                    }
                }
                SessionEvent::Selection { .. }
                | SessionEvent::Usage { .. }
                | SessionEvent::Trace { .. } => {}
            }
        }
        messages
    }
}

pub struct SessionStore {
    shared: Arc<SessionShared>,
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    default_cwd: Mutex<Option<String>>,
}

impl SessionStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            shared: Arc::new(SessionShared::default()),
            sessions: Mutex::new(HashMap::new()),
            default_cwd: Mutex::new(None),
        })
    }

    pub fn with_persistence(persistence: Arc<dyn SessionPersistence>) -> Arc<Self> {
        let store = Self::new();
        store.set_persistence(Some(persistence));
        store
    }

    pub fn set_persistence(&self, persistence: Option<Arc<dyn SessionPersistence>>) {
        *self.shared.persistence.lock().unwrap() = persistence;
    }

    pub fn persistence(&self) -> Option<Arc<dyn SessionPersistence>> {
        self.shared.persistence()
    }

    pub fn on_event(&self, cb: impl Fn(&SessionEventPayload) + Send + Sync + 'static) -> HookGuard {
        self.shared.on_event.on(cb)
    }

    pub fn on_created(
        &self,
        cb: impl Fn(&SessionCreatedPayload) + Send + Sync + 'static,
    ) -> HookGuard {
        self.shared.on_created.on(cb)
    }

    pub fn on_loaded(
        &self,
        cb: impl Fn(&SessionLoadedPayload) + Send + Sync + 'static,
    ) -> HookGuard {
        self.shared.on_loaded.on(cb)
    }

    pub fn on_title_updated(
        &self,
        cb: impl Fn(&SessionTitleUpdatedPayload) + Send + Sync + 'static,
    ) -> HookGuard {
        self.shared.on_title_updated.on(cb)
    }

    pub fn set_default_cwd(&self, cwd: Option<String>) {
        *self.default_cwd.lock().unwrap() = cwd;
    }

    pub fn default_cwd(&self) -> Option<String> {
        self.default_cwd.lock().unwrap().clone()
    }

    pub fn create(&self, id: Option<String>) -> Arc<Session> {
        self.create_with(SessionCreateOptions {
            id,
            ..Default::default()
        })
    }

    pub fn create_with(&self, opts: SessionCreateOptions) -> Arc<Session> {
        let id = opts.id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let now = now_millis();
        let mut header = SessionHeader::new(id.clone(), now);
        header.cwd = opts.cwd.or_else(|| self.default_cwd());
        header.title = opts.title;
        header.title_source = header.title.as_ref().map(|_| TitleSource::Manual);
        header.parent_id = opts.parent_id;
        header.fork_seq = opts.fork_seq;
        header.origin = opts.origin;

        let session = Arc::new(Session::from_parts(
            Arc::clone(&self.shared),
            header.clone(),
            Vec::new(),
        ));
        self.sessions
            .lock()
            .unwrap()
            .insert(id, Arc::clone(&session));

        if let Some(backend) = self.shared.persistence()
            && let Err(err) = backend.create_session(&header)
        {
            tracing::warn!(
                target: "mizar::session",
                "持久化创建会话失败 (session={}): {err}",
                session.id
            );
        }

        self.shared.on_created.emit(&SessionCreatedPayload {
            session: Arc::clone(&session),
        });
        session
    }

    pub fn contains_in_memory(&self, id: &str) -> bool {
        self.sessions.lock().unwrap().contains_key(id)
    }

    /// 只读加载或快照指定会话（不写入内存缓存，也不执行悬挂工具调用修复/落盘）。
    pub fn inspect_readonly(&self, id: &str) -> Option<LoadedSession> {
        if let Some(existing) = self.sessions.lock().unwrap().get(id).cloned() {
            let inner = existing.inner.lock().unwrap();
            let max_seq = inner
                .log
                .last()
                .map_or(0, |e| e.seq)
                .max(inner.min_last_seq);
            return Some(LoadedSession {
                header: inner.header.clone(),
                entries: inner.log.clone(),
                max_persisted_seq: (max_seq > 0).then_some(max_seq),
                skipped_events: 0,
            });
        }
        let backend = self.shared.persistence()?;
        backend.load_session(id).ok().flatten()
    }

    /// 异步只读加载或快照指定会话（优先走内存缓存，否则非阻塞委托持久化后端）。
    pub async fn inspect_readonly_async(&self, id: &str) -> Option<LoadedSession> {
        if let Some(existing) = self.sessions.lock().unwrap().get(id).cloned() {
            let inner = existing.inner.lock().unwrap();
            let max_seq = inner
                .log
                .last()
                .map_or(0, |e| e.seq)
                .max(inner.min_last_seq);
            return Some(LoadedSession {
                header: inner.header.clone(),
                entries: inner.log.clone(),
                max_persisted_seq: (max_seq > 0).then_some(max_seq),
                skipped_events: 0,
            });
        }
        let backend = self.shared.persistence()?;
        backend
            .load_session_async(id.to_string())
            .await
            .ok()
            .flatten()
    }

    /// 轻量只读获取会话头信息（避免深拷贝或全量反序列化 `entries`）。
    pub fn inspect_header(&self, id: &str) -> Option<SessionHeader> {
        if let Some(existing) = self.sessions.lock().unwrap().get(id).cloned() {
            return Some(existing.header());
        }
        let backend = self.shared.persistence()?;
        backend.load_header(id).ok().flatten()
    }

    /// 异步轻量只读获取会话头信息。
    pub async fn inspect_header_async(&self, id: &str) -> Option<SessionHeader> {
        if let Some(existing) = self.sessions.lock().unwrap().get(id).cloned() {
            return Some(existing.header());
        }
        let backend = self.shared.persistence()?;
        backend
            .load_header_async(id.to_string())
            .await
            .ok()
            .flatten()
    }

    /// 轻量只读窥视某个会话最后记录的选择：不水合、不克隆整段事件日志、不触发悬挂工具调用修复。
    pub fn peek_selection_light(&self, session_id: &str) -> Option<SessionSelection> {
        if let Some(existing) = self.sessions.lock().unwrap().get(session_id).cloned() {
            let inner = existing.inner.lock().unwrap();
            return inner
                .log
                .iter()
                .rev()
                .find_map(|entry| entry.event.as_selection());
        }
        let backend = self.shared.persistence()?;
        backend.latest_selection(session_id).ok().flatten()
    }

    /// 只读窥视某个会话最后记录的选择：不水合进内存、不触发悬挂工具调用修复（后者会写库）。
    pub fn peek_selection(&self, session_id: &str) -> Option<SessionSelection> {
        self.peek_selection_light(session_id)
    }

    /// 异步轻量只读窥视某个会话最后记录的选择。
    pub async fn peek_selection_async(&self, session_id: &str) -> Option<SessionSelection> {
        if let Some(existing) = self.sessions.lock().unwrap().get(session_id).cloned() {
            let inner = existing.inner.lock().unwrap();
            return inner
                .log
                .iter()
                .rev()
                .find_map(|entry| entry.event.as_selection());
        }
        let backend = self.shared.persistence()?;
        backend
            .latest_selection_async(session_id.to_string())
            .await
            .ok()
            .flatten()
    }

    /// 优先从内存返回；若内存中不存在且挂载了持久化后端，则自动从后端水合重建（包含悬挂工具调用自愈）。
    pub fn get(&self, id: &str) -> Option<Arc<Session>> {
        if let Some(existing) = self.sessions.lock().unwrap().get(id).cloned() {
            return Some(existing);
        }

        let backend = self.shared.persistence()?;
        match backend.load_session(id) {
            Ok(Some(mut loaded)) => {
                if loaded.skipped_events > 0 {
                    tracing::warn!(
                        target: "mizar::session",
                        "会话 {id} 有 {} 条事件无法解析已跳过（常见于旧版二进制读新版写入的会话），其余历史已正常载入",
                        loaded.skipped_events
                    );
                }
                let repaired = repair_dangling_tool_calls(&mut loaded);
                if !repaired.is_empty()
                    && let Err(err) = backend.append_events(id, &repaired)
                {
                    tracing::warn!(
                        target: "mizar::session",
                        "持久化写入崩溃恢复条目失败 (session={id}): {err}"
                    );
                }
                let session = Arc::new(Session::from_loaded_with_shared(
                    Arc::clone(&self.shared),
                    loaded,
                ));
                let (stored, inserted) = {
                    let mut guard = self.sessions.lock().unwrap();
                    if let Some(existing) = guard.get(id) {
                        (Arc::clone(existing), false)
                    } else {
                        guard.insert(id.to_string(), Arc::clone(&session));
                        (Arc::clone(&session), true)
                    }
                };
                if inserted {
                    self.shared.on_loaded.emit(&SessionLoadedPayload {
                        session: Arc::clone(&stored),
                    });
                }
                Some(stored)
            }
            Ok(None) => None,
            Err(err) => {
                tracing::warn!(
                    target: "mizar::session",
                    "从持久化后端加载会话失败 (session={id}): {err}"
                );
                None
            }
        }
    }

    pub fn get_or_create(&self, id: Option<String>) -> Arc<Session> {
        if let Some(ref existing_id) = id
            && !existing_id.is_empty()
            && let Some(session) = self.get(existing_id)
        {
            return session;
        }
        self.create(id)
    }

    /// 基于现有会话的 `up_to_seq` 前缀分叉出新会话（优先在持久化层同一事务内完成前缀复制）。
    pub fn fork(
        &self,
        from_session_id: &str,
        up_to_seq: u64,
        opts: SessionCreateOptions,
    ) -> PersistenceResult<Arc<Session>> {
        let parent = self
            .get(from_session_id)
            .ok_or_else(|| PersistenceError::NotFound(from_session_id.to_string()))?;

        let new_id = opts.id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let now = now_millis();
        let parent_header = parent.header();
        let mut new_header = SessionHeader::new(new_id.clone(), now);
        new_header.cwd = opts
            .cwd
            .or(parent_header.cwd)
            .or_else(|| self.default_cwd());
        new_header.title = opts.title;
        new_header.title_source = new_header.title.as_ref().map(|_| TitleSource::Manual);
        new_header.parent_id = Some(from_session_id.to_string());
        new_header.fork_seq = Some(up_to_seq);
        new_header.origin = opts.origin.or(Some(SessionOrigin::User));

        let mut loaded = if let Some(backend) = self.shared.persistence() {
            backend.fork_session(from_session_id, up_to_seq, &new_header)?
        } else {
            let entries: Vec<SessionEntry> = parent
                .all_entries()
                .into_iter()
                .filter(|e| e.seq <= up_to_seq)
                .collect();
            let max_seq = entries.last().map(|e| e.seq);
            LoadedSession {
                header: new_header,
                entries,
                max_persisted_seq: max_seq,
                skipped_events: 0,
            }
        };

        let repaired = repair_dangling_tool_calls(&mut loaded);
        if !repaired.is_empty()
            && let Some(backend) = self.shared.persistence()
        {
            backend.append_events(&new_id, &repaired)?;
        }

        let child = Arc::new(Session::from_loaded_with_shared(
            Arc::clone(&self.shared),
            loaded,
        ));
        self.sessions
            .lock()
            .unwrap()
            .insert(new_id, Arc::clone(&child));
        self.shared.on_created.emit(&SessionCreatedPayload {
            session: Arc::clone(&child),
        });
        Ok(child)
    }

    pub fn list(&self) -> Vec<Arc<Session>> {
        self.sessions.lock().unwrap().values().cloned().collect()
    }

    /// 沿 `parent_id` 向上回溯会话谱系链（首元素为当前 `session_id`，最多 64 代，防环）。
    pub fn lineage_session_ids(&self, session_id: &str) -> Vec<String> {
        let mut chain = Vec::new();
        let mut cur = Some(session_id.to_string());
        let mut depth = 0usize;
        while let Some(id) = cur {
            if depth >= 64 || chain.contains(&id) {
                break;
            }
            chain.push(id.clone());
            depth += 1;
            cur = self.inspect_header(&id).and_then(|h| h.parent_id);
        }
        chain
    }

    /// 异步沿 `parent_id` 向上回溯会话谱系链（轻量仅查头信息，最多 64 代，防环）。
    pub async fn lineage_session_ids_async(&self, session_id: &str) -> Vec<String> {
        let mut chain = Vec::new();
        let mut cur = Some(session_id.to_string());
        let mut depth = 0usize;
        while let Some(id) = cur {
            if depth >= 64 || chain.contains(&id) {
                break;
            }
            chain.push(id.clone());
            depth += 1;
            cur = self
                .inspect_header_async(&id)
                .await
                .and_then(|h| h.parent_id);
        }
        chain
    }

    /// 委托给持久化后端执行全文情景回溯（未启用或不支持索引时返回 `Ok(None)`）。
    pub fn search_recall(
        &self,
        query: &RecallSearchQuery,
    ) -> PersistenceResult<Option<Vec<RecallSearchHit>>> {
        if let Some(backend) = self.shared.persistence() {
            backend.search_recall(query)
        } else {
            Ok(None)
        }
    }

    /// 异步委托给持久化后端执行全文情景回溯。
    pub async fn search_recall_async(
        &self,
        query: RecallSearchQuery,
    ) -> PersistenceResult<Option<Vec<RecallSearchHit>>> {
        if let Some(backend) = self.shared.persistence() {
            backend.search_recall_async(query).await
        } else {
            Ok(None)
        }
    }

    fn list_in_memory_headers(&self, query: &SessionListQuery) -> Vec<SessionHeader> {
        let mut headers: Vec<SessionHeader> = self
            .sessions
            .lock()
            .unwrap()
            .values()
            .map(|s| s.header())
            .filter(|h| {
                (!query.exclude_subagents || !h.is_subagent_child())
                    && query
                        .cwd
                        .as_ref()
                        .is_none_or(|c| h.cwd.as_deref() == Some(c.as_str()))
            })
            .collect();
        headers.sort_by(|a, b| {
            b.updated_at_ms
                .cmp(&a.updated_at_ms)
                .then_with(|| a.id.cmp(&b.id))
        });
        if let Some(limit) = query.limit {
            headers.truncate(limit);
        }
        headers
    }

    /// 列出全部会话头信息（若挂载持久化后端则直接按索引查询，否则汇总内存会话），按最近活跃时间倒序。
    pub fn list_headers(&self, query: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
        if let Some(backend) = self.shared.persistence() {
            return backend.list_sessions(query);
        }
        Ok(self.list_in_memory_headers(query))
    }

    /// 异步列出全部会话头信息。
    pub async fn list_headers_async(
        &self,
        query: SessionListQuery,
    ) -> PersistenceResult<Vec<SessionHeader>> {
        if let Some(backend) = self.shared.persistence() {
            return backend.list_sessions_async(query).await;
        }
        Ok(self.list_in_memory_headers(&query))
    }

    pub fn flush(&self, session_id: Option<&str>) -> PersistenceResult<()> {
        if let Some(backend) = self.shared.persistence() {
            backend.flush(session_id)
        } else {
            Ok(())
        }
    }

    pub async fn flush_async(&self, session_id: Option<String>) -> PersistenceResult<()> {
        if let Some(backend) = self.shared.persistence() {
            backend.flush_async(session_id).await
        } else {
            Ok(())
        }
    }

    pub fn clear(&self) {
        self.sessions.lock().unwrap().clear();
    }
}
