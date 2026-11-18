//! 会话日志（对应 `NamiDSH/packages/session`）。
//!
//! 会话 = 一条只追加的事件日志。发给模型的消息列表是从日志「推导」出来的。
//! 支持可选的持久化后端（`SessionPersistence`）、上下文压缩（`Compaction`）
//! 以及同源承载 `LogOnly` 审计/轨迹事件。

pub mod model;
pub mod persistence;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{BoxFuture, Cleanup, Context, CordisError, Service};
use dsh_llm::{ContentBlock, Message};
use uuid::Uuid;

pub use model::{
    EntryVisibility, LoadedSession, SESSION_FORMAT_VERSION, SessionCreateOptions, SessionEntry,
    SessionEvent, SessionHeader, SessionHeaderPatch, SessionListQuery, SessionTraceEvent,
    TraceSeverity, derive_session_title,
};
pub use persistence::{
    PersistenceError, PersistenceResult, SERVICE_SESSION_PERSISTENCE, SessionPersistence,
    SessionPersistenceContextExt, SessionPersistenceHandle,
};

pub const EVENT_SESSION_EVENT: &str = "session/event";
pub const EVENT_SESSION_CREATED: &str = "session/created";

#[derive(Clone)]
pub struct SessionEventPayload {
    pub session: Arc<Session>,
    pub entry: SessionEntry,
}

#[derive(Clone)]
pub struct SessionCreatedPayload {
    pub session: Arc<Session>,
}

struct SessionInner {
    header: SessionHeader,
    log: Vec<SessionEntry>,
}

pub struct Session {
    pub id: String,
    inner: Mutex<SessionInner>,
    turn_counter: AtomicU64,
    ctx: Context,
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

/// 检查从持久化或分叉截断加载的日志尾部是否残留未返回 `ToolResult` 的工具调用。
/// 若存在悬挂工具调用，则自动追加合成的 `TOOL_OUTCOME_UNKNOWN` 错误结果以保持对话协议合法。
pub fn repair_dangling_tool_calls(loaded: &mut LoadedSession) -> Vec<SessionEntry> {
    let visible = collect_visible_indices(&loaded.entries);
    let mut pending: Vec<(String, String)> = Vec::new();

    for idx in visible {
        match &loaded.entries[idx].event {
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
            SessionEvent::ContextNotice { .. } | SessionEvent::Trace { .. } => {}
        }
    }

    if pending.is_empty() {
        return Vec::new();
    }

    let base_seq = loaded.entries.last().map_or(1, |e| e.seq + 1);
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
    repaired
}

impl Session {
    pub fn new(ctx: Context, id: impl Into<String>) -> Self {
        let id = id.into();
        let header = SessionHeader::new(id.clone(), now_millis());
        Self::from_parts(ctx, header, Vec::new())
    }

    pub fn from_loaded(ctx: Context, loaded: LoadedSession) -> Self {
        Self::from_parts(ctx, loaded.header, loaded.entries)
    }

    fn from_parts(ctx: Context, mut header: SessionHeader, log: Vec<SessionEntry>) -> Self {
        let id = header.id.clone();
        header.event_count = log.len() as u64;
        header.surface_count = log.iter().filter(|e| e.is_surface()).count() as u64;
        let turn_init = infer_turn_counter(&log);
        Self {
            id,
            inner: Mutex::new(SessionInner { header, log }),
            turn_counter: AtomicU64::new(turn_init),
            ctx,
        }
    }

    pub fn header(&self) -> SessionHeader {
        self.inner.lock().unwrap().header.clone()
    }

    /// 分配会话内单调递增的轮次号（从 1 开始）。
    pub fn next_turn(&self) -> u64 {
        self.turn_counter.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// 追加一条事件（`Surface` 或 `LogOnly`），同步写入可选持久化队列，并广播 `session/event`。
    pub fn append(self: &Arc<Self>, event: SessionEvent) {
        let persistence = self.ctx.session_persistence();
        let entry = {
            let mut inner = self.inner.lock().unwrap();
            let seq = inner.log.last().map_or(1, |e| e.seq + 1);
            let ts = now_millis();
            let entry = SessionEntry { seq, event, ts };

            inner.header.updated_at_ms = inner.header.updated_at_ms.max(ts);
            inner.header.event_count += 1;
            if entry.is_surface() {
                inner.header.surface_count += 1;
            }

            let mut title_patch: Option<SessionHeaderPatch> = None;
            match &entry.event {
                SessionEvent::User { text } if inner.header.title.is_none() => {
                    if let Some(title) = derive_session_title(text) {
                        inner.header.title = Some(title.clone());
                        title_patch = Some(SessionHeaderPatch {
                            title: Some(title),
                            updated_at_ms: Some(ts),
                            compacted_up_to_seq: None,
                        });
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
                        title: None,
                        updated_at_ms: Some(ts),
                        compacted_up_to_seq: inner.header.compacted_up_to_seq,
                    });
                }
                _ => {}
            }

            inner.log.push(entry.clone());

            if let Some(ref backend) = persistence {
                if let Err(err) = backend.append_events(&self.id, std::slice::from_ref(&entry)) {
                    self.ctx
                        .logger_named("session")
                        .warn(format!("持久化追加事件失败 (session={}): {err}", self.id));
                }
                if let Some(patch) = title_patch
                    && let Err(err) = backend.update_header(&self.id, patch)
                {
                    self.ctx
                        .logger_named("session")
                        .warn(format!("持久化更新标题失败 (session={}): {err}", self.id));
                }
            }

            entry
        };

        self.ctx.emit(
            EVENT_SESSION_EVENT,
            SessionEventPayload {
                session: Arc::clone(self),
                entry,
            },
        );
    }

    /// 便捷追加一条仅用于审计/轨迹回放的 `LogOnly` 边界事件。
    pub fn append_trace(self: &Arc<Self>, detail: SessionTraceEvent) {
        self.append(SessionEvent::Trace { detail });
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
                .filter(|e| {
                    e.is_surface() && !matches!(e.event, SessionEvent::Compaction { .. })
                })
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
        });
        let inner = self.inner.lock().unwrap();
        inner.log.last().map_or(first_kept_seq, |e| e.seq)
    }

    pub fn flush(&self) -> PersistenceResult<()> {
        if let Some(backend) = self.ctx.session_persistence() {
            backend.flush(Some(&self.id))
        } else {
            Ok(())
        }
    }

    pub async fn flush_async(&self) -> PersistenceResult<()> {
        if let Some(backend) = self.ctx.session_persistence() {
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

    /// 从日志推导出发给模型的消息列表（自动应用最新 `Compaction` 窗口并跳过 `LogOnly` 事件）。
    pub fn derive_messages(&self) -> Vec<Message> {
        let inner = self.inner.lock().unwrap();
        let indices = collect_visible_indices(&inner.log);
        let mut messages = Vec::with_capacity(indices.len());
        for idx in indices {
            match &inner.log[idx].event {
                SessionEvent::User { text } => messages.push(Message::User {
                    content: text.clone(),
                }),
                SessionEvent::ContextNotice { text, .. } => messages.push(Message::User {
                    content: text.clone(),
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
                    messages.push(Message::Tool {
                        content: content.clone(),
                        tool_call_id: tool_call_id.clone(),
                        is_error: *is_error,
                    });
                }
                SessionEvent::Compaction { summary, .. } => {
                    messages.push(Message::User {
                        content: format!("[历史对话摘要]\n{summary}"),
                    });
                }
                SessionEvent::Trace { .. } => {}
            }
        }
        messages
    }
}

pub struct SessionStore {
    ctx: Context,
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    default_cwd: Mutex<Option<String>>,
}

impl SessionStore {
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
        header.parent_id = opts.parent_id;
        header.fork_seq = opts.fork_seq;

        let session = Arc::new(Session::from_parts(
            self.ctx.clone(),
            header.clone(),
            Vec::new(),
        ));
        self.sessions
            .lock()
            .unwrap()
            .insert(id, Arc::clone(&session));

        if let Some(backend) = self.ctx.session_persistence()
            && let Err(err) = backend.create_session(&header)
        {
            self.ctx
                .logger_named("session")
                .warn(format!("持久化创建会话失败 (session={}): {err}", session.id));
        }

        self.ctx.emit(
            EVENT_SESSION_CREATED,
            SessionCreatedPayload {
                session: Arc::clone(&session),
            },
        );
        session
    }

    pub fn contains_in_memory(&self, id: &str) -> bool {
        self.sessions.lock().unwrap().contains_key(id)
    }

    /// 优先从内存返回；若内存中不存在且挂载了持久化后端，则自动从后端水合重建（包含悬挂工具调用自愈）。
    pub fn get(&self, id: &str) -> Option<Arc<Session>> {
        if let Some(existing) = self.sessions.lock().unwrap().get(id).cloned() {
            return Some(existing);
        }

        let backend = self.ctx.session_persistence()?;
        match backend.load_session(id) {
            Ok(Some(mut loaded)) => {
                let repaired = repair_dangling_tool_calls(&mut loaded);
                if !repaired.is_empty()
                    && let Err(err) = backend.append_events(id, &repaired)
                {
                    self.ctx
                        .logger_named("session")
                        .warn(format!("持久化写入崩溃恢复条目失败 (session={id}): {err}"));
                }
                let session = Arc::new(Session::from_loaded(self.ctx.clone(), loaded));
                let mut guard = self.sessions.lock().unwrap();
                Some(
                    guard
                        .entry(id.to_string())
                        .or_insert_with(|| Arc::clone(&session))
                        .clone(),
                )
            }
            Ok(None) => None,
            Err(err) => {
                self.ctx
                    .logger_named("session")
                    .warn(format!("从持久化后端加载会话失败 (session={id}): {err}"));
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
        new_header.cwd = opts.cwd.or(parent_header.cwd).or_else(|| self.default_cwd());
        new_header.title = opts.title.or(parent_header.title);
        new_header.parent_id = Some(from_session_id.to_string());
        new_header.fork_seq = Some(up_to_seq);

        let mut loaded = if let Some(backend) = self.ctx.session_persistence() {
            backend.fork_session(from_session_id, up_to_seq, &new_header)?
        } else {
            let entries: Vec<SessionEntry> = parent
                .all_entries()
                .into_iter()
                .filter(|e| e.seq <= up_to_seq)
                .collect();
            LoadedSession {
                header: new_header,
                entries,
            }
        };

        let repaired = repair_dangling_tool_calls(&mut loaded);
        if !repaired.is_empty()
            && let Some(backend) = self.ctx.session_persistence()
        {
            backend.append_events(&new_id, &repaired)?;
        }

        let child = Arc::new(Session::from_loaded(self.ctx.clone(), loaded));
        self.sessions
            .lock()
            .unwrap()
            .insert(new_id, Arc::clone(&child));
        self.ctx.emit(
            EVENT_SESSION_CREATED,
            SessionCreatedPayload {
                session: Arc::clone(&child),
            },
        );
        Ok(child)
    }

    pub fn list(&self) -> Vec<Arc<Session>> {
        self.sessions.lock().unwrap().values().cloned().collect()
    }

    /// 列出全部会话头信息（若挂载持久化后端则直接按索引查询，否则汇总内存会话），按最近活跃时间倒序。
    pub fn list_headers(&self, query: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
        if let Some(backend) = self.ctx.session_persistence() {
            return backend.list_sessions(query);
        }

        let mut headers: Vec<SessionHeader> = self
            .sessions
            .lock()
            .unwrap()
            .values()
            .map(|s| s.header())
            .filter(|h| query.cwd.as_ref().is_none_or(|c| h.cwd.as_deref() == Some(c.as_str())))
            .collect();
        headers.sort_by(|a, b| {
            b.updated_at_ms
                .cmp(&a.updated_at_ms)
                .then_with(|| a.id.cmp(&b.id))
        });
        if let Some(limit) = query.limit {
            headers.truncate(limit);
        }
        Ok(headers)
    }

    pub fn flush(&self, session_id: Option<&str>) -> PersistenceResult<()> {
        if let Some(backend) = self.ctx.session_persistence() {
            backend.flush(session_id)
        } else {
            Ok(())
        }
    }

    pub async fn flush_async(&self, session_id: Option<String>) -> PersistenceResult<()> {
        if let Some(backend) = self.ctx.session_persistence() {
            backend.flush_async(session_id).await
        } else {
            Ok(())
        }
    }

    pub fn clear(&self) {
        self.sessions.lock().unwrap().clear();
    }
}

impl Service for SessionStore {
    type Config = ();

    fn service_name() -> &'static str {
        "sessions"
    }

    fn plugin_name() -> &'static str {
        "SessionStore"
    }

    fn create(ctx: &Context, _config: Self::Config) -> Result<Arc<Self>, CordisError> {
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            sessions: Mutex::new(HashMap::new()),
            default_cwd: Mutex::new(None),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let store = Arc::downgrade(&self);
        let ctx = ctx.clone();
        Box::pin(async move {
            // 服务卸载前刷新持久化屏障，再清掉内存会话缓存
            ctx.effect("sessions.clear", || {
                Ok::<_, CordisError>(Some(Cleanup::sync(move || {
                    if let Some(s) = store.upgrade() {
                        let _ = s.flush(None);
                        s.clear();
                    }
                })))
            })?;
            Ok(())
        })
    }
}

pub trait SessionContextExt {
    fn sessions(&self) -> Result<Arc<SessionStore>, CordisError>;
}

impl SessionContextExt for Context {
    fn sessions(&self) -> Result<Arc<SessionStore>, CordisError> {
        self.service::<SessionStore>("sessions")
    }
}
