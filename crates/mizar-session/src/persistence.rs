//! 会话持久化抽象接口（Seam）：把领域层 `mizar-session` 与具体存储引擎（如 JSONL）解耦。

use std::ops::Deref;
use std::sync::Arc;

use cordis::{BoxFuture, Context};
use thiserror::Error;

use crate::model::{
    LoadedSession, SessionEntry, SessionHeader, SessionHeaderPatch, SessionListQuery,
};

pub const SERVICE_SESSION_PERSISTENCE: &str = "sessionPersistence";

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("session \"{0}\" not found")]
    NotFound(String),
    #[error("invalid persistence argument: {0}")]
    InvalidArgument(String),
    #[error("persistence backend is closed")]
    Closed,
    #[error("persistence backend error: {0}")]
    Backend(String),
}

pub type PersistenceResult<T> = Result<T, PersistenceError>;

/// 会话存储后端需要实现的最小持久化契约。
///
/// 约定：
/// - `create_session` / `append_events` / `update_header` 应尽量做到非阻塞入队，保证单会话内严格按调用顺序落盘；
/// - `flush` / `flush_async` 作为持久化同步屏障，确保在此之前的写入已原子提交；
/// - 写操作不得反向调用 `Session` 的内部锁，避免重入死锁。
pub trait SessionPersistence: Send + Sync + 'static {
    fn create_session(&self, header: &SessionHeader) -> PersistenceResult<()>;

    fn update_header(&self, session_id: &str, patch: SessionHeaderPatch) -> PersistenceResult<()>;

    fn load_session(&self, session_id: &str) -> PersistenceResult<Option<LoadedSession>>;

    fn list_sessions(&self, query: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>>;

    fn append_events(&self, session_id: &str, entries: &[SessionEntry]) -> PersistenceResult<()>;

    fn flush(&self, session_id: Option<&str>) -> PersistenceResult<()>;

    fn flush_async(&self, session_id: Option<String>) -> BoxFuture<'static, PersistenceResult<()>>;

    fn fork_session(
        &self,
        from_session_id: &str,
        up_to_seq: u64,
        new_header: &SessionHeader,
    ) -> PersistenceResult<LoadedSession>;

    /// 返回异步写入线程最近一次未恢复的持久化错误，供宿主及时感知并提示用户。
    fn last_error(&self) -> Option<String> {
        None
    }
}

/// 包装 `Arc<dyn SessionPersistence>` 以便通过 Cordis `ctx.provide` 注册与查找。
#[derive(Clone)]
pub struct SessionPersistenceHandle {
    backend: Arc<dyn SessionPersistence>,
}

impl SessionPersistenceHandle {
    pub fn new(backend: Arc<dyn SessionPersistence>) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &Arc<dyn SessionPersistence> {
        &self.backend
    }
}

impl Deref for SessionPersistenceHandle {
    type Target = dyn SessionPersistence;

    fn deref(&self) -> &Self::Target {
        &*self.backend
    }
}

pub trait SessionPersistenceContextExt {
    fn session_persistence(&self) -> Option<Arc<SessionPersistenceHandle>>;
}

impl SessionPersistenceContextExt for Context {
    fn session_persistence(&self) -> Option<Arc<SessionPersistenceHandle>> {
        self.get::<SessionPersistenceHandle>(SERVICE_SESSION_PERSISTENCE)
    }
}
