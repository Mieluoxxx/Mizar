//! 会话持久化抽象接口（Seam）：把领域层 `mizar-session` 与具体存储引擎（如 SQLite / JSONL）解耦。

use mizar_types::hooks::BoxFuture;
use thiserror::Error;

use crate::model::{
    LoadedSession, SessionEntry, SessionHeader, SessionHeaderPatch, SessionListQuery,
    SessionSelection,
};

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

    /// 轻量读取会话头信息，避免反序列化与深拷贝完整 `entries`。
    fn load_header(&self, session_id: &str) -> PersistenceResult<Option<SessionHeader>> {
        Ok(self.load_session(session_id)?.map(|s| s.header))
    }

    /// 轻量读取会话最后记录的 `Selection` 快照，避免加载完整事件日志。
    fn latest_selection(&self, session_id: &str) -> PersistenceResult<Option<SessionSelection>> {
        let Some(loaded) = self.load_session(session_id)? else {
            return Ok(None);
        };
        Ok(loaded
            .entries
            .iter()
            .rev()
            .find_map(|entry| entry.event.as_selection()))
    }

    fn list_sessions(&self, query: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>>;

    fn load_session_async(
        &self,
        session_id: String,
    ) -> BoxFuture<'static, PersistenceResult<Option<LoadedSession>>> {
        let res = self.load_session(&session_id);
        Box::pin(async move { res })
    }

    fn load_header_async(
        &self,
        session_id: String,
    ) -> BoxFuture<'static, PersistenceResult<Option<SessionHeader>>> {
        let res = self.load_header(&session_id);
        Box::pin(async move { res })
    }

    fn latest_selection_async(
        &self,
        session_id: String,
    ) -> BoxFuture<'static, PersistenceResult<Option<SessionSelection>>> {
        let res = self.latest_selection(&session_id);
        Box::pin(async move { res })
    }

    fn list_sessions_async(
        &self,
        query: SessionListQuery,
    ) -> BoxFuture<'static, PersistenceResult<Vec<SessionHeader>>> {
        let res = self.list_sessions(&query);
        Box::pin(async move { res })
    }

    fn append_events(&self, session_id: &str, entries: &[SessionEntry]) -> PersistenceResult<()>;

    fn flush(&self, session_id: Option<&str>) -> PersistenceResult<()>;

    fn flush_async(&self, session_id: Option<String>) -> BoxFuture<'static, PersistenceResult<()>>;

    fn fork_session(
        &self,
        from_session_id: &str,
        up_to_seq: u64,
        new_header: &SessionHeader,
    ) -> PersistenceResult<LoadedSession>;
}
