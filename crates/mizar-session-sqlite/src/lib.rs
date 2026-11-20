//! 基于嵌入式 SQLite（WAL + Single-Writer 后台线程）的 `SessionPersistence` 实现与 Cordis 插件。

pub mod recall;
pub mod schema;
pub mod writer;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Plugin};
use mizar_session::{
    LoadedSession, PersistenceError, PersistenceResult, RecallSearchHit, RecallSearchQuery,
    SERVICE_SESSION_PERSISTENCE, SessionEntry, SessionHeader, SessionHeaderPatch, SessionListQuery,
    SessionPersistence, SessionPersistenceHandle,
};
use rusqlite::{Connection, OpenFlags};
use tokio::sync::oneshot;

use crate::recall::search_recall_from_conn;
use crate::schema::{
    configure_reader_connection, init_schema, list_sessions_from_conn, load_session_from_conn,
    map_sqlite_err,
};
use crate::writer::{BarrierReply, WriteCmd, run_writer_loop};

#[derive(Debug, Clone)]
pub struct SqlitePersistenceConfig {
    /// 数据库文件路径（传 `PathBuf::from(":memory:")` 可用于纯内存测试）。
    pub db_path: PathBuf,
}

impl SqlitePersistenceConfig {
    pub fn new(db_path: impl Into<PathBuf>) -> Self {
        Self {
            db_path: db_path.into(),
        }
    }
}

enum WriterLifecycle {
    Open(JoinHandle<()>),
    Closing,
    Closed,
}

pub struct SqliteSessionPersistence {
    tx: mpsc::Sender<WriteCmd>,
    reader: Option<Mutex<Connection>>,
    lifecycle: Mutex<WriterLifecycle>,
    last_error: Arc<Mutex<Option<String>>>,
}

impl SqliteSessionPersistence {
    pub fn open(db_path: impl AsRef<Path>) -> PersistenceResult<Arc<Self>> {
        let path = db_path.as_ref();
        let is_memory = path == Path::new(":memory:");

        if !is_memory
            && let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let writer_conn = if is_memory {
            Connection::open_in_memory().map_err(map_sqlite_err)?
        } else {
            Connection::open(path).map_err(map_sqlite_err)?
        };
        init_schema(&writer_conn)?;

        let reader = if is_memory {
            None
        } else {
            let read_conn = Connection::open_with_flags(
                path,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
            .map_err(map_sqlite_err)?;
            configure_reader_connection(&read_conn)?;
            Some(Mutex::new(read_conn))
        };

        let (tx, rx) = mpsc::channel();
        let last_error = Arc::new(Mutex::new(None));
        let last_error_writer = Arc::clone(&last_error);
        let handle = std::thread::Builder::new()
            .name("mizar-sqlite-writer".to_string())
            .spawn(move || run_writer_loop(writer_conn, rx, last_error_writer))?;

        Ok(Arc::new(Self {
            tx,
            reader,
            lifecycle: Mutex::new(WriterLifecycle::Open(handle)),
            last_error,
        }))
    }

    pub fn shutdown(&self) -> PersistenceResult<()> {
        let handle = {
            let mut guard = self.lifecycle.lock().unwrap();
            match std::mem::replace(&mut *guard, WriterLifecycle::Closing) {
                WriterLifecycle::Open(h) => h,
                WriterLifecycle::Closing | WriterLifecycle::Closed => {
                    // 若未接手过 Handle（并发或二次关闭），保留既定 Closing/Closed 状态并幂等返回
                    if !matches!(*guard, WriterLifecycle::Closing) {
                        *guard = WriterLifecycle::Closed;
                    }
                    return Ok(());
                }
            }
        };
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let send_res = self.tx.send(WriteCmd::Shutdown(reply_tx));
        let res = if send_res.is_ok() {
            reply_rx.recv().unwrap_or(Err(PersistenceError::Closed))
        } else {
            Ok(())
        };
        let _ = handle.join();
        *self.lifecycle.lock().unwrap() = WriterLifecycle::Closed;
        res
    }

    fn send_cmd(&self, cmd: WriteCmd) -> PersistenceResult<()> {
        {
            let guard = self.lifecycle.lock().unwrap();
            if !matches!(*guard, WriterLifecycle::Open(_)) {
                return Err(PersistenceError::Closed);
            }
        }
        if let Some(ref err_msg) = *self.last_error.lock().unwrap() {
            return Err(PersistenceError::Backend(err_msg.clone()));
        }
        self.tx.send(cmd).map_err(|_| PersistenceError::Closed)
    }
}

impl Drop for SqliteSessionPersistence {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

impl SessionPersistence for SqliteSessionPersistence {
    fn create_session(&self, header: &SessionHeader) -> PersistenceResult<()> {
        self.send_cmd(WriteCmd::CreateSession(header.clone()))
    }

    fn update_header(&self, session_id: &str, patch: SessionHeaderPatch) -> PersistenceResult<()> {
        self.send_cmd(WriteCmd::UpdateHeader {
            session_id: session_id.to_string(),
            patch,
        })
    }

    fn load_session(&self, session_id: &str) -> PersistenceResult<Option<LoadedSession>> {
        if let Some(ref reader) = self.reader {
            self.flush(Some(session_id))?;
            let conn = reader.lock().unwrap();
            load_session_from_conn(&conn, session_id)
        } else {
            let (reply_tx, reply_rx) = mpsc::sync_channel(1);
            self.send_cmd(WriteCmd::LoadSession {
                session_id: session_id.to_string(),
                reply: reply_tx,
            })?;
            reply_rx.recv().map_err(|_| PersistenceError::Closed)?
        }
    }

    fn list_sessions(&self, query: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
        if let Some(ref reader) = self.reader {
            self.flush(None)?;
            let conn = reader.lock().unwrap();
            list_sessions_from_conn(&conn, query)
        } else {
            let (reply_tx, reply_rx) = mpsc::sync_channel(1);
            self.send_cmd(WriteCmd::ListSessions {
                query: query.clone(),
                reply: reply_tx,
            })?;
            reply_rx.recv().map_err(|_| PersistenceError::Closed)?
        }
    }

    fn append_events(&self, session_id: &str, entries: &[SessionEntry]) -> PersistenceResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.send_cmd(WriteCmd::AppendEvents {
            session_id: session_id.to_string(),
            entries: entries.to_vec(),
        })
    }

    fn flush(&self, _session_id: Option<&str>) -> PersistenceResult<()> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.send_cmd(WriteCmd::Flush(BarrierReply::Sync(reply_tx)))?;
        reply_rx.recv().map_err(|_| PersistenceError::Closed)?
    }

    fn flush_async(
        &self,
        _session_id: Option<String>,
    ) -> BoxFuture<'static, PersistenceResult<()>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let send_res = self.send_cmd(WriteCmd::Flush(BarrierReply::Async(reply_tx)));
        Box::pin(async move {
            send_res?;
            reply_rx.await.map_err(|_| PersistenceError::Closed)?
        })
    }

    fn fork_session(
        &self,
        from_session_id: &str,
        up_to_seq: u64,
        new_header: &SessionHeader,
    ) -> PersistenceResult<LoadedSession> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.send_cmd(WriteCmd::ForkSession {
            from_id: from_session_id.to_string(),
            up_to_seq,
            new_header: new_header.clone(),
            reply: reply_tx,
        })?;
        reply_rx.recv().map_err(|_| PersistenceError::Closed)?
    }

    fn search_recall(
        &self,
        query: &RecallSearchQuery,
    ) -> PersistenceResult<Option<Vec<RecallSearchHit>>> {
        if let Some(ref reader) = self.reader {
            self.flush(None)?;
            let conn = reader.lock().unwrap();
            search_recall_from_conn(&conn, query).map(Some)
        } else {
            let (reply_tx, reply_rx) = mpsc::sync_channel(1);
            self.send_cmd(WriteCmd::SearchRecall {
                query: query.clone(),
                reply: reply_tx,
            })?;
            reply_rx
                .recv()
                .map_err(|_| PersistenceError::Closed)?
                .map(Some)
        }
    }
}

/// 将 `SqliteSessionPersistence` 作为 Cordis 插件挂载到 `SERVICE_SESSION_PERSISTENCE`。
#[derive(Default, Clone, Copy)]
pub struct SqlitePersistencePlugin;

impl Plugin for SqlitePersistencePlugin {
    type Config = SqlitePersistenceConfig;

    fn name(&self) -> Option<&str> {
        Some("session-persistence-sqlite")
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let backend = SqliteSessionPersistence::open(&config.db_path)?;
            let handle = Arc::new(SessionPersistenceHandle::new(backend.clone()));
            ctx.provide(SERVICE_SESSION_PERSISTENCE, handle)?;

            ctx.effect("sessionPersistence.sqlite.shutdown", move || {
                let backend_for_cleanup = Arc::clone(&backend);
                Ok::<_, cordis::CordisError>(Some(Cleanup::sync(move || {
                    let _ = backend_for_cleanup.shutdown();
                })))
            })?;

            Ok(None)
        })
    }
}
