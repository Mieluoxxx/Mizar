//! 基于宽容单会话 JSONL（按 `cwd` 分桶、惰性落盘与 Blob 图片外置）的 `SessionPersistence` 实现与 Cordis 插件。

pub mod blob;
pub mod format;
pub mod writer;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Logger, Plugin};
use mizar_session::{
    LoadedSession, PersistenceError, PersistenceResult, SERVICE_SESSION_PERSISTENCE, SessionEntry,
    SessionHeader, SessionHeaderPatch, SessionListQuery, SessionPersistence,
    SessionPersistenceHandle,
};
use tokio::sync::oneshot;

use crate::blob::BlobStore;
use crate::format::{
    find_session_file, list_sessions_in_dir, read_session_file, validate_session_id,
};
use crate::writer::{BarrierReply, WriteCmd, run_writer_loop};

#[derive(Debug, Clone)]
pub struct JsonlPersistenceConfig {
    pub sessions_dir: PathBuf,
}

impl JsonlPersistenceConfig {
    pub fn new(sessions_dir: impl Into<PathBuf>) -> Self {
        Self {
            sessions_dir: sessions_dir.into(),
        }
    }
}

enum WriterLifecycle {
    Open(JoinHandle<()>),
    Closing,
    Closed,
}

pub struct JsonlSessionPersistence {
    sessions_dir: PathBuf,
    blobs: BlobStore,
    tx: mpsc::Sender<WriteCmd>,
    lifecycle: Mutex<WriterLifecycle>,
    last_error: Arc<Mutex<Option<String>>>,
    logger: Option<Logger>,
}

fn probe_writable_dir(dir: &Path) -> PersistenceResult<()> {
    if dir.as_os_str().is_empty() {
        return Err(PersistenceError::InvalidArgument(
            "sessions_dir cannot be empty".to_string(),
        ));
    }
    std::fs::create_dir_all(dir)?;
    let probe = dir.join(format!(".mizar-write-probe-{}.tmp", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&probe)?;
    use std::io::Write;
    let write_res = file.write_all(b"ok").and_then(|()| file.flush());
    drop(file);
    let remove_res = std::fs::remove_file(&probe);
    write_res?;
    remove_res?;
    Ok(())
}

impl JsonlSessionPersistence {
    pub fn open(sessions_dir: impl AsRef<Path>) -> PersistenceResult<Arc<Self>> {
        Self::open_with_logger(sessions_dir, None)
    }

    pub fn open_with_logger(
        sessions_dir: impl AsRef<Path>,
        logger: Option<Logger>,
    ) -> PersistenceResult<Arc<Self>> {
        let sessions_dir = sessions_dir.as_ref().to_path_buf();
        probe_writable_dir(&sessions_dir)?;

        let blobs = BlobStore::new(&sessions_dir);
        let (tx, rx) = mpsc::channel();
        let last_error = Arc::new(Mutex::new(None));
        let last_error_writer = Arc::clone(&last_error);
        let writer_dir = sessions_dir.clone();
        let writer_blobs = blobs.clone();
        let writer_logger = logger.clone();
        let handle = std::thread::Builder::new()
            .name("mizar-jsonl-writer".to_string())
            .spawn(move || {
                run_writer_loop(
                    writer_dir,
                    writer_blobs,
                    writer_logger,
                    rx,
                    last_error_writer,
                )
            })?;

        Ok(Arc::new(Self {
            sessions_dir,
            blobs,
            tx,
            lifecycle: Mutex::new(WriterLifecycle::Open(handle)),
            last_error,
            logger,
        }))
    }

    pub fn sessions_dir(&self) -> &Path {
        &self.sessions_dir
    }

    pub fn shutdown(&self) -> PersistenceResult<()> {
        let handle = {
            let mut guard = self.lifecycle.lock().unwrap();
            match std::mem::replace(&mut *guard, WriterLifecycle::Closing) {
                WriterLifecycle::Open(h) => h,
                WriterLifecycle::Closing | WriterLifecycle::Closed => {
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

impl Drop for JsonlSessionPersistence {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

impl SessionPersistence for JsonlSessionPersistence {
    fn create_session(&self, header: &SessionHeader) -> PersistenceResult<()> {
        validate_session_id(&header.id)?;
        self.send_cmd(WriteCmd::CreateSession(header.clone()))
    }

    fn update_header(&self, session_id: &str, patch: SessionHeaderPatch) -> PersistenceResult<()> {
        validate_session_id(session_id)?;
        self.send_cmd(WriteCmd::UpdateHeader {
            session_id: session_id.to_string(),
            patch,
        })
    }

    fn load_session(&self, session_id: &str) -> PersistenceResult<Option<LoadedSession>> {
        validate_session_id(session_id)?;
        self.flush(Some(session_id))?;
        let Some(path) = find_session_file(&self.sessions_dir, self.blobs.blobs_dir(), session_id)?
        else {
            return Ok(None);
        };
        read_session_file(&path, Some(&self.blobs), self.logger.as_ref(), true)
    }

    fn list_sessions(&self, query: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
        self.flush(None)?;
        list_sessions_in_dir(&self.sessions_dir, self.blobs.blobs_dir(), query)
    }

    fn append_events(&self, session_id: &str, entries: &[SessionEntry]) -> PersistenceResult<()> {
        validate_session_id(session_id)?;
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
        validate_session_id(from_session_id)?;
        validate_session_id(&new_header.id)?;
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.send_cmd(WriteCmd::ForkSession {
            from_id: from_session_id.to_string(),
            up_to_seq,
            new_header: new_header.clone(),
            reply: reply_tx,
        })?;
        reply_rx.recv().map_err(|_| PersistenceError::Closed)?
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().unwrap().clone()
    }
}

/// 将 `JsonlSessionPersistence` 作为 Cordis 插件挂载到 `SERVICE_SESSION_PERSISTENCE`。
#[derive(Default, Clone, Copy)]
pub struct JsonlPersistencePlugin;

impl Plugin for JsonlPersistencePlugin {
    type Config = JsonlPersistenceConfig;

    fn name(&self) -> Option<&str> {
        Some("session-persistence-jsonl")
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let logger = ctx.logger_named("session-jsonl");
            let backend = match JsonlSessionPersistence::open_with_logger(
                &config.sessions_dir,
                Some(logger.clone()),
            ) {
                Ok(b) => b,
                Err(err) => {
                    logger.error(format!(
                        "初始化 JSONL 会话持久化失败 ({}): {err}",
                        config.sessions_dir.display()
                    ));
                    return Err(err.into());
                }
            };
            let handle = Arc::new(SessionPersistenceHandle::new(backend.clone()));
            ctx.provide(SERVICE_SESSION_PERSISTENCE, handle)?;

            ctx.effect("sessionPersistence.jsonl.shutdown", move || {
                let backend_for_cleanup = Arc::clone(&backend);
                Ok::<_, cordis::CordisError>(Some(Cleanup::sync(move || {
                    let _ = backend_for_cleanup.shutdown();
                })))
            })?;

            Ok(None)
        })
    }
}
