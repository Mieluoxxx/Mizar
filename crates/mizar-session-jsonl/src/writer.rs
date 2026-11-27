//! 单后台写入线程（Single-Writer）：保障单会话事件顺序落盘、惰性创建与 `flush` 同步屏障。

use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

use mizar_session::{
    LoadedSession, PersistenceError, PersistenceResult, SessionEntry, SessionEvent, SessionHeader,
    SessionHeaderPatch,
};
use tokio::sync::oneshot;

use crate::blob::BlobStore;
use crate::format::{
    apply_header_patch, find_session_file, read_session_file, read_session_seqs,
    serialize_entry_line, serialize_header_patch_line, serialize_session_header_line,
    session_file_path,
};

const MAX_BATCH_COMMANDS: usize = 128;

pub enum BarrierReply {
    Sync(mpsc::SyncSender<PersistenceResult<()>>),
    Async(oneshot::Sender<PersistenceResult<()>>),
}

impl BarrierReply {
    fn send(self, res: PersistenceResult<()>) {
        match self {
            Self::Sync(tx) => {
                let _ = tx.send(res);
            }
            Self::Async(tx) => {
                let _ = tx.send(res);
            }
        }
    }
}

pub enum WriteCmd {
    CreateSession(SessionHeader),
    AppendEvents {
        session_id: String,
        entries: Vec<SessionEntry>,
    },
    UpdateHeader {
        session_id: String,
        patch: SessionHeaderPatch,
    },
    ForkSession {
        from_id: String,
        up_to_seq: u64,
        new_header: SessionHeader,
        reply: mpsc::SyncSender<PersistenceResult<LoadedSession>>,
    },
    Flush(BarrierReply),
    Shutdown(mpsc::SyncSender<PersistenceResult<()>>),
}

struct MaterializedSession {
    path: PathBuf,
    seen_seqs: HashSet<u64>,
}

struct WriterState {
    sessions_dir: PathBuf,
    blobs: BlobStore,
    logger: Option<cordis::Logger>,
    pending: HashMap<String, SessionHeader>,
    materialized: HashMap<String, MaterializedSession>,
}

impl WriterState {
    fn resolve_existing_path(&mut self, session_id: &str) -> PersistenceResult<Option<PathBuf>> {
        if let Some(mat) = self.materialized.get(session_id) {
            return Ok(Some(mat.path.clone()));
        }
        let Some(path) = find_session_file(&self.sessions_dir, self.blobs.blobs_dir(), session_id)?
        else {
            return Ok(None);
        };
        let seen_seqs = read_session_seqs(&path)?;
        self.materialized.insert(
            session_id.to_string(),
            MaterializedSession {
                path: path.clone(),
                seen_seqs,
            },
        );
        Ok(Some(path))
    }

    fn apply_create_session(&mut self, header: SessionHeader) -> PersistenceResult<()> {
        if self.materialized.contains_key(&header.id) {
            return Ok(());
        }
        if let Some(existing_path) =
            find_session_file(&self.sessions_dir, self.blobs.blobs_dir(), &header.id)?
        {
            let seen_seqs = read_session_seqs(&existing_path)?;
            self.materialized.insert(
                header.id,
                MaterializedSession {
                    path: existing_path,
                    seen_seqs,
                },
            );
            return Ok(());
        }
        if let Some(existing) = self.pending.get_mut(&header.id) {
            if existing.cwd.is_none() {
                existing.cwd = header.cwd;
            }
            if existing.title.is_none() {
                existing.title = header.title;
            }
            if existing.title_source.is_none() {
                existing.title_source = header.title_source;
            }
            if existing.parent_id.is_none() {
                existing.parent_id = header.parent_id;
            }
            if existing.fork_seq.is_none() {
                existing.fork_seq = header.fork_seq;
            }
            existing.created_at_ms = existing.created_at_ms.min(header.created_at_ms);
            existing.updated_at_ms = existing.updated_at_ms.max(header.updated_at_ms);
            existing.compacted_up_to_seq =
                match (existing.compacted_up_to_seq, header.compacted_up_to_seq) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };
        } else {
            self.pending.insert(header.id.clone(), header);
        }
        Ok(())
    }

    fn apply_update_header(
        &mut self,
        session_id: &str,
        patch: &SessionHeaderPatch,
    ) -> PersistenceResult<()> {
        if let Some(pending_header) = self.pending.get_mut(session_id) {
            apply_header_patch(pending_header, patch);
            return Ok(());
        }
        let Some(path) = self.resolve_existing_path(session_id)? else {
            return Ok(());
        };
        if let Some(line) = serialize_header_patch_line(patch)? {
            append_lines_to_file(&path, &[line])?;
        }
        Ok(())
    }

    fn apply_append_events(
        &mut self,
        session_id: &str,
        entries: &[SessionEntry],
    ) -> PersistenceResult<()> {
        if entries.is_empty() {
            return Ok(());
        }

        if let Some(pending_header) = self.pending.get(session_id).cloned() {
            let path = session_file_path(&self.sessions_dir, &pending_header)?;
            let (lines, seen_seqs) =
                build_initial_session_lines(&pending_header, entries, &self.blobs)?;
            write_new_file_atomic(&path, &lines)?;
            self.pending.remove(session_id);
            self.materialized.insert(
                session_id.to_string(),
                MaterializedSession { path, seen_seqs },
            );
            return Ok(());
        }

        if self.resolve_existing_path(session_id)?.is_none() {
            let header = SessionHeader::new(session_id, entries[0].ts);
            let path = session_file_path(&self.sessions_dir, &header)?;
            let (lines, seen_seqs) = build_initial_session_lines(&header, entries, &self.blobs)?;
            write_new_file_atomic(&path, &lines)?;
            self.materialized.insert(
                session_id.to_string(),
                MaterializedSession { path, seen_seqs },
            );
            return Ok(());
        }

        let mat = self
            .materialized
            .get_mut(session_id)
            .expect("resolved session must exist in materialized map");
        let mut lines = Vec::new();
        let mut newly_seen = Vec::new();
        for entry in entries {
            if mat.seen_seqs.contains(&entry.seq) || newly_seen.contains(&entry.seq) {
                continue;
            }
            lines.push(serialize_entry_line(entry, &self.blobs)?);
            newly_seen.push(entry.seq);
        }
        append_lines_to_file(&mat.path, &lines)?;
        mat.seen_seqs.extend(newly_seen);
        Ok(())
    }

    fn exec_fork(
        &mut self,
        from_id: &str,
        up_to_seq: u64,
        new_header: &SessionHeader,
    ) -> ForkOutcome {
        let copied_entries = if self.pending.contains_key(from_id) {
            Vec::new()
        } else {
            let from_path = match self.resolve_existing_path(from_id) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    return ForkOutcome::CallerErr(PersistenceError::NotFound(from_id.to_string()));
                }
                Err(err) => return ForkOutcome::StorageErr(err),
            };
            let loaded = match read_session_file(
                &from_path,
                Some(&self.blobs),
                self.logger.as_ref(),
                true,
            ) {
                Ok(Some(l)) => l,
                Ok(None) => {
                    return ForkOutcome::CallerErr(PersistenceError::NotFound(from_id.to_string()));
                }
                Err(err) => return ForkOutcome::StorageErr(err),
            };
            loaded
                .entries
                .into_iter()
                .filter(|e| e.seq <= up_to_seq)
                .collect()
        };

        let mut fresh_header = new_header.clone();
        fresh_header.event_count = 0;
        fresh_header.surface_count = 0;
        fresh_header.compacted_up_to_seq = copied_entries
            .iter()
            .filter_map(|e| match &e.event {
                SessionEvent::Compaction { first_kept_seq, .. } => Some(*first_kept_seq),
                _ => None,
            })
            .max();

        let child_path = match session_file_path(&self.sessions_dir, &fresh_header) {
            Ok(p) => p,
            Err(err) => return ForkOutcome::CallerErr(err),
        };
        let (lines, seen_seqs) =
            match build_initial_session_lines(&fresh_header, &copied_entries, &self.blobs) {
                Ok(res) => res,
                Err(err) => return ForkOutcome::StorageErr(err),
            };
        if let Err(err) = write_new_file_atomic(&child_path, &lines) {
            return ForkOutcome::StorageErr(err);
        }

        self.materialized.insert(
            fresh_header.id.clone(),
            MaterializedSession {
                path: child_path,
                seen_seqs,
            },
        );

        let max_persisted_seq = copied_entries.last().map(|e| e.seq);
        fresh_header.event_count = copied_entries.len() as u64;
        fresh_header.surface_count =
            copied_entries.iter().filter(|e| e.is_surface()).count() as u64;

        ForkOutcome::Ok(Box::new(LoadedSession {
            header: fresh_header,
            entries: copied_entries,
            max_persisted_seq,
            skipped_events: 0,
        }))
    }
}

enum ForkOutcome {
    Ok(Box<LoadedSession>),
    CallerErr(PersistenceError),
    StorageErr(PersistenceError),
}

fn build_initial_session_lines(
    header: &SessionHeader,
    entries: &[SessionEntry],
    blobs: &BlobStore,
) -> PersistenceResult<(Vec<String>, HashSet<u64>)> {
    let mut lines = Vec::with_capacity(1 + entries.len());
    lines.push(serialize_session_header_line(header)?);
    let mut seen_seqs = HashSet::with_capacity(entries.len());
    for entry in entries {
        if seen_seqs.insert(entry.seq) {
            lines.push(serialize_entry_line(entry, blobs)?);
        }
    }
    Ok((lines, seen_seqs))
}

fn write_new_file_atomic(path: &Path, lines: &[String]) -> PersistenceResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| PersistenceError::InvalidArgument("invalid session file path".into()))?;
    let tmp_path = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let mut buf = String::new();
    for line in lines {
        buf.push_str(line);
        if !line.ends_with('\n') {
            buf.push('\n');
        }
    }
    if let Err(err) = std::fs::write(&tmp_path, buf.as_bytes()) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err.into());
    }
    if let Err(err) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err.into());
    }
    Ok(())
}

fn append_lines_to_file(path: &Path, lines: &[String]) -> PersistenceResult<()> {
    if lines.is_empty() {
        return Ok(());
    }
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .append(true)
        .open(path)?;
    let len = file.metadata()?.len();
    let mut needs_newline = false;
    if len > 0 {
        file.seek(SeekFrom::End(-1))?;
        let mut last = [0u8; 1];
        file.read_exact(&mut last)?;
        if last[0] != b'\n' {
            needs_newline = true;
        }
    }
    let mut buf = String::new();
    if needs_newline {
        buf.push('\n');
    }
    for line in lines {
        buf.push_str(line);
        if !line.ends_with('\n') {
            buf.push('\n');
        }
    }
    file.write_all(buf.as_bytes())?;
    file.flush()?;
    Ok(())
}

fn stops_batch_collection(cmd: &WriteCmd) -> bool {
    matches!(cmd, WriteCmd::ForkSession { .. } | WriteCmd::Shutdown(_))
}

fn clone_err_res(res: &PersistenceResult<()>) -> PersistenceResult<()> {
    match res {
        Ok(()) => Ok(()),
        Err(err) => Err(PersistenceError::Backend(err.to_string())),
    }
}

pub fn run_writer_loop(
    sessions_dir: PathBuf,
    blobs: BlobStore,
    logger: Option<cordis::Logger>,
    rx: mpsc::Receiver<WriteCmd>,
    last_error: Arc<Mutex<Option<String>>>,
) {
    let mut state = WriterState {
        sessions_dir,
        blobs,
        logger,
        pending: HashMap::new(),
        materialized: HashMap::new(),
    };
    let mut sticky_error: Option<String> = None;

    while let Ok(first) = rx.recv() {
        let mut writes = Vec::new();
        let mut flushes = Vec::new();
        let mut tail_action: Option<WriteCmd> = None;

        let mut push_cmd = |cmd: WriteCmd| -> bool {
            let stop = stops_batch_collection(&cmd);
            match cmd {
                WriteCmd::CreateSession(_)
                | WriteCmd::AppendEvents { .. }
                | WriteCmd::UpdateHeader { .. } => writes.push(cmd),
                WriteCmd::Flush(reply) => flushes.push(reply),
                other => tail_action = Some(other),
            }
            stop
        };

        let mut stopped = push_cmd(first);
        let mut count = 1usize;
        while !stopped && count < MAX_BATCH_COMMANDS {
            match rx.try_recv() {
                Ok(cmd) => {
                    count += 1;
                    stopped = push_cmd(cmd);
                }
                Err(_) => break,
            }
        }

        let write_res = if let Some(ref err_msg) = sticky_error {
            Err(PersistenceError::Backend(err_msg.clone()))
        } else if writes.is_empty() {
            Ok(())
        } else {
            let mut batch_res = Ok(());
            for cmd in writes {
                let res = match cmd {
                    WriteCmd::CreateSession(header) => state.apply_create_session(header),
                    WriteCmd::AppendEvents {
                        session_id,
                        entries,
                    } => state.apply_append_events(&session_id, &entries),
                    WriteCmd::UpdateHeader { session_id, patch } => {
                        state.apply_update_header(&session_id, &patch)
                    }
                    _ => Ok(()),
                };
                if let Err(err) = res {
                    let msg = err.to_string();
                    sticky_error = Some(msg.clone());
                    *last_error.lock().unwrap() = Some(msg);
                    batch_res = Err(err);
                    break;
                }
            }
            batch_res
        };

        for reply in flushes {
            reply.send(clone_err_res(&write_res));
        }

        if let Some(action) = tail_action {
            match action {
                WriteCmd::ForkSession {
                    from_id,
                    up_to_seq,
                    new_header,
                    reply,
                } => {
                    let res = match write_res {
                        Ok(()) => match state.exec_fork(&from_id, up_to_seq, &new_header) {
                            ForkOutcome::Ok(loaded) => Ok(*loaded),
                            ForkOutcome::CallerErr(err) => Err(err),
                            ForkOutcome::StorageErr(err) => {
                                let msg = err.to_string();
                                sticky_error = Some(msg.clone());
                                *last_error.lock().unwrap() = Some(msg);
                                Err(err)
                            }
                        },
                        Err(err) => Err(err),
                    };
                    let _ = reply.send(res);
                }
                WriteCmd::Shutdown(reply) => {
                    let _ = reply.send(write_res);
                    break;
                }
                WriteCmd::CreateSession(_)
                | WriteCmd::AppendEvents { .. }
                | WriteCmd::UpdateHeader { .. }
                | WriteCmd::Flush(_) => {}
            }
        }
    }
}
