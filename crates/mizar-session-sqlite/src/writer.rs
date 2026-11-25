//! 专用 SQLite Single-Writer 后台线程：通过批量事务提交写入并隔离 Tokio 异步调度。

use std::sync::mpsc;

use mizar_session::{
    LoadedSession, PersistenceError, PersistenceResult, RecallSearchHit, RecallSearchQuery,
    SESSION_FORMAT_VERSION, SessionEntry, SessionEvent, SessionHeader, SessionHeaderPatch,
    SessionListQuery, TitlePatch, TitleSource, TitleSourcePatch,
};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use tokio::sync::oneshot;

use crate::recall::{index_entry_in_tx, index_session_events_in_tx, search_recall_from_conn};
use crate::schema::{
    is_sqlite_busy, list_sessions_from_conn, load_session_from_conn, map_sqlite_err, millis_to_i64,
    visibility_str,
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
    LoadSession {
        session_id: String,
        reply: mpsc::SyncSender<PersistenceResult<Option<LoadedSession>>>,
    },
    ListSessions {
        query: SessionListQuery,
        reply: mpsc::SyncSender<PersistenceResult<Vec<SessionHeader>>>,
    },
    SearchRecall {
        query: RecallSearchQuery,
        reply: mpsc::SyncSender<PersistenceResult<Vec<RecallSearchHit>>>,
    },
    Flush(BarrierReply),
    Shutdown(mpsc::SyncSender<PersistenceResult<()>>),
}

fn stops_batch_collection(cmd: &WriteCmd) -> bool {
    matches!(
        cmd,
        WriteCmd::ForkSession { .. }
            | WriteCmd::LoadSession { .. }
            | WriteCmd::ListSessions { .. }
            | WriteCmd::SearchRecall { .. }
            | WriteCmd::Shutdown(_)
    )
}

enum BatchWriteError {
    /// 跨进程锁竞争等瞬时 SQLite 冲突：当前批次失败，但不得污染写线程的 `sticky_error`。
    Transient(PersistenceError),
    Fatal(PersistenceError),
}

impl BatchWriteError {
    fn into_persistence(self) -> PersistenceError {
        match self {
            Self::Transient(err) | Self::Fatal(err) => err,
        }
    }
}

impl From<serde_json::Error> for BatchWriteError {
    fn from(err: serde_json::Error) -> Self {
        Self::Fatal(PersistenceError::Serialization(err))
    }
}

fn map_batch_sqlite_err(err: rusqlite::Error) -> BatchWriteError {
    if is_sqlite_busy(&err) {
        BatchWriteError::Transient(map_sqlite_err(err))
    } else {
        BatchWriteError::Fatal(map_sqlite_err(err))
    }
}

fn map_batch_persistence_err(err: PersistenceError) -> BatchWriteError {
    BatchWriteError::Fatal(err)
}

fn clone_err_res(res: &PersistenceResult<()>) -> PersistenceResult<()> {
    match res {
        Ok(()) => Ok(()),
        Err(err) => Err(PersistenceError::Backend(err.to_string())),
    }
}

pub fn run_writer_loop(
    mut conn: Connection,
    rx: mpsc::Receiver<WriteCmd>,
    last_error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
) {
    let mut sticky_error: Option<String> = None;
    let mut pending_transient_error: Option<String> = None;

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

        let has_observer = !flushes.is_empty() || tail_action.is_some();
        let write_res = if let Some(ref err_msg) = sticky_error {
            Err(PersistenceError::Backend(err_msg.clone()))
        } else if writes.is_empty() {
            if has_observer {
                match pending_transient_error.take() {
                    Some(err_msg) => Err(PersistenceError::Backend(err_msg)),
                    None => Ok(()),
                }
            } else {
                Ok(())
            }
        } else {
            match apply_write_batch(&mut conn, &writes) {
                Ok(()) => {
                    if has_observer {
                        match pending_transient_error.take() {
                            Some(err_msg) => Err(PersistenceError::Backend(err_msg)),
                            None => Ok(()),
                        }
                    } else {
                        Ok(())
                    }
                }
                Err(BatchWriteError::Transient(err)) => {
                    // 无观察者批次的瞬时忙碌不能直接丢弃，暂存到下一次 flush / 同步屏障消费一次
                    if has_observer {
                        pending_transient_error = None;
                    } else {
                        pending_transient_error = Some(err.to_string());
                    }
                    Err(err)
                }
                Err(BatchWriteError::Fatal(err)) => {
                    pending_transient_error = None;
                    let msg = err.to_string();
                    sticky_error = Some(msg.clone());
                    *last_error.lock().unwrap() = Some(msg);
                    Err(err)
                }
            }
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
                        Ok(()) => exec_fork(&mut conn, &from_id, up_to_seq, &new_header),
                        Err(err) => Err(err),
                    };
                    let _ = reply.send(res);
                }
                WriteCmd::LoadSession { session_id, reply } => {
                    let res = match write_res {
                        Ok(()) => load_session_from_conn(&conn, &session_id),
                        Err(err) => Err(err),
                    };
                    let _ = reply.send(res);
                }
                WriteCmd::ListSessions { query, reply } => {
                    let res = match write_res {
                        Ok(()) => list_sessions_from_conn(&conn, &query),
                        Err(err) => Err(err),
                    };
                    let _ = reply.send(res);
                }
                WriteCmd::SearchRecall { query, reply } => {
                    let res = match write_res {
                        Ok(()) => search_recall_from_conn(&conn, &query),
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

fn apply_write_batch(conn: &mut Connection, writes: &[WriteCmd]) -> Result<(), BatchWriteError> {
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(map_batch_sqlite_err)?;

    for cmd in writes {
        match cmd {
            WriteCmd::CreateSession(header) => insert_session_header(&tx, header)?,
            WriteCmd::AppendEvents {
                session_id,
                entries,
            } => append_entries_in_tx(&tx, session_id, entries)?,
            WriteCmd::UpdateHeader { session_id, patch } => {
                update_header_in_tx(&tx, session_id, patch)?
            }
            _ => {}
        }
    }

    tx.commit().map_err(map_batch_sqlite_err)?;
    Ok(())
}

fn insert_session_header(tx: &Transaction<'_>, h: &SessionHeader) -> Result<(), BatchWriteError> {
    let mut stmt = tx
        .prepare_cached(
            "INSERT INTO sessions (
                 id, format_version, cwd, title, title_source, parent_id, fork_seq,
                 created_at_ms, updated_at_ms, event_count, surface_count, compacted_up_to_seq
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(id) DO UPDATE SET
                 cwd = COALESCE(sessions.cwd, excluded.cwd),
                 title = COALESCE(sessions.title, excluded.title),
                 title_source = COALESCE(sessions.title_source, excluded.title_source),
                 parent_id = COALESCE(sessions.parent_id, excluded.parent_id),
                 fork_seq = COALESCE(sessions.fork_seq, excluded.fork_seq),
                 created_at_ms = MIN(sessions.created_at_ms, excluded.created_at_ms),
                 updated_at_ms = MAX(sessions.updated_at_ms, excluded.updated_at_ms),
                 event_count = MAX(sessions.event_count, excluded.event_count),
                 surface_count = MAX(sessions.surface_count, excluded.surface_count),
                 compacted_up_to_seq = CASE
                     WHEN excluded.compacted_up_to_seq IS NOT NULL THEN MAX(COALESCE(sessions.compacted_up_to_seq, 0), excluded.compacted_up_to_seq)
                     ELSE sessions.compacted_up_to_seq
                 END",
        )
        .map_err(map_batch_sqlite_err)?;

    stmt.execute(params![
        h.id,
        i64::from(h.format_version),
        h.cwd,
        h.title,
        h.title_source.map(TitleSource::as_str),
        h.parent_id,
        h.fork_seq.map(|v| v as i64),
        millis_to_i64(h.created_at_ms),
        millis_to_i64(h.updated_at_ms),
        h.event_count as i64,
        h.surface_count as i64,
        h.compacted_up_to_seq.map(|v| v as i64),
    ])
    .map_err(map_batch_sqlite_err)?;
    Ok(())
}

fn append_entries_in_tx(
    tx: &Transaction<'_>,
    session_id: &str,
    entries: &[SessionEntry],
) -> Result<(), BatchWriteError> {
    if entries.is_empty() {
        return Ok(());
    }

    let first_ts = millis_to_i64(entries[0].ts);
    {
        let mut ensure = tx
            .prepare_cached(
                "INSERT OR IGNORE INTO sessions (
                     id, format_version, created_at_ms, updated_at_ms
                 ) VALUES (?1, ?2, ?3, ?3)",
            )
            .map_err(map_batch_sqlite_err)?;
        ensure
            .execute(params![
                session_id,
                i64::from(SESSION_FORMAT_VERSION),
                first_ts
            ])
            .map_err(map_batch_sqlite_err)?;
    }

    let mut added_total = 0i64;
    let mut added_surface = 0i64;
    let mut max_ts = 0i64;
    let mut max_compacted: Option<u64> = None;

    {
        let mut stmt = tx
            .prepare_cached(
                "INSERT OR IGNORE INTO session_events (
                     session_id, seq, ts_ms, visibility, kind, payload_json
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )
            .map_err(map_batch_sqlite_err)?;

        for entry in entries {
            let ts_i64 = millis_to_i64(entry.ts);
            let payload_json = serde_json::to_string(&entry.event)?;
            let changed = stmt
                .execute(params![
                    session_id,
                    entry.seq as i64,
                    ts_i64,
                    visibility_str(entry.visibility()),
                    entry.event.kind_name(),
                    payload_json,
                ])
                .map_err(map_batch_sqlite_err)?;
            if changed > 0 {
                added_total += 1;
                if entry.is_surface() {
                    added_surface += 1;
                }
                max_ts = max_ts.max(ts_i64);
                if let SessionEvent::Compaction { first_kept_seq, .. } = &entry.event {
                    max_compacted = Some(
                        max_compacted.map_or(*first_kept_seq, |prev| prev.max(*first_kept_seq)),
                    );
                }
                index_entry_in_tx(tx, session_id, entry).map_err(map_batch_persistence_err)?;
            }
        }
    }

    if added_total > 0 {
        let mut upd = tx
            .prepare_cached(
                "UPDATE sessions
                 SET updated_at_ms = MAX(updated_at_ms, ?2),
                     event_count = event_count + ?3,
                     surface_count = surface_count + ?4,
                     compacted_up_to_seq = CASE
                         WHEN ?5 IS NOT NULL THEN MAX(COALESCE(compacted_up_to_seq, 0), ?5)
                         ELSE compacted_up_to_seq
                     END
                 WHERE id = ?1",
            )
            .map_err(map_batch_sqlite_err)?;
        upd.execute(params![
            session_id,
            max_ts,
            added_total,
            added_surface,
            max_compacted.map(|v| v as i64),
        ])
        .map_err(map_batch_sqlite_err)?;
    }

    Ok(())
}

fn update_header_in_tx(
    tx: &Transaction<'_>,
    session_id: &str,
    patch: &SessionHeaderPatch,
) -> Result<(), BatchWriteError> {
    let (apply_title, title_val): (i64, Option<&str>) = match &patch.title {
        TitlePatch::Keep => (0, None),
        TitlePatch::Set(v) => (1, Some(v.as_str())),
        TitlePatch::Clear => (1, None),
    };
    let (apply_source, source_val): (i64, Option<&str>) = match patch.title_source {
        TitleSourcePatch::Keep => (0, None),
        TitleSourcePatch::Set(s) => (1, Some(s.as_str())),
        TitleSourcePatch::Clear => (1, None),
    };
    let mut stmt = tx
        .prepare_cached(
            "UPDATE sessions
             SET title = CASE
                     WHEN ?2 != 0 THEN ?3
                     ELSE title
                 END,
                 title_source = CASE
                     WHEN ?4 != 0 THEN ?5
                     ELSE title_source
                 END,
                 updated_at_ms = CASE
                     WHEN ?6 IS NOT NULL THEN MAX(updated_at_ms, ?6)
                     ELSE updated_at_ms
                 END,
                 compacted_up_to_seq = CASE
                     WHEN ?7 IS NOT NULL THEN MAX(COALESCE(compacted_up_to_seq, 0), ?7)
                     ELSE compacted_up_to_seq
                 END
             WHERE id = ?1",
        )
        .map_err(map_batch_sqlite_err)?;
    stmt.execute(params![
        session_id,
        apply_title,
        title_val,
        apply_source,
        source_val,
        patch.updated_at_ms.map(millis_to_i64),
        patch.compacted_up_to_seq.map(|v| v as i64),
    ])
    .map_err(map_batch_sqlite_err)?;
    Ok(())
}

fn exec_fork(
    conn: &mut Connection,
    from_id: &str,
    up_to_seq: u64,
    new_header: &SessionHeader,
) -> PersistenceResult<LoadedSession> {
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(map_sqlite_err)?;

    let exists: Option<i64> = tx
        .query_row("SELECT 1 FROM sessions WHERE id = ?1", [from_id], |r| {
            r.get(0)
        })
        .optional()
        .map_err(map_sqlite_err)?;
    if exists.is_none() {
        return Err(PersistenceError::NotFound(from_id.to_string()));
    }

    let mut fresh_header = new_header.clone();
    fresh_header.event_count = 0;
    fresh_header.surface_count = 0;
    fresh_header.compacted_up_to_seq = None;
    insert_session_header(&tx, &fresh_header).map_err(BatchWriteError::into_persistence)?;

    tx.execute(
        "INSERT INTO session_events (session_id, seq, ts_ms, visibility, kind, payload_json)
         SELECT ?1, seq, ts_ms, visibility, kind, payload_json
         FROM session_events
         WHERE session_id = ?2 AND seq <= ?3
         ORDER BY seq ASC",
        params![new_header.id, from_id, up_to_seq as i64],
    )
    .map_err(map_sqlite_err)?;

    let copied_compacted_up_to_seq: Option<u64> = {
        let mut stmt = tx
            .prepare_cached(
                "SELECT payload_json
                 FROM session_events
                 WHERE session_id = ?1 AND kind = 'compaction'",
            )
            .map_err(map_sqlite_err)?;
        let rows = stmt
            .query_map([&new_header.id], |r| r.get::<_, String>(0))
            .map_err(map_sqlite_err)?;
        let mut max_kept: Option<u64> = None;
        for row in rows {
            let json = row.map_err(map_sqlite_err)?;
            if let Ok(SessionEvent::Compaction { first_kept_seq, .. }) =
                serde_json::from_str::<SessionEvent>(&json)
            {
                max_kept = Some(max_kept.map_or(first_kept_seq, |prev| prev.max(first_kept_seq)));
            }
        }
        max_kept
    };

    tx.execute(
        "UPDATE sessions
         SET event_count = (SELECT COUNT(*) FROM session_events WHERE session_id = ?1),
             surface_count = (SELECT COUNT(*) FROM session_events WHERE session_id = ?1 AND visibility = 'surface'),
             compacted_up_to_seq = ?2
         WHERE id = ?1",
        params![
            &new_header.id,
            copied_compacted_up_to_seq.map(|v| v as i64)
        ],
    )
    .map_err(map_sqlite_err)?;

    index_session_events_in_tx(&tx, &new_header.id)?;

    let loaded = load_session_from_conn(&tx, &new_header.id)?
        .ok_or_else(|| PersistenceError::NotFound(new_header.id.clone()))?;
    tx.commit().map_err(map_sqlite_err)?;
    Ok(loaded)
}
