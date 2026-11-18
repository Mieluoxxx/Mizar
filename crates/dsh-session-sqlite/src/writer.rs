//! 专用 SQLite Single-Writer 后台线程：通过批量事务提交写入并隔离 Tokio 异步调度。

use std::sync::mpsc;

use dsh_session::{
    LoadedSession, PersistenceError, PersistenceResult, SESSION_FORMAT_VERSION, SessionEntry,
    SessionEvent, SessionHeader, SessionHeaderPatch, SessionListQuery,
};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use tokio::sync::oneshot;

use crate::schema::{
    list_sessions_from_conn, load_session_from_conn, map_sqlite_err, millis_to_i64, visibility_str,
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
    Flush(BarrierReply),
    Shutdown(mpsc::SyncSender<PersistenceResult<()>>),
}

fn stops_batch_collection(cmd: &WriteCmd) -> bool {
    matches!(
        cmd,
        WriteCmd::ForkSession { .. }
            | WriteCmd::LoadSession { .. }
            | WriteCmd::ListSessions { .. }
            | WriteCmd::Shutdown(_)
    )
}

fn clone_err_res(res: &PersistenceResult<()>) -> PersistenceResult<()> {
    match res {
        Ok(()) => Ok(()),
        Err(err) => Err(PersistenceError::Backend(err.to_string())),
    }
}

pub fn run_writer_loop(mut conn: Connection, rx: mpsc::Receiver<WriteCmd>) {
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

        let write_res = if writes.is_empty() {
            Ok(())
        } else {
            apply_write_batch(&mut conn, &writes)
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

fn apply_write_batch(conn: &mut Connection, writes: &[WriteCmd]) -> PersistenceResult<()> {
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(map_sqlite_err)?;

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

    tx.commit().map_err(map_sqlite_err)?;
    Ok(())
}

fn insert_session_header(tx: &Transaction<'_>, h: &SessionHeader) -> PersistenceResult<()> {
    let mut stmt = tx
        .prepare_cached(
            "INSERT OR IGNORE INTO sessions (
                 id, format_version, cwd, title, parent_id, fork_seq,
                 created_at_ms, updated_at_ms, event_count, surface_count, compacted_up_to_seq
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )
        .map_err(map_sqlite_err)?;

    stmt.execute(params![
        h.id,
        i64::from(h.format_version),
        h.cwd,
        h.title,
        h.parent_id,
        h.fork_seq.map(|v| v as i64),
        millis_to_i64(h.created_at_ms),
        millis_to_i64(h.updated_at_ms),
        h.event_count as i64,
        h.surface_count as i64,
        h.compacted_up_to_seq.map(|v| v as i64),
    ])
    .map_err(map_sqlite_err)?;
    Ok(())
}

fn append_entries_in_tx(
    tx: &Transaction<'_>,
    session_id: &str,
    entries: &[SessionEntry],
) -> PersistenceResult<()> {
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
            .map_err(map_sqlite_err)?;
        ensure
            .execute(params![session_id, i64::from(SESSION_FORMAT_VERSION), first_ts])
            .map_err(map_sqlite_err)?;
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
            .map_err(map_sqlite_err)?;

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
                .map_err(map_sqlite_err)?;
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
            .map_err(map_sqlite_err)?;
        upd.execute(params![
            session_id,
            max_ts,
            added_total,
            added_surface,
            max_compacted.map(|v| v as i64),
        ])
        .map_err(map_sqlite_err)?;
    }

    Ok(())
}

fn update_header_in_tx(
    tx: &Transaction<'_>,
    session_id: &str,
    patch: &SessionHeaderPatch,
) -> PersistenceResult<()> {
    let mut stmt = tx
        .prepare_cached(
            "UPDATE sessions
             SET title = COALESCE(?2, title),
                 updated_at_ms = CASE
                     WHEN ?3 IS NOT NULL THEN MAX(updated_at_ms, ?3)
                     ELSE updated_at_ms
                 END,
                 compacted_up_to_seq = CASE
                     WHEN ?4 IS NOT NULL THEN MAX(COALESCE(compacted_up_to_seq, 0), ?4)
                     ELSE compacted_up_to_seq
                 END
             WHERE id = ?1",
        )
        .map_err(map_sqlite_err)?;
    stmt.execute(params![
        session_id,
        patch.title,
        patch.updated_at_ms.map(millis_to_i64),
        patch.compacted_up_to_seq.map(|v| v as i64),
    ])
    .map_err(map_sqlite_err)?;
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
    insert_session_header(&tx, &fresh_header)?;

    tx.execute(
        "INSERT INTO session_events (session_id, seq, ts_ms, visibility, kind, payload_json)
         SELECT ?1, seq, ts_ms, visibility, kind, payload_json
         FROM session_events
         WHERE session_id = ?2 AND seq <= ?3
         ORDER BY seq ASC",
        params![new_header.id, from_id, up_to_seq as i64],
    )
    .map_err(map_sqlite_err)?;

    tx.execute(
        "UPDATE sessions
         SET event_count = (SELECT COUNT(*) FROM session_events WHERE session_id = ?1),
             surface_count = (SELECT COUNT(*) FROM session_events WHERE session_id = ?1 AND visibility = 'surface')
         WHERE id = ?1",
        [&new_header.id],
    )
    .map_err(map_sqlite_err)?;

    let loaded = load_session_from_conn(&tx, &new_header.id)?
        .ok_or_else(|| PersistenceError::NotFound(new_header.id.clone()))?;
    tx.commit().map_err(map_sqlite_err)?;
    Ok(loaded)
}
