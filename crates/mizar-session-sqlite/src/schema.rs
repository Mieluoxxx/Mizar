//! SQLite Schema 初始化与连接参数配置（WAL + busy_timeout）。

use std::time::Duration;

use mizar_session::{
    EntryVisibility, LoadedSession, PersistenceError, PersistenceResult, SESSION_FORMAT_VERSION,
    SessionEntry, SessionEvent, SessionHeader, SessionListQuery,
};
use rusqlite::{Connection, OptionalExtension, params_from_iter};

pub const CURRENT_SCHEMA_VERSION: u32 = 1;

pub fn configure_connection(conn: &Connection) -> PersistenceResult<()> {
    conn.busy_timeout(Duration::from_millis(5000))
        .map_err(map_sqlite_err)?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;",
    )
    .map_err(map_sqlite_err)?;
    Ok(())
}

pub fn configure_reader_connection(conn: &Connection) -> PersistenceResult<()> {
    conn.busy_timeout(Duration::from_millis(5000))
        .map_err(map_sqlite_err)?;
    conn.execute_batch("PRAGMA query_only = ON;")
        .map_err(map_sqlite_err)?;
    Ok(())
}

pub fn init_schema(conn: &Connection) -> PersistenceResult<()> {
    configure_connection(conn)?;
    let version: u32 = conn
        .query_row("PRAGMA user_version;", [], |r| r.get(0))
        .map_err(map_sqlite_err)?;
    if version > CURRENT_SCHEMA_VERSION {
        return Err(PersistenceError::Backend(format!(
            "unsupported sqlite session schema version {version} (max supported: {CURRENT_SCHEMA_VERSION})"
        )));
    }
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sessions (
             id TEXT PRIMARY KEY,
             format_version INTEGER NOT NULL,
             cwd TEXT,
             title TEXT,
             parent_id TEXT,
             fork_seq INTEGER,
             created_at_ms INTEGER NOT NULL,
             updated_at_ms INTEGER NOT NULL,
             event_count INTEGER NOT NULL DEFAULT 0,
             surface_count INTEGER NOT NULL DEFAULT 0,
             compacted_up_to_seq INTEGER
         );
         CREATE INDEX IF NOT EXISTS idx_sessions_cwd_updated
             ON sessions(cwd, updated_at_ms DESC);
         CREATE INDEX IF NOT EXISTS idx_sessions_updated
             ON sessions(updated_at_ms DESC);
         CREATE TABLE IF NOT EXISTS session_events (
             session_id TEXT NOT NULL,
             seq INTEGER NOT NULL,
             ts_ms INTEGER NOT NULL,
             visibility TEXT NOT NULL,
             kind TEXT NOT NULL,
             payload_json TEXT NOT NULL,
             PRIMARY KEY (session_id, seq),
             FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
         );
         CREATE TABLE IF NOT EXISTS session_recall_docs (
             doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
             session_id TEXT NOT NULL,
             seq INTEGER NOT NULL,
             ts_ms INTEGER NOT NULL,
             kind TEXT NOT NULL,
             preview TEXT NOT NULL,
             UNIQUE(session_id, seq),
             FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
         );
         CREATE INDEX IF NOT EXISTS idx_recall_docs_session_seq
             ON session_recall_docs(session_id, seq);
         CREATE VIRTUAL TABLE IF NOT EXISTS session_recall_fts USING fts5(
             terms,
             content='',
             tokenize='ascii'
         );
         PRAGMA user_version = 1;",
    )
    .map_err(map_sqlite_err)?;

    Ok(())
}

pub fn map_sqlite_err(err: rusqlite::Error) -> PersistenceError {
    PersistenceError::Backend(format!("sqlite: {err}"))
}

pub fn millis_to_i64(ms: u128) -> i64 {
    i64::try_from(ms).unwrap_or(i64::MAX)
}

pub fn i64_to_millis(v: i64) -> u128 {
    u128::try_from(v.max(0)).unwrap_or(0)
}

pub fn visibility_str(v: EntryVisibility) -> &'static str {
    match v {
        EntryVisibility::Surface => "surface",
        EntryVisibility::LogOnly => "log_only",
    }
}

fn row_to_header(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionHeader> {
    let format_version: i64 = row.get(1)?;
    let fork_seq: Option<i64> = row.get(5)?;
    let created_at_ms: i64 = row.get(6)?;
    let updated_at_ms: i64 = row.get(7)?;
    let event_count: i64 = row.get(8)?;
    let surface_count: i64 = row.get(9)?;
    let compacted_up_to_seq: Option<i64> = row.get(10)?;

    Ok(SessionHeader {
        id: row.get(0)?,
        format_version: u32::try_from(format_version.max(1)).unwrap_or(SESSION_FORMAT_VERSION),
        cwd: row.get(2)?,
        title: row.get(3)?,
        parent_id: row.get(4)?,
        fork_seq: fork_seq.and_then(|v| u64::try_from(v).ok()),
        created_at_ms: i64_to_millis(created_at_ms),
        updated_at_ms: i64_to_millis(updated_at_ms),
        event_count: u64::try_from(event_count.max(0)).unwrap_or(0),
        surface_count: u64::try_from(surface_count.max(0)).unwrap_or(0),
        compacted_up_to_seq: compacted_up_to_seq.and_then(|v| u64::try_from(v).ok()),
    })
}

pub fn load_session_from_conn(
    conn: &Connection,
    session_id: &str,
) -> PersistenceResult<Option<LoadedSession>> {
    let header = conn
        .query_row(
            "SELECT id, format_version, cwd, title, parent_id, fork_seq,
                    created_at_ms, updated_at_ms, event_count, surface_count, compacted_up_to_seq
             FROM sessions
             WHERE id = ?1",
            [session_id],
            row_to_header,
        )
        .optional()
        .map_err(map_sqlite_err)?;

    let Some(header) = header else {
        return Ok(None);
    };

    let mut stmt = conn
        .prepare_cached(
            "SELECT seq, ts_ms, payload_json
             FROM session_events
             WHERE session_id = ?1
             ORDER BY seq ASC",
        )
        .map_err(map_sqlite_err)?;

    let rows = stmt
        .query_map([session_id], |row| {
            let seq: i64 = row.get(0)?;
            let ts_ms: i64 = row.get(1)?;
            let payload_json: String = row.get(2)?;
            Ok((seq, ts_ms, payload_json))
        })
        .map_err(map_sqlite_err)?;

    let mut entries = Vec::new();
    let mut skipped_events = 0usize;
    for item in rows {
        let (seq, ts_ms, payload_json) = item.map_err(map_sqlite_err)?;
        // 单条解析失败不能升级为整个会话加载失败：旧版二进制遇到新版写入的事件类型时
        // 应跳过该条并保留其余历史，否则回滚一次就丢全部会话。
        let event: SessionEvent = match serde_json::from_str(&payload_json) {
            Ok(event) => event,
            Err(_) => {
                // 具体错误由调用方根据 `skipped_events` 汇总告警，避免在写线程里直接写 stdout/stderr。
                skipped_events += 1;
                continue;
            }
        };
        entries.push(SessionEntry {
            seq: u64::try_from(seq.max(0)).unwrap_or(0),
            event,
            ts: i64_to_millis(ts_ms),
        });
    }

    Ok(Some(LoadedSession {
        header,
        entries,
        skipped_events,
    }))
}

pub fn list_sessions_from_conn(
    conn: &Connection,
    query: &SessionListQuery,
) -> PersistenceResult<Vec<SessionHeader>> {
    let mut sql = String::from(
        "SELECT id, format_version, cwd, title, parent_id, fork_seq,
                created_at_ms, updated_at_ms, event_count, surface_count, compacted_up_to_seq
         FROM sessions",
    );
    let mut params: Vec<rusqlite::types::Value> = Vec::new();

    if let Some(cwd) = &query.cwd {
        sql.push_str(" WHERE cwd = ?1");
        params.push(rusqlite::types::Value::Text(cwd.clone()));
    }

    sql.push_str(" ORDER BY updated_at_ms DESC, id ASC");

    if let Some(limit) = query.limit {
        let idx = params.len() + 1;
        sql.push_str(&format!(" LIMIT ?{idx}"));
        params.push(rusqlite::types::Value::Integer(
            i64::try_from(limit).unwrap_or(i64::MAX),
        ));
    }

    let mut stmt = conn.prepare(&sql).map_err(map_sqlite_err)?;
    let rows = stmt
        .query_map(params_from_iter(params), row_to_header)
        .map_err(map_sqlite_err)?;

    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(map_sqlite_err)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 写入一条无法解析的事件不得导致整个会话加载失败：
    /// 旧版二进制遇到新版写入的事件类型时，应跳过该条并保留其余历史。
    #[test]
    fn skips_unparsable_event_and_keeps_the_rest() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        let header = SessionHeader::new("sess-partial", 1000);
        insert_test_session(&conn, &header);
        insert_test_event(
            &conn,
            1,
            r#"{"type":"user","text":"第一条"}"#,
            EntryVisibility::Surface,
        );
        insert_test_event(
            &conn,
            2,
            r#"{"type":"unknown-future-event","payload":1}"#,
            EntryVisibility::LogOnly,
        );
        insert_test_event(
            &conn,
            3,
            r#"{"type":"user","text":"第三条"}"#,
            EntryVisibility::Surface,
        );

        let loaded = load_session_from_conn(&conn, "sess-partial")
            .expect("单条坏事件不应让整个会话加载失败")
            .expect("会话应存在");

        assert_eq!(loaded.skipped_events, 1);
        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(loaded.entries[0].seq, 1);
        assert_eq!(loaded.entries[1].seq, 3);
    }

    fn insert_test_session(conn: &Connection, header: &SessionHeader) {
        conn.execute(
            "INSERT INTO sessions (id, format_version, cwd, title, parent_id, fork_seq,
                                   created_at_ms, updated_at_ms, event_count, surface_count,
                                   compacted_up_to_seq)
             VALUES (?1, ?2, NULL, NULL, NULL, NULL, ?3, ?3, 0, 0, NULL)",
            rusqlite::params![
                header.id,
                i64::from(header.format_version),
                millis_to_i64(header.created_at_ms)
            ],
        )
        .unwrap();
    }

    fn insert_test_event(conn: &Connection, seq: i64, payload: &str, visibility: EntryVisibility) {
        conn.execute(
            "INSERT INTO session_events (session_id, seq, ts_ms, visibility, kind, payload_json)
             VALUES ('sess-partial', ?1, ?2, ?3, 'x', ?4)",
            rusqlite::params![seq, seq * 1000, visibility_str(visibility), payload],
        )
        .unwrap();
    }
}
