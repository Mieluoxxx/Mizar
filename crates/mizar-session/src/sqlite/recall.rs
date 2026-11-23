//! 基于 SQLite 无内容 FTS5（`tokenize='ascii'`）的 1..=3 Unicode scalar n-gram 情景回溯索引。
//!
//! 借鉴 `maka` 的无内容 n-gram 方案：
//! - 将 1..=3 个 Unicode 标量值打包为唯一 63-bit 整数并编码为 ASCII base-36 token（`g...`），
//!   同时支持中文短词、英文子串、Rust 标识符（`::` / `_`）与文件路径精确匹配；
//! - 索引与 `session_events` 在同一 SQLite 写事务内原子更新，零额外线程；
//! - 严禁把 `Compaction` 视为搜索壁垒：压缩前后全部表面事件均可检索。

use std::collections::HashSet;

use crate::{PersistenceResult, RecallSearchHit, RecallSearchQuery, SessionEntry, SessionEvent};
use mizar_types::llm::ContentBlock;
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};

const MAX_INDEX_CHARS: usize = 16_384;
const MAX_DOC_TOKENS: usize = 4_096;
const MAX_QUERY_TOKENS: usize = 24;
const MAX_PREVIEW_CHARS: usize = 4_000;
const SNIPPET_MAX_CHARS: usize = 240;
pub const MAX_RECALL_SEARCH_HITS: usize = 20;
pub const MAX_RECALL_SEARCH_TOTAL_BYTES: usize = 16 * 1024;

/// 从 `SessionEntry` 提取参与情景回溯索引的纯文本（跳过 `thinking`、`Trace` 与合成通知噪音）。
pub fn extract_indexable_text(event: &SessionEvent) -> Option<String> {
    let raw = match event {
        SessionEvent::User { text, .. } => text.clone(),
        SessionEvent::Assistant { blocks, .. } => {
            let mut parts = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text } if !text.trim().is_empty() => {
                        parts.push(text.clone());
                    }
                    ContentBlock::ToolCall {
                        name, arguments, ..
                    } => {
                        let args_snip = truncate_chars(arguments, 512);
                        parts.push(format!("[tool_call:{name}] {args_snip}"));
                    }
                    ContentBlock::Reasoning { .. } | ContentBlock::Text { .. } => {}
                }
            }
            if parts.is_empty() {
                return None;
            }
            parts.join("\n")
        }
        SessionEvent::ToolResult { name, content, .. } => {
            if content.trim().is_empty() {
                return None;
            }
            format!("[tool:{name}] {}", truncate_chars(content, MAX_INDEX_CHARS))
        }
        SessionEvent::BashExecution {
            command, output, ..
        } => {
            let out_snip = truncate_chars(output, MAX_INDEX_CHARS);
            format!("$ {command}\n{out_snip}")
        }
        SessionEvent::Compaction { summary, .. } => {
            if summary.trim().is_empty() {
                return None;
            }
            format!("[compaction] {}", truncate_chars(summary, MAX_INDEX_CHARS))
        }
        SessionEvent::ContextNotice { .. }
        | SessionEvent::Selection { .. }
        | SessionEvent::Usage { .. }
        | SessionEvent::Trace { .. } => return None,
    };

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(truncate_chars(trimmed, MAX_INDEX_CHARS))
    }
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut iter = s.chars();
    let taken: String = iter.by_ref().take(max_chars).collect();
    taken
}

fn pack_ngram(c0: char, c1: Option<char>, c2: Option<char>) -> u64 {
    let v0 = (c0 as u32) as u64;
    let v1 = c1.map_or(0u64, |c| (c as u32) as u64);
    let v2 = c2.map_or(0u64, |c| (c as u32) as u64);
    v0 | (v1 << 21) | (v2 << 42)
}

fn encode_base36_token(mut value: u64) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut buf = [0u8; 16];
    let mut len = 0usize;
    if value == 0 {
        buf[0] = b'0';
        len = 1;
    } else {
        while value > 0 {
            buf[len] = DIGITS[(value % 36) as usize];
            value /= 36;
            len += 1;
        }
    }
    let mut out = String::with_capacity(len + 1);
    out.push('g');
    for i in (0..len).rev() {
        out.push(buf[i] as char);
    }
    out
}

fn normalize_spans(text: &str) -> Vec<Vec<char>> {
    let mut spans = Vec::new();
    let mut current = Vec::new();
    for ch in text.chars() {
        if ch == '\0' {
            continue;
        }
        if ch.is_whitespace() {
            if !current.is_empty() {
                spans.push(std::mem::take(&mut current));
            }
        } else {
            for lower in ch.to_lowercase() {
                if lower != '\0' && !lower.is_whitespace() {
                    current.push(lower);
                }
            }
        }
    }
    if !current.is_empty() {
        spans.push(current);
    }
    spans
}

/// 将文档正文编码为空格分隔的 `g...` n-gram token 串，写入无内容 FTS5。
pub fn encode_document_terms(text: &str) -> String {
    let spans = normalize_spans(text);
    let mut seen: HashSet<u64> = HashSet::new();
    let mut tokens: Vec<String> = Vec::new();

    let mut push_key = |key: u64| -> bool {
        if tokens.len() >= MAX_DOC_TOKENS {
            return false;
        }
        if seen.insert(key) {
            tokens.push(encode_base36_token(key));
        }
        tokens.len() < MAX_DOC_TOKENS
    };

    'outer: for span in &spans {
        let n = span.len();
        if n == 0 {
            continue;
        }
        for i in 0..n {
            let c0 = span[i];
            // 1-gram：对所有非 ASCII（如中文单字）以及单字符 span 建索引，避免纯 ASCII 字母膨胀
            if (!c0.is_ascii() || n == 1) && !push_key(pack_ngram(c0, None, None)) {
                break 'outer;
            }
            // 2-gram：对非 ASCII 或短词（n == 2）建索引
            if i + 1 < n {
                let c1 = span[i + 1];
                if (!c0.is_ascii() || !c1.is_ascii() || n == 2)
                    && !push_key(pack_ngram(c0, Some(c1), None))
                {
                    break 'outer;
                }
            }
            // 3-gram：覆盖所有长度 >= 3 的中英文与代码标识符子串
            if i + 2 < n && !push_key(pack_ngram(c0, Some(span[i + 1]), Some(span[i + 2]))) {
                break 'outer;
            }
        }
    }

    tokens.join(" ")
}

/// 将查询词编码为分组后的 FTS5 token 列表（外层为各个空白分词段，内层为该段的 n-gram 约束）。
pub fn encode_query_groups(query: &str) -> Vec<Vec<String>> {
    let spans = normalize_spans(query);
    let mut groups = Vec::new();
    let mut total_tokens = 0usize;

    for span in spans {
        if total_tokens >= MAX_QUERY_TOKENS {
            break;
        }
        let n = span.len();
        if n == 0 {
            continue;
        }
        let mut group = Vec::new();
        let mut seen = HashSet::new();
        if n == 1 {
            let tok = encode_base36_token(pack_ngram(span[0], None, None));
            group.push(tok);
            total_tokens += 1;
        } else if n == 2 {
            let tok = encode_base36_token(pack_ngram(span[0], Some(span[1]), None));
            group.push(tok);
            total_tokens += 1;
        } else {
            let max_for_span = (MAX_QUERY_TOKENS - total_tokens).min(8);
            let total_trigrams = n - 2;
            let step = if total_trigrams <= max_for_span {
                1
            } else {
                total_trigrams.div_ceil(max_for_span)
            };
            let mut i = 0usize;
            while i < total_trigrams && group.len() < max_for_span {
                let key = pack_ngram(span[i], Some(span[i + 1]), Some(span[i + 2]));
                if seen.insert(key) {
                    group.push(encode_base36_token(key));
                    total_tokens += 1;
                }
                i += step;
            }
            // 确保尾端三元组也被纳入，提升精确后缀匹配
            let last_idx = total_trigrams - 1;
            if group.len() < max_for_span && total_tokens < MAX_QUERY_TOKENS {
                let key = pack_ngram(
                    span[last_idx],
                    Some(span[last_idx + 1]),
                    Some(span[last_idx + 2]),
                );
                if seen.insert(key) {
                    group.push(encode_base36_token(key));
                    total_tokens += 1;
                }
            }
        }
        if !group.is_empty() {
            groups.push(group);
        }
    }

    groups
}

fn build_fts_match_expr(groups: &[Vec<String>], join_groups_with_or: bool) -> Option<String> {
    if groups.is_empty() {
        return None;
    }
    let parts: Vec<String> = groups
        .iter()
        .filter(|g| !g.is_empty())
        .map(|g| {
            if g.len() == 1 {
                g[0].clone()
            } else {
                format!("({})", g.join(" AND "))
            }
        })
        .collect();
    if parts.is_empty() {
        None
    } else if join_groups_with_or {
        Some(parts.join(" OR "))
    } else {
        Some(parts.join(" AND "))
    }
}

/// 在当前事务内为单条 `SessionEntry` 写入（或幂等跳过）情景回溯索引。
pub fn index_entry_in_tx(
    tx: &rusqlite::Connection,
    session_id: &str,
    entry: &SessionEntry,
) -> PersistenceResult<()> {
    let Some(text) = extract_indexable_text(&entry.event) else {
        return Ok(());
    };
    let terms = encode_document_terms(&text);
    if terms.is_empty() {
        return Ok(());
    }
    let preview = truncate_chars(&text, MAX_PREVIEW_CHARS);
    let ts_i64 = super::schema::millis_to_i64(entry.ts);

    let mut insert_doc = tx
        .prepare_cached(
            "INSERT OR IGNORE INTO session_recall_docs (
                 session_id, seq, ts_ms, kind, preview
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
        )
        .map_err(super::schema::map_sqlite_err)?;

    let changed = insert_doc
        .execute(params![
            session_id,
            entry.seq as i64,
            ts_i64,
            entry.event.kind_name(),
            preview,
        ])
        .map_err(super::schema::map_sqlite_err)?;

    if changed > 0 {
        let doc_id = tx.last_insert_rowid();
        let mut insert_fts = tx
            .prepare_cached("INSERT INTO session_recall_fts (rowid, terms) VALUES (?1, ?2)")
            .map_err(super::schema::map_sqlite_err)?;
        insert_fts
            .execute(params![doc_id, terms])
            .map_err(super::schema::map_sqlite_err)?;
    }

    Ok(())
}

/// 供 `exec_fork` 在复制前缀事件后批量建立子会话的情景回溯索引。
pub fn index_session_events_in_tx(
    tx: &rusqlite::Connection,
    session_id: &str,
) -> PersistenceResult<()> {
    let rows: Vec<(i64, i64, String)> = {
        let mut stmt = tx
            .prepare(
                "SELECT seq, ts_ms, payload_json
                 FROM session_events
                 WHERE session_id = ?1
                 ORDER BY seq ASC",
            )
            .map_err(super::schema::map_sqlite_err)?;
        let iter = stmt
            .query_map([session_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .map_err(super::schema::map_sqlite_err)?;
        iter.collect::<Result<Vec<_>, _>>()
            .map_err(super::schema::map_sqlite_err)?
    };

    for (seq_i64, ts_i64, payload_json) in rows {
        let Ok(event) = serde_json::from_str::<SessionEvent>(&payload_json) else {
            continue;
        };
        let entry = SessionEntry {
            seq: u64::try_from(seq_i64.max(0)).unwrap_or(0),
            event,
            ts: super::schema::i64_to_millis(ts_i64),
        };
        index_entry_in_tx(tx, session_id, &entry)?;
    }

    Ok(())
}

/// 执行无内容 FTS5 n-gram 检索并构造有界摘要片段。
pub fn search_recall_from_conn(
    conn: &Connection,
    query: &RecallSearchQuery,
) -> PersistenceResult<Vec<RecallSearchHit>> {
    let trimmed = query.query.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let limit = query.limit.clamp(1, MAX_RECALL_SEARCH_HITS);
    let groups = encode_query_groups(trimmed);
    if groups.is_empty() {
        return Ok(Vec::new());
    }

    let strict_expr = build_fts_match_expr(&groups, false);
    let mut hits = if let Some(ref expr) = strict_expr {
        run_fts_query(conn, expr, query, trimmed, limit)?
    } else {
        Vec::new()
    };

    if hits.is_empty()
        && groups.len() > 1
        && let Some(or_expr) = build_fts_match_expr(&groups, true)
    {
        hits = run_fts_query(conn, &or_expr, query, trimmed, limit)?;
    }

    Ok(enforce_recall_byte_cap(hits, MAX_RECALL_SEARCH_TOTAL_BYTES))
}

#[derive(Clone)]
struct RecallScopeTarget {
    source_session_id: String,
    max_seq: Option<u64>,
    attributed_session_id: String,
    attributed_title: Option<String>,
}

fn resolve_session_scopes(
    conn: &Connection,
    session_ids: &[String],
) -> PersistenceResult<Vec<RecallScopeTarget>> {
    let requested: HashSet<&str> = session_ids.iter().map(String::as_str).collect();
    let mut scopes = Vec::with_capacity(session_ids.len());
    let mut stmt = conn
        .prepare_cached("SELECT title, parent_id, fork_seq FROM sessions WHERE id = ?1")
        .map_err(super::schema::map_sqlite_err)?;

    for sid in session_ids {
        scopes.push(RecallScopeTarget {
            source_session_id: sid.clone(),
            max_seq: None,
            attributed_session_id: sid.clone(),
            attributed_title: None,
        });

        let mut cur = sid.clone();
        let mut bound: Option<u64> = None;
        let mut root_title: Option<String> = None;
        let mut visited = HashSet::from([sid.clone()]);

        for depth in 0..64 {
            let row: Option<(Option<String>, Option<String>, Option<i64>)> = stmt
                .query_row([&cur], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .optional()
                .map_err(super::schema::map_sqlite_err)?;
            let Some((title, parent_id, fork_seq)) = row else {
                break;
            };
            if depth == 0 {
                root_title = title;
            }
            let (Some(pid), Some(fseq_raw)) = (parent_id, fork_seq) else {
                break;
            };
            let Ok(fseq) = u64::try_from(fseq_raw) else {
                break;
            };
            if visited.contains(&pid) || requested.contains(pid.as_str()) {
                break;
            }
            let next_bound = bound.map_or(fseq, |b| b.min(fseq));
            scopes.push(RecallScopeTarget {
                source_session_id: pid.clone(),
                max_seq: Some(next_bound),
                attributed_session_id: sid.clone(),
                attributed_title: root_title.clone(),
            });
            visited.insert(pid.clone());
            cur = pid;
            bound = Some(next_bound);
        }
    }

    Ok(scopes)
}

fn resolve_cwd_inherited_scopes(
    conn: &Connection,
    cwd: &str,
) -> PersistenceResult<Vec<RecallScopeTarget>> {
    let mut child_stmt = conn
        .prepare_cached(
            "SELECT id, title, parent_id, fork_seq
             FROM sessions
             WHERE cwd = ?1 AND parent_id IS NOT NULL AND fork_seq IS NOT NULL",
        )
        .map_err(super::schema::map_sqlite_err)?;
    let children: Vec<(String, Option<String>, String, i64)> = child_stmt
        .query_map([cwd], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .map_err(super::schema::map_sqlite_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(super::schema::map_sqlite_err)?;

    let mut parent_stmt = conn
        .prepare_cached("SELECT cwd, parent_id, fork_seq FROM sessions WHERE id = ?1")
        .map_err(super::schema::map_sqlite_err)?;
    let mut scopes = Vec::new();

    for (child_id, child_title, first_pid, first_fseq_raw) in children {
        let Ok(mut bound) = u64::try_from(first_fseq_raw) else {
            continue;
        };
        let mut cur = first_pid;
        let mut visited = HashSet::from([child_id.clone()]);

        for _ in 0..64 {
            if !visited.insert(cur.clone()) {
                break;
            }
            let row: Option<(Option<String>, Option<String>, Option<i64>)> = parent_stmt
                .query_row([&cur], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .optional()
                .map_err(super::schema::map_sqlite_err)?;
            let Some((anc_cwd, next_pid, next_fseq)) = row else {
                break;
            };
            if anc_cwd.as_deref() != Some(cwd) {
                scopes.push(RecallScopeTarget {
                    source_session_id: cur.clone(),
                    max_seq: Some(bound),
                    attributed_session_id: child_id.clone(),
                    attributed_title: child_title.clone(),
                });
            }
            let (Some(pid), Some(fseq_raw)) = (next_pid, next_fseq) else {
                break;
            };
            let Ok(fseq) = u64::try_from(fseq_raw) else {
                break;
            };
            bound = bound.min(fseq);
            cur = pid;
        }
    }

    Ok(scopes)
}

fn append_scope_filters(
    scopes: &[RecallScopeTarget],
    bind_values: &mut Vec<rusqlite::types::Value>,
) -> Vec<String> {
    let mut parts = Vec::with_capacity(scopes.len());
    for scope in scopes {
        let sid_idx = bind_values.len() + 1;
        bind_values.push(rusqlite::types::Value::Text(
            scope.source_session_id.clone(),
        ));
        if let Some(max_seq) = scope.max_seq {
            let seq_idx = bind_values.len() + 1;
            bind_values.push(rusqlite::types::Value::Integer(
                i64::try_from(max_seq).unwrap_or(i64::MAX),
            ));
            parts.push(format!(
                "(d.session_id = ?{sid_idx} AND d.seq <= ?{seq_idx})"
            ));
        } else {
            parts.push(format!("d.session_id = ?{sid_idx}"));
        }
    }
    parts
}

fn run_fts_query(
    conn: &Connection,
    match_expr: &str,
    query: &RecallSearchQuery,
    raw_query: &str,
    limit: usize,
) -> PersistenceResult<Vec<RecallSearchHit>> {
    let mut sql = String::from(
        "SELECT d.session_id, s.title, d.seq, d.ts_ms, d.kind, d.preview, s.cwd
         FROM session_recall_fts f
         JOIN session_recall_docs d ON d.doc_id = f.rowid
         JOIN sessions s ON s.id = d.session_id
         WHERE session_recall_fts MATCH ?1",
    );
    let mut bind_values: Vec<rusqlite::types::Value> =
        vec![rusqlite::types::Value::Text(match_expr.to_string())];

    let session_scopes = if !query.session_ids.is_empty() {
        resolve_session_scopes(conn, &query.session_ids)?
    } else {
        Vec::new()
    };
    let cwd_inherited_scopes = if query.session_ids.is_empty()
        && let Some(ref cwd) = query.cwd
    {
        resolve_cwd_inherited_scopes(conn, cwd)?
    } else {
        Vec::new()
    };

    if !session_scopes.is_empty() {
        let filters = append_scope_filters(&session_scopes, &mut bind_values);
        sql.push_str(&format!(" AND ({})", filters.join(" OR ")));
    } else if let Some(ref cwd) = query.cwd {
        let cwd_idx = bind_values.len() + 1;
        bind_values.push(rusqlite::types::Value::Text(cwd.clone()));
        if cwd_inherited_scopes.is_empty() {
            sql.push_str(&format!(" AND s.cwd = ?{cwd_idx}"));
        } else {
            let mut filters = vec![format!("s.cwd = ?{cwd_idx}")];
            filters.extend(append_scope_filters(
                &cwd_inherited_scopes,
                &mut bind_values,
            ));
            sql.push_str(&format!(" AND ({})", filters.join(" OR ")));
        }
    }

    let limit_idx = bind_values.len() + 1;
    // 多取候选并在 Rust 侧做轻量二次子串校验，消除 n-gram 碰撞误报
    let candidate_limit = (limit * 3).min(120) as i64;
    sql.push_str(&format!(
        " ORDER BY bm25(session_recall_fts) ASC, d.ts_ms DESC, d.seq DESC LIMIT ?{limit_idx}"
    ));
    bind_values.push(rusqlite::types::Value::Integer(candidate_limit));

    let mut stmt = conn.prepare(&sql).map_err(super::schema::map_sqlite_err)?;
    let rows = stmt
        .query_map(params_from_iter(bind_values.iter()), |row| {
            let session_id: String = row.get(0)?;
            let session_title: Option<String> = row.get(1)?;
            let seq: i64 = row.get(2)?;
            let ts_ms: i64 = row.get(3)?;
            let kind: String = row.get(4)?;
            let preview: String = row.get(5)?;
            let row_cwd: Option<String> = row.get(6)?;
            Ok((
                session_id,
                session_title,
                seq,
                ts_ms,
                kind,
                preview,
                row_cwd,
            ))
        })
        .map_err(super::schema::map_sqlite_err)?;

    let terms_lower: Vec<String> = raw_query
        .split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    let mut payload_stmt = conn
        .prepare_cached(
            "SELECT payload_json
             FROM session_events
             WHERE session_id = ?1 AND seq = ?2",
        )
        .map_err(super::schema::map_sqlite_err)?;

    let mut exact_hits = Vec::new();
    let mut fallback_hits = Vec::new();
    let mut seen_hits: HashSet<(String, u64)> = HashSet::new();

    for row in rows {
        let (session_id, session_title, seq, ts_ms, kind, preview, row_cwd) =
            row.map_err(super::schema::map_sqlite_err)?;
        let preview_lower = preview.to_lowercase();
        let mut all_match = terms_lower
            .iter()
            .all(|t| preview_lower.contains(t.as_str()));
        let mut any_match = terms_lower
            .iter()
            .any(|t| preview_lower.contains(t.as_str()));
        let mut snippet_source: Option<String> = None;

        // FTS5 索引覆盖前 MAX_INDEX_CHARS (16384) 字符，而 preview 仅缓存前 MAX_PREVIEW_CHARS (4000) 字符；
        // 若截断前缀未能完整命中，回查原始事件全文以防 4001..=16384 区间命中被静默丢弃。
        if !all_match
            && preview.chars().count() >= MAX_PREVIEW_CHARS
            && let Some(payload_json) = payload_stmt
                .query_row(params![&session_id, seq], |r| r.get::<_, String>(0))
                .ok()
            && let Ok(event) = serde_json::from_str::<SessionEvent>(&payload_json)
            && let Some(full_text) = extract_indexable_text(&event)
        {
            let full_lower = full_text.to_lowercase();
            all_match = terms_lower.iter().all(|t| full_lower.contains(t.as_str()));
            any_match = terms_lower.iter().any(|t| full_lower.contains(t.as_str()));
            if any_match || all_match {
                snippet_source = Some(full_text);
            }
        }

        if !all_match && !any_match {
            continue;
        }

        let seq_u64 = u64::try_from(seq.max(0)).unwrap_or(0);
        let snippet_text = snippet_source.as_deref().unwrap_or(preview.as_str());
        let snippet = build_snippet(snippet_text, &terms_lower, SNIPPET_MAX_CHARS);
        let ts = super::schema::i64_to_millis(ts_ms);

        let mut push_hit = |out_sid: String, out_title: Option<String>| {
            if !seen_hits.insert((out_sid.clone(), seq_u64)) {
                return;
            }
            let hit = RecallSearchHit {
                session_id: out_sid,
                session_title: out_title,
                seq: seq_u64,
                ts,
                kind: kind.clone(),
                snippet: snippet.clone(),
            };
            if all_match {
                exact_hits.push(hit);
            } else {
                fallback_hits.push(hit);
            }
        };

        if !session_scopes.is_empty() {
            for scope in &session_scopes {
                if scope.source_session_id == session_id
                    && scope.max_seq.is_none_or(|m| seq_u64 <= m)
                {
                    let out_title = if scope.max_seq.is_some() {
                        scope.attributed_title.clone()
                    } else {
                        session_title.clone()
                    };
                    push_hit(scope.attributed_session_id.clone(), out_title);
                }
            }
        } else {
            let direct_cwd_match = query
                .cwd
                .as_deref()
                .is_none_or(|expected| row_cwd.as_deref() == Some(expected));
            if direct_cwd_match {
                push_hit(session_id.clone(), session_title.clone());
            }
            for scope in &cwd_inherited_scopes {
                if scope.source_session_id == session_id
                    && scope.max_seq.is_none_or(|m| seq_u64 <= m)
                {
                    push_hit(
                        scope.attributed_session_id.clone(),
                        scope.attributed_title.clone(),
                    );
                }
            }
        }
    }

    exact_hits.extend(fallback_hits);
    exact_hits.truncate(limit);
    Ok(exact_hits)
}

fn hit_byte_len(hit: &RecallSearchHit) -> usize {
    hit.session_id.len()
        + hit.session_title.as_deref().map_or(0, str::len)
        + hit.kind.len()
        + hit.snippet.len()
}

fn truncate_utf8_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn enforce_recall_byte_cap(
    hits: Vec<RecallSearchHit>,
    max_total_bytes: usize,
) -> Vec<RecallSearchHit> {
    let mut bounded = Vec::with_capacity(hits.len());
    let mut total_bytes = 0usize;
    for mut hit in hits {
        let len = hit_byte_len(&hit);
        if total_bytes + len <= max_total_bytes {
            total_bytes += len;
            bounded.push(hit);
            continue;
        }
        if bounded.is_empty() {
            let fixed = hit.session_id.len()
                + hit.session_title.as_deref().map_or(0, str::len)
                + hit.kind.len();
            let avail = max_total_bytes.saturating_sub(fixed);
            hit.snippet = truncate_utf8_bytes(&hit.snippet, avail).to_string();
            bounded.push(hit);
        }
        break;
    }
    bounded
}

fn build_snippet(text: &str, terms_lower: &[String], max_chars: usize) -> String {
    let collapsed: String = text
        .chars()
        .map(|c| {
            if c == '\n' || c == '\r' || c == '\t' {
                ' '
            } else {
                c
            }
        })
        .collect();
    let chars: Vec<char> = collapsed.chars().collect();
    if chars.len() <= max_chars {
        return collapsed;
    }

    let lower = collapsed.to_lowercase();
    let match_char_idx = terms_lower
        .iter()
        .filter_map(|t| {
            lower
                .find(t.as_str())
                .map(|byte_pos| lower[..byte_pos].chars().count())
        })
        .min()
        .unwrap_or(0);

    let half = max_chars / 2;
    let start = match_char_idx.saturating_sub(half / 2);
    let end = (start + max_chars).min(chars.len());
    let start = end.saturating_sub(max_chars);

    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    let slice: String = chars[start..end].iter().collect();
    out.push_str(slice.trim());
    if end < chars.len() {
        out.push('…');
    }
    out
}
