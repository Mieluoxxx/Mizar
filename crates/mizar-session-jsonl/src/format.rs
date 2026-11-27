//! 宽容单会话 JSONL 文件格式、按 `cwd` 分桶的路径编解码与恢复扫描。

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};

use chrono::DateTime;
use mizar_session::{
    LoadedSession, PersistenceError, PersistenceResult, SESSION_FORMAT_VERSION, SessionEntry,
    SessionEvent, SessionHeader, SessionHeaderPatch, SessionListQuery, TitlePatch, TitleSource,
    TitleSourcePatch,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::blob::{BlobStore, sha256_hex};

const MAX_BUCKET_BYTES: usize = 120;
const GLOBAL_BUCKET: &str = "_global";

pub fn validate_session_id(session_id: &str) -> PersistenceResult<()> {
    if session_id.trim().is_empty()
        || session_id == "."
        || session_id.contains('/')
        || session_id.contains('\\')
        || session_id.contains("..")
        || session_id.contains('\0')
    {
        return Err(PersistenceError::InvalidArgument(format!(
            "invalid session id \"{session_id}\""
        )));
    }
    Ok(())
}

pub fn canonicalize_cwd(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    match std::fs::canonicalize(trimmed) {
        Ok(canon) => normalize_path_text(&canon.to_string_lossy()),
        Err(_) => normalize_lexical_path(trimmed),
    }
}

fn normalize_path_text(raw: &str) -> String {
    let stripped = raw.strip_prefix(r"\\?\").unwrap_or(raw);
    if stripped == "/" || (stripped.len() == 3 && stripped.ends_with(":\\")) {
        return stripped.to_string();
    }
    stripped.trim_end_matches(['/', '\\']).to_string()
}

fn normalize_lexical_path(raw: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut prefix_root = String::new();
    for comp in Path::new(raw).components() {
        match comp {
            Component::Prefix(p) => prefix_root.push_str(&p.as_os_str().to_string_lossy()),
            Component::RootDir => prefix_root.push('/'),
            Component::CurDir => {}
            Component::ParentDir => {
                if !parts.is_empty() && parts.last().map(String::as_str) != Some("..") {
                    parts.pop();
                } else if prefix_root.is_empty() {
                    parts.push("..".to_string());
                }
            }
            Component::Normal(seg) => parts.push(seg.to_string_lossy().into_owned()),
        }
    }
    if prefix_root.is_empty() {
        if parts.is_empty() {
            raw.to_string()
        } else {
            parts.join("/")
        }
    } else if parts.is_empty() {
        prefix_root
    } else {
        format!("{prefix_root}{}", parts.join("/"))
    }
}

/// 基于 canonicalize 后的 `cwd` 构造稳定且路径安全的目录桶名。
pub fn encode_cwd_bucket(cwd: Option<&str>) -> String {
    let Some(raw) = cwd.map(str::trim).filter(|s| !s.is_empty()) else {
        return GLOBAL_BUCKET.to_string();
    };
    let canon = canonicalize_cwd(raw);
    if canon.is_empty() {
        return GLOBAL_BUCKET.to_string();
    }

    let hash = sha256_hex(canon.as_bytes());
    let hash16 = &hash[..16];

    let mut sanitized = String::with_capacity(canon.len());
    for ch in canon.chars() {
        if matches!(ch, '/' | '\\' | ':' | '\0') || ch.is_ascii_control() {
            sanitized.push('-');
        } else {
            sanitized.push(ch);
        }
    }
    while sanitized.contains("..") {
        sanitized = sanitized.replace("..", "__");
    }
    if sanitized == GLOBAL_BUCKET || sanitized == "." || sanitized.is_empty() {
        return format!("_cwd-{hash16}");
    }

    if sanitized.len() > MAX_BUCKET_BYTES {
        let budget = MAX_BUCKET_BYTES - 1 - 16;
        let mut end = 0usize;
        for (idx, ch) in sanitized.char_indices() {
            let next = idx + ch.len_utf8();
            if next > budget {
                break;
            }
            end = next;
        }
        let prefix = sanitized[..end].trim_end_matches('-');
        let prefix = if prefix.is_empty() { "_cwd" } else { prefix };
        format!("{prefix}-{hash16}")
    } else {
        sanitized
    }
}

pub fn session_file_name(created_at_ms: u128, session_id: &str) -> String {
    let secs = i64::try_from(created_at_ms / 1_000).unwrap_or(0);
    let nanos = ((created_at_ms % 1_000) * 1_000_000) as u32;
    let dt = DateTime::from_timestamp(secs, nanos).unwrap_or(DateTime::UNIX_EPOCH);
    format!("{}_{session_id}.jsonl", dt.format("%Y%m%dT%H%M%SZ"))
}

pub fn session_file_path(
    sessions_dir: &Path,
    header: &SessionHeader,
) -> PersistenceResult<PathBuf> {
    validate_session_id(&header.id)?;
    let bucket = encode_cwd_bucket(header.cwd.as_deref());
    let file_name = session_file_name(header.created_at_ms, &header.id);
    Ok(sessions_dir.join(bucket).join(file_name))
}

fn is_utc_stamp(bytes: &[u8]) -> bool {
    bytes.len() == 16
        && bytes[..8].iter().all(u8::is_ascii_digit)
        && bytes[8] == b'T'
        && bytes[9..15].iter().all(u8::is_ascii_digit)
        && bytes[15] == b'Z'
}

fn matches_session_file(file_name: &str, target_id: &str) -> bool {
    let Some(stem) = file_name.strip_suffix(".jsonl") else {
        return false;
    };
    if stem.len() > 17 && stem.as_bytes()[16] == b'_' && is_utc_stamp(&stem.as_bytes()[..16]) {
        return &stem[17..] == target_id;
    }
    stem == target_id
}

fn deserialize_tolerant_format_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    let val = Option::<Value>::deserialize(deserializer)?;
    let version = val
        .as_ref()
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(SESSION_FORMAT_VERSION);
    Ok(version)
}

fn deserialize_double_option<'de, T, D>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

fn deserialize_tolerant_title_source<'de, D>(
    deserializer: D,
) -> Result<Option<TitleSource>, D::Error>
where
    D: Deserializer<'de>,
{
    let val = Option::<Value>::deserialize(deserializer)?;
    Ok(val
        .as_ref()
        .and_then(Value::as_str)
        .and_then(TitleSource::parse))
}

fn deserialize_tolerant_title_source_patch<'de, D>(
    deserializer: D,
) -> Result<Option<Option<TitleSource>>, D::Error>
where
    D: Deserializer<'de>,
{
    let val = Value::deserialize(deserializer)?;
    if val.is_null() {
        return Ok(Some(None));
    }
    match val.as_str().and_then(TitleSource::parse) {
        Some(ts) => Ok(Some(Some(ts))),
        None => Ok(None),
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionLineOut<'a> {
    record: &'static str,
    id: &'a str,
    format_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title_source: &'a Option<TitleSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_id: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fork_seq: &'a Option<u64>,
    created_at_ms: u128,
    updated_at_ms: u128,
    event_count: u64,
    surface_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    compacted_up_to_seq: &'a Option<u64>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
struct SessionLineIn {
    id: String,
    #[serde(deserialize_with = "deserialize_tolerant_format_version")]
    format_version: u32,
    cwd: Option<String>,
    title: Option<String>,
    #[serde(deserialize_with = "deserialize_tolerant_title_source")]
    title_source: Option<TitleSource>,
    parent_id: Option<String>,
    fork_seq: Option<u64>,
    created_at_ms: u128,
    updated_at_ms: u128,
    event_count: u64,
    surface_count: u64,
    compacted_up_to_seq: Option<u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EntryLineOut<'a> {
    record: &'static str,
    seq: u64,
    ts: u128,
    event: &'a SessionEvent,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HeaderPatchLineOut<'a> {
    record: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<Option<&'a str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title_source: Option<Option<TitleSource>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    updated_at_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compacted_up_to_seq: Option<u64>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
struct HeaderPatchLineIn {
    #[serde(deserialize_with = "deserialize_double_option")]
    title: Option<Option<String>>,
    #[serde(deserialize_with = "deserialize_tolerant_title_source_patch")]
    title_source: Option<Option<TitleSource>>,
    updated_at_ms: Option<u128>,
    compacted_up_to_seq: Option<u64>,
}

pub fn serialize_session_header_line(header: &SessionHeader) -> PersistenceResult<String> {
    let out = SessionLineOut {
        record: "session",
        id: &header.id,
        format_version: header.format_version,
        cwd: &header.cwd,
        title: &header.title,
        title_source: &header.title_source,
        parent_id: &header.parent_id,
        fork_seq: &header.fork_seq,
        created_at_ms: header.created_at_ms,
        updated_at_ms: header.updated_at_ms,
        // 首行写 0 计数，由读取端按实际 `entry` 行数累加还原，避免 append-only 模式重写首行
        event_count: 0,
        surface_count: 0,
        compacted_up_to_seq: &header.compacted_up_to_seq,
    };
    Ok(serde_json::to_string(&out)?)
}

pub fn serialize_entry_line(entry: &SessionEntry, blobs: &BlobStore) -> PersistenceResult<String> {
    let event = blobs.externalize_event(&entry.event)?;
    let out = EntryLineOut {
        record: "entry",
        seq: entry.seq,
        ts: entry.ts,
        event: event.as_ref(),
    };
    Ok(serde_json::to_string(&out)?)
}

pub fn serialize_header_patch_line(
    patch: &SessionHeaderPatch,
) -> PersistenceResult<Option<String>> {
    let title = match &patch.title {
        TitlePatch::Keep => None,
        TitlePatch::Set(v) => Some(Some(v.as_str())),
        TitlePatch::Clear => Some(None),
    };
    let title_source = match patch.title_source {
        TitleSourcePatch::Keep => None,
        TitleSourcePatch::Set(s) => Some(Some(s)),
        TitleSourcePatch::Clear => Some(None),
    };
    if title.is_none()
        && title_source.is_none()
        && patch.updated_at_ms.is_none()
        && patch.compacted_up_to_seq.is_none()
    {
        return Ok(None);
    }
    let out = HeaderPatchLineOut {
        record: "header-patch",
        title,
        title_source,
        updated_at_ms: patch.updated_at_ms,
        compacted_up_to_seq: patch.compacted_up_to_seq,
    };
    Ok(Some(serde_json::to_string(&out)?))
}

pub fn apply_header_patch(header: &mut SessionHeader, patch: &SessionHeaderPatch) {
    match &patch.title {
        TitlePatch::Keep => {}
        TitlePatch::Set(v) => header.title = Some(v.clone()),
        TitlePatch::Clear => header.title = None,
    }
    match patch.title_source {
        TitleSourcePatch::Keep => {}
        TitleSourcePatch::Set(s) => header.title_source = Some(s),
        TitleSourcePatch::Clear => header.title_source = None,
    }
    if let Some(ts) = patch.updated_at_ms {
        header.updated_at_ms = header.updated_at_ms.max(ts);
    }
    if let Some(seq) = patch.compacted_up_to_seq {
        header.compacted_up_to_seq =
            Some(header.compacted_up_to_seq.map_or(seq, |prev| prev.max(seq)));
    }
}

fn parse_u128_value(val: &Value) -> Option<u128> {
    if let Some(v) = val.as_u64() {
        return Some(u128::from(v));
    }
    serde_json::from_value::<u128>(val.clone()).ok()
}

pub fn read_session_seqs(path: &Path) -> PersistenceResult<HashSet<u64>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut raw_buf = Vec::new();
    let mut seqs = HashSet::new();
    loop {
        raw_buf.clear();
        if reader.read_until(b'\n', &mut raw_buf)? == 0 {
            break;
        }
        let Ok(line) = std::str::from_utf8(&raw_buf) else {
            continue;
        };
        let trimmed = line.trim().trim_start_matches('\u{FEFF}');
        if trimmed.is_empty() {
            continue;
        }
        let Ok(val) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        if val.get("record").and_then(Value::as_str) == Some("entry")
            && let Some(seq) = val.get("seq").and_then(Value::as_u64)
        {
            seqs.insert(seq);
        }
    }
    Ok(seqs)
}

pub fn read_session_file(
    path: &Path,
    blobs: Option<&BlobStore>,
    logger: Option<&cordis::Logger>,
    load_entries: bool,
) -> PersistenceResult<Option<LoadedSession>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let mut reader = BufReader::new(file);
    let mut raw_buf = Vec::new();

    let mut header: Option<SessionHeader> = None;
    let mut patches: Vec<HeaderPatchLineIn> = Vec::new();
    let mut entries = Vec::new();
    let mut seen_seqs = HashSet::new();
    let mut max_persisted_seq: Option<u64> = None;
    let mut skipped_events = 0usize;
    let mut counted_events = 0u64;
    let mut counted_surface = 0u64;
    let mut observed_max_ts = 0u128;
    let mut observed_compacted: Option<u64> = None;

    loop {
        raw_buf.clear();
        // 按字节读行，避免崩溃截断的半个 UTF-8 序列中断整文件恢复
        if reader.read_until(b'\n', &mut raw_buf)? == 0 {
            break;
        }
        let Ok(line) = std::str::from_utf8(&raw_buf) else {
            continue;
        };
        let trimmed = line.trim().trim_start_matches('\u{FEFF}');
        if trimmed.is_empty() {
            continue;
        }
        let Ok(val) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        let Some(record) = val.get("record").and_then(Value::as_str) else {
            continue;
        };

        match record {
            "session" => {
                if header.is_none()
                    && let Ok(rec) = serde_json::from_value::<SessionLineIn>(val)
                    && !rec.id.trim().is_empty()
                    && validate_session_id(&rec.id).is_ok()
                {
                    header = Some(SessionHeader {
                        id: rec.id,
                        format_version: rec.format_version,
                        cwd: rec.cwd,
                        title: rec.title,
                        title_source: rec.title_source,
                        parent_id: rec.parent_id,
                        fork_seq: rec.fork_seq,
                        created_at_ms: rec.created_at_ms,
                        updated_at_ms: rec.updated_at_ms,
                        event_count: rec.event_count,
                        surface_count: rec.surface_count,
                        compacted_up_to_seq: rec.compacted_up_to_seq,
                    });
                }
            }
            "entry" => {
                let Some(seq) = val.get("seq").and_then(Value::as_u64) else {
                    continue;
                };
                max_persisted_seq = Some(max_persisted_seq.map_or(seq, |m| m.max(seq)));
                let ts = val.get("ts").and_then(parse_u128_value).unwrap_or(0);
                observed_max_ts = observed_max_ts.max(ts);
                if !seen_seqs.insert(seq) {
                    continue;
                }
                let ev_val = match val {
                    Value::Object(mut map) => map.remove("event"),
                    _ => None,
                };
                match ev_val.and_then(|v| serde_json::from_value::<SessionEvent>(v).ok()) {
                    Some(mut event) => {
                        counted_events += 1;
                        if event.is_surface() {
                            counted_surface += 1;
                        }
                        if let SessionEvent::Compaction { first_kept_seq, .. } = &event {
                            observed_compacted = Some(
                                observed_compacted
                                    .map_or(*first_kept_seq, |p| p.max(*first_kept_seq)),
                            );
                        }
                        if load_entries {
                            if let Some(store) = blobs {
                                store.restore_event(&mut event, logger);
                            }
                            entries.push(SessionEntry { seq, event, ts });
                        }
                    }
                    None => {
                        if load_entries {
                            skipped_events += 1;
                        } else {
                            counted_events += 1;
                        }
                    }
                }
            }
            "header-patch" => {
                if let Ok(patch) = serde_json::from_value::<HeaderPatchLineIn>(val) {
                    patches.push(patch);
                }
            }
            _ => {}
        }
    }

    let Some(mut header) = header else {
        return Ok(None);
    };

    for patch in patches {
        if let Some(title) = patch.title {
            header.title = title;
        }
        if let Some(title_source) = patch.title_source {
            header.title_source = title_source;
        }
        if let Some(ts) = patch.updated_at_ms {
            observed_max_ts = observed_max_ts.max(ts);
        }
        if let Some(seq) = patch.compacted_up_to_seq {
            observed_compacted = Some(observed_compacted.map_or(seq, |p| p.max(seq)));
        }
    }

    header.updated_at_ms = header.updated_at_ms.max(observed_max_ts);
    header.event_count = header.event_count.max(counted_events);
    header.surface_count = header.surface_count.max(counted_surface);
    header.compacted_up_to_seq = match (header.compacted_up_to_seq, observed_compacted) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    if load_entries {
        entries.sort_by_key(|e| e.seq);
    }

    Ok(Some(LoadedSession {
        header,
        entries,
        max_persisted_seq,
        skipped_events,
    }))
}

pub fn find_session_file(
    sessions_dir: &Path,
    blobs_dir: &Path,
    session_id: &str,
) -> PersistenceResult<Option<PathBuf>> {
    validate_session_id(session_id)?;
    if !sessions_dir.exists() {
        return Ok(None);
    }
    let mut best: Option<(u128, PathBuf)> = None;
    for dir_entry in std::fs::read_dir(sessions_dir)? {
        let dir_entry = dir_entry?;
        let bucket_path = dir_entry.path();
        if bucket_path == blobs_dir || !bucket_path.is_dir() {
            continue;
        }
        for file_entry in std::fs::read_dir(&bucket_path)? {
            let file_entry = file_entry?;
            let file_path = file_entry.path();
            if !file_path.is_file() {
                continue;
            }
            let Some(file_name) = file_path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !matches_session_file(file_name, session_id) {
                continue;
            }
            if let Ok(Some(loaded)) = read_session_file(&file_path, None, None, false)
                && loaded.header.id == session_id
            {
                let ts = loaded.header.updated_at_ms;
                if best.as_ref().is_none_or(|(prev_ts, _)| ts >= *prev_ts) {
                    best = Some((ts, file_path));
                }
            }
        }
    }
    Ok(best.map(|(_, path)| path))
}

fn collect_headers_from_bucket(
    bucket_dir: &Path,
    by_id: &mut HashMap<String, SessionHeader>,
) -> PersistenceResult<()> {
    if !bucket_dir.is_dir() {
        return Ok(());
    }
    for file_entry in std::fs::read_dir(bucket_dir)? {
        let file_entry = file_entry?;
        let path = file_entry.path();
        if !path.is_file() {
            continue;
        }
        let is_jsonl = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "jsonl");
        if !is_jsonl {
            continue;
        }
        if let Ok(Some(loaded)) = read_session_file(&path, None, None, false) {
            let h = loaded.header;
            match by_id.get(&h.id) {
                Some(prev) if prev.updated_at_ms > h.updated_at_ms => {}
                _ => {
                    by_id.insert(h.id.clone(), h);
                }
            }
        }
    }
    Ok(())
}

pub fn list_sessions_in_dir(
    sessions_dir: &Path,
    blobs_dir: &Path,
    query: &SessionListQuery,
) -> PersistenceResult<Vec<SessionHeader>> {
    if !sessions_dir.exists() {
        return Ok(Vec::new());
    }
    let mut by_id = HashMap::new();
    if let Some(q_cwd) = &query.cwd {
        let trimmed = q_cwd.trim();
        let bucket_dir = if trimmed.is_empty() {
            sessions_dir.join(GLOBAL_BUCKET)
        } else {
            sessions_dir.join(encode_cwd_bucket(Some(trimmed)))
        };
        collect_headers_from_bucket(&bucket_dir, &mut by_id)?;
        let canon_q = canonicalize_cwd(trimmed);
        by_id.retain(|_, h| {
            if trimmed.is_empty() {
                h.cwd.as_deref().is_none_or(|c| c.trim().is_empty())
            } else {
                h.cwd.as_deref().map(canonicalize_cwd).as_deref() == Some(canon_q.as_str())
            }
        });
    } else {
        for dir_entry in std::fs::read_dir(sessions_dir)? {
            let dir_entry = dir_entry?;
            let bucket_path = dir_entry.path();
            if bucket_path == blobs_dir || !bucket_path.is_dir() {
                continue;
            }
            collect_headers_from_bucket(&bucket_path, &mut by_id)?;
        }
    }

    let mut headers: Vec<SessionHeader> = by_id.into_values().collect();
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
