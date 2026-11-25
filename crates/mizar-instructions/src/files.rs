use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use crate::config::{InstructionScopeKind, InstructionsConfig};
use crate::digest::trimmed_instruction_digest;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFile {
    pub canonical_path: PathBuf,
    pub display_path: String,
    pub scope_kind: InstructionScopeKind,
    pub content: String,
    pub digest: String,
    pub bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DiscoveryResult {
    pub project_root: PathBuf,
    pub global_files: Vec<DiscoveredFile>,
    pub project_files: Vec<DiscoveredFile>,
    pub warnings: Vec<String>,
}

pub fn resolve_mizar_home(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = explicit {
        return Some(dir.to_path_buf());
    }
    if let Ok(env) = std::env::var("MIZAR_HOME") {
        let trimmed = env.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    dirs::home_dir().map(|h| h.join(".mizar"))
}

pub fn resolve_agents_home(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = explicit {
        return Some(dir.to_path_buf());
    }
    if let Ok(env) = std::env::var("MIZAR_AGENTS_HOME") {
        let trimmed = env.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    dirs::home_dir().map(|h| h.join(".agents"))
}

pub fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(out.components().next_back(), Some(Component::Normal(_))) {
                    out.pop();
                } else if !matches!(
                    out.components().next_back(),
                    Some(Component::RootDir | Component::Prefix(_))
                ) {
                    out.push(Component::ParentDir);
                }
            }
            Component::Normal(s) => out.push(s),
        }
    }
    out
}

pub fn canonical_or_abs(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else if let Ok(cwd) = std::env::current_dir() {
        cwd.join(path)
    } else {
        path.to_path_buf()
    };
    let lexical = normalize_lexical(&abs);
    if let Ok(canon) = std::fs::canonicalize(&lexical) {
        return canon;
    }
    if let (Some(parent), Some(name)) = (lexical.parent(), lexical.file_name())
        && let Ok(canon_parent) = std::fs::canonicalize(parent)
    {
        return canon_parent.join(name);
    }
    lexical
}

pub fn sanitize_candidates(raw: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for item in raw {
        let trimmed = item.trim();
        if trimmed.is_empty()
            || trimmed == "."
            || trimmed == ".."
            || trimmed.contains('/')
            || trimmed.contains('\\')
        {
            continue;
        }
        if !out.iter().any(|existing| existing == trimmed) {
            out.push(trimmed.to_string());
        }
    }
    out
}

pub fn find_project_root(workspace: &Path, markers: &[String]) -> PathBuf {
    let effective_ws = canonical_or_abs(workspace);
    let clean_markers = sanitize_candidates(markers);
    if clean_markers.is_empty() {
        return effective_ws;
    }
    let mut cur = effective_ws.clone();
    loop {
        if clean_markers.iter().any(|m| cur.join(m).exists()) {
            return cur;
        }
        match cur.parent() {
            Some(parent) if parent != cur => cur = parent.to_path_buf(),
            _ => return effective_ws,
        }
    }
}

pub fn ancestor_chain(project_root: &Path, workspace: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut cur = workspace.to_path_buf();
    loop {
        dirs.push(cur.clone());
        if cur == project_root {
            break;
        }
        match cur.parent() {
            Some(parent) if parent != cur => cur = parent.to_path_buf(),
            _ => break,
        }
    }
    dirs.reverse();
    dirs
}

pub fn relative_display(from_dir: &Path, target: &Path) -> String {
    let from_comps: Vec<Component<'_>> = from_dir.components().collect();
    let target_comps: Vec<Component<'_>> = target.components().collect();
    let mut common = 0usize;
    while common < from_comps.len()
        && common < target_comps.len()
        && from_comps[common] == target_comps[common]
    {
        common += 1;
    }
    if common == 0 {
        return target.to_string_lossy().replace('\\', "/");
    }
    let mut parts: Vec<String> = Vec::new();
    for comp in &from_comps[common..] {
        if matches!(comp, Component::Normal(_)) {
            parts.push("..".to_string());
        }
    }
    for comp in &target_comps[common..] {
        match comp {
            Component::Normal(s) => parts.push(s.to_string_lossy().into_owned()),
            Component::CurDir => {}
            _ => return target.to_string_lossy().replace('\\', "/"),
        }
    }
    if parts.is_empty() {
        ".".to_string()
    } else {
        parts.join("/")
    }
}

fn format_global_display(dir: &Path, default_suffix: &str, filename: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let canon_home = canonical_or_abs(&home);
        let canon_dir = canonical_or_abs(dir);
        let default_dir = canon_home.join(default_suffix);
        if canon_dir == default_dir {
            return format!("~/{default_suffix}/{filename}");
        }
    }
    let dir_disp = dir.to_string_lossy().replace('\\', "/");
    format!("{dir_disp}/{filename}")
}

enum ReadCandidate {
    MissingOrEmpty,
    SkipWarn(String),
    Loaded {
        canonical_path: PathBuf,
        content: String,
        digest: String,
        bytes: usize,
    },
}

fn read_candidate_file(path: &Path, max_source_bytes: usize) -> ReadCandidate {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return ReadCandidate::MissingOrEmpty;
        }
        Err(err) => {
            return ReadCandidate::SkipWarn(format!(
                "读取指令文件元数据失败 ({}): {err}",
                path.display()
            ));
        }
    };
    if !meta.is_file() {
        return ReadCandidate::MissingOrEmpty;
    }
    let len = meta.len();
    if len > max_source_bytes as u64 {
        return ReadCandidate::SkipWarn(format!(
            "跳过超大指令文件 ({}，{} 字节 > 上限 {} 字节)",
            path.display(),
            len,
            max_source_bytes
        ));
    }
    let raw_bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(err) => {
            return ReadCandidate::SkipWarn(format!(
                "读取指令文件失败 ({}): {err}",
                path.display()
            ));
        }
    };
    if raw_bytes.len() > max_source_bytes {
        return ReadCandidate::SkipWarn(format!(
            "跳过超大指令文件 ({}，{} 字节 > 上限 {} 字节)",
            path.display(),
            raw_bytes.len(),
            max_source_bytes
        ));
    }
    let text = match String::from_utf8(raw_bytes) {
        Ok(s) => s,
        Err(_) => {
            return ReadCandidate::SkipWarn(format!("跳过非 UTF-8 指令文件 ({})", path.display()));
        }
    };
    let clean = text.strip_prefix('\u{FEFF}').unwrap_or(&text);
    if clean.trim().is_empty() {
        return ReadCandidate::MissingOrEmpty;
    }
    let canonical_path = std::fs::canonicalize(path).unwrap_or_else(|_| normalize_lexical(path));
    let digest = trimmed_instruction_digest(clean);
    let bytes = clean.len();
    ReadCandidate::Loaded {
        canonical_path,
        content: clean.to_string(),
        digest,
        bytes,
    }
}

struct GroupProbe<'a> {
    dir: &'a Path,
    scope_kind: InstructionScopeKind,
    filenames: &'a [String],
    default_global_suffix: Option<&'a str>,
}

fn probe_group(
    probe: GroupProbe<'_>,
    workspace: &Path,
    max_source_bytes: usize,
    seen_canonical: &mut HashSet<PathBuf>,
    digest_dedup: &mut HashSet<String>,
    out: &mut Vec<DiscoveredFile>,
    warnings: &mut Vec<String>,
) {
    for filename in probe.filenames {
        let path = probe.dir.join(filename);
        match read_candidate_file(&path, max_source_bytes) {
            ReadCandidate::MissingOrEmpty => {}
            ReadCandidate::SkipWarn(msg) => warnings.push(msg),
            ReadCandidate::Loaded {
                canonical_path,
                content,
                digest,
                bytes,
            } => {
                if !seen_canonical.insert(canonical_path.clone()) {
                    continue;
                }
                if !digest_dedup.insert(digest.clone()) {
                    continue;
                }
                let display_path = if let Some(suffix) = probe.default_global_suffix {
                    format_global_display(probe.dir, suffix, filename)
                } else {
                    let rel_dir = relative_display(workspace, probe.dir);
                    if rel_dir == "." {
                        filename.clone()
                    } else {
                        format!("{rel_dir}/{filename}")
                    }
                };
                out.push(DiscoveredFile {
                    canonical_path,
                    display_path,
                    scope_kind: probe.scope_kind.clone(),
                    content,
                    digest,
                    bytes,
                });
            }
        }
    }
}

pub fn discover_instructions(config: &InstructionsConfig) -> DiscoveryResult {
    let effective_ws = canonical_or_abs(&config.workspace);
    let project_root = find_project_root(&effective_ws, &config.project_root_markers);
    let base_candidates = sanitize_candidates(&config.candidates);
    let local_candidates = sanitize_candidates(&config.local_candidates);

    let mut seen_canonical: HashSet<PathBuf> = HashSet::new();
    let mut global_files = Vec::new();
    let mut project_files = Vec::new();
    let mut warnings = Vec::new();

    if config.include_default_global {
        let mut global_digests = HashSet::new();
        let mizar_candidates = if base_candidates.is_empty() {
            vec!["AGENTS.md".to_string()]
        } else {
            base_candidates.clone()
        };
        if let Some(mizar_home) = resolve_mizar_home(config.mizar_home.as_deref()) {
            probe_group(
                GroupProbe {
                    dir: &mizar_home,
                    scope_kind: InstructionScopeKind::UserGlobal,
                    filenames: &mizar_candidates,
                    default_global_suffix: Some(".mizar"),
                },
                &effective_ws,
                config.max_source_bytes,
                &mut seen_canonical,
                &mut global_digests,
                &mut global_files,
                &mut warnings,
            );
        }
        if let Some(agents_home) = resolve_agents_home(config.agents_home.as_deref()) {
            let agents_candidates = vec!["AGENTS.md".to_string()];
            probe_group(
                GroupProbe {
                    dir: &agents_home,
                    scope_kind: InstructionScopeKind::UserGlobal,
                    filenames: &agents_candidates,
                    default_global_suffix: Some(".agents"),
                },
                &effective_ws,
                config.max_source_bytes,
                &mut seen_canonical,
                &mut global_digests,
                &mut global_files,
                &mut warnings,
            );
        }
    }

    for dir in ancestor_chain(&project_root, &effective_ws) {
        let kind = if dir == effective_ws {
            InstructionScopeKind::Workspace
        } else {
            InstructionScopeKind::ProjectAncestor
        };
        if !base_candidates.is_empty() {
            let mut group_digests = HashSet::new();
            probe_group(
                GroupProbe {
                    dir: &dir,
                    scope_kind: kind.clone(),
                    filenames: &base_candidates,
                    default_global_suffix: None,
                },
                &effective_ws,
                config.max_source_bytes,
                &mut seen_canonical,
                &mut group_digests,
                &mut project_files,
                &mut warnings,
            );
        }
        if !local_candidates.is_empty() {
            let mut group_digests = HashSet::new();
            probe_group(
                GroupProbe {
                    dir: &dir,
                    scope_kind: kind,
                    filenames: &local_candidates,
                    default_global_suffix: None,
                },
                &effective_ws,
                config.max_source_bytes,
                &mut seen_canonical,
                &mut group_digests,
                &mut project_files,
                &mut warnings,
            );
        }
    }

    DiscoveryResult {
        project_root,
        global_files,
        project_files,
        warnings,
    }
}
