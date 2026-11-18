use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::config::{InstructionScopeKind, InstructionsConfig};
use crate::digest::trimmed_instruction_digest;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScopeRoot {
    DshGlobal,
    AgentsGlobal,
    Project(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScopeKey {
    pub root: ScopeRoot,
    pub filename: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsFingerprint {
    pub len: u64,
    pub modified_ns: Option<u128>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainKind {
    UserGlobal,
    Project,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFile {
    pub scope: ScopeKey,
    pub canonical_path: PathBuf,
    pub display_path: String,
    pub scope_dir_display: String,
    pub chain: ChainKind,
    pub scope_kind: InstructionScopeKind,
    pub content: String,
    pub digest: String,
    pub fingerprint: FsFingerprint,
    pub bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DiscoveryResult {
    pub project_root: PathBuf,
    pub global_files: Vec<DiscoveredFile>,
    pub project_files: Vec<DiscoveredFile>,
    pub warnings: Vec<String>,
}

pub fn resolve_dsh_home(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = explicit {
        return Some(dir.to_path_buf());
    }
    if let Ok(env) = std::env::var("DSH_RS_HOME") {
        let trimmed = env.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    dirs::home_dir().map(|h| h.join(".dsh-rs"))
}

pub fn resolve_agents_home(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = explicit {
        return Some(dir.to_path_buf());
    }
    if let Ok(env) = std::env::var("DSH_AGENTS_HOME") {
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

fn format_global_display(dir: &Path, default_suffix: &str, filename: &str) -> (String, String) {
    if let Some(home) = dirs::home_dir() {
        let canon_home = canonical_or_abs(&home);
        let canon_dir = canonical_or_abs(dir);
        let default_dir = canon_home.join(default_suffix);
        if canon_dir == default_dir {
            let dir_disp = format!("~/{default_suffix}");
            return (format!("{dir_disp}/{filename}"), dir_disp);
        }
    }
    let dir_disp = dir.to_string_lossy().replace('\\', "/");
    (format!("{dir_disp}/{filename}"), dir_disp)
}

pub fn read_fingerprint(path: &Path) -> std::io::Result<Option<FsFingerprint>> {
    match std::fs::metadata(path) {
        Ok(meta) => {
            if !meta.is_file() {
                return Ok(None);
            }
            let modified_ns = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos());
            Ok(Some(FsFingerprint {
                len: meta.len(),
                modified_ns,
            }))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err),
    }
}

enum ReadCandidate {
    MissingOrEmpty,
    SkipWarn(String),
    Loaded {
        canonical_path: PathBuf,
        content: String,
        digest: String,
        fingerprint: FsFingerprint,
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
    let modified_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos());
    let canonical_path = std::fs::canonicalize(path).unwrap_or_else(|_| normalize_lexical(path));
    let digest = trimmed_instruction_digest(clean);
    let bytes = clean.len();
    ReadCandidate::Loaded {
        canonical_path,
        content: clean.to_string(),
        digest,
        fingerprint: FsFingerprint { len, modified_ns },
        bytes,
    }
}

struct GroupProbe<'a> {
    dir: &'a Path,
    scope_root: ScopeRoot,
    chain: ChainKind,
    scope_kind: InstructionScopeKind,
    filenames: &'a [String],
    default_global_suffix: Option<&'a str>,
}

fn probe_group(
    probe: GroupProbe<'_>,
    workspace: &Path,
    max_source_bytes: usize,
    seen_canonical: &mut HashSet<PathBuf>,
    out: &mut Vec<DiscoveredFile>,
    warnings: &mut Vec<String>,
) {
    let mut group_digests: HashSet<String> = HashSet::new();
    for filename in probe.filenames {
        let path = probe.dir.join(filename);
        match read_candidate_file(&path, max_source_bytes) {
            ReadCandidate::MissingOrEmpty => {}
            ReadCandidate::SkipWarn(msg) => warnings.push(msg),
            ReadCandidate::Loaded {
                canonical_path,
                content,
                digest,
                fingerprint,
                bytes,
            } => {
                if !seen_canonical.insert(canonical_path.clone()) {
                    continue;
                }
                if !group_digests.insert(digest.clone()) {
                    continue;
                }
                let (display_path, scope_dir_display) =
                    if let Some(suffix) = probe.default_global_suffix {
                        format_global_display(probe.dir, suffix, filename)
                    } else {
                        let rel_dir = relative_display(workspace, probe.dir);
                        let file_disp = if rel_dir == "." {
                            filename.clone()
                        } else {
                            format!("{rel_dir}/{filename}")
                        };
                        (file_disp, rel_dir)
                    };
                out.push(DiscoveredFile {
                    scope: ScopeKey {
                        root: probe.scope_root.clone(),
                        filename: filename.clone(),
                    },
                    canonical_path,
                    display_path,
                    scope_dir_display,
                    chain: probe.chain,
                    scope_kind: probe.scope_kind.clone(),
                    content,
                    digest,
                    fingerprint,
                    bytes,
                });
            }
        }
    }
}

pub fn discover_instructions(
    config: &InstructionsConfig,
    extra_subdirs: &[PathBuf],
) -> DiscoveryResult {
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
        let dsh_candidates = if base_candidates.is_empty() {
            vec!["AGENTS.md".to_string()]
        } else {
            base_candidates.clone()
        };
        if let Some(dsh_home) = resolve_dsh_home(config.dsh_home.as_deref()) {
            let mut tmp = Vec::new();
            probe_group(
                GroupProbe {
                    dir: &dsh_home,
                    scope_root: ScopeRoot::DshGlobal,
                    chain: ChainKind::UserGlobal,
                    scope_kind: InstructionScopeKind::UserGlobal,
                    filenames: &dsh_candidates,
                    default_global_suffix: Some(".dsh-rs"),
                },
                &effective_ws,
                config.max_source_bytes,
                &mut seen_canonical,
                &mut tmp,
                &mut warnings,
            );
            for file in tmp {
                if global_digests.insert(file.digest.clone()) {
                    global_files.push(file);
                }
            }
        }
        if let Some(agents_home) = resolve_agents_home(config.agents_home.as_deref()) {
            let agents_candidates = vec!["AGENTS.md".to_string()];
            let mut tmp = Vec::new();
            probe_group(
                GroupProbe {
                    dir: &agents_home,
                    scope_root: ScopeRoot::AgentsGlobal,
                    chain: ChainKind::UserGlobal,
                    scope_kind: InstructionScopeKind::UserGlobal,
                    filenames: &agents_candidates,
                    default_global_suffix: Some(".agents"),
                },
                &effective_ws,
                config.max_source_bytes,
                &mut seen_canonical,
                &mut tmp,
                &mut warnings,
            );
            for file in tmp {
                if global_digests.insert(file.digest.clone()) {
                    global_files.push(file);
                }
            }
        }
    }

    let mut project_dirs: Vec<(PathBuf, InstructionScopeKind)> =
        ancestor_chain(&project_root, &effective_ws)
            .into_iter()
            .map(|dir| {
                let kind = if dir == effective_ws {
                    InstructionScopeKind::Workspace
                } else {
                    InstructionScopeKind::ProjectAncestor
                };
                (dir, kind)
            })
            .collect();

    let mut seen_dirs: HashSet<PathBuf> = project_dirs.iter().map(|(d, _)| d.clone()).collect();
    for sub in extra_subdirs {
        let canon_sub = canonical_or_abs(sub);
        if canon_sub.starts_with(&effective_ws)
            && canon_sub != effective_ws
            && seen_dirs.insert(canon_sub.clone())
        {
            project_dirs.push((canon_sub, InstructionScopeKind::Subdir));
        }
    }

    for (dir, kind) in project_dirs {
        let scope_root = ScopeRoot::Project(dir.clone());
        if !base_candidates.is_empty() {
            probe_group(
                GroupProbe {
                    dir: &dir,
                    scope_root: scope_root.clone(),
                    chain: ChainKind::Project,
                    scope_kind: kind.clone(),
                    filenames: &base_candidates,
                    default_global_suffix: None,
                },
                &effective_ws,
                config.max_source_bytes,
                &mut seen_canonical,
                &mut project_files,
                &mut warnings,
            );
        }
        if !local_candidates.is_empty() {
            probe_group(
                GroupProbe {
                    dir: &dir,
                    scope_root,
                    chain: ChainKind::Project,
                    scope_kind: kind,
                    filenames: &local_candidates,
                    default_global_suffix: None,
                },
                &effective_ws,
                config.max_source_bytes,
                &mut seen_canonical,
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

/// 抽取一次文件工具访问所穿越的 `workspace` 内部子目录（按浅到深顺序）。
pub fn descendant_dirs_between(workspace: &Path, raw_touched_path: &str) -> Vec<PathBuf> {
    let trimmed = raw_touched_path.trim();
    let without_at = trimmed.strip_prefix('@').unwrap_or(trimmed);
    if without_at.is_empty() {
        return Vec::new();
    }
    let expanded = if without_at == "~" {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"))
    } else if let Some(rest) = without_at.strip_prefix("~/") {
        dirs::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| PathBuf::from(without_at))
    } else {
        PathBuf::from(without_at)
    };

    let effective_ws = canonical_or_abs(workspace);
    let candidate = if expanded.is_absolute() {
        expanded
    } else {
        effective_ws.join(expanded)
    };
    let target = canonical_or_abs(&candidate);
    let Some(parent) = target.parent() else {
        return Vec::new();
    };
    let Ok(rel) = parent.strip_prefix(&effective_ws) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut cur = effective_ws;
    for comp in rel.components() {
        if let Component::Normal(seg) = comp {
            let s = seg.to_string_lossy();
            if s == ".git" || s == ".dsh-rs" {
                return Vec::new();
            }
            cur = cur.join(seg);
            out.push(cur.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "dsh_instr_files_{name}_{}_{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn loads_both_agents_and_claude_when_different_and_dedups_when_identical() {
        let root = temp_dir("both_and_dedup");
        let _ = std::fs::remove_dir_all(&root);
        let sub = root.join("packages/app");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(&sub).unwrap();

        // 根目录：AGENTS.md 与 CLAUDE.md 内容相同 -> 仅保留 AGENTS.md
        std::fs::write(root.join("AGENTS.md"), "root shared rule\n").unwrap();
        std::fs::write(root.join("CLAUDE.md"), "  root shared rule  ").unwrap();
        // 子项目目录：AGENTS.md 与 CLAUDE.md 内容不同 -> 两者均加载且 AGENTS.md 在前
        std::fs::write(sub.join("AGENTS.md"), "app agents rule").unwrap();
        std::fs::write(sub.join("CLAUDE.md"), "app claude rule").unwrap();
        std::fs::write(sub.join("CLAUDE.local.md"), "local override").unwrap();

        let mut cfg = InstructionsConfig::new(&sub);
        cfg.include_default_global = false;
        let res = discover_instructions(&cfg, &[]);

        let paths: Vec<&str> = res
            .project_files
            .iter()
            .map(|f| f.display_path.as_str())
            .collect();
        assert_eq!(
            paths,
            vec![
                "../../AGENTS.md",
                "AGENTS.md",
                "CLAUDE.md",
                "CLAUDE.local.md"
            ]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn collects_descendant_dirs_only_inside_workspace() {
        let root = temp_dir("descendants");
        let _ = std::fs::remove_dir_all(&root);
        let ws = root.join("repo");
        std::fs::create_dir_all(ws.join("src/nested")).unwrap();
        std::fs::create_dir_all(root.join("outside")).unwrap();

        let dirs = descendant_dirs_between(&ws, "src/nested/mod.rs");
        let canon_ws = canonical_or_abs(&ws);
        assert_eq!(
            dirs,
            vec![canon_ws.join("src"), canon_ws.join("src/nested")]
        );

        assert!(descendant_dirs_between(&ws, "lib.rs").is_empty());
        assert!(descendant_dirs_between(&ws, "../outside/secret.txt").is_empty());
        assert!(descendant_dirs_between(&ws, ".git/config").is_empty());

        let _ = std::fs::remove_dir_all(&root);
    }
}
