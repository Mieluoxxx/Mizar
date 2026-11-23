use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, UNIX_EPOCH};

use crate::skills::frontmatter::parse_skill_markdown;
use crate::skills::registry::{BoxFuture, SkillProvider};
use crate::skills::types::{
    BUNDLED_SKILL_RANK, SkillCandidate, SkillDefinition, SkillLookupOptions,
    SkillProviderObservation, SkillResourceBase, SkillSource, SkillSummary,
};

pub const PROJECT_AGENTS_RANK: u32 = 200;
pub const CUSTOM_RANK: u32 = 300;
pub const USER_AGENTS_RANK: u32 = 500;
const MAX_SKILL_SCAN_DEPTH: usize = 8;
const FINGERPRINT_SCAN_TTL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone)]
pub struct FilesystemSkillConfig {
    pub provider_name: String,
    pub include_default_roots: bool,
    pub include_user_roots: bool,
    pub agents_home: Option<PathBuf>,
    pub custom_skill_dirs: Vec<PathBuf>,
    pub bundled_skill_dir: Option<PathBuf>,
}

impl Default for FilesystemSkillConfig {
    fn default() -> Self {
        Self {
            provider_name: "filesystem".to_string(),
            include_default_roots: true,
            include_user_roots: true,
            agents_home: None,
            custom_skill_dirs: Vec::new(),
            bundled_skill_dir: None,
        }
    }
}

#[derive(Debug, Clone)]
struct SkillRoot {
    path: PathBuf,
    source: SkillSource,
    rank: u32,
}

#[derive(Debug, Clone)]
struct LocalLocator {
    path: PathBuf,
    directory: PathBuf,
    source: SkillSource,
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

/// 向上查找最近的 `.git` 祖先目录作为项目根；找不到时回退到 `cwd` 本身。
pub fn find_project_root(cwd: &Path) -> PathBuf {
    let mut cur = cwd.to_path_buf();
    loop {
        if cur.join(".git").exists() {
            return cur;
        }
        match cur.parent() {
            Some(parent) if parent != cur => cur = parent.to_path_buf(),
            _ => return cwd.to_path_buf(),
        }
    }
}

fn should_skip_entry_name(name: &str) -> bool {
    name.starts_with('.') || name == "node_modules"
}

/// 判断目标路径是否可能命中任一潜在技能路径（根下 `<name>.md` 或任意非隐藏子目录 `<dir>/.../SKILL.md`）。
pub fn is_potential_skill_path(root: &Path, target: &Path) -> bool {
    let Ok(rel) = target.strip_prefix(root) else {
        return false;
    };
    let comps: Vec<String> = rel
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    if comps.is_empty() {
        return false;
    }
    if comps.iter().any(|c| should_skip_entry_name(c)) {
        return false;
    }
    if comps.len() == 1 {
        comps[0].ends_with(".md") && comps[0] != "SKILL.md"
    } else {
        comps.last().is_some_and(|last| last == "SKILL.md")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    mtime_secs: u64,
    mtime_nanos: u32,
}

#[derive(Debug, Clone)]
struct CachedFingerprintEntry {
    scanned_at: Instant,
    root_stamps: Vec<(PathBuf, Option<FileStamp>)>,
    file_stamps: Vec<(PathBuf, Option<FileStamp>)>,
    fingerprint: u64,
}

pub struct FilesystemSkillProvider {
    config: FilesystemSkillConfig,
    default_cwd: Option<PathBuf>,
    fp_cache: Arc<Mutex<HashMap<Option<PathBuf>, CachedFingerprintEntry>>>,
}

impl FilesystemSkillProvider {
    pub fn new(config: FilesystemSkillConfig, default_cwd: Option<PathBuf>) -> Self {
        Self {
            config,
            default_cwd,
            fp_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn resolve_roots(&self, cwd: Option<&Path>) -> Vec<SkillRoot> {
        resolve_roots_for(&self.config, self.default_cwd.as_deref(), cwd)
    }

    pub fn matches_mutation_path(&self, target: &Path, cwd: Option<&Path>) -> bool {
        let abs_target = std::fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
        let matched = self.resolve_roots(cwd).into_iter().any(|root| {
            let root_abs = std::fs::canonicalize(&root.path).unwrap_or(root.path);
            is_potential_skill_path(&root_abs, &abs_target)
        });
        if matched {
            self.fp_cache.lock().unwrap().clear();
        }
        matched
    }
}

fn resolve_roots_for(
    config: &FilesystemSkillConfig,
    default_cwd: Option<&Path>,
    cwd: Option<&Path>,
) -> Vec<SkillRoot> {
    let mut roots = Vec::new();
    let mut seen_paths = HashSet::new();

    let excluded_user_key = config
        .include_default_roots
        .then(|| resolve_agents_home(config.agents_home.as_deref()).map(|h| h.join("skills")))
        .flatten()
        .map(|p| std::fs::canonicalize(&p).unwrap_or(p));
    let user_agents_dir = (config.include_default_roots && config.include_user_roots)
        .then(|| resolve_agents_home(config.agents_home.as_deref()).map(|h| h.join("skills")))
        .flatten();
    let user_agents_key = excluded_user_key;

    let effective_cwd = cwd.or(default_cwd);
    if config.include_default_roots
        && let Some(dir) = effective_cwd
    {
        let abs_cwd = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        let project_root = find_project_root(&abs_cwd);
        let mut candidates = Vec::new();
        if abs_cwd != project_root {
            candidates.push(abs_cwd.join(".agents").join("skills"));
        }
        candidates.push(project_root.join(".agents").join("skills"));
        for path in candidates {
            let key = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            if user_agents_key.as_ref() == Some(&key) {
                continue;
            }
            if seen_paths.insert(key) {
                roots.push(SkillRoot {
                    path,
                    source: SkillSource::ProjectAgents,
                    rank: PROJECT_AGENTS_RANK,
                });
            }
        }
    }

    for dir in &config.custom_skill_dirs {
        let key = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.clone());
        if seen_paths.insert(key) {
            roots.push(SkillRoot {
                path: dir.clone(),
                source: SkillSource::Custom,
                rank: CUSTOM_RANK,
            });
        }
    }

    if let Some(path) = user_agents_dir {
        let key = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if seen_paths.insert(key) {
            roots.push(SkillRoot {
                path,
                source: SkillSource::UserAgents,
                rank: USER_AGENTS_RANK,
            });
        }
    }

    let bundled = config.bundled_skill_dir.clone().or_else(|| {
        config
            .include_default_roots
            .then(|| std::env::var("MIZAR_BUNDLED_SKILL_DIR").ok())
            .flatten()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
    });
    if let Some(bundled_dir) = bundled {
        let key = std::fs::canonicalize(&bundled_dir).unwrap_or_else(|_| bundled_dir.clone());
        if seen_paths.insert(key) {
            roots.push(SkillRoot {
                path: bundled_dir,
                source: SkillSource::Bundled,
                rank: BUNDLED_SKILL_RANK,
            });
        }
    }

    roots
}

fn read_file_stamp(path: &Path) -> Option<FileStamp> {
    let meta = std::fs::metadata(path).ok()?;
    let (mtime_secs, mtime_nanos) = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or((0, 0), |d| (d.as_secs(), d.subsec_nanos()));
    Some(FileStamp {
        len: meta.len(),
        mtime_secs,
        mtime_nanos,
    })
}

fn hash_stamp(stamp: &FileStamp, hasher: &mut DefaultHasher) {
    stamp.len.hash(hasher);
    stamp.mtime_secs.hash(hasher);
    stamp.mtime_nanos.hash(hasher);
}

/// 扫描单个技能根目录，返回排序后的 `(skill_md_path, resource_base_dir)` 列表。
/// 若某子目录直接包含 `SKILL.md`，加载后停止继续下钻；否则递归扫描下级目录以兼容技能包结构。
fn scan_root_skill_files(root_path: &Path) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    if !root_path.exists() {
        return Ok(Vec::new());
    }

    let mut visited_dirs = HashSet::new();
    let mut stack = vec![(root_path.to_path_buf(), true, 0usize)];
    let mut discovered = Vec::new();

    while let Some((dir, is_root, depth)) = stack.pop() {
        let canonical_dir = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        if !visited_dirs.insert(canonical_dir) {
            continue;
        }

        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(err) if is_root => return Err(format!("read_dir {}: {err}", dir.display())),
            Err(_) => continue,
        };

        let mut entries: Vec<(String, PathBuf)> = rd
            .flatten()
            .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut next_subdirs = Vec::new();
        for (name, entry_path) in entries {
            if should_skip_entry_name(&name) {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&entry_path) else {
                continue;
            };
            if meta.is_dir() {
                let skill_md = entry_path.join("SKILL.md");
                if std::fs::metadata(&skill_md).is_ok_and(|m| m.is_file()) {
                    discovered.push((skill_md, entry_path));
                } else if depth < MAX_SKILL_SCAN_DEPTH {
                    next_subdirs.push((entry_path, false, depth + 1));
                }
            } else if is_root && meta.is_file() && name.ends_with(".md") && name != "SKILL.md" {
                discovered.push((entry_path, root_path.to_path_buf()));
            }
        }

        // LIFO 栈逆序压入以保持字典序遍历
        for item in next_subdirs.into_iter().rev() {
            stack.push(item);
        }
    }

    discovered.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(discovered)
}

fn compute_roots_fingerprint(roots: &[SkillRoot]) -> CachedFingerprintEntry {
    let mut hasher = DefaultHasher::new();
    let mut root_stamps = Vec::with_capacity(roots.len());
    let mut file_stamps = Vec::new();
    for root in roots {
        root.path.hash(&mut hasher);
        let root_stamp = read_file_stamp(&root.path);
        root_stamps.push((root.path.clone(), root_stamp.clone()));
        let Some(ref rs) = root_stamp else {
            false.hash(&mut hasher);
            continue;
        };
        true.hash(&mut hasher);
        hash_stamp(rs, &mut hasher);
        let Ok(files) = scan_root_skill_files(&root.path) else {
            continue;
        };
        for (file_path, _) in files {
            let stamp = read_file_stamp(&file_path);
            if let Some(ref st) = stamp {
                file_path.hash(&mut hasher);
                hash_stamp(st, &mut hasher);
            }
            file_stamps.push((file_path, stamp));
        }
    }
    CachedFingerprintEntry {
        scanned_at: Instant::now(),
        root_stamps,
        file_stamps,
        fingerprint: hasher.finish(),
    }
}

fn cached_stamps_still_valid(entry: &CachedFingerprintEntry) -> bool {
    if entry.scanned_at.elapsed() >= FINGERPRINT_SCAN_TTL {
        return false;
    }
    entry
        .root_stamps
        .iter()
        .all(|(p, prev)| read_file_stamp(p) == *prev)
        && entry
            .file_stamps
            .iter()
            .all(|(p, prev)| read_file_stamp(p) == *prev)
}

async fn discover_root(
    provider_name: &str,
    root: &SkillRoot,
) -> Result<Vec<SkillCandidate>, String> {
    let root_path = root.path.clone();
    let files = tokio::task::spawn_blocking(move || scan_root_skill_files(&root_path))
        .await
        .map_err(|err| format!("scan_root_skill_files join error: {err}"))??;

    let mut candidates = Vec::new();
    for (skill_path, resource_dir) in files {
        let locator = LocalLocator {
            path: skill_path,
            directory: resource_dir,
            source: root.source.clone(),
        };

        match read_and_parse_skill(&locator.path).await {
            Ok(Some((canonical_path, parsed))) => {
                candidates.push(SkillCandidate {
                    summary: SkillSummary {
                        path: Some(canonical_path),
                        name: parsed.name,
                        description: parsed.description,
                        when_to_use: parsed.when_to_use,
                        invocation: parsed.invocation,
                        source: root.source.clone(),
                        provider: provider_name.to_string(),
                        resource_base: Some(SkillResourceBase::Directory {
                            path: locator.directory.clone(),
                        }),
                    },
                    rank: root.rank,
                    locator: Arc::new(locator),
                    metadata: parsed.metadata,
                });
            }
            Ok(None) => {}
            Err(reason) => {
                tracing::warn!(
                    target: "mizar::skills",
                    "skill file {} ignored: {reason}",
                    locator.path.display()
                );
            }
        }
    }

    Ok(candidates)
}

async fn read_and_parse_skill(
    path: &Path,
) -> Result<Option<(PathBuf, crate::skills::frontmatter::ParsedSkillFile)>, String> {
    let canonical = match tokio::fs::canonicalize(path).await {
        Ok(p) => p,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let raw = match tokio::fs::read_to_string(&canonical).await {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", canonical.display(),)),
    };
    let parsed = parse_skill_markdown(&raw)?;
    Ok(Some((canonical, parsed)))
}

async fn run_cancellable<T, F>(
    cancel: &tokio_util::sync::CancellationToken,
    fut: F,
) -> Result<T, String>
where
    F: Future<Output = Result<T, String>>,
{
    tokio::select! {
        biased;
        () = cancel.cancelled() => Err("skill operation cancelled".to_string()),
        res = fut => res,
    }
}

impl SkillProvider for FilesystemSkillProvider {
    fn name(&self) -> &str {
        &self.config.provider_name
    }

    fn fingerprint(
        &self,
        options: &SkillLookupOptions,
    ) -> Pin<Box<dyn Future<Output = Option<u64>> + Send + '_>> {
        let config = self.config.clone();
        let default_cwd = self.default_cwd.clone();
        let cwd = options.cwd.clone();
        let fp_cache = Arc::clone(&self.fp_cache);
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                // 仅在锁内克隆缓存条目，文件系统 stat 与递归扫描完全在锁外执行
                let cached = { fp_cache.lock().unwrap().get(&cwd).cloned() };
                if let Some(entry) = cached
                    && cached_stamps_still_valid(&entry)
                {
                    return entry.fingerprint;
                }

                let roots = resolve_roots_for(&config, default_cwd.as_deref(), cwd.as_deref());
                let entry = compute_roots_fingerprint(&roots);
                let fp = entry.fingerprint;
                let mut guard = fp_cache.lock().unwrap();
                if guard.len() >= 16 && !guard.contains_key(&cwd) {
                    guard.clear();
                }
                guard.insert(cwd, entry);
                fp
            })
            .await
            .ok()
        })
    }

    fn list(
        &self,
        options: SkillLookupOptions,
    ) -> BoxFuture<'_, Result<SkillProviderObservation, String>> {
        Box::pin(async move {
            let config = self.config.clone();
            let default_cwd = self.default_cwd.clone();
            let cwd = options.cwd.clone();
            let roots = tokio::task::spawn_blocking(move || {
                resolve_roots_for(&config, default_cwd.as_deref(), cwd.as_deref())
            })
            .await
            .map_err(|err| format!("resolve_roots join error: {err}"))?;
            let mut candidates = Vec::new();
            let mut complete = true;

            for root in &roots {
                if options.cancel.is_cancelled() {
                    return Err("skill discovery cancelled".to_string());
                }
                match run_cancellable(
                    &options.cancel,
                    discover_root(&self.config.provider_name, root),
                )
                .await
                {
                    Ok(found) => candidates.extend(found),
                    Err(err) => {
                        if options.cancel.is_cancelled() {
                            return Err(err);
                        }
                        complete = false;
                        tracing::warn!(
                            target: "mizar::skills",
                            "skill root `{}` skipped: {err}",
                            root.path.display()
                        );
                    }
                }
            }

            Ok(SkillProviderObservation {
                candidates,
                complete,
            })
        })
    }

    fn get(
        &self,
        candidate: SkillCandidate,
        options: SkillLookupOptions,
    ) -> BoxFuture<'_, Result<Option<SkillDefinition>, String>> {
        Box::pin(async move {
            let Some(locator) = candidate.locator.downcast_ref::<LocalLocator>() else {
                return Err("invalid skill locator".to_string());
            };
            let fut = read_and_parse_skill(&locator.path);
            let parsed_opt = run_cancellable(&options.cancel, fut).await?;
            let Some((canonical, parsed)) = parsed_opt else {
                return Ok(None);
            };
            Ok(Some(SkillDefinition {
                summary: SkillSummary {
                    path: Some(canonical),
                    name: parsed.name,
                    description: parsed.description,
                    when_to_use: parsed.when_to_use,
                    invocation: parsed.invocation,
                    source: locator.source.clone(),
                    provider: self.config.provider_name.clone(),
                    resource_base: Some(SkillResourceBase::Directory {
                        path: locator.directory.clone(),
                    }),
                },
                content: parsed.body,
                metadata: parsed.metadata,
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn fingerprint_uses_ttl_cache_without_missing_direct_skill_edits() {
        let uniq = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base =
            std::env::temp_dir().join(format!("mizar_skills_fp_{}_{uniq}", std::process::id()));
        let skill_dir = base.join(".agents/skills/alpha");
        std::fs::create_dir_all(&skill_dir).unwrap();
        let skill_file = skill_dir.join("SKILL.md");
        std::fs::write(&skill_file, "---\nname: alpha\ndescription: v1\n---\nv1").unwrap();

        let cfg = FilesystemSkillConfig {
            include_user_roots: false,
            ..Default::default()
        };
        let provider = FilesystemSkillProvider::new(cfg, Some(base.clone()));
        let opts = SkillLookupOptions {
            cwd: Some(base.clone()),
            cancel: CancellationToken::new(),
        };

        let fp1 = provider.fingerprint(&opts).await.unwrap();
        let scanned_at_1 = provider
            .fp_cache
            .lock()
            .unwrap()
            .get(&Some(base.clone()))
            .unwrap()
            .scanned_at;

        // TTL 内再次查询应命中缓存（不重新全量扫盘）
        let fp2 = provider.fingerprint(&opts).await.unwrap();
        assert_eq!(fp1, fp2);
        let scanned_at_2 = provider
            .fp_cache
            .lock()
            .unwrap()
            .get(&Some(base.clone()))
            .unwrap()
            .scanned_at;
        assert_eq!(scanned_at_1, scanned_at_2);

        // 直接改写已存在 SKILL.md 后（不同长度触发 stamp 变更），即使在 TTL 内也必须立即刷新指纹
        std::fs::write(
            &skill_file,
            "---\nname: alpha\ndescription: v2 longer\n---\nv2 longer body",
        )
        .unwrap();
        let fp3 = provider.fingerprint(&opts).await.unwrap();
        assert_ne!(fp1, fp3);

        let _ = std::fs::remove_dir_all(&base);
    }
}
