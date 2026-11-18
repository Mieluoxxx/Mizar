use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use cordis::Context;

use crate::frontmatter::parse_skill_markdown;
use crate::registry::{BoxFuture, SkillProvider};
use crate::types::{
    BUNDLED_SKILL_RANK, SkillCandidate, SkillDefinition, SkillLookupOptions,
    SkillProviderObservation, SkillResourceBase, SkillSource, SkillSummary,
};

pub const PROJECT_AGENTS_RANK: u32 = 200;
pub const CUSTOM_RANK: u32 = 300;
pub const USER_AGENTS_RANK: u32 = 500;
const MAX_SKILL_SCAN_DEPTH: usize = 8;

#[derive(Debug, Clone)]
pub struct FilesystemSkillConfig {
    pub provider_name: String,
    pub include_default_roots: bool,
    pub agents_home: Option<PathBuf>,
    pub custom_skill_dirs: Vec<PathBuf>,
    pub bundled_skill_dir: Option<PathBuf>,
}

impl Default for FilesystemSkillConfig {
    fn default() -> Self {
        Self {
            provider_name: "filesystem".to_string(),
            include_default_roots: true,
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
    if let Ok(env) = std::env::var("DSH_AGENTS_HOME") {
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

pub struct FilesystemSkillProvider {
    ctx: Context,
    config: FilesystemSkillConfig,
    default_cwd: Option<PathBuf>,
}

impl FilesystemSkillProvider {
    pub fn new(ctx: Context, config: FilesystemSkillConfig, default_cwd: Option<PathBuf>) -> Self {
        Self {
            ctx,
            config,
            default_cwd,
        }
    }

    fn resolve_roots(&self, cwd: Option<&Path>) -> Vec<SkillRoot> {
        let mut roots = Vec::new();
        let mut seen_paths = HashSet::new();

        let user_agents_dir = self
            .config
            .include_default_roots
            .then(|| resolve_agents_home(self.config.agents_home.as_deref()).map(|h| h.join("skills")))
            .flatten();
        let user_agents_key = user_agents_dir
            .as_ref()
            .map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()));

        let effective_cwd = cwd.or(self.default_cwd.as_deref());
        if self.config.include_default_roots
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

        for dir in &self.config.custom_skill_dirs {
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

        let bundled = self.config.bundled_skill_dir.clone().or_else(|| {
            self.config
                .include_default_roots
                .then(|| std::env::var("DSH_BUNDLED_SKILL_DIR").ok())
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

    pub fn matches_mutation_path(&self, target: &Path, cwd: Option<&Path>) -> bool {
        let abs_target = std::fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
        self.resolve_roots(cwd).into_iter().any(|root| {
            let root_abs = std::fs::canonicalize(&root.path).unwrap_or(root.path);
            is_potential_skill_path(&root_abs, &abs_target)
        })
    }
}

fn hash_mtime(meta: &std::fs::Metadata, hasher: &mut DefaultHasher) {
    meta.len().hash(hasher);
    if let Ok(mtime) = meta.modified()
        && let Ok(dur) = mtime.duration_since(UNIX_EPOCH)
    {
        dur.as_secs().hash(hasher);
        dur.subsec_nanos().hash(hasher);
    }
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

fn compute_roots_fingerprint(roots: &[SkillRoot]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for root in roots {
        root.path.hash(&mut hasher);
        let Ok(root_meta) = std::fs::metadata(&root.path) else {
            false.hash(&mut hasher);
            continue;
        };
        true.hash(&mut hasher);
        hash_mtime(&root_meta, &mut hasher);
        let Ok(files) = scan_root_skill_files(&root.path) else {
            continue;
        };
        for (file_path, _) in files {
            if let Ok(meta) = std::fs::metadata(&file_path) {
                file_path.hash(&mut hasher);
                hash_mtime(&meta, &mut hasher);
            }
        }
    }
    hasher.finish()
}

async fn discover_root(
    ctx: &Context,
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
                ctx.logger_named("skills").warn(format!(
                    "skill file {} ignored: {reason}",
                    locator.path.display()
                ));
            }
        }
    }

    Ok(candidates)
}

async fn read_and_parse_skill(
    path: &Path,
) -> Result<Option<(PathBuf, crate::frontmatter::ParsedSkillFile)>, String> {
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

async fn run_cancellable<T, F>(cancel: &tokio_util::sync::CancellationToken, fut: F) -> Result<T, String>
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

    fn fingerprint(&self, options: &SkillLookupOptions) -> Pin<Box<dyn Future<Output = Option<u64>> + Send + '_>> {
        let roots = self.resolve_roots(options.cwd.as_deref());
        Box::pin(async move { Some(compute_roots_fingerprint(&roots)) })
    }

    fn list(&self, options: SkillLookupOptions) -> BoxFuture<'_, Result<SkillProviderObservation, String>> {
        Box::pin(async move {
            let roots = self.resolve_roots(options.cwd.as_deref());
            let mut candidates = Vec::new();
            let mut complete = true;

            for root in &roots {
                if options.cancel.is_cancelled() {
                    return Err("skill discovery cancelled".to_string());
                }
                match run_cancellable(
                    &options.cancel,
                    discover_root(&self.ctx, &self.config.provider_name, root),
                )
                .await
                {
                    Ok(found) => candidates.extend(found),
                    Err(err) => {
                        if options.cancel.is_cancelled() {
                            return Err(err);
                        }
                        complete = false;
                        self.ctx.logger_named("skills").warn(format!(
                            "skill root `{}` skipped: {err}",
                            root.path.display()
                        ));
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
