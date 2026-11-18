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

pub const PROJECT_DSH_RANK: u32 = 100;
pub const PROJECT_AGENTS_RANK: u32 = 200;
pub const CUSTOM_RANK: u32 = 300;
pub const USER_DSH_RANK: u32 = 400;
pub const USER_AGENTS_RANK: u32 = 500;

#[derive(Debug, Clone)]
pub struct FilesystemSkillConfig {
    pub provider_name: String,
    pub include_default_roots: bool,
    pub dsh_home: Option<PathBuf>,
    pub agents_home: Option<PathBuf>,
    pub custom_skill_dirs: Vec<PathBuf>,
    pub bundled_skill_dir: Option<PathBuf>,
}

impl Default for FilesystemSkillConfig {
    fn default() -> Self {
        Self {
            provider_name: "filesystem".to_string(),
            include_default_roots: true,
            dsh_home: None,
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
    skip_system: bool,
}

#[derive(Debug, Clone)]
struct LocalLocator {
    path: PathBuf,
    directory: PathBuf,
    source: SkillSource,
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

/// 判断目标路径是否可能命中任一潜在技能路径（`<root>/<name>.md` 或 `<root>/<name>/SKILL.md`）。
pub fn is_potential_skill_path(root: &Path, target: &Path, skip_system: bool) -> bool {
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
    if comps.is_empty() || comps.len() > 2 {
        return false;
    }
    if skip_system && comps[0] == ".system" {
        return false;
    }
    if comps.len() == 1 {
        comps[0].ends_with(".md")
    } else {
        comps[1] == "SKILL.md"
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
        let effective_cwd = cwd.or(self.default_cwd.as_deref());
        if self.config.include_default_roots
            && let Some(dir) = effective_cwd
        {
            let abs = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
            let project_root = find_project_root(&abs);
            roots.push(SkillRoot {
                path: project_root.join(".dsh-rs").join("skills"),
                source: SkillSource::ProjectDsh,
                rank: PROJECT_DSH_RANK,
                skip_system: false,
            });
            roots.push(SkillRoot {
                path: project_root.join(".agents").join("skills"),
                source: SkillSource::ProjectAgents,
                rank: PROJECT_AGENTS_RANK,
                skip_system: false,
            });
        }

        for dir in &self.config.custom_skill_dirs {
            roots.push(SkillRoot {
                path: dir.clone(),
                source: SkillSource::Custom,
                rank: CUSTOM_RANK,
                skip_system: false,
            });
        }

        if self.config.include_default_roots {
            if let Some(dsh_home) = resolve_dsh_home(self.config.dsh_home.as_deref()) {
                roots.push(SkillRoot {
                    path: dsh_home.join("skills"),
                    source: SkillSource::UserDsh,
                    rank: USER_DSH_RANK,
                    skip_system: true,
                });
            }
            if let Some(agents_home) = resolve_agents_home(self.config.agents_home.as_deref()) {
                roots.push(SkillRoot {
                    path: agents_home.join("skills"),
                    source: SkillSource::UserAgents,
                    rank: USER_AGENTS_RANK,
                    skip_system: false,
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
            roots.push(SkillRoot {
                path: bundled_dir,
                source: SkillSource::Bundled,
                rank: BUNDLED_SKILL_RANK,
                skip_system: false,
            });
        }

        roots
    }

    pub fn matches_mutation_path(&self, target: &Path, cwd: Option<&Path>) -> bool {
        let abs_target = std::fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
        self.resolve_roots(cwd).into_iter().any(|root| {
            let root_abs = std::fs::canonicalize(&root.path).unwrap_or(root.path);
            is_potential_skill_path(&root_abs, &abs_target, root.skip_system)
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
        let Ok(rd) = std::fs::read_dir(&root.path) else {
            continue;
        };
        let mut names: Vec<(String, PathBuf)> = rd
            .flatten()
            .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
            .collect();
        names.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, path) in names {
            if root.skip_system && name == ".system" {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            if meta.is_dir() {
                let skill_file = path.join("SKILL.md");
                if let Ok(sf_meta) = std::fs::metadata(&skill_file)
                    && sf_meta.is_file()
                {
                    name.hash(&mut hasher);
                    hash_mtime(&sf_meta, &mut hasher);
                }
            } else if meta.is_file() && name.ends_with(".md") {
                name.hash(&mut hasher);
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
    if !root.path.exists() {
        return Ok(Vec::new());
    }
    let mut rd = tokio::fs::read_dir(&root.path)
        .await
        .map_err(|e| format!("read_dir {}: {e}", root.path.display()))?;

    let mut entries = Vec::new();
    while let Some(entry) = rd
        .next_entry()
        .await
        .map_err(|e| format!("next_entry {}: {e}", root.path.display()))?
    {
        let name = entry.file_name().to_string_lossy().into_owned();
        if root.skip_system && name == ".system" {
            continue;
        }
        entries.push((name, entry.path()));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut candidates = Vec::new();
    for (name, entry_path) in entries {
        let Ok(meta) = tokio::fs::metadata(&entry_path).await else {
            continue;
        };
        let locator = if meta.is_dir() {
            let skill_md = entry_path.join("SKILL.md");
            if tokio::fs::metadata(&skill_md)
                .await
                .is_ok_and(|m| m.is_file())
            {
                LocalLocator {
                    path: skill_md,
                    directory: entry_path,
                    source: root.source.clone(),
                }
            } else {
                continue;
            }
        } else if meta.is_file() && name.ends_with(".md") {
            LocalLocator {
                path: entry_path,
                directory: root.path.clone(),
                source: root.source.clone(),
            }
        } else {
            continue;
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
