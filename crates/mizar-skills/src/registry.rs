use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use cordis::{BoxFuture as CordisBoxFuture, Cleanup, Context, CordisError, Service};

#[cfg(all(mizar_loom, test))]
mod sync {
    pub(super) use loom::sync::Mutex;
    pub(super) use loom::sync::atomic::{AtomicU64, Ordering};
}

#[cfg(not(all(mizar_loom, test)))]
mod sync {
    pub(super) use std::sync::Mutex;
    pub(super) use std::sync::atomic::{AtomicU64, Ordering};
}

use crate::types::{
    RUNTIME_SKILL_RANK, SkillCandidate, SkillCatalogSnapshot, SkillDefinition, SkillLookupOptions,
    SkillProviderObservation, SkillRegistration, SkillSummary, is_skill_name,
};

pub const EVENT_SKILLS_CHANGE: &str = "skills/change";
const DEFAULT_COLLECT_CACHE_MAX: usize = 128;
const MAX_COLLECT_ATTEMPTS: usize = 2;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub trait SkillProvider: Send + Sync + 'static {
    fn name(&self) -> &str;

    fn fingerprint(&self, _options: &SkillLookupOptions) -> BoxFuture<'_, Option<u64>> {
        Box::pin(async { None })
    }

    fn list(
        &self,
        options: SkillLookupOptions,
    ) -> BoxFuture<'_, Result<SkillProviderObservation, String>>;

    fn get(
        &self,
        candidate: SkillCandidate,
        options: SkillLookupOptions,
    ) -> BoxFuture<'_, Result<Option<SkillDefinition>, String>>;
}

#[derive(Clone)]
struct RegisteredProvider {
    id: u64,
    order: i64,
    provider: Arc<dyn SkillProvider>,
}

#[derive(Clone)]
struct RuntimeEntry {
    id: u64,
    definition: SkillDefinition,
}

#[derive(Clone)]
struct IndexedCandidate {
    candidate: SkillCandidate,
    provider: Option<Arc<dyn SkillProvider>>,
    provider_id: u64,
    provider_order: i64,
    local_order: usize,
    runtime_def: Option<SkillDefinition>,
}

struct CatalogData<P, R> {
    providers: Vec<P>,
    runtime_skills: Vec<R>,
}

#[derive(Clone)]
struct CachedEntry<V> {
    revision: u64,
    value: V,
}

/// 带单调版本号的目录容器：`data` 锁内原子递增 `revision`（`Release`），
/// 并在释放 `data` 锁后于 `cache` 锁内仅淘汰旧版本缓存，防止跨锁撕裂与陈旧快照写回。
struct VersionedCatalog<P, R, K, V> {
    cache_max: usize,
    revision: sync::AtomicU64,
    data: sync::Mutex<CatalogData<P, R>>,
    cache: sync::Mutex<HashMap<K, CachedEntry<V>>>,
}

impl<P, R, K, V> VersionedCatalog<P, R, K, V>
where
    P: Clone,
    R: Clone,
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    fn new(cache_max: usize) -> Self {
        Self::with_initial_data(cache_max, Vec::new(), Vec::new())
    }

    fn with_initial_data(cache_max: usize, providers: Vec<P>, runtime_skills: Vec<R>) -> Self {
        Self {
            cache_max: cache_max.max(1),
            revision: sync::AtomicU64::new(1),
            data: sync::Mutex::new(CatalogData {
                providers,
                runtime_skills,
            }),
            cache: sync::Mutex::new(HashMap::new()),
        }
    }

    fn mutate_data<T>(&self, f: impl FnOnce(&mut Vec<P>, &mut Vec<R>) -> (T, bool)) -> (T, bool) {
        let (res, new_rev) = {
            let mut data = self.data.lock().unwrap();
            let guard = &mut *data;
            let (res, changed) = f(&mut guard.providers, &mut guard.runtime_skills);
            let new_rev = if changed {
                Some(self.revision.fetch_add(1, sync::Ordering::Release) + 1)
            } else {
                None
            };
            (res, new_rev)
        };
        if let Some(rev) = new_rev {
            let mut cache = self.cache.lock().unwrap();
            // 仅剔除旧版本条目，保留并发期间已基于最新 rev 完成写入的新快照
            cache.retain(|_, entry| entry.revision >= rev);
            (res, true)
        } else {
            (res, false)
        }
    }

    fn invalidate(&self) {
        let mut cache = self.cache.lock().unwrap();
        self.revision.fetch_add(1, sync::Ordering::Release);
        cache.clear();
    }

    fn try_get_cached(&self, key: &K) -> (u64, Option<V>) {
        let cache = self.cache.lock().unwrap();
        let rev = self.revision.load(sync::Ordering::Acquire);
        let hit = cache
            .get(key)
            .filter(|entry| entry.revision == rev)
            .map(|entry| entry.value.clone());
        (rev, hit)
    }

    fn is_revision_unchanged(&self, rev_before: u64) -> bool {
        self.revision.load(sync::Ordering::Acquire) == rev_before
    }

    fn snapshot_providers(&self) -> Vec<P> {
        self.data.lock().unwrap().providers.clone()
    }

    fn inspect_providers<T>(&self, f: impl FnOnce(&[P]) -> T) -> T {
        f(&self.data.lock().unwrap().providers)
    }

    fn snapshot_data(&self) -> (u64, Vec<P>, Vec<R>) {
        let data = self.data.lock().unwrap();
        let rev = self.revision.load(sync::Ordering::Acquire);
        (rev, data.providers.clone(), data.runtime_skills.clone())
    }

    fn commit_cache(&self, key: K, rev_snapshot: u64, value: V) -> bool {
        let mut cache = self.cache.lock().unwrap();
        if self.revision.load(sync::Ordering::Acquire) != rev_snapshot {
            return false;
        }
        if !cache.contains_key(&key)
            && cache.len() >= self.cache_max
            && let Some(first_key) = cache.keys().next().cloned()
        {
            cache.remove(&first_key);
        }
        cache.insert(
            key,
            CachedEntry {
                revision: rev_snapshot,
                value,
            },
        );
        true
    }
}

#[derive(Clone)]
struct CachedCollect {
    entries: HashMap<String, IndexedCandidate>,
    fingerprints: Vec<(u64, Option<u64>)>,
}

#[derive(Debug, Clone, Default)]
pub struct SkillRegistryConfig {
    pub collect_cache_max_entries: Option<usize>,
}

pub struct SkillRegistry {
    ctx: Context,
    next_id: sync::AtomicU64,
    catalog: VersionedCatalog<RegisteredProvider, RuntimeEntry, Option<PathBuf>, CachedCollect>,
}

impl SkillRegistry {
    pub fn register_provider(
        self: &Arc<Self>,
        provider: Arc<dyn SkillProvider>,
    ) -> Result<Cleanup, String> {
        let name = provider.name().to_string();
        if name == "runtime" {
            return Err("\"runtime\" is reserved for runtime skill registrations".to_string());
        }
        let id = self.next_id.fetch_add(1, sync::Ordering::Relaxed);
        let (res, changed) = self.catalog.mutate_data(|providers, _| {
            if providers.iter().any(|p| p.provider.name() == name) {
                return (
                    Err(format!("skill provider \"{name}\" is already registered")),
                    false,
                );
            }
            providers.push(RegisteredProvider {
                id,
                order: id as i64,
                provider,
            });
            (Ok(()), true)
        });
        res?;
        if changed {
            self.ctx.emit(EVENT_SKILLS_CHANGE, ());
        }

        let weak = Arc::downgrade(self);
        Ok(Cleanup::sync(move || {
            if let Some(reg) = weak.upgrade() {
                let (_, changed) = reg.catalog.mutate_data(|providers, _| {
                    let before = providers.len();
                    providers.retain(|p| p.id != id);
                    ((), providers.len() != before)
                });
                if changed {
                    reg.ctx.emit(EVENT_SKILLS_CHANGE, ());
                }
            }
        }))
    }

    pub fn register(self: &Arc<Self>, skill: SkillRegistration) -> Result<Cleanup, String> {
        if !is_skill_name(&skill.name) {
            return Err(format!("invalid skill name \"{}\"", skill.name));
        }
        if skill.description.trim().is_empty() {
            return Err(format!("skill \"{}\" requires a description", skill.name));
        }
        let id = self.next_id.fetch_add(1, sync::Ordering::Relaxed);
        let skill_name = skill.name.clone();
        let (inserted, changed) = self.catalog.mutate_data(|_, runtime_skills| {
            if runtime_skills
                .iter()
                .any(|e| e.definition.summary.name == skill_name)
            {
                return (false, false);
            }
            runtime_skills.push(RuntimeEntry {
                id,
                definition: SkillDefinition {
                    summary: SkillSummary {
                        path: skill.path,
                        name: skill.name,
                        description: skill.description,
                        when_to_use: skill.when_to_use,
                        invocation: skill.invocation.unwrap_or_default(),
                        source: skill.source,
                        provider: "runtime".to_string(),
                        resource_base: skill.resource_base,
                    },
                    content: skill.content,
                    metadata: skill.metadata,
                },
            });
            (true, true)
        });
        if !inserted {
            self.ctx.logger_named("skills").warn(format!(
                "runtime skill \"{skill_name}\" ignored because it is already registered"
            ));
            return Ok(Cleanup::sync(|| {}));
        }
        if changed {
            self.ctx.emit(EVENT_SKILLS_CHANGE, ());
        }

        let weak = Arc::downgrade(self);
        Ok(Cleanup::sync(move || {
            if let Some(reg) = weak.upgrade() {
                let (_, changed) = reg.catalog.mutate_data(|_, runtime_skills| {
                    let before = runtime_skills.len();
                    runtime_skills.retain(|e| e.id != id);
                    ((), runtime_skills.len() != before)
                });
                if changed {
                    reg.ctx.emit(EVENT_SKILLS_CHANGE, ());
                }
            }
        }))
    }

    pub fn invalidate_cache(&self) {
        self.catalog.invalidate();
        self.ctx.emit(EVENT_SKILLS_CHANGE, ());
    }

    async fn is_cache_fresh(
        &self,
        cached: &CachedCollect,
        providers: &[RegisteredProvider],
        options: &SkillLookupOptions,
    ) -> bool {
        if cached.fingerprints.len() != providers.len() {
            return false;
        }
        for (reg, (cached_id, cached_fp)) in providers.iter().zip(&cached.fingerprints) {
            if reg.id != *cached_id {
                return false;
            }
            let current = reg.provider.fingerprint(options).await;
            if current != *cached_fp {
                return false;
            }
        }
        true
    }

    async fn collect(
        &self,
        options: &SkillLookupOptions,
    ) -> (HashMap<String, IndexedCandidate>, bool) {
        let mut attempt = 0;
        loop {
            let (rev_before, cache_hit) = self.catalog.try_get_cached(&options.cwd);
            if let Some(cached) = cache_hit {
                let providers = self.catalog.snapshot_providers();
                if self.is_cache_fresh(&cached, &providers, options).await
                    && self.catalog.is_revision_unchanged(rev_before)
                {
                    return (cached.entries, true);
                }
            }

            let (rev_snapshot, providers, mut rt) = self.catalog.snapshot_data();
            let mut all_candidates: Vec<IndexedCandidate> = Vec::new();
            let mut complete = true;
            let mut fingerprints = Vec::with_capacity(providers.len());

            // 运行时注册 skill（rank = 250）
            rt.sort_by(|a, b| a.definition.summary.name.cmp(&b.definition.summary.name));
            for (idx, entry) in rt.into_iter().enumerate() {
                let def = entry.definition;
                let candidate = SkillCandidate {
                    summary: def.summary.clone(),
                    rank: RUNTIME_SKILL_RANK,
                    locator: Arc::new(()),
                    metadata: def.metadata.clone(),
                };
                all_candidates.push(IndexedCandidate {
                    candidate,
                    provider: None,
                    provider_id: entry.id,
                    provider_order: -1,
                    local_order: idx,
                    runtime_def: Some(def),
                });
            }

            // 所有已注册提供方
            for reg in &providers {
                let fp = reg.provider.fingerprint(options).await;
                fingerprints.push((reg.id, fp));

                match reg.provider.list(options.clone()).await {
                    Ok(obs) => {
                        if !obs.complete {
                            complete = false;
                        }
                        for (idx, candidate) in obs.candidates.into_iter().enumerate() {
                            if !is_skill_name(&candidate.summary.name)
                                || candidate.summary.description.is_empty()
                            {
                                complete = false;
                                continue;
                            }
                            all_candidates.push(IndexedCandidate {
                                candidate,
                                provider: Some(Arc::clone(&reg.provider)),
                                provider_id: reg.id,
                                provider_order: reg.order,
                                local_order: idx,
                                runtime_def: None,
                            });
                        }
                    }
                    Err(err) => {
                        complete = false;
                        self.ctx.logger_named("skills").warn(format!(
                            "skill provider `{}` skipped: {err}",
                            reg.provider.name()
                        ));
                    }
                }
            }

            // 按 rank 升序 -> provider_order 升序 -> local_order 升序去重，先胜出
            all_candidates.sort_by(|a, b| {
                a.candidate
                    .rank
                    .cmp(&b.candidate.rank)
                    .then_with(|| a.provider_order.cmp(&b.provider_order))
                    .then_with(|| a.local_order.cmp(&b.local_order))
            });

            let mut winners = HashMap::new();
            let mut seen = HashSet::new();
            for entry in all_candidates {
                let name = entry.candidate.summary.name.clone();
                if !seen.insert(name.clone()) {
                    self.ctx.logger_named("skills").warn(format!(
                        "skill \"{name}\" from {} ignored because a higher-priority skill already exists",
                        entry.candidate.summary.source.as_str()
                    ));
                    continue;
                }
                winners.insert(name, entry);
            }

            if complete {
                if !self.catalog.commit_cache(
                    options.cwd.clone(),
                    rev_snapshot,
                    CachedCollect {
                        entries: winners.clone(),
                        fingerprints,
                    },
                ) {
                    attempt += 1;
                    if attempt < MAX_COLLECT_ATTEMPTS {
                        continue;
                    }
                    return (winners, false);
                }
                return (winners, true);
            }

            if !self.catalog.is_revision_unchanged(rev_snapshot) {
                attempt += 1;
                if attempt < MAX_COLLECT_ATTEMPTS {
                    continue;
                }
                return (winners, false);
            }

            return (winners, complete);
        }
    }

    pub async fn snapshot(&self, options: SkillLookupOptions) -> SkillCatalogSnapshot {
        let (entries, complete) = self.collect(&options).await;
        let mut skills: Vec<SkillSummary> =
            entries.into_values().map(|e| e.candidate.summary).collect();
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        SkillCatalogSnapshot { skills, complete }
    }

    pub async fn list(&self, options: SkillLookupOptions) -> Vec<SkillSummary> {
        self.snapshot(options).await.skills
    }

    pub async fn get(
        &self,
        name: &str,
        options: SkillLookupOptions,
    ) -> Result<Option<SkillDefinition>, String> {
        if !is_skill_name(name) {
            return Ok(None);
        }
        let (entries, _) = self.collect(&options).await;
        let Some(matched) = entries.get(name).cloned() else {
            return Ok(None);
        };
        if options.cancel.is_cancelled() {
            return Err("skill load cancelled".to_string());
        }
        if let Some(def) = matched.runtime_def {
            return Ok(Some(def));
        }
        let Some(provider) = matched.provider else {
            return Ok(None);
        };
        let loaded = provider.get(matched.candidate, options).await?;
        let Some(def) = loaded else {
            return Ok(None);
        };
        if def.summary.name != name {
            let still_live = self.catalog.inspect_providers(|providers| {
                providers.iter().any(|p| p.id == matched.provider_id)
            });
            if still_live {
                self.invalidate_cache();
            }
            return Ok(None);
        }
        Ok(Some(def))
    }
}

impl Service for SkillRegistry {
    type Config = SkillRegistryConfig;

    fn service_name() -> &'static str {
        "skills"
    }

    fn plugin_name() -> &'static str {
        "SkillRegistry"
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        let cache_max = config
            .collect_cache_max_entries
            .unwrap_or(DEFAULT_COLLECT_CACHE_MAX)
            .max(1);
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            next_id: sync::AtomicU64::new(1),
            catalog: VersionedCatalog::new(cache_max),
        }))
    }

    fn init(self: Arc<Self>, _ctx: &Context) -> CordisBoxFuture<'static, Result<(), CordisError>> {
        Box::pin(async { Ok(()) })
    }
}

pub trait SkillsContextExt {
    fn skills(&self) -> Result<Arc<SkillRegistry>, CordisError>;
}

impl SkillsContextExt for Context {
    fn skills(&self) -> Result<Arc<SkillRegistry>, CordisError> {
        self.service::<SkillRegistry>("skills")
    }
}

#[cfg(all(mizar_loom, test))]
mod loom_tests {
    use super::*;
    use loom::sync::Arc as LoomArc;
    use loom::sync::atomic::{AtomicBool, Ordering};
    use loom::thread;

    #[test]
    fn loom_concurrent_mutation_never_publishes_stale_cache() {
        loom::model(|| {
            let catalog = LoomArc::new(VersionedCatalog::<u32, u32, (), u32>::with_initial_data(
                4,
                vec![1],
                vec![],
            ));

            let w_cat = LoomArc::clone(&catalog);
            let r_cat = LoomArc::clone(&catalog);

            let writer = thread::spawn(move || {
                w_cat.mutate_data(|providers, _| {
                    providers[0] = 2;
                    ((), true)
                });
            });

            let reader = thread::spawn(move || {
                let (rev, providers, _) = r_cat.snapshot_data();
                let _ = r_cat.commit_cache((), rev, providers[0]);
            });

            writer.join().unwrap();
            reader.join().unwrap();

            if let (_, Some(cached)) = catalog.try_get_cached(&()) {
                assert_eq!(cached, 2, "stale value committed to cache after mutation");
            }
        });
    }

    #[test]
    fn loom_fast_path_never_hits_stale_after_mutation_returns() {
        loom::model(|| {
            let catalog = LoomArc::new(VersionedCatalog::<u32, u32, (), u32>::with_initial_data(
                4,
                vec![1],
                vec![],
            ));
            let (rev0, p0, _) = catalog.snapshot_data();
            assert!(catalog.commit_cache((), rev0, p0[0]));

            let mutated_done = LoomArc::new(AtomicBool::new(false));
            let w_cat = LoomArc::clone(&catalog);
            let w_flag = LoomArc::clone(&mutated_done);
            let r_cat = LoomArc::clone(&catalog);
            let r_flag = LoomArc::clone(&mutated_done);

            let writer = thread::spawn(move || {
                w_cat.mutate_data(|providers, _| {
                    providers[0] = 2;
                    ((), true)
                });
                w_flag.store(true, Ordering::Release);
            });

            let reader = thread::spawn(move || {
                let (rev_before, hit) = r_cat.try_get_cached(&());
                if r_flag.load(Ordering::Acquire)
                    && let Some(val) = hit
                    && r_cat.is_revision_unchanged(rev_before)
                {
                    assert_eq!(
                        val, 2,
                        "fast path served stale value after mutation returned"
                    );
                }
            });

            writer.join().unwrap();
            reader.join().unwrap();
        });
    }

    #[test]
    fn loom_invalidate_races_with_commit_cache() {
        loom::model(|| {
            let catalog = LoomArc::new(VersionedCatalog::<u32, u32, (), u32>::with_initial_data(
                4,
                vec![1],
                vec![],
            ));
            let (rev_snap, providers, _) = catalog.snapshot_data();

            let w_cat = LoomArc::clone(&catalog);
            let r_cat = LoomArc::clone(&catalog);

            let writer = thread::spawn(move || {
                w_cat.invalidate();
            });

            let reader = thread::spawn(move || {
                let _ = r_cat.commit_cache((), rev_snap, providers[0]);
            });

            writer.join().unwrap();
            reader.join().unwrap();

            assert!(
                catalog.try_get_cached(&()).1.is_none(),
                "invalidated revision leaked back into cache"
            );
        });
    }
}
