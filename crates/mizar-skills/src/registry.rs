use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cordis::{BoxFuture as CordisBoxFuture, Cleanup, Context, CordisError, Service};

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

#[derive(Clone)]
struct CachedCollect {
    revision: u64,
    entries: HashMap<String, IndexedCandidate>,
    fingerprints: Vec<(u64, Option<u64>)>,
}

#[derive(Debug, Clone, Default)]
pub struct SkillRegistryConfig {
    pub collect_cache_max_entries: Option<usize>,
}

pub struct SkillRegistry {
    ctx: Context,
    cache_max: usize,
    next_id: AtomicU64,
    revision: AtomicU64,
    providers: Mutex<Vec<RegisteredProvider>>,
    runtime_skills: Mutex<Vec<RuntimeEntry>>,
    collect_cache: Mutex<HashMap<Option<PathBuf>, CachedCollect>>,
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
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut list = self.providers.lock().unwrap();
            if list.iter().any(|p| p.provider.name() == name) {
                return Err(format!("skill provider \"{name}\" is already registered"));
            }
            list.push(RegisteredProvider {
                id,
                order: id as i64,
                provider,
            });
        }
        self.invalidate_cache();

        let weak = Arc::downgrade(self);
        Ok(Cleanup::sync(move || {
            if let Some(reg) = weak.upgrade() {
                let mut list = reg.providers.lock().unwrap();
                let before = list.len();
                list.retain(|p| p.id != id);
                if list.len() != before {
                    drop(list);
                    reg.invalidate_cache();
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
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut list = self.runtime_skills.lock().unwrap();
            if list.iter().any(|e| e.definition.summary.name == skill.name) {
                self.ctx.logger_named("skills").warn(format!(
                    "runtime skill \"{}\" ignored because it is already registered",
                    skill.name
                ));
                return Ok(Cleanup::sync(|| {}));
            }
            list.push(RuntimeEntry {
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
        }
        self.invalidate_cache();

        let weak = Arc::downgrade(self);
        Ok(Cleanup::sync(move || {
            if let Some(reg) = weak.upgrade() {
                let mut list = reg.runtime_skills.lock().unwrap();
                let before = list.len();
                list.retain(|e| e.id != id);
                if list.len() != before {
                    drop(list);
                    reg.invalidate_cache();
                }
            }
        }))
    }

    pub fn invalidate_cache(&self) {
        {
            let mut cache = self.collect_cache.lock().unwrap();
            self.revision.fetch_add(1, Ordering::SeqCst);
            cache.clear();
        }
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
            let (rev_before, cache_hit) = {
                let cache = self.collect_cache.lock().unwrap();
                (
                    self.revision.load(Ordering::SeqCst),
                    cache.get(&options.cwd).cloned(),
                )
            };
            let providers = self.providers.lock().unwrap().clone();
            if let Some(cached) = cache_hit
                && cached.revision == rev_before
                && self.is_cache_fresh(&cached, &providers, options).await
                && self.revision.load(Ordering::SeqCst) == rev_before
            {
                return (cached.entries, true);
            }

            let mut all_candidates: Vec<IndexedCandidate> = Vec::new();
            let mut complete = true;
            let mut fingerprints = Vec::with_capacity(providers.len());

            // 运行时注册 skill（rank = 250）
            {
                let mut rt = self.runtime_skills.lock().unwrap().clone();
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
                let mut cache = self.collect_cache.lock().unwrap();
                if self.revision.load(Ordering::SeqCst) != rev_before {
                    drop(cache);
                    attempt += 1;
                    if attempt < MAX_COLLECT_ATTEMPTS {
                        continue;
                    }
                    return (winners, false);
                }
                if cache.len() >= self.cache_max
                    && let Some(first_key) = cache.keys().next().cloned()
                {
                    cache.remove(&first_key);
                }
                cache.insert(
                    options.cwd.clone(),
                    CachedCollect {
                        revision: rev_before,
                        entries: winners.clone(),
                        fingerprints,
                    },
                );
                return (winners, true);
            }

            if self.revision.load(Ordering::SeqCst) != rev_before {
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
            let still_live = self
                .providers
                .lock()
                .unwrap()
                .iter()
                .any(|p| p.id == matched.provider_id);
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
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            cache_max: config
                .collect_cache_max_entries
                .unwrap_or(DEFAULT_COLLECT_CACHE_MAX)
                .max(1),
            next_id: AtomicU64::new(1),
            revision: AtomicU64::new(1),
            providers: Mutex::new(Vec::new()),
            runtime_skills: Mutex::new(Vec::new()),
            collect_cache: Mutex::new(HashMap::new()),
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
