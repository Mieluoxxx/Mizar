use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cordis::{BoxFuture, Cleanup, Context, CordisError, Service};
use futures::Stream;

use crate::types::{GenerateOptions, ModelInfo, ProviderInfo, StreamChunk};
use mizar_llm_types::LlmError;

pub const EVENT_ADAPTERS_UPDATED: &str = "llm/adapters-updated";
pub const EVENT_LLM_STREAM: &str = "llm/stream";

pub type BoxChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>;

/// 模型适配器：把统一的 `GenerateOptions` 翻译成具体模型协议的流式输出。
pub trait LlmAdapter: Send + Sync + 'static {
    fn stream(&self, options: GenerateOptions) -> BoxChunkStream;

    /// 返回当前路由的协议与静态模型目录信息（可选）。
    fn info(&self) -> Option<ProviderInfo> {
        None
    }

    /// 异步发现远端或目录中的可用模型（默认返回 `info()` 里的静态列表）。
    fn discover_models(&self) -> BoxFuture<'static, Result<Vec<ModelInfo>, LlmError>> {
        let models = self.info().map(|i| i.models).unwrap_or_default();
        Box::pin(async move { Ok(models) })
    }

    /// 带额外请求头的模型发现（供路由层注入 provider headers）；默认回退到 `discover_models()`。
    fn discover_models_with_headers(
        &self,
        extra_headers: &[(String, String)],
    ) -> BoxFuture<'static, Result<Vec<ModelInfo>, LlmError>> {
        let _ = extra_headers;
        self.discover_models()
    }
}

struct AdapterEntry {
    id: u64,
    provider: String,
    adapter: Arc<dyn LlmAdapter>,
}

pub struct LlmRuntime {
    ctx: Context,
    next_id: AtomicU64,
    adapters: Mutex<Vec<AdapterEntry>>,
    default_provider: Mutex<Option<String>>,
}

impl LlmRuntime {
    /// 适配器插件调用它注册自己。返回的 `Cleanup` 放入 `ctx.effect` 即可随插件卸载自动注销。
    pub fn register_adapter(
        self: &Arc<Self>,
        provider: impl Into<String>,
        adapter: Arc<dyn LlmAdapter>,
    ) -> Result<Cleanup, LlmError> {
        let provider = provider.into();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut list = self.adapters.lock().unwrap();
            if list.iter().any(|e| e.provider == provider) {
                return Err(LlmError::duplicate_provider(&provider));
            }
            list.push(AdapterEntry {
                id,
                provider: provider.clone(),
                adapter,
            });
        }
        self.ctx.emit(EVENT_ADAPTERS_UPDATED, ());

        Ok(self.make_unregister_cleanup(id, provider))
    }

    /// 原子替换（或新增）指定路由的适配器，适用于路由配置热更新。
    pub fn upsert_adapter(
        self: &Arc<Self>,
        provider: impl Into<String>,
        adapter: Arc<dyn LlmAdapter>,
    ) -> Cleanup {
        let provider = provider.into();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut list = self.adapters.lock().unwrap();
            if let Some(entry) = list.iter_mut().find(|e| e.provider == provider) {
                entry.id = id;
                entry.adapter = adapter;
            } else {
                list.push(AdapterEntry {
                    id,
                    provider: provider.clone(),
                    adapter,
                });
            }
        }
        self.ctx.emit(EVENT_ADAPTERS_UPDATED, ());

        self.make_unregister_cleanup(id, provider)
    }

    /// 显式移除指定路由适配器。
    pub fn remove_adapter(&self, provider: &str) -> bool {
        let removed = {
            let mut list = self.adapters.lock().unwrap();
            if let Some(idx) = list.iter().position(|e| e.provider == provider) {
                list.remove(idx);
                true
            } else {
                false
            }
        };
        if removed {
            self.ctx.emit(EVENT_ADAPTERS_UPDATED, ());
        }
        removed
    }

    /// 原子同步指定集合的适配器（先移除再原地更新/追加），只触发一次 `EVENT_ADAPTERS_UPDATED`。
    pub fn sync_adapters(
        self: &Arc<Self>,
        removed: &[String],
        upserts: Vec<(String, Arc<dyn LlmAdapter>)>,
    ) {
        if removed.is_empty() && upserts.is_empty() {
            return;
        }
        let changed = {
            let mut list = self.adapters.lock().unwrap();
            let mut dirty = false;
            if !removed.is_empty() {
                let before = list.len();
                list.retain(|e| !removed.iter().any(|r| r == &e.provider));
                if list.len() != before {
                    dirty = true;
                }
            }
            for (provider, adapter) in upserts {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                if let Some(entry) = list.iter_mut().find(|e| e.provider == provider) {
                    entry.id = id;
                    entry.adapter = adapter;
                } else {
                    list.push(AdapterEntry {
                        id,
                        provider,
                        adapter,
                    });
                }
                dirty = true;
            }
            dirty
        };
        if changed {
            self.ctx.emit(EVENT_ADAPTERS_UPDATED, ());
        }
    }

    /// 显式设置默认 provider（`resolve_adapter(None)` 优先选用）；传 `None` 则恢复按注册序回退。
    pub fn set_default_provider(&self, provider: Option<String>) {
        *self.default_provider.lock().unwrap() = provider;
    }

    pub fn default_provider(&self) -> Option<String> {
        self.default_provider.lock().unwrap().clone()
    }

    fn make_unregister_cleanup(self: &Arc<Self>, id: u64, provider: String) -> Cleanup {
        let runtime = Arc::downgrade(self);
        Cleanup::sync(move || {
            if let Some(rt) = runtime.upgrade() {
                let removed = {
                    let mut list = rt.adapters.lock().unwrap();
                    if let Some(idx) = list
                        .iter()
                        .position(|e| e.id == id && e.provider == provider)
                    {
                        list.remove(idx);
                        true
                    } else {
                        false
                    }
                };
                if removed {
                    rt.ctx.emit(EVENT_ADAPTERS_UPDATED, ());
                }
            }
        })
    }

    pub fn providers(&self) -> Vec<String> {
        self.adapters
            .lock()
            .unwrap()
            .iter()
            .map(|e| e.provider.clone())
            .collect()
    }

    pub fn provider_infos(&self) -> Vec<ProviderInfo> {
        self.adapters
            .lock()
            .unwrap()
            .iter()
            .map(|e| {
                e.adapter.info().unwrap_or_else(|| ProviderInfo {
                    id: e.provider.clone(),
                    name: e.provider.clone(),
                    protocol: "custom".to_string(),
                    models: Vec::new(),
                })
            })
            .collect()
    }

    fn resolve_adapter(&self, provider: Option<&str>) -> Option<Arc<dyn LlmAdapter>> {
        let pinned_default = if provider.is_none() {
            self.default_provider.lock().unwrap().clone()
        } else {
            None
        };
        let list = self.adapters.lock().unwrap();
        if let Some(target) = provider {
            return list
                .iter()
                .find(|e| e.provider == target)
                .map(|e| Arc::clone(&e.adapter));
        }
        if let Some(ref default_id) = pinned_default
            && let Some(found) = list.iter().find(|e| &e.provider == default_id)
        {
            return Some(Arc::clone(&found.adapter));
        }
        list.first().map(|e| Arc::clone(&e.adapter))
    }

    pub fn discover_models(
        &self,
        provider: Option<&str>,
    ) -> BoxFuture<'static, Result<Vec<ModelInfo>, LlmError>> {
        let Some(adapter) = self.resolve_adapter(provider) else {
            return Box::pin(async { Err(LlmError::no_adapter()) });
        };
        adapter.discover_models()
    }

    /// 直接调用匹配的适配器流（绕过 `llm/stream` waterfall，供重试等中间件使用）。
    pub fn stream_direct(&self, options: GenerateOptions) -> Result<BoxChunkStream, LlmError> {
        let adapter = self
            .resolve_adapter(options.provider.as_deref())
            .ok_or_else(LlmError::no_adapter)?;
        Ok(adapter.stream(options))
    }

    /// 流式调用。外层包了一道 `waterfall("llm/stream")`：
    /// 其他插件可以在这里加重试、日志或改写请求，无需修改核心代码。
    pub fn stream(&self, options: GenerateOptions) -> Result<BoxChunkStream, LlmError> {
        let adapter = self
            .resolve_adapter(options.provider.as_deref())
            .ok_or_else(LlmError::no_adapter)?;

        Ok(self
            .ctx
            .waterfall(EVENT_LLM_STREAM, options, move |opts| adapter.stream(opts)))
    }
}

impl Service for LlmRuntime {
    type Config = ();

    fn service_name() -> &'static str {
        "llm"
    }

    fn plugin_name() -> &'static str {
        "LlmRuntime"
    }

    fn create(ctx: &Context, _config: Self::Config) -> Result<Arc<Self>, CordisError> {
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            next_id: AtomicU64::new(1),
            adapters: Mutex::new(Vec::new()),
            default_provider: Mutex::new(None),
        }))
    }
}

/// 为 `cordis::Context` 补充 `ctx.llm()` 扩展方法（对应 TS `declare module`）。
pub trait LlmContextExt {
    fn llm(&self) -> Result<Arc<LlmRuntime>, CordisError>;
}

impl LlmContextExt for Context {
    fn llm(&self) -> Result<Arc<LlmRuntime>, CordisError> {
        self.service::<LlmRuntime>("llm")
    }
}
