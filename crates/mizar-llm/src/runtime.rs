use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use cordis::{BoxFuture, Cleanup, Context, CordisError, Service};
use futures::Stream;

use crate::types::{
    FinishReason, GenerateOptions, ModelInfo, ProviderInfo, StreamChunk, TokenUsage,
};
use mizar_types::llm::LlmError;

pub const EVENT_ADAPTERS_UPDATED: &str = "llm/adapters-updated";
pub const EVENT_LLM_STREAM: &str = "llm/stream";
pub const EVENT_LLM_AFTER_RESPONSE: &str = "llm/after-response";

pub type BoxChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>;

/// 模型单次流正常结束时的只读回执事件（由 `LlmRuntime::stream` 内核统一发出）。
#[derive(Debug, Clone)]
pub struct LlmAfterResponseEvent {
    pub request: Arc<GenerateOptions>,
    pub provider: String,
    pub model: String,
    pub finish_reason: FinishReason,
    pub usage: Option<TokenUsage>,
}

/// 模型适配器：把统一的 `GenerateOptions` 翻译成具体模型协议的流式输出。
pub trait LlmAdapter: Send + Sync + 'static {
    fn stream(&self, options: GenerateOptions) -> BoxChunkStream;

    /// 在冻结只读快照与进入 `llm/stream` 洋葱链前对请求参数执行幂等规整。
    fn normalize_options(&self, options: GenerateOptions) -> GenerateOptions {
        options
    }

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

    fn resolve_adapter_entry(
        &self,
        provider: Option<&str>,
    ) -> Option<(String, Arc<dyn LlmAdapter>)> {
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
                .map(|e| (e.provider.clone(), Arc::clone(&e.adapter)));
        }
        if let Some(ref default_id) = pinned_default
            && let Some(found) = list.iter().find(|e| &e.provider == default_id)
        {
            return Some((found.provider.clone(), Arc::clone(&found.adapter)));
        }
        list.first()
            .map(|e| (e.provider.clone(), Arc::clone(&e.adapter)))
    }

    pub fn discover_models(
        &self,
        provider: Option<&str>,
    ) -> BoxFuture<'static, Result<Vec<ModelInfo>, LlmError>> {
        let Some((_, adapter)) = self.resolve_adapter_entry(provider) else {
            return Box::pin(async { Err(LlmError::no_adapter()) });
        };
        adapter.discover_models()
    }

    /// 直接调用匹配的适配器流（绕过 `llm/stream` waterfall 与 `llm/after-response`，供重试与缓存预热使用）。
    pub fn stream_direct(&self, options: GenerateOptions) -> Result<BoxChunkStream, LlmError> {
        let (_, adapter) = self
            .resolve_adapter_entry(options.provider.as_deref())
            .ok_or_else(LlmError::no_adapter)?;
        let normalized = adapter.normalize_options(options);
        Ok(adapter.stream(normalized))
    }

    /// 流式调用。先经适配器幂等规整并冻结快照，再走 `waterfall("llm/stream")` 并在流正常收尾时发出 `llm/after-response`。
    pub fn stream(&self, options: GenerateOptions) -> Result<BoxChunkStream, LlmError> {
        let (resolved_provider, adapter) = self
            .resolve_adapter_entry(options.provider.as_deref())
            .ok_or_else(LlmError::no_adapter)?;

        let normalized = adapter.normalize_options(options);
        let snapshot = Arc::new(normalized.clone());
        let inner_adapter = Arc::clone(&adapter);
        let stream = self
            .ctx
            .waterfall(EVENT_LLM_STREAM, normalized, move |opts| {
                inner_adapter.stream(opts)
            });

        Ok(Box::pin(AfterResponseStream {
            inner: stream,
            ctx: self.ctx.clone(),
            snapshot,
            provider: resolved_provider,
            latest_usage: None,
            saw_error: false,
            emitted: false,
        }))
    }
}

struct AfterResponseStream {
    inner: BoxChunkStream,
    ctx: Context,
    snapshot: Arc<GenerateOptions>,
    provider: String,
    latest_usage: Option<TokenUsage>,
    saw_error: bool,
    emitted: bool,
}

impl Stream for AfterResponseStream {
    type Item = Result<StreamChunk, LlmError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.inner.as_mut().poll_next(cx);
        if let Poll::Ready(Some(ref item)) = poll {
            match item {
                Ok(StreamChunk::Usage { usage }) => {
                    self.latest_usage = Some(*usage);
                }
                Ok(StreamChunk::Finish { reason, replay }) => {
                    let is_clean_finish =
                        !matches!(reason, FinishReason::Aborted | FinishReason::Error { .. });
                    if is_clean_finish
                        && !self.saw_error
                        && !self.emitted
                        && !self.snapshot.cancel.is_cancelled()
                    {
                        self.emitted = true;
                        let provider = self
                            .snapshot
                            .provider
                            .clone()
                            .unwrap_or_else(|| self.provider.clone());
                        let model = self
                            .snapshot
                            .model
                            .clone()
                            .or_else(|| replay.as_ref().map(|r| r.model.clone()))
                            .unwrap_or_default();
                        self.ctx.emit(
                            EVENT_LLM_AFTER_RESPONSE,
                            LlmAfterResponseEvent {
                                request: Arc::clone(&self.snapshot),
                                provider,
                                model,
                                finish_reason: reason.clone(),
                                usage: self.latest_usage,
                            },
                        );
                    }
                }
                Err(_) => {
                    self.saw_error = true;
                }
                Ok(_) => {}
            }
        }
        poll
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, RequestOrigin};
    use cordis::Next;
    use futures::{StreamExt, executor::block_on, stream};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    enum FakeOutcome {
        Clean {
            finish: FinishReason,
            usage: Option<TokenUsage>,
        },
        TransportErr,
        ErrorChunk,
        AbortedChunk,
        DoubleFinish,
    }

    struct FakeAdapter {
        default_model: Option<String>,
        outcomes: Mutex<Vec<FakeOutcome>>,
        seen_options: Mutex<Vec<GenerateOptions>>,
        calls: AtomicUsize,
    }

    impl FakeAdapter {
        fn new(default_model: Option<&str>, outcomes: Vec<FakeOutcome>) -> Self {
            Self {
                default_model: default_model.map(ToString::to_string),
                outcomes: Mutex::new(outcomes),
                seen_options: Mutex::new(Vec::new()),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl LlmAdapter for FakeAdapter {
        fn normalize_options(&self, mut options: GenerateOptions) -> GenerateOptions {
            if options.model.is_none() {
                options.model = self.default_model.clone();
            }
            if !options
                .headers
                .iter()
                .any(|(k, _)| k.eq_ignore_ascii_case("x-norm"))
            {
                options
                    .headers
                    .push(("x-norm".to_string(), "1".to_string()));
            }
            options
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let idx = self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen_options.lock().unwrap().push(options);
            let outcome = {
                let guard = self.outcomes.lock().unwrap();
                guard
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| guard.last().cloned().unwrap())
            };
            let chunks: Vec<Result<StreamChunk, LlmError>> = match outcome {
                FakeOutcome::Clean { finish, usage } => {
                    let mut out = vec![Ok(StreamChunk::text("ok"))];
                    if let Some(u) = usage {
                        out.push(Ok(StreamChunk::Usage { usage: u }));
                    }
                    out.push(Ok(StreamChunk::Finish {
                        reason: finish,
                        replay: None,
                    }));
                    out
                }
                FakeOutcome::TransportErr => vec![
                    Err(LlmError::transport("boom")),
                    Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }),
                ],
                FakeOutcome::ErrorChunk => vec![Ok(StreamChunk::Finish {
                    reason: FinishReason::Error {
                        code: mizar_types::llm::LlmErrorCode::Server,
                        message: "bad".into(),
                        status: Some(500),
                        retry_after_ms: None,
                    },
                    replay: None,
                })],
                FakeOutcome::AbortedChunk => vec![Ok(StreamChunk::Finish {
                    reason: FinishReason::Aborted,
                    replay: None,
                })],
                FakeOutcome::DoubleFinish => vec![
                    Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }),
                    Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }),
                ],
            };
            Box::pin(stream::iter(chunks))
        }
    }

    #[test]
    fn stream_normalizes_before_snapshot_and_emits_after_response_once() {
        block_on(async {
            let app = Context::new();
            let llm = LlmRuntime::create(&app, ()).unwrap();

            let adapter = Arc::new(FakeAdapter::new(
                Some("norm-model"),
                vec![FakeOutcome::Clean {
                    finish: FinishReason::Stop,
                    usage: Some(TokenUsage {
                        input_tokens: 120,
                        output_tokens: 15,
                        cache_read_tokens: Some(800),
                        ..Default::default()
                    }),
                }],
            ));
            let _reg = llm
                .register_adapter("p1", Arc::clone(&adapter) as Arc<dyn LlmAdapter>)
                .unwrap();

            let waterfall_seen_model = Arc::new(Mutex::new(None::<Option<String>>));
            let wf_cap = Arc::clone(&waterfall_seen_model);
            app.on_waterfall(
                EVENT_LLM_STREAM,
                move |opts: GenerateOptions, next: Next<GenerateOptions, BoxChunkStream>| {
                    *wf_cap.lock().unwrap() = Some(opts.model.clone());
                    next.call(opts)
                },
            )
            .unwrap();

            let events = Arc::new(Mutex::new(Vec::<LlmAfterResponseEvent>::new()));
            let ev_cap = Arc::clone(&events);
            app.on(
                EVENT_LLM_AFTER_RESPONSE,
                move |ev: &LlmAfterResponseEvent| {
                    ev_cap.lock().unwrap().push(ev.clone());
                },
            )
            .unwrap();

            let mut s = llm
                .stream(GenerateOptions {
                    messages: vec![Message::user("hi")],
                    origin: RequestOrigin::SessionTurn {
                        session_id: "s1".into(),
                        turn: 1,
                        step: 0,
                    },
                    ..Default::default()
                })
                .unwrap();
            while let Some(item) = s.next().await {
                let _ = item.unwrap();
            }

            // waterfall 必须看到已经经过 normalize_options 补齐的模型名
            assert_eq!(
                *waterfall_seen_model.lock().unwrap(),
                Some(Some("norm-model".into()))
            );
            let captured = events.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(captured[0].provider, "p1");
            assert_eq!(captured[0].model, "norm-model");
            assert_eq!(captured[0].finish_reason, FinishReason::Stop);
            assert_eq!(captured[0].usage.unwrap().prompt_tokens(), 920);
            assert_eq!(captured[0].request.model.as_deref(), Some("norm-model"));
            assert_eq!(
                captured[0].request.headers,
                vec![("x-norm".to_string(), "1".to_string())]
            );
            assert!(captured[0].request.origin.is_session_turn());
        });
    }

    #[test]
    fn after_response_skips_errors_aborts_cancel_stream_direct_and_duplicates() {
        block_on(async {
            let app = Context::new();
            let llm = LlmRuntime::create(&app, ()).unwrap();

            let adapter = Arc::new(FakeAdapter::new(
                Some("m-edge"),
                vec![
                    FakeOutcome::TransportErr,
                    FakeOutcome::ErrorChunk,
                    FakeOutcome::AbortedChunk,
                    FakeOutcome::Clean {
                        finish: FinishReason::Stop,
                        usage: None,
                    },
                    FakeOutcome::Clean {
                        finish: FinishReason::Stop,
                        usage: None,
                    },
                    FakeOutcome::DoubleFinish,
                ],
            ));
            let _reg = llm
                .register_adapter("p1", Arc::clone(&adapter) as Arc<dyn LlmAdapter>)
                .unwrap();

            let events = Arc::new(Mutex::new(Vec::<LlmAfterResponseEvent>::new()));
            let ev_cap = Arc::clone(&events);
            app.on(
                EVENT_LLM_AFTER_RESPONSE,
                move |ev: &LlmAfterResponseEvent| {
                    ev_cap.lock().unwrap().push(ev.clone());
                },
            )
            .unwrap();

            // 1. Err(_) 后即使出现 Finish 也不发出
            let mut s1 = llm.stream(GenerateOptions::default()).unwrap();
            while s1.next().await.is_some() {}
            assert_eq!(events.lock().unwrap().len(), 0);

            // 2. FinishReason::Error 不发出
            let mut s2 = llm.stream(GenerateOptions::default()).unwrap();
            while s2.next().await.is_some() {}
            assert_eq!(events.lock().unwrap().len(), 0);

            // 3. FinishReason::Aborted 不发出
            let mut s3 = llm.stream(GenerateOptions::default()).unwrap();
            while s3.next().await.is_some() {}
            assert_eq!(events.lock().unwrap().len(), 0);

            // 4. cancel 已被触发时不发出
            let cancel = tokio_util::sync::CancellationToken::new();
            cancel.cancel();
            let mut s4 = llm
                .stream(GenerateOptions {
                    cancel,
                    ..Default::default()
                })
                .unwrap();
            while s4.next().await.is_some() {}
            assert_eq!(events.lock().unwrap().len(), 0);

            // 5. stream_direct 不触发 llm/after-response，但仍应执行 normalize_options
            let mut s5 = llm.stream_direct(GenerateOptions::default()).unwrap();
            while s5.next().await.is_some() {}
            assert_eq!(events.lock().unwrap().len(), 0);
            assert_eq!(
                adapter.seen_options.lock().unwrap()[4].model.as_deref(),
                Some("m-edge")
            );

            // 6. 同一条流即使意外产出两个 Finish，也至多发出一次
            let mut s6 = llm.stream(GenerateOptions::default()).unwrap();
            while s6.next().await.is_some() {}
            assert_eq!(events.lock().unwrap().len(), 1);
        });
    }

    #[test]
    fn retry_via_stream_direct_emits_after_response_once_on_final_success() {
        block_on(async {
            let app = Context::new();
            let llm = LlmRuntime::create(&app, ()).unwrap();

            let adapter = Arc::new(FakeAdapter::new(
                Some("m-retry"),
                vec![
                    FakeOutcome::TransportErr,
                    FakeOutcome::Clean {
                        finish: FinishReason::ToolCalls,
                        usage: Some(TokenUsage {
                            input_tokens: 50,
                            output_tokens: 10,
                            ..Default::default()
                        }),
                    },
                ],
            ));
            let _reg = llm
                .register_adapter("p1", Arc::clone(&adapter) as Arc<dyn LlmAdapter>)
                .unwrap();

            let llm_for_retry = Arc::clone(&llm);
            app.on_waterfall(
                EVENT_LLM_STREAM,
                move |opts: GenerateOptions, next: Next<GenerateOptions, BoxChunkStream>| {
                    let first = next.call(opts.clone());
                    let llm_inner = Arc::clone(&llm_for_retry);
                    Box::pin(stream::once(async move {
                        let items: Vec<_> = first.collect().await;
                        if items.iter().any(Result::is_err) {
                            llm_inner
                                .stream_direct(opts)
                                .unwrap()
                                .collect::<Vec<_>>()
                                .await
                        } else {
                            items
                        }
                    }))
                    .flat_map(stream::iter)
                    .boxed()
                },
            )
            .unwrap();

            let events = Arc::new(Mutex::new(Vec::<LlmAfterResponseEvent>::new()));
            let ev_cap = Arc::clone(&events);
            app.on(
                EVENT_LLM_AFTER_RESPONSE,
                move |ev: &LlmAfterResponseEvent| {
                    ev_cap.lock().unwrap().push(ev.clone());
                },
            )
            .unwrap();

            let mut s = llm.stream(GenerateOptions::default()).unwrap();
            while let Some(item) = s.next().await {
                let _ = item.unwrap();
            }

            let captured = events.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(captured[0].finish_reason, FinishReason::ToolCalls);
            assert_eq!(captured[0].usage.unwrap().input_tokens, 50);
        });
    }
}
