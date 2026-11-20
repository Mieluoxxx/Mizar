//! 事件总线（对应 `packages/cordis/src/events.ts`）。
//!
//! 保留全部五种触发方式：`emit` / `parallel` / `serial` / `bail` / `waterfall`。
//! 所有通过 `ctx.on*` 注册的监听都会作为 Fiber 的 effect 记账，插件卸载时自动摘除。

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::future::join_all;

use crate::context::Context;
use crate::fiber::{CordisError, DisposableHandle, Fiber, FiberState};
use crate::utils::{BoxError, BoxFuture};

pub const INTERNAL_PLUGIN: &str = "internal/plugin";
pub const INTERNAL_STATUS: &str = "internal/status";
pub const INTERNAL_SERVICE: &str = "internal/service";
pub const INTERNAL_DISPATCH: &str = "internal/dispatch";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchMode {
    Emit,
    Parallel,
    Serial,
    Bail,
    Waterfall,
}

/// `internal/plugin` 事件载荷：插件挂载或卸载时发出。
#[derive(Clone)]
pub struct InternalPluginEvent {
    pub fiber: Fiber,
}

/// `internal/status` 事件载荷：插件 Fiber 状态变化时发出。
#[derive(Clone)]
pub struct InternalStatusEvent {
    pub fiber: Fiber,
    pub old_state: FiberState,
    pub new_state: FiberState,
}

/// `internal/service` 事件载荷：服务上线或下线时发出。
#[derive(Debug, Clone)]
pub struct InternalServiceEvent {
    pub name: String,
    pub available: bool,
}

/// `internal/dispatch` 事件载荷：任何非 `internal/` 事件被触发时发出。
#[derive(Debug, Clone)]
pub struct InternalDispatchEvent {
    pub mode: DispatchMode,
    pub name: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EventOptions {
    /// 插到同优先级区段之首，而不是之尾。
    pub prepend: bool,
    /// 调度优先级（数值越小越先执行，默认 `0`）。
    pub priority: i32,
}

/// `parallel` 模式下收集所有失败 listener 错误的聚合错误。
#[derive(Debug)]
pub struct AggregateError {
    pub errors: Vec<BoxError>,
}

impl fmt::Display for AggregateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AggregateError({} errors):", self.errors.len())?;
        for err in &self.errors {
            write!(f, "\n  - {err}")?;
        }
        Ok(())
    }
}

impl std::error::Error for AggregateError {}

/// 洋葱模型（`waterfall`）的下一层调用句柄。
pub struct Next<I, O> {
    func: Box<dyn FnOnce(I) -> O + Send>,
}

impl<I, O> Next<I, O> {
    pub fn new(func: impl FnOnce(I) -> O + Send + 'static) -> Self {
        Self {
            func: Box::new(func),
        }
    }

    /// 放行并把（可能改写后的）输入传给下一层。
    pub fn call(self, input: I) -> O {
        (self.func)(input)
    }
}

type DynAnySend = Box<dyn Any + Send>;
type DynNextFn = Box<dyn FnOnce(DynAnySend) -> DynAnySend + Send>;

type SyncEmitFn = Arc<dyn Fn(&dyn Any) + Send + Sync>;
type SyncSignalFn = Arc<dyn Fn() + Send + Sync>;
type SyncBailFn = Arc<dyn Fn(&dyn Any) -> Option<DynAnySend> + Send + Sync>;
type AsyncSerialFn = Arc<dyn Fn(&dyn Any) -> BoxFuture<'static, Option<DynAnySend>> + Send + Sync>;
type AsyncParallelFn =
    Arc<dyn Fn(&dyn Any) -> BoxFuture<'static, Result<(), BoxError>> + Send + Sync>;
type WaterfallFn = Arc<dyn Fn(DynAnySend, DynNextFn) -> DynAnySend + Send + Sync>;

#[derive(Clone)]
pub(crate) enum HookKind {
    Emit(SyncEmitFn),
    EmitSignal(SyncSignalFn),
    Bail(SyncBailFn),
    Serial(AsyncSerialFn),
    Parallel(AsyncParallelFn),
    Waterfall(WaterfallFn),
}

#[derive(Clone)]
struct HookEntry {
    id: u64,
    priority: i32,
    kind: HookKind,
}

pub struct EventsService {
    next_id: AtomicU64,
    hooks: Mutex<HashMap<String, Vec<HookEntry>>>,
}

impl Default for EventsService {
    fn default() -> Self {
        Self::new()
    }
}

impl EventsService {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            hooks: Mutex::new(HashMap::new()),
        }
    }

    /// 取出某个事件的当前监听列表快照，并按需触发 `internal/dispatch`。
    /// 锁在返回前立即释放，避免监听函数内部再次操作事件总线时死锁。
    fn dispatch(&self, mode: DispatchMode, name: &str) -> Vec<HookKind> {
        if !name.starts_with("internal/") {
            self.emit(
                INTERNAL_DISPATCH,
                InternalDispatchEvent {
                    mode,
                    name: name.to_string(),
                },
            );
        }
        let hooks = self.hooks.lock().unwrap();
        hooks
            .get(name)
            .map(|list| list.iter().map(|entry| entry.kind.clone()).collect())
            .unwrap_or_default()
    }

    /// 广播：同步调用所有监听者，不管返回值。
    pub fn emit<P: Any>(&self, name: &str, payload: P) {
        let any: &dyn Any = &payload;
        for kind in self.dispatch(DispatchMode::Emit, name) {
            match kind {
                HookKind::Emit(cb) => cb(any),
                HookKind::EmitSignal(cb) => cb(),
                _ => {}
            }
        }
    }

    /// 同步版 `serial`：按顺序调用，谁返回 `Some(result)` 就停下并返回它。
    pub fn bail<P: Any, R: 'static>(&self, name: &str, payload: P) -> Option<R> {
        let any: &dyn Any = &payload;
        for kind in self.dispatch(DispatchMode::Bail, name) {
            if let HookKind::Bail(cb) = kind
                && let Some(boxed) = cb(any)
            {
                let res = *boxed
                    .downcast::<R>()
                    .expect("bail listener return type mismatch");
                return Some(res);
            }
        }
        None
    }

    /// 串行异步：逐个 await，谁返回 `Some(result)` 就停下并返回它。
    pub async fn serial<P: Any, R: 'static>(&self, name: &str, payload: P) -> Option<R> {
        let futs: Vec<_> = {
            let any: &dyn Any = &payload;
            self.dispatch(DispatchMode::Serial, name)
                .into_iter()
                .filter_map(|kind| match kind {
                    HookKind::Serial(cb) => Some(cb(any)),
                    _ => None,
                })
                .collect()
        };
        for fut in futs {
            if let Some(boxed) = fut.await {
                let res = *boxed
                    .downcast::<R>()
                    .expect("serial listener return type mismatch");
                return Some(res);
            }
        }
        None
    }

    /// 并行异步：所有监听者同时跑，全部结束后若有失败则聚合抛出 `AggregateError`。
    pub async fn parallel<P: Any>(&self, name: &str, payload: P) -> Result<(), AggregateError> {
        let futs: Vec<_> = {
            let any: &dyn Any = &payload;
            self.dispatch(DispatchMode::Parallel, name)
                .into_iter()
                .filter_map(|kind| match kind {
                    HookKind::Parallel(cb) => Some(cb(any)),
                    _ => None,
                })
                .collect()
        };
        let results = join_all(futs).await;
        let errors: Vec<BoxError> = results.into_iter().filter_map(Result::err).collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(AggregateError { errors })
        }
    }

    /// 瀑布（洋葱模型）：`inner` 是最里层的默认实现。
    /// 每个监听者拿到 `(input, next)`：调用 `next.call(input)` 往里传，不调用即拦截。
    pub fn waterfall<I, O, F>(&self, name: &str, input: I, inner: F) -> O
    where
        I: Send + 'static,
        O: Send + 'static,
        F: FnOnce(I) -> O + Send + 'static,
    {
        let hooks: Vec<WaterfallFn> = self
            .dispatch(DispatchMode::Waterfall, name)
            .into_iter()
            .filter_map(|kind| match kind {
                HookKind::Waterfall(cb) => Some(cb),
                _ => None,
            })
            .collect();

        if hooks.is_empty() {
            return inner(input);
        }

        let event_name = name.to_string();
        let event_name_in = event_name.clone();
        let mut current: DynNextFn = Box::new(move |boxed: DynAnySend| {
            let typed = *boxed.downcast::<I>().unwrap_or_else(|_| {
                panic!(
                    "waterfall input type mismatch on \"{event_name_in}\": expected {}",
                    std::any::type_name::<I>()
                )
            });
            Box::new(inner(typed))
        });

        for cb in hooks.into_iter().rev() {
            let next = current;
            current = Box::new(move |boxed: DynAnySend| cb(boxed, next));
        }

        let out = current(Box::new(input));
        *out.downcast::<O>().unwrap_or_else(|_| {
            panic!(
                "waterfall output type mismatch on \"{event_name}\": expected {}",
                std::any::type_name::<O>()
            )
        })
    }

    /// 注册监听（通过 `ctx.fiber().effect(...)` 记账，插件卸载时自动取消）。
    pub(crate) fn register_hook(
        self: &Arc<Self>,
        ctx: &Context,
        name: impl Into<String>,
        options: EventOptions,
        kind: HookKind,
    ) -> Result<DisposableHandle, CordisError> {
        let name = name.into();
        let events = Arc::clone(self);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let label = format!("ctx.on({name:?})");

        ctx.fiber().effect(&label, move || {
            {
                let mut hooks = events.hooks.lock().unwrap();
                let list = hooks.entry(name.clone()).or_default();
                let entry = HookEntry {
                    id,
                    priority: options.priority,
                    kind,
                };
                let insert_idx = if options.prepend {
                    list.iter()
                        .position(|e| e.priority >= options.priority)
                        .unwrap_or(list.len())
                } else {
                    list.iter()
                        .position(|e| e.priority > options.priority)
                        .unwrap_or(list.len())
                };
                list.insert(insert_idx, entry);
            }
            let events_cleanup = Arc::clone(&events);
            Ok::<_, CordisError>(Some(crate::fiber::Cleanup::sync(move || {
                let mut hooks = events_cleanup.hooks.lock().unwrap();
                if let Some(list) = hooks.get_mut(&name) {
                    if let Some(idx) = list.iter().position(|e| e.id == id) {
                        list.remove(idx);
                    }
                    if list.is_empty() {
                        hooks.remove(&name);
                    }
                }
            })))
        })
    }

    pub fn on<P, F>(
        self: &Arc<Self>,
        ctx: &Context,
        name: impl Into<String>,
        listener: F,
        options: EventOptions,
    ) -> Result<DisposableHandle, CordisError>
    where
        P: 'static,
        F: Fn(&P) + Send + Sync + 'static,
    {
        let cb: SyncEmitFn = Arc::new(move |any: &dyn Any| {
            if let Some(payload) = any.downcast_ref::<P>() {
                listener(payload);
            }
        });
        self.register_hook(ctx, name, options, HookKind::Emit(cb))
    }

    pub fn on_signal<F>(
        self: &Arc<Self>,
        ctx: &Context,
        name: impl Into<String>,
        listener: F,
        options: EventOptions,
    ) -> Result<DisposableHandle, CordisError>
    where
        F: Fn() + Send + Sync + 'static,
    {
        let cb: SyncSignalFn = Arc::new(listener);
        self.register_hook(ctx, name, options, HookKind::EmitSignal(cb))
    }

    pub fn on_bail<P, R, F>(
        self: &Arc<Self>,
        ctx: &Context,
        name: impl Into<String>,
        listener: F,
        options: EventOptions,
    ) -> Result<DisposableHandle, CordisError>
    where
        P: 'static,
        R: Send + 'static,
        F: Fn(&P) -> Option<R> + Send + Sync + 'static,
    {
        let cb: SyncBailFn = Arc::new(move |any: &dyn Any| {
            let payload = any.downcast_ref::<P>()?;
            listener(payload).map(|r| Box::new(r) as DynAnySend)
        });
        self.register_hook(ctx, name, options, HookKind::Bail(cb))
    }

    pub fn on_serial<P, R, Fut, F>(
        self: &Arc<Self>,
        ctx: &Context,
        name: impl Into<String>,
        listener: F,
        options: EventOptions,
    ) -> Result<DisposableHandle, CordisError>
    where
        P: Clone + Send + Sync + 'static,
        R: Send + 'static,
        Fut: std::future::Future<Output = Option<R>> + Send + 'static,
        F: Fn(P) -> Fut + Send + Sync + 'static,
    {
        let cb: AsyncSerialFn = Arc::new(move |any: &dyn Any| {
            if let Some(payload) = any.downcast_ref::<P>() {
                let fut = listener(payload.clone());
                Box::pin(async move { fut.await.map(|r| Box::new(r) as DynAnySend) })
            } else {
                Box::pin(async { None })
            }
        });
        self.register_hook(ctx, name, options, HookKind::Serial(cb))
    }

    pub fn on_parallel<P, E, Fut, F>(
        self: &Arc<Self>,
        ctx: &Context,
        name: impl Into<String>,
        listener: F,
        options: EventOptions,
    ) -> Result<DisposableHandle, CordisError>
    where
        P: Clone + Send + Sync + 'static,
        E: Into<BoxError>,
        Fut: std::future::Future<Output = Result<(), E>> + Send + 'static,
        F: Fn(P) -> Fut + Send + Sync + 'static,
    {
        let cb: AsyncParallelFn = Arc::new(move |any: &dyn Any| {
            if let Some(payload) = any.downcast_ref::<P>() {
                let fut = listener(payload.clone());
                Box::pin(async move { fut.await.map_err(Into::into) })
            } else {
                Box::pin(async { Ok(()) })
            }
        });
        self.register_hook(ctx, name, options, HookKind::Parallel(cb))
    }

    pub fn on_waterfall<I, O, F>(
        self: &Arc<Self>,
        ctx: &Context,
        name: impl Into<String>,
        listener: F,
        options: EventOptions,
    ) -> Result<DisposableHandle, CordisError>
    where
        I: Send + 'static,
        O: Send + 'static,
        F: Fn(I, Next<I, O>) -> O + Send + Sync + 'static,
    {
        let name_str = name.into();
        let event_name = name_str.clone();
        let cb: WaterfallFn = Arc::new(move |boxed: DynAnySend, next_fn: DynNextFn| {
            let typed_in = *boxed.downcast::<I>().unwrap_or_else(|_| {
                panic!(
                    "waterfall listener input type mismatch on \"{event_name}\": expected {}",
                    std::any::type_name::<I>()
                )
            });
            let event_name_inner = event_name.clone();
            let typed_next = Next::new(move |i: I| {
                let out_any = next_fn(Box::new(i));
                *out_any.downcast::<O>().unwrap_or_else(|_| {
                    panic!(
                        "waterfall listener output type mismatch on \"{event_name_inner}\": expected {}",
                        std::any::type_name::<O>()
                    )
                })
            });
            let out = listener(typed_in, typed_next);
            Box::new(out)
        });
        self.register_hook(ctx, name_str, options, HookKind::Waterfall(cb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[tokio::test]
    async fn serial_hooks_respect_priority_across_reregistration_and_prepend() {
        let ctx = Context::new();
        let trace = Arc::new(StdMutex::new(Vec::new()));

        let push = |tag: &'static str, trace: &Arc<StdMutex<Vec<&'static str>>>| {
            let t = Arc::clone(trace);
            move |_p: ()| {
                let t = Arc::clone(&t);
                async move {
                    t.lock().unwrap().push(tag);
                    None::<()>
                }
            }
        };

        // 即使低优先级(20)先注册、高优先级(-20)后注册，仍按 priority 从小到大执行
        let _h_recall = ctx
            .on_serial_with(
                "pre-step",
                push("recall:20", &trace),
                EventOptions {
                    priority: 20,
                    ..Default::default()
                },
            )
            .unwrap();
        let _h_mid1 = ctx.on_serial("pre-step", push("mid1:0", &trace)).unwrap();
        let h_compact = ctx
            .on_serial_with(
                "pre-step",
                push("compact:-20", &trace),
                EventOptions {
                    priority: -20,
                    ..Default::default()
                },
            )
            .unwrap();
        let _h_mid_prepend = ctx
            .on_serial_with(
                "pre-step",
                push("mid-prepend:0", &trace),
                EventOptions {
                    prepend: true,
                    priority: 0,
                },
            )
            .unwrap();

        let _: Option<()> = ctx.events().serial("pre-step", ()).await;
        assert_eq!(
            *trace.lock().unwrap(),
            vec!["compact:-20", "mid-prepend:0", "mid1:0", "recall:20"]
        );

        // 卸载后重新挂载高优先级钩子（模拟插件热重启），仍排在最前
        h_compact.dispose().await;
        let _h_compact2 = ctx
            .on_serial_with(
                "pre-step",
                push("compact2:-20", &trace),
                EventOptions {
                    priority: -20,
                    ..Default::default()
                },
            )
            .unwrap();
        trace.lock().unwrap().clear();
        let _: Option<()> = ctx.events().serial("pre-step", ()).await;
        assert_eq!(
            *trace.lock().unwrap(),
            vec!["compact2:-20", "mid-prepend:0", "mid1:0", "recall:20"]
        );

        ctx.fiber().dispose().await;
    }
}
