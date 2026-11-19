//! Context：插件看到的「世界」（对应 `packages/cordis/src/context.ts`）。
//!
//! 根 `Context` 创建内置的四个服务（`logger` / `fiber` / `reflect` / `registry` / `events`），
//! 每个插件拿到一个绑定了自己 `Fiber` 的子 `Context`。

use std::any::Any;
use std::future::Future;
use std::sync::Arc;

use crate::events::{AggregateError, EventOptions, EventsService, Next};
use crate::fiber::{Cleanup, CordisError, DisposableHandle, Fiber};
use crate::logger::Logger;
use crate::reflect::ReflectService;
use crate::registry::{Inject, IntoPluginReturn, Plugin, RegistryService};
use crate::utils::BoxError;

pub(crate) struct RootContext {
    pub(crate) logger: Logger,
    pub(crate) reflect: Arc<ReflectService>,
    pub(crate) registry: Arc<RegistryService>,
    pub(crate) events: Arc<EventsService>,
    pub(crate) root_fiber: Fiber,
}

/// 插件上下文：轻量可 `Clone`，每个插件持有指向自己 `Fiber` 的子 `Context`。
#[derive(Clone)]
pub struct Context {
    root: Arc<RootContext>,
    fiber: Fiber,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// 创建根 `Context`，初始化四个内置服务与根 `Fiber`。
    pub fn new() -> Self {
        let root = Arc::new_cyclic(|weak| RootContext {
            logger: Logger::default(),
            reflect: Arc::new(ReflectService::new(weak.clone())),
            registry: Arc::new(RegistryService::new()),
            events: Arc::new(EventsService::new()),
            root_fiber: Fiber::new_root(weak.clone()),
        });
        let fiber = root.root_fiber.clone();
        Self { root, fiber }
    }

    pub(crate) fn from_parts(root: Arc<RootContext>, fiber: Fiber) -> Self {
        Self { root, fiber }
    }

    pub(crate) fn root_arc(&self) -> Arc<RootContext> {
        Arc::clone(&self.root)
    }

    /// 根 `Context`。
    pub fn root(&self) -> Self {
        Self {
            root: Arc::clone(&self.root),
            fiber: self.root.root_fiber.clone(),
        }
    }

    /// 派生绑定指定 `Fiber` 的子 `Context`（对应 `ctx.extend({ fiber })`）。
    pub fn extend(&self, fiber: Fiber) -> Self {
        Self {
            root: Arc::clone(&self.root),
            fiber,
        }
    }

    pub fn fiber(&self) -> &Fiber {
        &self.fiber
    }

    pub fn events(&self) -> &Arc<EventsService> {
        &self.root.events
    }

    pub fn reflect(&self) -> &Arc<ReflectService> {
        &self.root.reflect
    }

    pub fn registry(&self) -> &Arc<RegistryService> {
        &self.root.registry
    }

    pub fn logger(&self) -> &Logger {
        &self.root.logger
    }

    /// 创建命名子日志器（对应 JS 版 `ctx.logger('name')`）。
    pub fn logger_named(&self, name: impl Into<String>) -> Logger {
        self.root.logger.child(name)
    }

    /// 挂载插件（若当前 Context 未激活则返回 `Err(CordisError::InactiveEffect)`）。
    pub fn try_plugin<P: Plugin>(
        &self,
        plugin: P,
        config: P::Config,
    ) -> Result<Fiber, CordisError> {
        self.root.registry.plugin(self, plugin, config)
    }

    /// 挂载插件（对应 `ctx.plugin(plugin, config)`）。
    pub fn plugin<P: Plugin>(&self, plugin: P, config: P::Config) -> Fiber {
        self.try_plugin(plugin, config)
            .expect("cannot mount plugin on inactive context")
    }

    /// 声明依赖并在依赖到齐后执行 `callback`（匿名子插件）。
    pub fn try_inject<F, Fut, R>(
        &self,
        deps: impl Into<Inject>,
        callback: F,
    ) -> Result<Fiber, CordisError>
    where
        F: Fn(Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoPluginReturn + 'static,
    {
        self.root.registry.inject(self, deps, callback)
    }

    /// 声明依赖并在依赖到齐后执行 `callback`（对应 `ctx.inject(deps, cb)`）。
    pub fn inject<F, Fut, R>(&self, deps: impl Into<Inject>, callback: F) -> Fiber
    where
        F: Fn(Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoPluginReturn + 'static,
    {
        self.try_inject(deps, callback)
            .expect("cannot register inject on inactive context")
    }

    /// 提供服务（以 `Arc<T>` 形式注册，卸载时自动下线）。
    pub fn provide<T: Send + Sync + 'static>(
        &self,
        name: impl Into<String>,
        value: Arc<T>,
    ) -> Result<DisposableHandle, CordisError> {
        self.root.reflect.provide(self, name, value)
    }

    /// 提供值类型服务（自动包裹为 `Arc::new(value)`）。
    pub fn provide_value<T: Send + Sync + 'static>(
        &self,
        name: impl Into<String>,
        value: T,
    ) -> Result<DisposableHandle, CordisError> {
        self.provide(name, Arc::new(value))
    }

    /// 按名字取服务（无需 `inject`，取不到或未激活返回 `None`）。适合可选依赖。
    pub fn get<T: Send + Sync + 'static>(&self, name: &str) -> Option<Arc<T>> {
        self.root.reflect.get(name)
    }

    /// 按 `ctx.xxx` 规则读取已注入的服务：
    /// 没声明 `inject` 返回 `MissingInject`，尚未就绪返回 `InactiveService`。
    pub fn service<T: Send + Sync + 'static>(&self, name: &str) -> Result<Arc<T>, CordisError> {
        self.root.reflect.resolve_for_ctx(self, name)
    }

    /// 注册同步副作用。
    pub fn effect<E, F>(
        &self,
        label: impl Into<String>,
        execute: F,
    ) -> Result<DisposableHandle, CordisError>
    where
        E: Into<BoxError>,
        F: FnOnce() -> Result<Option<Cleanup>, E>,
    {
        self.fiber.effect(label, execute)
    }

    /// 注册异步副作用。
    pub fn effect_async<E, Fut, F>(
        &self,
        label: impl Into<String>,
        execute: F,
    ) -> Result<DisposableHandle, CordisError>
    where
        E: Into<BoxError>,
        Fut: Future<Output = Result<Option<Cleanup>, E>> + Send + 'static,
        F: FnOnce() -> Fut,
    {
        self.fiber.effect_async(label, execute)
    }

    pub fn on<P, F>(
        &self,
        name: impl Into<String>,
        listener: F,
    ) -> Result<DisposableHandle, CordisError>
    where
        P: 'static,
        F: Fn(&P) + Send + Sync + 'static,
    {
        self.root
            .events
            .on(self, name, listener, EventOptions::default())
    }

    pub fn on_with<P, F>(
        &self,
        name: impl Into<String>,
        listener: F,
        options: EventOptions,
    ) -> Result<DisposableHandle, CordisError>
    where
        P: 'static,
        F: Fn(&P) + Send + Sync + 'static,
    {
        self.root.events.on(self, name, listener, options)
    }

    pub fn on_signal<F>(
        &self,
        name: impl Into<String>,
        listener: F,
    ) -> Result<DisposableHandle, CordisError>
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.root
            .events
            .on_signal(self, name, listener, EventOptions::default())
    }

    pub fn on_bail<P, R, F>(
        &self,
        name: impl Into<String>,
        listener: F,
    ) -> Result<DisposableHandle, CordisError>
    where
        P: 'static,
        R: Send + 'static,
        F: Fn(&P) -> Option<R> + Send + Sync + 'static,
    {
        self.root
            .events
            .on_bail(self, name, listener, EventOptions::default())
    }

    pub fn on_serial<P, R, Fut, F>(
        &self,
        name: impl Into<String>,
        listener: F,
    ) -> Result<DisposableHandle, CordisError>
    where
        P: Clone + Send + Sync + 'static,
        R: Send + 'static,
        Fut: Future<Output = Option<R>> + Send + 'static,
        F: Fn(P) -> Fut + Send + Sync + 'static,
    {
        self.root
            .events
            .on_serial(self, name, listener, EventOptions::default())
    }

    pub fn on_parallel<P, E, Fut, F>(
        &self,
        name: impl Into<String>,
        listener: F,
    ) -> Result<DisposableHandle, CordisError>
    where
        P: Clone + Send + Sync + 'static,
        E: Into<BoxError>,
        Fut: Future<Output = Result<(), E>> + Send + 'static,
        F: Fn(P) -> Fut + Send + Sync + 'static,
    {
        self.root
            .events
            .on_parallel(self, name, listener, EventOptions::default())
    }

    pub fn on_waterfall<I, O, F>(
        &self,
        name: impl Into<String>,
        listener: F,
    ) -> Result<DisposableHandle, CordisError>
    where
        I: Send + 'static,
        O: Send + 'static,
        F: Fn(I, Next<I, O>) -> O + Send + Sync + 'static,
    {
        self.root
            .events
            .on_waterfall(self, name, listener, EventOptions::default())
    }

    pub fn on_waterfall_with<I, O, F>(
        &self,
        name: impl Into<String>,
        listener: F,
        options: EventOptions,
    ) -> Result<DisposableHandle, CordisError>
    where
        I: Send + 'static,
        O: Send + 'static,
        F: Fn(I, Next<I, O>) -> O + Send + Sync + 'static,
    {
        self.root.events.on_waterfall(self, name, listener, options)
    }

    pub fn emit<P: Any>(&self, name: &str, payload: P) {
        self.root.events.emit(name, payload);
    }

    pub fn bail<P: Any, R: 'static>(&self, name: &str, payload: P) -> Option<R> {
        self.root.events.bail(name, payload)
    }

    pub async fn serial<P: Any, R: 'static>(&self, name: &str, payload: P) -> Option<R> {
        self.root.events.serial(name, payload).await
    }

    pub async fn parallel<P: Any>(&self, name: &str, payload: P) -> Result<(), AggregateError> {
        self.root.events.parallel(name, payload).await
    }

    pub fn waterfall<I, O, F>(&self, name: &str, input: I, inner: F) -> O
    where
        I: Send + 'static,
        O: Send + 'static,
        F: FnOnce(I) -> O + Send + 'static,
    {
        self.root.events.waterfall(name, input, inner)
    }
}
