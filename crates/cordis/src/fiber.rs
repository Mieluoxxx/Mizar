//! Fiber：一个插件实例（对应 `packages/cordis/src/fiber.ts`）。
//!
//! 负责状态机（6 种状态）、effect 记账与倒序撤销、异步加载/卸载串行化（inertia）、
//! 错误隔离，以及基于 epoch 的依赖变更自动重启。

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::future::{Future, IntoFuture};
use std::sync::{Arc, Mutex, Weak};

use futures::future::join_all;
use tokio::sync::watch;

use crate::context::{Context, RootContext};
use crate::events::{INTERNAL_PLUGIN, INTERNAL_STATUS, InternalPluginEvent, InternalStatusEvent};
use crate::reflect::ServiceImpl;
use crate::registry::{Inject, PluginRuntime, ValidationError};
use crate::utils::{BoxError, BoxFuture, DisposableList};

/// epoch 的特殊值：依赖不齐。
const INACTIVE: &str = "__INACTIVE__";

/// 六种状态，数值和真 Cordis 完全一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FiberState {
    /// 依赖没到齐，等着
    Pending = 0,
    /// 正在执行 apply（可能是异步的）
    Loading = 1,
    /// 运行中
    Active = 2,
    /// apply 出错了，被隔离，不影响别人
    Failed = 3,
    /// 已卸载
    Disposed = 4,
    /// 正在执行清理
    Unloading = 5,
}

impl fmt::Display for FiberState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Pending => "PENDING",
            Self::Loading => "LOADING",
            Self::Active => "ACTIVE",
            Self::Failed => "FAILED",
            Self::Disposed => "DISPOSED",
            Self::Unloading => "UNLOADING",
        };
        f.write_str(s)
    }
}

/// Cordis 核心错误类型。
#[derive(Debug, Clone)]
pub enum CordisError {
    InactiveEffect,
    Validation(ValidationError),
    InactiveService(String),
    MissingInject(String),
    TypeMismatch { name: String, expected: &'static str },
    DuplicateService { name: String, owner: String },
    Message(String),
}

impl CordisError {
    pub fn from_box(err: BoxError) -> Self {
        match err.downcast::<CordisError>() {
            Ok(ce) => *ce,
            Err(err) => match err.downcast::<ValidationError>() {
                Ok(ve) => CordisError::Validation(*ve),
                Err(err) => CordisError::Message(err.to_string()),
            },
        }
    }

    pub fn code(&self) -> Option<&'static str> {
        match self {
            Self::InactiveEffect => Some("INACTIVE_EFFECT"),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Validation(_) => "ValidationError",
            Self::InactiveEffect => "CordisError",
            _ => "Error",
        }
    }
}

impl fmt::Display for CordisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InactiveEffect => write!(f, "cannot create effect on inactive context"),
            Self::Validation(v) => write!(f, "{v}"),
            Self::InactiveService(name) => {
                write!(f, "cannot get required service \"{name}\" in inactive context")
            }
            Self::MissingInject(name) => {
                write!(f, "cannot get property \"{name}\" without inject")
            }
            Self::TypeMismatch { name, expected } => {
                write!(f, "service \"{name}\" type mismatch: expected {expected}")
            }
            Self::DuplicateService { name, owner } => {
                write!(f, "service \"{name}\" has been registered at <{owner}>")
            }
            Self::Message(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for CordisError {}

/// effect 返回的撤销动作：可以是同步函数，也可以是带同步前缀 + 异步尾巴的清理函数。
pub enum Cleanup {
    Sync(Box<dyn FnOnce() + Send>),
    Async(Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>),
}

impl Cleanup {
    pub fn sync(f: impl FnOnce() + Send + 'static) -> Self {
        Self::Sync(Box::new(f))
    }

    pub fn async_fn<Fut>(f: impl FnOnce() -> Fut + Send + 'static) -> Self
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        Self::Async(Box::new(move || Box::pin(f())))
    }

    pub fn async_raw(f: impl FnOnce() -> BoxFuture<'static, ()> + Send + 'static) -> Self {
        Self::Async(Box::new(f))
    }
}

enum EffectState {
    SyncExecuting,
    AsyncPending { disposed_early: bool },
    Ready { cleanup: Option<Cleanup> },
    Disposed,
}

struct EffectInner {
    #[allow(dead_code)]
    label: String,
    fiber: Weak<FiberInner>,
    sn: Mutex<Option<u64>>,
    state: Mutex<EffectState>,
    done_tx: watch::Sender<bool>,
    done_rx: watch::Receiver<bool>,
}

/// 一条可撤销注册的句柄。调用 `.dispose()` 可提前撤销。
#[derive(Clone)]
pub struct DisposableHandle(Arc<EffectInner>);

impl DisposableHandle {
    /// 撤销此注册。同步前缀会立即执行，异步部分在后台任务中完成；可选择 `.await` 等待清理结束。
    pub fn dispose(&self) -> BoxFuture<'static, ()> {
        let sn = self.0.sn.lock().unwrap().take();
        if let Some(sn) = sn
            && let Some(fiber) = self.0.fiber.upgrade()
        {
            fiber.mutable.lock().unwrap().disposables.remove(sn);
        }

        let action = {
            let mut state = self.0.state.lock().unwrap();
            match std::mem::replace(&mut *state, EffectState::Disposed) {
                EffectState::SyncExecuting => {
                    let _ = self.0.done_tx.send(true);
                    None
                }
                EffectState::AsyncPending { .. } => {
                    *state = EffectState::AsyncPending {
                        disposed_early: true,
                    };
                    None
                }
                EffectState::Ready { cleanup } => match cleanup {
                    None => {
                        let _ = self.0.done_tx.send(true);
                        None
                    }
                    Some(c) => Some(c),
                },
                EffectState::Disposed => None,
            }
        };

        if let Some(cleanup) = action {
            match cleanup {
                Cleanup::Sync(f) => {
                    f();
                    let _ = self.0.done_tx.send(true);
                }
                Cleanup::Async(f) => {
                    // 立即调用 f()，让它的同步前缀（如标记 uid=None、更新 epoch）马上生效，
                    // 再把返回的异步尾放到 Tokio 上执行，确保即使调用方不 await 也能完成清理。
                    let fut = f();
                    let done_tx = self.0.done_tx.clone();
                    tokio::spawn(async move {
                        fut.await;
                        let _ = done_tx.send(true);
                    });
                }
            }
        }

        let mut rx = self.0.done_rx.clone();
        Box::pin(async move {
            loop {
                if *rx.borrow_and_update() {
                    break;
                }
                if rx.changed().await.is_err() {
                    break;
                }
            }
        })
    }
}

#[derive(Clone, Copy)]
enum TransitionStep {
    Reload,
    Unload,
}

struct FiberMutable {
    uid: Option<u64>,
    state: FiberState,
    /// 运行期间可见的服务快照：注入的 + 自己提供的。
    store: Option<HashMap<String, ServiceImpl>>,
    /// 依赖检查的最新结果：服务名 → 当前提供者。
    shadow_store: HashMap<String, ServiceImpl>,
    /// epoch = 依赖提供者 uid 拼成的串，不齐时为 INACTIVE。
    epoch: String,
    config_raw: Arc<dyn Any + Send + Sync>,
    error: Option<CordisError>,
    /// 是否有状态切换循环正在跑，确保加载与卸载串行进行。
    inertia_active: bool,
    disposables: DisposableList<DisposableHandle>,
    self_dispose: Option<DisposableHandle>,
}

impl FiberMutable {
    fn compute_state(&self) -> FiberState {
        if self.uid.is_none() {
            FiberState::Disposed
        } else if self.error.is_some() {
            FiberState::Failed
        } else if self.epoch != INACTIVE {
            FiberState::Active
        } else {
            FiberState::Pending
        }
    }
}

pub(crate) struct FiberInner {
    pub(crate) root: Weak<RootContext>,
    pub(crate) parent: Option<Weak<FiberInner>>,
    pub(crate) inject: Inject,
    pub(crate) runtime: Option<Arc<PluginRuntime>>,
    mutable: Mutex<FiberMutable>,
    inertia_tx: watch::Sender<bool>,
    inertia_rx: watch::Receiver<bool>,
}

/// Fiber 句柄，代表一个挂载中的插件实例。
///
/// 实现了 `IntoFuture`，因此既可以忽略返回值（异步后台加载），也可以直接 `app.plugin(...).await?`。
#[derive(Clone)]
pub struct Fiber(pub(crate) Arc<FiberInner>);

impl Fiber {
    pub(crate) fn from_inner(inner: Arc<FiberInner>) -> Self {
        Self(inner)
    }

    pub(crate) fn downgrade(&self) -> Weak<FiberInner> {
        Arc::downgrade(&self.0)
    }

    /// 创建根 Fiber：一出生就是 `ACTIVE`。
    pub(crate) fn new_root(root: Weak<RootContext>) -> Self {
        let (inertia_tx, inertia_rx) = watch::channel(false);
        Self(Arc::new(FiberInner {
            root,
            parent: None,
            inject: Inject::default(),
            runtime: None,
            mutable: Mutex::new(FiberMutable {
                uid: Some(0),
                state: FiberState::Active,
                store: Some(HashMap::new()),
                shadow_store: HashMap::new(),
                epoch: String::new(),
                config_raw: Arc::new(()),
                error: None,
                inertia_active: false,
                disposables: DisposableList::new(),
                self_dispose: None,
            }),
            inertia_tx,
            inertia_rx,
        }))
    }

    /// 创建子插件 Fiber，挂载到 `parent` 下并开始异步加载。
    pub(crate) fn new_child(
        parent: &Context,
        raw_config: Arc<dyn Any + Send + Sync>,
        inject: Inject,
        runtime: Arc<PluginRuntime>,
    ) -> Result<Self, CordisError> {
        let root_arc = parent.root_arc();
        let uid = root_arc.registry.next_uid();
        let (inertia_tx, inertia_rx) = watch::channel(false);

        let child = Arc::new(FiberInner {
            root: Arc::downgrade(&root_arc),
            parent: Some(parent.fiber().downgrade()),
            inject,
            runtime: Some(Arc::clone(&runtime)),
            mutable: Mutex::new(FiberMutable {
                uid: Some(uid),
                state: FiberState::Pending,
                store: None,
                shadow_store: HashMap::new(),
                epoch: INACTIVE.to_string(),
                config_raw: raw_config,
                error: None,
                inertia_active: false,
                disposables: DisposableList::new(),
                self_dispose: None,
            }),
            inertia_tx,
            inertia_rx,
        });

        let rt_for_effect = Arc::clone(&runtime);
        let child_for_effect = Arc::clone(&child);
        let dispose_handle = parent.fiber().effect("ctx.plugin()", move || {
            let sn = rt_for_effect.push_fiber(Fiber(Arc::clone(&child_for_effect)));
            let rt_cleanup = Arc::clone(&rt_for_effect);
            let child_cleanup = Arc::clone(&child_for_effect);
            Ok::<_, CordisError>(Some(Cleanup::async_raw(move || {
                {
                    let mut m = child_cleanup.mutable.lock().unwrap();
                    m.uid = None;
                }
                if let Some(root) = child_cleanup.root.upgrade() {
                    root.events.emit(
                        INTERNAL_PLUGIN,
                        InternalPluginEvent {
                            fiber: Fiber(Arc::clone(&child_cleanup)),
                        },
                    );
                    if root.registry.has_key(&rt_cleanup.key) {
                        rt_cleanup.remove_fiber(sn);
                        if rt_cleanup.is_empty() {
                            root.registry.delete_by_key(&rt_cleanup.key);
                        }
                    }
                }
                child_cleanup.set_epoch(INACTIVE.to_string());
                Box::pin(async move {
                    let _ = Fiber(child_cleanup).wait_settled().await;
                })
            })))
        })?;

        child.mutable.lock().unwrap().self_dispose = Some(dispose_handle);

        let fiber = Fiber(child);
        root_arc.events.emit(
            INTERNAL_PLUGIN,
            InternalPluginEvent {
                fiber: fiber.clone(),
            },
        );

        if fiber.uid().is_some() && parent.fiber().state() != FiberState::Unloading {
            let names: Vec<String> = fiber.0.inject.iter().map(str::to_string).collect();
            for name in &names {
                fiber.check_impl(name);
            }
            fiber.refresh();
        }

        Ok(fiber)
    }

    pub fn uid(&self) -> Option<u64> {
        self.0.mutable.lock().unwrap().uid
    }

    pub fn state(&self) -> FiberState {
        self.0.mutable.lock().unwrap().state
    }

    pub fn has_runtime(&self) -> bool {
        self.0.runtime.is_some()
    }

    pub fn runtime(&self) -> Option<Arc<PluginRuntime>> {
        self.0.runtime.clone()
    }

    pub fn parent_fiber(&self) -> Option<Fiber> {
        self.0
            .parent
            .as_ref()
            .and_then(Weak::upgrade)
            .map(Fiber::from_inner)
    }

    /// 绑定该 Fiber 的 `Context`（无 `Arc` 环，按需构造）。
    pub fn ctx(&self) -> Context {
        let root = self
            .0
            .root
            .upgrade()
            .expect("Cordis RootContext has been dropped");
        Context::from_parts(root, self.clone())
    }

    /// 插件显示名：向上查找最近的具名插件，否则返回 `"root"`。
    pub fn name(&self) -> String {
        let mut current = Some(Arc::clone(&self.0));
        while let Some(fiber) = current {
            if let Some(rt) = &fiber.runtime
                && let Some(name) = &rt.name
            {
                return name.clone();
            }
            current = fiber.parent.as_ref().and_then(Weak::upgrade);
        }
        "root".to_string()
    }

    pub fn assert_active(&self) -> Result<(), CordisError> {
        if self.uid().is_none() {
            return Err(CordisError::InactiveEffect);
        }
        Ok(())
    }

    pub(crate) fn has_inject(&self, name: &str) -> bool {
        self.0.inject.contains(name)
    }

    pub(crate) fn get_from_snapshot(&self, name: &str) -> Option<ServiceImpl> {
        self.0
            .mutable
            .lock()
            .unwrap()
            .store
            .as_ref()
            .and_then(|s| s.get(name).cloned())
    }

    pub(crate) fn insert_into_snapshot(&self, name: &str, impl_: ServiceImpl) {
        if let Some(store) = &mut self.0.mutable.lock().unwrap().store {
            store.insert(name.to_string(), impl_);
        }
    }

    pub(crate) fn remove_from_snapshot(&self, name: &str) {
        if let Some(store) = &mut self.0.mutable.lock().unwrap().store {
            store.remove(name);
        }
    }

    /// 同步注册一个可撤销副作用（对应 `ctx.effect(() => cleanup)`）。
    pub fn effect<E, F>(
        &self,
        label: impl Into<String>,
        execute: F,
    ) -> Result<DisposableHandle, CordisError>
    where
        E: Into<BoxError>,
        F: FnOnce() -> Result<Option<Cleanup>, E>,
    {
        let label = label.into();
        let (done_tx, done_rx) = watch::channel(false);
        let record = Arc::new(EffectInner {
            label,
            fiber: Arc::downgrade(&self.0),
            sn: Mutex::new(None),
            state: Mutex::new(EffectState::SyncExecuting),
            done_tx,
            done_rx,
        });
        let handle = DisposableHandle(Arc::clone(&record));

        // 先占位置，再执行：保证后注册的先撤销
        let sn = {
            let mut m = self.0.mutable.lock().unwrap();
            if m.uid.is_none() || m.state == FiberState::Unloading {
                return Err(CordisError::InactiveEffect);
            }
            m.disposables.push(handle.clone())
        };
        *record.sn.lock().unwrap() = Some(sn);

        match execute() {
            Ok(cleanup) => {
                let mut early_cleanup = None;
                {
                    let mut st = record.state.lock().unwrap();
                    match *st {
                        EffectState::SyncExecuting => {
                            *st = EffectState::Ready { cleanup };
                        }
                        EffectState::Disposed => {
                            early_cleanup = cleanup;
                        }
                        _ => {}
                    }
                }
                if let Some(c) = early_cleanup {
                    match c {
                        Cleanup::Sync(f) => f(),
                        Cleanup::Async(f) => {
                            let fut = f();
                            tokio::spawn(fut);
                        }
                    }
                }
                Ok(handle)
            }
            Err(err) => {
                *record.sn.lock().unwrap() = None;
                self.0.mutable.lock().unwrap().disposables.remove(sn);
                *record.state.lock().unwrap() = EffectState::Disposed;
                let _ = record.done_tx.send(true);
                Err(CordisError::from_box(err.into()))
            }
        }
    }

    /// 异步注册一个可撤销副作用（对应 `ctx.effect(async () => cleanup)`）。
    ///
    /// 不会阻塞当前 `apply`，但卸载时会等该异步过程结束后再执行它返回的清理函数。
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
        let label = label.into();
        let (done_tx, done_rx) = watch::channel(false);
        let record = Arc::new(EffectInner {
            label,
            fiber: Arc::downgrade(&self.0),
            sn: Mutex::new(None),
            state: Mutex::new(EffectState::AsyncPending {
                disposed_early: false,
            }),
            done_tx,
            done_rx,
        });
        let handle = DisposableHandle(Arc::clone(&record));

        let sn = {
            let mut m = self.0.mutable.lock().unwrap();
            if m.uid.is_none() || m.state == FiberState::Unloading {
                return Err(CordisError::InactiveEffect);
            }
            m.disposables.push(handle.clone())
        };
        *record.sn.lock().unwrap() = Some(sn);

        let fut = execute();
        let fiber_weak = Arc::downgrade(&self.0);
        let record_task = Arc::clone(&record);
        tokio::spawn(async move {
            let res = fut.await.map_err(Into::into);
            match res {
                Ok(cleanup) => {
                    let run_now = {
                        let mut st = record_task.state.lock().unwrap();
                        match *st {
                            EffectState::AsyncPending {
                                disposed_early: true,
                            }
                            | EffectState::Disposed => {
                                *st = EffectState::Disposed;
                                true
                            }
                            _ => {
                                *st = EffectState::Ready { cleanup };
                                return;
                            }
                        }
                    };
                    if run_now {
                        if let Some(c) = cleanup {
                            match c {
                                Cleanup::Sync(f) => f(),
                                Cleanup::Async(f) => f().await,
                            }
                        }
                        let _ = record_task.done_tx.send(true);
                    }
                }
                Err(err) => {
                    let err = CordisError::from_box(err);
                    *record_task.sn.lock().unwrap() = None;
                    if let Some(fiber) = fiber_weak.upgrade() {
                        fiber.mutable.lock().unwrap().disposables.remove(sn);
                        if let Some(root) = fiber.root.upgrade() {
                            root.logger.error(&err);
                        }
                    }
                    *record_task.state.lock().unwrap() = EffectState::Disposed;
                    let _ = record_task.done_tx.send(true);
                }
            }
        });

        Ok(handle)
    }

    /// 检查某个依赖现在由谁提供（只认 `ACTIVE` 的提供者）。
    pub(crate) fn check_impl(&self, name: &str) {
        let Some(root) = self.0.root.upgrade() else {
            return;
        };
        let impl_opt = root.reflect.get_impl(name, true);
        let mut m = self.0.mutable.lock().unwrap();
        if let Some(impl_) = impl_opt {
            m.shadow_store.insert(name.to_string(), impl_);
        } else {
            m.shadow_store.remove(name);
        }
    }

    /// 重新计算 epoch：依赖是否齐了？提供者是否换了？
    pub(crate) fn refresh(&self) {
        let epoch = {
            let m = self.0.mutable.lock().unwrap();
            let mut buf = String::new();
            let mut missing = false;
            for name in self.0.inject.iter() {
                if let Some(impl_) = m.shadow_store.get(name) {
                    buf.push(':');
                    buf.push_str(&impl_.fiber_uid.to_string());
                } else {
                    missing = true;
                    break;
                }
            }
            if missing {
                INACTIVE.to_string()
            } else {
                buf
            }
        };
        self.0.set_epoch(epoch);
    }

    /// 卸载当前插件（若为根 Fiber，则等价于 `restart()`）。
    pub fn dispose(&self) -> BoxFuture<'static, ()> {
        let handle = self.0.mutable.lock().unwrap().self_dispose.clone();
        match handle {
            Some(h) => h.dispose(),
            None => {
                let this = self.clone();
                Box::pin(async move {
                    let _ = this.restart().await;
                })
            }
        }
    }

    /// 等待所有进行中的状态切换结束；若插件加载失败则把错误返回给等待者。
    pub async fn wait_settled(&self) -> Result<(), CordisError> {
        let mut rx = self.0.inertia_rx.clone();
        loop {
            if !*rx.borrow_and_update() {
                break;
            }
            if rx.changed().await.is_err() {
                break;
            }
        }
        if let Some(err) = self.0.mutable.lock().unwrap().error.clone() {
            return Err(err);
        }
        Ok(())
    }

    /// 重启当前插件（卸载后重新加载）。
    pub async fn restart(&self) -> Result<(), CordisError> {
        self.assert_active()?;
        self.0.set_epoch(INACTIVE.to_string());
        self.refresh();
        self.wait_settled().await
    }

    /// 更新配置并用新配置重启插件。
    pub fn update<C: Send + Sync + Clone + 'static>(
        &self,
        config: C,
    ) -> BoxFuture<'static, Result<(), CordisError>> {
        if let Err(err) = self.assert_active() {
            return Box::pin(async move { Err(err) });
        }
        let was_active = {
            let mut m = self.0.mutable.lock().unwrap();
            m.config_raw = Arc::new(config);
            m.error = None;
            m.state == FiberState::Active
        };
        self.0.set_epoch(INACTIVE.to_string());
        self.refresh();
        if was_active {
            let this = self.clone();
            Box::pin(async move { this.wait_settled().await })
        } else {
            Box::pin(async { Ok(()) })
        }
    }
}

impl IntoFuture for Fiber {
    type Output = Result<Fiber, CordisError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            self.wait_settled().await?;
            Ok(self)
        })
    }
}

impl FiberInner {
    fn set_epoch(self: &Arc<Self>, new_epoch: String) {
        let transition = {
            let mut m = self.mutable.lock().unwrap();
            if m.epoch == new_epoch {
                return;
            }
            let old_epoch = std::mem::replace(&mut m.epoch, new_epoch.clone());
            // 正在切换中：等当前切换结束后，它会自动对照最新 epoch 再决定下一步
            if m.inertia_active {
                return;
            }
            m.inertia_active = true;
            let _ = self.inertia_tx.send(true);
            let step = if new_epoch != INACTIVE && old_epoch == INACTIVE {
                TransitionStep::Reload
            } else {
                TransitionStep::Unload
            };
            let target_state = match step {
                TransitionStep::Reload => FiberState::Loading,
                TransitionStep::Unload => FiberState::Unloading,
            };
            let old_state = std::mem::replace(&mut m.state, target_state);
            Some((step, old_state, target_state))
        };

        if let Some((step, old_state, target_state)) = transition {
            self.after_state_change(old_state, target_state);
            let fiber = Arc::clone(self);
            tokio::spawn(async move {
                fiber.run_transitions(step).await;
            });
        }
    }

    async fn run_transitions(self: Arc<Self>, mut step: TransitionStep) {
        loop {
            let next = match step {
                TransitionStep::Reload => self.reload_step().await,
                TransitionStep::Unload => self.unload_step().await,
            };
            match next {
                Some(n) => step = n,
                None => break,
            }
        }
    }

    async fn reload_step(self: &Arc<Self>) -> Option<TransitionStep> {
        let (old_epoch, raw_config) = {
            let mut m = self.mutable.lock().unwrap();
            m.store = Some(m.shadow_store.clone());
            (m.epoch.clone(), Arc::clone(&m.config_raw))
        };

        // 让出一拍：确保 ctx.plugin() 先返回，再异步执行插件本体
        tokio::task::yield_now().await;

        let still_valid = self.mutable.lock().unwrap().epoch == old_epoch;
        if still_valid && let Some(runtime) = &self.runtime {
            let ctx = Fiber(Arc::clone(self)).ctx();
            let result = runtime.plugin.validate_and_apply(ctx, raw_config).await;
            match result {
                Ok(cleanup_opt) => {
                    if let Some(cleanup) = cleanup_opt {
                        let _ = Fiber(Arc::clone(self))
                            .effect("plugin cleanup", || Ok::<_, CordisError>(Some(cleanup)));
                    }
                    self.mutable.lock().unwrap().error = None;
                }
                Err(reason) => {
                    if let Some(root) = self.root.upgrade() {
                        root.logger.error(&reason);
                    }
                    let mut m = self.mutable.lock().unwrap();
                    m.error = Some(reason);
                    m.epoch = INACTIVE.to_string();
                }
            }
        }

        let (old_state, new_state, next_step) = {
            let mut m = self.mutable.lock().unwrap();
            if m.epoch == old_epoch {
                let target = m.compute_state();
                let old = std::mem::replace(&mut m.state, target);
                m.inertia_active = false;
                let _ = self.inertia_tx.send(false);
                (old, target, None)
            } else {
                let old = std::mem::replace(&mut m.state, FiberState::Unloading);
                (old, FiberState::Unloading, Some(TransitionStep::Unload))
            }
        };

        self.after_state_change(old_state, new_state);
        next_step
    }

    async fn unload_step(self: &Arc<Self>) -> Option<TransitionStep> {
        let disposables = self.mutable.lock().unwrap().disposables.clear();
        let futs = disposables.into_iter().map(|handle| async move {
            tokio::task::yield_now().await;
            handle.dispose().await;
        });
        join_all(futs).await;

        let (old_state, new_state, next_step) = {
            let mut m = self.mutable.lock().unwrap();
            m.store = None;
            if m.epoch == INACTIVE {
                let target = m.compute_state();
                let old = std::mem::replace(&mut m.state, target);
                m.inertia_active = false;
                let _ = self.inertia_tx.send(false);
                (old, target, None)
            } else {
                let old = std::mem::replace(&mut m.state, FiberState::Loading);
                (old, FiberState::Loading, Some(TransitionStep::Reload))
            }
        };

        self.after_state_change(old_state, new_state);
        next_step
    }

    fn after_state_change(self: &Arc<Self>, old_state: FiberState, new_state: FiberState) {
        if old_state == new_state {
            return;
        }
        let Some(root) = self.root.upgrade() else {
            return;
        };
        root.events.emit(
            INTERNAL_STATUS,
            InternalStatusEvent {
                fiber: Fiber(Arc::clone(self)),
                old_state,
                new_state,
            },
        );
        // 进入或离开 ACTIVE 时，通知依赖本插件服务的其他插件
        if old_state != FiberState::Active && new_state != FiberState::Active {
            return;
        }
        let owned = root.reflect.owned_service_names(self);
        for name in owned {
            root.reflect.notify(&[name.as_str()]);
        }
    }
}
