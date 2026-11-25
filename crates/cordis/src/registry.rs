//! 插件登记表：`ctx.plugin()` 和 `ctx.inject()` 的实现（对应 `packages/cordis/src/registry.ts`）。

use std::any::{Any, TypeId};
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::context::Context;
use crate::fiber::{Cleanup, CordisError, Fiber};
use crate::utils::{BoxError, BoxFuture, DisposableList};

/// 依赖声明：保存所需的服务名列表（保留声明顺序并去重）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Inject {
    names: Vec<String>,
}

impl Inject {
    pub fn new<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut names = Vec::new();
        for item in iter {
            let name = item.into();
            if !names.contains(&name) {
                names.push(name);
            }
        }
        Self { names }
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }
}

impl<const N: usize> From<[&str; N]> for Inject {
    fn from(arr: [&str; N]) -> Self {
        Self::new(arr)
    }
}

impl From<&[&str]> for Inject {
    fn from(slice: &[&str]) -> Self {
        Self::new(slice.iter().copied())
    }
}

impl From<Vec<&str>> for Inject {
    fn from(vec: Vec<&str>) -> Self {
        Self::new(vec)
    }
}

impl From<Vec<String>> for Inject {
    fn from(vec: Vec<String>) -> Self {
        Self::new(vec)
    }
}

/// 单条配置校验问题。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub message: String,
    pub path: Vec<String>,
}

impl ValidationIssue {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            path: Vec::new(),
        }
    }

    pub fn at(mut self, path: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.path = path.into_iter().map(Into::into).collect();
        self
    }
}

/// 配置校验失败错误（对应 `fiber.ts` 的 `ValidationError`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationError {
    pub fn new(issues: Vec<ValidationIssue>) -> Self {
        Self { issues }
    }

    pub fn single(message: impl Into<String>) -> Self {
        Self {
            issues: vec![ValidationIssue::new(message)],
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid config:")?;
        for issue in &self.issues {
            write!(f, "\n  - {}", issue.message)?;
            if !issue.path.is_empty() {
                write!(f, " (at {})", issue.path.join("."))?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for ValidationError {}

/// 插件在登记表中的唯一身份。
///
/// 具名结构体插件默认以 `TypeId` 作为身份；闭包 / 动态构造的 `PluginDef` 使用实例 ID。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PluginKey {
    Type(TypeId),
    Instance(u64),
}

/// 插件 trait：对应 JS 版的函数 / 对象 / Service 类三种写法的统一抽象。
pub trait Plugin: Send + Sync + 'static {
    type Config: Send + Sync + Clone + 'static;

    fn key(&self) -> PluginKey {
        PluginKey::Type(TypeId::of::<Self>())
    }

    fn name(&self) -> Option<&str> {
        None
    }

    fn inject(&self) -> Inject {
        Inject::default()
    }

    fn validate(&self, _config: &Self::Config) -> Result<(), ValidationError> {
        Ok(())
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>>;
}

/// 将外部闭包返回值统一转换为 `Result<Option<Cleanup>, BoxError>`。
pub trait IntoPluginReturn {
    fn into_plugin_return(self) -> Result<Option<Cleanup>, BoxError>;
}

impl IntoPluginReturn for () {
    fn into_plugin_return(self) -> Result<Option<Cleanup>, BoxError> {
        Ok(None)
    }
}

impl IntoPluginReturn for Cleanup {
    fn into_plugin_return(self) -> Result<Option<Cleanup>, BoxError> {
        Ok(Some(self))
    }
}

impl IntoPluginReturn for Option<Cleanup> {
    fn into_plugin_return(self) -> Result<Option<Cleanup>, BoxError> {
        Ok(self)
    }
}

impl<T: IntoPluginReturn, E: Into<BoxError>> IntoPluginReturn for Result<T, E> {
    fn into_plugin_return(self) -> Result<Option<Cleanup>, BoxError> {
        match self {
            Ok(v) => v.into_plugin_return(),
            Err(e) => Err(e.into()),
        }
    }
}

type ValidatorFn<C> = Box<dyn Fn(&C) -> Result<(), ValidationError> + Send + Sync>;
type ApplyFn<C> =
    Box<dyn Fn(Context, C) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> + Send + Sync>;

struct PluginDefInner<C> {
    id: u64,
    name: Option<String>,
    inject: Inject,
    validator: Option<ValidatorFn<C>>,
    apply_fn: ApplyFn<C>,
}

/// 便捷插件构建器：用于把同步/异步闭包快速组装成带 `name`、`inject`、`validate` 的插件。
pub struct PluginDef<C = ()>(Arc<PluginDefInner<C>>);

impl<C> Clone for PluginDef<C> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

static NEXT_DEF_ID: AtomicU64 = AtomicU64::new(1);

impl<C: Send + Sync + Clone + 'static> PluginDef<C> {
    /// 用异步闭包创建插件。
    pub fn new<F, Fut, R>(name: impl Into<String>, apply_fn: F) -> Self
    where
        F: Fn(Context, C) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoPluginReturn + 'static,
    {
        Self::build(Some(name.into()), apply_fn)
    }

    /// 创建匿名插件（`name` 为空时，其 `Fiber::name()` 会向上继承父插件名）。
    pub fn anonymous<F, Fut, R>(apply_fn: F) -> Self
    where
        F: Fn(Context, C) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoPluginReturn + 'static,
    {
        Self::build(None, apply_fn)
    }

    fn build<F, Fut, R>(name: Option<String>, apply_fn: F) -> Self
    where
        F: Fn(Context, C) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoPluginReturn + 'static,
    {
        let id = NEXT_DEF_ID.fetch_add(1, Ordering::Relaxed);
        let name = name.filter(|s| !s.is_empty() && s != "apply");
        Self(Arc::new(PluginDefInner {
            id,
            name,
            inject: Inject::default(),
            validator: None,
            apply_fn: Box::new(move |ctx, cfg| {
                let fut = apply_fn(ctx, cfg);
                Box::pin(async move { fut.await.into_plugin_return() })
            }),
        }))
    }

    /// 用同步闭包创建插件。
    pub fn sync<F, R>(name: impl Into<String>, apply_fn: F) -> Self
    where
        F: Fn(Context, C) -> R + Send + Sync + 'static,
        R: IntoPluginReturn + 'static,
    {
        let id = NEXT_DEF_ID.fetch_add(1, Ordering::Relaxed);
        let name = Some(name.into()).filter(|s| !s.is_empty() && s != "apply");
        Self(Arc::new(PluginDefInner {
            id,
            name,
            inject: Inject::default(),
            validator: None,
            apply_fn: Box::new(move |ctx, cfg| {
                let res = apply_fn(ctx, cfg).into_plugin_return();
                Box::pin(async move { res })
            }),
        }))
    }

    pub fn inject(self, inject: impl Into<Inject>) -> Self {
        let mut inner = Arc::try_unwrap(self.0)
            .unwrap_or_else(|_| panic!("PluginDef::inject must be called before cloning/mounting"));
        inner.inject = inject.into();
        Self(Arc::new(inner))
    }

    pub fn validate<F>(self, validator: F) -> Self
    where
        F: Fn(&C) -> Result<(), ValidationError> + Send + Sync + 'static,
    {
        let mut inner = Arc::try_unwrap(self.0).unwrap_or_else(|_| {
            panic!("PluginDef::validate must be called before cloning/mounting")
        });
        inner.validator = Some(Box::new(validator));
        Self(Arc::new(inner))
    }
}

impl<C: Send + Sync + Clone + 'static> Plugin for PluginDef<C> {
    type Config = C;

    fn key(&self) -> PluginKey {
        PluginKey::Instance(self.0.id)
    }

    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    fn inject(&self) -> Inject {
        self.0.inject.clone()
    }

    fn validate(&self, config: &Self::Config) -> Result<(), ValidationError> {
        if let Some(validator) = &self.0.validator {
            validator(config)?;
        }
        Ok(())
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        (self.0.apply_fn)(ctx, config)
    }
}

pub(crate) trait ErasedPlugin: Send + Sync {
    fn validate_and_apply(
        &self,
        ctx: Context,
        raw_config: Arc<dyn Any + Send + Sync>,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, CordisError>>;
}

impl<P: Plugin> ErasedPlugin for P {
    fn validate_and_apply(
        &self,
        ctx: Context,
        raw_config: Arc<dyn Any + Send + Sync>,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, CordisError>> {
        let config = match raw_config.downcast::<P::Config>() {
            Ok(cfg) => (*cfg).clone(),
            Err(_) => {
                let expected = std::any::type_name::<P::Config>();
                return Box::pin(async move {
                    Err(CordisError::Validation(ValidationError::single(format!(
                        "invalid config type: expected {expected}"
                    ))))
                });
            }
        };

        if let Err(err) = self.validate(&config) {
            return Box::pin(async move { Err(CordisError::Validation(err)) });
        }

        let fut = self.apply(ctx, config);
        Box::pin(async move { fut.await.map_err(CordisError::from_box) })
    }
}

/// 同一个插件（不管挂了几份）共享一个 `PluginRuntime`。
pub struct PluginRuntime {
    pub key: PluginKey,
    pub name: Option<String>,
    pub(crate) plugin: Arc<dyn ErasedPlugin>,
    pub(crate) fibers: Mutex<DisposableList<Fiber>>,
}

impl PluginRuntime {
    pub(crate) fn push_fiber(&self, fiber: Fiber) -> u64 {
        self.fibers.lock().unwrap().push(fiber)
    }

    pub(crate) fn remove_fiber(&self, sn: u64) {
        self.fibers.lock().unwrap().remove(sn);
    }

    pub fn is_empty(&self) -> bool {
        self.fibers.lock().unwrap().is_empty()
    }

    pub fn fibers(&self) -> Vec<Fiber> {
        self.fibers.lock().unwrap().snapshot()
    }
}

pub struct RegistryService {
    counter: AtomicU64,
    runtimes: Mutex<Vec<Arc<PluginRuntime>>>,
}

impl Default for RegistryService {
    fn default() -> Self {
        Self::new()
    }
}

impl RegistryService {
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
            runtimes: Mutex::new(Vec::new()),
        }
    }

    /// 每个插件实例分配一个递增 `uid`（根 Fiber 为 0，插件从 1 开始）。
    pub(crate) fn next_uid(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn get<P: Plugin>(&self, plugin: &P) -> Option<Arc<PluginRuntime>> {
        self.get_by_key(&plugin.key())
    }

    pub fn get_by_type<P: Plugin>(&self) -> Option<Arc<PluginRuntime>> {
        self.get_by_key(&PluginKey::Type(TypeId::of::<P>()))
    }

    pub fn get_by_key(&self, key: &PluginKey) -> Option<Arc<PluginRuntime>> {
        let runtimes = self.runtimes.lock().unwrap();
        runtimes.iter().find(|r| &r.key == key).cloned()
    }

    pub fn has<P: Plugin>(&self, plugin: &P) -> bool {
        self.has_key(&plugin.key())
    }

    pub fn has_key(&self, key: &PluginKey) -> bool {
        let runtimes = self.runtimes.lock().unwrap();
        runtimes.iter().any(|r| &r.key == key)
    }

    /// 删除一个插件（按实例或类型）：卸载它的全部 Fiber 实例。
    pub fn delete<P: Plugin>(&self, plugin: &P) -> Option<Arc<PluginRuntime>> {
        self.delete_by_key(&plugin.key())
    }

    pub fn delete_by_type<P: Plugin>(&self) -> Option<Arc<PluginRuntime>> {
        self.delete_by_key(&PluginKey::Type(TypeId::of::<P>()))
    }

    pub fn delete_by_key(&self, key: &PluginKey) -> Option<Arc<PluginRuntime>> {
        let removed = {
            let mut runtimes = self.runtimes.lock().unwrap();
            let idx = runtimes.iter().position(|r| &r.key == key)?;
            runtimes.remove(idx)
        };
        for fiber in removed.fibers() {
            drop(fiber.dispose());
        }
        Some(removed)
    }

    /// 按注册顺序返回当前所有 `PluginRuntime` 的快照。
    pub fn values(&self) -> Vec<Arc<PluginRuntime>> {
        self.runtimes.lock().unwrap().clone()
    }

    /// `ctx.inject(['a'], cb)`：创建一个只有依赖声明的匿名子插件。
    pub fn inject<F, Fut, R>(
        &self,
        ctx: &Context,
        inject: impl Into<Inject>,
        callback: F,
    ) -> Result<Fiber, CordisError>
    where
        F: Fn(Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoPluginReturn + 'static,
    {
        let def = PluginDef::<()>::anonymous(move |c, ()| callback(c)).inject(inject);
        self.plugin(ctx, def, ())
    }

    /// `ctx.plugin()`：创建一个 Fiber 并开始异步加载。
    pub fn plugin<P: Plugin>(
        &self,
        ctx: &Context,
        plugin: P,
        config: P::Config,
    ) -> Result<Fiber, CordisError> {
        ctx.fiber().assert_active()?;
        let key = plugin.key();
        let inject = plugin.inject();

        let runtime = {
            let mut runtimes = self.runtimes.lock().unwrap();
            if let Some(existing) = runtimes.iter().find(|r| r.key == key) {
                Arc::clone(existing)
            } else {
                let name = plugin
                    .name()
                    .filter(|s| !s.is_empty() && *s != "apply")
                    .map(str::to_string);
                let rt = Arc::new(PluginRuntime {
                    key,
                    name,
                    plugin: Arc::new(plugin),
                    fibers: Mutex::new(DisposableList::new()),
                });
                runtimes.push(Arc::clone(&rt));
                rt
            }
        };

        let raw_config: Arc<dyn Any + Send + Sync> = Arc::new(config);
        Fiber::new_child(ctx, raw_config, inject, runtime)
    }
}
