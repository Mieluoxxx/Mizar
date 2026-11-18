//! 服务基类抽象（对应 `packages/cordis/src/service.ts`）。
//!
//! 对应 JS 版写法：
//! ```ignore
//! class AgentLoop extends Service {
//!   static inject = ['llm', 'tools']
//!   constructor(ctx, config) { super(ctx, 'agentLoop') }
//! }
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use crate::context::Context;
use crate::fiber::{Cleanup, CordisError};
use crate::registry::{Inject, Plugin, ValidationError};
use crate::utils::{BoxError, BoxFuture, short_type_name};

/// 服务 trait：实现它即可自动通过 `ServicePlugin<S>` 或 `S::plugin()` 挂载为 Cordis 插件。
pub trait Service: Sized + Send + Sync + 'static {
    type Config: Send + Sync + Clone + 'static;

    /// 服务注册名（例如 `"llm"`、`"tools"`、`"agentLoop"`）。
    fn service_name() -> &'static str;

    /// 插件展示名，默认取结构体短类型名（与 JS `class.name` 行为一致）。
    fn plugin_name() -> &'static str {
        short_type_name::<Self>()
    }

    /// 声明所依赖的服务名。
    fn inject() -> Inject {
        Inject::default()
    }

    /// 可选配置校验（对应 `static Config`）。
    fn validate(_config: &Self::Config) -> Result<(), ValidationError> {
        Ok(())
    }

    /// 构造服务实例。框架会在构造完成后自动 `ctx.provide(Self::service_name(), instance)`。
    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError>;

    /// 可选异步启动钩子（对应 `async [Service.init]()`），框架会等它完成后才进入 `ACTIVE`。
    fn init(self: Arc<Self>, _ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        Box::pin(async { Ok(()) })
    }

    /// 便捷构造方法：返回可直接传给 `ctx.plugin(...)` 的插件适配器。
    fn plugin() -> ServicePlugin<Self> {
        ServicePlugin::new()
    }
}

/// 将任意 `Service` 适配为 Cordis `Plugin`。
pub struct ServicePlugin<S>(PhantomData<fn() -> S>);

impl<S> Default for ServicePlugin<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Clone for ServicePlugin<S> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<S> ServicePlugin<S> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<S: Service> Plugin for ServicePlugin<S> {
    type Config = S::Config;

    fn name(&self) -> Option<&str> {
        Some(S::plugin_name())
    }

    fn inject(&self) -> Inject {
        S::inject()
    }

    fn validate(&self, config: &Self::Config) -> Result<(), ValidationError> {
        S::validate(config)
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let instance = S::create(&ctx, config)?;
            ctx.provide(S::service_name(), Arc::clone(&instance))?;
            instance.init(&ctx).await?;
            Ok(None)
        })
    }
}
