//! 迷你版 Cordis 插件框架（对应 `NamiDSH/packages/cordis`）。
//!
//! 保留核心机制：Context 树、Fiber 状态机、effect 记账与倒序撤销、服务依赖注入与通知、
//! 五种事件分发（emit / parallel / serial / bail / waterfall）、配置校验与错误隔离。

pub mod context;
pub mod events;
pub mod fiber;
pub mod logger;
pub mod reflect;
pub mod registry;
pub mod service;
pub mod utils;

pub use context::Context;
pub use events::{
    AggregateError, DispatchMode, EventOptions, EventsService, INTERNAL_DISPATCH, INTERNAL_PLUGIN,
    INTERNAL_SERVICE, INTERNAL_STATUS, InternalDispatchEvent, InternalPluginEvent,
    InternalServiceEvent, InternalStatusEvent, Next,
};
pub use fiber::{Cleanup, CordisError, DisposableHandle, Fiber, FiberState};
pub use logger::{LogHandler, LogLevel, LogRecord, LogSink, Logger};
pub use reflect::{ReflectService, ServiceImpl};
pub use registry::{
    Inject, IntoPluginReturn, Plugin, PluginDef, PluginKey, PluginRuntime, RegistryService,
    ValidationError, ValidationIssue,
};
pub use service::{Service, ServicePlugin};
pub use utils::{BoxError, BoxFuture, DisposableList, short_type_name};
