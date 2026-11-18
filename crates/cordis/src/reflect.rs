//! 服务注册与依赖查找（对应 `packages/cordis/src/reflect.ts`）。
//!
//! JS 版通过 Proxy 实现 `ctx.llm` 读取；Rust 版通过 `ctx.service::<T>("llm")`
//! 及各服务包提供的扩展 trait（如 `ctx.llm()?`）实现完全一致的沿树查找与权限校验规则。

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};

use futures::future::join_all;

use crate::context::Context;
use crate::events::{INTERNAL_SERVICE, InternalServiceEvent};
use crate::fiber::{Cleanup, CordisError, DisposableHandle, Fiber, FiberInner, FiberState};

/// 一个服务实现：名字、值、提供者 Fiber 的编号与弱引用。
#[derive(Clone)]
pub struct ServiceImpl {
    pub name: String,
    pub value: Arc<dyn Any + Send + Sync>,
    pub fiber_uid: u64,
    pub(crate) fiber: Weak<FiberInner>,
}

impl ServiceImpl {
    pub fn fiber(&self) -> Option<Fiber> {
        self.fiber.upgrade().map(Fiber::from_inner)
    }
}

struct ReflectState {
    /// 全局服务表：名字 → 实现。同一名字同时只能有一个提供者。
    store: HashMap<String, ServiceImpl>,
    /// 出现过的服务名集合。
    props: HashSet<String>,
}

pub struct ReflectService {
    root: Weak<crate::context::RootContext>,
    state: Mutex<ReflectState>,
}

impl ReflectService {
    pub(crate) fn new(root: Weak<crate::context::RootContext>) -> Self {
        Self {
            root,
            state: Mutex::new(ReflectState {
                store: HashMap::new(),
                props: HashSet::new(),
            }),
        }
    }

    /// 判断某个服务名是否曾经注册过。
    pub fn has_prop(&self, name: &str) -> bool {
        self.state.lock().unwrap().props.contains(name)
    }

    /// 取出当前服务实现。`strict = true` 时只承认 `ACTIVE` 状态的提供者。
    pub(crate) fn get_impl(&self, name: &str, strict: bool) -> Option<ServiceImpl> {
        let impl_ = {
            let state = self.state.lock().unwrap();
            state.store.get(name).cloned()
        }?;
        if strict {
            let fiber = impl_.fiber()?;
            if fiber.state() != FiberState::Active {
                return None;
            }
        }
        Some(impl_)
    }

    /// 按名字取服务（无需 `inject`）。取不到或提供者未激活时返回 `None`。
    pub fn get<T: Send + Sync + 'static>(&self, name: &str) -> Option<Arc<T>> {
        self.get_with_strict(name, true)
    }

    pub(crate) fn get_with_strict<T: Send + Sync + 'static>(
        &self,
        name: &str,
        strict: bool,
    ) -> Option<Arc<T>> {
        let impl_ = self.get_impl(name, strict)?;
        impl_.value.downcast::<T>().ok()
    }

    /// 按 `ctx.xxx` Proxy 规则沿插件树向上查找已注入的服务：
    /// 1. 根 ctx 不受限制，直接从全局服务表取（不要求 ACTIVE）；
    /// 2. 子 ctx 沿父链查找 `fiber.store`；
    /// 3. 若声明了 `inject` 但还没拿到快照，报 `InactiveService`；
    /// 4. 走到根仍未找到且没声明 `inject`，报 `MissingInject`。
    pub(crate) fn resolve_for_ctx<T: Send + Sync + 'static>(
        &self,
        ctx: &Context,
        name: &str,
    ) -> Result<Arc<T>, CordisError> {
        if !ctx.fiber().has_runtime() {
            let impl_ = self
                .get_impl(name, false)
                .ok_or_else(|| CordisError::MissingInject(name.to_string()))?;
            return impl_.value.downcast::<T>().map_err(|_| CordisError::TypeMismatch {
                name: name.to_string(),
                expected: std::any::type_name::<T>(),
            });
        }

        let mut current = Some(ctx.fiber().clone());
        while let Some(fiber) = current {
            if let Some(impl_) = fiber.get_from_snapshot(name) {
                return impl_.value.downcast::<T>().map_err(|_| CordisError::TypeMismatch {
                    name: name.to_string(),
                    expected: std::any::type_name::<T>(),
                });
            }
            if fiber.has_inject(name) {
                return Err(CordisError::InactiveService(name.to_string()));
            }
            if !fiber.has_runtime() {
                return Err(CordisError::MissingInject(name.to_string()));
            }
            current = fiber.parent_fiber();
        }
        Err(CordisError::MissingInject(name.to_string()))
    }

    /// 列出属于某个 Fiber 的所有当前在线服务名。
    pub(crate) fn owned_service_names(&self, fiber: &Arc<FiberInner>) -> Vec<String> {
        let ptr = Arc::as_ptr(fiber);
        let state = self.state.lock().unwrap();
        state
            .store
            .iter()
            .filter(|(_, impl_)| Weak::as_ptr(&impl_.fiber) == ptr)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// 提供服务。本身是一个 effect：插件停止时自动下线。
    ///
    /// 下线顺序（先存盘，再拔电源）：
    /// 1. 从全局服务表删除；
    /// 2. 通知依赖方并等待它们清理完；
    /// 3. 最后才从提供者自己的快照里删掉。
    pub fn provide<T: Send + Sync + 'static>(
        self: &Arc<Self>,
        ctx: &Context,
        name: impl Into<String>,
        value: Arc<T>,
    ) -> Result<DisposableHandle, CordisError> {
        let name = name.into();
        let reflect = Arc::clone(self);
        let ctx_cloned = ctx.clone();
        let label = format!("ctx.provide({name:?})");

        ctx.fiber().effect(&label, move || {
            let fiber = ctx_cloned.fiber();
            let fiber_uid = fiber.uid().ok_or(CordisError::InactiveEffect)?;
            let impl_ = ServiceImpl {
                name: name.clone(),
                value: value as Arc<dyn Any + Send + Sync>,
                fiber_uid,
                fiber: fiber.downgrade(),
            };

            {
                let mut state = reflect.state.lock().unwrap();
                state.props.insert(name.clone());
                if let Some(existing) = state.store.get(&name) {
                    let owner = existing
                        .fiber()
                        .map(|f| f.name())
                        .unwrap_or_else(|| "unknown".to_string());
                    return Err(CordisError::DuplicateService {
                        name: name.clone(),
                        owner,
                    });
                }
                state.store.insert(name.clone(), impl_.clone());
            }

            fiber.insert_into_snapshot(&name, impl_);

            if fiber.state() == FiberState::Active {
                reflect.notify(&[name.as_str()]);
            }

            let reflect_cleanup = Arc::clone(&reflect);
            let fiber_weak = fiber.downgrade();
            let name_cleanup = name.clone();
            Ok(Some(Cleanup::async_fn(move || async move {
                {
                    let mut state = reflect_cleanup.state.lock().unwrap();
                    state.store.remove(&name_cleanup);
                }
                let fibers = reflect_cleanup.notify(&[name_cleanup.as_str()]);
                let waits = fibers.into_iter().map(|f| async move {
                    let _ = f.wait_settled().await;
                });
                join_all(waits).await;
                if let Some(f) = fiber_weak.upgrade() {
                    Fiber::from_inner(f).remove_from_snapshot(&name_cleanup);
                }
            })))
        })
    }

    /// 服务变化时唤醒所有 `inject` 了这些服务名的插件重新计算 epoch。
    pub fn notify(&self, names: &[&str]) -> Vec<Fiber> {
        let mut affected = Vec::new();
        let Some(root) = self.root.upgrade() else {
            return affected;
        };

        let runtimes = root.registry.values();
        for runtime in runtimes {
            let fibers = runtime.fibers();
            for fiber in fibers {
                let mut has_update = false;
                for &name in names {
                    if !fiber.has_inject(name) {
                        continue;
                    }
                    has_update = true;
                    fiber.check_impl(name);
                }
                if !has_update {
                    continue;
                }
                fiber.refresh();
                affected.push(fiber);
            }
        }

        for &name in names {
            let available = self.get_impl(name, false).is_some();
            root.events.emit(
                INTERNAL_SERVICE,
                InternalServiceEvent {
                    name: name.to_string(),
                    available,
                },
            );
        }

        affected
    }
}
