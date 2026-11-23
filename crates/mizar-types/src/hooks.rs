//! 轻量级强类型钩子原语与 RAII 清理守卫。
//!
//! 取代原微内核动态事件总线：提供 `SyncEventBus`（同步广播）、
//! `WaterfallHook`（洋葱拦截链）、`SerialHook`（异步惰性短路链）与 `HookGuard`（RAII 析构注销）。

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// RAII 钩子/清理守卫：离开作用域或显式 `drop` 时立即执行注销或清理逻辑。
#[must_use = "dropping HookGuard unregisters the hook or cleanup immediately"]
pub struct HookGuard {
    cleanup: Mutex<Option<Box<dyn FnOnce() + Send + Sync>>>,
}

impl HookGuard {
    pub fn new(cleanup: impl FnOnce() + Send + Sync + 'static) -> Self {
        Self {
            cleanup: Mutex::new(Some(Box::new(cleanup))),
        }
    }

    pub fn noop() -> Self {
        Self {
            cleanup: Mutex::new(None),
        }
    }

    /// 取消析构时的自动清理（常用于把生命周期提升为静态常驻）。
    pub fn detach(&mut self) {
        let _ = self.cleanup.lock().unwrap().take();
    }

    /// 合并多个守卫；释放时按**逆序（LIFO）**逐个清理。
    pub fn group(guards: Vec<HookGuard>) -> Self {
        if guards.is_empty() {
            return Self::noop();
        }
        Self::new(move || {
            for guard in guards.into_iter().rev() {
                drop(guard);
            }
        })
    }
}

impl Default for HookGuard {
    fn default() -> Self {
        Self::noop()
    }
}

impl std::fmt::Debug for HookGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let armed = self.cleanup.lock().unwrap().is_some();
        f.debug_struct("HookGuard").field("armed", &armed).finish()
    }
}

impl Drop for HookGuard {
    fn drop(&mut self) {
        if let Some(cb) = self.cleanup.lock().unwrap().take() {
            cb();
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HookOptions {
    /// 插到同优先级区段之首，而不是之尾。
    pub prepend: bool,
    /// 调度优先级（数值越小越先执行，默认 `0`）。
    pub priority: i32,
}

type EventSubscriber<E> = Arc<dyn Fn(&E) + Send + Sync>;

struct EventBusState<E> {
    entries: Vec<(u64, EventSubscriber<E>)>,
    snapshot: Arc<[EventSubscriber<E>]>,
}

struct SyncEventBusInner<E> {
    next_id: AtomicU64,
    subscriber_count: AtomicUsize,
    state: RwLock<EventBusState<E>>,
}

/// 线程安全的强类型同步事件广播总线（零订阅快速跳过，派发前释放读锁防重入死锁）。
pub struct SyncEventBus<E> {
    inner: Arc<SyncEventBusInner<E>>,
}

impl<E: 'static> Default for SyncEventBus<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: 'static> SyncEventBus<E> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(SyncEventBusInner {
                next_id: AtomicU64::new(1),
                subscriber_count: AtomicUsize::new(0),
                state: RwLock::new(EventBusState {
                    entries: Vec::new(),
                    snapshot: Arc::from([]),
                }),
            }),
        }
    }

    pub fn subscriber_count(&self) -> usize {
        self.inner.subscriber_count.load(Ordering::Acquire)
    }

    pub fn on(&self, cb: impl Fn(&E) + Send + Sync + 'static) -> HookGuard {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let cb: Arc<dyn Fn(&E) + Send + Sync> = Arc::new(cb);
        {
            let mut st = self.inner.state.write().unwrap();
            st.entries.push((id, cb));
            st.snapshot = Arc::from(
                st.entries
                    .iter()
                    .map(|(_, f)| Arc::clone(f))
                    .collect::<Vec<_>>(),
            );
            self.inner.subscriber_count.fetch_add(1, Ordering::Release);
        }

        let weak = Arc::downgrade(&self.inner);
        HookGuard::new(move || {
            let Some(inner) = weak.upgrade() else {
                return;
            };
            let mut st = inner.state.write().unwrap();
            if let Some(idx) = st.entries.iter().position(|(eid, _)| *eid == id) {
                st.entries.remove(idx);
                st.snapshot = Arc::from(
                    st.entries
                        .iter()
                        .map(|(_, f)| Arc::clone(f))
                        .collect::<Vec<_>>(),
                );
                inner.subscriber_count.fetch_sub(1, Ordering::AcqRel);
            }
        })
    }

    pub fn emit(&self, ev: &E) {
        if self.inner.subscriber_count.load(Ordering::Acquire) == 0 {
            return;
        }
        let snapshot = {
            let st = self.inner.state.read().unwrap();
            Arc::clone(&st.snapshot)
        };
        for cb in snapshot.iter() {
            cb(ev);
        }
    }
}

/// 洋葱模型（`WaterfallHook`）的下一层调用句柄。
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

struct OrderedEntry<F> {
    id: u64,
    priority: i32,
    func: F,
}

fn insert_ordered<F: Clone>(
    entries: &mut Vec<OrderedEntry<F>>,
    entry: OrderedEntry<F>,
    opts: HookOptions,
) -> Arc<[F]> {
    let idx = if opts.prepend {
        entries
            .iter()
            .position(|e| e.priority >= opts.priority)
            .unwrap_or(entries.len())
    } else {
        entries
            .iter()
            .position(|e| e.priority > opts.priority)
            .unwrap_or(entries.len())
    };
    entries.insert(idx, entry);
    Arc::from(entries.iter().map(|e| e.func.clone()).collect::<Vec<_>>())
}

fn remove_ordered<F: Clone>(entries: &mut Vec<OrderedEntry<F>>, id: u64) -> Option<Arc<[F]>> {
    let idx = entries.iter().position(|e| e.id == id)?;
    entries.remove(idx);
    Some(Arc::from(
        entries.iter().map(|e| e.func.clone()).collect::<Vec<_>>(),
    ))
}

type WaterfallFn<I, O> = Arc<dyn Fn(I, Next<I, O>) -> O + Send + Sync>;
type OrderedBucketState<F> = (Vec<OrderedEntry<F>>, Arc<[F]>);

struct HookBucket<F> {
    next_id: AtomicU64,
    state: RwLock<OrderedBucketState<F>>,
}

impl<F: Clone + Send + Sync + 'static> HookBucket<F> {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            next_id: AtomicU64::new(1),
            state: RwLock::new((Vec::new(), Arc::from([]))),
        })
    }

    fn insert(self: &Arc<Self>, opts: HookOptions, func: F) -> HookGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut st = self.state.write().unwrap();
            let snapshot = insert_ordered(
                &mut st.0,
                OrderedEntry {
                    id,
                    priority: opts.priority,
                    func,
                },
                opts,
            );
            st.1 = snapshot;
        }
        let weak = Arc::downgrade(self);
        HookGuard::new(move || {
            let Some(bucket) = weak.upgrade() else {
                return;
            };
            let mut st = bucket.state.write().unwrap();
            if let Some(snapshot) = remove_ordered(&mut st.0, id) {
                st.1 = snapshot;
            }
        })
    }

    fn snapshot(&self) -> Arc<[F]> {
        Arc::clone(&self.state.read().unwrap().1)
    }
}

/// 强类型洋葱模型拦截链：支持同步流返回或异步 `BoxFuture<'static, T>`。
pub struct WaterfallHook<I, O> {
    bucket: Arc<HookBucket<WaterfallFn<I, O>>>,
}

impl<I: 'static, O: 'static> Default for WaterfallHook<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: 'static, O: 'static> WaterfallHook<I, O> {
    pub fn new() -> Self {
        Self {
            bucket: HookBucket::new(),
        }
    }

    pub fn register(&self, f: impl Fn(I, Next<I, O>) -> O + Send + Sync + 'static) -> HookGuard {
        self.register_with(HookOptions::default(), f)
    }

    pub fn register_with(
        &self,
        opts: HookOptions,
        f: impl Fn(I, Next<I, O>) -> O + Send + Sync + 'static,
    ) -> HookGuard {
        self.bucket.insert(opts, Arc::new(f))
    }

    pub fn call(&self, input: I, inner: impl FnOnce(I) -> O + Send + 'static) -> O
    where
        I: Send + 'static,
        O: Send + 'static,
    {
        let snapshot = self.bucket.snapshot();
        if snapshot.is_empty() {
            return inner(input);
        }
        let mut current = Next::new(inner);
        for cb in snapshot.iter().rev() {
            let cb = Arc::clone(cb);
            let next = current;
            current = Next::new(move |val| cb(val, next));
        }
        current.call(input)
    }
}

impl<I: Send + 'static, T: Send + 'static> WaterfallHook<I, BoxFuture<'static, T>> {
    pub async fn run(
        &self,
        input: I,
        inner: impl FnOnce(I) -> BoxFuture<'static, T> + Send + 'static,
    ) -> T {
        self.call(input, inner).await
    }
}

type SerialFn<I, O> = Arc<dyn Fn(&I) -> BoxFuture<'static, Option<O>> + Send + Sync>;

/// 串行异步短路链：按优先级惰性依次调用，首个返回 `Some(out)` 即停止后续调用。
pub struct SerialHook<I, O> {
    bucket: Arc<HookBucket<SerialFn<I, O>>>,
}

impl<I: 'static, O: 'static> Default for SerialHook<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: 'static, O: 'static> SerialHook<I, O> {
    pub fn new() -> Self {
        Self {
            bucket: HookBucket::new(),
        }
    }

    pub fn register(
        &self,
        f: impl Fn(&I) -> BoxFuture<'static, Option<O>> + Send + Sync + 'static,
    ) -> HookGuard {
        self.register_with(HookOptions::default(), f)
    }

    pub fn register_with(
        &self,
        opts: HookOptions,
        f: impl Fn(&I) -> BoxFuture<'static, Option<O>> + Send + Sync + 'static,
    ) -> HookGuard {
        self.bucket.insert(opts, Arc::new(f))
    }

    pub async fn run(&self, input: I) -> Option<O> {
        let snapshot = self.bucket.snapshot();
        for cb in snapshot.iter() {
            if let Some(out) = cb(&input).await {
                return Some(out);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::task::{Context, Poll, Waker};

    fn block_on_ready<F: Future>(fut: F) -> F::Output {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut fut = std::pin::pin!(fut);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("test future unexpectedly returned Pending"),
        }
    }

    #[test]
    fn hook_guard_drop_detach_and_reverse_group_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        {
            let l1 = Arc::clone(&log);
            let l2 = Arc::clone(&log);
            let l3 = Arc::clone(&log);
            let mut detached = HookGuard::new(move || l3.lock().unwrap().push(99));
            detached.detach();
            let _group = HookGuard::group(vec![
                HookGuard::new(move || l1.lock().unwrap().push(1)),
                HookGuard::new(move || l2.lock().unwrap().push(2)),
            ]);
            assert!(log.lock().unwrap().is_empty());
        }
        assert_eq!(*log.lock().unwrap(), vec![2, 1]);
    }

    #[test]
    fn sync_event_bus_unregisters_and_allows_reentrant_mutations() {
        let bus = SyncEventBus::<i32>::new();
        assert_eq!(bus.subscriber_count(), 0);
        bus.emit(&1);

        let received = Arc::new(Mutex::new(Vec::new()));
        let rec1 = Arc::clone(&received);
        let g1 = bus.on(move |v| rec1.lock().unwrap().push(*v));

        let holder: Arc<Mutex<Option<HookGuard>>> = Arc::new(Mutex::new(None));
        let holder_ref = Arc::clone(&holder);
        let g2 = bus.on(move |_| {
            let _ = holder_ref.lock().unwrap().take();
        });
        let rec3 = Arc::clone(&received);
        *holder.lock().unwrap() = Some(bus.on(move |v| rec3.lock().unwrap().push(*v * 10)));

        assert_eq!(bus.subscriber_count(), 3);
        bus.emit(&7);
        assert_eq!(*received.lock().unwrap(), vec![7, 70]);
        assert_eq!(bus.subscriber_count(), 2);

        drop(g1);
        drop(g2);
        assert_eq!(bus.subscriber_count(), 0);
        bus.emit(&9);
        assert_eq!(*received.lock().unwrap(), vec![7, 70]);
    }

    #[test]
    fn waterfall_hook_preserves_priority_prepend_and_short_circuit() {
        let hook = WaterfallHook::<Vec<String>, Vec<String>>::new();
        let _g_default = hook.register(|mut v, next| {
            v.push("default-1".into());
            next.call(v)
        });
        let _g_default_2 = hook.register(|mut v, next| {
            v.push("default-2".into());
            next.call(v)
        });
        let _g_prepend = hook.register_with(
            HookOptions {
                prepend: true,
                priority: 0,
            },
            |mut v, next| {
                v.push("prepend-0".into());
                next.call(v)
            },
        );
        let _g_low_prio = hook.register_with(
            HookOptions {
                prepend: false,
                priority: -20,
            },
            |mut v, next| {
                v.push("prio--20".into());
                next.call(v)
            },
        );
        let _g_high_prio = hook.register_with(
            HookOptions {
                prepend: false,
                priority: 10,
            },
            |mut v, next| {
                v.push("prio-10".into());
                next.call(v)
            },
        );

        let trace = hook.call(Vec::new(), |mut v| {
            v.push("inner".into());
            v
        });
        assert_eq!(
            trace,
            vec![
                "prio--20",
                "prepend-0",
                "default-1",
                "default-2",
                "prio-10",
                "inner",
            ]
        );

        let _g_blocker = hook.register_with(
            HookOptions {
                prepend: true,
                priority: -100,
            },
            |mut v, _next| {
                v.push("blocked".into());
                v
            },
        );
        let blocked = hook.call(Vec::new(), |mut v| {
            v.push("inner".into());
            v
        });
        assert_eq!(blocked, vec!["blocked"]);
    }

    #[test]
    fn waterfall_async_and_serial_hook_lazy_short_circuit() {
        let wf = WaterfallHook::<i32, BoxFuture<'static, i32>>::new();
        let _g = wf.register(|x, next| Box::pin(async move { next.call(x + 1).await * 2 }));
        let res = block_on_ready(wf.run(3, |x| Box::pin(async move { x + 10 })));
        assert_eq!(res, 28);

        let serial = SerialHook::<i32, &'static str>::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let c1 = Arc::clone(&calls);
        let _s1 = serial.register_with(
            HookOptions {
                prepend: false,
                priority: 10,
            },
            move |_| {
                let c = Arc::clone(&c1);
                Box::pin(async move {
                    c.fetch_add(1, Ordering::Relaxed);
                    Some("late")
                })
            },
        );
        let c2 = Arc::clone(&calls);
        let _s2 = serial.register_with(
            HookOptions {
                prepend: false,
                priority: -5,
            },
            move |v| {
                let c = Arc::clone(&c2);
                let val = *v;
                Box::pin(async move {
                    c.fetch_add(1, Ordering::Relaxed);
                    (val > 0).then_some("early")
                })
            },
        );

        let out = block_on_ready(serial.run(5));
        assert_eq!(out, Some("early"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
