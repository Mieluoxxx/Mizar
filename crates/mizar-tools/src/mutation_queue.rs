//! 同路径文件写锁队列：同一路径的 `write`/`edit` 操作严格串行，不同文件仍可并发执行。

use std::collections::HashMap;
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use tokio::sync::Mutex as AsyncMutex;

#[cfg(all(mizar_loom, test))]
mod sync {
    pub(super) use loom::sync::{Arc, Mutex};
}

#[cfg(not(all(mizar_loom, test)))]
mod sync {
    pub(super) use std::sync::{Arc, Mutex};
}

struct SlotEntry<L> {
    lock: sync::Arc<L>,
    waiters: usize,
}

/// 按键串行化的 RAII 租约锁表：正常返回、Future 取消或 panic 均会严格先释放锁再回收空槽位。
pub(crate) struct KeyedLockTable<K, L> {
    entries: sync::Mutex<HashMap<K, SlotEntry<L>>>,
}

pub(crate) struct SlotLease<'a, K: Eq + std::hash::Hash + Clone, L> {
    table: &'a KeyedLockTable<K, L>,
    key: K,
    lock: Option<sync::Arc<L>>,
}

impl<K: Eq + std::hash::Hash + Clone, L> KeyedLockTable<K, L> {
    pub(crate) fn new() -> Self {
        Self {
            entries: sync::Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn acquire(&self, key: K, make_lock: impl FnOnce() -> L) -> SlotLease<'_, K, L> {
        let lock = {
            let mut map = self.entries.lock().unwrap();
            let entry = map.entry(key.clone()).or_insert_with(|| SlotEntry {
                lock: sync::Arc::new(make_lock()),
                waiters: 0,
            });
            entry.waiters += 1;
            sync::Arc::clone(&entry.lock)
        };
        SlotLease {
            table: self,
            key,
            lock: Some(lock),
        }
    }

    #[cfg(all(mizar_loom, test))]
    pub(crate) fn active_key_count(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    #[cfg(all(test, not(mizar_loom)))]
    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.entries.lock().unwrap().contains_key(key)
    }
}

impl<K: Eq + std::hash::Hash + Clone, L> SlotLease<'_, K, L> {
    pub(crate) fn lock(&self) -> &L {
        self.lock.as_deref().expect("slot lease already dropped")
    }
}

impl<K: Eq + std::hash::Hash + Clone, L> Drop for SlotLease<'_, K, L> {
    fn drop(&mut self) {
        // 先在表锁外卸掉内部锁引用，避免持表锁期间触发外部状态清理
        drop(self.lock.take());
        let mut map = self.table.entries.lock().unwrap();
        if let Some(entry) = map.get_mut(&self.key) {
            entry.waiters -= 1;
            if entry.waiters == 0 {
                map.remove(&self.key);
            }
        }
    }
}

static FILE_LOCKS: OnceLock<KeyedLockTable<PathBuf, AsyncMutex<()>>> = OnceLock::new();

fn lock_table() -> &'static KeyedLockTable<PathBuf, AsyncMutex<()>> {
    FILE_LOCKS.get_or_init(KeyedLockTable::new)
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(out.components().next_back(), Some(Component::Normal(_))) {
                    out.pop();
                } else if !matches!(
                    out.components().next_back(),
                    Some(Component::RootDir | Component::Prefix(_))
                ) {
                    out.push(Component::ParentDir);
                }
            }
            Component::Normal(s) => out.push(s),
        }
    }
    out
}

async fn mutation_queue_key(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else if let Ok(cwd) = std::env::current_dir() {
        cwd.join(path)
    } else {
        path.to_path_buf()
    };
    let lexical = normalize_lexical(&abs);

    if let Ok(canon) = tokio::fs::canonicalize(&lexical).await {
        return canon;
    }
    // 文件尚未创建时，尝试解析其已存在的最近祖先目录，避免软链接指向同一目录却产生不同 key
    if let (Some(parent), Some(name)) = (lexical.parent(), lexical.file_name())
        && let Ok(canon_parent) = tokio::fs::canonicalize(parent).await
    {
        return canon_parent.join(name);
    }
    lexical
}

/// 按路径串行化执行文件变更操作。
pub async fn with_file_mutation_queue<F, Fut, T>(path: &Path, op: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    let key = mutation_queue_key(path).await;
    // 严格先绑定 lease 再获取异步锁：局部变量按逆序 Drop，确保完成、取消或 panic 时均先释放锁再回收表项
    let lease = lock_table().acquire(key, || AsyncMutex::new(()));
    let _guard = lease.lock().lock().await;
    op().await
}

#[cfg(all(test, not(mizar_loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::Notify;

    #[tokio::test]
    async fn serializes_same_path_and_allows_parallel_different_paths() {
        let dir = std::env::temp_dir().join(format!("mizar_mq_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file_a = dir.join("a.txt");
        let file_b = dir.join("b.txt");
        std::fs::write(&file_a, "a").unwrap();
        std::fs::write(&file_b, "b").unwrap();

        let in_flight_a = Arc::new(AtomicUsize::new(0));
        let max_in_flight_a = Arc::new(AtomicUsize::new(0));

        let mut tasks = Vec::new();
        for i in 0..4 {
            let p = if i < 3 {
                file_a.clone()
            } else {
                file_b.clone()
            };
            let cur = Arc::clone(&in_flight_a);
            let max = Arc::clone(&max_in_flight_a);
            let is_a = i < 3;
            tasks.push(tokio::spawn(async move {
                with_file_mutation_queue(&p, || async move {
                    if is_a {
                        let now = cur.fetch_add(1, Ordering::SeqCst) + 1;
                        max.fetch_max(now, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(15)).await;
                        cur.fetch_sub(1, Ordering::SeqCst);
                    } else {
                        tokio::time::sleep(Duration::from_millis(15)).await;
                    }
                })
                .await;
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(max_in_flight_a.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn cancelled_waiter_and_holder_drain_lock_table() {
        let dir = std::env::temp_dir().join(format!("mizar_mq_cancel_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file_c = dir.join("c.txt");
        std::fs::write(&file_c, "c").unwrap();
        let canon_c = mutation_queue_key(&file_c).await;

        let entered = Arc::new(Notify::new());
        let entered_holder = Arc::clone(&entered);
        let path_holder = file_c.clone();
        let holder = tokio::spawn(async move {
            with_file_mutation_queue(&path_holder, || async move {
                entered_holder.notify_one();
                std::future::pending::<()>().await;
            })
            .await;
        });

        entered.notified().await;
        assert!(lock_table().contains_key(&canon_c));

        let waiter_entered = Arc::new(AtomicUsize::new(0));
        let waiter_entered_flag = Arc::clone(&waiter_entered);
        let path_waiter = file_c.clone();
        let waiter = tokio::spawn(async move {
            with_file_mutation_queue(&path_waiter, || async move {
                waiter_entered_flag.store(1, Ordering::SeqCst);
            })
            .await;
        });
        tokio::task::yield_now().await;

        waiter.abort();
        let waiter_res = waiter.await;
        assert!(waiter_res.unwrap_err().is_cancelled());
        assert_eq!(waiter_entered.load(Ordering::SeqCst), 0);

        holder.abort();
        let holder_res = holder.await;
        assert!(holder_res.unwrap_err().is_cancelled());
        assert!(
            !lock_table().contains_key(&canon_c),
            "cancelled holder/waiter leaked path entry"
        );

        let ran = with_file_mutation_queue(&file_c, || async { 42usize }).await;
        assert_eq!(ran, 42);
        assert!(!lock_table().contains_key(&canon_c));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(all(mizar_loom, test))]
mod loom_tests {
    use super::*;
    use loom::cell::UnsafeCell;
    use loom::model::Builder;
    use loom::sync::atomic::{AtomicUsize, Ordering};
    use loom::thread;

    #[test]
    fn loom_same_key_strictly_serialized_and_table_drains() {
        loom::model(|| {
            let table = sync::Arc::new(KeyedLockTable::<u32, sync::Mutex<()>>::new());
            let in_flight = sync::Arc::new(AtomicUsize::new(0));
            let mut handles = Vec::new();
            for _ in 0..2 {
                let t = sync::Arc::clone(&table);
                let cur = sync::Arc::clone(&in_flight);
                handles.push(thread::spawn(move || {
                    let lease = t.acquire(42, || sync::Mutex::new(()));
                    let _guard = lease.lock().lock().unwrap();
                    let n = cur.fetch_add(1, Ordering::SeqCst) + 1;
                    assert_eq!(n, 1, "same key entered concurrently");
                    cur.fetch_sub(1, Ordering::SeqCst);
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
            assert_eq!(table.active_key_count(), 0);
        });
    }

    #[test]
    fn loom_early_drop_before_lock_does_not_leak_or_break_mutual_exclusion() {
        let mut builder = Builder::new();
        builder.preemption_bound = Some(2);
        builder.check(|| {
            let table = sync::Arc::new(KeyedLockTable::<u32, sync::Mutex<()>>::new());
            let counter = sync::Arc::new(UnsafeCell::new(0usize));
            let mut handles = Vec::new();

            for _ in 0..2 {
                let t = sync::Arc::clone(&table);
                let cell = sync::Arc::clone(&counter);
                handles.push(thread::spawn(move || {
                    let lease = t.acquire(7, || sync::Mutex::new(()));
                    let _guard = lease.lock().lock().unwrap();
                    // 仅在持有路径互斥锁后改写 UnsafeCell，任何互斥失效都会被 loom 立即捕获为数据竞争
                    cell.with_mut(|ptr| unsafe {
                        *ptr += 1;
                    });
                }));
            }

            let t_cancel = sync::Arc::clone(&table);
            handles.push(thread::spawn(move || {
                // 模拟等待者或取消方拿到租约后未进入临界区即丢弃
                let lease = t_cancel.acquire(7, || sync::Mutex::new(()));
                drop(lease);
            }));

            for h in handles {
                h.join().unwrap();
            }
            counter.with(|ptr| unsafe {
                assert_eq!(*ptr, 2);
            });
            assert_eq!(table.active_key_count(), 0);
        });
    }
}
