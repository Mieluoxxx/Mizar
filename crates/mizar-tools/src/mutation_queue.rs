//! 同路径文件写锁队列：同一路径的 `write`/`edit` 操作严格串行，不同文件仍可并发执行。

use std::collections::HashMap;
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use tokio::sync::Mutex as AsyncMutex;

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
