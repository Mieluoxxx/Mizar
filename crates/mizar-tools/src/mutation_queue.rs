//! 同路径文件写锁队列：同一路径的 `write`/`edit` 操作严格串行，不同文件仍可并发执行。

use std::collections::HashMap;
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use tokio::sync::Mutex as AsyncMutex;

static FILE_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>> = OnceLock::new();

fn lock_map() -> &'static Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>> {
    FILE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
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
    let file_mutex = {
        let mut map = lock_map().lock().unwrap();
        Arc::clone(
            map.entry(key.clone())
                .or_insert_with(|| Arc::new(AsyncMutex::new(()))),
        )
    };

    let _guard = file_mutex.lock().await;
    let out = op().await;
    drop(_guard);

    // 尽量清理已经无人持有的条目，避免长时间运行后 map 无限增长
    drop(file_mutex);
    let mut map = lock_map().lock().unwrap();
    if let Some(entry) = map.get(&key)
        && Arc::strong_count(entry) == 1
    {
        map.remove(&key);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

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
}
