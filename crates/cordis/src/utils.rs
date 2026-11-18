//! 框架内部使用的工具类型与辅助函数（对应 `packages/cordis/src/utils.ts`）。

use std::collections::BTreeMap;
use std::error::Error;
use std::future::Future;
use std::pin::Pin;

/// 统一异步 Future 别名。
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// 统一动态错误类型。
pub type BoxError = Box<dyn Error + Send + Sync>;

/// 可撤销列表：`push` 返回唯一序号，`clear` 按**倒序**弹出（后注册的先撤销）。
///
/// Fiber 用它记录「这个插件做过的所有注册」。
#[derive(Debug)]
pub struct DisposableList<T> {
    sn: u64,
    map: BTreeMap<u64, T>,
}

impl<T> Default for DisposableList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DisposableList<T> {
    pub fn new() -> Self {
        Self {
            sn: 0,
            map: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// 注册一条记录，返回可用于 `remove` 的唯一序号。
    pub fn push(&mut self, value: T) -> u64 {
        self.sn += 1;
        let sn = self.sn;
        self.map.insert(sn, value);
        sn
    }

    /// 按序号移除单条记录。
    pub fn remove(&mut self, sn: u64) -> Option<T> {
        self.map.remove(&sn)
    }

    /// 清空并返回所有元素，**倒序**排列（后注册的先撤销）。
    pub fn clear(&mut self) -> Vec<T> {
        let map = std::mem::take(&mut self.map);
        map.into_values().rev().collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.map.values()
    }
}

impl<T: Clone> DisposableList<T> {
    /// 按正序克隆当前全部元素快照。
    pub fn snapshot(&self) -> Vec<T> {
        self.map.values().cloned().collect()
    }
}

/// 获取不含模块路径的短类型名，用作 `Service` 默认插件名。
pub fn short_type_name<T>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full)
}
