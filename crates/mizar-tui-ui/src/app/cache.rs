//! 对话流虚拟化渲染双层缓存（行数索引 + LRU 富文本行缓存 + 流式 Markdown 前缀增量缓存）。

use std::collections::HashMap;
use std::sync::Arc;

use crate::app::ChatItem;
use crate::markdown::RenderedLine;

/// 单个消息条目的行级渲染缓存键。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemRenderKey {
    pub width: usize,
    pub theme_id: &'static str,
    pub emphasis_enabled: bool,
    pub round_caps: bool,
    pub expanded: bool,
    pub status_code: u8,
    pub reasoning_len: usize,
    pub text_len: usize,
    pub preview_len: usize,
}

pub const MAX_CACHED_RENDER_ITEMS: usize = 96;

#[derive(Debug, Clone)]
pub struct CachedItemLines {
    pub key: ItemRenderKey,
    pub lines: Arc<Vec<RenderedLine>>,
}

#[derive(Debug, Clone)]
struct CachedItemHeight {
    key: ItemRenderKey,
    line_count: usize,
}

#[derive(Debug, Clone)]
struct CachedToolExpandable {
    width: usize,
    status_code: u8,
    extra_len: usize,
    preview_len: usize,
    expandable: bool,
}

#[derive(Debug, Clone)]
struct LruRenderEntry {
    item: CachedItemLines,
    last_used: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct StreamingMarkdownPrefixCache {
    pub item_idx: usize,
    pub width: usize,
    pub theme_id: &'static str,
    pub emphasis_enabled: bool,
    pub round_caps: bool,
    pub prefix_text: String,
    pub lines: Vec<RenderedLine>,
}

/// 对话渲染双层缓存：
/// - `heights` 仅保存每个条目的渲染行数（轻量常驻），支撑 O(N) 整数扫描与视口裁剪；
/// - `entries` 用 LRU 上限仅缓存视口工作集附近的完整 `Arc<Vec<RenderedLine>>` 高亮结果；
/// - `expandable` 缓存工具卡展开性判定，避免跨帧重复反序列化 JSON 与渲染 Diff。
#[derive(Debug, Default, Clone)]
pub struct TranscriptRenderCache {
    heights: HashMap<usize, CachedItemHeight>,
    entries: HashMap<usize, LruRenderEntry>,
    expandable: HashMap<usize, CachedToolExpandable>,
    pub(crate) span_scratch: Vec<(usize, usize, ItemRenderKey)>,
    pub(crate) streaming_prefix: Option<StreamingMarkdownPrefixCache>,
    clock: u64,
}

impl TranscriptRenderCache {
    pub fn clear(&mut self) {
        self.heights.clear();
        self.entries.clear();
        self.expandable.clear();
        self.streaming_prefix = None;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.heights.is_empty()
    }

    pub fn height_count(&self) -> usize {
        self.heights.len()
    }

    /// 局部失效指定条目（及其相邻工具组），替代全表 `clear()`。
    pub fn invalidate_item(&mut self, idx: usize) {
        self.heights.remove(&idx);
        self.entries.remove(&idx);
        self.expandable.remove(&idx);
        if self
            .streaming_prefix
            .as_ref()
            .is_some_and(|p| p.item_idx == idx)
        {
            self.streaming_prefix = None;
        }
    }

    pub fn invalidate_tool_neighborhood(&mut self, items: &[ChatItem], idx: usize) {
        self.invalidate_item(idx);
        // 向前失效同一连续工具组（前驱的 `is_group_end` / `show_expand_hint` 可能变化）
        let mut cur = idx;
        while cur > 0 {
            cur -= 1;
            if matches!(items.get(cur), Some(ChatItem::ToolCall { .. })) {
                self.invalidate_item(cur);
            } else {
                break;
            }
        }
        // 向后失效同一连续工具组（后续卡片可能是 group_end，受当前展开性变化影响）
        let mut next = idx + 1;
        while matches!(items.get(next), Some(ChatItem::ToolCall { .. })) {
            self.invalidate_item(next);
            next += 1;
        }
    }

    pub fn get_height(&self, idx: usize, key: &ItemRenderKey) -> Option<usize> {
        self.heights
            .get(&idx)
            .filter(|h| &h.key == key)
            .map(|h| h.line_count)
    }

    pub fn get_lines(&mut self, idx: usize, key: &ItemRenderKey) -> Option<Arc<Vec<RenderedLine>>> {
        let matched = self.entries.get(&idx).is_some_and(|e| &e.item.key == key);
        if !matched {
            return None;
        }
        self.clock = self.clock.wrapping_add(1);
        let stamp = self.clock;
        let entry = self.entries.get_mut(&idx)?;
        entry.last_used = stamp;
        Some(Arc::clone(&entry.item.lines))
    }

    pub fn record_height_only(&mut self, idx: usize, key: ItemRenderKey, line_count: usize) {
        self.heights
            .insert(idx, CachedItemHeight { key, line_count });
    }

    pub fn insert_lines(&mut self, idx: usize, key: ItemRenderKey, lines: Arc<Vec<RenderedLine>>) {
        self.heights.insert(
            idx,
            CachedItemHeight {
                key: key.clone(),
                line_count: lines.len(),
            },
        );
        self.clock = self.clock.wrapping_add(1);
        let stamp = self.clock;
        self.entries.insert(
            idx,
            LruRenderEntry {
                item: CachedItemLines { key, lines },
                last_used: stamp,
            },
        );
        if self.entries.len() > MAX_CACHED_RENDER_ITEMS
            && let Some((&evict_idx, _)) = self.entries.iter().min_by_key(|(_, e)| e.last_used)
        {
            self.entries.remove(&evict_idx);
        }
    }

    pub fn get_expandable(
        &self,
        idx: usize,
        width: usize,
        status_code: u8,
        extra_len: usize,
        preview_len: usize,
    ) -> Option<bool> {
        self.expandable.get(&idx).and_then(|c| {
            (c.width == width
                && c.status_code == status_code
                && c.extra_len == extra_len
                && c.preview_len == preview_len)
                .then_some(c.expandable)
        })
    }

    pub fn insert_expandable(
        &mut self,
        idx: usize,
        width: usize,
        status_code: u8,
        extra_len: usize,
        preview_len: usize,
        expandable: bool,
    ) {
        self.expandable.insert(
            idx,
            CachedToolExpandable {
                width,
                status_code,
                extra_len,
                preview_len,
                expandable,
            },
        );
    }
}
