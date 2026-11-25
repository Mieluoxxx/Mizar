//! 客户端消息排队队列：入队、唯一 ID 出队、草稿交换撤回与选择器游标导航。
//!
//! 迁移自 `mizar-tui-ui/src/app.rs` 的排队与选择器语义，供 TUI 与 Merak 共用。

use mizar_llm_types::ImageAttachment;

use crate::types::QueuedPrompt;

/// 客户端待发送消息队列。
///
/// 用单调递增 `id` 而非下标寻址：后端完成一轮后的自动出队会瞬间改变向量下标，
/// 避免选择器「高亮第二条却撤回第三条」。
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct PromptQueue {
    items: Vec<QueuedPrompt>,
    selected: Option<u64>,
    next_id: u64,
}

impl PromptQueue {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc_id(&mut self) -> u64 {
        // `Default` 将 `next_id` 初始化为 0；从 1 起分配可把 0 保留为无效哨兵值。
        let id = self.next_id.max(1);
        self.next_id = id.wrapping_add(1).max(1);
        id
    }

    fn sync_selection(&mut self) {
        if let Some(selected) = self.selected
            && !self.items.iter().any(|item| item.id == selected)
        {
            self.selected = None;
        }
    }

    /// 推入一条待发送消息并清空当前选择器高亮，返回分配的唯一条目 ID。
    pub fn enqueue(
        &mut self,
        text: impl Into<String>,
        images: Vec<(String, ImageAttachment)>,
    ) -> u64 {
        let id = self.alloc_id();
        self.items.push(QueuedPrompt {
            id,
            text: text.into(),
            images,
        });
        self.selected = None;
        id
    }

    /// 直接压入已带有 `id` 的队列项（例如恢复持久化队列），同时推进 `next_id` 防止后续碰撞。
    pub fn push_raw(&mut self, item: QueuedPrompt) {
        self.next_id = self.next_id.max(item.id.saturating_add(1));
        self.items.push(item);
    }

    /// 按唯一 ID 出队；若被移除项正好被选中，自动清空 `selected`。
    pub fn dequeue_by_id(&mut self, id: u64) -> Option<QueuedPrompt> {
        let index = self.items.iter().position(|item| item.id == id)?;
        let item = self.items.remove(index);
        self.sync_selection();
        Some(item)
    }

    /// 取出队首一条待发送消息。
    pub fn dequeue_next(&mut self) -> Option<QueuedPrompt> {
        let id = self.items.first()?.id;
        self.dequeue_by_id(id)
    }

    /// 把指定队列项撤回为草稿：草稿为空则直接返回该项内容；非空则将当前草稿插回队尾，
    /// 保留其余队列项原有顺序并清空 `selected`。
    pub fn retract_by_id(
        &mut self,
        id: u64,
        draft_text: &str,
        draft_images: Vec<(String, ImageAttachment)>,
    ) -> Option<(String, Vec<(String, ImageAttachment)>)> {
        let item = self.dequeue_by_id(id)?;
        let trimmed_draft = draft_text.trim();
        if !trimmed_draft.is_empty() || !draft_images.is_empty() {
            let swap_id = self.alloc_id();
            self.items.push(QueuedPrompt {
                id: swap_id,
                text: trimmed_draft.to_string(),
                images: draft_images,
            });
        }
        self.selected = None;
        Some((item.text, item.images))
    }

    /// 一步撤回队尾消息（支持 `Alt+↑`）；草稿非空时与草稿交换。
    pub fn retract_tail(
        &mut self,
        draft_text: &str,
        draft_images: Vec<(String, ImageAttachment)>,
    ) -> Option<(String, Vec<(String, ImageAttachment)>)> {
        let last_id = self.items.last()?.id;
        self.retract_by_id(last_id, draft_text, draft_images)
    }

    /// 丢弃指定 ID 的队列项（不发送）。
    pub fn discard_by_id(&mut self, id: u64) -> bool {
        self.dequeue_by_id(id).is_some()
    }

    /// 将选择器打开并聚焦到队尾条目；队列为空时保持关闭并返回 `false`。
    pub fn open_selector_at_tail(&mut self) -> bool {
        if let Some(last) = self.items.last() {
            self.selected = Some(last.id);
            true
        } else {
            self.selected = None;
            false
        }
    }

    /// 选择器向上移动一项：
    /// - 未选中且队列非空时进入队尾并保持选择器开启（返回 `true`）；
    /// - 已在队首时越界关闭选择器（置 `None` 并返回 `false`），交还给输入框历史回溯；
    /// - 否则移向前一项并返回 `true`。
    pub fn move_selection_up(&mut self) -> bool {
        if self.items.is_empty() {
            self.selected = None;
            return false;
        }
        let Some(selected) = self.selected else {
            return self.open_selector_at_tail();
        };
        let Some(index) = self.items.iter().position(|item| item.id == selected) else {
            self.selected = None;
            return false;
        };
        if index == 0 {
            self.selected = None;
            false
        } else {
            self.selected = Some(self.items[index - 1].id);
            true
        }
    }

    /// 选择器向下移动一项：
    /// - 已在队尾或未打开时关闭/保持关闭并返回 `false`，交还给输入框导航；
    /// - 否则移向后一项并返回 `true`。
    pub fn move_selection_down(&mut self) -> bool {
        let Some(selected) = self.selected else {
            return false;
        };
        let Some(index) = self.items.iter().position(|item| item.id == selected) else {
            self.selected = None;
            return false;
        };
        if index + 1 >= self.items.len() {
            self.selected = None;
            false
        } else {
            self.selected = Some(self.items[index + 1].id);
            true
        }
    }

    pub fn selected_id(&self) -> Option<u64> {
        self.selected
    }

    pub fn set_selected(&mut self, id: Option<u64>) {
        self.selected = id;
        self.sync_selection();
    }

    pub fn clear_selection(&mut self) {
        self.selected = None;
    }

    pub fn items(&self) -> &[QueuedPrompt] {
        &self.items
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.selected = None;
    }
}
