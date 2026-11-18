//! 30fps 指数衰减平滑吐字控制器（借鉴 `dsh-TUI` `smoothReveal.ts` / `oh-my-pi`）：
//! 将后端突发到达的大块流式 delta 平滑化为稳定节奏，同时保证安全落在 UTF-8 字符边界。

use std::collections::{HashMap, HashSet};

pub const REVEAL_MIN_STEP_BYTES: usize = 4;
pub const REVEAL_CATCHUP_FRAMES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RevealSlot {
    AssistantText(usize),
    AssistantReasoning(usize),
}

#[derive(Debug, Clone)]
struct RevealCursor {
    total_bytes: usize,
    revealed_bytes: usize,
}

#[derive(Debug, Default)]
pub struct RevealController {
    pub enabled: bool,
    cursors: HashMap<RevealSlot, RevealCursor>,
    completed: HashSet<RevealSlot>,
}

impl RevealController {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            cursors: HashMap::new(),
            completed: HashSet::new(),
        }
    }

    pub fn clear(&mut self) {
        self.cursors.clear();
        self.completed.clear();
    }

    pub fn has_pending(&self) -> bool {
        self.enabled
            && self
                .cursors
                .values()
                .any(|c| c.revealed_bytes < c.total_bytes)
    }

    /// 注册或更新流式条目长度；若尚未追踪且处于激活流式阶段则从当前进度开始平滑推进。
    pub fn observe(&mut self, slot: RevealSlot, full_text: &str, active: bool) {
        let total = full_text.len();
        if !self.enabled || !active {
            self.cursors.remove(&slot);
            if total > 0 {
                self.completed.insert(slot);
            }
            return;
        }
        if self.completed.contains(&slot) {
            return;
        }
        self.cursors
            .entry(slot)
            .and_modify(|c| {
                c.total_bytes = total;
                if c.revealed_bytes > total {
                    c.revealed_bytes = total;
                }
            })
            .or_insert(RevealCursor {
                total_bytes: total,
                revealed_bytes: 0,
            });
    }

    /// 立即对齐所有未完成槽位（例如取消、切换会话或关闭平滑吐字）。
    pub fn snap_all(&mut self) {
        for (slot, c) in self.cursors.drain() {
            if c.total_bytes > 0 {
                self.completed.insert(slot);
            }
        }
    }

    /// 推进一帧动画；返回是否有可见进度发生变化。
    pub fn tick(&mut self) -> bool {
        if !self.enabled || self.cursors.is_empty() {
            return false;
        }
        let mut changed = false;
        let mut finished = Vec::new();
        for (&slot, c) in &mut self.cursors {
            if c.revealed_bytes >= c.total_bytes {
                finished.push(slot);
                continue;
            }
            let backlog = c.total_bytes - c.revealed_bytes;
            let step = reveal_step(backlog);
            c.revealed_bytes = (c.revealed_bytes + step).min(c.total_bytes);
            changed = true;
            if c.revealed_bytes >= c.total_bytes {
                finished.push(slot);
            }
        }
        for slot in finished {
            self.cursors.remove(&slot);
            self.completed.insert(slot);
        }
        changed
    }

    /// 获取当前帧允许显示的 UTF-8 安全前缀切片。
    pub fn visible_slice<'a>(&self, slot: RevealSlot, full_text: &'a str) -> &'a str {
        if !self.enabled {
            return full_text;
        }
        let Some(c) = self.cursors.get(&slot) else {
            return full_text;
        };
        if c.revealed_bytes >= full_text.len() {
            return full_text;
        }
        let mut end = c.revealed_bytes;
        while end < full_text.len() && !full_text.is_char_boundary(end) {
            end += 1;
        }
        &full_text[..end]
    }
}

pub fn reveal_step(backlog: usize) -> usize {
    let decay = backlog.div_ceil(REVEAL_CATCHUP_FRAMES);
    decay.max(REVEAL_MIN_STEP_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothly_catches_up_respecting_utf8_boundaries() {
        let mut ctrl = RevealController::new(true);
        let text = "你好世界，这是一段测试文本，用来验证平滑吐字不会切碎多字节UTF-8字符。";
        let slot = RevealSlot::AssistantText(0);
        ctrl.observe(slot, text, true);
        assert_eq!(ctrl.visible_slice(slot, text), "");

        let mut steps = 0;
        while ctrl.has_pending() {
            assert!(ctrl.tick());
            let slice = ctrl.visible_slice(slot, text);
            assert!(text.starts_with(slice));
            steps += 1;
            assert!(steps < 64);
        }
        assert_eq!(ctrl.visible_slice(slot, text), text);
    }
}
