//! 30fps 指数衰减平滑吐字控制器（借鉴 `dsh-TUI` `smoothReveal.ts` / `oh-my-pi`）：
//! 将后端突发到达的大块流式 delta 平滑化为稳定节奏，同时保证安全落在 UTF-8 字符边界。

use std::collections::HashMap;

pub const REVEAL_MIN_STEP_BYTES: usize = 4;
pub const REVEAL_CATCHUP_FRAMES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RevealSlot {
    AssistantText(usize),
    AssistantReasoning(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotRevealState {
    /// 正在平滑推进揭示进度。
    Animating {
        revealed_bytes: usize,
        total_bytes: usize,
    },
    /// 当前活跃流已追平到 `revealed_bytes`，后续新文本到达时从该偏移继续平滑揭示。
    CaughtUp { revealed_bytes: usize },
    /// 条目已完成流式输出（或已强制对齐），固定展示完整文本。
    Finalized,
}

#[derive(Debug, Default)]
pub struct RevealController {
    pub enabled: bool,
    slots: HashMap<RevealSlot, SlotRevealState>,
    generation: u64,
}

impl RevealController {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            slots: HashMap::new(),
            generation: 0,
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    /// 新草稿会复用原下标，不能继承旧草稿的揭示进度。
    pub fn remove_item(&mut self, item_idx: usize) {
        let text = self.slots.remove(&RevealSlot::AssistantText(item_idx));
        let reasoning = self.slots.remove(&RevealSlot::AssistantReasoning(item_idx));
        if text.is_some() || reasoning.is_some() {
            self.generation = self.generation.wrapping_add(1);
        }
    }

    pub fn has_pending(&self) -> bool {
        self.enabled
            && self.slots.values().any(|s| {
                matches!(
                    s,
                    SlotRevealState::Animating {
                        revealed_bytes,
                        total_bytes
                    } if revealed_bytes < total_bytes
                )
            })
    }

    /// 注册或更新流式条目长度；若已追平后又收到新文本，从上次已揭示偏移继续平滑推进。
    pub fn observe(&mut self, slot: RevealSlot, total_bytes: usize, active: bool) {
        if !self.enabled || !active {
            let changed = if total_bytes > 0 {
                !matches!(
                    self.slots.insert(slot, SlotRevealState::Finalized),
                    Some(SlotRevealState::Finalized) | None
                )
            } else {
                self.slots.remove(&slot).is_some()
            };
            if changed {
                self.generation = self.generation.wrapping_add(1);
            }
            return;
        }
        match self.slots.get_mut(&slot) {
            Some(SlotRevealState::Finalized) => {}
            Some(state @ SlotRevealState::CaughtUp { .. }) => {
                let SlotRevealState::CaughtUp { revealed_bytes } = *state else {
                    unreachable!();
                };
                let clamped = revealed_bytes.min(total_bytes);
                if clamped < revealed_bytes {
                    self.generation = self.generation.wrapping_add(1);
                }
                if total_bytes > clamped {
                    *state = SlotRevealState::Animating {
                        revealed_bytes: clamped,
                        total_bytes,
                    };
                } else {
                    *state = SlotRevealState::CaughtUp {
                        revealed_bytes: clamped,
                    };
                }
            }
            Some(state @ SlotRevealState::Animating { .. }) => {
                let SlotRevealState::Animating { revealed_bytes, .. } = *state else {
                    unreachable!();
                };
                let clamped = revealed_bytes.min(total_bytes);
                if clamped < revealed_bytes {
                    self.generation = self.generation.wrapping_add(1);
                }
                if clamped >= total_bytes {
                    *state = SlotRevealState::CaughtUp {
                        revealed_bytes: total_bytes,
                    };
                } else {
                    *state = SlotRevealState::Animating {
                        revealed_bytes: clamped,
                        total_bytes,
                    };
                }
            }
            None => {
                let init = if total_bytes == 0 {
                    SlotRevealState::CaughtUp { revealed_bytes: 0 }
                } else {
                    SlotRevealState::Animating {
                        revealed_bytes: 0,
                        total_bytes,
                    }
                };
                self.slots.insert(slot, init);
            }
        }
    }

    /// 立即对齐所有槽位（例如取消、切换会话或关闭平滑吐字）。
    pub fn snap_all(&mut self) {
        let mut changed = false;
        self.slots.retain(|_, state| match *state {
            SlotRevealState::Finalized => true,
            SlotRevealState::CaughtUp { revealed_bytes } => {
                if revealed_bytes > 0 {
                    *state = SlotRevealState::Finalized;
                    changed = true;
                    true
                } else {
                    changed = true;
                    false
                }
            }
            SlotRevealState::Animating { total_bytes, .. } => {
                if total_bytes > 0 {
                    *state = SlotRevealState::Finalized;
                    changed = true;
                    true
                } else {
                    changed = true;
                    false
                }
            }
        });
        if changed {
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// 推进一帧动画；返回是否有可见进度发生变化。
    pub fn tick(&mut self) -> bool {
        if !self.enabled || self.slots.is_empty() {
            return false;
        }
        let mut changed = false;
        for state in self.slots.values_mut() {
            if let SlotRevealState::Animating {
                revealed_bytes,
                total_bytes,
            } = *state
            {
                if revealed_bytes >= total_bytes {
                    *state = SlotRevealState::CaughtUp {
                        revealed_bytes: total_bytes,
                    };
                    continue;
                }
                let backlog = total_bytes - revealed_bytes;
                let step = reveal_step(backlog);
                let next = (revealed_bytes + step).min(total_bytes);
                changed = true;
                if next >= total_bytes {
                    *state = SlotRevealState::CaughtUp {
                        revealed_bytes: total_bytes,
                    };
                } else {
                    *state = SlotRevealState::Animating {
                        revealed_bytes: next,
                        total_bytes,
                    };
                }
            }
        }
        if changed {
            self.generation = self.generation.wrapping_add(1);
        }
        changed
    }

    /// 获取当前帧允许显示的 UTF-8 安全前缀切片。
    pub fn visible_slice<'a>(&self, slot: RevealSlot, full_text: &'a str) -> &'a str {
        if !self.enabled {
            return full_text;
        }
        let revealed_bytes = match self.slots.get(&slot) {
            None | Some(SlotRevealState::Finalized) => return full_text,
            Some(SlotRevealState::CaughtUp { revealed_bytes })
            | Some(SlotRevealState::Animating { revealed_bytes, .. }) => *revealed_bytes,
        };
        if revealed_bytes >= full_text.len() {
            return full_text;
        }
        let mut end = revealed_bytes;
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
