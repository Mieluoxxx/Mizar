//! 纯内存会话信箱（`SessionInbox`）与轮次输入身份区分（`TurnInput`）。
//!
//! 统一收口步间插话（`Steer`）、忙时排队/闲时唤醒（`FollowUp`）与静默上下文附着（`Inject`），
//! 并以按会话计数的 `auto_wake_limit` 防止后台插件连续自动唤醒形成无限推理循环。

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use mizar_llm::ImageAttachment;
use mizar_session::SessionEvent;
use tokio::sync::broadcast;

/// 连续非人类 `Notice` 自动唤醒的默认上限；超出后自动降级为 `DeliveryMode::Inject`。
pub const DEFAULT_AUTO_WAKE_LIMIT: usize = 5;

const WAKE_CHANNEL_CAPACITY: usize = 256;

/// 轮次/步间输入载荷：明确区分人类用户直接指令与插件/外部事件通知。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnInput {
    User {
        text: String,
        images: Vec<ImageAttachment>,
    },
    Notice {
        source: String,
        text: String,
        images: Vec<ImageAttachment>,
    },
}

impl TurnInput {
    pub fn user(text: impl Into<String>) -> Self {
        Self::User {
            text: text.into(),
            images: Vec::new(),
        }
    }

    pub fn user_with_images(text: impl Into<String>, images: Vec<ImageAttachment>) -> Self {
        Self::User {
            text: text.into(),
            images,
        }
    }

    pub fn notice(source: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Notice {
            source: source.into(),
            text: text.into(),
            images: Vec::new(),
        }
    }

    pub fn notice_with_images(
        source: impl Into<String>,
        text: impl Into<String>,
        images: Vec<ImageAttachment>,
    ) -> Self {
        Self::Notice {
            source: source.into(),
            text: text.into(),
            images,
        }
    }

    pub fn is_user(&self) -> bool {
        matches!(self, Self::User { .. })
    }

    pub fn text(&self) -> &str {
        match self {
            Self::User { text, .. } | Self::Notice { text, .. } => text,
        }
    }

    pub fn images(&self) -> &[ImageAttachment] {
        match self {
            Self::User { images, .. } | Self::Notice { images, .. } => images,
        }
    }

    pub fn into_session_event(self) -> SessionEvent {
        match self {
            Self::User { text, images } => SessionEvent::user_with_images(text, images),
            Self::Notice {
                source,
                text,
                images,
            } => SessionEvent::ContextNotice {
                source,
                text,
                images,
            },
        }
    }
}

impl From<String> for TurnInput {
    fn from(text: String) -> Self {
        Self::user(text)
    }
}

impl From<&str> for TurnInput {
    fn from(text: &str) -> Self {
        Self::user(text)
    }
}

impl From<(&str, Vec<ImageAttachment>)> for TurnInput {
    fn from((text, images): (&str, Vec<ImageAttachment>)) -> Self {
        Self::user_with_images(text, images)
    }
}

impl From<(String, Vec<ImageAttachment>)> for TurnInput {
    fn from((text, images): (String, Vec<ImageAttachment>)) -> Self {
        Self::user_with_images(text, images)
    }
}

/// 信箱投递语义。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryMode {
    /// 步间插话：运行中在下步前排空；空闲时按 `FollowUp` 尝试唤醒。
    Steer,
    /// 排入下一轮：忙时排队等待当前轮结束，空闲时触发唤醒。
    FollowUp,
    /// 仅沉淀为上下文，不触发新一轮推理。
    Inject,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InboxItem {
    pub id: u64,
    pub mode: DeliveryMode,
    pub input: TurnInput,
    pub coalesce_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InboxPushResult {
    pub id: u64,
    pub effective_mode: DeliveryMode,
    pub degraded: bool,
}

#[derive(Debug)]
struct SessionMailbox {
    items: Vec<InboxItem>,
    auto_wakes_since_human: usize,
    auto_wake_limit: usize,
}

impl Default for SessionMailbox {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            auto_wakes_since_human: 0,
            auto_wake_limit: DEFAULT_AUTO_WAKE_LIMIT,
        }
    }
}

#[derive(Debug, Default)]
struct InboxState {
    next_id: u64,
    active_sessions: HashSet<String>,
    mailboxes: HashMap<String, SessionMailbox>,
}

/// 按 `session_id` 隔离的纯内存 Actor 收件箱。
pub struct SessionInbox {
    state: Mutex<InboxState>,
    wake_tx: broadcast::Sender<String>,
}

impl Default for SessionInbox {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionInbox {
    pub fn new() -> Self {
        let (wake_tx, _) = broadcast::channel(WAKE_CHANNEL_CAPACITY);
        Self {
            state: Mutex::new(InboxState::default()),
            wake_tx,
        }
    }

    pub fn subscribe_wake(&self) -> broadcast::Receiver<String> {
        self.wake_tx.subscribe()
    }

    /// 显式标记人类直接提交了输入，清零该会话的连续自动唤醒计数。
    pub fn record_human_input(&self, session_id: &str) {
        let mut state = self.state.lock().unwrap();
        let mailbox = state.mailboxes.entry(session_id.to_string()).or_default();
        mailbox.auto_wakes_since_human = 0;
    }

    /// 供 `AgentLoop` 与独立单测标记会话忙闲态。
    pub fn set_session_active(&self, session_id: &str, active: bool) {
        if active {
            let mut state = self.state.lock().unwrap();
            state.active_sessions.insert(session_id.to_string());
        } else {
            self.release_session(session_id);
        }
    }

    pub fn is_session_active(&self, session_id: &str) -> bool {
        self.state
            .lock()
            .unwrap()
            .active_sessions
            .contains(session_id)
    }

    /// 原子抢占会话执行权；若已在运行返回 `false`。
    pub(crate) fn try_acquire_session(&self, session_id: &str) -> bool {
        self.state
            .lock()
            .unwrap()
            .active_sessions
            .insert(session_id.to_string())
    }

    /// 释放会话执行权，并将竞争窗口内遗留的 `Steer` 转为 `FollowUp`/`Inject` 后发出唤醒。
    pub(crate) fn release_session(&self, session_id: &str) {
        let has_followup = {
            let mut state = self.state.lock().unwrap();
            state.active_sessions.remove(session_id);
            if let Some(mailbox) = state.mailboxes.get_mut(session_id) {
                for item in &mut mailbox.items {
                    if item.mode == DeliveryMode::Steer {
                        if item.input.is_user() {
                            item.mode = DeliveryMode::FollowUp;
                        } else {
                            let next_wakes = mailbox.auto_wakes_since_human.saturating_add(1);
                            if next_wakes > mailbox.auto_wake_limit {
                                item.mode = DeliveryMode::Inject;
                            } else {
                                mailbox.auto_wakes_since_human = next_wakes;
                                item.mode = DeliveryMode::FollowUp;
                            }
                        }
                    }
                }
                mailbox
                    .items
                    .iter()
                    .any(|item| item.mode == DeliveryMode::FollowUp)
            } else {
                false
            }
        };
        if has_followup {
            let _ = self.wake_tx.send(session_id.to_string());
        }
    }

    pub fn push(&self, session_id: &str, mode: DeliveryMode, input: TurnInput) -> InboxPushResult {
        self.push_coalesced(session_id, mode, input, None)
    }

    pub fn push_coalesced(
        &self,
        session_id: &str,
        mode: DeliveryMode,
        input: TurnInput,
        coalesce_key: Option<String>,
    ) -> InboxPushResult {
        let result = {
            let mut state = self.state.lock().unwrap();
            let is_active = state.active_sessions.contains(session_id);
            let is_human = input.is_user();

            // 预判断同 key 可否原地覆写，避免高频刷屏重复消耗自动唤醒预算。
            let existing_followup_match = if let Some(ref key) = coalesce_key {
                state.mailboxes.get(session_id).is_some_and(|mb| {
                    mb.items.iter().any(|item| {
                        item.mode == DeliveryMode::FollowUp
                            && item.coalesce_key.as_deref() == Some(key.as_str())
                    })
                })
            } else {
                false
            };

            let mailbox = state.mailboxes.entry(session_id.to_string()).or_default();
            if is_human {
                mailbox.auto_wakes_since_human = 0;
            }

            let (effective_mode, degraded) = match mode {
                DeliveryMode::Inject => (DeliveryMode::Inject, false),
                DeliveryMode::Steer if is_active => (DeliveryMode::Steer, false),
                DeliveryMode::Steer | DeliveryMode::FollowUp => {
                    if is_human || existing_followup_match {
                        (DeliveryMode::FollowUp, false)
                    } else {
                        let next_wakes = mailbox.auto_wakes_since_human.saturating_add(1);
                        if next_wakes > mailbox.auto_wake_limit {
                            (DeliveryMode::Inject, true)
                        } else {
                            mailbox.auto_wakes_since_human = next_wakes;
                            (DeliveryMode::FollowUp, false)
                        }
                    }
                }
            };

            if let Some(ref key) = coalesce_key
                && let Some(existing) = mailbox.items.iter_mut().find(|item| {
                    item.mode == effective_mode
                        && item.coalesce_key.as_deref() == Some(key.as_str())
                })
            {
                existing.input = input;
                InboxPushResult {
                    id: existing.id,
                    effective_mode,
                    degraded,
                }
            } else {
                state.next_id = state.next_id.wrapping_add(1).max(1);
                let id = state.next_id;
                state
                    .mailboxes
                    .entry(session_id.to_string())
                    .or_default()
                    .items
                    .push(InboxItem {
                        id,
                        mode: effective_mode,
                        input,
                        coalesce_key,
                    });
                InboxPushResult {
                    id,
                    effective_mode,
                    degraded,
                }
            }
        };

        if result.effective_mode == DeliveryMode::FollowUp {
            let _ = self.wake_tx.send(session_id.to_string());
        }
        result
    }

    /// 排空本步可消费的 `Steer` 与 `Inject` 条目（保留顺序；`FollowUp` 留待下轮）。
    pub fn drain_step_inputs(&self, session_id: &str) -> Vec<TurnInput> {
        let mut state = self.state.lock().unwrap();
        let Some(mailbox) = state.mailboxes.get_mut(session_id) else {
            return Vec::new();
        };
        let mut drained = Vec::new();
        let mut remaining = Vec::with_capacity(mailbox.items.len());
        for item in mailbox.items.drain(..) {
            match item.mode {
                DeliveryMode::Steer | DeliveryMode::Inject => drained.push(item.input),
                DeliveryMode::FollowUp => remaining.push(item),
            }
        }
        mailbox.items = remaining;
        drained
    }

    pub fn has_pending_steer(&self, session_id: &str) -> bool {
        let state = self.state.lock().unwrap();
        state
            .mailboxes
            .get(session_id)
            .is_some_and(|mb| mb.items.iter().any(|item| item.mode == DeliveryMode::Steer))
    }

    pub fn has_pending_user_steer(&self, session_id: &str) -> bool {
        let state = self.state.lock().unwrap();
        state.mailboxes.get(session_id).is_some_and(|mb| {
            mb.items
                .iter()
                .any(|item| item.mode == DeliveryMode::Steer && item.input.is_user())
        })
    }

    /// 当停轮守卫或自动化 `Notice` 耗尽每轮防爆预算时，将剩余 `Notice` `Steer` 降级为 `Inject`，
    /// 既防止停轮后被 `release_session` 二次唤醒，又保留上下文内容。
    pub(crate) fn degrade_notice_steers_to_inject(&self, session_id: &str) -> usize {
        let mut state = self.state.lock().unwrap();
        let Some(mailbox) = state.mailboxes.get_mut(session_id) else {
            return 0;
        };
        let mut count = 0usize;
        for item in &mut mailbox.items {
            if item.mode == DeliveryMode::Steer && !item.input.is_user() {
                item.mode = DeliveryMode::Inject;
                count += 1;
            }
        }
        count
    }

    pub fn pop_followup(&self, session_id: &str) -> Option<TurnInput> {
        let mut state = self.state.lock().unwrap();
        let mailbox = state.mailboxes.get_mut(session_id)?;
        let idx = mailbox
            .items
            .iter()
            .position(|item| item.mode == DeliveryMode::FollowUp)?;
        Some(mailbox.items.remove(idx).input)
    }

    pub fn snapshot(&self, session_id: &str) -> Vec<InboxItem> {
        let state = self.state.lock().unwrap();
        state
            .mailboxes
            .get(session_id)
            .map(|mb| mb.items.clone())
            .unwrap_or_default()
    }

    pub fn clear(&self, session_id: &str) {
        let mut state = self.state.lock().unwrap();
        if let Some(mailbox) = state.mailboxes.get_mut(session_id) {
            mailbox.items.clear();
        }
    }

    pub fn set_auto_wake_limit(&self, session_id: &str, limit: usize) {
        let mut state = self.state.lock().unwrap();
        let mailbox = state.mailboxes.entry(session_id.to_string()).or_default();
        mailbox.auto_wake_limit = limit;
    }

    pub fn auto_wake_limit(&self, session_id: &str) -> usize {
        let state = self.state.lock().unwrap();
        state
            .mailboxes
            .get(session_id)
            .map_or(DEFAULT_AUTO_WAKE_LIMIT, |mb| mb.auto_wake_limit)
    }

    pub fn auto_wakes_since_human(&self, session_id: &str) -> usize {
        let state = self.state.lock().unwrap();
        state
            .mailboxes
            .get(session_id)
            .map_or(0, |mb| mb.auto_wakes_since_human)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_input_conversions_and_session_event_mapping() {
        let from_str: TurnInput = "hello".into();
        assert_eq!(from_str, TurnInput::user("hello"));
        assert!(from_str.is_user());

        let from_string: TurnInput = String::from("world").into();
        assert_eq!(from_string, TurnInput::user("world"));

        let img = ImageAttachment::new("image/png", "AQID");
        let from_tuple: TurnInput = ("look", vec![img.clone()]).into();
        assert_eq!(from_tuple.images().len(), 1);

        let notice = TurnInput::notice("watcher", "file changed");
        assert!(!notice.is_user());
        assert!(matches!(
            notice.into_session_event(),
            SessionEvent::ContextNotice { source, text, .. }
            if source == "watcher" && text == "file changed"
        ));
    }

    #[tokio::test]
    async fn followup_and_idle_steer_degrade_past_auto_wake_limit_and_reset_on_human_user() {
        let inbox = SessionInbox::new();
        let mut wake_rx = inbox.subscribe_wake();
        inbox.set_auto_wake_limit("s1", 2);

        // 1. 前两次 Notice FollowUp（及空闲 Steer）允许唤醒
        let r1 = inbox.push(
            "s1",
            DeliveryMode::FollowUp,
            TurnInput::notice("cron", "n1"),
        );
        assert_eq!(r1.effective_mode, DeliveryMode::FollowUp);
        assert!(!r1.degraded);
        assert_eq!(wake_rx.try_recv().unwrap(), "s1");

        let r2 = inbox.push("s1", DeliveryMode::Steer, TurnInput::notice("cron", "n2"));
        assert_eq!(r2.effective_mode, DeliveryMode::FollowUp);
        assert!(!r2.degraded);
        assert_eq!(inbox.auto_wakes_since_human("s1"), 2);
        assert_eq!(wake_rx.try_recv().unwrap(), "s1");

        // 2. 第 3 次超出 limit=2：降级为 Inject 且不触发唤醒
        let r3 = inbox.push(
            "s1",
            DeliveryMode::FollowUp,
            TurnInput::notice("cron", "n3"),
        );
        assert_eq!(r3.effective_mode, DeliveryMode::Inject);
        assert!(r3.degraded);
        assert!(wake_rx.try_recv().is_err());

        // 消费掉两次 FollowUp 后，第 3 条仍作为 Inject 留存在队列中
        assert_eq!(inbox.pop_followup("s1").unwrap().text(), "n1");
        assert_eq!(inbox.pop_followup("s1").unwrap().text(), "n2");
        assert!(inbox.pop_followup("s1").is_none());
        assert_eq!(inbox.snapshot("s1").len(), 1);
        assert_eq!(inbox.snapshot("s1")[0].mode, DeliveryMode::Inject);

        // 3. 人类 TurnInput::User 投递后立即清零计数，后续自动化 FollowUp 重新恢复唤醒
        let r_human = inbox.push("s1", DeliveryMode::FollowUp, TurnInput::user("human reset"));
        assert_eq!(r_human.effective_mode, DeliveryMode::FollowUp);
        assert!(!r_human.degraded);
        assert_eq!(inbox.auto_wakes_since_human("s1"), 0);
        assert_eq!(wake_rx.try_recv().unwrap(), "s1");

        let r4 = inbox.push(
            "s1",
            DeliveryMode::FollowUp,
            TurnInput::notice("cron", "n4"),
        );
        assert_eq!(r4.effective_mode, DeliveryMode::FollowUp);
        assert!(!r4.degraded);
        assert_eq!(inbox.auto_wakes_since_human("s1"), 1);

        // drain_step_inputs 仅排空 Inject/Steer，不吞掉 FollowUp
        let drained = inbox.drain_step_inputs("s1");
        assert_eq!(drained, vec![TurnInput::notice("cron", "n3")]);
        assert_eq!(inbox.snapshot("s1").len(), 2);
    }

    #[test]
    fn active_steer_coalesce_and_release_promotion() {
        let inbox = SessionInbox::new();
        inbox.set_session_active("s1", true);

        let r1 = inbox.push_coalesced(
            "s1",
            DeliveryMode::Steer,
            TurnInput::notice("build", "step 1"),
            Some("build-status".into()),
        );
        assert_eq!(r1.effective_mode, DeliveryMode::Steer);
        assert!(inbox.has_pending_steer("s1"));

        let r2 = inbox.push_coalesced(
            "s1",
            DeliveryMode::Steer,
            TurnInput::notice("build", "step 2"),
            Some("build-status".into()),
        );
        assert_eq!(r2.id, r1.id);
        assert_eq!(inbox.snapshot("s1").len(), 1);
        assert_eq!(inbox.snapshot("s1")[0].input.text(), "step 2");

        // 释放会话时，未及消费的 Steer 自动提升为 FollowUp 并唤醒宿主
        let mut rx = inbox.subscribe_wake();
        inbox.set_session_active("s1", false);
        assert!(!inbox.has_pending_steer("s1"));
        assert_eq!(rx.try_recv().unwrap(), "s1");
        assert_eq!(inbox.pop_followup("s1").unwrap().text(), "step 2");
    }
}
