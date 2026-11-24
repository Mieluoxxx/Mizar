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

    pub fn append_batch(&mut self, next: TurnInput, batch_source: &str) {
        let next_source = match &next {
            Self::Notice { source, .. } => Some(source.clone()),
            Self::User { .. } => None,
        };
        let (next_text, mut next_images) = match next {
            Self::User { text, images } | Self::Notice { text, images, .. } => (text, images),
        };
        match self {
            Self::User { text, images } => {
                if !next_text.is_empty() {
                    if !text.is_empty() {
                        text.push_str("\n\n---\n\n");
                    }
                    text.push_str(&next_text);
                }
                images.append(&mut next_images);
            }
            Self::Notice {
                source,
                text,
                images,
            } => {
                if let Some(ns) = next_source {
                    if source.as_str() != ns.as_str() && !batch_source.is_empty() {
                        *source = batch_source.to_string();
                    }
                } else if !batch_source.is_empty() {
                    *source = batch_source.to_string();
                }
                if !next_text.is_empty() {
                    if !text.is_empty() {
                        text.push_str("\n\n---\n\n");
                    }
                    text.push_str(&next_text);
                }
                images.append(&mut next_images);
            }
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

/// 子代理结构化批次类别（对应 `omp` `YieldQueue` 的延迟去重通道）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentSlotKind {
    Settled,
    Watchdog,
}

impl SubagentSlotKind {
    pub const fn coalesce_key(self) -> &'static str {
        match self {
            Self::Settled => "subagent:settled",
            Self::Watchdog => "subagent:watchdog",
        }
    }

    pub const fn batch_source(self) -> &'static str {
        match self {
            Self::Settled => "subagent:batch",
            Self::Watchdog => "subagent-watchdog:batch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentSlot {
    pub child_session_id: String,
    pub run_id: u64,
    pub source: String,
    pub text: String,
    /// 失败 / 超时 / 看门狗等紧急告警会立即打破 `yield` barrier 提前唤醒。
    pub urgent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MailboxPayload {
    Plain(TurnInput),
    SubagentBatch {
        kind: SubagentSlotKind,
        slots: Vec<SubagentSlot>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MailboxEntry {
    id: u64,
    mode: DeliveryMode,
    coalesce_key: Option<String>,
    payload: MailboxPayload,
}

impl MailboxEntry {
    fn is_user(&self) -> bool {
        matches!(&self.payload, MailboxPayload::Plain(input) if input.is_user())
    }

    fn is_subagent_batch(&self) -> bool {
        matches!(self.payload, MailboxPayload::SubagentBatch { .. })
    }

    fn has_urgent_slot(&self) -> bool {
        matches!(
            &self.payload,
            MailboxPayload::SubagentBatch { slots, .. } if slots.iter().any(|s| s.urgent)
        )
    }

    fn materialize_input(&self) -> TurnInput {
        match &self.payload {
            MailboxPayload::Plain(input) => input.clone(),
            MailboxPayload::SubagentBatch { kind, slots } => {
                if slots.len() == 1 {
                    TurnInput::notice(slots[0].source.clone(), slots[0].text.clone())
                } else {
                    let mut combined = String::new();
                    for slot in slots {
                        if !slot.text.is_empty() {
                            if !combined.is_empty() {
                                combined.push_str("\n\n---\n\n");
                            }
                            combined.push_str(&slot.text);
                        }
                    }
                    TurnInput::notice(kind.batch_source(), combined)
                }
            }
        }
    }

    fn to_inbox_item(&self) -> InboxItem {
        InboxItem {
            id: self.id,
            mode: self.mode,
            input: self.materialize_input(),
            coalesce_key: self.coalesce_key.clone(),
        }
    }
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
    items: Vec<MailboxEntry>,
    auto_wakes_since_human: usize,
    auto_wake_limit: usize,
    /// 按 `child_session_id` 记录已知最新运行代次 `run_id`，用于在 flush 前过滤旧代次残留槽位。
    latest_child_run: HashMap<String, u64>,
    /// 主代理主动调用 `subagent(action="yield")` 后持有的并发等待 barrier。
    yield_barrier: bool,
}

impl Default for SessionMailbox {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            auto_wakes_since_human: 0,
            auto_wake_limit: DEFAULT_AUTO_WAKE_LIMIT,
            latest_child_run: HashMap::new(),
            yield_barrier: false,
        }
    }
}

impl SessionMailbox {
    /// 按最新代次过滤结构化子代理槽位，槽位全空时整条移除并归还未投递的自动唤醒计数。
    fn prune_and_compact(&mut self) {
        let latest = &self.latest_child_run;
        let mut dropped_followups = 0usize;
        self.items.retain_mut(|entry| {
            let MailboxPayload::SubagentBatch { slots, .. } = &mut entry.payload else {
                return true;
            };
            let mut kept_rev = Vec::with_capacity(slots.len());
            let mut seen_children = HashSet::new();
            for slot in slots.iter().rev() {
                if let Some(&latest_run) = latest.get(&slot.child_session_id)
                    && slot.run_id < latest_run
                {
                    continue;
                }
                if seen_children.insert(slot.child_session_id.clone()) {
                    kept_rev.push(slot.clone());
                }
            }
            kept_rev.reverse();
            *slots = kept_rev;
            let keep = !slots.is_empty();
            if !keep && entry.mode == DeliveryMode::FollowUp {
                dropped_followups = dropped_followups.saturating_add(1);
            }
            keep
        });
        if dropped_followups > 0 {
            self.auto_wakes_since_human = self
                .auto_wakes_since_human
                .saturating_sub(dropped_followups);
        }
    }

    fn is_entry_deliverable_followup(&self, entry: &MailboxEntry) -> bool {
        entry.mode == DeliveryMode::FollowUp
            && !(self.yield_barrier && entry.is_subagent_batch() && !entry.has_urgent_slot())
    }

    fn has_deliverable_followup(&self) -> bool {
        self.items
            .iter()
            .any(|entry| self.is_entry_deliverable_followup(entry))
    }

    fn has_urgent_or_user_followup(&self) -> bool {
        self.items.iter().any(|entry| {
            entry.mode == DeliveryMode::FollowUp && (entry.is_user() || entry.has_urgent_slot())
        })
    }
}

#[derive(Debug, Default)]
struct InboxState {
    next_id: u64,
    active_sessions: HashSet<String>,
    mailboxes: HashMap<String, SessionMailbox>,
}

fn resolve_push_delivery(
    mailbox: &mut SessionMailbox,
    is_active: bool,
    mode: DeliveryMode,
    is_human: bool,
    coalesce_key: Option<&str>,
) -> (DeliveryMode, bool) {
    let existing_followup_match = coalesce_key.is_some_and(|key| {
        mailbox.items.iter().any(|item| {
            item.mode == DeliveryMode::FollowUp && item.coalesce_key.as_deref() == Some(key)
        })
    });

    if is_human {
        mailbox.auto_wakes_since_human = 0;
    }

    match mode {
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
    }
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

    /// 显式标记人类直接提交了输入，清零连续自动唤醒计数并立即打破 `yield` barrier。
    pub fn record_human_input(&self, session_id: &str) {
        let mut state = self.state.lock().unwrap();
        let mailbox = state.mailboxes.entry(session_id.to_string()).or_default();
        mailbox.auto_wakes_since_human = 0;
        mailbox.yield_barrier = false;
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
                mailbox.prune_and_compact();
                for item in &mut mailbox.items {
                    if item.mode == DeliveryMode::Steer {
                        if item.is_user() {
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
                mailbox.has_deliverable_followup()
            } else {
                false
            }
        };
        if has_followup {
            let _ = self.wake_tx.send(session_id.to_string());
        }
    }

    /// 为主动调用 `subagent(action="yield")` 的父会话挂载并发等待 barrier；
    /// 若箱内已存在人类输入或看门狗/失败等紧急 `FollowUp` 则立即拒绝/打破并返回 `false`。
    pub fn arm_yield_barrier(&self, session_id: &str) -> bool {
        let mut state = self.state.lock().unwrap();
        let mailbox = state.mailboxes.entry(session_id.to_string()).or_default();
        mailbox.prune_and_compact();
        if mailbox.has_urgent_or_user_followup() {
            mailbox.yield_barrier = false;
            return false;
        }
        mailbox.yield_barrier = true;
        true
    }

    /// 在所有子代理结算（或被停止归零）时释放 `yield` barrier；若有可投递 `FollowUp` 则唤醒一次。
    pub fn release_yield_barrier(&self, session_id: &str) {
        let should_wake = {
            let mut state = self.state.lock().unwrap();
            let Some(mailbox) = state.mailboxes.get_mut(session_id) else {
                return;
            };
            let was_barrier = mailbox.yield_barrier;
            mailbox.yield_barrier = false;
            mailbox.prune_and_compact();
            was_barrier && mailbox.has_deliverable_followup()
        };
        if should_wake {
            let _ = self.wake_tx.send(session_id.to_string());
        }
    }

    pub fn has_yield_barrier(&self, session_id: &str) -> bool {
        let state = self.state.lock().unwrap();
        state
            .mailboxes
            .get(session_id)
            .is_some_and(|mb| mb.yield_barrier)
    }

    /// 记录子代理推进到更新代次 `run_id`，并清除该子代理旧代次残留槽位。
    pub fn advance_subagent_generation(
        &self,
        session_id: &str,
        child_session_id: &str,
        new_run_id: u64,
    ) {
        let mut state = self.state.lock().unwrap();
        let Some(mailbox) = state.mailboxes.get_mut(session_id) else {
            return;
        };
        let entry = mailbox
            .latest_child_run
            .entry(child_session_id.to_string())
            .or_insert(new_run_id);
        if new_run_id > *entry {
            *entry = new_run_id;
        }
        let target_gen = *entry;
        for item in &mut mailbox.items {
            if let MailboxPayload::SubagentBatch { slots, .. } = &mut item.payload {
                slots.retain(|slot| {
                    slot.child_session_id != child_session_id || slot.run_id >= target_gen
                });
            }
        }
        mailbox.prune_and_compact();
    }

    /// 按 `child_session_id`（及可选槽位类别）立即移除待投递条目；槽位全空时整条移除不唤醒。
    pub fn invalidate_subagent_slots(
        &self,
        session_id: &str,
        child_session_id: &str,
        kind: Option<SubagentSlotKind>,
    ) -> usize {
        let mut state = self.state.lock().unwrap();
        let Some(mailbox) = state.mailboxes.get_mut(session_id) else {
            return 0;
        };
        let mut removed = 0usize;
        for item in &mut mailbox.items {
            if let MailboxPayload::SubagentBatch {
                kind: batch_kind,
                slots,
            } = &mut item.payload
                && kind.is_none_or(|k| k == *batch_kind)
            {
                let before = slots.len();
                slots.retain(|slot| slot.child_session_id != child_session_id);
                removed += before.saturating_sub(slots.len());
            }
        }
        mailbox.prune_and_compact();
        removed
    }

    /// 推送结构化子代理通知槽位（`subagent:settled` / `subagent:watchdog`）：
    /// - 以 `(child_session_id, run_id)` 去重并清理旧代次/已过时的看门狗槽位；
    /// - 在 `yield` barrier 期间缓存非紧急结算，直到全员结算（`release_barrier`）或触发紧急打破（`urgent`）。
    #[allow(clippy::too_many_arguments)]
    pub fn push_subagent_slot(
        &self,
        session_id: &str,
        mode: DeliveryMode,
        kind: SubagentSlotKind,
        child_session_id: impl Into<String>,
        run_id: u64,
        source: impl Into<String>,
        text: impl Into<String>,
        urgent: bool,
        release_barrier: bool,
    ) -> InboxPushResult {
        let child_id = child_session_id.into();
        let source = source.into();
        let text = text.into();

        let (result, should_wake) = {
            let mut state = self.state.lock().unwrap();
            let is_active = state.active_sessions.contains(session_id);
            let mailbox = state.mailboxes.entry(session_id.to_string()).or_default();
            let was_barrier = mailbox.yield_barrier;

            if let Some(&latest_gen) = mailbox.latest_child_run.get(&child_id)
                && run_id < latest_gen
            {
                if urgent || release_barrier {
                    mailbox.yield_barrier = false;
                }
                mailbox.prune_and_compact();
                let wake =
                    !mailbox.yield_barrier && was_barrier && mailbox.has_deliverable_followup();
                (
                    InboxPushResult {
                        id: 0,
                        effective_mode: DeliveryMode::Inject,
                        degraded: false,
                    },
                    wake,
                )
            } else {
                mailbox.latest_child_run.insert(child_id.clone(), run_id);

                // 同子代理结算时清理其残留看门狗和旧结算；发新看门狗时覆盖同子代理旧看门狗
                for item in &mut mailbox.items {
                    if let MailboxPayload::SubagentBatch {
                        kind: batch_kind,
                        slots,
                    } = &mut item.payload
                        && (*batch_kind == SubagentSlotKind::Watchdog
                            || kind == SubagentSlotKind::Settled)
                    {
                        slots.retain(|slot| slot.child_session_id != child_id);
                    }
                }
                mailbox.prune_and_compact();

                let batch_key = kind.coalesce_key();
                let (effective_mode, degraded) =
                    resolve_push_delivery(mailbox, is_active, mode, false, Some(batch_key));

                let new_slot = SubagentSlot {
                    child_session_id: child_id,
                    run_id,
                    source,
                    text,
                    urgent,
                };

                let res = if let Some(existing) = mailbox.items.iter_mut().find(|item| {
                    item.mode == effective_mode
                        && matches!(
                            item.payload,
                            MailboxPayload::SubagentBatch { kind: k, .. } if k == kind
                        )
                }) {
                    if let MailboxPayload::SubagentBatch { slots, .. } = &mut existing.payload {
                        slots.push(new_slot);
                    }
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
                        .push(MailboxEntry {
                            id,
                            mode: effective_mode,
                            coalesce_key: Some(batch_key.to_string()),
                            payload: MailboxPayload::SubagentBatch {
                                kind,
                                slots: vec![new_slot],
                            },
                        });
                    InboxPushResult {
                        id,
                        effective_mode,
                        degraded,
                    }
                };

                let mailbox = state.mailboxes.get_mut(session_id).expect("mailbox exists");
                if urgent || release_barrier {
                    mailbox.yield_barrier = false;
                }
                let wake = !mailbox.yield_barrier
                    && (res.effective_mode == DeliveryMode::FollowUp
                        || (was_barrier && mailbox.has_deliverable_followup()));
                (res, wake)
            }
        };

        if should_wake {
            let _ = self.wake_tx.send(session_id.to_string());
        }
        result
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
        self.push_internal(session_id, mode, input, coalesce_key, None)
    }

    /// 批次追加投递：同 `(effective_mode, batch_key)` 条目存在时无论为 `FollowUp`、`Inject` 还是 `Steer`，
    /// 一律按分隔符拼接正文并切换为 `batch_source`，绝不覆写丢回执。
    pub fn push_appending(
        &self,
        session_id: &str,
        mode: DeliveryMode,
        input: TurnInput,
        batch_key: impl Into<String>,
        batch_source: impl Into<String>,
    ) -> InboxPushResult {
        let key = batch_key.into();
        let src = batch_source.into();
        self.push_internal(session_id, mode, input, Some(key), Some(src))
    }

    fn push_internal(
        &self,
        session_id: &str,
        mode: DeliveryMode,
        input: TurnInput,
        coalesce_key: Option<String>,
        append_batch_source: Option<String>,
    ) -> InboxPushResult {
        let (result, should_wake) = {
            let mut state = self.state.lock().unwrap();
            let is_active = state.active_sessions.contains(session_id);
            let is_human = input.is_user();

            let mailbox = state.mailboxes.entry(session_id.to_string()).or_default();
            mailbox.prune_and_compact();
            let was_barrier = mailbox.yield_barrier;
            if is_human {
                mailbox.yield_barrier = false;
            }
            let (effective_mode, degraded) =
                resolve_push_delivery(mailbox, is_active, mode, is_human, coalesce_key.as_deref());

            let res = if let Some(ref key) = coalesce_key
                && let Some(existing) = mailbox.items.iter_mut().find(|item| {
                    item.mode == effective_mode
                        && item.coalesce_key.as_deref() == Some(key.as_str())
                }) {
                let mut current = existing.materialize_input();
                if let Some(ref batch_source) = append_batch_source {
                    current.append_batch(input, batch_source);
                    existing.payload = MailboxPayload::Plain(current);
                } else {
                    existing.payload = MailboxPayload::Plain(input);
                }
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
                    .push(MailboxEntry {
                        id,
                        mode: effective_mode,
                        coalesce_key,
                        payload: MailboxPayload::Plain(input),
                    });
                InboxPushResult {
                    id,
                    effective_mode,
                    degraded,
                }
            };

            let mailbox = state.mailboxes.get(session_id).expect("mailbox exists");
            let wake = !mailbox.yield_barrier
                && (res.effective_mode == DeliveryMode::FollowUp
                    || (is_human && was_barrier && mailbox.has_deliverable_followup()));
            (res, wake)
        };

        if should_wake {
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
        mailbox.prune_and_compact();
        let mut drained = Vec::new();
        let mut remaining = Vec::with_capacity(mailbox.items.len());
        for item in mailbox.items.drain(..) {
            match item.mode {
                DeliveryMode::Steer | DeliveryMode::Inject => {
                    drained.push(item.materialize_input())
                }
                DeliveryMode::FollowUp => remaining.push(item),
            }
        }
        mailbox.items = remaining;
        drained
    }

    pub fn has_pending_steer(&self, session_id: &str) -> bool {
        let mut state = self.state.lock().unwrap();
        state.mailboxes.get_mut(session_id).is_some_and(|mb| {
            mb.prune_and_compact();
            mb.items.iter().any(|item| item.mode == DeliveryMode::Steer)
        })
    }

    pub fn has_pending_user_steer(&self, session_id: &str) -> bool {
        let mut state = self.state.lock().unwrap();
        state.mailboxes.get_mut(session_id).is_some_and(|mb| {
            mb.prune_and_compact();
            mb.items
                .iter()
                .any(|item| item.mode == DeliveryMode::Steer && item.is_user())
        })
    }

    /// 当停轮守卫或自动化 `Notice` 耗尽每轮防爆预算时，将剩余 `Notice` `Steer` 降级为 `Inject`，
    /// 既防止停轮后被 `release_session` 二次唤醒，又保留上下文内容。
    pub(crate) fn degrade_notice_steers_to_inject(&self, session_id: &str) -> usize {
        let mut state = self.state.lock().unwrap();
        let Some(mailbox) = state.mailboxes.get_mut(session_id) else {
            return 0;
        };
        mailbox.prune_and_compact();
        let mut count = 0usize;
        for item in &mut mailbox.items {
            if item.mode == DeliveryMode::Steer && !item.is_user() {
                item.mode = DeliveryMode::Inject;
                count += 1;
            }
        }
        count
    }

    pub fn pop_followup(&self, session_id: &str) -> Option<TurnInput> {
        let mut state = self.state.lock().unwrap();
        let mailbox = state.mailboxes.get_mut(session_id)?;
        mailbox.prune_and_compact();
        let idx = mailbox
            .items
            .iter()
            .position(|item| mailbox.is_entry_deliverable_followup(item))?;
        Some(mailbox.items.remove(idx).materialize_input())
    }

    pub fn snapshot(&self, session_id: &str) -> Vec<InboxItem> {
        let mut state = self.state.lock().unwrap();
        let Some(mailbox) = state.mailboxes.get_mut(session_id) else {
            return Vec::new();
        };
        mailbox.prune_and_compact();
        mailbox
            .items
            .iter()
            .map(MailboxEntry::to_inbox_item)
            .collect()
    }

    pub fn clear(&self, session_id: &str) {
        let mut state = self.state.lock().unwrap();
        if let Some(mailbox) = state.mailboxes.get_mut(session_id) {
            mailbox.items.clear();
            mailbox.yield_barrier = false;
        }
    }

    /// 清除所有未投递的用户输入条目，并将已到箱的非人类 `FollowUp`/`Steer` 通知转为 `Inject` 保留正文，
    /// 供交互 `Esc` 取消时立刻安静停住而不误删后台子代理已落箱成果。
    pub fn clear_user_items_and_defer_notices(&self, session_id: &str) -> usize {
        let mut state = self.state.lock().unwrap();
        let Some(mailbox) = state.mailboxes.get_mut(session_id) else {
            return 0;
        };
        mailbox.prune_and_compact();
        mailbox.yield_barrier = false;
        let before = mailbox.items.len();
        mailbox.items.retain(|item| !item.is_user());
        let mut affected = before.saturating_sub(mailbox.items.len());
        for item in &mut mailbox.items {
            if matches!(item.mode, DeliveryMode::FollowUp | DeliveryMode::Steer) {
                item.mode = DeliveryMode::Inject;
                affected += 1;
            }
        }
        affected
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

    pub fn remaining_auto_wakes(&self, session_id: &str) -> usize {
        let state = self.state.lock().unwrap();
        state
            .mailboxes
            .get(session_id)
            .map_or(DEFAULT_AUTO_WAKE_LIMIT, |mb| {
                mb.auto_wake_limit.saturating_sub(mb.auto_wakes_since_human)
            })
    }

    pub fn has_pending_followup_key(&self, session_id: &str, key: &str) -> bool {
        let mut state = self.state.lock().unwrap();
        state.mailboxes.get_mut(session_id).is_some_and(|mb| {
            mb.prune_and_compact();
            mb.items.iter().any(|item| {
                item.mode == DeliveryMode::FollowUp && item.coalesce_key.as_deref() == Some(key)
            })
        })
    }

    /// 判断当前是否存在可立即投递（未被 `yield` barrier 拦住、且非空）的 `FollowUp`。
    pub fn has_deliverable_followup(&self, session_id: &str) -> bool {
        let mut state = self.state.lock().unwrap();
        state.mailboxes.get_mut(session_id).is_some_and(|mb| {
            mb.prune_and_compact();
            mb.has_deliverable_followup()
        })
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

    #[test]
    fn push_appending_batches_followups_and_preserves_degraded_injects() {
        let inbox = SessionInbox::new();
        inbox.set_auto_wake_limit("s1", 2);
        assert_eq!(inbox.remaining_auto_wakes("s1"), 2);

        // 1. 同一波次 6 条并发通知合并入同一 FollowUp，只消耗 1 格预算且不降级
        for i in 1..=6 {
            let res = inbox.push_appending(
                "s1",
                DeliveryMode::FollowUp,
                TurnInput::notice(format!("subagent:w{i}"), format!("done {i}")),
                "subagent:settled",
                "subagent:batch",
            );
            assert_eq!(res.effective_mode, DeliveryMode::FollowUp);
            assert!(!res.degraded);
        }
        assert_eq!(inbox.auto_wakes_since_human("s1"), 1);
        assert_eq!(inbox.remaining_auto_wakes("s1"), 1);
        assert!(inbox.has_pending_followup_key("s1", "subagent:settled"));

        let merged = inbox.pop_followup("s1").unwrap();
        match merged {
            TurnInput::Notice { source, text, .. } => {
                assert_eq!(source, "subagent:batch");
                for i in 1..=6 {
                    assert!(
                        text.contains(&format!("done {i}")),
                        "missing done {i}: {text}"
                    );
                }
            }
            other => panic!("expected notice, got {other:?}"),
        }
        assert!(!inbox.has_pending_followup_key("s1", "subagent:settled"));

        // 2. 限额为 0 时多条通知均降级为 Inject，但仍然拼接保留全部正文绝不覆写
        inbox.set_auto_wake_limit("s2", 0);
        let d1 = inbox.push_appending(
            "s2",
            DeliveryMode::FollowUp,
            TurnInput::notice("subagent:a", "alpha result"),
            "subagent:settled",
            "subagent:batch",
        );
        let d2 = inbox.push_appending(
            "s2",
            DeliveryMode::FollowUp,
            TurnInput::notice("subagent:b", "beta result"),
            "subagent:settled",
            "subagent:batch",
        );
        assert_eq!(d1.effective_mode, DeliveryMode::Inject);
        assert!(d1.degraded);
        assert_eq!(d2.effective_mode, DeliveryMode::Inject);
        assert!(d2.degraded);

        let drained = inbox.drain_step_inputs("s2");
        assert_eq!(drained.len(), 1);
        assert!(drained[0].text().contains("alpha result"));
        assert!(drained[0].text().contains("beta result"));
    }

    #[test]
    fn clear_user_items_and_defer_notices_drops_user_and_keeps_notice_text() {
        let inbox = SessionInbox::new();
        inbox.set_auto_wake_limit("s1", 5);
        inbox.push("s1", DeliveryMode::FollowUp, TurnInput::user("user queued"));
        inbox.push_appending(
            "s1",
            DeliveryMode::FollowUp,
            TurnInput::notice("subagent:w1", "important summary"),
            "subagent:settled",
            "subagent:batch",
        );

        let affected = inbox.clear_user_items_and_defer_notices("s1");
        assert_eq!(affected, 2);
        assert!(inbox.pop_followup("s1").is_none());

        let snap = inbox.snapshot("s1");
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].mode, DeliveryMode::Inject);
        assert_eq!(snap[0].input.text(), "important summary");
    }

    #[test]
    fn subagent_slots_dedup_by_generation_and_drop_empty_batch_and_clear_watchdog() {
        let inbox = SessionInbox::new();
        inbox.set_auto_wake_limit("p1", 1);

        // 1. child-a 触发看门狗后结算：结算槽位自动清掉同 child 的未投递看门狗槽位
        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Watchdog,
            "child-a",
            1,
            "subagent-watchdog:a",
            "wd-1",
            true,
            true,
        );
        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Settled,
            "child-a",
            1,
            "subagent:a",
            "settled-run1",
            false,
            true,
        );
        let snap = inbox.snapshot("p1");
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].input.text(), "settled-run1");
        assert!(!inbox.has_pending_followup_key("p1", "subagent:watchdog"));

        // 2. 推进代次（send 续跑 run2）：旧代次 settled 槽位失效并整条移除，且归还自动唤醒额度
        inbox.advance_subagent_generation("p1", "child-a", 2);
        assert!(!inbox.has_pending_followup_key("p1", "subagent:settled"));
        assert!(inbox.pop_followup("p1").is_none());
        assert_eq!(inbox.remaining_auto_wakes("p1"), 1);

        // 3. 过期 run1 再尝试投递必须被忽略；run2 与 child-b 合并，失效 child-a 后仅保留 child-b
        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Settled,
            "child-a",
            1,
            "subagent:a",
            "stale-run1",
            false,
            false,
        );
        assert!(inbox.pop_followup("p1").is_none());

        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Settled,
            "child-a",
            2,
            "subagent:a",
            "settled-run2",
            false,
            false,
        );
        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Settled,
            "child-b",
            3,
            "subagent:b",
            "settled-b",
            false,
            true,
        );
        assert_eq!(
            inbox.invalidate_subagent_slots("p1", "child-a", Some(SubagentSlotKind::Settled)),
            1
        );
        let msg = inbox.pop_followup("p1").unwrap();
        assert_eq!(msg.text(), "settled-b");
        assert!(inbox.pop_followup("p1").is_none());
    }

    #[test]
    fn yield_barrier_holds_non_urgent_settlements_and_breaks_on_urgent_or_human() {
        let inbox = SessionInbox::new();
        let mut wake_rx = inbox.subscribe_wake();

        // 1. 挂载 barrier 后，首个子代理正常完成不得提前唤醒或被 pop_followup 消费
        assert!(inbox.arm_yield_barrier("p1"));
        assert!(inbox.has_yield_barrier("p1"));
        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Settled,
            "c1",
            1,
            "subagent:c1",
            "result-c1",
            false,
            false,
        );
        assert!(wake_rx.try_recv().is_err());
        assert!(!inbox.has_deliverable_followup("p1"));
        assert!(inbox.pop_followup("p1").is_none());

        // 第二个子代理全部完成（release_barrier = true），一次唤醒打包投递两者
        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Settled,
            "c2",
            2,
            "subagent:c2",
            "result-c2",
            false,
            true,
        );
        assert!(!inbox.has_yield_barrier("p1"));
        assert_eq!(wake_rx.try_recv().unwrap(), "p1");
        let batched = inbox.pop_followup("p1").unwrap();
        assert!(batched.text().contains("result-c1"));
        assert!(batched.text().contains("result-c2"));
        assert!(inbox.pop_followup("p1").is_none());

        // 2. 子代理失败/看门狗（urgent = true）立即打破 barrier
        assert!(inbox.arm_yield_barrier("p1"));
        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Settled,
            "c3",
            3,
            "subagent:c3",
            "ok-c3",
            false,
            false,
        );
        assert!(wake_rx.try_recv().is_err());
        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Watchdog,
            "c4",
            4,
            "subagent-watchdog:c4",
            "stalled-c4",
            true,
            false,
        );
        assert!(!inbox.has_yield_barrier("p1"));
        assert_eq!(wake_rx.try_recv().unwrap(), "p1");
        assert!(inbox.pop_followup("p1").unwrap().text().contains("ok-c3"));
        assert!(
            inbox
                .pop_followup("p1")
                .unwrap()
                .text()
                .contains("stalled-c4")
        );

        // 3. 人类输入立即打破 barrier
        assert!(inbox.arm_yield_barrier("p1"));
        inbox.push_subagent_slot(
            "p1",
            DeliveryMode::FollowUp,
            SubagentSlotKind::Settled,
            "c5",
            5,
            "subagent:c5",
            "ok-c5",
            false,
            false,
        );
        assert!(inbox.pop_followup("p1").is_none());
        inbox.push(
            "p1",
            DeliveryMode::FollowUp,
            TurnInput::user("human interrupt"),
        );
        assert!(!inbox.has_yield_barrier("p1"));
        assert_eq!(wake_rx.try_recv().unwrap(), "p1");
        assert!(inbox.pop_followup("p1").is_some());
        assert!(inbox.pop_followup("p1").is_some());
    }
}
