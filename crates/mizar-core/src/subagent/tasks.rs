//! 后台 Subagent 任务注册表、实时活性追踪与唯一代次状态机。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::agent_loop::inbox::SubagentSlotKind;
use crate::agent_loop::{DeliveryMode, SessionInbox, TurnInput};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::subagent::{SubagentConfig, SubagentError, SubagentMode};

const MAX_SETTLED_PER_PARENT: usize = 32;
pub const SUBAGENT_OUTPUT_INLINE_MAX_CHARS: usize = 12_000;
pub const SUBAGENT_OUTPUT_PREVIEW_CHARS: usize = 5_000;

/// 截取前 `max_chars` 个 Unicode 标量字符的预览片段，优先在后半段换行边界断开避免截断 Markdown 行。
pub(crate) fn preview_head_chars(text: &str, max_chars: usize) -> String {
    let mut cut_byte = text.len();
    for (char_count, (idx, _)) in text.char_indices().enumerate() {
        if char_count == max_chars {
            cut_byte = idx;
            break;
        }
    }
    let slice = &text[..cut_byte];
    if let Some(nl) = slice.rfind('\n')
        && slice[..nl].chars().count() >= max_chars / 2
    {
        return slice[..nl].to_string();
    }
    slice.to_string()
}

/// 格式化终态子代理输出：`<= 12,000` 字符直接内联；超长自动落盘临时文件并附 `5,000` 字符预览与路径。
pub(crate) fn format_terminal_output_for_delivery(
    raw_output: &str,
    child_session_id: &str,
    run_id: u64,
) -> (String, Option<PathBuf>) {
    let total_chars = raw_output.chars().count();
    if total_chars <= SUBAGENT_OUTPUT_INLINE_MAX_CHARS {
        return (raw_output.to_string(), None);
    }

    let preview = preview_head_chars(raw_output, SUBAGENT_OUTPUT_PREVIEW_CHARS);
    let preview_chars = preview.chars().count();
    let dir = std::env::temp_dir().join("mizar-subagent-outputs");
    let file_path = dir.join(format!("{child_session_id}-run{run_id}.txt"));

    match std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&file_path, raw_output)) {
        Ok(()) => {
            let formatted = format!(
                "{preview}\n\n[Output truncated: {total_chars} chars total, showing {preview_chars}-char preview. Full output saved to: {} (inspect with `read`)]",
                file_path.display()
            );
            (formatted, Some(file_path))
        }
        Err(err) => {
            let formatted = format!(
                "{preview}\n\n[Output truncated: {total_chars} chars total, showing {preview_chars}-char preview; failed to spill full output to temp file: {err}]"
            );
            (formatted, None)
        }
    }
}

/// 后台或同步 Subagent 单次运行的生命周期状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentRunStatus {
    Running,
    Completed,
    MaxSteps,
    Truncated,
    Failed,
    TimedOut,
    Stopped,
}

impl SubagentRunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::MaxSteps => "max_steps",
            Self::Truncated => "truncated",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
            Self::Stopped => "stopped",
        }
    }

    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }
}

/// 面向调用方查询与状态投影的 Subagent 任务只读快照。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentTaskSnapshot {
    pub name: String,
    pub parent_session_id: String,
    pub child_session_id: String,
    pub mode: SubagentMode,
    pub prompt_preview: String,
    pub status: SubagentRunStatus,
    pub steps: usize,
    pub current_step: usize,
    pub last_tool: Option<String>,
    pub idle_secs: u64,
    pub watchdog_alerts: usize,
    pub elapsed_secs: u64,
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
}

/// 通过强类型状态事件总线广播的 Subagent 生命周期变更通知。
#[derive(Debug, Clone)]
pub struct SubagentStateEvent {
    pub parent_session_id: String,
    pub name: String,
    pub child_session_id: String,
    pub mode: SubagentMode,
    pub status: SubagentRunStatus,
    pub running_count: usize,
    pub notice: Option<String>,
}

/// 单个活跃子会话的轻量无阻塞活性与软步数预算追踪器（供事件监听刷新与零 Token 看门狗读取）。
pub struct ChildActivityTracker {
    started_at: Instant,
    last_activity_ms: AtomicU64,
    step_offset: AtomicUsize,
    current_step: AtomicUsize,
    last_tool: Mutex<Option<String>>,
    soft_budget: usize,
    stop_threshold: Option<usize>,
    budget_steer_sent: AtomicBool,
    in_forced_wrapup: AtomicBool,
}

impl ChildActivityTracker {
    pub fn new(step_offset: usize) -> Self {
        Self::with_soft_budget(step_offset, 0)
    }

    pub fn with_soft_budget(step_offset: usize, soft_budget: usize) -> Self {
        Self {
            started_at: Instant::now(),
            last_activity_ms: AtomicU64::new(0),
            step_offset: AtomicUsize::new(step_offset),
            current_step: AtomicUsize::new(step_offset),
            last_tool: Mutex::new(None),
            soft_budget,
            stop_threshold: SubagentConfig::force_stop_steps(soft_budget),
            budget_steer_sent: AtomicBool::new(false),
            in_forced_wrapup: AtomicBool::new(false),
        }
    }

    pub fn reset_turn_budget(&self) {
        self.budget_steer_sent.store(false, Ordering::Release);
        self.in_forced_wrapup.store(false, Ordering::Release);
    }

    pub fn enter_forced_wrapup(&self) {
        self.in_forced_wrapup.store(true, Ordering::Release);
    }

    pub fn take_budget_steer_notice_if_due(&self, turn_step: usize) -> Option<String> {
        if self.soft_budget == 0 || self.in_forced_wrapup.load(Ordering::Acquire) {
            return None;
        }
        if turn_step < self.soft_budget {
            return None;
        }
        if self
            .budget_steer_sent
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return None;
        }
        let stop_at = self.stop_threshold.unwrap_or(self.soft_budget);
        Some(format!(
            "[budget notice] You have used {turn_step} steps in this run (soft budget: {}). Wrap up now: finish your findings and deliver your final report without starting unnecessary extra steps. At {stop_at} steps the free tool loop will stop and you will be asked to summarize what you have.",
            self.soft_budget
        ))
    }

    pub fn touch(&self) {
        let elapsed_ms = u64::try_from(self.started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.last_activity_ms.store(elapsed_ms, Ordering::Relaxed);
    }

    pub fn set_turn_step(&self, turn_step: usize) {
        let offset = self.step_offset.load(Ordering::Relaxed);
        self.current_step
            .store(offset.saturating_add(turn_step), Ordering::Relaxed);
        self.touch();
    }

    pub fn advance_step_offset(&self, delta_steps: usize) {
        let next = self
            .step_offset
            .fetch_add(delta_steps, Ordering::Relaxed)
            .saturating_add(delta_steps);
        self.current_step.store(next, Ordering::Relaxed);
        self.touch();
    }

    pub fn set_tool(&self, label: &str) {
        let trimmed = label.trim();
        if !trimmed.is_empty() {
            let capped: String = trimmed.chars().take(64).collect();
            *self.last_tool.lock().unwrap() = Some(capped);
        }
        self.touch();
    }

    pub fn last_tool(&self) -> Option<String> {
        self.last_tool.lock().unwrap().clone()
    }

    pub fn last_activity_instant(&self) -> Instant {
        let ms = self.last_activity_ms.load(Ordering::Relaxed);
        self.started_at
            .checked_add(Duration::from_millis(ms))
            .unwrap_or(self.started_at)
    }

    pub fn idle_duration(&self) -> Duration {
        Instant::now().saturating_duration_since(self.last_activity_instant())
    }

    pub fn snapshot(&self) -> (usize, Option<String>, u64) {
        (
            self.current_step.load(Ordering::Relaxed),
            self.last_tool(),
            self.idle_duration().as_secs(),
        )
    }
}

struct ManagedTask {
    name: String,
    parent_session_id: String,
    child_session_id: String,
    mode: SubagentMode,
    provider: Option<String>,
    model: Option<String>,
    prompt_preview: String,
    status: SubagentRunStatus,
    steps: usize,
    last_turn_steps: usize,
    last_tool: Option<String>,
    summary: Option<String>,
    last_output: Option<String>,
    last_output_path: Option<PathBuf>,
    started_at: Instant,
    finished_at: Option<Instant>,
    cancel: CancellationToken,
    stop_requested: bool,
    active_run_id: Option<u64>,
    watchdog_alerts: usize,
    activity: Arc<ChildActivityTracker>,
}

impl ManagedTask {
    fn snapshot(&self) -> SubagentTaskSnapshot {
        let elapsed_secs = self
            .finished_at
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(self.started_at)
            .as_secs();
        let (current_step, last_tool, idle_secs) =
            if !self.status.is_terminal() && self.active_run_id.is_some() {
                self.activity.snapshot()
            } else {
                (self.steps, self.last_tool.clone(), 0)
            };
        SubagentTaskSnapshot {
            name: self.name.clone(),
            parent_session_id: self.parent_session_id.clone(),
            child_session_id: self.child_session_id.clone(),
            mode: self.mode,
            prompt_preview: self.prompt_preview.clone(),
            status: self.status,
            steps: self.steps,
            current_step,
            last_tool,
            idle_secs,
            watchdog_alerts: self.watchdog_alerts,
            elapsed_secs,
            summary: self.summary.clone(),
            output_path: self
                .last_output_path
                .as_ref()
                .map(|p| p.display().to_string()),
        }
    }
}

pub(crate) struct SettledRollback {
    status: SubagentRunStatus,
    prompt_preview: String,
    summary: Option<String>,
    last_output: Option<String>,
    last_output_path: Option<PathBuf>,
    started_at: Instant,
    finished_at: Option<Instant>,
    cancel: CancellationToken,
    watchdog_alerts: usize,
    activity: Arc<ChildActivityTracker>,
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum SendOutcome {
    SteeredRunning {
        name: String,
        child_session_id: String,
        mode: SubagentMode,
    },
    Reactivated {
        run_id: u64,
        name: String,
        child_session_id: String,
        mode: SubagentMode,
        prev_provider: Option<String>,
        prev_model: Option<String>,
        activity: Arc<ChildActivityTracker>,
        cancel: CancellationToken,
        rollback: SettledRollback,
    },
}

pub(crate) type SettlementNoticeBuilder<'a> =
    &'a dyn Fn(SubagentRunStatus, usize, u64) -> Option<(String, bool)>;

pub(crate) enum WorkerTurnAction {
    Continue { next_input: TurnInput },
    SettledWaitGuard { was_stop: bool },
}

/// 校验并归一化自定义子代理名称：1..=32 位小写字母/数字/`-`/`_`，且首字符为字母或数字、不可为 `"all"`。
pub(crate) fn validate_custom_name(raw: &str) -> Result<String, SubagentError> {
    let trimmed = raw.trim();
    let normalized = trimmed.to_ascii_lowercase();
    if normalized.is_empty()
        || normalized.len() > 32
        || normalized == "all"
        || !normalized
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        || !normalized
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_'))
    {
        return Err(SubagentError::InvalidName(trimmed.to_string()));
    }
    Ok(normalized)
}

pub(crate) struct SingleStatusDetail {
    pub snapshot: SubagentTaskSnapshot,
    pub last_output: Option<String>,
    pub last_output_path: Option<PathBuf>,
}

/// 按父会话隔离的纯内存后台 Subagent 注册表。
#[derive(Default)]
pub(crate) struct BackgroundTaskRegistry {
    next_run_id: u64,
    next_auto_seq: HashMap<String, usize>,
    entries: HashMap<String, Vec<ManagedTask>>,
}

impl BackgroundTaskRegistry {
    fn allocate_next_run_id(counter: &mut u64) -> u64 {
        *counter = counter.wrapping_add(1).max(1);
        *counter
    }

    fn prune_settled_overflow(list: &mut Vec<ManagedTask>, parent_id: &str, inbox: &SessionInbox) {
        loop {
            let settled_idle_count = list
                .iter()
                .filter(|t| t.status.is_terminal() && t.active_run_id.is_none())
                .count();
            if settled_idle_count <= MAX_SETTLED_PER_PARENT {
                break;
            }
            let oldest_idx = list
                .iter()
                .enumerate()
                .filter(|(_, t)| t.status.is_terminal() && t.active_run_id.is_none())
                .min_by_key(|(idx, t)| (t.finished_at, *idx))
                .map(|(idx, _)| idx);
            let Some(idx) = oldest_idx else {
                break;
            };
            let removed = list.remove(idx);
            inbox.invalidate_subagent_slots(parent_id, &removed.child_session_id, None);
        }
    }

    /// 创建子会话前预先检查并发上限与自定义名称冲突，避免无谓创建孤儿会话。
    pub fn preflight_spawn(
        &self,
        parent_id: &str,
        requested_name: Option<&str>,
        max_concurrent: usize,
    ) -> Result<(), SubagentError> {
        let cap = max_concurrent.max(1);
        let active = self.active_run_count(parent_id);
        if active >= cap {
            return Err(SubagentError::ConcurrencyLimitReached(cap));
        }
        if let Some(raw) = requested_name {
            let name = validate_custom_name(raw)?;
            let conflict = self.entries.get(parent_id).and_then(|list| {
                list.iter().find(|t| {
                    t.name == name && (!t.status.is_terminal() || t.active_run_id.is_some())
                })
            });
            if let Some(entry) = conflict {
                if entry.stop_requested || entry.status.is_terminal() {
                    return Err(SubagentError::StoppingInProgress(name));
                }
                return Err(SubagentError::NameAlreadyRunning(name));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register_running(
        &mut self,
        parent_id: &str,
        requested_name: Option<&str>,
        child_session_id: String,
        mode: SubagentMode,
        provider: Option<String>,
        model: Option<String>,
        prompt_preview: String,
        cancel: CancellationToken,
        max_concurrent: usize,
        soft_budget: usize,
        inbox: &SessionInbox,
    ) -> Result<(u64, String, usize, Arc<ChildActivityTracker>), SubagentError> {
        self.preflight_spawn(parent_id, requested_name, max_concurrent)?;

        let run_id = Self::allocate_next_run_id(&mut self.next_run_id);
        let assigned_name = match requested_name {
            Some(raw) => {
                let name = validate_custom_name(raw)?;
                if let Some(list) = self.entries.get_mut(parent_id) {
                    // 仅清理同名已彻底终态（active_run_id.is_none()）旧条目，并同步撤回其残留信箱槽位
                    let mut i = 0;
                    while i < list.len() {
                        if list[i].name == name {
                            let old = list.remove(i);
                            inbox.invalidate_subagent_slots(parent_id, &old.child_session_id, None);
                        } else {
                            i += 1;
                        }
                    }
                }
                name
            }
            None => {
                let seq = self.next_auto_seq.entry(parent_id.to_string()).or_insert(0);
                loop {
                    *seq = seq.saturating_add(1);
                    let candidate = format!("agent-{}", *seq);
                    let in_use = self
                        .entries
                        .get(parent_id)
                        .is_some_and(|list| list.iter().any(|t| t.name == candidate));
                    if !in_use {
                        break candidate;
                    }
                }
            }
        };

        let activity = Arc::new(ChildActivityTracker::with_soft_budget(0, soft_budget));
        let list = self.entries.entry(parent_id.to_string()).or_default();
        list.push(ManagedTask {
            name: assigned_name.clone(),
            parent_session_id: parent_id.to_string(),
            child_session_id,
            mode,
            provider,
            model,
            prompt_preview,
            status: SubagentRunStatus::Running,
            steps: 0,
            last_turn_steps: 0,
            last_tool: None,
            summary: None,
            last_output: None,
            last_output_path: None,
            started_at: Instant::now(),
            finished_at: None,
            cancel,
            stop_requested: false,
            active_run_id: Some(run_id),
            watchdog_alerts: 0,
            activity: Arc::clone(&activity),
        });

        let running_count = list.iter().filter(|t| !t.status.is_terminal()).count();
        Ok((run_id, assigned_name, running_count, activity))
    }

    /// 在单一 `tasks` 锁内原子判定 `send` 是向运行中子代理推入 `Steer`，还是抢占重激活已休眠分身。
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_send(
        &mut self,
        parent_id: &str,
        raw_name: &str,
        prompt_preview: String,
        requested_mode: Option<SubagentMode>,
        requested_model: Option<&str>,
        inbox: &SessionInbox,
        prompt: &str,
        cfg: &SubagentConfig,
        dispose_token: &CancellationToken,
    ) -> Result<SendOutcome, SubagentError> {
        let name = validate_custom_name(raw_name)?;
        let cap = cfg.effective_max_concurrent();
        let active = self.active_run_count(parent_id);

        let Some(list) = self.entries.get_mut(parent_id) else {
            return Err(SubagentError::SubagentNotFound(name));
        };
        let Some(idx) = list
            .iter()
            .position(|t| t.name == name || t.name.eq_ignore_ascii_case(&name))
        else {
            return Err(SubagentError::SubagentNotFound(name));
        };

        if let Some(req_mode) = requested_mode
            && req_mode != list[idx].mode
        {
            return Err(SubagentError::ModeChangeNotAllowed(
                list[idx].name.clone(),
                list[idx].mode.as_str().to_string(),
                req_mode.as_str().to_string(),
            ));
        }

        if list[idx].active_run_id.is_some() {
            if list[idx].status == SubagentRunStatus::Running && !list[idx].stop_requested {
                if requested_model.is_some() {
                    return Err(SubagentError::ModelChangeWhileRunning(
                        list[idx].name.clone(),
                    ));
                }
                let child_id = list[idx].child_session_id.clone();
                let mode = list[idx].mode;
                let assigned = list[idx].name.clone();
                // 保持 tasks 锁内推进 inbox，消除与 worker 结算退出的 TOCTOU 窗口（锁序: tasks -> inbox.state）
                inbox.invalidate_subagent_slots(
                    parent_id,
                    &child_id,
                    Some(SubagentSlotKind::Watchdog),
                );
                inbox.push(&child_id, DeliveryMode::Steer, TurnInput::user(prompt));
                return Ok(SendOutcome::SteeredRunning {
                    name: assigned,
                    child_session_id: child_id,
                    mode,
                });
            }
            return Err(SubagentError::StoppingInProgress(list[idx].name.clone()));
        }

        if active >= cap {
            return Err(SubagentError::ConcurrencyLimitReached(cap));
        }

        let run_id = Self::allocate_next_run_id(&mut self.next_run_id);
        let mut entry = list.remove(idx);
        let rollback = SettledRollback {
            status: entry.status,
            prompt_preview: entry.prompt_preview.clone(),
            summary: entry.summary.clone(),
            last_output: entry.last_output.clone(),
            last_output_path: entry.last_output_path.clone(),
            started_at: entry.started_at,
            finished_at: entry.finished_at,
            cancel: entry.cancel.clone(),
            watchdog_alerts: entry.watchdog_alerts,
            activity: Arc::clone(&entry.activity),
        };

        // 唤醒续跑推进 run_id，立即清除该子代理旧代次遗留的 settled / watchdog 通知槽位
        inbox.advance_subagent_generation(parent_id, &entry.child_session_id, run_id);

        let new_cancel = dispose_token.child_token();
        let new_activity = Arc::new(ChildActivityTracker::with_soft_budget(
            entry.steps,
            cfg.soft_budget_for(entry.mode),
        ));
        entry.status = SubagentRunStatus::Running;
        entry.prompt_preview = prompt_preview;
        entry.summary = None;
        entry.last_output = None;
        entry.last_output_path = None;
        entry.started_at = Instant::now();
        entry.finished_at = None;
        entry.cancel = new_cancel.clone();
        entry.stop_requested = false;
        entry.active_run_id = Some(run_id);
        entry.watchdog_alerts = 0;
        entry.activity = Arc::clone(&new_activity);

        let out = SendOutcome::Reactivated {
            run_id,
            name: entry.name.clone(),
            child_session_id: entry.child_session_id.clone(),
            mode: entry.mode,
            prev_provider: entry.provider.clone(),
            prev_model: entry.model.clone(),
            activity: new_activity,
            cancel: new_cancel,
            rollback,
        };
        list.push(entry);
        Ok(out)
    }

    pub fn rollback_reactivation(
        &mut self,
        parent_id: &str,
        name: &str,
        child_session_id: &str,
        expected_run_id: u64,
        rollback: SettledRollback,
    ) {
        let Some(list) = self.entries.get_mut(parent_id) else {
            return;
        };
        let Some(entry) = list.iter_mut().find(|t| {
            t.name == name
                && t.child_session_id == child_session_id
                && t.active_run_id == Some(expected_run_id)
        }) else {
            return;
        };
        entry.active_run_id = None;
        entry.activity = rollback.activity;
        if entry.stop_requested {
            entry.status = SubagentRunStatus::Stopped;
            entry.finished_at.get_or_insert_with(Instant::now);
        } else {
            entry.status = rollback.status;
            entry.prompt_preview = rollback.prompt_preview;
            entry.summary = rollback.summary;
            entry.last_output = rollback.last_output;
            entry.last_output_path = rollback.last_output_path;
            entry.started_at = rollback.started_at;
            entry.finished_at = rollback.finished_at;
            entry.cancel = rollback.cancel;
            entry.watchdog_alerts = rollback.watchdog_alerts;
        }
    }

    pub fn update_route(
        &mut self,
        parent_id: &str,
        name: &str,
        child_session_id: &str,
        expected_run_id: u64,
        provider: Option<String>,
        model: Option<String>,
    ) {
        if let Some(list) = self.entries.get_mut(parent_id)
            && let Some(entry) = list.iter_mut().find(|t| {
                t.name == name
                    && t.child_session_id == child_session_id
                    && t.active_run_id == Some(expected_run_id)
            })
        {
            entry.provider = provider;
            entry.model = model;
        }
    }

    pub fn record_watchdog_alert(
        &mut self,
        parent_id: &str,
        name: &str,
        child_session_id: &str,
        expected_run_id: u64,
    ) -> Option<(usize, usize)> {
        let list = self.entries.get_mut(parent_id)?;
        let entry = list.iter_mut().find(|t| {
            t.name == name
                && t.child_session_id == child_session_id
                && t.active_run_id == Some(expected_run_id)
                && !t.status.is_terminal()
        })?;
        entry.watchdog_alerts = entry.watchdog_alerts.saturating_add(1);
        let alerts = entry.watchdog_alerts;
        let running = list.iter().filter(|t| !t.status.is_terminal()).count();
        Some((alerts, running))
    }

    /// 正常轮次结束后同一把 `tasks` 锁内判定是否续跑挂起 `send` 指令；
    /// 若无挂起指令（或已请求停止），先置终态暂保 `active_run_id` 阻断过渡期重入，待释放守卫后再调用 `finish_active_run`。
    #[allow(clippy::too_many_arguments)]
    pub fn begin_normal_finish_or_continue(
        &mut self,
        parent_id: &str,
        name: &str,
        child_session_id: &str,
        expected_run_id: u64,
        turn_steps: usize,
        status_if_settled: SubagentRunStatus,
        summary_if_settled: Option<String>,
        last_output_if_settled: Option<String>,
        last_output_path_if_settled: Option<PathBuf>,
        inbox: &SessionInbox,
        shorten_preview: impl Fn(&str) -> String,
    ) -> Option<WorkerTurnAction> {
        let now = Instant::now();
        let list = self.entries.get_mut(parent_id)?;
        let entry = list.iter_mut().find(|t| {
            t.name == name
                && t.child_session_id == child_session_id
                && t.active_run_id == Some(expected_run_id)
        })?;

        entry.steps = entry.steps.saturating_add(turn_steps);
        entry.last_turn_steps = turn_steps;
        entry.last_tool = entry.activity.last_tool();

        if !entry.stop_requested
            && let Some(next_input) = inbox.pop_followup(child_session_id)
        {
            entry.activity.advance_step_offset(turn_steps);
            let preview = shorten_preview(next_input.text());
            if !preview.is_empty() {
                entry.prompt_preview = preview;
            }
            return Some(WorkerTurnAction::Continue { next_input });
        }

        if entry.stop_requested {
            inbox.clear(child_session_id);
        }
        let was_stop = entry.stop_requested;
        entry.status = if was_stop {
            SubagentRunStatus::Stopped
        } else {
            status_if_settled
        };
        entry.summary = summary_if_settled;
        entry.last_output = last_output_if_settled;
        entry.last_output_path = last_output_path_if_settled;
        entry.finished_at.get_or_insert(now);
        Some(WorkerTurnAction::SettledWaitGuard { was_stop })
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_and_notify_inbox(
        list: &mut Vec<ManagedTask>,
        parent_id: &str,
        name: &str,
        child_session_id: &str,
        expected_run_id: u64,
        was_stop: bool,
        final_status: SubagentRunStatus,
        total_steps: usize,
        elapsed_secs: u64,
        inbox: &SessionInbox,
        build_notice: Option<SettlementNoticeBuilder<'_>>,
    ) -> (bool, usize, usize, u64, bool) {
        Self::prune_settled_overflow(list, parent_id, inbox);
        let running_count = list.iter().filter(|t| !t.status.is_terminal()).count();
        let release_barrier = running_count == 0;
        let mut degraded = false;
        if was_stop {
            if release_barrier {
                inbox.release_yield_barrier(parent_id);
            }
        } else if let Some(make_notice) = build_notice
            && let Some((inbox_text, urgent)) = make_notice(final_status, total_steps, elapsed_secs)
        {
            let res = inbox.push_subagent_slot(
                parent_id,
                DeliveryMode::FollowUp,
                SubagentSlotKind::Settled,
                child_session_id,
                expected_run_id,
                format!("subagent:{name}"),
                inbox_text,
                urgent,
                release_barrier,
            );
            degraded = res.degraded;
        } else if release_barrier {
            inbox.release_yield_barrier(parent_id);
        }
        (was_stop, total_steps, running_count, elapsed_secs, degraded)
    }

    /// 在 `child_guard` 已释放后校验匹配 `expected_run_id` 并清理 `active_run_id = None`、淘汰与原子投递信箱。
    #[allow(clippy::too_many_arguments)]
    pub fn finish_active_run(
        &mut self,
        parent_id: &str,
        name: &str,
        child_session_id: &str,
        expected_run_id: u64,
        inbox: &SessionInbox,
        build_notice: Option<SettlementNoticeBuilder<'_>>,
    ) -> Option<(bool, usize, usize, u64, bool)> {
        let now = Instant::now();
        let list = self.entries.get_mut(parent_id)?;
        let entry = list.iter_mut().find(|t| {
            t.name == name
                && t.child_session_id == child_session_id
                && t.active_run_id == Some(expected_run_id)
        })?;
        let was_stop = entry.stop_requested;
        let final_status = entry.status;
        let total_steps = entry.steps;
        let finished_at = *entry.finished_at.get_or_insert(now);
        let elapsed_secs = finished_at
            .saturating_duration_since(entry.started_at)
            .as_secs();
        entry.active_run_id = None;

        Some(Self::finalize_and_notify_inbox(
            list,
            parent_id,
            name,
            child_session_id,
            expected_run_id,
            was_stop,
            final_status,
            total_steps,
            elapsed_secs,
            inbox,
            build_notice,
        ))
    }

    /// 供异常/取消/超时/卸载路径在 `drop(child_guard)` 后一步完成结算、清 `active_run_id = None` 与信箱投递。
    #[allow(clippy::too_many_arguments)]
    pub fn mark_settled(
        &mut self,
        parent_id: &str,
        name: &str,
        child_session_id: &str,
        expected_run_id: u64,
        status: SubagentRunStatus,
        add_turn_steps: usize,
        summary: Option<String>,
        last_output: Option<String>,
        last_output_path: Option<PathBuf>,
        inbox: &SessionInbox,
        build_notice: Option<SettlementNoticeBuilder<'_>>,
    ) -> Option<(bool, usize, usize, u64, bool)> {
        let now = Instant::now();
        let list = self.entries.get_mut(parent_id)?;
        let entry = list.iter_mut().find(|t| {
            t.name == name
                && t.child_session_id == child_session_id
                && t.active_run_id == Some(expected_run_id)
        })?;

        let was_stop = entry.stop_requested;
        let final_status = if was_stop {
            SubagentRunStatus::Stopped
        } else {
            status
        };
        entry.status = final_status;
        entry.steps = entry.steps.saturating_add(add_turn_steps);
        entry.last_turn_steps = add_turn_steps;
        entry.last_tool = entry.activity.last_tool();
        entry.summary = summary;
        entry.last_output = last_output;
        entry.last_output_path = last_output_path;
        let total_steps = entry.steps;
        let finished_at = *entry.finished_at.get_or_insert(now);
        let elapsed_secs = finished_at
            .saturating_duration_since(entry.started_at)
            .as_secs();
        entry.active_run_id = None;

        Some(Self::finalize_and_notify_inbox(
            list,
            parent_id,
            name,
            child_session_id,
            expected_run_id,
            was_stop,
            final_status,
            total_steps,
            elapsed_secs,
            inbox,
            build_notice,
        ))
    }

    pub fn request_stop(
        &mut self,
        parent_id: &str,
        target: &str,
        inbox: &SessionInbox,
    ) -> Vec<(String, String, SubagentMode)> {
        let Some(list) = self.entries.get_mut(parent_id) else {
            return Vec::new();
        };
        let now = Instant::now();
        let normalized_target = target.trim();
        let match_all = normalized_target.eq_ignore_ascii_case("all");
        let mut stopped = Vec::new();
        for entry in list.iter_mut() {
            if entry.status.is_terminal() {
                continue;
            }
            if match_all
                || entry.name == normalized_target
                || entry.name.eq_ignore_ascii_case(normalized_target)
                || entry.child_session_id == normalized_target
            {
                entry.stop_requested = true;
                entry.status = SubagentRunStatus::Stopped;
                entry.finished_at.get_or_insert(now);
                // 绝不清除 active_run_id：仅由对应 run_id 的 Worker 退出或 rollback 清理
                entry.cancel.cancel();
                inbox.invalidate_subagent_slots(parent_id, &entry.child_session_id, None);
                stopped.push((
                    entry.name.clone(),
                    entry.child_session_id.clone(),
                    entry.mode,
                ));
            }
        }
        if !stopped.is_empty() && self.running_count(parent_id) == 0 {
            inbox.release_yield_barrier(parent_id);
        }
        stopped
    }

    pub fn cancel_parent(
        &mut self,
        parent_id: &str,
        inbox: &SessionInbox,
    ) -> Vec<(String, String, SubagentMode)> {
        self.request_stop(parent_id, "all", inbox)
    }

    /// 取消全部运行中子代理，并为每个受影响的父会话返回一条代表记录以便广播 `running_count = 0`。
    pub fn cancel_all(&mut self) -> Vec<(String, String, String, SubagentMode)> {
        let now = Instant::now();
        let mut affected = Vec::new();
        for (parent_id, list) in self.entries.iter_mut() {
            let mut last_stopped = None;
            for entry in list.iter_mut() {
                if !entry.status.is_terminal() {
                    entry.stop_requested = true;
                    entry.status = SubagentRunStatus::Stopped;
                    entry.finished_at.get_or_insert(now);
                    entry.cancel.cancel();
                    last_stopped = Some((
                        parent_id.clone(),
                        entry.name.clone(),
                        entry.child_session_id.clone(),
                        entry.mode,
                    ));
                }
            }
            if let Some(rep) = last_stopped {
                affected.push(rep);
            }
        }
        affected
    }

    pub fn running_count(&self, parent_id: &str) -> usize {
        self.entries.get(parent_id).map_or(0, |list| {
            list.iter().filter(|t| !t.status.is_terminal()).count()
        })
    }

    /// 统计当前仍在占用真实后台执行槽位的任务数（含处于 stop 过渡期、尚未清空 `active_run_id` 的任务）。
    pub fn active_run_count(&self, parent_id: &str) -> usize {
        self.entries.get(parent_id).map_or(0, |list| {
            list.iter().filter(|t| t.active_run_id.is_some()).count()
        })
    }

    pub fn snapshots(
        &self,
        parent_id: &str,
        filter_name: Option<&str>,
    ) -> Vec<SubagentTaskSnapshot> {
        let Some(list) = self.entries.get(parent_id) else {
            return Vec::new();
        };
        let filter = filter_name.map(str::trim).filter(|s| !s.is_empty());
        list.iter()
            .filter(|t| match filter {
                None => true,
                Some(f) => t.name == f || t.name.eq_ignore_ascii_case(f) || t.child_session_id == f,
            })
            .map(ManagedTask::snapshot)
            .collect()
    }

    /// 查询子代理状态；当指定 `filter_name` 精确命中单个终态子代理时返回完整输出（或预览+路径）并消费其未投递回执槽位。
    pub fn inspect_status(
        &self,
        parent_id: &str,
        filter_name: Option<&str>,
        inbox: &SessionInbox,
    ) -> (Vec<SubagentTaskSnapshot>, usize, Option<SingleStatusDetail>) {
        let Some(list) = self.entries.get(parent_id) else {
            return (Vec::new(), 0, None);
        };
        let filter = filter_name.map(str::trim).filter(|s| !s.is_empty());
        let matched: Vec<&ManagedTask> = list
            .iter()
            .filter(|t| match filter {
                None => true,
                Some(f) => t.name == f || t.name.eq_ignore_ascii_case(f) || t.child_session_id == f,
            })
            .collect();
        let snapshots: Vec<SubagentTaskSnapshot> = matched.iter().map(|t| t.snapshot()).collect();
        let running = list.iter().filter(|t| !t.status.is_terminal()).count();
        let single_detail =
            if filter.is_some() && matched.len() == 1 && matched[0].status.is_terminal() {
                let task = matched[0];
                inbox.invalidate_subagent_slots(parent_id, &task.child_session_id, None);
                Some(SingleStatusDetail {
                    snapshot: snapshots[0].clone(),
                    last_output: task.last_output.clone(),
                    last_output_path: task.last_output_path.clone(),
                })
            } else {
                None
            };
        (snapshots, running, single_detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_tracker_truncates_utf8_and_advances_step_offset_monotonically() {
        let tracker = ChildActivityTracker::new(0);
        tracker.set_turn_step(3);
        tracker.set_tool("bash: echo 你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界");
        let (step, tool, _) = tracker.snapshot();
        assert_eq!(step, 3);
        let t = tool.unwrap();
        assert_eq!(t.chars().count(), 64);

        // 续跑新一轮，turn_step 从 1 重算时 current_step 不得倒退
        tracker.advance_step_offset(3);
        tracker.set_turn_step(1);
        assert_eq!(tracker.snapshot().0, 4);
    }

    #[test]
    fn stop_keeps_active_run_id_until_worker_finishes_blocking_early_reactivation() {
        let mut reg = BackgroundTaskRegistry::default();
        let inbox = SessionInbox::new();
        let dispose = CancellationToken::new();

        let (run1, name, running, _) = reg
            .register_running(
                "parent-1",
                Some("worker"),
                "child-1".into(),
                SubagentMode::ReadOnly,
                Some("mock".into()),
                Some("m1".into()),
                "first task".into(),
                dispose.child_token(),
                4,
                100,
                &inbox,
            )
            .unwrap();
        assert_eq!(running, 1);
        assert_eq!(name, "worker");

        let stopped = reg.request_stop("parent-1", "worker", &inbox);
        assert_eq!(stopped.len(), 1);
        assert_eq!(reg.running_count("parent-1"), 0);

        // Worker 尚未退出清除 active_run_id 前，禁止重激活第二条 Worker
        let default_cfg = SubagentConfig::default();
        let err = reg
            .prepare_send(
                "parent-1",
                "worker",
                "second task".into(),
                None,
                None,
                &inbox,
                "second task",
                &default_cfg,
                &dispose,
            )
            .err()
            .expect("must block reactivation while winding down");
        assert!(matches!(err, SubagentError::StoppingInProgress(_)));

        // Worker 正式退出结算后清除 active_run_id，随后 send 可安全分配 run2 重激活
        assert!(
            reg.mark_settled(
                "parent-1",
                "worker",
                "child-1",
                run1,
                SubagentRunStatus::Stopped,
                2,
                Some("stopped".into()),
                None,
                None,
                &inbox,
                None,
            )
            .is_some()
        );

        let res = reg
            .prepare_send(
                "parent-1",
                "worker",
                "resume task".into(),
                None,
                None,
                &inbox,
                "resume task",
                &default_cfg,
                &dispose,
            )
            .unwrap();
        match res {
            SendOutcome::Reactivated {
                run_id,
                prev_provider,
                prev_model,
                ..
            } => {
                assert!(run_id > run1);
                assert_eq!(prev_provider.as_deref(), Some("mock"));
                assert_eq!(prev_model.as_deref(), Some("m1"));
            }
            SendOutcome::SteeredRunning { .. } => panic!("expected reactivation"),
        }
    }

    #[test]
    fn stopping_transition_still_counts_toward_concurrency_limit_until_settled() {
        let mut reg = BackgroundTaskRegistry::default();
        let inbox = SessionInbox::new();
        let dispose = CancellationToken::new();

        // 预先放入一个已结算的休眠任务 worker-idle 供后续 prepare_send 测试
        let (run0, _, _, _) = reg
            .register_running(
                "parent-1",
                Some("worker-idle"),
                "child-0".into(),
                SubagentMode::ReadOnly,
                None,
                None,
                "idle".into(),
                dispose.child_token(),
                1,
                100,
                &inbox,
            )
            .unwrap();
        reg.mark_settled(
            "parent-1",
            "worker-idle",
            "child-0",
            run0,
            SubagentRunStatus::Completed,
            1,
            None,
            None,
            None,
            &inbox,
            None,
        )
        .unwrap();

        let (run1, _, _, _) = reg
            .register_running(
                "parent-1",
                Some("worker-1"),
                "child-1".into(),
                SubagentMode::ReadOnly,
                None,
                None,
                "first".into(),
                dispose.child_token(),
                1,
                100,
                &inbox,
            )
            .unwrap();

        let stopped = reg.request_stop("parent-1", "worker-1", &inbox);
        assert_eq!(stopped.len(), 1);
        assert_eq!(reg.running_count("parent-1"), 0);
        assert_eq!(reg.active_run_count("parent-1"), 1);

        // stop 过渡期 worker-1 仍占槽位，禁止新 spawn 或重激活第二个任务击穿 cap=1
        assert!(matches!(
            reg.preflight_spawn("parent-1", Some("worker-2"), 1),
            Err(SubagentError::ConcurrencyLimitReached(1))
        ));
        let cap1_cfg = SubagentConfig {
            max_concurrent_per_parent: 1,
            ..SubagentConfig::default()
        };
        assert!(matches!(
            reg.prepare_send(
                "parent-1",
                "worker-idle",
                "resume".into(),
                None,
                None,
                &inbox,
                "resume",
                &cap1_cfg,
                &dispose,
            ),
            Err(SubagentError::ConcurrencyLimitReached(1))
        ));

        // 结算清掉 active_run_id 后，槽位真正释放
        reg.mark_settled(
            "parent-1",
            "worker-1",
            "child-1",
            run1,
            SubagentRunStatus::Stopped,
            1,
            None,
            None,
            None,
            &inbox,
            None,
        )
        .unwrap();
        assert_eq!(reg.active_run_count("parent-1"), 0);
        assert!(
            reg.register_running(
                "parent-1",
                Some("worker-2"),
                "child-2".into(),
                SubagentMode::ReadOnly,
                None,
                None,
                "second".into(),
                dispose.child_token(),
                1,
                100,
                &inbox,
            )
            .is_ok()
        );
    }

    #[test]
    fn activity_tracker_emits_single_budget_steer_at_soft_budget() {
        let tracker = ChildActivityTracker::with_soft_budget(0, 4);
        assert!(tracker.take_budget_steer_notice_if_due(3).is_none());
        let notice = tracker
            .take_budget_steer_notice_if_due(4)
            .expect("must fire once at 1.0x soft budget");
        assert!(notice.contains("soft budget: 4"), "{notice}");
        assert!(notice.contains("At 6 steps"), "{notice}");
        assert!(tracker.take_budget_steer_notice_if_due(5).is_none());

        tracker.reset_turn_budget();
        assert!(tracker.take_budget_steer_notice_if_due(4).is_some());
        tracker.enter_forced_wrapup();
        tracker.reset_turn_budget();
        tracker.enter_forced_wrapup();
        assert!(tracker.take_budget_steer_notice_if_due(10).is_none());
    }

    #[test]
    fn terminal_output_inlines_up_to_12k_and_spills_larger_with_5k_preview() {
        let inline_raw = "a".repeat(SUBAGENT_OUTPUT_INLINE_MAX_CHARS);
        let (inline_out, inline_path) =
            format_terminal_output_for_delivery(&inline_raw, "child_inline", 1);
        assert!(inline_path.is_none());
        assert_eq!(inline_out.len(), SUBAGENT_OUTPUT_INLINE_MAX_CHARS);

        let large_line = format!("{}\n", "x".repeat(99));
        let large_raw = large_line.repeat(130); // 13,000 chars > 12,000
        let (spilled_out, spilled_path) =
            format_terminal_output_for_delivery(&large_raw, "child_spill_test", 99);
        let path = spilled_path.expect("must spill to temp file over 12k chars");
        assert!(path.exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), large_raw);
        let preview_part = spilled_out
            .split("\n\n[Output truncated:")
            .next()
            .expect("must include preview header");
        assert!(preview_part.chars().count() <= SUBAGENT_OUTPUT_PREVIEW_CHARS);
        assert!(spilled_out.contains(&path.display().to_string()));
        let _ = std::fs::remove_file(path);
    }
}
