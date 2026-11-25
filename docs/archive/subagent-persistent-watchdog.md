# Mizar Subagent 常驻分身、零 Token 活性看门狗与无头正交精简设计（v3）

> **状态**：已实现并全量验收通过（v3）  
> **适用范围**：`mizar-session` / `mizar-session-sqlite` / `mizar-session-title` / `mizar-agent-loop` / `mizar-subagent` / `mizar-agent-spine` / `mizar-tui` / `mizar-tui-ui` / [plugins.md](../plugins.md)  
> **v0.3 路径映射说明**：v0.3 工作区拍扁后，文中历史 crate 名 `mizar-session-sqlite` 与 `mizar-session-title` / `mizar-agent-loop` / `mizar-subagent` / `mizar-agent-spine` 已分别收口进 `mizar-session` 与 `mizar-core`（现行 10-Crate 拓扑见 [plugins.md](../plugins.md)）。  
> **前置背景与关联文档**：当前装配拓扑见 [plugins.md](../plugins.md)；编译隔离与仓库规范见 [AGENTS.md](../../AGENTS.md)。

---

## 一、背景与核心定位

Mizar 当前已实现第一版「Fire-and-Notify」单层后台 Subagent（`spawn` / `status` / `stop` + 无头 `mizar exec` 强制 `Sync` 同步降级），但在对照生产诉求及参考实现后，暴露出四类本质问题与结构税：

1. **工具与无头重叠、维护双轨**：无头 `mizar exec` 并无人机陪聊诉求，其并行加速与上下文减负完全由 `codemode`（脚本确定性 `Promise.all` + 过滤裁剪）承担；硬在无头里维护 `SubagentExecutionMode::Sync` 双轨分支（以及未来的收敛循环）属于纯粹的偶然复杂度。
2. **一次性子代理无法中途纠偏或唤醒续聊**：子代理偏航或接近步数上限停下后，主 Agent 无法发消息继续利用其既有上下文，只能从零 `spawn`。
3. **运行期黑盒与无活性看门狗**：主模型没有生物钟；子代理在后台静默卡死（例如测试挂住或工具阻塞）时，主 Agent 既收不到主动提醒，调 `status` 也看不到当前步数、最近工具和静默时间。
4. **底座隐蔽隐患**：
   - `SessionInbox::pop_followup` 单条出队且每个 `FollowUp` 独立加 1 预算，导致并行扇出的 4 个子代理同时完成时会连续唤醒父代理 4 轮、迅速打爆 `DEFAULT_AUTO_WAKE_LIMIT = 5`；
   - `cancel_active_run`（`Esc`）调用 `inbox().clear(...)` 会误删尚未消费的后台子代理回执与看门狗通知；
   - `SessionHeader::is_subagent_child()` 通过 `parent_id.is_some() && fork_seq.is_none()` 隐式推断子代理身份（且 `mizar-session-title` 也直接手写该条件），为未来演进埋下暗雷；
   - `Timeout` 直接丢弃 `run_fut` 会在内存会话留下悬挂 `ToolCall`（后续若通过 `send` 唤醒必触发 LLM 400 协议错误）；
   - `SubagentConfig` 尚未透出至 `settings.json`。

### 架构收敛定位

- **双工具 100% 正交**：
  - `codemode`（`ToolExposure::ModelOnly`）：面向交互与无头 `mizar exec` 的确定性脚本并行编排与数据过滤；
  - `subagent`（`ToolExposure::ModelOnly`）：专用于交互宿主（TUI / Merak Desktop）的长周期异步分身与主对话释放。两者均为 `ModelOnly`，平级隔离、互不套娃。
- **做减法**：无头 `mizar exec` 直接关闭 `subagent`（`spine_cfg.subagent = None`），彻底删除 `SubagentExecutionMode` 枚举、`sync_default()`、`run_sync_subagent` 与 `effective_exec_subagent_mode`。
- **取 AGY Swarm 之魂、不背其壳**：给予分身「常驻休眠 + `send` 唤醒/插话 + 零 Token Rust 活性看门狗主动提醒 + 陪聊分流提示词」，坚决不上网状群聊板、LLM Monitor 与多后端 CoW 文件系统。

---

## 二、总体改造拓扑

```text
1. mizar-session / mizar-session-sqlite / mizar-session-title
   ──► 显式 SessionOrigin（v2 -> v3 自动迁移回填）+ Session::repair_dangling_tool_calls 内存自愈
2. mizar-agent-loop (SessionInbox)
   ──► push_appending（FollowUp/Inject 全分支追加合并，绝不覆写）+ 预算余量查询 + clear_user_items_and_defer_notices
3. mizar-subagent
   ──► 删 Sync、改 ModelOnly、active_run_id/generation 统一代次、零 Token 看门狗、拆分 5 模块
4. mizar-agent-spine
   ──► settings.json 透出 subagent.* 配置并覆盖至 Spine（严格 #[cfg(feature = "subagent")] 门控）
5. mizar-tui (exec / runner) & mizar-tui-ui
   ──► exec 禁用 subagent；TUI 回放与实时横幅过滤 subagent-watchdog:；Esc 保留并延后 Notice 且清积压 wake_rx
6. docs/plugins.md + 回归/全量验收
   ──► 文档同步 + 迁改与新增测试 + 全量 Clippy/边界校验
```

---

## 三、分阶段详细设计

### 阶段 1：显式 `SessionOrigin` 与内存会话自愈（`mizar-session` + `mizar-session-sqlite` + `mizar-session-title`）

1. **`mizar-session/src/model.rs` & `lib.rs`**：
   - 新增公开枚举（并由 `mizar-session` 根导出）：
     ```rust
     #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
     #[serde(rename_all = "snake_case")]
     pub enum SessionOrigin {
         User,
         Subagent,
     }
     ```
     提供 `as_str(&self) -> &'static str` 与 `parse(raw: &str) -> Option<Self>`。
   - `SessionHeader` 与 `SessionCreateOptions` 增加：
     ```rust
     #[serde(default, skip_serializing_if = "Option::is_none")]
     pub origin: Option<SessionOrigin>,
     ```
   - `SessionHeader::new` 默认 `origin: None`。
   - `SessionHeader::is_subagent_child(&self)` 判定规则：
     ```rust
     match self.origin {
         Some(SessionOrigin::Subagent) => true,
         Some(SessionOrigin::User) => false,
         None => self.parent_id.is_some() && self.fork_seq.is_none(),
     }
     ```
   - `SessionStore::create_with` 写入 `header.origin = opts.origin`；`SessionStore::fork` 显式写入 `new_header.origin = Some(SessionOrigin::User)`。
   - **新增内存会话悬挂 `ToolCall` 修复方法 `Session::repair_dangling_tool_calls(&self) -> usize`**：
     - 读取当前 `Session` 内存轨迹中的可见事件（复用与 `repair_dangling_tool_calls(&mut LoadedSession)` 完全一致的可见性与配对扫描规则），找出尚未配对 `ToolResult` 的 `(tool_call_id, name)`；
     - 若非空，逐条通过 `self.append(SessionEvent::ToolResult { tool_call_id, name, content: "[会话恢复：上次运行意外中断或超时，工具结果未知 (TOOL_OUTCOME_UNKNOWN)]".to_string(), is_error: true })` 追加合成失败结果（自动更新内存事件与触发异步持久化队列）；返回修复条数（幂等：无悬挂返回 `0`）。
2. **`mizar-session-sqlite/src/schema.rs` & `writer.rs`**：
   - `CURRENT_SCHEMA_VERSION` 从 `2` 升为 `3`。
   - `sessions` 建表增加 `origin TEXT` 列；在 `locked_version < 3` 时执行幂等迁移：
     1. 若 `!has_column(conn, "sessions", "origin")?`，执行 `ALTER TABLE sessions ADD COLUMN origin TEXT;`；
     2. 回填历史记录：`UPDATE sessions SET origin = CASE WHEN parent_id IS NOT NULL AND fork_seq IS NULL THEN 'subagent' ELSE 'user' END WHERE origin IS NULL;`；
     3. 更新 `PRAGMA user_version = 3;`。
   - 更新 `schema.rs` 所有 `SELECT` / `INSERT` / `row_to_header` 以及 `writer.rs` 的 `UPSERT`（使用 `origin = COALESCE(sessions.origin, excluded.origin)` 防乱序写入丢字段）。
   - `exclude_subagents` SQL WHERE 条件升级为：
     ```sql
     (origin = 'user' OR (origin IS NULL AND (parent_id IS NULL OR fork_seq IS NOT NULL)))
     ```
3. **`mizar-session-title/src/service.rs:164`**：
   - 将 `if header.parent_id.is_some() && header.fork_seq.is_none()` 统一改为 `if header.is_subagent_child()`，消除双重推断分裂。

---

### 阶段 2：`SessionInbox` 批量追加、唤醒余量与 `Esc` 保护（`mizar-agent-loop`）

在 [crates/mizar-agent-loop/src/inbox.rs](file:///Users/moguw/workspace/dsh-space/mizar/crates/mizar-agent-loop/src/inbox.rs) 扩展以下纯信箱原语（不引入具体插件业务逻辑）：

1. **预算查询方法**：
   - `remaining_auto_wakes(&self, session_id: &str) -> usize`：返回 `mailbox.auto_wake_limit.saturating_sub(mailbox.auto_wakes_since_human)`（未建邮箱时返回 `DEFAULT_AUTO_WAKE_LIMIT`）。
   - `has_pending_followup_key(&self, session_id: &str, key: &str) -> bool`：判断是否存在 `item.mode == DeliveryMode::FollowUp && item.coalesce_key.as_deref() == Some(key)` 的待消费条目。
2. **`push_appending(session_id, mode, input, batch_key, batch_source) -> InboxPushResult`**（修复降级覆写漏洞）：
   - **第一步：计算 `effective_mode` 与 `degraded`（绝不回退走覆写的 `push_coalesced`）**：
     - 若 `is_human` 为真：清零 `auto_wakes_since_human`；
     - 按传入 `mode` 与 `is_active` 初步归类：`Inject` 得 `Inject`；`Steer` 且 `is_active` 得 `Steer`；其余唤醒诉求（`FollowUp` 或空闲 `Steer`）检查：
       - 若 `is_human` 或邮箱中已存在 `item.mode == DeliveryMode::FollowUp && item.coalesce_key == Some(batch_key)` -> `(DeliveryMode::FollowUp, false)`（同波次合并，不增加 `auto_wakes_since_human`）；
       - 否则校验 `auto_wakes_since_human + 1 > auto_wake_limit`：超限得 `(DeliveryMode::Inject, true)`，未超限递增 `auto_wakes_since_human` 得 `(DeliveryMode::FollowUp, false)`。
   - **第二步：按 `(effective_mode, batch_key)` 拼接追加或新建条目**：
     - 查找邮箱中 `item.mode == effective_mode && item.coalesce_key.as_deref() == Some(&batch_key)` 的现有条目（无论是 `FollowUp`、`Inject` 还是 `Steer`）：
       - **命中时**：将新 `input` 正文以 `"\n\n---\n\n"` 拼接到现有条目末尾，合并 `images`；若原 `TurnInput::Notice { source, .. }` 与新 `source` 不一致则更新为 `batch_source`（如 `"subagent:batch"` / `"subagent-watchdog:batch"`），**绝不覆盖旧正文**；
       - **未命中时**：插入新条目，携带 `coalesce_key = Some(batch_key)`。
   - **第三步**：仅当 `effective_mode == DeliveryMode::FollowUp` 时向 `wake_tx` 发送 `session_id`。
3. **`clear_user_items_and_defer_notices(&self, session_id: &str) -> usize`**：
   - 用于交互模式按 `Esc` 中断主轮次：
     - 移除所有 `item.input.is_user()` 的待发条目；
     - 将邮箱中此刻尚未消费的 `DeliveryMode::FollowUp` / `DeliveryMode::Steer` 非人类 `Notice` 统一转为 `DeliveryMode::Inject`（保留全部正文与图片，等用户下次发言同轮随 `drain_step_inputs` 注入上下文，既**绝对不丢后台子代理已落箱结果**，又避免 `Esc` 刚按下瞬间自动把新回合顶起来）；
     - 返回被清理或降级的条目影响数，而 `Esc` **之后**新结算的后台子代理仍可按正常 `FollowUp` 唤醒父会话。原有 `clear()` 保持不变，专供 `/new`、`/resume`、`/tree` 切会话全量清空。

---

### 阶段 3：`mizar-subagent` 升级、原子并发控制与模块化拆分

遵守 `AGENTS.md`（不用 `mod.rs`），将 `crates/mizar-subagent/src/` 拆分为 5 个单职责文件；同时在 `crates/mizar-subagent/Cargo.toml` 为 `tokio` 显式启用 `features = ["macros", "time"]` 保证单 crate 可独立通过 `--lib` 检查。

| 文件 | 职责 |
| :--- | :--- |
| `src/lib.rs` | `SubagentConfig`、错误类型、`SubagentRuntime`、`Service` 生命周期挂载（含事件监听与静态系统提示词） |
| `src/tasks.rs` | `ChildActivityTracker`、`ManagedTask`、`BackgroundTaskRegistry`（`active_run_id` 分配与校验、原子 `prepare_send` / `rollback_reactivation` / `mark_settled` / `record_watchdog_alert`） |
| `src/guardrails.rs` | `compute_allowed_tools`、`tools/pre-execute(-100)` 深度 1 拦截、绑定 `run_id` 的 `ChildSessionGuard` |
| `src/tool.rs` | `build_subagent_tool_definition`（`ToolExposure::ModelOnly`）及 4 动作分发：`spawn` / `send` / `status` / `stop` |
| `src/worker.rs` | 后台 worker 续跑循环、内嵌零 Token 看门狗、超时宽限与会话自愈、结算同锁判定与批量回执投递 |

#### 3.1 `SubagentConfig` 与新增 `SubagentError`（`src/lib.rs`）
- 删除 `SubagentExecutionMode`、`sync_default()`；保留单一后台异步引擎。
- 新增公开常量与字段：
  - `DEFAULT_SUBAGENT_STALL_TIMEOUT_SECS: u64 = 45`（`0` 表示关闭活性看门狗）
  - `DEFAULT_SUBAGENT_MAX_WATCHDOG_ALERTS: usize = 2`
  - `SubagentConfig`：`default_max_steps = 20`, `max_steps_cap = 40`, `timeout_secs = 180`, `max_concurrent_per_parent = 4`, `stall_timeout_secs = 45`, `max_watchdog_alerts = 2`。
  - 新增 `effective_timeout_secs(&self) -> u64 { self.timeout_secs.max(1) }`，防止 `settings.json` 误配 `0` 导致刚启动就超时。
- **新增 `SubagentError` 变体**：
  - `MissingParentSession`（`"缺少父会话上下文，无法启动后台子代理"`）
  - `MissingSendTarget`（`"send 操作缺少目标 name 参数（必须为具体子代理名称，不可为 \"all\"）"`）
  - `SubagentNotFound(String)`（`"未找到名为 `{0}` 的子代理；请先使用 action=\"spawn\" 启动"`）
  - `StoppingInProgress(String)`（`"子代理 `{0}` 正在停止收尾中，请稍后再试"`）
  - `ModeChangeNotAllowed`（`"不允许修改已有子代理 `{0}` 的权限模式（原 mode={1}，请求 mode={2}）"`）
  - `ModelChangeWhileRunning(String)`（`"子代理 `{0}` 正在运行中，无法在运行期间热切 model；请等待其休眠后 send 或 stop 后重派"`）
  - `SessionUnavailable(String)`（`"无法加载子代理 `{0}` 对应的会话 `{1}`"`）

#### 3.2 统一单向锁序 & `active_run_id` / `generation` 唯一代次防越权（`src/guardrails.rs` & `src/tasks.rs`）
- **严格单向锁序**（全 crate 唯一锁获取顺序，严禁反向加锁，严禁跨 `.await` 或 `ctx.emit` 持锁）：
  `state_emit_lock -> tasks -> active_children -> activity_trackers -> inbox.state`
- **统一单调代次 `active_run_id: Option<u64>`（彻底封死双 Worker 并发与反序 `generation` 越权）**：
  - `BackgroundTaskRegistry` 持有 `next_run_id: u64`（严格在 `tasks` 锁内单调递增分配）；`ManagedTask` 新增 `active_run_id: Option<u64>`。
  - `register_running`（`spawn`）与 `prepare_send`（`Settled -> Running` 抢占）均在 `tasks` 锁内分配新 `run_id` 并写入 `entry.active_run_id = Some(run_id)`，返回给调用方。
  - `request_stop` / `cancel_parent` / `cancel_all` 保持同步设 `stop_requested = true`、`status = Stopped`、`cancel.cancel()`（保留 `running_count == 0` 语义），但**绝不清除 `active_run_id`**；`active_run_id` **仅允许**由持有匹配 `expected_run_id` 的 `rollback_reactivation(..., expected_run_id)` 或 Worker 退出时的 `mark_settled(..., expected_run_id)` 在确认 `entry.active_run_id == Some(expected_run_id)` 后清为 `None`（不匹配则直接忽略过时结算）。`record_watchdog_alert` 同样校验 `entry.active_run_id == Some(expected_run_id)`。
- **`ChildSessionGuard` 直接绑定 `run_id`（`guardrails.rs`）**：
  - `ChildSessionGuard::register(..., generation: u64, ...)` **不再在锁外自增 generation**，而是直接接收 `register_running` / `prepare_send` 在 `tasks` 锁内预分配的 `run_id` 作为 `generation`，向 `active_children: Mutex<HashMap<String, (u64, SubagentMode)>>` 与 `activity_trackers: Mutex<HashMap<String, (u64, Arc<ChildActivityTracker>)>>` 写入带该代次的条目，并设置 `tools.set_active_tools`。
  - `ChildSessionGuard::Drop` 时仅当 map 中当前条目的 `generation == self.generation` 才执行移除和 `tools.reset_session_activations`；配合 `active_run_id` 单实例互斥，同一 `child_session_id` 在任一时刻绝不可能存在两个未退出 Worker 或反序重叠守卫。
- **`is_subagent_child_session`**（`guardrails.rs`）：直接调用 `loaded.header.is_subagent_child()`。

#### 3.3 `ChildActivityTracker` 与步数语义（`src/tasks.rs`）
- **明确的步数语义**：
  - `max_steps`（`effective_max_steps()`）为**单次唤醒轮次（per-turn）步数上限**；
  - `ManagedTask.steps` 记录累计总步数（各唤醒轮次之和）；`ManagedTask.last_turn_steps` 记录最近一次唤醒轮次所走步数（`MaxSteps` 提示按 `{last_turn_steps}/{max_steps}` 展示，绝不出现 `35/20 步`）。
  - `ChildActivityTracker`：
    - `started_at: Instant`
    - `last_activity_ms: AtomicU64`（初始 `0`，`touch()` 使用 `Ordering::Relaxed` 写入 `started_at.elapsed().as_millis() as u64`）
    - `step_offset: AtomicUsize`（单轮续跑切换时调用 `advance_step_offset(delta)` 推进基准偏移，保证 `current_step = step_offset + turn_step` 单调递增绝不倒退归 1）
    - `current_step: AtomicUsize`（由 `set_turn_step(ev.step)` 写入 `step_offset + ev.step`）
    - `last_tool: Mutex<Option<String>>`（`tool-call`/`tool-update` 更新时使用 `.chars().take(64).collect()` 按 UTF-8 字符安全截断）
    - `last_activity_instant(&self) -> Instant`：使用 `self.started_at.checked_add(Duration::from_millis(ms)).unwrap_or(self.started_at)` 防极端溢出。
- **Registry 原子原语（`src/tasks.rs`）**：
  - `ManagedTask` 完整保存：`name`, `parent_session_id`, `child_session_id`, `mode`, `provider: Option<String>`, `model: Option<String>`, `prompt_preview`, `status`, `steps`, `last_turn_steps`, `last_tool: Option<String>`, `summary`, `started_at`, `finished_at`, `cancel`, `stop_requested`, `active_run_id: Option<u64>`, `watchdog_alerts`, `activity: Arc<ChildActivityTracker>`。
  - `MAX_SETTLED_PER_PARENT = 32` 淘汰策略改为：当满足 `t.status.is_terminal() && t.active_run_id.is_none()` 的条目超过 32 条时，精准删除其中 **`finished_at` 最早（最久远结束）** 的条目（绝不淘汰仍在停止收尾中的 `StoppingInProgress` 条目；并在重激活时将条目移到对应列表末尾），避免刚被 `send` 唤醒并再次休眠的分身误遭头部淘汰。
  - `prepare_send(&mut self, parent_id, name, prompt_preview, requested_mode, requested_model, inbox: &SessionInbox, prompt: &str, max_concurrent, dispose_token)`（**单一 `tasks` 锁内完成**：`tasks -> inbox.state`）：
    1. 查找匹配 `name` 的条目：找不到返回 `Err(SubagentError::SubagentNotFound)`；
    2. 校验权限：若 `requested_mode` 为 `Some(req)` 且 `req != entry.mode`，立刻返回 `Err(SubagentError::ModeChangeNotAllowed)`；
    3. 若 `entry.active_run_id.is_some()`：
       - 若 `entry.status == SubagentRunStatus::Running && !entry.stop_requested`：
         - 若 `requested_model.is_some()`，直接返回 `Err(SubagentError::ModelChangeWhileRunning)`；
         - 在保持 `tasks` 锁的同时立即调用 `inbox.push(&entry.child_session_id, DeliveryMode::Steer, TurnInput::user(prompt.to_string()))`，返回 `SendOutcome::SteeredRunning`（完全消除 `Running` 查询与 `worker` 结算退出的 TOCTOU 窗口！）；
       - 否则（`stop_requested == true` 或正在停止收尾/启动过渡期）：**禁止重激活启动第二条 Worker**，直接返回 `Err(SubagentError::StoppingInProgress)`；
    4. 仅当 `entry.active_run_id.is_none()` 且 `entry.status.is_terminal()` 时允许抢占重激活：
       - 检验并发上限 `running < max_concurrent.max(1)`；
       - 分配新 `run_id = next_run_id`；先保存 `SettledRollback` 快照（原 `status`, `prompt_preview`, `summary`, `finished_at`, `started_at`, `watchdog_alerts`, `activity`, `cancel`, `provider`, `model`）；
       - 原地原子抢占迁为 `Running`：设 `active_run_id = Some(run_id)`、重置 `cancel = dispose_token.child_token()`、`stop_requested = false`、`started_at = Instant::now()`、`finished_at = None`、`watchdog_alerts = 0`、新建 `activity`（预设 `step_offset = entry.steps`）、更新 `prompt_preview` 并移至末尾；
       - 返回 `SendOutcome::Reactivated { run_id, child_session_id, mode, prev_provider, prev_model, activity, cancel, running_count, rollback }`。
  - `rollback_reactivation(&mut self, parent_id, name, child_session_id, expected_run_id, rollback)`：仅当 `entry.active_run_id == Some(expected_run_id)` 匹配时清 `entry.active_run_id = None`；若过渡期已收到 `stop`（`entry.stop_requested == true`），保持 `entry.status = SubagentRunStatus::Stopped` 与 `finished_at`，否则完整恢复 `SettledRollback` 原终态字段，绝不留脏状态或被并发覆盖。

#### 3.4 窄工具 4 动作执行流程（`src/tool.rs`）
- `build_subagent_tool_definition`：
  - 暴露级别为 **`ToolExposure::ModelOnly`**，`ToolExecutionMode::Parallel`；
  - `action` 枚举为 `["spawn", "send", "status", "stop"]`（默认 `"spawn"`）。
- `spawn` 流程：
  - 强制要求存在 `parent_session_id`（缺失返回 `SubagentError::MissingParentSession`）；
  - `sessions.create_with` 传入 `origin: Some(SessionOrigin::Subagent)`；
  - 在 `tasks` 锁内 `register_running` 成功拿到独占名、`run_id` 与 `activity` 后，**仅由成功注册方**调用 `ChildSessionGuard::register(..., run_id, ...)` 并投递 `spawn_background_worker(..., run_id, ...)`（若分配 `run_id` 前已发现取消则清理）。
- `send` 流程（彻底消除并行 `send(Settled)` 与 `stop` 交叉越权竞态）：
  1. 校验必填 `name`（合规自定义名、禁止 `"all"`）与非空 `prompt`、解析可选 `mode` / `model`；
  2. 在 `tasks` 锁内原子调用 `prepare_send(...)`（同时校验 `active_run_id` 状态、`StoppingInProgress`、并发上限、`mode` 锁定、以及 `Running` 分支直接在锁内推入 `Steer`）；
  3. 若结果为 `SteeredRunning` -> 立刻返回 `Sent steering update to running subagent "{name}"`；
  4. 若结果为 `Reactivated { run_id, ... }`：
     - 首先检查 `child_cancel.is_cancelled() || call_ctx.cancel.is_cancelled() || self.dispose_token.is_cancelled()`，命中则立刻调 `rollback_reactivation(..., run_id, rollback)` 并返回取消错误；
     - 调用 `sessions.get(&child_session_id)` 取回原会话（若 `None` 则调 `rollback_reactivation(..., run_id, rollback)` 并返回 `SubagentError::SessionUnavailable`）；
     - 幂等修复潜在悬挂工具调用：`child_session.repair_dangling_tool_calls()`；
     - 计算当前父会话可用工具白名单 `compute_allowed_tools`（按锁定原 `mode`）、挂上绑定 **`run_id`** 的 `ChildSessionGuard::register(..., run_id, ...)` 与 `tools.set_active_tools`（挂载后再次微检 `child_cancel.is_cancelled()`，若刚好被 `stop` 取消则 drop guard + `rollback_reactivation` 返回取消，由于 `active_run_id` 在退出 `rollback` 前一直占位，绝无第二条 worker 可抢跑）；
     - 计算模型路由（传了 `model` 参数则重解析否则沿用 `prev_provider`/`prev_model`，并更新至任务路由记录）、广播 `SubagentStateEvent`（`status: Running`）、启动 `spawn_background_worker(..., run_id, ...)` 继续在该 `child_session` 跑下一轮！
- `status`：格式化输出实时 `step`、`tool`、`elapsed`、`idle`、`watchdog` 与终态摘要。

#### 3.5 后台 Worker、零 Token 看门狗与无竞态续跑（`src/worker.rs`）
- 外层续跑循环（持有 `run_id` 与唯一 `child_guard`，直至确认彻底结算才释放）：
  1. 启动或续跑前保存当轮 `prompt_for_turn`，构造当轮 `hard_deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs)`，重置当轮 `alerts_sent = 0`、`last_alert_at = None`。
  2. 内层看门狗 + 执轮循环（`tokio::pin!(run_fut)`）：
     - 计算 `watchdog_fut`：仅当 `stall_timeout_secs > 0 && alerts_sent < max_watchdog_alerts` 时，`wake_at = max(tracker.last_activity_instant(), last_alert_at) + stall_timeout`，使用 `tokio::time::sleep_until(wake_at.into())`；否则使用 `std::future::pending::<()>()`。
     - `tokio::select! { biased; }` 严格优先级顺序：
       1. `_ = rt.dispose_token.cancelled()` -> `child_cancel.cancel()`，清空 `inbox.clear(&child_id)`，释放 `child_guard`，调 `child_session.repair_dangling_tool_calls()`，加锁 `tasks` 调用 `mark_settled(..., run_id, ...)` 并返回；
       2. `res = &mut run_fut` -> 跳出内层循环进入结算/续跑判定（**优先级高于 `hard_deadline` 与 `watchdog_fut`**，防同 tick 误报超时/告警！）；
       3. `_ = tokio::time::sleep_until(hard_deadline)` -> 执行优雅超时宽限：`child_cancel.cancel()`，并在 `tokio::time::timeout(Duration::from_millis(200), &mut run_fut)` 等最多 200ms 让子代理写盘收敛；若 200ms 后仍未退出再 Drop `run_fut`，标记为 `TimedOut`；
       4. `_ = watchdog_fut` -> 复核 `tracker.idle_duration() >= stall_timeout` 且距上次告警已满 `stall_timeout`（且未完成未取消）：
          - 递增 `alerts_sent`，记录 `last_alert_at = Some(Instant::now())`，调用 `tasks.record_watchdog_alert(..., run_id)`；
          - **保留最后 1 格自动唤醒预算给结算**：先检查 `inbox.has_pending_followup_key(&parent_id, "subagent:watchdog") || inbox.remaining_auto_wakes(&parent_id) >= 2` 决定使用 `DeliveryMode::FollowUp`（可合并唤醒）还是保底 `DeliveryMode::Inject`（静默附着不挤占最后一个唤醒名额）；
          - 统一使用 `inbox.push_appending(&parent_id, wd_mode, TurnInput::notice(format!("subagent-watchdog:{assigned_name}"), watchdog_text), "subagent:watchdog", "subagent-watchdog:batch")` 投递（绝不覆盖彼此）；
          - 广播 `SubagentStateEvent`（`status: Running`，附简短中文 UI 提示）。
  3. 离开内层循环后的收尾与续跑原子判定：
     - **正常结束条件**（`TurnEndReason::Completed | MaxSteps | Truncated` 且 `!child_cancel.is_cancelled()` 且 `!rt.dispose_token.is_cancelled()`）：
       - 加锁 `tasks`：先检测 `stop_requested`；未请求停止则在同一把 `tasks` 锁内调用 `agent_loop.inbox().pop_followup(&child_id)`：
         - **命中挂起 `send` 指令** -> 在同锁内累加本轮步数到 `entry.steps`、`entry.last_turn_steps = report.steps`，推进 `tracker.advance_step_offset(report.steps)`，重置状态保持 `Running` 并更新 `prompt_preview`，释放锁进入下一轮循环直接续跑（`ChildSessionGuard` 全程未释放，零缝隙！）；
         - **无挂起指令（或 `stop_requested`）** -> 若 `stop_requested` 先清 `inbox.clear(&child_id)`；在同锁内先将 `entry.status` 置为终态并暂保 `active_run_id = Some(run_id)`（阻断过渡期并发重入），释放 `tasks` 锁并 `drop(child_guard)` 彻底卸载旧守卫与 `tools` 策略后，再加锁 `tasks` 调用 `finish_active_run(..., run_id)`（校验匹配后将 `active_run_id` 清为 `None` 并执行超 32 条淘汰），实现真正的零纳秒竞争窗口。
     - **异常 / 取消 / 超时分支**（`TimedOut` / `Cancelled` / `Error` / `Err(AgentError)`）：
       - 清空临界窗口可能残留的 `agent_loop.inbox().clear(&child_id)`；
       - 释放 `child_guard` 后调用 `child_session.repair_dangling_tool_calls()` 补齐可能存在的悬挂 `ToolCall`（关键防线：确保未来 `send(Settled)` 唤醒 `TimedOut`/`Error` 子会话绝对不会触发 LLM 400！）；
       - 加锁 `tasks` 调用 `mark_settled(..., run_id, ...)`（清 `active_run_id = None`）。
  4. 投递最终父会话回执：
     - 使用 `inbox.push_appending(&parent_id, DeliveryMode::FollowUp, TurnInput::notice(format!("subagent:{assigned_name}"), inbox_text), "subagent:settled", "subagent:batch")`（即使在 `auto_wake_limit == 0` 下降级为 `Inject` 也只拼接绝不覆盖）；
     - 以 `state_emit_lock -> tasks` 锁序读取实时 `running_count` 并发出终态 `SubagentStateEvent`（附 `degraded` 提示逻辑）。

#### 3.6 主 Agent 心智提示词（`src/lib.rs`）
更新 `subagent.prompt-guidance` 静态英文提示词：
- 1~3 步轻活顺手完成；重活、高噪音脏上下文、多路独立排查优先 `spawn` 后台分身并**立刻回复用户交还对话权**；
- 支持同轮并行 `spawn`；用 `send` 对运行中分身插话纠偏或唤醒已休眠分身接着干；
- 收到 `subagent-watchdog:*` 时根据快照判断继续等（简短同步用户）、`send` 纠偏或 `stop` 止损；严禁 `sleep` 或轮询。

---

### 阶段 4：`settings.json` 透出配置（`mizar-agent-spine`）

1. **`crates/mizar-agent-spine/src/config/settings.rs`**：新增 `SubagentSettings`（`max_steps`, `max_steps_cap`, `timeout_secs`, `max_concurrent`, `stall_timeout_secs`, `max_watchdog_alerts`），接入 `SettingsConfig` 与配置合并。
2. **`crates/mizar-agent-spine/src/config.rs`**：`apply_to_spine` 与相关单元测试严格配 `#[cfg(feature = "subagent")]` 门控。
3. **`crates/mizar-agent-spine/src/lib.rs`**：移除 `SubagentExecutionMode` re-export。

---

### 阶段 5：无头关闭 + TUI 适配（`mizar-tui` & `mizar-tui-ui`）

1. **`crates/mizar-tui/src/exec.rs`**：移除 `effective_exec_subagent_mode` 与旧单测；`run_exec_cli` 中直接 `spine_cfg.subagent = None;`（通过小辅助函数方便单测验证）。
2. **`crates/mizar-tui/src/runner/actions.rs` & `runner.rs`**：
   - `drain_next_inbox_followup` 放行静默来源条件扩展为 `!(source.starts_with("subagent:") || source.starts_with("subagent-watchdog:"))`；
   - `cancel_active_run`（`Esc`）将 `inbox().clear(...)` 替换为 `inbox().clear_user_items_and_defer_notices(&active_session.id)`；并在 `runner.rs` 处理 `AppAction::CancelRun` 时用兼容 `TryRecvError::Lagged` 的循环排空 `wake_rx` 积压广播（`loop { match rx.try_recv() { Ok(_) | Err(TryRecvError::Lagged(_)) => continue, Err(_) => break } }`），确保 `Esc` 后瞬间安静停住，不丢失已落箱结果，且后续新完成分身仍正常唤醒。
3. **`crates/mizar-tui-ui/src/app.rs:1166`**：
   - `summarize_context_notice_for_replay` 同时匹配 `source.starts_with("subagent:") || source.starts_with("subagent-watchdog:")`，避免 `/resume` 回放长看门狗告警刷屏。

---

## 四、关键不变量总结

1. **依赖边界**：四大纯核心 crate 与 `mizar-client` 零违规依赖（`AGENTS.md` 边界校验必须 100% 通过）。
2. **单向锁序**：`state_emit_lock -> tasks -> active_children -> activity_trackers -> inbox.state`，严禁跨 `.await` 或 `ctx.emit` 持锁。
3. **沙箱与权限不变量**：
   - `send(Settled)` 仅在 `active_run_id.is_none()` 时在 `tasks` 锁内分配唯一 `run_id` 抢占独占权，`request_stop` 不清 `active_run_id`，停止收尾中调用 `send` 统一拒绝并返回 `StoppingInProgress`，配合绑定 `run_id` 的 `ChildSessionGuard` 彻底消除双 Worker 与反序 `generation` 越权；
   - `send` 锁定原 `mode` 禁止提权，运行中禁止换 `model`。
4. **会话协议不变量**：
   - `TimedOut` 先 200ms 宽限再 Drop + 内存 `Session::repair_dangling_tool_calls()` 自愈兜底，保证任何后续 `send(Settled)` 零 400 错误。
5. **明确克制的边界**：不做网状群聊板、LLM Monitor、本期不做 `git worktree`。

---

## 五、测试与验收矩阵

### 1. 分包改造与新增测试
- `mizar-session`（`tests/session_unit.rs`）：`SessionOrigin` 判定、`fork` 设 `User`、`Session::repair_dangling_tool_calls` 内存补齐与幂等验证。
- `mizar-session-sqlite`（`tests/sqlite_persistence.rs`）：更新 line 572 `ver == 3`、v2 -> v3 自动迁移回填、显式 `origin='subagent'` 且非空 `fork_seq` 被 `exclude_subagents` 排除验证。
- `mizar-agent-loop`（`src/inbox.rs` 单测）：
  - 6 次 `push_appending` 在 `limit=2` 下只扣 1 格预算且合并为 `subagent:batch`；
  - `limit=0` 下多次 `push_appending` 降级 `Inject` 且**完整拼接保留全部正文绝不覆写**；
  - `clear_user_items_and_defer_notices` 只删用户项、将已到 `Notice` `FollowUp` 降 `Inject` 保留正文；
  - `remaining_auto_wakes` 与 `has_pending_followup_key` 验证。
- `mizar-subagent`（`tests/subagent_integration.rs`）：
  - 迁改既有 8 项 `sync_default` 测试 + 更新 `background_spawn_returns_immediately_delivers_followup_and_emits_state_events` 适配 `subagent:batch`；
  - `ToolExposure::ModelOnly` 验证；
  - `send(Settled)` 唤醒复用 `child_session_id` 与前轮记忆、**并发两次 `send(Settled)` 绝不破坏 `read_only` 护栏**、拒绝 `read_only -> edit` 提权、拒绝超 `max_concurrent`；
  - `stop_while_winding_down_blocks_send_with_stopping_in_progress_then_allows_resume`：子代理在阻塞屏障中收到 `stop` 尚未退出前立刻 `send`，断言返回 `StoppingInProgress` 且不破坏原 Worker，等原 Worker 收尾释放 `active_run_id` 后再 `send` 可正常唤醒续跑；
  - `send(Running)` 步间 `Steer` 与结束临界窗口不丢指令（`ModelChangeWhileRunning` 拒绝运行中改模型）；
  - `TimedOut` 后自动修复悬挂 `ToolCall` 且随后 `send(Settled)` 合法续跑；
  - 零 token 看门狗（`stall_timeout_secs = 1`）：静默告警投递且保留最后 1 格 `auto_wake_limit` 给结算回执，持续活动分身不误报；
  - `status` 运行中实时包含 `step=`、`tool=`、`idle=`。
- `mizar-agent-spine`：`SubagentSettings` JSON 解析、合并、`apply_to_spine` 及 `--no-default-features` 编译。
- `mizar-tui` & `mizar-tui-ui`：`exec` 关闭 `subagent`；`Esc` 清用户输入保留 `Notice`；回放摘要匹配 `subagent-watchdog:`。

### 2. 全量 CI 与边界命令
- `cargo fmt --all -- --check`
- `cargo check --locked -p mizar-agent-spine --no-default-features --all-targets`
- `cargo clippy --locked -p mizar-agent-spine --no-default-features --features subagent --all-targets -- -D warnings`
- `cargo clippy --locked --workspace --all-targets -- -D warnings`
- `cargo test --locked --workspace --no-fail-fast`
- 架构边界 `rg` 校验。

---

## 六、独立审查与复审结论（PASS）

- **初审 / 二审（`Subagent Design Reviewer`）**：先后定位出 `send(Settled)` 并发越权、`send(Running)` 结算 TOCTOU、`TimedOut` Drop 残留悬挂 `ToolCall`、`push_appending` 降级 `Inject` 覆写、看门狗挤爆唤醒预算、`stop -> send(Settled)` 双 Worker 与反序 `generation` 隐患等关键问题，均已在 v2/v3 设计中逐项闭环。
- **最终复审结论**：`VERDICT: PASS`（确认 `active_run_id` 与 `generation` 唯一代次绑定、内存会话自愈、全分支 `push_appending`、`Esc` 保护与 `#[cfg(feature = "subagent")]` 门控均无阻塞缺陷；并按复审低危建议在正常结算路径采用「置终态暂保 `active_run_id` -> 放锁 `drop(child_guard)` -> 清 `active_run_id = None`」彻底消灭 `tools.reset_session_activations` 纳秒间隙）。
