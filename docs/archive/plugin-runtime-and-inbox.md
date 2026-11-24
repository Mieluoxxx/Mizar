# Mizar 外部插件载体与事件推送（SessionInbox）架构设计

> **状态**：设计定稿（Proposed / 尚未落地代码）  
> **适用范围**：Mizar Rust Core（`mizar-agent-loop` / `mizar-session` / `mizar-agent-spine`）、终端宿主（`mizar-tui`）及上层桌面宿主（如未来 GPUI App、`rquickjs` / 双向子进程插件宿主）  
> **关联文档**：当前已装配 crate 清单与服务拓扑见 [plugins.md](../plugins.md)；隔离与编码约束见 [AGENTS.md](../../AGENTS.md)。本文档中的 `TurnInput`、`SessionInbox` 等均为**拟议新增设计**，尚未合入仓库代码。

---

## 一、设计背景与核心问题

随着 Mizar 面向桌面端（如 GPUI App）与外部扩展演进，插件系统面临一个典型的双向需求：

1. **动态注册工具 / Prompt / Skill**：类似 MCP，使模型可以按需发现并调用外部扩展提供的能力。
2. **主动向 Agent 推送事件**：外部世界发生状态变化（文件监听告警、异步构建/CI 完成、Webhook 回调、桌面看板交互、定时触发、用户中途插话）时，插件需要反过来把事件送进 Agent——要么插进当前轮次的下一步纠偏，要么唤醒空闲中的会话，要么静默沉淀为上下文。

围绕这一需求，经常出现两个工程疑问：

- **Rust 没有类似 TS/Python 的源码级动态加载，是否要靠 `.dll` / `.so` / `.dylib` ABI 插件解决？**  
  **不需要，也不应该。** Rust 没有稳定的跨编译版本 ABI；直接走 `extern "C"` ABI 难以传递 `async Future` 与 Tokio 运行时上下文，跨 FFI 边界内存管理脆弱，单个插件 `panic` 会直接拖垮主进程，且跨平台多架构分发代价极高。
- **为什么常规 MCP 做不到「给 Agent 发事件」？**  
  以 Mizar 当前 [McpClient](../../crates/mizar-mcp/src/client.rs) 与 [McpManager](../../crates/mizar-mcp/src/lib.rs) 为例，链路是纯单向拉取：**「模型决定调工具 → 宿主转发 `call_tool` → Server 返回结果」**。模型未发起调用或会话空闲时，外部 Server 没有入口把事件推入运行循环。

**核心判断**：能否同时支撑「注册工具 + 推事件」，关键不在语言是否支持动态 `import`，而在于 **Core 必须先开好「热注册 + 异步收件箱」两扇正交的门**；至于插件以 JS 脚本还是子进程形态存在，交给上层宿主载体解决。

---

## 二、Core 与 App 的职责边界

Mizar 的核心原则是**保持最小、稳固、可独立运行 TUI 的 Rust + Cordis 内核**，绝不把 JS 宿主或桌面工作台耦合进底层。

| 能力维度 | 归属层级 | 具体内容 | 决策理由 |
| :--- | :--- | :--- | :--- |
| **工具 / Prompt / Skill 注册与清理** | **Core** | `ToolRuntime::register`、`SystemPrompt::section`、`SkillRegistry::register` 返回 `Cleanup` | 已就绪；借助 Cordis `effect` 倒序清理防止泄漏 |
| **消息队列、投递语义与步边界消费** | **Core** | `TurnInput` 身份区分、`SessionInbox`（`Steer` / `FollowUp` / `Inject`）、`wakeBudget` / 去重合并、`AgentLoop` 步间 drain | App 处于循环外部，无法切入 `step` 与 `step` 之间的原子缝隙；多前端不能各自维护相互冲突的提示词队列 |
| **闲时拉起执行调度** | **宿主层（TUI / GPUI App）** | 订阅 `SessionInbox` 待唤醒通知，在会话非忙时调用 `AgentLoop::run_with_options` | 保持 `AgentLoop` 自身是被动按 `Run` 调用的纯粹执行器，各前端自行持有流式事件消费与取消句柄 |
| **外部插件运行时与通信桥** | **App / 可选扩展宿主** | `rquickjs` 产品插件宿主、双向子进程协议桥（JSON-RPC / NDJSON over stdio）、Wasm（远期） | 避免将重型动态脚本宿主与平台胶水绑死进最小内核 |
| **Bundle 分发与桌面交互扩展** | **App 层** | `plugin.json` 清单发现、GPUI 工作台/看板面板、Gen-UI 渲染与交互事件绑定 | 纯产品与交互层关注点，与核心推理循环无关 |

> **不可突破的编译边界**（同 [AGENTS.md](../../AGENTS.md)）：`mizar-tui-ui`、`mizar-llm-types`、`mizar-trajectory-core` 禁止引入 `tokio`、`cordis`、`reqwest`、`rquickjs`、`rusqlite`。本文档所有内核新增能力均不破坏该边界。

---

## 三、外部插件运行载体选型

既然 Core 将接口收敛为「JSON Schema 工具注册 + 文本/结构化事件投递」，外部插件不再需要二进制 ABI 链接，可按场景选用上层载体：

| 载体方案 | 定位与适用场景 | 优点 | 缺点与边界 | Mizar 结论 |
| :--- | :--- | :--- | :--- | :--- |
| **宿主内嵌 `rquickjs`** | App 层轻量扩展、桌面工作流脚本、UI 事件监听 | 仓库已引入 `rquickjs` 依赖；零平台编译分发、跨 OS 一致、可通过绑定函数直接接入注册与推送接口 | 与宿主同进程（依赖沙箱/超时隔离）；不适合重型系统 I/O 或原生库生态 | **桌面/产品插件首选**（放 App 层，与内置 `mizar-codemode` 隔离） |
| **双向子进程（stdio / Socket + NDJSON / JSON-RPC）** | 跨语言（Python/Go/Node/Rust）、重型后端能力、高隔离需求 | 天然进程崩溃隔离、语言无关、无 Rust ABI 隐患；同一长连接既可注册/执行工具，又可主动上行推送事件 | 有进程拉起与序列化开销 | **跨语言与重型扩展首选**（能力严格超集于单向 MCP） |
| **Wasm 组件模型（WIT）** | 强沙箱第三方二进制分发 | 跨平台安全沙箱 | 异步 I/O 与宿主机能力桥接偏重 | **远期预留**，首期不引入 |
| **原生 `.dll` / `.so`（C ABI）** | — | 理论上零序列化开销 | 无稳定 Rust ABI、`async` 跨界困难、`panic` 拖垮主进程、跨平台构建矩阵重 | **明确不采用** |

---

## 四、业界参考对照与取舍

结合 `pi` 与 `deepseek-harness`（简称 `dsh`）两套参考实现，明确 Mizar 的借鉴与裁剪边界（注：二者为行为规范参考源，非仓库内已移植代码）：

### 4.1 架构对比

- **`pi`（轻量内存队列 + Session 适配）**：
  - 核心 `agent` 包内置内存态 `steeringQueue`（`steer`）与 `followUpQueue`（`followUp`），循环内每步优先排空 `steer`，内层无更多工具调用时再取 `followUp` 开启下一轮。
  - 外层 `AgentSession` 提供面向扩展的 `sendCustomMessage(..., { deliverAs, triggerTurn })`，用 `role: "custom"` 与人类输入隔离，并在空闲时主动触发新一轮。
- **`deepseek-harness`（正交 Inbox 语义 + 并发护栏）**：
  - 将投递抽象为 `Agent.send(message, target: 'next-turn' | 'next-step', wakeup: boolean)`，收口为 `followup` / `steer` / `inject` 三操作，并在无工具调用准备停轮前派发 `agent/turn-stopping` 允许拦截停轮。
  - 外围 `tool-jobs` 与 `webhook` 实现明确的来源归属与 **`wakeBudget` 忙时降级**：仅当 `owner.status === 'idle'` 且未超出唤醒预算时调用 `followup`，一旦忙碌或耗尽预算则自动降级为 `inject`（只写入下文、不主动唤醒），避免多任务并发完成引发唤醒风暴。
  - 通过 `agent/inbox/spliced` 事件把 Inbox 修改也持久化进会话日志。

### 4.2 Mizar 的取舍

1. **取 `pi` 之简**：首期采用**纯内存态 `SessionInbox`**，暂不引入 `dsh` 式的 `agent/inbox/spliced` 持久化事件溯源，保持低复杂度与零落盘负担。
2. **取 `dsh` 之稳**：在内核落地 **`Steer` / `FollowUp` / `Inject` 三语义 + `wakeBudget` 熔断降级 + `coalesce_key` 合并**，堵住后台插件循环唤醒烧 token 的风险。
3. **复用 Mizar 已有机制**：不另起一套 `custom` 消息协议，直接复用 [mizar-session/src/model.rs](../../crates/mizar-session/src/model.rs) 已定义并被轨迹/UI/压缩全链路支持的 `SessionEvent::ContextNotice { source, text }`。

---

## 五、Mizar 现状诊断（改造锚点）

经核查现有代码，Mizar 的工具热注册已完备，但事件推送链路存在 4 处结构性卡点与 1 处可选增强点：

1. **队列所有权停留在 TUI 前端循环**：`pending_steer: Option<String>` 仅存在于 [crates/mizar-tui/src/runner.rs](../../crates/mizar-tui/src/runner.rs)，`queued_prompts: Vec<String>` 仅存在于 [crates/mizar-tui-ui/src/app.rs](../../crates/mizar-tui-ui/src/app.rs)。未来 GPUI App 与外部插件无法共享该队列。
2. **`SteerPrompt` 为强中断重启，而非步间插话**：当前 `runner.rs` 收到 `AppAction::SteerPrompt` 时直接调用 `cancel.cancel()` 中止在途轮次，等待 `UiEvent::RunFinished` 后再 `SubmitPrompt` 重启新轮，丢失了正在进行中的步间连续性。
3. **轮次入口写死人类身份**：[AgentLoop::run_with_options](../../crates/mizar-agent-loop/src/lib.rs) 仅接收 `input: impl Into<String>`，并在内部固定追加 `SessionEvent::User { text }`。插件或 Webhook 若复用该入口，会被记为人类发言（与真实用户指令混淆）。
4. **现有上下文注入仅支持被动触发**：`mizar-instructions`、`mizar-skills` 与 TUI `@` 引用（见 [mentions.rs](../../crates/mizar-tui/src/mentions.rs)）均挂在 `agent/pre-step` 被动触发，不能在会话空闲时主动唤醒新轮次。
5. **（可选增强）单 MCP Server 热启停入口**：[McpManager](../../crates/mizar-mcp/src/lib.rs) 内部已按 server 维护各自独立的 `ConnectedServer { entry, client, state, cleanups }`，但对外仅暴露批量的 `connect_configured` 与 `close_all`；桌面端若需单独开关某一个 MCP Server，后续可按单 server 粒度暴露增删方法。

---

## 六、拟议架构与详细设计（Proposed）

### 6.1 整体数据流

```text
  ┌─ 上层宿主 / 插件载体（App 层，Core 无感知）──────────────────────────────┐
  │  TUI 输入 │ GPUI 交互 │ rquickjs 插件 │ 双向子进程桥（JSON-RPC/NDJSON） │
  └─────┬──────────────────────────┬───────────────────────────┬────────────┘
        │ ① 注册工具/Prompt        │ ② 推送事件                 │ ③ 空闲监听拉起
        v                          v                           v
┌───────┴──────────────────────────┴───────────────────────────┴────────────┐
│ Core 契约层（mizar-tools / mizar-session / mizar-agent-loop）              │
│                                                                           │
│  ToolRuntime::register -> Cleanup      SessionInbox (内存态，按 session)   │
│  (Direct / Codemode / Deferred)        ├─ Steer    (步间插话/停轮拦截)     │
│                                        ├─ FollowUp (闲时唤醒/忙时排队)     │
│                                        ├─ Inject   (仅沉淀上下文不唤醒)    │
│                                        └─ 护栏: wakeBudget + coalesce_key │
│                                                    │                      │
│  AgentLoop::run_turn_locked                        v                      │
│   ├─ 入口写入 TurnInput (User 或 ContextNotice)                           │
│   ├─ 点位 A (每步调 LLM 前): drain Steer + Inject -> ContextNotice         │
│   └─ 点位 B (无 tool calls 拟退出前): 检查新 Steer -> 有则 continue 下一步 │
└───────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Core 层改造要点

#### 1. `TurnInput`：在轮次入口区分人类指令与事件通知

在 `mizar-agent-loop` 新增轻量输入枚举：

```rust
pub enum TurnInput {
    /// 人类用户直接发送的提示词 -> 写入 SessionEvent::User
    User { text: String },
    /// 插件 / 外部事件唤醒 -> 写入 SessionEvent::ContextNotice
    Notice { source: String, text: String },
}
```

- 为 `String` / `&str` 实现 `From<T> for TurnInput`（映射为 `TurnInput::User`），保持现有 `AgentLoop::run` 与 `run_with_options` 签名向后兼容。
- 当以 `TurnInput::Notice { source, text }` 启动新轮时，[mizar-session](../../crates/mizar-session/src/lib.rs) 的 `derive_messages()` 照常将其投影为模型可见的 `Message::User`，同时回放视图、轨迹观测（`mizar-trajectory`）和压缩摘要都能明确识别事件来源，不与主人输入混淆。

#### 2. `SessionInbox`：收口队列、语义与防爆护栏

在 `mizar-agent-loop`（随 `"agentLoop"` 暴露或拆出薄子模块）维护按 `session_id` 索引的内存收件箱：

- **投递模式 `DeliveryMode`**：
  - **`Inject`**：只进入会话待消费槽位，不主动触发推理；由运行中循环的下一步或下一次外部唤醒顺带写入 `ContextNotice`。
  - **`Steer`**：高优步间纠偏。若目标会话当前在 `active_sessions` 中运行，在下一步开始前或停轮前立即排空写入当前轮（**不取消正在执行的工具**）；若会话处于空闲态，按 `FollowUp` 尝试唤醒（同样受 `wakeBudget` 约束）。
  - **`FollowUp`**：若目标会话忙，排入下一轮队尾等待当前轮正常结束后由宿主继续拉起；若空闲，发出就绪通知由宿主拉起新轮。
- **内置护栏**：
  - **连续自动唤醒预算（`wake_budget`）**：按 `session_id` 统计「无人类介入期间由非人类 `Notice` 触发的自动唤醒次数」。每次人类投递 `TurnInput::User` 时计数器归零；一旦超出 `wake_budget`，后续来自插件的 `FollowUp` / 闲时 `Steer` 自动降级为 `Inject`（保留事件文本供下次阅读，不再触发推理），防止插件循环唤醒死锁。
  - **按键去重合并（`coalesce_key: Option<String>`）**：同一来源高频触发（如持续保存触发的文件检查、构建进度更新）带相同 `coalesce_key` 投递时，原地更新/替换队列中未消费的旧条目，防止上下文被重复通知淹没。
  - **可观测与可撤销**：提供按 `session_id` 查看待发快照（供 TUI/GPUI 展示待发队列计数）及清理接口（如切换会话或用户显式取消时处理）。

#### 3. `AgentLoop` 主循环的双挂载点

在 `crates/mizar-agent-loop/src/lib.rs` 的 `run_turn_locked` 循环中接入两个固定边界（不引入额外并发锁冲突）：

- **点位 A（每步开始、拼装模型输入之前）**：
  - 在 `agent/pre-step` 之前（或紧邻其后、`system_prompt.build()` 与 `session.derive_messages()` 之前），从 `SessionInbox` 排空当前会话待消费的 `Steer` 与 `Inject` 条目，按序追加到 `session`（`TurnInput::User` 写 `User`，`TurnInput::Notice` 写 `ContextNotice`）。
  - 这样第 `N` 步工具执行完成（`additional_contexts` 写入之后）紧接着第 `N+1` 步调模型前，插件刚推入的纠偏消息无需取消整轮即可生效。
- **点位 B（无工具调用、即将结束本轮之前）**：
  - 当前实现在 `if calls.is_empty()` 处直接设置 `TurnEndReason::Completed` 并 `return Ok(text)`。
  - 改造后在真正退出前检查 `SessionInbox` 是否已有新到达的 `Steer`（必要时可先派发轻量收尾检查钩子）；若存在待消费 `Steer`，则跳过退出并 `continue` 进入下一 `step`，否则正常结束本轮。

### 6.3 宿主层对接（TUI 与未来 GPUI App）

1. **唤醒驱动（`WakeDriver`）**：
   - TUI（[runner.rs](../../crates/mizar-tui/src/runner.rs)）与未来 GPUI App 各自运行一个轻量监听循环（或复用共享辅助函数），订阅 `SessionInbox` 的「有待运行 `FollowUp`」通知；当目标会话非忙时出队下一条 `TurnInput` 并拉起 `AgentLoop::run_with_options`。
   - **错误处理约定**：对齐当前 TUI 行为，若上一轮以取消或错误结束，常规人类 `FollowUp` 排队不自动吞掉连发，保留在队列中等待用户处理或手动清空。
2. **显式「强制中断」与「温和 `Steer`」并行保留**：
   - TUI 快捷键若语义明确为「立刻打断当前生成并改发新消息」，仍可走 `cancel.cancel()` + 投递新轮次；插件系统触发的高优纠偏优先走无中断的步间 `Steer`。
3. **GPUI App 外部插件宿主闭环**：
   - **`rquickjs` 宿主**：加载插件包时向 JS 上下文暴露宿主能力：`registerTool(spec, handler)` 底层调用 Core `ToolRuntime::register` 并记录 `Cleanup`；`pushEvent(payload)` 底层调用 Core `SessionInbox::push`。
   - **双向子进程宿主**：扩展一条长连接上的 JSON-RPC / NDJSON 消息类型：子进程下行声明 `tools/register` 与响应 `tools/call`，上行随时发送 `agent/push_event`（指定 `steer` / `followup` / `inject` 与 `coalesce_key`），桥接层收到后直接送入 Core `SessionInbox`。

---

## 七、分阶段实施路线与验证计划

### Phase 1：Core 原语与步间 `Steer`（零 UI 变更）
- **改动范围**：`crates/mizar-agent-loop`（新增 `TurnInput`、`SessionInbox`、`WakeBudget` 状态与点位 A/B 消费）、`crates/mizar-agent-spine`（按需导出类型）。
- **验证标准**：
  - 集成测试：运行中第 1 步工具执行期间投递 `Steer`，断言本轮不被取消、第 2 步模型上下文收到对应 `ContextNotice`；
  - 集成测试：模型未返回工具调用将停轮前若存在 `Steer`，断言拦截停轮并继续下一步；
  - 集成测试：连续自动 `FollowUp` 超过 `wake_budget` 后自动降级为 `Inject` 且不再唤醒，直到收到 `TurnInput::User` 后恢复；
  - 必过全量门禁与最小特性检查：`cargo check -p mizar-agent-spine --no-default-features`、`cargo test --workspace`、`cargo clippy --workspace --all-targets -- -D warnings`。

### Phase 2：收口 TUI 私有队列
- **改动范围**：`crates/mizar-tui/src/runner.rs`、`crates/mizar-tui-ui/src/app.rs`（`mizar-tui-ui` 仍不引入异步/内核依赖，仅通过 `UiEvent` / 动作指令同步待发预览）。
- **验证标准**：复用并补充 `mizar-tui` 无头测试（`tui_headless.rs`），验证现有排队提示、错误保留排队、手动取消与新步间 `Steer` 无回归。

### Phase 3：GPUI / App 外部插件宿主与单 MCP 热启停
- **改动范围**：上层 GPUI App crate（`rquickjs` 产品插件宿主 + 双向子进程桥接 + `WakeDriver`），视需要为 `crates/mizar-mcp` 补充单 server `add_server` / `remove_server` 热启停接口，并同步更新 [plugins.md](../plugins.md)。
- **验证标准**：端到端验证一个子进程（或 JS）插件同时完成「注册工具给模型调用」与「外部触发事件唤醒空闲会话 / 插话运行中会话」，且插件卸载后通过 `Cleanup` 彻底移除工具与回调。
