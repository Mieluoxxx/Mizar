# Mizar 极简 Hooks、缓存续命与单层 Subagent 架构设计

> **状态**：设计定稿（Proposed / 尚未落地代码）  
> **适用范围**：Mizar Rust Core（`cordis` / `mizar-agent-loop` / `mizar-llm` / `mizar-llm-router` / `mizar-llm-messages` / `mizar-session` / `mizar-tools` / `mizar-agent-spine`）、跨端宿主层（`mizar-client` / `mizar-tui` / 未来桌面端 Merak）及拟议可选脊柱模块（`mizar-cache-warming` / `mizar-subagent`）  
> **关联文档**：当前已装配 Crate 拓扑与服务清单见 [plugins.md](../plugins.md)；`SessionInbox` 与外部插件载体边界见 [plugin-runtime-and-inbox.md](plugin-runtime-and-inbox.md)；局部性与内存优化设计见 [locality-and-memory-optimization.md](locality-and-memory-optimization.md)；编译隔离与仓库规范见 [AGENTS.md](../../AGENTS.md)。本文档中的新增钩子、字段与可选模块均为**拟议设计**，尚未合入仓库代码。

---

## 一、设计哲学与红线（大道至简 + 经典软件工程映射）

Mizar 的演进目标不是堆砌扩展机制或重造一套分布式运行框架，而是在「本地单机 CLI + 桌面端零终端直嵌」定位上守住极低的**偶然复杂度**：用重写判据（拿需求从零重写一遍会自己长回来的才是本质复杂度）和删除判据（删掉后行为不变且无用户损失的多半是自我征收的结构税）收敛全盘设计，并与经典软件工程第一性原理对齐：

| 工程问题 | 经典软件工程原型 | Mizar 收敛落点 |
| :--- | :--- | :--- |
| **真相源与可回放性** | 数据库 **WAL** / **Git** 不可变日志（状态只是日志折叠视图） | 坚持 `Model-visible means logged`（与 DeepSeek Harness 一致）及冻结不可变 `Arc<GenerateOptions>`；历史消息仅从持久 `SessionEvent` 确定性投影，禁止发包前黑盒篡改 |
| **多插件拦截与异步插话** | **Erlang Actor** / **Go CSP**（靠信箱传消息，不靠返回值互掐） | 内存态 `SessionInbox` + 顺序无关、无返回值的 `agent/turn-stopping`；谁要拦截停轮或唤醒，只向信箱投递不可变消息 |
| **模型与工具拦截扩展** | **Ring** / **Rack** / Rust **`Tower`** 洋葱中间件（`Request -> Response`） | 复用 `cordis` `waterfall` 收口输入门禁、模型流链、错误处理与工具三段管线，不额外发明独立 Hook 引擎或优先级规则体系 |
| **接口粒度与安全底线** | **UNIX 机制与策略分离** & Ousterhout **深模块（Deep Modules）** | 窄接口隐藏流尾观测、缓存续命与成本审计细节；审批拒签与工作区边界等安全门禁写死在内核固定检查位，不依赖插件顺序或注册表维护 |
| **跨端与未来外部扩展** | **Neovim** 无头内核 + **LSP** 单一文本协议 | 保持 `mizar-client-core` / `mizar-client` 零终端依赖；未来对外开放扩展时优先走单一双向 stdio JSON-RPC 协议，不并行维护双套运行时宿主 |
| **运行形态与存储定位** | **SQLite** 单进程哲学（单机单文件做到极致，拒绝分布式幻想） | 保持轻量内存 Tokio 执行循环 + 单机 SQLite WAL，明确不引入重型持久化任务状态机或常驻多租户 BFF |

### 不可退让的 4 条架构红线

1. **绝不引入分布式 Durable Task 状态机**：不将工具执行或流式切片塞进跨进程 checkpoint 事务状态机，维持内存 `AgentLoop` + 追加式 `SessionEvent` 的轻快响应。
2. **绝不做原生 `.dylib` / `.so` / `.dll` ABI 插件**：规避跨编译版本 Rust ABI 不稳定、`async Future` 跨 FFI 边界及单插件 `panic` 拖垮主进程的风险。
3. **绝不击穿纯核编译隔离边界**（见 [AGENTS.md](../../AGENTS.md)）：`mizar-client-core`、`mizar-tui-ui`、`mizar-llm-types`、`mizar-trajectory-core` 严禁引入 `tokio`、`cordis`、`reqwest`、`rquickjs` 或 `rusqlite`；`mizar-client` 严禁引入 `ratatui`、`crossterm` 或 `mizar-tui-ui`。
4. **绝不退回「发包前黑盒改写历史」**：拒绝类似 Pi 1.0/2.0 `prepare-request` 的发前临时改写钩子；所有模型可见文本必须先定型落盘为 `SessionEvent` 或由已注册的确定性能力投影算出。

---

## 二、主循环与不可变数据流总图

收敛后的完整单步数据流如下（兼顾**可回放**、**顺序无关拦截**与**零克隆预热重放**）：

```text
用户输入 / 内部提醒 / 外部事件
  │
  ▼  TurnInput::User | TurnInput::Notice
[ SessionInbox ] ──(Steer 步间插话 / FollowUp 唤醒与排队 / Inject 静默附着; auto_wake_limit 熔断)
  │
  ▼  认领本步待入账输入
[ agent/pre-step (waterfall) ]  ◀── 唯一允许改写/展开/拒收新输入的关口（FileMentions、环境上下文）
  │                                  （Reject 则直接停步；Enter 后立即写库定型）
  ▼  不可变落盘
[ SessionStore (SQLite WAL) ]   ──► User / ContextNotice / ToolResultPrune / Usage(LogOnly)
  │
  ▼  确定性 derive_messages() + systemPrompt + declared tools
[ Arc<GenerateOptions> (冻结只读; origin: RequestOrigin, disable_continuation) ]
  │
  ▼  LlmRouter 前置参数规整 + 确定性模型能力投影（如纯文本模型自动剥离图片）
[ llm/stream (waterfall: 重试 / 路由) ]
  │        ├── 失败 / 上下文溢出 ─► [ agent/request-error (waterfall) ] ─► 压缩/卸载/降级重试
  │        └── 正常完成          ─► [ llm/after-response (emit 只读) ]
  │                                       │
  │                                       ├─► mizar-cache-warming（按四道门禁 + $0.05 期望收益测算）
  │                                       │     └─► stream_direct(max_tokens=1, disable_continuation=true)
  │                                       │     └─► 成功写入 SessionEvent::Usage("cache_warm") + 刷新 compaction 活跃时间
  │                                       └─► Cache Re-billed 浪费审计（/session 展示）
  ▼  若存在 tool_calls
[ tools/pre-execute (含内核固定安全检查) ─► tools/execute ─► tools/post-execute ] (waterfall)
  │
  ▼  若无 tool_calls 且 SessionInbox 为空
[ agent/turn-stopping (serial 无返回值) ] ──► 守卫向 SessionInbox 投递 Steer ─► 复读信箱非空 ─► 继续下一 step
  │
  ▼  SessionInbox 仍空（或耗尽 auto_wake_limit）
TurnEndReason::Completed
```

---

## 三、极简 Hooks 体系设计

### 1. 核心约束：不造第二套“Hook 引擎”，只分 2 种语义角色

Mizar 的 [cordis 事件总线](../../crates/cordis/src/events.rs) 已原生具备 `emit` / `parallel` / `serial` / `bail` / `waterfall` 与 `effect` 自动卸载能力；`mizar-tools` 与 `mizar-llm` 也已跑通核心 `waterfall`。因此 Hooks 体系**直接复用 `cordis`**，并在语义上锁死为 **2 种角色**：

1. **管道包裹（`waterfall`）**：全系统**唯一允许**改写载荷、包裹执行流、重试或拒收放行的挂点（对应中间件模型）。
2. **只读观察（`emit` / 无返回值 `serial`）**：物理上拿不到上游载荷的修改权，不以返回值裁决主流程，且监听者异常互相隔离；它们只能记账、统计，或者向 `SessionInbox` 投递一条消息（对应 Actor 信箱模型）。

### 2. 全系统锁死的 7 个正交挂点（现状已有 4 个，拟议仅升级/补齐 3 个）

| 位置 | 挂点名 | 模式 | 唯一职责 | 现状 vs 拟议改造 |
| :--- | :--- | :--- | :--- | :--- |
| **入库前关口** | `agent/pre-step` | `waterfall` | **唯一允许改写、展开、注入或拒收新输入的关口**；放行后立即落盘 `SessionEvent`，之后不可篡改 | **现状**为只读 `serial`（[mizar-agent-loop](../../crates/mizar-agent-loop/src/lib.rs)）；**拟议**升级为 `waterfall`，收口 [FileMentions](../../crates/mizar-client/src/mentions.rs) 并剥离主循环硬编码的 `ensure_environment_context` |
| **模型调用链** | `llm/stream` | `waterfall` | 标准 `Request -> Stream` 中间件链（重试、主备切换、路由），输入为冻结的 `Arc<GenerateOptions>` | **现状**已存在（[mizar-llm](../../crates/mizar-llm/src/runtime.rs)）；**拟议**将 [RoutedAdapter](../../crates/mizar-llm-router/src/lib.rs) 默认参数补全与确定性能力投影前移至进链之前，确保快照与真实出网参数一致 |
| **模型完成回执** | `llm/after-response` | `emit`（只读） | `LlmRuntime` 内核统一包一次流尾；仅当请求正常完成（非取消/错误）时广播 `(Arc<GenerateOptions>, provider, model, finish_reason, usage)`，消除各观察插件重复包装异步流的开销与出错面 | **拟议新增**（[mizar-llm](../../crates/mizar-llm/src/runtime.rs) 深模块收口） |
| **请求错误自愈** | `agent/request-error` | `waterfall` | 统一承接模型请求失败后的语义恢复（上下文溢出压缩、图片超限卸载、限流切换路由等） | **现状**为狭义 `agent/context-overflow`（[mizar-agent-loop](../../crates/mizar-agent-loop/src/lib.rs)）；**拟议**泛化升级为 `agent/request-error` |
| **工具三段管线** | `tools/pre-execute`<br>`tools/execute`<br>`tools/post-execute` | `waterfall` | 审批拦截 → 执行包裹（超时/统计）→ 结果改写与重复调用守卫；**硬安全边界（工作区限制、审批拒签）固定在内核检查位，不依赖插件顺序、不可被外部绕过** | **现状**已完整存在（[mizar-tools](../../crates/mizar-tools/src/lib.rs)）；保持骨架，只确保内核硬安全检查位于不可跳过位置 |
| **停轮检查点** | `agent/turn-stopping` | `serial`（无返回值） | 在无工具调用且 `SessionInbox` 为空、准备返回 `TurnEndReason::Completed` 前触发；认为不应结束的守卫插件直接往 `SessionInbox` 投递 `Steer`，主循环复读信箱非空即延续下一步 | **拟议新增**（[mizar-agent-loop](../../crates/mizar-agent-loop/src/lib.rs)） |

### 3. 外部用户自定义 Hooks 的边界

未来若需支持用户配置外部命令钩子（如 shell hook），**核心层完全不感知其存在**。只需在应用层挂载一个普通桥接插件订阅上述内部挂点（如 `tools/pre-execute`、`agent/turn-stopping`）：命令非零退出即拒绝该工具调用，输出文本即转为 `Steer` 投递到 `SessionInbox`。内核与外部扩展共享同一套不可变日志与防爆约束。

### 4. 明确砍掉的设计负担

- **砍掉 38 事件扁平总线**：不设立类似 Pi 1.0 的巨型中心 `ExtensionRunner`，不为每个事件手写专用调度器。
- **砍掉四阶段组合框架与独立 Guard 注册表**：在单一内核控制流下，把安全门禁固定在内核不可跳过位置、把观察类收口到 `emit` / 无返回值 `serial`，即可根治顺序打架，无需引入多阶段优先级注册表。
- **砍掉 `agent/prepare-request` 黑盒改写钩子**：杜绝发包前无日志留痕地临时修改消息列表。

---

## 四、极简内存 Actor `SessionInbox` 收敛口径

在 [plugin-runtime-and-inbox.md](plugin-runtime-and-inbox.md) 既定方案基础上，按最小实用原则收敛运行契约：

1. **入口身份区分（`TurnInput`）**：在 `mizar-agent-loop` 新增 `TurnInput::User { text }` 与 `TurnInput::Notice { source, text }`，为 `String` / `&str` 保留 `From` 转换以兼容现有签名；所有来自插件或唤醒的 `Notice` 落库为 `SessionEvent::ContextNotice`，绝不伪造成主人发言。
2. **纯内存三语义队列（`SessionInbox`）**：在 `mizar-agent-loop` 按 `session_id` 维护纯内存信箱，统合 `Steer`（步间插话，**不中断正在执行的工具**）、`FollowUp`（忙时排入下一轮，闲时唤醒）、`Inject`（仅沉淀上下文，不主动触发推理）；替换目前滞留在 [mizar-client/src/events.rs](../../crates/mizar-client/src/events.rs) 与 TUI `runner.rs` 中的私有队列与强杀重开式 `pending_steer`。
3. **双轨打断保留**：插件纠偏、停轮守卫、普通排队的步间插话走不毁掉当步工具成果的 `Steer`；TUI 与未来桌面端仍保留显式强打断快捷键（直接触发 `CancellationToken` 中止当前生成流）。
4. **单整数防爆护栏（`auto_wake_limit`）**：按 `session_id` 计数连续由非人类 `Notice` 触发的自动唤醒次数；人类投递 `TurnInput::User` 时立即清零；超出预算后后续自动化 `FollowUp` / 闲时 `Steer` 自动降级为 `Inject`（保留内容、切断无限推理循环）。
5. **暂不预支的复杂度**：首期**不做**持久化 Inbox 事件溯源（不写 `agent/inbox/spliced`）也不做复杂的合并规则引擎；当实际场景出现高频通知刷屏时，再增补简单的 `coalesce_key` 原地覆写即可。

---

## 五、Prompt Cache 续命与浪费审计深模块

Prompt Cache 续命本身在服务端与计费规则上有真实外部边界，按 **深模块（Deep Module）** 思路把复杂数学与容错逻辑全部封装在独立可选插件内，保持对外契约极简。

### 1. 前置基础补齐（跨越本质复杂度）

- **请求归属与线程控制**（[mizar-llm/src/types.rs](../../crates/mizar-llm/src/types.rs)）：`GenerateOptions` 新增 `origin: RequestOrigin`（`SessionTurn` / `Compaction` / `SessionTitle` / `CacheWarm` / `Subagent` / `Custom`）与 `disable_continuation: bool`（防止保活请求推进或覆盖服务端有状态会话线程）。
- **目录元数据下沉**（[mizar-llm-router/src/config.rs](../../crates/mizar-llm-router/src/config.rs)）：将 `PromptCacheConfig { short, long }`、`ModelCostConfig`（含 `tiers` 阶梯定价）与 `force_adaptive_thinking` 能力下沉为纯 DTO，随 `mizar-llm-types` / `LlmRuntime` 暴露。
- **Anthropic 真实缓存断点**（[mizar-llm-messages/src/lib.rs](../../crates/mizar-llm-messages/src/lib.rs)）：补齐 `CacheRetention`（`Short` 5 分钟 / `Long` 1 小时）对应的 `cache_control` 透传，与 `Completions` / `Responses` 的 `prompt_cache_key` 能力拉齐。
- **零迁移非上下文账本事件**（[mizar-session/src/model.rs](../../crates/mizar-session/src/model.rs) & [mizar-session-sqlite/src/schema.rs](../../crates/mizar-session-sqlite/src/schema.rs)）：新增 `SessionEvent::Usage { kind, provider, model, usage, note }`（`EntryVisibility::LogOnly`）。依托 SQLite 通用的 `(kind, payload_json)` 表结构，**无需任何 Schema 迁移**即可落盘，且 `derive_messages()` 天然将其排除在模型上下文之外。

### 2. 可选脊柱模块 `mizar-cache-warming`（挂载 `SpinePlugins`）

该模块直接受 [SpinePlugins](../../crates/mizar-agent-spine/src/plugins.rs) 热启停管理，对主循环零侵入：

- **零拷贝重放机制**：订阅 `llm/after-response`，仅处理 `RequestOrigin::SessionTurn`；直接持有并原样重放冻结的 `Arc<GenerateOptions>`，通过 `stream_direct` 发送 `max_tokens: Some(1)`、`disable_continuation: true`、无重试的最小请求，借助 DSH 不可变快照铁律保证字节与前缀完全一致、无并发篡改风险。
- **4 道硬门禁**：
  1. **目录与重放安全门禁**：仅当模型在 `models.json` 显式声明 `promptCache` TTL 与 `cost` 时启用；若开启推理且为非 `force_adaptive_thinking` 的旧版 Anthropic budget-based thinking，禁止重放（防止 `max_tokens: 1` 改写 `budget_tokens` 破坏服务端缓存键）。
  2. **期望净收益门禁**：每次到期前按 `E[savings] = P(continue) * missCost - warmCost` 测算（`streaming` 阶段取 `1.0`，`idle` 阶段取 `0.15`），仅当净期望收益 `>= $0.05` 时才续命。
  3. **窗口硬顶与休眠防错峰**：设置 60 分钟（`streaming`）/ 30 分钟（`idle`）最长保活上限，并计算 `refresh_deadline = next_warm_at + (ttl - delay) / 2`；若系统休眠或事件循环卡顿导致唤醒时已过截止线，宁可放弃续命也不冒险在失效瞬间打出全额 `cacheWrite`。
  4. **一次性/子任务保护**：无头模式 `mizar exec` 与 `RequestOrigin::Subagent` 强制禁用 `idle` 续命，避免短命任务在后台空耗预算。
- **跨模块联动与浪费审计**：
  - 每次 `cache_warm` 成功后，除写入 `SessionEvent::Usage(kind = "cache_warm")` 外，必须同步刷新 [mizar-compaction](../../crates/mizar-compaction/src/service.rs) 的 `last_activity_at`，避免“刚成功保活又被压缩服务按 300s `cold_cache_idle_secs` 误判缓存已冷而压掉前缀”。
  - 利用 `llm/after-response` 连续记录与 `Usage("cache_warm")` 节点在 `/session` 面板展示实时状态及意外失效造成的 `Cache Re-billed` 浪费金额，用真实观测数据指导提示词与工具稳定性迭代。
- **克制裁剪**：首期**不新增** `cache/warming-decision` 二次覆盖钩子，避免过早把内部经济测算逻辑暴露为插件竞态点。

---

## 六、UNIX 进程式单层 `mizar-subagent` 设计

### 1. 必要性判据（通过本质复杂度检验）

Subagent 解决的是单上下文窗口下的两个硬物理约束，而非概念炫技：
- **隔离检索污染**：调研代码库、追踪调用链会产生数万 token 的中间工具输出；将其封装在独立子会话中执行并只向主会话交回精炼结论，可保住主上下文注意力与前缀缓存命中率。
- **只读并行扇出与独立视角**：一次对多个模块并发调查，或以干净上下文执行独立 Review。

Mizar 现已具备全部底层基石：[mizar-session](../../crates/mizar-session/src/lib.rs) 原生支持 `header.parent_id` 会话谱系与 `mizar-recall` 谱系回溯；`mizar-agent-loop` 按 `session_id` 独立加锁并接受 `CancellationToken`；`mizar-tools` 支持工具白名单与 `ReadOnly` 预设。

### 2. 5 条极简不变量（独立可选插件 `mizar-subagent`）

将 Subagent 设计成一个类似 UNIX `fork/exec + wait` 的普通可选脊柱插件，对 `AgentLoop` 零特殊分支：

1. **形态即工具（同步优先）**：向模型暴露窄接口工具 `subagent(prompt, mode?: "read_only" | "edit", model?)`。主模型像调用 `grep` 一样调度它并同步取回总结文本作为 `tool_result`；由于工具批量执行本身支持并行，模型单步发起多个只读 `subagent` 调用即可免费获得并发调研。
2. **独立子会话 + 取消级联**：每次调用在 `SessionStore` 创建 `parent_id = 父 session_id` 的子会话（完整落盘 SQLite 以备审计与 `recall`，绝不污染主会话历史），并绑定 `parent_cancel.child_token()`——用户一旦取消主任务，所有子任务瞬间联动终止，零孤儿后台任务。
3. **默认只读，写权慎开**：`mode` 默认为 `"read_only"`（仅开放 `read` / `grep` / `find` 等只读能力，天然并发安全）；`edit` 模式显式按需授权，首期**不内置**自动 `git worktree` 编排或文件级冲突调度器。
4. **3 条防脱缰硬护栏**：
   - **严格单层（禁止套娃）**：子会话的可调用工具集中直接过滤隐藏 `subagent` 工具，无需引入树状 Scope 容器即可杜绝递归爆炸；
   - **强制步数与超时封顶**：子会话默认强制设置 `max_steps`（如 `20`）与调用超时，防止失控空转；
   - **身份隔离标签**：所有模型调用统一标注 `RequestOrigin::Subagent`，自动跳过 `idle` 缓存预热与自动标题生成。
5. **未来平滑演进通道**：首期只做同步 `spawn + wait`；若后续确有后台长耗时子任务需求，只需在子任务收尾时向父会话的 `SessionInbox` 投递一条 `TurnInput::Notice`（`FollowUp` 或 `Steer`），完全复用已有 Actor 信箱机制。

---

## 七、明确延后的能力边界（「痛了再长，绝不预支」）

| 候选特性 | 当前决定 | 未来触发条件与零返工演进路径 |
| :--- | :--- | :--- |
| **Virtual Models 虚拟模型路由** | 延后 | 等确需「强模型规划 + 廉价模型执行」或自动主备切换时，作为 `llm/stream` `waterfall` 中的一层中间件直接包入，不改内核模型结构 |
| **树状作用域 `dsh-scope`** | 延后 | 仅在开放多层 Subagent 树且出现跨层工具继承/事件冒泡冲突时引入；当前单层 `subagent` + `session_id` 过滤已足够且零心智税 |
| **非破坏性会话分支 `/tree`** | 延后 | 底层 `SessionHeader::parent_id` 与前缀 Fork 接口已就位，待高频分叉探索需求明确后在持久化与 UI 视图层顺势接通 |
| **外部扩展宿主载体** | 延后 | 等桌面端 Merak 真正开放第三方扩展时，优先采用单一 **LSP 式双向 stdio JSON-RPC** 桥接协议（或薄扩 MCP），避免同时维护 `rquickjs` 与子进程两套宿主 |
| **UI 历史冷详情按需卸载** | 延后 | 严格按 [locality-and-memory-optimization.md](locality-and-memory-optimization.md) P2 规划，在超长会话成为常态瓶颈时再启用折叠卡片正文按需回读 |

---

## 八、现状复杂度审计：已确认的过度设计与保留边界

基于当前工作区 33 个 crate、约 77k 行 `src/` 源码的静态盘点，下文记录**已确认的偶然复杂度**与**不应误删的本质复杂度**，作为后续演进时的瘦身与止血基准（不盲目大重构，只在增量迭代中逐步收口）。

### 1. 已确认的 4 处偶然复杂度（及经典软件工程替代思路）

| 现状痛点与实锤定位 | 引入的结构税 | 经典软件工程更优雅解法 |
| :--- | :--- | :--- |
| **将 JS 动态运行时 `cordis` 原样搬入静态编译的 Rust 单体**<br>（`cordis` 共 3,207 行，含 1,123 行 [fiber.rs](../../crates/cordis/src/fiber.rs) 与 254 行 [reflect.rs](../../crates/cordis/src/reflect.rs)；`parallel` / `bail` / `isolate` / `priority` / `prepend` 生产调用均为 0 次；18 个 `Plugin` 与 12 个 `Service` 全在编译期静态装配） | 把编译期缺失参数错误退化成运行期静默失败（`MissingInject`）；手写 14 个重复 `*ContextExt` 样板；本来同步可完成的服务装配变成异步竞态，迫使 [mizar-client/src/events.rs](../../crates/mizar-client/src/events.rs) 维护 `wait_for_services` 探针 | **静态构造注入（Composition Root）+ RAII `Drop`**：在单一装配点按 DAG 显式传入 `Arc<T>` 构造核心服务，由编译器担保依赖完备性，用 `Vec<Cleanup>` 或 `Drop` 负责资源逆序清理 |
| **核心流式数据面拆成 16 个字符串 `emit` 与 1:1 镜像枚举**<br>（[mizar-agent-loop/src/lib.rs](../../crates/mizar-agent-loop/src/lib.rs) 拆出 16 个 `EVENT_AGENT_*` 字符串；[mizar-client/src/events.rs](../../crates/mizar-client/src/events.rs) 再写十几段同构 `ctx.on` 转发及 `AgentStatus` / `TurnEndReason` / `StepOutcome` 镜像投影） | 每次 token 增量都要经过 `HashMap<String, ...>` 与 `&dyn Any` 向下转型，失去编译期穷举检查并多出整套 1:1 搬运胶水 | **封闭代数数据类型 `enum` + 强类型单通道 + 纯值类型下沉共享**：流式事件收口为单个封闭 `enum AgentEvent` 走强类型广播通道，纯值枚举直接下沉无运行时纯类型 crate 共享 |
| **局部 Crate 切分过细与历史 `legacy` feature 包袱**<br>（`mizar-tool-time` 43 行、`mizar-system-prompt` 163 行、`mizar-llm-mock` 207 行、`mizar-tool-fs` 258 行、`mizar-llm-retry` 439 行；[mizar-agent-spine/src/lib.rs](../../crates/mizar-agent-spine/src/lib.rs) 与 [plugins.rs](../../crates/mizar-agent-spine/src/plugins.rs) 需维护 7 组 `#[cfg(feature)]` 门控） | 相同依赖边界的组件被硬拆成多套 `Cargo.toml` 与 feature 组合矩阵，抬高汇编编排和 CI 组合检查成本 | **Ousterhout 深模块 + 语言原生 `mod`**：仅在需要切开编译/依赖边界（如零异步纯核、零终端宿主、隔离重型 HTTP/SQLite/QuickJS）时才独立成 crate，其余用 `mod` + `pub(crate)` 封装 |
| **自造私有孤岛日志门面 `cordis::Logger`**<br>（[cordis/src/logger.rs](../../crates/cordis/src/logger.rs) 共 199 行；仅服务少数调用 `ctx.logger()` 的文件，前端另需 `LogRecord -> UiLogRecord` 投影） | 生态库与底层模块日志无法走同一管线，额外维护自定义Sink 与级别解析 | **标准日志门面（SLF4J / Rust `tracing` 或 `log`）**：用生态标准门面统一日志生产端与 TUI/文件输出端 |

### 2. 反过度审查护栏（明确不应乱砍的 4 块本质复杂度）

以下模块体量虽大，但均可通过“重写测试”——删去即直接损失核心能力或架构边界，**禁止当作过度设计误删**：

1. **纯核零运行时编译隔离**（`mizar-client-core` / `mizar-tui-ui` / `mizar-llm-types` / `mizar-trajectory-core`）：直接保障 UI 秒级增量编译、无异步确定性单测及未来 Merak Desktop 零终端直嵌。
2. **上下文压缩与情景回溯闭环**（`mizar-compaction` + `mizar-recall` + `mizar-session-sqlite` 的 Single-Writer 与 FTS5 n-gram）：解决窗口溢出、压后防陈旧 `write`、SQLite 写锁争用与穿透压缩边界检索等硬物理约束。
3. **五级工具暴露治理链**（`ToolExposure` + `mizar-codemode` + `mizar-tool-search`）：对抗大量 MCP 工具注入造成的上下文爆炸与中间结果污染。
4. **高密度终端渲染域**（`mizar-tui-ui`）：体量源自 Markdown 渲染、应用级选区、时间轴导轨与三泳道轨迹分析的功能密度本身，且已按视口局部性优化，属于本质产品复杂度。

### 3. 务实治理口径（只止血与顺手瘦身，不搞推倒重写）

由于 `cordis` 已稳定运行并具备完整单测与并发模型检验，**不为历史遗产单独发起高风险大重写**，后续所有改造遵守 3 条轻量止血规则：

1. **冻结 `cordis` 扩张**：不再向 `cordis` 增添新状态、新分发模式或日志特性；增量代码只把它当作「生命周期清理器 + 7 个正交挂点」使用。
2. **抑制微 crate 膨胀并伺机清理历史包袱**：除非需要切出新的编译/依赖边界，新小能力优先以现有 crate 内 `mod` 组织而非新建微 crate；在确认不再需要旧套件时再统一退役 `legacy`（`mizar-tool-time` / `mizar-tool-fs`）以削薄 `mizar-agent-spine` 的 feature 矩阵。
3. **跨端纯 DTO 直接共享**：后续新增的跨端纯数据结构与状态枚举（如 `RequestOrigin`、预热状态、账单摘要）直接下沉到无运行时纯类型 crate 共享，不再向 `mizar-client` 追加 1:1 `to_ui_xxx` 镜像搬运代码。

---

## 九、分阶段落地路线与验证门禁

### PR 1：铁律底座 + 7 挂点 Hooks + `SessionInbox` + `Usage` 账本
- **核心改动**：
  - `mizar-llm` / `mizar-llm-types`：`GenerateOptions` 补充 `origin: RequestOrigin` 与 `disable_continuation: bool`；`LlmRuntime` 新增只读流尾广播 `llm/after-response`；
  - `mizar-llm-router`：将默认参数规整与模型多模态能力投影前移至 `llm/stream` 之前；
  - `mizar-session`：`SessionEvent` 新增 `Usage(LogOnly)` 账务变体；
  - `mizar-agent-loop`：新增 `TurnInput`、`SessionInbox`（`Steer` / `FollowUp` / `Inject` + `auto_wake_limit`）；将 `agent/pre-step` 升级为 `waterfall` 并剥离硬编码 `ensure_environment_context`；新增无返回值 `agent/turn-stopping`；将 `agent/context-overflow` 泛化为 `agent/request-error`；
  - `mizar-client` / `mizar-tui`：将原客户端私有排队与强杀式 `pending_steer` 收口至 `SessionInbox`（保留快捷键强打断双轨并行）。

### PR 2：真实缓存链 + `mizar-cache-warming` + 浪费审计
- **核心改动**：
  - `mizar-llm-messages`：落实 Anthropic `cache_control`（5m / 1h）断点；
  - `mizar-llm` / `mizar-llm-types`：下沉 `PromptCacheConfig` / `ModelCostConfig` 纯 DTO 与无状态期望收益测算函数；
  - 新增可选脊柱模块 `mizar-cache-warming` 并接入 `SpinePlugins`：落地四道门禁、休眠截止线与 [mizar-compaction](../../crates/mizar-compaction/src/service.rs) 冷缓存活跃时间同步；
  - `mizar-client-core` / `mizar-client` / `mizar-tui-ui`：在 `/session` 面板展示实时预热状态与 `Cache Re-billed` 浪费审计（同步更新 [plugins.md](../plugins.md)）。

### PR 3：UNIX 进程式单层 `mizar-subagent`
- **核心改动**：
  - 新增可选脊柱模块 `mizar-subagent` 并接入 `SpinePlugins`：暴露 `subagent` 工具，使用 `parent_id` 创建子会话并绑定 `child_token()`，默认 `read_only`，强制深度 1 过滤、`max_steps` / 超时封顶与 `RequestOrigin::Subagent`（同步更新 [plugins.md](../plugins.md)）。

### 全量门禁要求（每个 PR 合并前必须全部通过）

```sh
cargo check --locked -p mizar-agent-spine --no-default-features --all-targets
cargo test --locked --workspace --no-fail-fast
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
```
