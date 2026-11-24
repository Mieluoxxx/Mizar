# Mizar 模块与依赖清单

本文档介绍 `crates/` 下共 **10 个 crate** 的分层作用，以及它们通过 **`mizar-core::spine::MizarRuntime` 静态组合根 + `mizar_types::hooks` 强类型钩子 / RAII `HookGuard` 清理** 构建的 Cargo 无环 DAG。

目前仓库为纯 Virtual Workspace，唯一的终端入口为 `crates/mizar-tui/src/main.rs`（`mizar-tui`），通过 `MizarRuntime::build` 按编译期确定顺序装配 Agent 能力，经跨端共享层（`mizar-types` / `mizar-client`）连接纯前端工作台 `mizar-tui-ui`（同时可供桌面端 Merak 直嵌复用）。**纯核心契约/纯 UI/强类型钩子原语**与**异步运行时宿主层**已彻底解耦。

---

## 一、纯类型、强类型钩子与纯 UI 层（无 `tokio` / 无 HTTP / 无重运行时）

| Crate | 模块 / 定位 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- |
| **`mizar-types`** | `hooks` / `llm` / `trajectory` / `client` / `rpc` | **纯核心契约与纯算法层**（严禁 `tokio` / `reqwest` / `rquickjs` / `rusqlite` / `tracing` / `ratatui` / `crossterm`）：<br>• `hooks`：`BoxError`、`BoxFuture`、RAII 清理守卫 `HookGuard`（支持分组逆序清理）、`HookOptions`、`SyncEventBus<E>`、可插拔拦截链 `WaterfallHook<I, O>`（与 `Next`）以及短路钩子 `SerialHook<I, O>`<br>• `llm`：统一 `Message`、`ContentBlock`、`BlockAssembler`、`RequestOrigin`、`PromptCacheLifetime`、`ModelCostRates`、`cache_audit`（被动 `Cache Re-billed` 审计）、`LlmError` 与 `SseParser`<br>• `trajectory`：`TraceEvent`、`TrajectorySnapshot`、`fold_trajectory`、三泳道时间轴 `derive_timeline` 与异常检测 `detect_anomalies`<br>• `client`：`HostExtensionFlags`（10 个官方扩展开关与 `"tui"` 级联派生）、`PromptQueue`、`TpsLedger`、`WorkspaceFileCache`、`ClientEvent`、`mentions`、`reveal`、`tool_format` 与纯 DTO<br>• `rpc`：跨端 JSON-RPC 2.0 NDJSON 协议契约（`SessionSubscribeSnapshot`、8 类 `RpcNotification`、12 类请求/响应 DTO 与锁/会话错误码） | 无 |
| **`mizar-tui-ui`** | 纯终端工作台前端 | `AppState` 状态机、`TuiFeatures` 扩展开关、`pi-emphasis` 视觉语法、虚拟化 Markdown/代码渲染、双栏侧栏、TimelineRail、Overlay、主题与轨迹全屏视图；`terminal` 发射 OSC 7501（含 `pid=<pid>`）与 OSC 9;4（严禁 `tokio` / `reqwest` / `rquickjs` / `rusqlite` / `tracing`） | `mizar-types` |

---

## 二、基础能力层（传输、工具与持久化）

| Crate | 模块 | 关键类型 / 安装入口 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- |
| **`mizar-llm`** | `runtime` / `http` / `completions` / `messages` / `responses` / `router` / `retry` / `mock` | `LlmRuntime`、`LlmRouter`、`install_retry`、`CompletionsAdapter`、`ResponsesAdapter`、`MessagesAdapter`、`register_mock_adapter` | 抽象 LLM 服务与 `stream_hook` / `on_after_response` / `on_adapters_updated` 强类型钩子；共享 `reqwest` 客户端与 SSE 驱动（`http`）；三大协议适配器；声明式多提供方路由与热更新（`router`）；未发可见内容前瞬时错误自动退避重试（`retry`）；离线规则假模型（`mock`） | `mizar-types` |
| **`mizar-tools`** | `lib` / `truncate` / `mutation_queue` / `coding` / `search` | `ToolRuntime`、`register_coding_tools`、`register_tool_search` | 工具注册表、三段强类型管线（`pre_execute_hook` → `execute_hook` → `post_execute_hook`）、事件总线（`on_change` / `on_result`）与同路径写串行队列；编码 6 件套（`read`/`bash`/`edit`/`write`/`grep`/`find`，三档预设 `All`/`MinimalCoding`/`ReadOnly` + 纯函数 `build_tools_guide`）；BM25 `tool_search` 工具检索与按需激活 | `mizar-types` |
| **`mizar-session`** | `lib` / `model` / `persistence` / `sqlite` | `SessionStore`、`SqliteSessionPersistence` | 只追加会话事件日志（含 `Surface` / `LogOnly` 分级、`Selection` / `Usage` / `ToolResult.meta` / `Approval` / `ToolResultPrune` / `StepEnd.error` / `TurnEnd.{error,summary}` 账务、`derive_messages` 上下文零污染投影、`list_events_paged` 双向分页与 `on_event`/`on_created`/`on_loaded`/`on_title_updated` 总线）；嵌入式 SQLite（WAL + Single-Writer 后台线程，schema v4 自动清理历史 FTS 表） | `mizar-types` |

### 工具分层暴露策略（`mizar-tools::ToolExposure`）

| `ToolExposure` | 直接声明给模型 | 可被 `codemode` 脚本调用 | 列在 `codemode` 描述中 | 典型使用者 |
| :--- | :--- | :--- | :--- | :--- |
| `Direct` | ✓ 激活后 | ✓ | — | `read` / `bash` / `edit` / `write` / `grep` / `find` / `skill` |
| `ModelOnly` | ✓ 激活后 | ✗（自身是编排工具） | — | `tool_search`、`codemode`、`subagent` |
| `Codemode` | ✗（需激活） | ✓ | ✓ | MCP 工具默认、`codemode` 内部依赖的工具 |
| `Deferred` | ✗（需激活） | ✓ | ✗（靠 `tool_search` / `searchTools()` 发现） | MCP 工具可选项 |
| `Hidden` | ✗ | ✗ | ✗ | 显式屏蔽（`exclude_tools`） |

激活与排除优先级依次为（`mizar-tools` 的 `is_default_active`）：**`global_active_overrides`** > **`include_tools` 通配白名单** > **`auto_default_active`**（`codemode`/`tool_search` 按需自动启用） > 工具的 `default_active`；任何情况下 `ToolExposure::Hidden` 与命中 `exclude_tools` 的工具都不可激活。

---

## 三、重量级隔离扩展层（沙箱与外部协议）

| Crate | 关键类型 / 安装入口 | 注册工具 | 注入依赖 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-codemode`** | `register_codemode` | `codemode`（`ModelOnly`） | `Arc<ToolRuntime>` | **基于 QuickJS（`rquickjs`）沙箱的多工具 JS 编排引擎**：单独隔离 C 依赖 `rquickjs`，把多步工具调用折叠进一次脚本执行，内建 `searchTools()`、并发信号量（默认 2）、超时取消传播（默认 120 s）与内联预算（默认 3 000 字符） | `mizar-types`、`mizar-tools` |
| **`mizar-mcp`** | `McpManager::start` | `mcp__<server>__<tool>`、`list_mcp_resources`、`list_mcp_resource_templates`、`read_mcp_resource` | `Arc<ToolRuntime>` | **外部 MCP 客户端**：加载 `~/.mizar/mcp.json` 与 `<workspace>/.mizar/mcp.json`，支持 stdio 与 Streamable HTTP（复用 `mizar-llm::http`），按 Server 与逐工具分级暴露；不修改静态 `SystemPrompt`，按需提供 `render_servers_section()` | `mizar-types`、`mizar-llm`、`mizar-tools` |

---

## 四、Agent 核心领域与装配层（`mizar-core`）

`crates/mizar-core` 收拢了 Agent 全部领域循环、上下文管理与统一静态装配，内部按 `src/<mod>.rs` 划分 9 个模块：

| 模块 | 关键类型 / 入口 | 注入依赖 | 作用 |
| :--- | :--- | :--- | :--- |
| `system_prompt` | `SystemPrompt`、`PromptSection` | 无 | 按 `order` 动态聚合系统提示词段落（内置 `persona` 与 `runtime_protocol`），`section()` 返回 `HookGuard` |
| `agent_loop` | `AgentLoop`、`AgentEvent`、`SessionInbox`、`TurnInput` | `llm`、`tools`、`sessions`、`system_prompt` | 托管 `SessionInbox`（`Steer`/`FollowUp`/`Inject`），串联强类型 `pre_step_hook` → 冻结 `GenerateOptions` 调模型 → 工具并发/串行管线 → `request_error_hook` / `turn_stopping_hook`，通过 `AgentEventBus`（`subscribe` / `on_event`）广播强类型 `AgentEvent` |
| `compaction` | `CompactionRuntime::install` | `llm`、`sessions`、`tools`、`agent_loop`、`system_prompt` | 单阶段检查点压缩（真实可用窗口 80% 触发、`<mizar-digest>` 硬事实、膨胀保护、溢出 `retain=0` 恢复）、压后写文件/读去重守卫，以及 Cache-Aware 尾部剪枝（`supersedeReads` / `dropUseless`、8K 热尾 / 90m 冷缓存 Prompt Cache 守卫与 `ToolResultPrune` 持久化标记），提供 `on_start`/`on_progress`/`on_end` |
| `instructions` | `AgentInstructions::install` | `system_prompt` | 启动时静态扫描 `~/.mizar/`、`~/.agents/` 与项目根到工作区祖先链的 `AGENTS.md` / `CLAUDE.md`（及 `.local.md`），去重裁切后通过 `SystemPrompt::section("agent_instructions", 50)` 静态注入（移除子目录动态发现与 `ContextNotice`） |
| `skills` | `SkillRegistry`、`install_skills` | `tools`、`sessions`、`agent_loop` | 分层发现 `SKILL.md`、注册 `skill` 工具、目录注入、写入命中热失效及 `on_change` |
| `trajectory` | `TrajectoryRuntime::install`、`install_repeat_guard` | `agent_loop`、`tools`、`sessions` | 实时采集 `TraceEvent`、`on_anomaly` 异常检测、`trajectory.jsonl` 导出与单文件实时流式追加（`export_file` / `--trace-jsonl` / `MIZAR_TRACE_JSONL`）、以及连续重复失败工具调用拦截 |
| `session_title` | `SessionTitleRuntime::install` | `llm`、`agent_loop` | 异步生成与刷新 `MMDD｜TYPE｜Topic` 会话标题，尊重 `/rename` 人工所有权，自动跳过子代理 |
| `subagent` | `SubagentRuntime::install` | `llm`、`tools`、`sessions`、`agent_loop`、`system_prompt` | UNIX 进程式单层后台常驻分身 `subagent`（`ModelOnly`），支持 `spawn`/`send`/`status`/`stop`/`yield`、按模式软步数预算（`read_only=100` / `edit=200` + `1.0× Steer` / `1.5×` 空工具集收尾）、默认无墙钟硬超时、零 Token 活性看门狗、结构化回执去重、`yield` 并发屏障、终态全文/超限落盘与深度 1 护栏 |
| `spine` | `MizarRuntime::build`、`SpinePlugins`、`build_tools_guide_section`、`LoadedMizarConfig` | 静态组合根 | 按编译期 DAG 顺序静态构造核心服务与 11 个内置可选模块，托管 `HookGuard` 并在 `shutdown()` 时按 LIFO 逆序清理；`SpinePlugins` 控制可选模块原子启停（`mcp` 启动禁用时延后首次按需启动；核心自保护、`tools_guide` `order=20` 段落、识别 `KNOWN_HOST_EXTENSION_IDS` 10 个官方宿主扩展 ID），并承载 `settings.json` / `models.json` / CLI 路由解析 |

**内部 Crate 依赖**：`mizar-types`、`mizar-llm`、`mizar-tools`、`mizar-session`，以及按 Cargo feature 选配的 `mizar-codemode`、`mizar-mcp`。

`subagent(action="yield", message="…")` 用于交还主代理回合：若 `message` 非空则发射并持久化一条助手回复，在无在跑子任务且无待投递回执时拒绝执行，写回全部工具结果后触发并发 barrier（正常结算等待全部在跑子任务完成后聚合唤醒，失败/看门狗/人工输入立即打断）；单次无头 `exec` 不启用子代理。

### Cargo Features 清单

| Crate | Feature | 默认开启 | 作用 |
| :--- | :--- | :--- | :--- |
| `mizar-core` | `codemode` | ✓ | 启用 `mizar-codemode` QuickJS 沙箱模块（`AgentToolsMode::Coding::codemode`） |
| `mizar-core` | `mcp` | ✓ | 启用 `mizar-mcp` 外部 MCP 客户端模块（`AgentToolsMode::Coding::mcp`） |
| `mizar-session` | `bundled` | ✓ | 启用 `rusqlite/bundled` 内置编译 SQLite（关闭后可链接系统 SQLite） |

---

## 五、宿主接入与终端入口层

| Crate | 入口 / 挂载函数 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- |
| **`mizar-client`** | `SessionActor`、`RpcServer`、`ProcessRegistration`、`PendingInteractions`、`install_file_mentions`、`install_approval_gate` | **跨端宿主运行时（严禁引入 `ratatui` / `crossterm` / `mizar-tui-ui`）**：供 `mizar-tui`、Herdr Mod 与桌面端 Merak 共用。包含单会话核 `SessionActor`/`SessionActorHandle`（`actor`：`ActiveTurnBuffer`、单源 `queued_inputs`、`FollowUp` 唤醒循环、双向事件分页、`session.fork`/`tree` 与能力清单）、Per-PID 运行态与单写者锁（`run_dir`：`~/.mizar/run/p-<pid>.sock` 与 `0600` `p-<pid>.json`，含 `without_secret()` 脱敏投影）、Unix Socket + `127.0.0.1:0` 认证回环 TCP 双通道 JSON-RPC 2.0 服务端（`rpc_server`：TCP 首帧常量时间 `auth.hello` 鉴权/`-32003`、`40ms` `stream.delta` 聚合与 `256` 帧慢客户端断连保护）、高危审批门禁与双端原子抢占（`approval`：`PendingInteractions`）、`@` 引用展开（`mentions`）、图片压缩与 EXIF 校正（`images`）、会话投影（`session_view`）、设置热写回（`settings`）、本地 Shell（`local_bash`）、运行控制（`events`）及工作区面板快照（`workspace`） | `mizar-types`、`mizar-llm`、`mizar-tools`、`mizar-session`、`mizar-mcp`、`mizar-core` |
| **`mizar-tui`** | `mizar` 二进制、`mizar serve`、`install_tui_emphasis` | **薄终端宿主客户端与回环网关**：解析 CLI 参数、`mizar exec` 无头单次执行与 `mizar serve` 回环 HTTP/WebSocket 网关（`serve.rs`：强制回环地址绑定与严格 `Origin` 校验、`/health`、脱敏 `/api/processes`、`/api/sessions`、`/ws` 代理活 `p-<pid>.sock` 或为静态 SQLite 会话按需拉起 headless `SessionActor`），实例化 `MizarRuntime` 与 `SessionActor`，通过进程内零序列化 `ClientEvent` 流交互（并可选自动启动 `RpcServer`），专注终端 I/O、剪贴板、`mizar-tui-ui` 渲染桥接及按 `HostExtensionFlags` 挂载终端扩展 | `mizar-types`、`mizar-llm`、`mizar-tools`、`mizar-session`、`mizar-mcp`、`mizar-core`、`mizar-client`、`mizar-tui-ui` |

---

## 六、依赖拓扑简图（10-Crate 无环 DAG）

```text
┌─ 终端入口与跨端宿主层 ──────────────────────────────────────────────────────────┐
│  mizar-tui (CLI + exec + serve + 终端桥接)      Merak Desktop（桌面端直嵌）     │
│    ├─► mizar-tui-ui (纯终端 UI：零 tokio)             │                         │
│    │     └────────────────────────────────────────────┼──► mizar-types          │
│    └───────────────────────┬──────────────────────────┘                         │
│                            v                                                    │
│  mizar-client（前端无关宿主运行时：零 ratatui / crossterm / mizar-tui-ui）      │
│    ├─► mizar-core（agent_loop / system_prompt / compaction / instructions /     │
│    │               skills / trajectory / session_title / subagent / spine）     │
└────┴───────────────────────┬──────────────────────────┬─────────────────────────┘
                             │                          │ (optional features)
                             v                          v
┌─ 会话与隔离扩展层 ─────────┴──────────┐    ┌──────────┴─────────────────────────┐
│  mizar-session (sessions + sqlite)    │    │  mizar-codemode (rquickjs 隔离)    │
│  （仅依赖 mizar-types）               │    │  mizar-mcp (外部 stdio/HTTP MCP)   │
└────────────────┬──────────────────────┘    └──────────┬─────────────────────────┘
                 │                                      │
                 v                                      v
┌─ 协议与工具基础层 ──────────────────────────────────────────────────────────────┐
│  mizar-llm (runtime + http + completions/messages/responses + router/retry/mock)│
│  mizar-tools (tools runtime + mutation_queue + coding 6 tools + tool_search)    │
└───────────────────────────────┬──────────────────────┬──────────────────────────┘
                                v                      v
     [ mizar-types（纯契约 / 轨迹模型 / 客户端算法 / 强类型钩子与 HookGuard） ]
```

**编译隔离边界**：
- `mizar-types` 与 `mizar-tui-ui` 均不依赖 `tokio`、`reqwest`、`rquickjs`、`rusqlite` 或 `tracing`（其中 `mizar-types` 零内部 crate 依赖且亦禁止 `ratatui` / `crossterm`）。
- `mizar-client` 与 `mizar-types` **严禁引入 `ratatui`、`crossterm` 或 `mizar-tui-ui`**，确保 `Merak Desktop` 直嵌零终端依赖。

---

## 七、配置字段速查

全局 `~/.mizar/` 与项目 `<workspace>/.mizar/` 会按目录顺序叠加，后者覆盖前者（`excludeTools`、`customDirs` 与 `disabledPlugins` 是合并/追加而非覆盖）。目录可用 `MIZAR_HOME` 改写。

`settings.json`（`SettingsConfig`，camelCase，全部字段可选）：

| 字段 | 类型 | 作用 |
| :--- | :--- | :--- |
| `defaultProvider` / `defaultModel` | string | 默认提供方与模型 |
| `reasoningEffort`（别名 `defaultThinkingLevel`） | string | 思考强度 `off`..`max` |
| `maxSteps` | number | 单轮工具步数上限，省略则不限 |
| `persona` | string | 覆盖内置人设 |
| `toolsPreset` | `all` / `minimal` / `read-only` | 编码工具预设，对应 `--minimal` / `--read-only` |
| `tools` | string[] | 工具白名单（支持 `*` 通配） |
| `excludeTools` | string[] | 工具黑名单（支持 `*` 通配，多级目录累加） |
| `disabledPlugins` | string[] | 声明式屏蔽内置可选模块 ID 或官方客户端/TUI 扩展 ID（全局 + 项目合并去重排序；支持 TUI `/settings` `[模块]` 运行期零销毁原子启停并原子回写；受保护核心服务忽略并告警） |
| `restrictToWorkspace` | bool | 将文件工具限制在工作区内（**默认 `false`**） |
| `sessionDb` | string | SQLite 落盘路径；等价环境变量 `MIZAR_SESSION_DB`（`none` / `off` 关闭） |
| `codemode` | object | `autoEnable` / `onlyMode` / `inlineBudget` / `timeoutMs` |
| `toolSearch` | object | `autoEnable` / `defaultLimit` |
| `skills` | object | `enabled` / `customDirs` / `bundledDir` / `catalogDescriptionMaxLength` |
| `instructions` | object | `enabled` / `candidates` / `localCandidates` / `projectRootMarkers` / `maxBytes` / `maxSourceBytes` |
| `compaction` | object | `enabled` / `autoCompact` / `defaultContextWindow` / `maxContextTokens` / `triggerRatio` / `keepRecentTokens` |
| `sessionTitle` | object | `enabled`（默认 `true`）/ `model`（可选 `provider/model`，省略则跟随当前会话模型）/ `thinkingLevel`（默认 `"low"`） |
| `subagent` | object | `readOnlySoftBudget`（默认 `100`，`0` 关闭）/ `editSoftBudget`（默认 `200`，`0` 关闭）/ `timeoutSecs`（默认 `0`，`0` 关闭墙钟硬超时）/ `maxConcurrent`（默认 `4`）/ `stallTimeoutSecs`（默认 `45`，设为 `0` 关闭看门狗）/ `maxWatchdogAlerts`（默认 `2`） |
| `ui` | object | `theme` / `screenMode` / `showThinking` / `verboseExpanded` / `sidePanelOpen` / `mouseCapture` / `approvalMode`（可通过 TUI `/settings` 交互修改并自动保存） |

### 客户端宿主与 TUI 扩展模块 ID（`disabledPlugins`）

`mizar-core::spine` 在 `KNOWN_HOST_EXTENSION_IDS` 中登记了以下 **10 个官方客户端宿主与 TUI 扩展模块 ID**。`SpinePlugins::initialize` 会将其识别为合法已知扩展（不触发未知模块告警或启动错误），由 `mizar-types::client::HostExtensionFlags::from_disabled_plugins`、`mizar-tui-ui::TuiFeatures` 及客户端宿主按开关控制挂载：

| 扩展 ID | 常量名（`mizar-types::client`） | 作用域 | 受 `"tui"` 级联 | 控制的能力与禁用效果 |
| :--- | :--- | :--- | :--- | :--- |
| `file-mentions` | `PLUGIN_FILE_MENTIONS` | 跨端宿主扩展 | ✗ 独立 | `mizar-client::install_file_mentions`（在 `agent_loop.pre_step_hook` 自动展开用户输入中的 `@` 文件/目录/图片引用并注入 `ContextNotice`） |
| `local-bash` | `PLUGIN_LOCAL_BASH` | 跨端宿主扩展 | ✗ 独立 | 本地 `!`（进上下文）/ `!!`（仅本地展示）Shell 命令模式与 `mizar-client::local_bash` 执行 |
| `approval-gate` | `PLUGIN_APPROVAL_GATE` | 跨端宿主扩展 | ✗ 独立 | `mizar-client::install_approval_gate`（`tools.pre_execute_hook` 高危工具审批拦截与决策桥接） |
| `tui` | `PLUGIN_TUI` | 终端宿主总开关 | 总开关 | 控制交互式 `mizar-tui` 工作台启用，并**级联关闭全部 6 个 `tui-*` 子扩展**（不影响 `file-mentions` / `local-bash` / `approval-gate` 等跨端核心能力） |
| `tui-emphasis` | `PLUGIN_TUI_EMPHASIS` | 终端 UI / 提示词 | ✓ 级联关闭 | `mizar-tui::install_tui_emphasis` 注入 `tui_reply_guidance` 系统提示词与 `pi-emphasis` 回复视觉语法渲染（禁用后不再向模型注入终端胶囊/图标引导规则） |
| `tui-side-panel` | `PLUGIN_TUI_SIDE_PANEL` | 终端 UI 扩展 | ✓ 级联关闭 | 右侧四页签分栏面板（`Context` / `Files` / `Tools` / `Stats`、`Ctrl+B`、分割线拖拽与侧栏/Git 采样） |
| `tui-timeline-rail` | `PLUGIN_TUI_TIMELINE_RAIL` | 终端 UI 扩展 | ✓ 级联关闭 | 2 列语义对话时间轴导轨、顶部吸顶回合提示与导轨点击跳转 |
| `tui-trajectory-view` | `PLUGIN_TUI_TRAJECTORY_VIEW` | 终端 UI 扩展 | ✓ 级联关闭 | `Ctrl+T` / `/trace` 全屏三泳道轨迹分析场景（WaveBand + Timeline/Hotspots + Inspector） |
| `tui-selection` | `PLUGIN_TUI_SELECTION` | 终端 UI 扩展 | ✓ 级联关闭 | 全屏应用级鼠标文本选区（字符/词/行粒度、多栏围栏复制） |
| `tui-clipboard` | `PLUGIN_TUI_CLIPBOARD` | 终端 UI 扩展 | ✓ 级联关闭 | `Ctrl+V` 系统剪贴板与 SSH OSC 5522 图片/文件/文本读取粘贴 |

> **Merak Desktop 直嵌说明**：桌面端 Merak 内置 Mizar 时，只需在配置中设置 `disabledPlugins: ["tui"]`，即可一键级联禁用全部 `tui` 与 `tui-*` 终端扩展（含终端 `tui_reply_guidance` 系统提示词注入），同时完整保留 `file-mentions`、`local-bash`、`approval-gate` 以及后端 `MizarRuntime` 全部核心能力。

`models.json` 为 `RouterConfig`（`providers` 映射），字段细节见 `crates/mizar-llm/src/router/config.rs` 顶部注释；`mcp.json` 使用标准 `mcpServers` 结构，并额外支持每个 Server 的 `exposure` 与 `toolExposure`。

> **注意**：`restrictToWorkspace` 默认关闭且 `TuiConfig::approval_mode` 默认为 `Auto`，即开箱状态下 `bash` / `write` / `edit` 可触达工作区外路径且无审批弹窗；需要沙箱感时请显式开启 `restrictToWorkspace` 并搭配 `--approve`（或通过 `/settings` / `/approval ask` 运行期开启审批）。
