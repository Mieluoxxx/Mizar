# Mizar 插件与模块清单

本文档介绍 `crates/` 下共 **11 个 crate** 的分层作用，以及它们在 **Cordis 运行时（`inject` / `provide` 服务）** 与 **Cargo 包依赖** 两个维度上的依赖关系（100% 无环 DAG）。

目前仓库为纯 Virtual Workspace，唯一的终端入口为 `crates/mizar-tui/src/main.rs`（`mizar-tui`），通过 `mizar-core::spine` 按需装配 Agent 能力，经跨端共享层（`mizar-types` / `mizar-client`）连接纯前端工作台 `mizar-tui-ui`（同时可供桌面端 Merak 直嵌复用）。**纯模型/纯 UI/纯客户端契约层**与**异步插件宿主层**已彻底解耦。

---

## 一、底层微内核与纯类型/UI 层（无 `tokio` / 无 HTTP / 无重运行时）

| Crate | 模块 / 定位 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- |
| **`cordis`** | 插件微内核框架 | 提供 `Context` 上下文树、`Fiber` 生命周期状态机、`effect` 记账与倒序清理、`Service` 抽象、`inject` 服务依赖注入与 5 种事件分发模式（`emit` / `parallel` / `serial` / `bail` / `waterfall`） | 无 |
| **`mizar-types`** | `llm` / `trajectory` / `client` | **纯核心契约与纯算法层**（严禁 `tokio` / `cordis` / `reqwest` / `rquickjs` / `rusqlite` / `ratatui` / `crossterm`）：<br>• `llm`：统一 `Message`、`ContentBlock`、`BlockAssembler`、`RequestOrigin`、`PromptCacheLifetime`、`ModelCostRates`、`cache_audit`（被动 `Cache Re-billed` 审计）、`LlmError` 与 `SseParser`<br>• `trajectory`：`TraceEvent`、`TrajectorySnapshot`、`fold_trajectory`、三泳道时间轴 `derive_timeline` 与异常检测 `detect_anomalies`<br>• `client`：`HostExtensionFlags`（10 个官方扩展开关与 `"tui"` 级联派生）、`PromptQueue`、`TpsLedger`、`WorkspaceFileCache`、`ClientEvent`、`mentions`、`reveal`、`tool_format` 与纯 DTO | 无 |
| **`mizar-tui-ui`** | 纯终端工作台前端 | `AppState` 状态机、`TuiFeatures` 插件特性开关、`pi-emphasis` 视觉语法、虚拟化 Markdown/代码渲染、双栏侧栏、TimelineRail、Overlay、主题与轨迹全屏视图（严禁 `tokio` / `cordis` / `reqwest` / `rquickjs` / `rusqlite`） | `mizar-types` |

---

## 二、基础能力层（传输、工具与持久化）

| Crate | 模块 | 插件 / 服务 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- |
| **`mizar-llm`** | `runtime` / `types` / `http` / `completions` / `messages` / `responses` / `router` / `retry` / `mock` | `LlmRuntime` (`"llm"`)、`LlmRouter` (`"llmRouter"`)、`LlmRetryPlugin`、`CompletionsPlugin`、`ResponsesPlugin`、`MessagesPlugin`、`MockPlugin` | 抽象 LLM 服务与 `llm/stream` / `llm/after-response` 拦截链；共享 `reqwest` 客户端与 SSE 驱动（`http`）；三大协议适配器；声明式多提供方路由与热更新（`router`）；未发可见内容前瞬时错误自动退避重试（`retry`）；离线规则假模型（`mock`） | `cordis`、`mizar-types` |
| **`mizar-tools`** | `lib` / `truncate` / `mutation_queue` / `coding` / `search` | `ToolRuntime` (`"tools"`)、`CodingToolsPlugin` (`"tools-coding"`)、`ToolSearchPlugin` (`"tool-search"`) | 工具注册表、三段执行管线（`tools/pre-execute` → `tools/execute` → `tools/post-execute`）与同路径写串行队列；编码 6 件套（`read`/`bash`/`edit`/`write`/`grep`/`find`，三档预设 `All`/`MinimalCoding`/`ReadOnly` + 纯函数 `build_tools_guide`）；BM25 `tool_search` 工具检索与按需激活 | `cordis`、`mizar-types` |
| **`mizar-session`** | `lib` / `model` / `persistence` / `sqlite` / `recall` | `SessionStore` (`"sessions"`)、`SqlitePersistencePlugin` (`"sessionPersistence"`)、`RecallPlugin` (`"recall"`) | 只追加会话事件日志（含 `Surface` / `LogOnly` 分级、`Selection` / `Usage` 账务与 `derive_messages` 推导）；嵌入式 SQLite（WAL + Single-Writer 后台线程 + FTS5 1..=3 Unicode scalar n-gram 索引）；`recall` 穿透压缩与谱系边界只读回溯工具 | `cordis`、`mizar-types`、`mizar-tools` |

### 工具分层暴露策略（`mizar-tools::ToolExposure`）

| `ToolExposure` | 直接声明给模型 | 可被 `codemode` 脚本调用 | 列在 `codemode` 描述中 | 典型使用者 |
| :--- | :--- | :--- | :--- | :--- |
| `Direct` | ✓ 激活后 | ✓ | — | `read` / `bash` / `edit` / `write` / `grep` / `find` / `skill` / `recall` |
| `ModelOnly` | ✓ 激活后 | ✗（自身是编排工具） | — | `tool_search`、`codemode`、`subagent` |
| `Codemode` | ✗（需激活） | ✓ | ✓ | MCP 工具默认、`codemode` 内部依赖的工具 |
| `Deferred` | ✗（需激活） | ✓ | ✗（靠 `tool_search` / `searchTools()` 发现） | MCP 工具可选项 |
| `Hidden` | ✗ | ✗ | ✗ | 显式屏蔽（`exclude_tools`） |

激活与排除优先级依次为（`mizar-tools` 的 `is_default_active`）：**`global_active_overrides`** > **`include_tools` 通配白名单** > **`auto_default_active`**（`codemode`/`tool_search` 按需自动启用） > 工具的 `default_active`；任何情况下 `ToolExposure::Hidden` 与命中 `exclude_tools` 的工具都不可激活。

---

## 三、重量级隔离扩展层（沙箱与外部协议）

| Crate | 插件 / 服务结构体 | 提供服务 / 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-codemode`** | `CodemodePlugin` | `"codemode"` | `codemode`（`ModelOnly`） | `["tools"]` | **基于 QuickJS（`rquickjs`）沙箱的多工具 JS 编排引擎**：单独隔离 C 依赖 `rquickjs`，把多步工具调用折叠进一次脚本执行，内建 `searchTools()`、并发信号量（默认 2）、超时取消传播（默认 120 s）与内联预算（默认 3 000 字符） | `cordis`、`mizar-types`、`mizar-tools` |
| **`mizar-mcp`** | `McpManager` | `"mcp"` | `mcp__<server>__<tool>`、`list_mcp_resources`、`list_mcp_resource_templates`、`read_mcp_resource` | `["tools"]` | **外部 MCP 客户端**：加载 `~/.mizar/mcp.json` 与 `<workspace>/.mizar/mcp.json`，支持 stdio 与 Streamable HTTP（复用 `mizar-llm::http`），按 Server 与逐工具分级暴露；不修改静态 `SystemPrompt`，按需提供 `render_servers_section()` | `cordis`、`mizar-types`、`mizar-llm`、`mizar-tools` |

---

## 四、Agent 核心领域与装配层（`mizar-core`）

`crates/mizar-core` 收拢了 Agent 全部领域循环、上下文管理与统一装配，内部按 `src/<mod>.rs` 划分 9 个模块：

| 模块 | 原对应包 / 服务名 | 关键类型 / 插件 | `inject` | 作用 |
| :--- | :--- | :--- | :--- | :--- |
| `system_prompt` | `"systemPrompt"` | `SystemPrompt`、`PromptSection` | 无 | 按 `order` 动态聚合系统提示词段落（内置 `persona` 与 `runtime_protocol`） |
| `agent_loop` | `"agentLoop"` | `AgentLoop`、`AgentEvent`、`SessionInbox`、`TurnInput` | `["llm", "tools", "sessions", "systemPrompt"]` | 托管 `SessionInbox`（`Steer`/`FollowUp`/`Inject`），串联 `agent/pre-step` → 冻结 `GenerateOptions` 调模型 → 工具并发/串行管线 → `agent/turn-stopping`，通过 `AgentEventBus`（`subscribe` / `on_event`）广播强类型 `AgentEvent` |
| `compaction` | `"compaction"` | `CompactionRuntime` | `["llm", "sessions", "tools", "systemPrompt"]` | 单阶段检查点压缩（真实可用窗口 80% 触发、`<mizar-digest>` 硬事实、膨胀保护、溢出 `retain=0` 恢复）与压后写文件/读去重守卫 |
| `instructions` | `"agentInstructions"` | `AgentInstructions` | 无（事件订阅） | 扫描 `~/.mizar/`、`~/.agents/`、项目根到工作区祖先链及触达子目录的 `AGENTS.md` / `CLAUDE.md`，去重裁切并在 `agent/pre-step` 注入 `ContextNotice` |
| `skills` | `"skills"` / `"skills-plugin"` | `SkillRegistry`、`SkillsPlugin` | `["tools", "skills"]` | 分层发现 `SKILL.md`、注册 `skill` 工具、目录注入与写入命中热失效 |
| `trajectory` | `"trajectory"` / `"repeat-tool-guard"` | `TrajectoryRuntime`、`RepeatToolGuardPlugin` | 无（事件订阅） | 实时采集 `TraceEvent`、异常检测、`trajectory.jsonl` 导出与连续重复失败工具调用拦截 |
| `session_title` | `"sessionTitle"` | `SessionTitleRuntime` | `["llm", "sessions"]` | 异步生成与刷新 `MMDD｜TYPE｜Topic` 会话标题，尊重 `/rename` 人工所有权，自动跳过子代理 |
| `subagent` | `"subagent"` | `SubagentRuntime`、`SubagentPlugin` | `["tools", "sessions", "agentLoop", "systemPrompt"]` | UNIX 进程式单层后台常驻分身 `subagent`（`ModelOnly`），支持 `spawn`/`send`/`status`/`stop`、零 Token 活性看门狗与深度 1 护栏 |
| `spine` | `"spinePlugins"` / `"agent-spine"` | `AgentSpinePlugin`、`SpinePlugins`、`ToolsGuidePromptPlugin`、`LoadedMizarConfig` | 装配根 | 静态挂载核心服务与 12 个内置可选模块，通过轻量原子开关与 `ToolRuntime` 分组屏蔽控制运行期启停（`mcp` 启动禁用时延迟初始化；Gate 2 核心自保护、`tools_guide` `order=20` 提示词段落挂载、识别 `KNOWN_HOST_EXTENSION_IDS` 10 个官方宿主扩展 ID），并承载 `settings.json` / `models.json` / CLI 路由解析 |

**内部 Crate 依赖**：`cordis`、`mizar-types`、`mizar-llm`、`mizar-tools`、`mizar-session`，以及按 Cargo feature 选配的 `mizar-codemode`、`mizar-mcp`。

### Cargo Features 清单

| Crate | Feature | 默认开启 | 作用 |
| :--- | :--- | :--- | :--- |
| `mizar-core` | `codemode` | ✓ | 启用 `mizar-codemode` QuickJS 沙箱模块（`AgentToolsMode::Coding::codemode`） |
| `mizar-core` | `mcp` | ✓ | 启用 `mizar-mcp` 外部 MCP 客户端模块（`AgentToolsMode::Coding::mcp`） |
| `mizar-session` | `bundled` | ✓ | 启用 `rusqlite/bundled` 内置编译 SQLite（关闭后可链接系统 SQLite） |

---

## 五、宿主接入与终端入口层

| Crate | 入口 / 结构体 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- |
| **`mizar-client`** | `FileMentionsPlugin`、`ApprovalGatePlugin`、宿主运行时模块 | **跨端宿主运行时（严禁引入 `ratatui` / `crossterm` / `mizar-tui-ui`）**：供 `mizar-tui` 与桌面端 Merak 共用。包含 `@` 引用展开（`mentions`）、高危审批门禁（`approval`）、图片压缩与 EXIF 校正（`images`）、会话投影与选择优先级解析（`session_view`）、设置热写回（`settings`）、本地 Shell（`local_bash`）、事件桥与代次控制（`events`）及工作区面板快照（`workspace`） | `cordis`、`mizar-types`、`mizar-llm`、`mizar-tools`、`mizar-session`、`mizar-mcp`、`mizar-core` |
| **`mizar-tui`** | `mizar` 二进制、`TuiEmphasisPlugin` | **薄终端宿主适配器**：解析 CLI 参数与 `mizar exec` 无头单次执行，装配 `AgentSpinePlugin`，将事件订阅、审批、本地 Bash、会话投影与配置热写回委托给 `mizar-client`，自身专注终端 I/O、剪贴板、`mizar-tui-ui` 渲染桥接及按 `HostExtensionFlags` 挂载终端扩展 | `cordis`、`mizar-types`、`mizar-llm`、`mizar-tools`、`mizar-session`、`mizar-mcp`、`mizar-core`、`mizar-client`、`mizar-tui-ui` |

---

## 六、依赖拓扑简图（11-Crate 无环 DAG）

```text
┌─ 终端入口与跨端宿主层 ──────────────────────────────────────────────────────────┐
│  mizar-tui (CLI + exec + 终端桥接)              Merak Desktop（桌面端直嵌）     │
│    ├─► mizar-tui-ui (纯终端 UI：零 tokio/cordis)      │                         │
│    │     └────────────────────────────────────────────┼──► mizar-types          │
│    └───────────────────────┬──────────────────────────┘                         │
│                            v                                                    │
│  mizar-client（前端无关宿主运行时：零 ratatui / crossterm / mizar-tui-ui）      │
│    ├─► mizar-core（agent_loop / system_prompt / compaction / instructions /     │
│    │               skills / trajectory / session_title / subagent / spine）     │
└────┴───────────────────────┬──────────────────────────┬─────────────────────────┘
                             │                          │ (optional features)
                             v                          v
┌─ 会话/工具/隔离扩展层 ─────┴──────────┐    ┌──────────┴─────────────────────────┐
│  mizar-session (sessions + sqlite     │    │  mizar-codemode (rquickjs 隔离)    │
│                 + recall)             │    │  mizar-mcp (外部 stdio/HTTP MCP)   │
└────────────────┬──────────────────────┘    └──────────┬─────────────────────────┘
                 │                                      │
                 v                                      v
┌─ 协议与工具基础层 ──────────────────────────────────────────────────────────────┐
│  mizar-llm (runtime + http + completions/messages/responses + router/retry/mock)│
│  mizar-tools (tools runtime + mutation_queue + coding 6 tools + tool_search)    │
└───────────────────────────────┬──────────────────────┬──────────────────────────┘
                                v                      v
     [ mizar-types（纯契约/轨迹模型/客户端算法：零运行时、零内部依赖） ]
     [ cordis（微内核：Context / Fiber / effect / 五种事件分发） ]
```

**编译隔离边界**：
- `mizar-types` 与 `mizar-tui-ui` 均不依赖 `tokio`、`cordis`、`reqwest`、`rquickjs` 或 `rusqlite`（其中 `mizar-types` 零内部 crate 依赖且亦禁止 `ratatui` / `crossterm`）。
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
| `instructions` | object | `enabled` / `candidates` / `localCandidates` / `projectRootMarkers` / `maxBytes` / `maxSourceBytes` / `followSubdirs` |
| `compaction` | object | `enabled` / `autoCompact` / `defaultContextWindow` / `maxContextTokens` / `triggerRatio` / `keepRecentTokens` |
| `recall` | object | `enabled` / `defaultSearchLimit`（默认 `8`）/ `readCharLimit`（默认 `6000`） |
| `sessionTitle` | object | `enabled`（默认 `true`）/ `model`（可选 `provider/model`，省略则跟随当前会话模型）/ `thinkingLevel`（默认 `"low"`） |
| `subagent` | object | `maxSteps`（默认 `20`）/ `maxStepsCap`（默认 `40`）/ `timeoutSecs`（默认 `180`，最小钳位 `1`）/ `maxConcurrent`（默认 `4`）/ `stallTimeoutSecs`（默认 `45`，设为 `0` 关闭看门狗）/ `maxWatchdogAlerts`（默认 `2`） |
| `ui` | object | `theme` / `screenMode` / `showThinking` / `verboseExpanded` / `sidePanelOpen` / `mouseCapture` / `approvalMode`（可通过 TUI `/settings` 交互修改并自动保存） |

### 客户端宿主与 TUI 扩展模块 ID（`disabledPlugins`）

`mizar-core::spine` 在 `KNOWN_HOST_EXTENSION_IDS` 中登记了以下 **10 个官方客户端宿主与 TUI 扩展模块 ID**。`SpinePlugins::initialize` 会将其识别为合法已知扩展（不触发未知模块告警或启动错误），由 `mizar-types::client::HostExtensionFlags::from_disabled_plugins`、`mizar-tui-ui::TuiFeatures` 及客户端宿主按开关控制挂载：

| 扩展 ID | 常量名（`mizar-types::client`） | 作用域 | 受 `"tui"` 级联 | 控制的能力与禁用效果 |
| :--- | :--- | :--- | :--- | :--- |
| `file-mentions` | `PLUGIN_FILE_MENTIONS` | 跨端宿主扩展 | ✗ 独立 | `mizar-client::FileMentionsPlugin`（在 `agent/pre-step` 自动展开用户输入中的 `@` 文件/目录/图片引用并注入 `ContextNotice`） |
| `local-bash` | `PLUGIN_LOCAL_BASH` | 跨端宿主扩展 | ✗ 独立 | 本地 `!`（进上下文）/ `!!`（仅本地展示）Shell 命令模式与 `mizar-client::local_bash` 执行 |
| `approval-gate` | `PLUGIN_APPROVAL_GATE` | 跨端宿主扩展 | ✗ 独立 | `mizar-client::ApprovalGatePlugin`（`tools/pre-execute` 高危工具审批拦截与决策桥接） |
| `tui` | `PLUGIN_TUI` | 终端宿主总开关 | 总开关 | 控制交互式 `mizar-tui` 工作台启用，并**级联关闭全部 6 个 `tui-*` 子扩展**（不影响 `file-mentions` / `local-bash` / `approval-gate` 等跨端核心能力） |
| `tui-emphasis` | `PLUGIN_TUI_EMPHASIS` | 终端 UI / 提示词 | ✓ 级联关闭 | `mizar-tui::TuiEmphasisPlugin` 注入 `tui_reply_guidance` 系统提示词与 `pi-emphasis` 回复视觉语法渲染（禁用后不再向模型注入终端胶囊/图标引导规则） |
| `tui-side-panel` | `PLUGIN_TUI_SIDE_PANEL` | 终端 UI 扩展 | ✓ 级联关闭 | 右侧四页签分栏面板（`Context` / `Files` / `Tools` / `Stats`、`Ctrl+B`、分割线拖拽与侧栏/Git 采样） |
| `tui-timeline-rail` | `PLUGIN_TUI_TIMELINE_RAIL` | 终端 UI 扩展 | ✓ 级联关闭 | 2 列语义对话时间轴导轨、顶部吸顶回合提示与导轨点击跳转 |
| `tui-trajectory-view` | `PLUGIN_TUI_TRAJECTORY_VIEW` | 终端 UI 扩展 | ✓ 级联关闭 | `Ctrl+T` / `/trace` 全屏三泳道轨迹分析场景（WaveBand + Timeline/Hotspots + Inspector） |
| `tui-selection` | `PLUGIN_TUI_SELECTION` | 终端 UI 扩展 | ✓ 级联关闭 | 全屏应用级鼠标文本选区（字符/词/行粒度、多栏围栏复制） |
| `tui-clipboard` | `PLUGIN_TUI_CLIPBOARD` | 终端 UI 扩展 | ✓ 级联关闭 | `Ctrl+V` 系统剪贴板与 SSH OSC 5522 图片/文件/文本读取粘贴 |

> **Merak Desktop 直嵌说明**：桌面端 Merak 内置 Mizar 时，只需在配置中设置 `disabledPlugins: ["tui"]`，即可一键级联禁用全部 `tui` 与 `tui-*` 终端扩展（含终端 `tui_reply_guidance` 系统提示词注入），同时完整保留 `file-mentions`、`local-bash`、`approval-gate` 以及后端 `AgentSpinePlugin` 全部核心能力。

`models.json` 为 `RouterConfig`（`providers` 映射），字段细节见 `crates/mizar-llm/src/router/config.rs` 顶部注释；`mcp.json` 使用标准 `mcpServers` 结构，并额外支持每个 Server 的 `exposure` 与 `toolExposure`。

> **注意**：`restrictToWorkspace` 默认关闭且 `TuiConfig::approval_mode` 默认为 `Auto`，即开箱状态下 `bash` / `write` / `edit` 可触达工作区外路径且无审批弹窗；需要沙箱感时请显式开启 `restrictToWorkspace` 并搭配 `--approve`（或通过 `/settings` / `/approval ask` 运行期开启审批）。
