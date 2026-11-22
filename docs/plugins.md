# Mizar 插件与模块清单

本文档介绍 `crates/` 下共 **35 个 crate** 的分层作用，以及它们在 **Cordis 运行时（`inject` / `provide` 服务）** 与 **Cargo 包依赖** 两个维度上的依赖关系。

目前仓库为纯 Virtual Workspace，唯一的终端入口为 `crates/mizar-tui/src/main.rs`（`mizar-tui`），通过 `mizar-agent-spine` 按需装配 Agent 能力，经跨端共享层（`mizar-client-core` / `mizar-client`）连接纯前端工作台 `mizar-tui-ui`（同时可供桌面端 Merak 直嵌复用）。**纯模型/纯 UI/纯客户端契约层**与**异步插件宿主层**已彻底解耦。

---

## 一、底层微内核与纯类型/核心层（无 `tokio` / 无 HTTP / 无重运行时）

以下 5 个底层 crate 只保留数据结构、纯逻辑或微内核抽象，可独立进行毫秒级 `cargo check` 与单测：

| Crate | 类型 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- |
| **`cordis`** | 插件微内核框架 | 提供 `Context` 上下文树、`Fiber` 生命周期状态机、`effect` 记账与倒序清理、`Service` 抽象、`inject` 服务依赖注入与 5 种事件分发模式（`emit` / `parallel` / `serial` / `bail` / `waterfall`） | 无 |
| **`mizar-llm-types`** | 纯 Schema / SSE Core | 定义统一消息、`ContentBlock`、`BlockAssembler`、`RequestOrigin`、`PromptCacheLifetime`、`ModelCostRates`、`CacheWarmingMode`、错误分类与流式 `SseParser`（零 `tokio` / `cordis` / `reqwest`） | 无 |
| **`mizar-trajectory-core`** | 纯轨迹建模与检测 | 轨迹事件模型、折叠快照、三泳道时间轴建模与异常检测规则（零 `tokio` / `cordis` / 运行时插件依赖） | `mizar-llm-types` |
| **`mizar-client-core`** | 纯客户端契约与算法层 | 跨 TUI 与 Merak Desktop 共享的纯数据结构与算法：`HostExtensionFlags`（10 个官方扩展开关与 `"tui"` 级联派生）、`PromptQueue`（ID 稳定的待发送队列）、`TpsLedger`（解码吞吐账本）、`WorkspaceFileCache`（`.gitignore` 感知后台索引）、`cache_audit`（`CacheWarmingSessionStatus` / `CacheMissAudit` / `Cache Re-billed` 审计）、`ClientEvent` 统一事件、`mentions` 解析、`reveal` 平滑吐字、`tool_format` 摘要与纯 DTO（零 `tokio` / `cordis` / `reqwest` / `rquickjs` / `rusqlite` / `ratatui` / `crossterm`） | `mizar-llm-types` |
| **`mizar-tui-ui`** | 纯终端工作台前端 | `AppState` 状态机、`TuiFeatures` 插件特性开关、`pi-emphasis` 视觉语法、虚拟化 Markdown/代码渲染、双栏侧栏、TimelineRail、Overlay、主题与轨迹全屏视图（零 `tokio` / `cordis` / `reqwest` / `rquickjs` / `rusqlite`） | `mizar-client-core`、`mizar-llm-types`、`mizar-trajectory-core` |

---

## 二、核心基础服务插件（向 `Context` 提供服务）

以下 4 个插件实现了 `cordis::Service`，负责向容器注册核心能力，均不依赖其他运行时服务：

| Crate | 插件 / 服务结构体 | 提供服务名 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-llm`** | `LlmRuntime` | `"llm"` | 抽象 LLM 服务：重导出 `mizar-llm-types`，提供 `normalize_options` 前置规整、不可变快照 `llm/stream` 拦截链、流尾只读回执 `llm/after-response`、纯 `cache_warming` 数学策略与协议适配器管理（无内置 HTTP 栈） | 无 | `cordis`、`mizar-llm-types` |
| **`mizar-session`** | `SessionStore` | `"sessions"` | 会话事件日志管理：以只追加日志保存对话、工具调用、`LogOnly` `Usage` 账务记录与回放状态，推导出发往模型的消息列表，并定义 `SessionPersistence` 持久化 trait 占位 `"sessionPersistence"` | 无 | `cordis`、`mizar-llm` |
| **`mizar-system-prompt`** | `SystemPrompt` | `"systemPrompt"` | 系统提示词动态组装：支持多插件按 `order` 贡献段落并自动拼接（内置 `persona` 与当前日期） | 无 | `cordis` |
| **`mizar-tools`** | `ToolRuntime` | `"tools"` | 工具注册表与三段执行管线（`tools/pre-execute` → `tools/execute` → `tools/post-execute`），生成工具 Schema 并统一派发执行；定义 `ToolExposure` / `ToolExecutionMode` / `ToolAnnotations` | 无 | `cordis`、`mizar-llm` |

---

## 三、LLM HTTP 客户端、协议适配器与路由插件

HTTP 传输实现收敛在 `mizar-llm-http`，其余协议与路由插件运行时均注入 `"llm"` 服务：

| Crate | 插件 / 服务结构体 | 插件名 / 提供服务 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-llm-http`** | 工具库（非插件） | — | 共享异步 `reqwest` 客户端、超时控制与 SSE 流驱动封装（隔离重型 TLS/HTTP 依赖图） | — | `mizar-llm` |
| **`mizar-llm-completions`** | `CompletionsPlugin` | `"llm-completions"` | **OpenAI Chat Completions 兼容协议适配器**：支持 OpenAI、DeepSeek、OpenRouter、Qwen、vLLM 等端点及 `CompletionsCompat` 方言开关（思考格式、`max_tokens` 字段等） | `["llm"]` | `cordis`、`mizar-llm`、`mizar-llm-http` |
| **`mizar-llm-responses`** | `ResponsesPlugin` | `"llm-responses"` | **OpenAI Responses 协议适配器**：对接 `/responses` 接口，将 `System` 转为 `instructions` 并解析文本、思考摘要、函数调用与 `usage` 流式事件 | `["llm"]` | `cordis`、`mizar-llm`、`mizar-llm-http` |
| **`mizar-llm-messages`** | `MessagesPlugin` | `"llm-messages"` | **Anthropic Messages 协议适配器**：对接 `/v1/messages`，自动处理 `system` 抽取、`cache_control`（5m / 1h）断点透传、连续 `tool_result` 合并、`thinking` 签名与前缀摘要校验 | `["llm"]` | `cordis`、`mizar-llm`、`mizar-llm-http` |
| **`mizar-llm-mock`** | `MockPlugin` | `"llm-mock"` | **离线假模型适配器**：无需 API Key，按关键词规则模拟 `get_time` / `list_dir` 工具调用与逐字流式输出，用于演示与无网测试 | `["llm"]` | `cordis`、`mizar-llm` |
| **`mizar-llm-retry`** | `LlmRetryPlugin` | `"llm-retry"` | **流式自动重试中间件**：拦截 `llm/stream` waterfall，在未发出可见内容前对 `RateLimit` / `Server` / `Timeout` / `Transport` 等瞬时故障按退避策略重试 | `["llm"]` | `cordis`、`mizar-llm` |
| **`mizar-llm-router`** | `LlmRouter` | `"llmRouter"`（服务） | **多提供方声明式路由与热重载插件**：将多路由配置（`~/.mizar/models.json`）映射到上述协议适配器，支持 `protocol` 选择、`api_key_env` 与 `$VAR` / `${VAR}` / `!command` 动态解析、`models` / `modelOverrides` 目录、请求头扩展钩子，并暴露 `apply_config` 原子热更新接口（**当前无内置文件监听**，需宿主在配置变更时主动调用） | `["llm"]` | `cordis`、`mizar-llm`、`mizar-llm-http`、`mizar-llm-completions`、`mizar-llm-responses`、`mizar-llm-messages`、`mizar-llm-mock` |

---

## 四、工具插件

以下插件中，4.1–4.5 及 4.7 的 `mizar-recall` 向 `"tools"` 运行时注册可执行工具并注入 `"tools"` 服务；**4.6 `mizar-instructions` 与 4.7 `mizar-compaction` 是例外**：前者不注册任何工具、仅订阅事件注入 `ContextNotice`，后者提供 `"compaction"` 服务并通过步前与工具管线执行上下文压缩与压后护栏。

工具的可达性由 `mizar-tools` 的 **`ToolExposure`** 统一控制：

| `ToolExposure` | 直接声明给模型 | 可被 `codemode` 脚本调用 | 列在 `codemode` 描述中 | 典型使用者 |
| :--- | :--- | :--- | :--- | :--- |
| `Direct` | ✓ 激活后 | ✓ | — | `read` / `bash` / `edit` / `write` / `grep` / `find` / `skill` |
| `ModelOnly` | ✓ 激活后 | ✗（自身是编排工具） | — | `tool_search`、`codemode` |
| `Codemode` | ✗（需激活） | ✓ | ✓ | MCP 工具默认、`codemode` 内部依赖的工具 |
| `Deferred` | ✗（需激活） | ✓ | ✗（靠 `tool_search` / `searchTools()` 发现） | MCP 工具可选项 |
| `Hidden` | ✗ | ✗ | ✗ | 显式屏蔽（`exclude_tools`） |

激活与排除的优先级依次为（`mizar-tools` 的 `is_default_active`）：**`global_active_overrides`** > **`include_tools` 通配白名单** > **`auto_default_active`**（`codemode`/`tool_search` 按需自动启用） > 工具的 `default_active`；任何情况下 `ToolExposure::Hidden` 与命中 `exclude_tools` 的工具都不可激活。显式激活过的工具会在下一次模型调用时重新参与声明，而 `Codemode` / `Deferred` 工具即使未激活也仍可被脚本或 `tool_search` 调用（`resolve_callable_infos`）。

### 4.1 最小只读套件（Legacy 模式）

| Crate | 插件结构体 | 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-tool-time`** | `ToolTimePlugin` | `"tool-time"` | `get_time` | `["tools"]` | 返回格式化的本地当前日期与时间 | `cordis`、`mizar-tools` |
| **`mizar-tool-fs`** | `ToolFsPlugin` | `"tool-fs"` | `list_dir`、`read_file` | `["tools"]` | 只读文件工具，内建路径防逃逸检查（限制在 `root` 目录内）、UTF-8 安全字节截断（`max_bytes` 默认 20 000） | `cordis`、`mizar-tools` |

### 4.2 核心编码工具集

| Crate | 插件结构体 | 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-tools-coding`** | `CodingToolsPlugin` | `"tools-coding"` | `read`、`write`、`edit`、`bash`、`grep`、`find` | `["tools"]` | 编码工具体系：`read` 支持行偏移与长度限制，`edit` 做词级 diff 替换，`grep` / `find` 走 `ignore` 库（尊重 `.gitignore`），`bash` 支持超时与输出截断。三档预设：`All`（6 个）、`MinimalCoding`（4 个：`read`/`bash`/`edit`/`write`）、`ReadOnly`（3 个：`read`/`grep`/`find`）；可选向 `SystemPrompt` 贡献「可用工具 + 工作目录」段落 | `cordis`、`mizar-system-prompt`、`mizar-tools` |

### 4.3 工具发现与 JS 编排

| Crate | 插件结构体 | 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-tool-search`** | `ToolSearchPlugin` | `"tool-search"` | `tool_search` | `["tools"]` | **BM25 工具发现**：对 `Codemode` / `Deferred` 暴露等级的工具建索引（名称权重 4.0、命名空间 2.0），命中后按会话（或全局）激活，使其出现在下一次模型调用的声明中。存在可搜索工具时自动激活，`default_limit` 默认 8 | `cordis`、`mizar-tools` |
| **`mizar-codemode`** | `CodemodePlugin` | `"codemode"` | `codemode` | `["tools"]` | **基于 QuickJS（`rquickjs`）沙箱的多工具 JS 编排引擎**：把多个工具调用折叠成一次脚本执行，中间结果不进上下文。提供 `searchTools()` 内建检索、并发信号量（`max_workers` 默认 2）、脚本超时（默认 120 s，支持跨异步嵌套工具调用取消传播）与内联预算（默认 3 000 字符，`DEFAULT_CODEMODE_INLINE_BUDGET`）；`only_mode` 可隐藏 `Direct` 工具的直接声明，全部改由脚本调度 | `cordis`、`mizar-tool-search`、`mizar-tools` |

### 4.4 MCP 接入

| Crate | 服务结构体 | 提供服务名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-mcp`** | `McpManager` | `"mcp"` | `mcp__<server>__<tool>`、`list_mcp_resources`、`list_mcp_resource_templates`、`read_mcp_resource` | `["tools"]` | **MCP 客户端**：从 `~/.mizar/mcp.json` 与 `<workspace>/.mizar/mcp.json` 加载配置（项目级省略 `command`/`url` 的同名条目作为增量补丁覆盖 `exposure` / `toolExposure` 等字段），支持 **stdio**（`command` / `args` / `env` / `cwd`）与 **Streamable HTTP**（`url` / `headers`，复用 `mizar-llm-http` 连接池与自定义 CA 根，支持 `text/event-stream` 增量分块读取）两种传输；按 Server 与逐工具两级配置暴露等级（`McpExposure` 默认 `Codemode`，支持通配符匹配），后台异步连接不阻塞启动；`close_all` 通过 `McpManagerPhase` 状态机、`CancellationToken` 与传输层 `StdioRpcState` 安全退出并唤醒等待者；向 `SystemPrompt` 贡献 `<mcp_servers>` 段落 | `cordis`、`mizar-llm`、`mizar-llm-http`、`mizar-system-prompt`、`mizar-tools` |

### 4.5 技能系统

| Crate | 服务 / 插件结构体 | 提供服务名 / 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-skills`** | `SkillRegistry` + `SkillsPlugin` | `"skills"` / `"skills-plugin"` | `skill` | `["tools", "skills"]` | **渐进式技能发现与按需加载**：扫描分层技能根（项目 `.agents/skills` → 自定义目录 → 用户 `~/.agents/skills` → 内置包），解析 `SKILL.md` frontmatter，按 rank 升序仲裁同名技能；通过合成消息注入技能目录并支持热失效——监听 `write` / `edit` 写入命中技能目录时自动刷新缓存，广播 `skills/change` | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-tools` |

### 4.6 工作区指令系统

| Crate | 服务结构体 | 提供服务名 / 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-instructions`** | `AgentInstructions` | `"agentInstructions"` / `"agent-instructions"` | —（不注册工具） | 无（事件订阅 `tools/result` + `agent/pre-step`） | **多层级 `AGENTS.md` / `CLAUDE.md` 自动加载与热同步**：扫描 `~/.mizar/`（`MIZAR_HOME`）与 `~/.agents/`（`MIZAR_AGENTS_HOME`）、由 `.git` 项目根到工作区的祖先链，以及 `read`/`write`/`edit` 触达的子目录（`follow_subdirs`）；默认候选 `AGENTS.md` / `CLAUDE.md` 与 `.local.md` 变体。同组按去空白 SHA-256 摘要去重、不同内容并存；超预算时**优先淘汰最宽泛的祖先**以保住近目录指令，并在正文做 UTF-8 安全截断。默认 `max_bytes` 65 536、`max_source_bytes` 1 MiB（超出即跳过该文件）；通过 `agent/pre-step` 注入 `ContextNotice`，文件增删改时发增量提示，并在历史压缩后自动重建基线 | `cordis`、`mizar-agent-loop`、`mizar-session`、`mizar-tools` |

### 4.7 上下文压缩与无损情景回溯

| Crate | 服务 / 插件结构体 | 提供服务名 / 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-compaction`** | `CompactionRuntime` | `"compaction"` / `"CompactionRuntime"` | —（不注册工具） | `["llm", "sessions", "tools", "systemPrompt"]` | **渐进上下文压缩与压后工具护栏**：步前水位估算（含 380k 软顶、regrowth 防抖与 300s 冷缓存机会窗）→ L1 零模型批量 `ToolResultPrune` 剪枝锁存 → L2 软硬分离检查点摘要（确定性 `<mizar-digest>` + 同前缀尾置指令 + `reasoning: off` + `<= 0.5*B` Re-plan + `answeredRoute`）→ L3 溢出与隐性连续截断（`cutOffs >= 2`）自动恢复；挂载 `tools/pre-execute`（压后拦陈旧整文件 `write` 放行 `edit`）与 `tools/post-execute`（跨压缩感知 `read-dedup`）；暴露 `compact_now` 供 TUI `/compact [focus]` 调用，且在 `AgentSpinePlugin` 中先于 `instructions` / `skills` / `recall` 装配以便压缩后同步触发重注与激活 | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-system-prompt`、`mizar-tools` |
| **`mizar-recall`** | `RecallPlugin` | `"recall"` | `recall_search`、`recall_read` | `["sessions", "tools"]` | **无损情景回溯工具**：默认以 `ToolExposure::Deferred` 零开销挂载，一旦本会话发生首次剪枝/压缩或处于 `parent_id` 谱系中即自动激活为模型可见；穿透压缩边界检索 SQLite FTS5 1..=3 n-gram 索引（无持久化时自动退回内存扫描），支持 `current` / `lineage` / `workspace` 作用域校验、`is_current_session` 标记、字符分页与 `to_file` 旁路导出 | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-tools` |

---

### 4.8 单层子代理

| Crate | 服务 / 插件结构体 | 提供服务名 / 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-subagent`** | `SubagentRuntime` + `SubagentPlugin` | `"subagent"` / `"SubagentRuntime"` | `subagent`（`ModelOnly`） | `["tools", "sessions", "agentLoop", "systemPrompt"]` | **UNIX 进程式单层后台常驻分身**：向模型暴露窄接口工具 `subagent(action?, prompt?, mode?, model?, name?)`（`spawn` 默认 / `send` / `status` / `stop`，`ToolExposure::ModelOnly` 禁止 `codemode` 脚本套娃）；默认 `tokio::spawn` 后台异步执行（并发上限 4），支持 `send` 对运行中分身步间 `Steer` 插话纠偏或唤醒已休眠 SQLite 子会话继续执行（锁定原权限模式防提权），并引入 `active_run_id` 单调代次与绑定代次的 `ChildSessionGuard` 消除停止过渡并发越权；内嵌**零 Token Rust 活性看门狗**（订阅 `agent/*` 刷新实时步数/工具/静默时长，超时向父 `SessionInbox` 投递可合并告警并保留最后 1 格自动唤醒预算给结算回执）；超限异常与超时自动宽限 200ms 并修复内存悬挂 `ToolCall`；并行完成回执合并批醒 `subagent:batch`；`mizar exec` 无头模式自动关闭本插件（由 `codemode` 承担确定性并发）；强制深度 1（子会话剔除 `subagent` 并经 `tools/pre-execute(-100)` 守卫拦截）、`RequestOrigin::Subagent` 与显式 `SessionOrigin::Subagent` 身份隔离 | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-system-prompt`、`mizar-tools` |

---

## 五、会话持久化、标题与缓存保活

| Crate | 插件结构体 | 插件名 | 提供服务名 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-session-sqlite`** | `SqlitePersistencePlugin` | `"session-persistence-sqlite"` | `"sessionPersistence"` | 无 | **嵌入式 SQLite 持久化 + FTS5 情景索引**：WAL 模式 + `busy_timeout`，独占 **Single-Writer 后台线程**通过批量事务提交把写入隔离在 Tokio 调度器外；只读连接加 `query_only`；内建无内容 FTS5 1..=3 Unicode scalar n-gram 索引（`session_recall_docs` + `session_recall_fts`），供 `mizar-recall` 毫秒级穿透检索中英文与代码标识符；支持关闭 `bundled` feature 以链接系统 SQLite。默认路径 `~/.mizar/sessions.sqlite`（`MIZAR_SESSION_DB` 可覆盖，设为 `none`/`off` 关闭落盘） | `cordis`、`mizar-llm-types`、`mizar-session` |
| **`mizar-session-title`** | `SessionTitleRuntime` | `"session-title"` | `"sessionTitle"` | `["llm", "sessions"]` | **后台异步会话标题自动生成**：订阅 `agent/turn-input`（仅提取 `SessionEvent::User` 消息，通过 `SessionHeader::is_subagent_child()` 自动跳过子代理会话），首轮输入及每隔 `refresh_interval_turns`（默认 20）轮异步生成 `MMDD｜TYPE｜Topic` 标题（`FEA/DES/FIX/OPT/REL/EXP/DOC/RES`）；自动跳过 `Offline Mock` 并尊重人工 `/rename` 所有权（`TitleSource::Manual`）与并发改名保护；失败静默不重试并保留 `derive_session_title` 兜底清洗标题；提供 `regenerate_now` 供 `/rename` 无参手动重生成 | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session` |
| **`mizar-cache-warming`** | `CacheWarmingRuntime` | `"cacheWarming"` / `"CacheWarmingRuntime"` | `"cacheWarming"` | `["llm", "sessions"]` | **Prompt Cache 后台定时续命保活**：监听 `llm/after-response`（仅 `RequestOrigin::SessionTurn`）与 `agent/turn-start`/`end`，在四道硬门禁（模型显式缓存与重放安全、`E[savings] >= $0.05`、60m/30m 上限与休眠截止线、禁用 `mizar exec` 与子代理 `idle`）通过后，通过 `stream_direct(max_tokens=1, disable_continuation=true)` 保活；成功后写入 `SessionEvent::Usage("cache_warm")` 并动态查找可选 `"compaction"` 服务刷新冷缓存活跃时间戳 | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-client-core`（可选 `mizar-compaction`） |

---

## 六、Agent 编排与观测

| Crate | 插件 / 服务结构体 | 插件名 / 提供服务 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-agent-loop`** | `AgentLoop` | `"AgentLoop"` / 提供 `"agentLoop"` | **Agent 主循环**：托管 `TurnInput` 与纯内存 `SessionInbox`（`Steer`/`FollowUp`/`Inject` + `auto_wake_limit`），串联「`agent/pre-step` (`waterfall`) 准入落盘 → 拼装提示词与历史 → 冻结 `Arc<GenerateOptions>` 流式调模型（含 `agent/request-error` `waterfall` 自愈）→ 工具管线 → 无工具时 `agent/turn-stopping` (`serial`) 数据驱动检查」，广播全套 `agent/*` 流式事件；`max_steps` 默认 `None` 不限步数 | `["llm", "tools", "sessions", "systemPrompt"]` | `cordis`、`mizar-llm`、`mizar-session`、`mizar-system-prompt`、`mizar-tools` |
| **`mizar-trajectory`** | `TrajectoryRuntime` + `RepeatToolGuardPlugin` | `"trajectory"`（服务）/ `"repeat-tool-guard"` | **轨迹观测运行时**：基于 `mizar-trajectory-core` 模型与检测器提供 `"trajectory"` 服务，订阅会话与 `agent/*` 事件实时采集 `TraceEvent`（广播 `trajectory/anomaly`），可选落盘 `{export_dir}/{session_id}.jsonl`；`RepeatToolGuardPlugin` 在 `tools/post-execute` waterfall 上捕捉连续重复调用并追加提醒或强制拦截 | 无（全部通过事件订阅接入） | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-tools`、`mizar-trajectory-core` |

---

## 七、装配与接入层

| Crate | 插件 / 入口 | 插件名 / 形态 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-agent-spine`** | `AgentSpinePlugin` / `SpinePlugins` | `"agent-spine"` / 提供服务 `"spinePlugins"` | **核心装配套件与模块控制器**：先挂载核心内核服务（含按 `sqlite` feature 与 `session_db_path` 挂载的 `sessionPersistence`），再按顺序通过 `SpinePlugins`（`SpinePluginsContextExt`）管理 **15 个内置可选模块**（`compaction`、`tool-time`、`tool-fs`、`tools-coding`、`tool-search`、`codemode`、`mcp`、`skills`、`instructions`、`recall`、`session-title`、`cache-warming`、`subagent`、`trajectory`、`repeat-guard`）的 `Fiber` 生命周期与运行期热启停（`set_enabled`）；对 `llm` / `sessions` / `tools` / `systemPrompt` / `agentLoop` / `spinePlugins` 实施 Gate 2 自保护（严禁禁用），同时识别 `KNOWN_HOST_EXTENSION_IDS` 10 个官方宿主/TUI 扩展 ID 以避免 `disabledPlugins` 误报未知模块警告。支持 Cargo features（`legacy` / `coding` / `compaction` / `sqlite` / `session-title` / `cache-warming` / `subagent` / `trajectory` / `router`）裁剪（`--no-default-features` 仅拉起核心服务），并将配置加载与路由解析拆分为 `config.rs` + `config/{settings,route}.rs`（提供 `load_mizar_config` / `patch_settings_file` / `persist_disabled_plugins` 原子读写） | 无（自身负责装配与控制） | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-system-prompt`、`mizar-tools` 及按 Cargo features 选配的后端 crate |
| **`mizar-client`** | `FileMentionsPlugin`、`ApprovalGatePlugin` 与宿主运行时模块 | `"file-mentions"` / `"approval-gate"`（前端无关宿主运行时） | **跨端宿主运行时（零 `ratatui` / `crossterm` / `mizar-tui-ui`）**：供 `mizar-tui` 与未来桌面端 Merak 共用。内置两个标准 Cordis 插件——`FileMentionsPlugin`（`mentions`：在 `agent/pre-step` 展开 `@` 文件/目录/图片引用为 `ContextNotice`）与 `ApprovalGatePlugin`（`approval`：在 `tools/pre-execute` 拦截高危工具执行并等待决策）；并提供前端无关运行时子模块：`images`（图片嗅探/EXIF 校正/动图首帧与自适应压缩）、`session_view`（只读会话投影/模型选择优先级解析/会话信息报告）、`settings`（设置条目构建、热更新与 `SpinePlugins` 启停原子写回）、`local_bash`（`!` / `!!` 本地 Shell 执行与会话审计）、`events`（核心服务等待、事件订阅与 `LoopClientMsg` / `RunController` 代次控制）以及 `workspace`（`WorkspacePanelSnapshot` 工具/MCP/指令/Git/轨迹聚合） | `FileMentionsPlugin` 依赖 `["llm"]`；`ApprovalGatePlugin` 依赖 `["tools"]`；其余运行时工具按需访问 `agentLoop`、`sessions`、`spinePlugins`、`compaction`、`sessionTitle`、`instructions`、`mcp`、`skills`、`trajectory` | `cordis`、`mizar-agent-loop`、`mizar-agent-spine`、`mizar-client-core`、`mizar-instructions`、`mizar-llm`、`mizar-mcp`、`mizar-session`、`mizar-session-title`、`mizar-skills`、`mizar-subagent`、`mizar-tools`、`mizar-tools-coding`、`mizar-trajectory` |
| **`mizar-tui`** | 可执行文件 `mizar`（含 `TuiEmphasisPlugin`） | 终端与无头 CLI 入口 / `"tui-emphasis"` | **薄终端宿主适配器**：解析 CLI 参数并装配 `AgentSpinePlugin`，同时提供交互式 TUI 与非交互无头单次执行（`mizar exec` / `-p` / `--prompt-file`）。TUI 模式（`runner.rs` + `runner/{terminal_io,actions}.rs`）将事件订阅、审批门禁、`@` 引用展开、图片处理、本地 Bash、会话投影与配置热写回全部委托给 `mizar-client` / `mizar-client-core`，自身专注终端 I/O、系统/OSC 5522 剪贴板（`clipboard`）、`mizar-tui-ui` 渲染桥接，以及按 `HostExtensionFlags` 控制 `TuiEmphasisPlugin`（`tui_reply_guidance` 系统提示词挂载）、`FileMentionsPlugin`、审批钩子与各前端特性开关（若 `disabledPlugins` 包含 `"tui"` 则直接拒绝启动而不进入终端 raw mode），支持欢迎画布、`!` / `!!` 本地 Bash 模式、可取消的 `/compact [focus]` 进度压缩、`/resume` 交互式会话选择器与 `/rename [title]` 改名/重生成，以及 `/settings`（`/config`）实时切换；无头 `exec` 模式复用同套装配，支持 `--provider` / `--model` / `--protocol`（`--api-format`）/ `--base-url` / `--reasoning-effort` / `--max-steps` / `--timeout` / `--isolated`，严格校验路由与凭据且仅在 `stdout` 输出最终答案。CLI 同时支持 `--workspace/-C`、`--minimal`、`--read-only`、`--tools/-t`、`--exclude-tools/-xt`、`--inline`、`--approve`、`--continue/-c`、`--resume/-r [id]`、`--no-persist`、`--no-instructions`、`--theme` 与 `mizar update` | `["agentLoop", "sessions", "llm"]`（启动前 `wait_for_services` 探测；`TuiEmphasisPlugin` 注入 `["systemPrompt"]`；按需访问 `spinePlugins`、`compaction`、`sessionTitle`、`skills`、`mcp`、`instructions`、`trajectory`、`llmRouter`、`tools`） | `mizar-client`、`mizar-client-core`、`mizar-tui-ui`、`mizar-agent-spine` 及宿主桥接 crate |

---

## 八、依赖拓扑简图

```text
┌─ 前端与跨端宿主接入层 ──────────────────────────────────────────────────────────┐
│  crates/mizar-tui (CLI + 终端宿主适配)        Merak Desktop（未来桌面端直嵌）   │
│    ├─► crates/mizar-tui-ui                          │                           │
│    │     └─► mizar-client-core / mizar-llm-types / mizar-trajectory-core        │
│    └────────────────────────┬───────────────────────┘                           │
│                             v                                                   │
│  crates/mizar-client（前端无关宿主运行时：零 ratatui / crossterm / mizar-tui-ui）│
│    ├─► crates/mizar-client-core ──► mizar-llm-types                             │
│    └─► crates/mizar-agent-spine（feature-gated 装配层 + spinePlugins）          │
└─────────────────────────────┬───────────────────────────────────────────────────┘
                              v
┌─ 编排与观测层 ─────────────────────────────────────────────────────────────┐
│  mizar-agent-loop      mizar-trajectory ──► mizar-trajectory-core          │
│  (agentLoop + Inbox)   (trajectory + repeat-tool-guard)                    │
│  mizar-session-sqlite  mizar-session-title  mizar-cache-warming            │
│  (sessionPersistence)  (sessionTitle)       (cacheWarming)                 │
└───────────────────────────────┬────────────────────────────────────────────┘
                                v
┌─ 工具与上下文管理插件层 ───────────────────────────────────────────────────┐
│  coding: mizar-tools-coding / mizar-tool-search / mizar-codemode           │
│  legacy: mizar-tool-fs / mizar-tool-time                                   │
│  extra : mizar-mcp / mizar-skills / mizar-recall / mizar-subagent          │
│  ctx   : mizar-compaction（compaction）/ mizar-instructions（ContextNotice）│
└───────────────────────────────┬────────────────────────────────────────────┘
                                v
┌─ 基础服务与模型层 ─────────────────────────────────────────────────────────┐
│  mizar-llm (llm) ──► mizar-llm-types                                       │
│  mizar-llm-router ──► mizar-llm-http + completions/responses/messages/mock │
│  mizar-llm-retry │ mizar-session │ mizar-system-prompt │ mizar-tools       │
└───────────────────────────────┬────────────────────────────────────────────┘
                                v
        [ cordis 微内核：Context / Fiber / effect / 五种事件分发 ]
```

**编译隔离边界**：
- `mizar-client-core`、`mizar-tui-ui`、`mizar-llm-types`、`mizar-trajectory-core` 均不依赖 `tokio`、`cordis`、`reqwest`、`rquickjs` 或 `rusqlite`（其中 `mizar-client-core` **亦禁止**依赖 `ratatui` / `crossterm`）；修改 UI 状态、渲染或交互逻辑时，只需重编极轻量的 `mizar-tui-ui` 与薄宿主 `mizar-tui`。
- `mizar-client` 与 `mizar-client-core` **严禁引入 `ratatui`、`crossterm` 或 `mizar-tui-ui`**，确保未来 `Merak Desktop` 直嵌零终端依赖。

---

## 九、配置字段速查

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
| `disabledPlugins` | string[] | 声明式屏蔽内置可选 Cordis 模块 ID 或官方客户端/TUI 扩展 ID（全局 + 项目合并去重排序；支持 TUI `/settings` `[模块]` 运行期热启停并原子回写；受保护核心服务忽略并告警） |
| `restrictToWorkspace` | bool | 将文件工具限制在工作区内（**默认 `false`**） |
| `sessionDb` | string | SQLite 落盘路径；等价环境变量 `MIZAR_SESSION_DB`（`none` / `off` 关闭） |
| `codemode` | object | `autoEnable` / `onlyMode` / `inlineBudget` / `timeoutMs` |
| `toolSearch` | object | `autoEnable` / `defaultLimit` |
| `skills` | object | `enabled` / `customDirs` / `bundledDir` / `catalogDescriptionMaxLength` |
| `instructions` | object | `enabled` / `candidates` / `localCandidates` / `projectRootMarkers` / `maxBytes` / `maxSourceBytes` / `followSubdirs` |
| `compaction` | object | `enabled` / `autoCompact` / `autoPrune` / `softWindowCap`（默认 `380000`）/ `triggerRatio`（默认 `0.80`）/ `pruneThresholdChars`（默认 `6000`）/ `coldCacheIdleSecs`（默认 `300`）/ `reusePrefixCache` / `disableSummaryReasoning` / `enableReadDedup` / `enableWriteAfterCompactGuard` |
| `recall` | object | `enabled` / `autoActivateOnCompaction` / `defaultSearchLimit`（默认 `8`）/ `readCharLimit`（默认 `6000`） |
| `sessionTitle` | object | `enabled`（默认 `true`）/ `model`（可选 `provider/model`，省略则跟随当前会话模型）/ `thinkingLevel`（默认 `"low"`） |
| `subagent` | object | `maxSteps`（默认 `20`）/ `maxStepsCap`（默认 `40`）/ `timeoutSecs`（默认 `180`，最小钳位 `1`）/ `maxConcurrent`（默认 `4`）/ `stallTimeoutSecs`（默认 `45`，设为 `0` 关闭看门狗）/ `maxWatchdogAlerts`（默认 `2`） |
| `cacheWarming` | string | Prompt Cache 保活模式：`idle`（默认）/ `streaming` / `off`（`mizar exec` 无头模式下自动禁用 `idle`） |
| `showCacheMissNotices` | bool | 是否在对话流显示连续折叠的缓存续命提示（默认 `false`） |
| `ui` | object | `theme` / `screenMode` / `showThinking` / `verboseExpanded` / `sidePanelOpen` / `mouseCapture` / `approvalMode`（可通过 TUI `/settings` 交互修改并自动保存） |

### 客户端宿主与 TUI 扩展模块 ID（`disabledPlugins`）

`mizar-agent-spine` 在 `KNOWN_HOST_EXTENSION_IDS` 中登记了以下 **10 个官方客户端宿主与 TUI 扩展模块 ID**。`SpinePlugins::initialize` 会将其识别为合法已知扩展（不触发未知模块告警或启动错误），由 `mizar-client-core::HostExtensionFlags::from_disabled_plugins`、`mizar-tui-ui::TuiFeatures` 及客户端宿主按开关控制挂载：

| 扩展 ID | 常量名（`mizar-client-core`） | 作用域 | 受 `"tui"` 级联 | 控制的能力与禁用效果 |
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

`models.json` 为 `RouterConfig`（`providers` 映射），字段细节见 `crates/mizar-llm-router/src/config.rs` 顶部注释；`mcp.json` 使用标准 `mcpServers` 结构，并额外支持每个 Server 的 `exposure` 与 `toolExposure`。

> **注意**：`restrictToWorkspace` 默认关闭且 `TuiConfig::approval_mode` 默认为 `Auto`，即开箱状态下 `bash` / `write` / `edit` 可触达工作区外路径且无审批弹窗；需要沙箱感时请显式开启 `restrictToWorkspace` 并搭配 `--approve`（或通过 `/settings` / `/approval ask` 运行期开启审批）。
