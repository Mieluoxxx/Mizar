# Mizar 插件与模块清单

本文档介绍 `crates/` 下各 crate 的分层作用，以及它们在 **Cordis 运行时（`inject` / `provide` 服务）** 与 **Cargo 包依赖** 两个维度上的依赖关系。

目前仓库为纯 Virtual Workspace，唯一的终端入口为 `crates/mizar-tui/src/main.rs`（`mizar-tui`），通过 `mizar-agent-spine` 按需装配 Agent 能力，经跨端共享层（`mizar-client-core` / `mizar-client`）连接纯前端工作台 `mizar-tui-ui`（同时可供桌面端 Merak 直嵌复用）。**纯模型/纯 UI/纯客户端契约层**与**异步插件宿主层**已彻底解耦。

---

## 一、底层微内核与纯类型/核心层（无 `tokio` / 无 HTTP / 无重运行时）

底层 crate 提供数据结构、纯逻辑或微内核抽象。

| Crate | 类型 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- |
| **`cordis`** | 插件微内核框架 | 提供 `Context` 上下文树、`Fiber` 生命周期状态机、`effect` 记账与倒序清理、`Service` 抽象、`inject` 服务依赖注入与 5 种事件分发模式（`emit` / `parallel` / `serial` / `bail` / `waterfall`） | 无 |
| **`mizar-llm-types`** | 纯 Schema / SSE Core | 定义统一消息、`ContentBlock`、`BlockAssembler`、错误分类与流式 `SseParser`（零 `tokio` / `cordis` / `reqwest`） | 无 |
| **`mizar-image`** | 共享图片处理库 | 按文件头识别 PNG、JPEG、WebP、GIF。解码前校验 1 亿像素上限，再校正 EXIF 方向、提取动图首帧，按 2048 像素长边和 5 MB base64 预算生成 `ImageAttachment` | `mizar-llm-types` |
| **`mizar-trajectory-core`** | 纯轨迹建模与检测 | 轨迹事件模型、折叠快照、三泳道时间轴建模与异常检测规则（零 `tokio` / `cordis` / 运行时插件依赖） | `mizar-llm-types` |
| **`mizar-client-core`** | 纯客户端契约与算法层 | 跨 TUI 与 Merak Desktop 共享的纯数据结构与算法：`HostExtensionFlags`（10 个官方扩展开关与 `"tui"` 级联派生）、`PromptQueue`（ID 稳定的待发送队列）、`TpsLedger`（解码吞吐账本）、`WorkspaceFileCache`（`.gitignore` 感知后台索引）、`ClientEvent` 统一事件、`mentions` 解析、`reveal` 平滑吐字、`tool_format` 摘要与纯 DTO（零 `tokio` / `cordis` / `reqwest` / `rquickjs` / `ratatui` / `crossterm`） | `mizar-llm-types` |
| **`mizar-tui-ui`** | 纯终端工作台前端 | `AppState` 状态机、`TuiFeatures` 插件特性开关、`pi-emphasis` 视觉语法、虚拟化 Markdown/代码渲染、双栏侧栏、TimelineRail、Overlay、主题与轨迹全屏视图（零 `tokio` / `cordis` / `reqwest` / `rquickjs`） | `mizar-client-core`、`mizar-llm-types`、`mizar-trajectory-core` |

---

## 二、核心基础服务插件（向 `Context` 提供服务）

以下 4 个插件实现了 `cordis::Service`，负责向容器注册核心能力，均不依赖其他运行时服务：

| Crate | 插件 / 服务结构体 | 提供服务名 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-llm`** | `LlmRuntime` | `"llm"` | 抽象 LLM 服务：重导出 `mizar-llm-types`，提供 `llm/stream` 拦截通道。按目标适配器的模型目录推断视觉能力，显式 `GenerateOptions.supports_images` 优先。目录缺项时允许图片 | 无 | `cordis`、`mizar-llm-types` |
| **`mizar-session`** | `SessionStore` | `"sessions"` | 会话事件日志管理：以只追加日志保存对话、工具调用与回放状态。用户消息和工具回执可带图片，剪枝投影同时移除旧回执的文本和附件。定义 `SessionPersistence` 持久化 trait 占位 `"sessionPersistence"` | 无 | `cordis`、`mizar-llm` |
| **`mizar-system-prompt`** | `SystemPrompt` | `"systemPrompt"` | 按 `order` 拼接插件段落。`identity`（0）给出 Mizar 身份与回复语言。显式配置的非空 `personality`（20）排在工具指南（30）之前。Spine 提供 `runtime_protocol`（40）说明消息中的上下文标签 | 无 | `cordis` |
| **`mizar-tools`** | `ToolRuntime` | `"tools"` | 工具注册表与三段执行管线（`tools/pre-execute` → `tools/execute` → `tools/post-execute`），生成工具 Schema 并统一派发执行。每次工具回执最多保留 20 张图片，超限会在正文说明。定义 `ToolExposure` / `ToolExecutionMode` / `ToolAnnotations` | 无 | `cordis`、`mizar-llm` |

---

## 三、LLM HTTP 客户端、协议适配器与路由插件

HTTP 传输实现收敛在 `mizar-llm-http`，其余协议与路由插件运行时均注入 `"llm"` 服务：

| Crate | 插件 / 服务结构体 | 插件名 / 提供服务 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-llm-http`** | 工具库（非插件） | — | 共享异步 `reqwest` 客户端、超时控制与 SSE 流驱动封装（隔离重型 TLS/HTTP 依赖图） | — | `mizar-llm` |
| **`mizar-llm-completions`** | `CompletionsPlugin` | `"llm-completions"` | **OpenAI Chat Completions 兼容协议适配器**：支持 OpenAI、DeepSeek、OpenRouter、Qwen、vLLM 等端点及 `CompletionsCompat` 方言开关（思考格式、`max_tokens` 字段等） | `["llm"]` | `cordis`、`mizar-llm`、`mizar-llm-http` |
| **`mizar-llm-responses`** | `ResponsesPlugin` | `"llm-responses"` | **OpenAI Responses 协议适配器**：对接 `/responses` 接口，默认将 `System` 作为 `developer` 内联 `input` 首位以命中 Prompt Cache（`supportsDeveloperRole: false` 降级为 `system`，可选顶层 `instructions`），受 `supportsStore`/`supportsTemperature`/`supportsReasoningEffort`/`supportsPromptCacheKey` 门控字段，保序回放 `outputItems` 并解析文本、思考摘要、函数调用与 `usage` 流式事件 | `["llm"]` | `cordis`、`mizar-llm`、`mizar-llm-http` |
| **`mizar-llm-messages`** | `MessagesPlugin` | `"llm-messages"` | **Anthropic Messages 协议适配器**：对接 `/v1/messages`，自动处理 `system` 抽取、连续 `tool_result` 合并、`thinking` 签名与前缀摘要校验，并在计算 `prefixDigest` 后注入 `system` / 末尾 `tools`（受 `supportsCacheControlOnTools` 门控）/ 末尾 `user` 块的 `cache_control` 断点以保住签名思考回放 | `["llm"]` | `cordis`、`mizar-llm`、`mizar-llm-http` |
| **`mizar-llm-mock`** | `MockPlugin` | `"llm-mock"` | **离线假模型适配器**：无需 API Key，按关键词规则模拟 `get_time` / `list_dir` 工具调用与逐字流式输出，用于演示与无网测试 | `["llm"]` | `cordis`、`mizar-llm` |
| **`mizar-llm-retry`** | 策略与流收集库 | — | 提供 `RetryConfig`、带抖动的退避、可取消等待和 `collect_with_retry`。主循环、压缩器和标题生成各自持有请求的重试预算 | — | `mizar-llm` |
| **`mizar-llm-router`** | `LlmRouter` | `"llmRouter"`（服务） | **多提供方声明式路由与热重载插件**：将多路由配置（`~/.mizar/models.json`）映射到上述协议适配器，支持 `protocol` 选择、`api_key_env` 与 `$VAR` / `${VAR}` / `!command` 动态解析、`models` / `modelOverrides` 目录、请求头扩展钩子，并暴露 `apply_config` 原子热更新接口（**当前无内置文件监听**，需宿主在配置变更时主动调用） | `["llm"]` | `cordis`、`mizar-llm`、`mizar-llm-http`、`mizar-llm-completions`、`mizar-llm-responses`、`mizar-llm-messages`、`mizar-llm-mock` |

视觉模型接收用户和工具图片。Anthropic 将工具的文本和图片放在 `tool_result.content` 数组，Responses 使用 `function_call_output.output` 数组。Chat Completions 先发出同一批的所有 `role: "tool"` 回执，再用一条 `role: "user"` 消息按顺序附图。纯文本工具回执沿用原来的字符串格式。目标模型仅支持文本时，三种协议都在序列化前将图片替换为 `[image omitted: current model does not support image input]`。

---

## 四、工具插件

以下插件中，4.1–4.4 向 `"tools"` 运行时注册可执行工具并注入 `"tools"` 服务；**4.5 `mizar-skills`、4.6 `mizar-instructions` 与 4.7 `mizar-compaction` 是例外**：`mizar-skills` 不注册工具（仅读取声明表按需注入 `<available_skills>` 与展开 `/<name>` 手势），`mizar-instructions` 仅在启动期向 `"systemPrompt"` 注入静态工作区指令段落，`mizar-compaction` 提供 `"compaction"` 服务并通过步前与工具管线执行缓存感知尾部剪枝、上下文压缩与压后护栏。

工具的可达性由 `mizar-tools` 的 **`ToolExposure`** 统一控制：

| `ToolExposure` | 直接声明给模型 | 可被 `codemode` 脚本调用 | 列在 `codemode` 描述中 | 典型使用者 |
| :--- | :--- | :--- | :--- | :--- |
| `Direct` | ✓ 激活后 | ✓ 激活后 | `onlyMode` 预算内的本地工具 | `read` / `bash` / `edit` / `write` / `grep` / `find` |
| `ModelOnly` | ✓ 激活后 | ✗（自身是编排工具） | — | `tool_search`、`codemode` |
| `Codemode` | 激活后可声明 | ✓ | 通过沙箱检索 | MCP 工具默认、`codemode` 内部依赖的工具 |
| `Deferred` | ✗（需激活） | ✓ | ✗（靠 `tool_search` / `searchTools()` 发现） | MCP 工具可选项 |
| `Hidden` | ✗ | ✗ | ✗ | 显式屏蔽（`exclude_tools`） |

激活与排除的优先级依次为（`mizar-tools` 的 `is_default_active`）：**`global_active_overrides`** > **`include_tools` 通配白名单** > **`auto_default_active`**（MCP 按配置激活 `codemode`，`tool_search` 随可搜索工具激活） > 工具的 `default_active`；任何情况下 `ToolExposure::Hidden` 与命中 `exclude_tools` 的工具都不可激活。显式激活过的工具会在下一次模型调用时重新参与声明，而 `Codemode` / `Deferred` 工具即使未激活也仍可被脚本或 `tool_search` 调用（`resolve_callable_infos`）。

### 4.1 最小只读套件（Legacy 模式）

| Crate | 插件结构体 | 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-tool-time`** | `ToolTimePlugin` | `"tool-time"` | `get_time` | `["tools"]` | 返回格式化的本地当前日期与时间 | `cordis`、`mizar-tools` |
| **`mizar-tool-fs`** | `ToolFsPlugin` | `"tool-fs"` | `list_dir`、`read_file` | `["tools"]` | 只读文件工具，内建路径防逃逸检查（限制在 `root` 目录内）、UTF-8 安全字节截断（`max_bytes` 默认 20 000） | `cordis`、`mizar-tools` |

### 4.2 核心编码工具集

| Crate | 插件结构体 | 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-tools-coding`** | `CodingToolsPlugin` | `"tools-coding"` | `read`、`bash`、`edit`、`write`、`grep`、`find` | `["tools"]`，可选 `systemPrompt` | 默认 `MinimalCoding` 声明前四项。`All` 或 `--tools` 可启用独立的 `grep` 和 `find`，`ReadOnly` 提供 `read`、`grep` 和 `find`。文本可按行分页读取，读图按文件头识别并检查 32 MiB 输入上限。`edit` 支持批量局部替换。按会话声明生成 `<tools>` 和 `<rules>`。`bash` 给模型展示首尾摘要（20 KiB、400 行、每行 768 字节），给脚本的 `structured_content.output` 最多 1 MiB，不论 `<= 64 KiB` 或流式溢写均以原始字节落盘临时文件（`full_output_path`）且不吞写错误；被 Unix 信号终止时退出码记为 `128 + sig` 并提示信号编号 | `cordis`、`mizar-image`、`mizar-system-prompt`、`mizar-tools` |

`<tools>` 按声明顺序列出工具与 `prompt_snippet`。`<rules>` 根据 `bash`、`grep`、`find` 的可用性给出搜索建议，再按原顺序收集各工具的 `prompt_guidelines` 并去重，最后提示并行执行独立调用、简洁回复和写清文件路径。

`grep` 先在 `$MIZAR_HOME/bin`（默认 `~/.mizar/bin`）查找可执行的 `rg`，再查系统 `PATH`。两处都缺失时才调用宿主提供的 `ensure_bin`。`bash` 把同一托管目录追加在子进程 `PATH` 末尾，保留系统工具优先级。独立 `find` 使用 Rust 的 `ignore` 遍历目录。`fd` 供 shell 命令使用，本地解析兼容 Debian 的 `fdfind`。

### 4.3 工具发现与 JS 编排

| Crate | 插件结构体 | 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-tool-search`** | `ToolSearchPlugin` | `"tool-search"` | `tool_search` | `["tools"]` | **BM25 工具发现**：对 `Codemode` / `Deferred` 暴露等级的工具建索引（名称权重 4.0、命名空间 2.0），命中后按会话（或全局）激活，使其出现在下一次模型调用的声明中。存在可搜索工具时自动激活，`default_limit` 默认 8 | `cordis`、`mizar-tools` |
| **`mizar-codemode`** | `CodemodePlugin` | `"codemode"` | `codemode` | `["tools"]` | 可选 QuickJS 编排工具。默认休眠，`--tools codemode`、`codemode.enabled: true`、`onlyMode` 或 MCP 静态配置可激活。工具进入会话声明时，`<rules>` 收录它用 `Promise.allSettled` 批量调用、串联工具和裁剪输出的指南。显式 `enabled: false` 禁用工具。混合模式使用固定声明，MCP 工具由 `searchTools()`、`describeTool()` 和 `describeNamespace()` 发现。`onlyMode` 隐藏直连声明，按命名空间轮询列出预算内的本地直连工具（默认 3000 token）。脚本接收图片元数据，宿主随外层结果附图。默认 2 个并发工作位，超时 120 秒，取消传递给嵌套调用 | `cordis`、`mizar-tool-search`、`mizar-tools` |

### 4.4 MCP 接入

| Crate | 服务结构体 | 提供服务名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-mcp`** | `McpManager` | `"mcp"` | `mcp__<server>__<tool>`、`list_mcp_resources`、`list_mcp_resource_templates`、`read_mcp_resource` | `["tools"]`，可选 `systemPrompt` | 合并全局与项目 `mcp.json`，项目同名条目可覆盖 `exposure` 和 `toolExposure`。`autoEnableCodemode` 默认开启。启用的服务或其工具覆盖使用 `codemode` 暴露时，初始化立即激活沙箱。`<mcp_servers>` 按配置名排序，描述取自配置。stdio 和 Streamable HTTP 后台握手、工具发现和远端 instructions 不改变这段摘要。MCP `image` 块经解码、校验、压缩后作为附件返回，损坏图片返回文本说明。卸载撤销自动激活并关闭连接 | `cordis`、`mizar-image`、`mizar-llm`、`mizar-llm-http`、`mizar-system-prompt`、`mizar-tools` |

混合模式下，后台握手完成前后，`codemode` 的声明和 `<mcp_servers>` 的字节相同。`tool_search` 有独立的自动激活条件，发现可搜索工具后仍可能加入声明。无头 `exec` 显式选择直连 MCP 工具时会等待远端工具清单，确保第一步即可调用。

### 4.5 技能系统

| Crate | 服务 / 插件结构体 | 提供服务名 / 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-skills`** | `SkillRegistry` + `SkillsPlugin` | `"skills"` / `"skills-plugin"` | —（不注册工具） | `["tools", "skills"]` | **渐进式技能发现与目录注入**：扫描分层技能根（项目 `.agents/skills` → 自定义目录 → 用户 `~/.agents/skills` → 内置包），解析 `SKILL.md` frontmatter，按 rank 升序仲裁同名技能；当会话声明了 `read` 或 `bash` 时通过合成消息注入含 `<location>`（`SKILL.md` 绝对路径）的 `<available_skills>` 目录并支持热失效——监听 `write` / `edit` 写入命中技能目录时自动刷新缓存，广播 `skills/change`；同时保留 `/<name>` 手势直接展开 `<skill_content>` | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-tools` |

### 4.6 工作区指令系统

| Crate | 服务结构体 | 提供服务名 / 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-instructions`** | `AgentInstructions` | `"agentInstructions"` / `"agent-instructions"` | —（不注册工具） | `["systemPrompt"]` | **启动期静态 `AGENTS.md` / `CLAUDE.md` 发现与系统提示词注入**：启动期一次性扫描 `~/.mizar/`（`MIZAR_HOME`）与 `~/.agents/`（`MIZAR_AGENTS_HOME`）以及由 `.git` 项目根到工作区的祖先链；默认候选 `AGENTS.md` / `CLAUDE.md` 与 `.local.md` 变体。同组按去空白 SHA-256 摘要去重、不同内容并存；超预算时**优先淘汰最宽泛的祖先**以保住近目录指令，并在正文做 UTF-8 安全截断。默认 `max_bytes` 65 536、`max_source_bytes` 1 MiB；通过 `ctx.effect` 向 `SystemPrompt` 注入静态只读段落，运行期不做每步磁盘 `stat`，保持 KV Cache 前缀稳定 | `cordis`、`mizar-agent-loop`、`mizar-session`、`mizar-system-prompt`、`mizar-tools` |

### 4.7 上下文压缩

| Crate | 服务 / 插件结构体 | 提供服务名 / 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-compaction`** | `CompactionRuntime` | `"compaction"` / `"CompactionRuntime"` | —（不注册工具） | `["llm", "sessions", "tools", "systemPrompt"]` | 步前剪枝可折叠重复 `read` 和零命中搜索，保留 `SKILL.md` 读取与原始事件记录。热缓存期保护启用缓存的 OpenAI Responses、可思考的 Anthropic Messages，以及仍含 Responses 回放或签名思考块的历史。其余协议默认可剪最近 8,000 token，闲置 90 分钟可清仓。受保护会话达到阈值时先全量剪枝，重估仍达阈值才请求检查点摘要。窗口按 `--context-window`、`MIZAR_CONTEXT_WINDOW`、模型目录、`compaction.defaultContextWindow`、128,000 的顺序取值。预算预留输出并应用 `maxContextTokens` 上限，默认在可用窗口的 80% 触发。摘要沿用最近检查点、确定性摘要与完整工具批次。`compaction/end` 广播会话 ID、触发原因、结果及压缩前后 token 数。溢出恢复可使用净剪枝量。保留 `tools/pre-execute` 写前读护栏与 `/compact [focus]` 手动入口 | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-llm-retry`、`mizar-session`、`mizar-system-prompt`、`mizar-tools` |

---

## 五、会话持久化与标题管理

| Crate | 插件结构体 | 插件名 | 提供服务名 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-session-jsonl`** | `JsonlPersistencePlugin` | `"session-persistence-jsonl"` | `"sessionPersistence"` | 无 | **宽容单会话 JSONL 持久化 + Blob 外置**：按 canonicalize 后的 `cwd` 分桶存储（`<sessions_dir>/<encoded-cwd>/<YYYYMMDDTHHMMSSZ>_<session_id>.jsonl`），无 `cwd` 归入 `_global`；启动期执行可写探针，初始化/写入失败显式暴露给宿主；**惰性落盘**（空会话不建空文件，首次非空追加或 `fork` 才写盘）；`fork` 拒绝重复会话 ID 或覆盖既有文件；用户/上下文/工具图片 base64 经纯 Rust RFC 6234 SHA-256 摘要原子外置到同级 `blobs/<sha256>`（`blob:sha256:<hex>` 引用，缺失仅剔除丢失附件并告警）；Single-Writer 后台线程通过 `mpsc` 顺序追加与 `flush` 同步屏障保障有序落盘并暴露 `last_error`；宽容读取跳过空行/未知记录/崩溃截断尾行且严禁用整数版本号拒开。默认目录 `~/.mizar/sessions`（`MIZAR_SESSIONS_DIR` 可覆盖，设为 `none`/`off`/`memory` 关闭落盘） | `cordis`、`chrono`、`mizar-llm-types`、`mizar-session` |
| **`mizar-session-title`** | `SessionTitleRuntime` | `"session-title"` | `"sessionTitle"` | `["llm", "sessions"]` | **后台异步会话标题自动生成**：订阅 `agent/turn-input`（仅提取 `SessionEvent::User` 消息），首轮输入及每隔 `refresh_interval_turns`（默认 20）轮异步生成 `MMDD｜TYPE｜Topic` 标题（`FEA/DES/FIX/OPT/REL/EXP/DOC/RES`）；自动跳过 `Offline Mock` 并尊重人工 `/rename` 所有权（`TitleSource::Manual`）与并发改名保护；失败静默不重试并保留 `derive_session_title` 兜底清洗标题；提供 `regenerate_now` 供 `/rename` 无参手动重生成 | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-llm-retry`、`mizar-session` |

---

## 六、Agent 编排与观测

| Crate | 插件 / 服务结构体 | 插件名 / 提供服务 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-agent-loop`** | `AgentLoop` | `"AgentLoop"` / 提供 `"agentLoop"` | **Agent 主循环**：串联「记录用户输入 → 拼装提示词与历史 → 流式调模型 → 执行工具并写回会话 → 直到最终回答」，同一步内重试首字前或仅有思考的瞬时断流，通过 `agent/retry` 通知客户端清理草稿；overflow 压缩恢复后按 `session_id` 刷新系统提示词与声明工具；直调工具强制校验当前步公布白名单并独立统计各工具真实耗时（`duration_ms`）；`max_steps` 默认 `None` 不限步数，靠模型收敛、重复守卫与 `CancellationToken` 终止 | `["llm", "tools", "sessions", "systemPrompt"]` | `cordis`、`mizar-llm`、`mizar-llm-retry`、`mizar-session`、`mizar-system-prompt`、`mizar-tools` |
| **`mizar-trajectory`** | `TrajectoryRuntime` + `RepeatToolGuardPlugin` | `"trajectory"`（服务）/ `"repeat-tool-guard"` | **轨迹观测运行时**：基于 `mizar-trajectory-core` 模型与检测器提供 `"trajectory"` 服务，订阅会话与 `agent/*` 事件实时采集 `TraceEvent`（广播 `trajectory/anomaly`），支持结构化 `TrajectoryExport`（`mizar exec --trace-out` / `MIZAR_TRACE_OUT`）与可选 `{export_dir}/{session_id}.jsonl` 落盘；`RepeatToolGuardPlugin` 在 `tools/post-execute` waterfall 上捕捉连续重复调用并追加提醒或强制拦截 | 无（全部通过事件订阅接入） | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-tools`、`mizar-trajectory-core` |

---

## 七、装配与接入层

| Crate | 插件 / 入口 | 插件名 / 形态 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`mizar-agent-spine`** | `AgentSpinePlugin` / `SpinePlugins` | `"agent-spine"` / 提供服务 `"spinePlugins"` | **核心装配套件与模块控制器**：先挂载核心内核服务（含按 `jsonl` feature 与 `sessions_dir` 挂载的 `sessionPersistence`），再按顺序通过 `SpinePlugins`（`SpinePluginsContextExt`）管理内置可选模块（`compaction`、`tool-time`、`tool-fs`、`tools-coding`、`tool-search`、`codemode`、`mcp`、`skills`、`instructions`、`session-title`、`trajectory`、`repeat-guard`）的 `Fiber` 生命周期与运行期热启停（`set_enabled`）；对 `llm` / `sessions` / `tools` / `systemPrompt` / `agentLoop` / `spinePlugins`实施 Gate 2 自保护（严禁禁用），同时识别 `KNOWN_HOST_EXTENSION_IDS` 10 个官方宿主/TUI 扩展 ID 以避免 `disabledPlugins` 误报未知模块警告。支持 Cargo features（`legacy` / `coding` / `compaction` / `jsonl` / `session-title` / `trajectory` / `router`）裁剪（`--no-default-features` 仅拉起核心服务），并将配置加载与路由解析拆分为 `config.rs` + `config/{settings,route}.rs`（提供 `load_mizar_config` / `patch_settings_file` / `persist_disabled_plugins` 原子读写） | 无（自身负责装配与控制） | `cordis`、`mizar-agent-loop`、`mizar-llm`、`mizar-session`、`mizar-system-prompt`、`mizar-tools` 及按 Cargo features 选配的后端 crate |
| **`mizar-client`** | `FileMentionsPlugin`、`ApprovalGatePlugin` 与宿主运行时模块 | `"file-mentions"` / `"approval-gate"`（前端无关宿主运行时） | **跨端宿主运行时（零 `ratatui` / `crossterm` / `mizar-tui-ui`）**：供 `mizar-tui` 与未来桌面端 Merak 共用。内置两个标准 Cordis 插件——`FileMentionsPlugin`（`mentions`：在 `agent/pre-step` 展开 `@` 文件/目录/图片引用为 `ContextNotice`）与 `ApprovalGatePlugin`（`approval`：在 `tools/pre-execute` 拦截高危工具执行并等待决策）；并提供前端无关运行时子模块：`images`（重导出 `mizar-image` 的共享管线）、`session_view`（只读会话投影/模型选择优先级解析/会话信息报告）、`settings`（设置条目构建、热更新与 `SpinePlugins` 启停原子写回）、`local_bash`（`!` / `!!` 本地 Shell 执行与会话审计）、`events`（核心服务等待、事件订阅与 `LoopClientMsg` / `RunController` 代次控制）以及 `workspace`（`WorkspacePanelSnapshot` 工具/MCP/指令/Git/轨迹聚合） | `FileMentionsPlugin` 依赖 `["llm"]`；`ApprovalGatePlugin` 依赖 `["tools"]`；其余运行时工具按需访问 `agentLoop`、`sessions`、`spinePlugins`、`compaction`、`sessionTitle`、`instructions`、`mcp`、`skills`、`trajectory` | `cordis`、`mizar-agent-loop`、`mizar-agent-spine`、`mizar-client-core`、`mizar-image`、`mizar-instructions`、`mizar-llm`、`mizar-mcp`、`mizar-session`、`mizar-session-title`、`mizar-skills`、`mizar-tools`、`mizar-tools-coding`、`mizar-trajectory` |
| **`mizar-tui`** | 可执行文件 `mizar`（含 `TuiEmphasisPlugin`） | 终端与无头 CLI 入口 / `"tui-emphasis"` | **薄终端宿主适配器**：解析 CLI 参数并装配 `AgentSpinePlugin`，同时提供交互式 TUI 与非交互无头单次执行（`mizar exec` / `-p` / `--prompt-file`）。TUI 模式（`runner.rs` + `runner/{terminal_io,actions}.rs`）将事件订阅、审批门禁、`@` 引用展开、图片处理、本地 Bash、会话投影与配置热写回全部委托给 `mizar-client` / `mizar-client-core`，自身专注终端 I/O、系统/OSC 5522 剪贴板（`clipboard`）、`mizar-tui-ui` 渲染桥接，以及按 `HostExtensionFlags` 控制 `TuiEmphasisPlugin`（`tui_reply_guidance` 系统提示词挂载）、`FileMentionsPlugin`、审批钩子与各前端特性开关（若 `disabledPlugins` 包含 `"tui"` 则直接拒绝启动而不进入终端 raw mode），支持欢迎画布、`!` / `!!` 本地 Bash 模式、可取消的 `/compact [focus]` 进度压缩、`/resume` 交互式会话选择器与 `/rename [title]` 改名/重生成，以及 `/settings`（`/config`）实时切换；无头 `exec` 模式复用同套装配，支持 `--provider` / `--model` / `--protocol`（`--api-format`）/ `--base-url` / `--reasoning-effort` / `--max-steps` / `--timeout` / `--isolated`，校验路由与凭据，标准错误报告重试；TUI 与无头模式都支持 `--context-window <tokens>`，TUI 的 `/session` 与侧栏使用生效窗口。文本模式仅在成功时向标准输出写答案。`exec --json`、`--output-format json` 和 `--output-format ndjson` 使用下述事件流。CLI 同时支持 `--workspace/-C`、`--minimal`、`--read-only`、`--tools/-t`、`--exclude-tools/-xt`、`--inline`、`--approve`、`--continue/-c`、`--resume/-r [id]`、`--no-persist`、`--no-instructions`、`--theme`、`mizar update` 与 `mizar install-tools` | `["agentLoop", "sessions", "llm"]`（启动前 `wait_for_services` 探测；`TuiEmphasisPlugin` 注入 `["systemPrompt"]`；按需访问 `spinePlugins`、`compaction`、`sessionTitle`、`skills`、`mcp`、`instructions`、`trajectory`、`llmRouter`、`tools`） | `mizar-client`、`mizar-client-core`、`mizar-tui-ui`、`mizar-agent-spine` 及宿主桥接 crate |

`mizar-tui` 负责 `rg` 和 `fd` 的下载。交互式 TUI 挂载后在后台并发补齐两个工具。TUI 的 `grep` 通过 `AgentSpineConfig::with_ensure_bin` 复用同一次 `rg` 安装。无头 `exec` 只使用已有工具。`mizar install-tools [--force] [rg|fd ...]` 默认处理两者，显式指定名字可只装一项。`--force` 把指定工具重新安装到托管目录。命令打印来源、路径及 `--version` 输出。

安装器支持 macOS、Linux 的 x86_64 和 aarch64，Linux 使用 musl 发行包。Intel Mac 的 `fd` 固定为 10.3.0，其余组合从 `releases/latest` 的重定向取 tag。版本查询失败、下载响应头超过 10 秒、HTTP 错误、慢速或停滞都可触发镜像回退。每次下载限时 120 秒，压缩包和提取的单个二进制各限 64 MiB。安装器用系统 `tar` 在独立暂存目录提取目标文件，运行 `--version`，退出码为 0 且标准输出含非空白内容时才原子替换目标。

| 工具安装环境变量 | 取值及默认行为 |
| --- | --- |
| `MIZAR_OFFLINE`、`PI_OFFLINE` | 任一设为 `1`、`true` 或 `yes` 时只查本地，缺失时报告离线状态。 |
| `MIZAR_GITHUB_BASE_URL` | 默认为 `https://github.com`。 |
| `MIZAR_GITHUB_MIRROR` | 默认为 `https://ghfast.top`。镜像请求地址为 `{mirror}/{原始完整 URL}`。设为 `off`、`none` 或空串可关闭。 |
| `MIZAR_BIN_SLOW_BPS`、`MIZAR_BIN_WARMUP_MS` | 默认预热 5,000 毫秒。直连平均速率低于 65,536 字节/秒时，转到已启用的镜像。 |
| `MIZAR_BIN_STALL_MS` | 连续收不到新字节的时限，默认为 8,000 毫秒。 |

`mizar exec` 的 NDJSON 标准输出每行是一条带 `type` 和 `ts_ms` 的 JSON 对象，每行写完即刷新。会话和路由就绪时发送 `session`。`step_start` 和 `step_end` 标记模型调用，`step_end` 给出耗时、首 token 延迟和该步最终尝试的 `usage`。`tool_call` 携带单行脱敏参数预览，最多 512 个字符。`tool_result` 给出调用 ID、工具名、错误状态、单工具真实执行耗时（并行批次中各自独立计时，而非整批墙钟）和图片张数。`retry` 记录尝试次数、等待毫秒数、原因及是否撤销草稿，标准错误同时写一行 `[W retry]`。`compaction` 报告触发原因、结果和前后 token 数。

启用持久化时，`result` 在会话写入完成或报告存储错误后输出。`reason`、`exit_code`、`text` 和错误时的 `error` 描述终态。本轮指标包括 `steps`、`duration_ms`、`llm_ms`、`ttft_ms`、`tool_ms`、`tool_calls`、`tool_errors`、`retries`、`compactions` 和 `usage`。`session_id`、`resumed`、`session_turns`、`session_steps` 和 `session_usage` 对应整个会话。用量分为 `input_tokens`、`output_tokens`、`total_tokens`、`cache_read_tokens`、`cache_write_tokens`、`reasoning_tokens`。`input_tokens` 是未命中缓存的输入，`total_tokens` 合计净输入、缓存读写和输出，推理用量是输出的细分。输入或配置校验失败时仅输出一行 `invalid_request` 终态。

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
│  (agentLoop)         (trajectory + repeat-tool-guard)                      │
│  mizar-session-jsonl (sessionPersistence) │ mizar-session-title (sessionTitle) │
└───────────────────────────────┬────────────────────────────────────────────┘
                                v
┌─ 工具与上下文管理插件层 ───────────────────────────────────────────────────┐
│  coding: mizar-tools-coding / mizar-tool-search / mizar-codemode           │
│  legacy: mizar-tool-fs / mizar-tool-time                                   │
│  extra : mizar-mcp / mizar-skills                                          │
│  ctx   : mizar-compaction（compaction）/ mizar-instructions（SystemPrompt）│
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
- `mizar-client-core`、`mizar-tui-ui`、`mizar-llm-types`、`mizar-trajectory-core`、`mizar-image` 均不依赖 `tokio`、`cordis`、`reqwest`、`rquickjs`（其中 `mizar-client-core` **亦禁止**依赖 `ratatui` / `crossterm`）；修改 UI 状态、渲染或交互逻辑时，只需重编极轻量的 `mizar-tui-ui` 与薄宿主 `mizar-tui`。
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
| `personality` | string | 在身份定位之后插入用户指定的沟通风格。省略、空串和纯空白均占 0 字节 |
| `toolsPreset` | `minimal` / `minimal-coding` / `all` / `read-only` / `readonly` | 默认 `minimal`，可用 `--minimal` 或 `--read-only` 覆盖项目配置 |
| `tools` | string[] | 工具白名单（支持 `*` 通配） |
| `excludeTools` | string[] | 工具黑名单（支持 `*` 通配，多级目录累加） |
| `disabledPlugins` | string[] | 声明式屏蔽内置可选 Cordis 模块 ID 或官方客户端/TUI 扩展 ID（全局 + 项目合并去重排序；支持 TUI `/settings` `[模块]` 运行期热启停并原子回写；受保护核心服务忽略并告警） |
| `restrictToWorkspace` | bool | 将文件工具限制在工作区内（**默认 `false`**） |
| `sessionsDir` | string | JSONL 会话持久化目录；等价环境变量 `MIZAR_SESSIONS_DIR`（`none` / `off` / `memory` 关闭） |
| `codemode` | object | `enabled`（缺省按需激活）/ `onlyMode` / `inlineBudget`（token）/ `timeoutMs` |
| `toolSearch` | object | `autoEnable` / `defaultLimit` |
| `skills` | object | `enabled` / `customDirs` / `bundledDir` / `catalogDescriptionMaxLength` |
| `instructions` | object | `enabled` / `candidates` / `localCandidates` / `projectRootMarkers` / `maxBytes` / `maxSourceBytes` |
| `compaction` | object | `enabled` / `autoCompact` / `defaultContextWindow`（未知模型默认窗口 `128000`）/ `maxContextTokens`（可选软上限，默认 `null`）/ `triggerRatio`（默认 `0.80`）/ `keepRecentTokens`（默认 `20000`）/ `supersedeReads` / `dropUseless` |
| `sessionTitle` | object | `enabled`（默认 `true`）/ `model`（可选 `provider/model`，省略则跟随当前会话模型）/ `thinkingLevel`（默认 `"low"`） |
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

`models.json` 为 `RouterConfig`（`providers` 映射），字段细节见 `crates/mizar-llm-router/src/config.rs` 顶部注释；`mcp.json` 使用标准 `mcpServers` 结构，支持顶层 `autoEnableCodemode` 以及每个 Server 的 `exposure` 与 `toolExposure`。

> **注意**：`restrictToWorkspace` 默认关闭且 `TuiConfig::approval_mode` 默认为 `Auto`，即开箱状态下 `bash` / `write` / `edit` 可触达工作区外路径且无审批弹窗；需要沙箱感时请显式开启 `restrictToWorkspace` 并搭配 `--approve`（或通过 `/settings` / `/approval ask` 运行期开启审批）。
