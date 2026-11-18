# dsh-rs 插件与模块清单

本文档介绍 `crates/` 下共 **24 个 crate** 的作用，以及它们在 **Cordis 运行时（`inject` / `provide` 服务）** 与 **Cargo 包依赖** 两个维度上的依赖关系。

两个入口（`src/main.rs` 网页端、`crates/dsh-tui/src/main.rs` 终端端）都只做三件事：加载配置、挂载 `dsh-agent-spine`、选择界面与模型来源。**Agent 能力**与**接入方式**是彻底解耦的。

---

## 一、底层框架

### `cordis`
* **位置**：`crates/cordis`
* **类型**：插件微内核框架（非插件）
* **作用**：提供 `Context` 上下文树、`Fiber` 生命周期状态机、`effect` 记账与倒序清理、`Service` 抽象、基于 `inject` 的服务依赖注入，以及 5 种事件分发模式（`emit` / `parallel` / `serial` / `bail` / `waterfall`）。
* **依赖**：不依赖任何内部 crate。

---

## 二、核心基础服务插件（向 `Context` 提供服务）

以下 4 个插件实现了 `cordis::Service`，负责向容器注册核心能力，均不依赖其他服务：

| Crate | 插件 / 服务结构体 | 提供服务名 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-llm`** | `LlmRuntime` | `"llm"` | 抽象 LLM 服务：定义统一消息、内容块、流式增量、`BlockAssembler`、错误分类、SSE 解析及 `llm/stream` 拦截通道，管理适配器注册 | 无 | `cordis` |
| **`dsh-session`** | `SessionStore` | `"sessions"` | 会话事件日志管理：以只追加日志保存对话、工具调用与回放状态，并推导出发往模型的消息列表；同时定义 `SessionPersistence` 持久化 trait 占位 `"sessionPersistence"` | 无 | `cordis`、`dsh-llm` |
| **`dsh-system-prompt`** | `SystemPrompt` | `"systemPrompt"` | 系统提示词动态组装：支持多插件按 `order` 贡献段落并自动拼接（内置 `persona` 与当前日期） | 无 | `cordis` |
| **`dsh-tools`** | `ToolRuntime` | `"tools"` | 工具注册表与三段执行管线（`tools/pre-execute` → `tools/execute` → `tools/post-execute`），生成工具 Schema 并统一派发执行；定义 `ToolExposure` / `ToolExecutionMode` / `ToolAnnotations` | 无 | `cordis`、`dsh-llm` |

---

## 三、LLM 协议、路由与中间件插件

这些插件负责接入不同模型协议、路由策略与流式容错，运行时均注入 `"llm"` 服务：

| Crate | 插件 / 服务结构体 | 插件名 / 提供服务 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-llm-completions`** | `CompletionsPlugin` | `"llm-completions"` | **OpenAI Chat Completions 兼容协议适配器**：支持 OpenAI、DeepSeek、OpenRouter、Qwen、vLLM 等端点及 `CompletionsCompat` 方言开关（思考格式、`max_tokens` 字段等） | `["llm"]` | `cordis`、`dsh-llm`（`http`） |
| **`dsh-llm-responses`** | `ResponsesPlugin` | `"llm-responses"` | **OpenAI Responses 协议适配器**：对接 `/responses` 接口，将 `System` 转为 `instructions` 并解析文本、思考摘要、函数调用与 `usage` 流式事件 | `["llm"]` | `cordis`、`dsh-llm`（`http`） |
| **`dsh-llm-messages`** | `MessagesPlugin` | `"llm-messages"` | **Anthropic Messages 协议适配器**：对接 `/v1/messages`，自动处理 `system` 抽取、连续 `tool_result` 合并、`thinking` 签名与前缀摘要校验 | `["llm"]` | `cordis`、`dsh-llm`（`http`） |
| **`dsh-llm-mock`** | `MockPlugin` | `"llm-mock"` | **离线假模型适配器**：无需 API Key，按关键词规则模拟 `get_time` / `list_dir` 工具调用与逐字流式输出，用于演示与测试 | `["llm"]` | `cordis`、`dsh-llm` |
| **`dsh-llm-retry`** | `LlmRetryPlugin` | `"llm-retry"` | **流式自动重试中间件**：拦截 `llm/stream` waterfall，在未发出可见内容前对 `RateLimit` / `Server` / `Timeout` / `Transport` 等瞬时故障按退避策略重试 | `["llm"]` | `cordis`、`dsh-llm` |
| **`dsh-llm-router`** | `LlmRouter` | `"llmRouter"`（服务） | **多提供方声明式路由与热重载插件**：将一份多路由配置（`~/.dsh-rs/models.json`）映射到上述协议适配器，支持 `protocol` 选择、`api_key_env` 动态解析、`models` / `model_overrides` 目录、请求头扩展钩子与原子热更新 | `["llm"]` | `cordis`、`dsh-llm`（`http`）、`dsh-llm-completions`、`dsh-llm-responses`、`dsh-llm-messages`、`dsh-llm-mock` |

---

## 四、工具插件

以下插件均向 `"tools"` 运行时注册可执行工具，运行时均注入 `"tools"` 服务。

工具的可达性由 `dsh-tools` 的 **`ToolExposure`** 统一控制，这是理解整个工具生态的关键：

| `ToolExposure` | 直接声明给模型 | 可被 `codemode` 脚本调用 | 列在 `codemode` 描述中 | 典型使用者 |
| :--- | :--- | :--- | :--- | :--- |
| `Direct` | ✓ 激活后 | ✓ | — | `read` / `bash` / `edit` / `write` / `grep` / `find` / `skill` |
| `ModelOnly` | ✓ 激活后 | ✗（自身是编排工具） | — | `tool_search`、`codemode` |
| `Codemode` | ✗（需激活） | ✓ | ✓ | MCP 工具默认、`codemode` 内部依赖的工具 |
| `Deferred` | ✗（需激活） | ✓ | ✗（靠 `tool_search` / `searchTools()` 发现） | MCP 工具可选项 |
| `Hidden` | ✗ | ✗ | ✗ | 显式屏蔽（`exclude_tools`） |

### 4.1 最小只读套件（Legacy 模式）

| Crate | 插件结构体 | 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-tool-time`** | `ToolTimePlugin` | `"tool-time"` | `get_time` | `["tools"]` | 返回格式化的本地当前日期与时间 | `cordis`、`dsh-tools` |
| **`dsh-tool-fs`** | `ToolFsPlugin` | `"tool-fs"` | `list_dir`、`read_file` | `["tools"]` | 只读文件工具，内建路径防逃逸检查（限制在 `root` 目录内）、UTF-8 安全字节截断（`max_bytes` 默认 20 000） | `cordis`、`dsh-tools` |

### 4.2 核心编码工具集

| Crate | 插件结构体 | 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-tools-coding`** | `CodingToolsPlugin` | `"tools-coding"` | `read`、`write`、`edit`、`bash`、`grep`、`find` | `["tools"]` | 对齐 `pi` 的编码工具体系：`read` 支持行偏移与长度限制，`edit` 做词级 diff 替换，`grep` / `find` 走 `ignore` 库（尊重 `.gitignore`），`bash` 支持超时与输出截断。三档预设：`All`（6 个）、`MinimalCoding`（4 个：`read`/`bash`/`edit`/`write`）、`ReadOnly`（3 个：`read`/`grep`/`find`）；可选向 `SystemPrompt` 贡献「可用工具 + 工作目录」段落 | `cordis`、`dsh-system-prompt`、`dsh-tools` |

### 4.3 工具发现与 JS 编排

| Crate | 插件结构体 | 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-tool-search`** | `ToolSearchPlugin` | `"tool-search"` | `tool_search` | `["tools"]` | **BM25 工具发现**：对 `Codemode` / `Deferred` 暴露等级的工具建索引（名称权重 4.0、命名空间 2.0），命中后按会话（或全局）激活，使其出现在下一次模型调用的声明中。存在可搜索工具时自动激活，`default_limit` 默认 8 | `cordis`、`dsh-tools` |
| **`dsh-codemode`** | `CodemodePlugin` | `"codemode"` | `codemode` | `["tools"]` | **基于 QuickJS（`rquickjs`）沙箱的多工具 JS 编排引擎**：把多个工具调用折叠成一次脚本执行，中间结果不进上下文。提供 `searchTools()` 内建检索、并发信号量（`max_workers` 默认 2）、脚本超时（默认 120 s）与内联预算（默认 6 000 字符）；`only_mode` 可隐藏 `Direct` 工具的直接声明，全部改由脚本调度 | `cordis`、`dsh-tool-search`、`dsh-tools` |

### 4.4 MCP 接入

| Crate | 服务结构体 | 提供服务名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-mcp`** | `McpManager` | `"mcp"` | `mcp__<server>__<tool>`、`list_mcp_resources`、`list_mcp_resource_templates`、`read_mcp_resource` | `["tools"]` | **MCP 客户端**：从 `~/.dsh-rs/mcp.json` 与 `<workspace>/.dsh-rs/mcp.json` 加载配置，支持 **stdio**（`command` / `args` / `env` / `cwd`）与 **Streamable HTTP**（`url` / `headers`）两种传输；按 Server 与逐工具两级配置暴露等级（默认 `Codemode`）；后台异步连接不阻塞启动；向 `SystemPrompt` 贡献 `<mcp_servers>` 段落 | `cordis`、`dsh-llm`、`dsh-system-prompt`、`dsh-tools` |

### 4.5 技能系统

| Crate | 服务 / 插件结构体 | 提供服务名 / 插件名 | 注册工具 | `inject` | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-skills`** | `SkillRegistry` + `SkillsPlugin` | `"skills"` / `"skills-plugin"` | `skill` | `["tools", "skills"]` | **渐进式技能发现与按需加载**：扫描分层技能根（项目 `.agents/skills` → 自定义目录 → 用户 `~/.agents/skills` → 内置包），解析 `SKILL.md` frontmatter，按 rank 升序仲裁同名技能；通过合成消息注入技能目录并支持热失效——监听 `write` / `edit` 写入命中技能目录时自动刷新缓存，广播 `skills/change` | `cordis`、`dsh-agent-loop`、`dsh-llm`、`dsh-session`、`dsh-tools` |

---

## 五、会话持久化

| Crate | 插件结构体 | 插件名 | 提供服务名 | 作用 | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-session-sqlite`** | `SqlitePersistencePlugin` | `"session-persistence-sqlite"` | `"sessionPersistence"` | **嵌入式 SQLite 持久化**：WAL 模式 + `busy_timeout`，独占 **Single-Writer 后台线程**通过批量事务提交，把写入从 Tokio 调度器中隔离；只读连接加 `query_only`。默认路径 `~/.dsh-rs/sessions.sqlite`（`DSH_RS_SESSION_DB` 可覆盖，设为 `none`/`off` 关闭落盘） | `cordis`、`dsh-session` |

---

## 六、Agent 编排与观测

| Crate | 插件 / 服务结构体 | 插件名 / 提供服务 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-agent-loop`** | `AgentLoop` | `"AgentLoop"` / 提供 `"agentLoop"` | **Agent 主循环**：串联「记录用户输入 → 拼装提示词与历史 → 流式调模型 → 执行工具并写回会话 → 直到最终回答」，广播全套 `agent/*` 流式事件（`turn-start` / `step-start` / `step-first-token` / `text-delta` / `reasoning-delta` / `tool-call` / `tool-result` / `usage` / `step-end` / `turn-end` / `status`）；`max_steps` 默认 `None` 不限步数，靠模型收敛、重复守卫与 `CancellationToken` 终止 | `["llm", "tools", "sessions", "systemPrompt"]` | `cordis`、`dsh-llm`、`dsh-session`、`dsh-system-prompt`、`dsh-tools` |
| **`dsh-trajectory`** | `TrajectoryRuntime` + `RepeatToolGuardPlugin` | `"trajectory"` / `"repeat-tool-guard"` | **轨迹观测层**：订阅会话与 `agent/*` 事件采集 `TraceEvent`，做纯折叠投影、三泳道时间轴建模与异常检测（广播 `trajectory/anomaly`），可选把每条轨迹实时落盘为 `{export_dir}/{session_id}.jsonl`；`RepeatToolGuardPlugin` 在 `tools/post-execute` waterfall 上捕捉连续重复调用，按 `[3, 5, 8]` 阈值向模型追加提醒，可选 `hard_block_after` 强制拦截，广播 `guard/repeat-tool` | 无（全部通过事件订阅接入） | `cordis`、`dsh-agent-loop`、`dsh-llm`、`dsh-session`、`dsh-tools` |

---

## 七、装配与接入层

| Crate | 插件 / 入口 | 插件名 / 形态 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-agent-spine`** | `AgentSpinePlugin` | `"agent-spine"` | **核心装配套件**：一次性挂载 `LlmRuntime`、可选 `SqlitePersistencePlugin`、`SessionStore`、`ToolRuntime`、`SystemPrompt`、工具套件（`Legacy` / `Coding` 两种模式）、`AgentLoop`、`TrajectoryRuntime` 与 `RepeatToolGuardPlugin`，并统一应用工具黑白名单。另提供 `load_dsh_config` 读取 `~/.dsh-rs/` 与 `<workspace>/.dsh-rs/` 下的 `settings.json` / `models.json` / `mcp.json` | 无（自身负责装配） | 15 个 `dsh-*` crate + `cordis`（见拓扑图） |
| **`dsh-webserver`** | `WebServerPlugin` | `"webserver"` | **HTTP / SSE 网页服务**：静态页 `GET /`、`GET /index.html`；聊天流 `POST /api/chat`，推送 `session` / `delta` / `reasoning` / `tool` / `usage` / `done` / `error` 事件；会话查询 `GET /api/sessions`、`GET /api/sessions/{id}/trajectory`、`GET /api/sessions/{id}/export`、`POST /api/trajectory/inspect`（轨迹相关路由按需读取 `"trajectory"` 服务，未挂载时返回 `trajectory service not mounted`） | `["agentLoop", "sessions"]` | `cordis`、`dsh-agent-loop`、`dsh-session`、`dsh-trajectory` |
| **`dsh-tui`** | 可执行文件 `dsh-tui` | 接入层（非插件） | **交互式终端工作台**：双栏布局 + 语义化时间轴导轨、Oklab 主题体系（`auto` / `mist-dark` / `mist-light` / `sakura-dark` / `system` / `ansi16`）、虚拟化流式渲染、鼠标选区、词级 diff、`pi-emphasis` 视觉语法与三泳道轨迹视图。CLI 支持 `--workspace/-C`、`--minimal`、`--read-only`、`--tools/-t`、`--exclude-tools/-xt`、`--inline`、`--approve`、`--continue/-c`、`--resume/-r`、`--no-persist`、`--theme` | `["agentLoop", "sessions", "llm"]`（启动前 `wait_for_services` 探测） | 13 个内部 crate |

---

## 八、依赖拓扑简图

按「层」自下而上读：

```text
┌─ 入口层 ────────────────────────────────────────────────────────────────┐
│  src/main.rs                  crates/dsh-tui/src/main.rs                 │
│  （网页 · Legacy 工具）        （终端 · Coding 工具）                     │
└──────────────┬────────────────────────────┬──────────────────────────────┘
               └────────────┬───────────────┘
                            v
┌─ 装配层 ────────────────────────────────────────────────────────────────┐
│  dsh-agent-spine   ← 唯一入口，按 Legacy / Coding 两种模式装配            │
│                     依赖 15 个 dsh-* crate + cordis                      │
└──────────────┬───────────────────────────────────────────────────────────┘
               v
┌─ 编排与观测层 ──────────────────────────────────────────────────────────┐
│  dsh-agent-loop      dsh-trajectory          dsh-session-sqlite          │
│  提供 agentLoop      提供 trajectory         提供 sessionPersistence      │
│        └──────────── 事件订阅 ────────────┘                              │
└──────────────┬───────────────────────────────────────────────────────────┘
               v
┌─ 工具层（均 inject ["tools"]） ─────────────────────────────────────────┐
│  dsh-tools-coding   read/write/edit/bash/grep/find                       │
│  dsh-tool-fs        list_dir/read_file        （Legacy 模式）            │
│  dsh-tool-time      get_time                  （Legacy 模式）            │
│  dsh-tool-search    tool_search        ← BM25 发现                       │
│  dsh-codemode       codemode           ← QuickJS 编排，依赖 tool-search  │
│  dsh-mcp            mcp__* / *_mcp_resource                              │
│  dsh-skills         skill              ← 另提供 skills 服务              │
└──────────────┬───────────────────────────────────────────────────────────┘
               v
┌─ 基础服务层 ────────────────────────────────────────────────────────────┐
│  dsh-llm (llm)   dsh-session (sessions)   dsh-system-prompt (systemPrompt)│
│  dsh-tools (tools)                                                        │
└──────────────┬───────────────────────────────────────────────────────────┘
               v
┌─ LLM 接入层（均 inject ["llm"]） ──────────────────────────────────────┐
│  dsh-llm-router  (llmRouter)                                              │
│    ├─ dsh-llm-completions   OpenAI Chat Completions 兼容                  │
│    ├─ dsh-llm-responses     OpenAI Responses                             │
│    ├─ dsh-llm-messages      Anthropic Messages                           │
│    └─ dsh-llm-mock          离线假模型                                    │
│  dsh-llm-retry     拦截 llm/stream waterfall，瞬时故障退避重试             │
└──────────────┬───────────────────────────────────────────────────────────┘
               v
┌─ 接入层 ────────────────────────────────────────────────────────────────┐
│  dsh-webserver   inject ["agentLoop", "sessions"]                       │
└──────────────┬───────────────────────────────────────────────────────────┘
               v
        [ cordis 微内核：Context / Fiber / effect / 五种事件分发 ]
```

**反向依赖（工具 → 基础服务）**：工具插件依赖 `tools` 服务而非反向；`dsh-mcp` 额外依赖 `dsh-system-prompt`（注入 `<mcp_servers>` 段落），`dsh-skills` 额外依赖 `dsh-agent-loop`（监听工具结果做热失效），`dsh-session-sqlite` 只依赖 `dsh-session` 的 trait。

---

## 九、两个入口的差异

| 维度 | `src/main.rs`（网页端） | `crates/dsh-tui`（终端端） |
| :--- | :--- | :--- |
| 装配模式 | `AgentSpineConfig::new()` → **`Legacy`**：`get_time` + `list_dir` + `read_file` | `AgentSpineConfig::coding()` → **`Coding`**：`read`/`write`/`edit`/`bash`/`grep`/`find` + `tool_search` + `codemode` + MCP + skills |
| 接入层 | `WebServerPlugin`（默认 `127.0.0.1:3000`） | `run_tui`（`Fullscreen` 或 `--inline`） |
| 会话持久化 | `resolve_default_session_db_path()` | 同上，`--no-persist` 可关闭 |
| 高危工具审批 | 无 | `--approve` 开启交互式闸门 |
| 视图/主题 | 浏览器端 HTML | 6 套 Oklab 主题 + 三泳道轨迹 |

> **注意**：网页端当前是**只读工具集**，不具备写文件、执行命令的能力。若要让网页端具备完整编码能力，需要把 `src/main.rs` 的装配从 `AgentSpineConfig::new()` 换成 `AgentSpineConfig::coding()`，并在根 `Cargo.toml` 补上 `dsh-tools-coding` 等依赖。