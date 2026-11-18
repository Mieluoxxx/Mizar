# dsh-rs 插件与模块清单

本文档介绍 `crates/` 下共 **16 个 crate** 的作用，以及它们在 **Cordis 运行时（`inject` / `provide` 服务）** 与 **Cargo 包依赖** 两个维度上的依赖关系。

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
| **`dsh-session`** | `SessionStore` | `"sessions"` | 会话事件日志管理：以只追加日志保存对话、工具调用与回放状态，并推导出发往模型的消息列表 | 无 | `cordis`、`dsh-llm` |
| **`dsh-system-prompt`** | `SystemPrompt` | `"systemPrompt"` | 系统提示词动态组装：支持多插件按 `order` 贡献段落并自动拼接（内置 `persona` 与当前日期） | 无 | `cordis` |
| **`dsh-tools`** | `ToolRuntime` | `"tools"` | 工具注册表与三段执行管线（`tools/pre-execute` → `tools/execute` → `tools/post-execute`），生成工具 Schema 并统一派发执行 | 无 | `cordis`、`dsh-llm` |

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
| **`dsh-llm-router`** | `LlmRouter` | `"llmRouter"`（服务） | **多提供方声明式路由与热重载插件**：将一份多路由配置映射到上述协议适配器，支持 `api_key_env` 动态解析、请求头扩展钩子与原子热更新 | `["llm"]` | `cordis`、`dsh-llm`（`http`）、`dsh-llm-completions`、`dsh-llm-responses`、`dsh-llm-messages`、`dsh-llm-mock` |

---

## 四、工具插件

以下插件向工具运行时注册具体的可执行工具，运行时均注入 `"tools"` 服务：

| Crate | 插件结构体 | 插件名 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-tool-time`** | `ToolTimePlugin` | `"tool-time"` | 注册 `get_time` 工具，返回格式化的本地当前日期与时间 | `["tools"]` | `cordis`、`dsh-tools` |
| **`dsh-tool-fs`** | `ToolFsPlugin` | `"tool-fs"` | 注册只读文件工具 `list_dir` 与 `read_file`，内建路径防逃逸检查（限制在 `root` 目录内）与 UTF-8 安全字节截断 | `["tools"]` | `cordis`、`dsh-tools` |

---

## 五、Agent 编排、装配与接入层

| Crate | 插件 / 服务结构体 | 插件名 / 提供服务 | 作用 | 运行时服务依赖（`inject`） | 内部 Crate 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`dsh-agent-loop`** | `AgentLoop` | `"AgentLoop"` / 提供 `"agentLoop"` | **Agent 主循环**：串联「记录用户输入 → 拼装提示词与历史 → 流式调模型 → 执行工具并写回会话 → 直到最终回答」，广播 `agent/*` 流式事件 | `["llm", "tools", "sessions", "systemPrompt"]` | `cordis`、`dsh-llm`、`dsh-session`、`dsh-system-prompt`、`dsh-tools` |
| **`dsh-agent-spine`** | `AgentSpinePlugin` | `"agent-spine"` | **核心装配套件**：一次性挂载 `LlmRuntime`、`SessionStore`、`ToolRuntime`、`SystemPrompt`、`ToolTimePlugin`、`ToolFsPlugin` 与 `AgentLoop` | 无（自身负责装配核心基础服务与工具） | `cordis`、`dsh-llm`、`dsh-session`、`dsh-system-prompt`、`dsh-tools`、`dsh-tool-time`、`dsh-tool-fs`、`dsh-agent-loop` |
| **`dsh-webserver`** | `WebServerPlugin` | `"webserver"` | **HTTP / SSE 网页服务**：提供静态页面（`GET /`）与聊天流接口（`POST /api/chat`），通过 SSE 推送 `session` / `delta` / `reasoning` / `tool` / `usage` / `done` 事件 | `["agentLoop", "sessions"]` | `cordis`、`dsh-agent-loop`、`dsh-session` |

---

## 六、依赖拓扑简图

```text
                       [ src/main.rs ]
                     /        |         \
                    v         v          v
       dsh-agent-spine   dsh-llm-router   dsh-webserver
      (装配核心 + 工具)    /   |   |   \     /        \
       /   |   |   \     v    v   v    v   v          \
      v    v   v    v  comp resp msg mock agent-loop   \
   tools  fs  time  sp   \    |   |   /   /   |   \     \
     \              |     \   |   |  /   /    |    \     \
      `-------------+-----> dsh-llm <---+   tools  sessions
                             ^                 |      |
                             |                 v      v
                       dsh-llm-retry        [ cordis 微内核 ]
```
