# AGENTS.md

给在本仓库工作的 AI coding agent 的说明。

**Mizar** = Rust 重写的终端 AI 编程 Agent：`mizar`（`mizar-tui`）是唯一可执行入口，
10 个 crate 通过 `mizar-core::spine::MizarRuntime` 静态组合根与 `mizar_types::hooks` 强类型钩子装配成「流式调模型 → 执行工具 → 回写会话」的主循环。

架构权威文本是 **[docs/plugins.md](docs/plugins.md)**。本文件不复述它，只记「反直觉、看代码看不出来」的规则。

---

## 一、不可破坏的边界

**这两个纯核/纯前端 crate 禁止引入 `tokio` / `reqwest` / `rquickjs` / `rusqlite` / `tracing`**：
`mizar-types`、`mizar-tui-ui`（其中 `mizar-types` **亦禁止**引入 `ratatui` / `crossterm` 及任何内部 crate）。

加了能编过，但架构就废了。需要异步或运行时能力，上移到 `mizar-client` / `mizar-tui` / `mizar-core` 宿主层。

**Merak Desktop 直嵌边界**：**`mizar-client`（及 `mizar-types`）禁止引入 `ratatui` / `crossterm` / `mizar-tui-ui`**，确保跨端宿主运行时零终端依赖。

**workspace 规则**：根 `Cargo.toml` 无 `[package]`；依赖一律进 `[workspace.dependencies]`，crate 内写
`{ workspace = true }`，不写死版本号；每个 crate 带 `[lints] workspace = true`。Rust 固定 **1.98.0**、edition 2024。

**`references/` 已被 `.gitignore` 排除**（`pi`、`deepseek-harness`、`dsh-TUI`、`dsh-ops` 等 TS 原版仓库）。
它们是移植的行为规范来源，**不要 `git add`**，也不要把其中未移植的路径当事实。

---

## 二、命令

```sh
cargo check -p mizar-tui-ui     # 改 UI 的快速回路，秒级
cargo check --workspace       # 全量；改跨 crate 枚举/字段必跑
cargo check --locked -p mizar-core --no-default-features --all-targets
cargo test --locked --workspace --no-fail-fast
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
```

CI 跑后四条，**clippy 的 warning 也算失败**。仓库**没有** xtask / rustfmt.toml / `tempfile` 依赖（`scripts/install.sh` 仅用于安装预编译二进制）。

---

## 三、约定

- **不用 `mod.rs`**：`src/foo.rs` + `src/foo/bar.rs`。生产代码 `use crate::xxx`；测试模块内 `use super::*`。
- 每个 crate 与顶层模块应有 `//!` 模块文档（中文），标注对应的 TS 原版路径。现存例外：`mizar-llm/src/http.rs`。
- 公共错误用 `thiserror`，不用 `anyhow`；统一 `mizar_types::BoxError`。
- **`.lock().unwrap()` 是被接受的**。Mutex 中毒不处理 —— 别引 `parking_lot`、别包一层、别当技术债去「修」。

### 装配与钩子

- **显式组合根**：`MizarRuntime::build(config)` 按固定编译期 DAG 顺序装配全部服务（`LlmRuntime` → `SqliteSessionPersistence` / `SessionStore` → `ToolRuntime` → `SystemPrompt` → `AgentLoop` → 编码工具 / 扩展 / `SpinePlugins`），所有运行时服务均可直接访问，不再存在 `inject` 漏写导致的静默消失。
- **RAII `HookGuard` 是 `#[must_use]`**：`SyncEventBus::on`、`WaterfallHook::register*`、`SerialHook::register*`、`ToolRuntime::register`、`SystemPrompt::section` 与各类 `install_*` 返回 `HookGuard`，**丢弃返回值会立即注销**；必须绑定具名变量、存入宿主集合或交由 `runtime.push_guard(...)` 托管（`MizarRuntime::shutdown` 按 LIFO 逆序卸载）。
- **保持钩子优先级与顺序不变量**：如 `SubagentRuntime` 在 `tools.pre_execute_hook` 最外层挂载（`priority: -100, prepend: true`）、`FileMentions` 在 `agent_loop.pre_step_hook` 以 `priority: -50` 展开、`CompactionRuntime`（`-20`）、`AgentLoop` `<environment_context>`（`-10`）、`Skills`（`5`）。
- **运行期模块开关统一走 `runtime.modules`（`SpinePlugins`）**：核心模块自保护禁止禁用，`mcp` 启动禁用时延后在首次启用时按需连接。

### 工具

| 想要 | `ToolExposure` |
| :--- | :--- |
| 模型直接可调（`read`/`bash`/`grep`） | `Direct` |
| 模型可见但禁止脚本嵌套（`codemode`/`tool_search`） | `ModelOnly` |
| 只给脚本调、不占 context（MCP 默认） | `Codemode` |
| 只靠 `tool_search` 发现 | `Deferred` |
| 屏蔽 | `Hidden` |

选错 = 工具静默不可用。同路径 `write` / `edit` 必须走 `with_file_mutation_queue` 串行。

### 文案语言（别统一成一种）

| 位置 | 语言 |
| :--- | :--- |
| `//!` 与行内注释 | 中文 |
| 工具 `description`、参数 `description`（给模型看） | **英文** |
| 工具运行时错误（给人看） | 中文（`"缺少 path 参数"`） |
| TUI 文案 / commit message | 中文 / **英文** |

把工具描述改成中文会降低模型的工具调用准确率。

---

## 四、测试

- **没有 `tempfile`**：用 `std::env::temp_dir().join(format!("mizar_xxx_{}", std::process::id()))`。
- `mizar-core` 的集成测试由 `tests/integration/main.rs` 聚合、按 `#[cfg(feature)]` 门控 ——
  **新加测试要同步改 `main.rs`**，否则不编译。
- 新行为要带回退即失败的断言。别用 `#[ignore]` / 重试掩盖并发不稳。

---

## 五、提交

```
feat(scope,scope): 一句话

- 说为什么，不复述 diff。跨 crate 改动按 crate 分组。

Verified:
- cargo clippy --workspace --all-targets -- -D warnings: passed (0 warnings).
- cargo test --workspace: 105 passed, 0 failed.
```

`Verified:` 只写**实际跑过**的命令。不加 AI 署名。

---

## 六、坑

- **改 `mizar-types` / `mizar-tui-ui` 的枚举或事件，`mizar-client` / `mizar-tui` 会一起受影响**。`ClientEvent`（`UiEvent`）、`OverlayKind`、`OverlayAction` 跨 crate 传递，一次改完再跑 `cargo check --workspace`，别只 check 单个。
- **feature 组合会漏编**。`mizar-core` 有 `codemode` / `mcp` 两个可选 feature（`mizar-session` 有默认开启的 `bundled`），
  可选依赖的 `use` 必须配 `#[cfg(feature)]`。验最小组合（含测试目标）：`cargo check --locked -p mizar-core --no-default-features --all-targets`。
- **改 crate 结构 / 模块装配或服务依赖时，`docs/plugins.md` 的表格与拓扑图要同步改**，否则立刻过期。
- **沙箱默认关闭**：`restrictToWorkspace` 默认 `false`，`bash` 能碰工作区外。别假设有隔离。
- **本文件会被 `mizar-core::instructions` 加载**，预算 65 536 字节，写长了触发祖先淘汰或正文截断。
- **`pi-emphasis` 是双向契约**：`mizar-tui-ui/src/emphasis.rs` 的 `reply_guidance_prompt()` 生成注入文本，
  同文件渲染器消费它。改一边必须改另一边，测试要同时断言。

---

## 七、从哪读起

| 想做什么 | 看 |
| :--- | :--- |
| 整体架构 | `docs/plugins.md` |
| 跨端纯契约、算法与强类型钩子 | `crates/mizar-types/src/{lib,hooks,client,llm,trajectory}.rs`（`HookGuard`、`SyncEventBus`、`WaterfallHook`、`SerialHook`、`HostExtensionFlags`、`ClientEvent`） |
| 跨端宿主运行时 | `crates/mizar-client/src/lib.rs`（`install_file_mentions`、`install_approval_gate`、`events`、`settings`、`session_view`、`workspace`） |
| 加工具 | `crates/mizar-tools/src/{lib,coding,search}.rs` |
| 加 LLM 协议适配器 / 路由 | `crates/mizar-llm/src/{lib,runtime,http,completions,messages,responses,router,retry,mock}.rs` |
| 会话日志 / SQLite / Recall | `crates/mizar-session/src/{lib,model,persistence,sqlite,recall}.rs` |
| 主循环与 Agent 领域核 | `crates/mizar-core/src/{lib,agent_loop,system_prompt,compaction,instructions,skills,trajectory,session_title,subagent}.rs` |
| UI 渲染与状态机 | `crates/mizar-tui-ui/src/`（`ui.rs`、`markdown.rs`、`features.rs`、`app.rs`、`app/{cache,events,input}.rs`） |
| TUI 宿主循环与 CLI 参数 | `crates/mizar-tui/src/runner.rs`、`runner/{terminal_io,actions}.rs`；CLI 参数表见 `crates/mizar-tui/src/main.rs` 顶部 `//!` |
| 静态组合根与扩展 ID | `crates/mizar-core/src/spine.rs`（`MizarRuntime::build`）与 `crates/mizar-core/src/spine/plugins.rs`（`SpinePlugins`、`KNOWN_HOST_EXTENSION_IDS`、`build_tools_guide_section`） |
| 配置字段 | `crates/mizar-core/src/spine/config.rs`、`config/{settings,route}.rs` + `docs/plugins.md` 第七节 |
