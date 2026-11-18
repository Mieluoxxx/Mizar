# AGENTS.md

给在本仓库工作的 AI coding agent 的说明。

**Mizar** = Rust 重写的终端 AI 编程 Agent：`mizar`（`mizar-tui`）是唯一可执行入口，`cordis` 是自研微内核，
28 个 crate 靠 `inject` / `provide` 装配成「流式调模型 → 执行工具 → 回写会话」的主循环。

架构权威文本是 **[docs/plugins.md](docs/plugins.md)**。本文件不复述它，只记「反直觉、看代码看不出来」的规则。

---

## 一、不可破坏的边界

**这三个 crate 禁止引入 `tokio` / `cordis` / `reqwest` / `rquickjs` / `rusqlite`**：
`mizar-tui-ui`、`mizar-llm-types`、`mizar-trajectory-core`。

加了能编过，但架构就废了。需要异步或运行时能力，上移到 `mizar-tui` / `mizar-agent-spine` 宿主层。

**workspace 规则**：根 `Cargo.toml` 无 `[package]`；依赖一律进 `[workspace.dependencies]`，crate 内写
`{ workspace = true }`，不写死版本号；每个 crate 带 `[lints] workspace = true`。Rust 固定 **1.98.0**、edition 2024。

**`references/` 已被 `.gitignore` 排除**（`pi`、`deepseek-harness`、`dsh-TUI`、`dsh-ops` 等 TS 原版仓库）。
它们是移植的行为规范来源，**不要 `git add`**，也不要把其中未移植的路径当事实。

---

## 二、命令

```sh
cargo check -p mizar-tui-ui     # 改 UI 的快速回路，秒级
cargo check --workspace       # 全量；改跨 crate 枚举/字段必跑
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

CI 跑后三条，**clippy 的 warning 也算失败**。仓库**没有** xtask / scripts / rustfmt.toml / `tempfile` 依赖。

---

## 三、约定

- **不用 `mod.rs`**：`src/foo.rs` + `src/foo/bar.rs`。生产代码 `use crate::xxx`；测试模块内 `use super::*`（41 处都如此）。
- 每个 crate 应有 `//!` 模块文档（中文），标注对应的 TS 原版路径。现存例外：`mizar-llm-http`。
- 公共错误用 `thiserror`，不用 `anyhow`；统一 `cordis::BoxError`。
- **`.lock().unwrap()` 是被接受的（204 处）**。Mutex 中毒不处理 —— 别引 `parking_lot`、别包一层、别当技术债去「修」。

### 插件

```rust
impl Plugin for XxxPlugin {
    type Config = ();
    fn name(&self) -> Option<&str> { Some("xxx-plugin") }   // kebab-case
    fn inject(&self) -> Inject { Inject::from(["tools"]) }  // 服务名 camelCase
    fn apply(&self, ctx, cfg) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> { /* ... */ }
}
```

- **`inject` 漏写不 panic**：`ctx.service()` 返回 `MissingInject`，插件进 `FAILED`，工具**静默消失** —— 比 panic 难查得多。
- **副作用必须包 `ctx.effect("id", ...)`**，id 用 `"<crate>-<动作>"`。卸载时倒序自动清理，这是核心不变量。
- 服务名：`llm` `tools` `sessions` `systemPrompt` `agentLoop` `skills` `mcp` `trajectory`。
  一律走 `<Name>ContextExt`（`ctx.tools()?`），不裸取。

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
- `mizar-agent-spine` 的集成测试由 `tests/integration/main.rs` 聚合、按 `#[cfg(feature)]` 门控 ——
  **新加测试要同步改 `main.rs`**，否则不编译。
- `cordis/tests/compat.rs` 用 `compat_NN_<描述>` 对齐 TS 原版顺序，编号保持连续。
- 新行为要带回退即失败的断言。别用 `#[ignore]` / 重试掩盖并发不稳。

---

## 五、提交

```
feat(scope,scope): 一句话

- 说为什么，不复述 diff。跨 crate 改动按 crate 分组。

Verified:
- cargo clippy --workspace --all-targets -- -D warnings: passed (0 warnings).
- cargo test --workspace: 105 passed, 0 failed (1 ignored doctest in cordis).
```

`Verified:` 只写**实际跑过**的命令。不加 AI 署名。

---

## 六、坑

- **改 `mizar-tui-ui` 的枚举/结构体字段，`mizar-tui` 的构造点会一起断**。`OverlayKind` / `OverlayAction` / `UiEvent`
  横跨两个 crate，一次改完再跑 `cargo check --workspace`，别只 check 单个。
- **feature 组合会漏编**。`mizar-agent-spine` 有 `legacy`/`coding`/`sqlite`/`trajectory`/`router`，
  可选依赖的 `use` 必须配 `#[cfg(feature)]`。验最小组合：`cargo check -p mizar-agent-spine --no-default-features`。
- **改 crate 结构 / `inject` / 服务名时，`docs/plugins.md` 的表格与拓扑图要同步改**，否则立刻过期。
- **沙箱默认关闭**：`restrictToWorkspace` 默认 `false`，`bash` 能碰工作区外。别假设有隔离。
- **本文件会被 `mizar-instructions` 加载**，预算 65 536 字节，写长了触发祖先淘汰或正文截断。
- **`pi-emphasis` 是双向契约**：`mizar-tui-ui/src/emphasis.rs` 的 `reply_guidance_prompt()` 生成注入文本，
  同文件渲染器消费它。改一边必须改另一边，测试要同时断言。

---

## 七、从哪读起

| 想做什么 | 看 |
| :--- | :--- |
| 整体架构 | `docs/plugins.md` |
| 加工具（最小样例 41 行） | `crates/mizar-tool-time/src/lib.rs` |
| 加 LLM 协议适配器 | `crates/mizar-llm-mock/src/lib.rs` |
| 主循环 | `crates/mizar-agent-loop/src/lib.rs` |
| 工具管线（三道 waterfall） | `crates/mizar-tools/src/lib.rs` |
| 微内核 | `crates/cordis/src/`（`fiber.rs` 状态机、`context.rs` 注入） |
| UI 渲染 | `crates/mizar-tui-ui/src/`（`ui.rs` 布局、`markdown.rs`、`app.rs`） |
| CLI 参数 | `crates/mizar-tui/src/main.rs` 顶部 `//!` 即完整参数表 |
| 装配顺序 | `crates/mizar-agent-spine/src/lib.rs` 的 `apply` |
| 配置字段 | `crates/mizar-agent-spine/src/config.rs` + `docs/plugins.md` 第九节 |