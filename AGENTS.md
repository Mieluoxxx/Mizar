# AGENTS.md

Mizar 是 Rust 编写的终端 AI 编程 Agent（微内核 `cordis` + 多 crate 插件装配）。架构与模块目录按需查 [docs/plugins.md](docs/plugins.md)，本文件只记**全局硬边界与反直觉约束**。

## 架构边界

- **零异步/运行时边界**：`mizar-client-core`、`mizar-tui-ui`、`mizar-llm-types`、`mizar-trajectory-core`、`mizar-image` 禁止引入 `tokio` / `cordis` / `reqwest` / `rquickjs` / `rusqlite`。
- **零终端依赖边界**：`mizar-client` 与 `mizar-client-core` 禁止引入 `ratatui` / `crossterm` / `mizar-tui-ui`（确保 Merak Desktop 可直嵌）。
- **Workspace 依赖**：新增依赖统一进根 `Cargo.toml` 的 `[workspace.dependencies]`，子 crate 写 `{ workspace = true }` 并带 `[lints] workspace = true`。
- **文档与参考仓**：改 crate 结构 / `inject` / 服务名时同步更新 `docs/plugins.md`；`references/` 为 TS 原版参考，**禁止 `git add`**。

## 验证策略

**禁止编写任何测试代码**：不建 `tests/`，不写 `#[cfg(test)]`，不加 `[dev-dependencies]`。

- **静态门禁**：结构、类型、feature 组合与 lint 由 `rustc` 和 `clippy -D warnings` 兜底。
- **行为验收只写用户故事**（见 [docs/stories.md](docs/stories.md)）：UI 交互交给 Computer Use 在真实终端实操验收，代码与 CLI 行为交给独立上下文的 Agent 实跑验证。
- 新增核心能力或修完真坑时，只增补/更新对应用户故事（写清执行方、前置条件、操作步骤与可观测验收结果）。

## CI 与提交

CI 门禁（clippy warning 即失败）：

```sh
cargo check --locked -p mizar-agent-spine --no-default-features --all-targets
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
```

提交信息 `Verified:` 只列实际跑过的命令，不加 AI 署名。
