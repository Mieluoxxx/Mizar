<p align="center">
  <img src="site/public/favicon.svg" width="36" alt="Mizar" />
</p>

<h1 align="center">Mizar</h1>

<p align="center">
  Terminal-native AI coding workbench written in Rust.
</p>

<p align="center">
  <a href="site/">Docs (中文 / EN)</a> · <a href="docs/plugins.md">Architecture</a> · <a href="AGENTS.md">Contributing</a>
</p>

---

**Mizar** is a Rust workspace (`cordis` plugin microkernel + `mizar-tui`) for heavy command-line coding workflows—featuring Fullscreen/Inline dual viewports, local `!` / `!!` shell execution, hierarchical `AGENTS.md` rules, MCP servers, and an isolated QuickJS orchestration sandbox.

### Quick Start

Requires **Rust 1.98.0+** (edition 2024, pinned in [`rust-toolchain.toml`](rust-toolchain.toml)).

```sh
# Launch offline demo mode (when no API key or models.json is configured)
cargo run -p mizar-tui --bin mizar

# Or connect directly with an API key
export DEEPSEEK_API_KEY="sk-..."   # also supports OPENAI_API_KEY / ANTHROPIC_API_KEY
cargo run -p mizar-tui --bin mizar -- -C /path/to/repo
```

Build and install the `mizar` binary locally:

```sh
cargo install --path crates/mizar-tui
```

### Highlights

- **Dual-Mode TUI (`F3` / `--inline`)**: Switch seamlessly between a full-screen dashboard and an Inline viewport that preserves terminal scrollback, with `@file#L1-40` mentions, split diagnostics (`Ctrl+B`), and native copy toggle (`F2`).
- **Local Bash (`!` / `!!`)**: Run `!cmd` to feed command output into the next model turn, or `!!cmd` to run locally with zero token cost.
- **Rules & Skills**: Hierarchical `AGENTS.md` / `CLAUDE.md` discovery under a `64 KiB` budget plus hot-reloaded `SKILL.md` workflows.
- **MCP & Codemode**: Connect Stdio/HTTP MCP servers via `mcp.json`, discover large tool catalogs on demand with `tool_search`, and batch-call tools inside an isolated QuickJS (`rquickjs`) sandbox.
- **Providers & Persistence**: OpenAI Completions/Responses, Anthropic Messages, and custom gateways configured via `~/.mizar/models.json`, backed by WAL SQLite sessions (`/tree` branching & `/trace` profiler).

### Documentation & Development

- **Bilingual Docs Site**: [`site/`](site/) (Chinese & English Astro + Pagefind site; deployed via [`.github/workflows/pages.yml`](.github/workflows/pages.yml))
  ```sh
  pnpm --dir site install --frozen-lockfile
  pnpm --dir site dev
  ```
- **Architecture & Crates**: [`docs/plugins.md`](docs/plugins.md) (28 workspace crates built on `cordis`)
- **Agent & Contributor Rules**: [`AGENTS.md`](AGENTS.md)

### License

MIT
