<p align="center">
  <img src="site/public/favicon.svg" width="44" alt="Mizar" />
</p>

<h1 align="center">Mizar</h1>

<p align="center">
  Terminal-native AI coding workbench written in Rust.
</p>

<p align="center">
  <a href="https://github.com/Mieluoxxx/Mizar/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Mieluoxxx/Mizar/ci.yml?branch=main&label=CI" alt="CI" /></a>
  <a href="https://github.com/Mieluoxxx/Mizar/releases/latest"><img src="https://img.shields.io/github/v/release/Mieluoxxx/Mizar" alt="Latest Release" /></a>
  <a href="https://mieluoxxx.github.io/Mizar/en/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-0969da" alt="Docs" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Mieluoxxx/Mizar" alt="MIT License" /></a>
</p>

<p align="center">
  <a href="https://mieluoxxx.github.io/Mizar/en/">Docs</a> · <a href="https://mieluoxxx.github.io/Mizar/">中文文档</a> · <a href="https://github.com/Mieluoxxx/Mizar/releases">Releases</a> · <a href="docs/plugins.md">Architecture</a> · <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

**Mizar** is a Rust workspace (10-crate static `MizarRuntime` + `mizar-tui`) for heavy command-line coding workflows—featuring Fullscreen/Inline dual viewports, headless `mizar exec`, persistent background `subagent` workers, local `!` / `!!` shell execution, hierarchical `AGENTS.md` rules, MCP servers, and an isolated QuickJS orchestration sandbox.

### Quick Start

Install the prebuilt binary on macOS (Apple Silicon / Intel) or Linux (`x86_64` / `arm64`) from [GitHub Releases](https://github.com/Mieluoxxx/Mizar/releases):

```sh
curl -fsSL https://raw.githubusercontent.com/Mieluoxxx/Mizar/main/scripts/install.sh | sh
```

Once installed, check or upgrade to the latest release anytime with `mizar update` (set `MIZAR_SKIP_VERSION_CHECK=1` to disable startup version checks). Note: the built-in `grep` tool requires `ripgrep` (`rg`) on `PATH`.

Or build from source (requires **Rust 1.98.0+**, edition 2024, pinned in [`rust-toolchain.toml`](rust-toolchain.toml)):

```sh
# Launch offline demo mode (when no API key or models.json is configured)
cargo run -p mizar-tui --bin mizar

# Or connect directly with an API key
export DEEPSEEK_API_KEY="sk-..."   # also supports OPENAI_API_KEY / ANTHROPIC_API_KEY
cargo run -p mizar-tui --bin mizar -- -C /path/to/repo

# Or install locally from the workspace
cargo install --path crates/mizar-tui
```

### Highlights

- **Dual-Mode TUI (`F3` / `--inline`)**: Switch seamlessly between a full-screen dashboard and an Inline viewport that preserves terminal scrollback, with `@file#L1-40` mentions, split diagnostics (`Ctrl+B`), and native copy toggle (`F2`).
- **Local Bash (`!` / `!!`)**: Run `!cmd` to feed command output into the next model turn, or `!!cmd` to run locally with zero token cost.
- **Rules & Skills**: Hierarchical `AGENTS.md` / `CLAUDE.md` discovery under a `64 KiB` budget plus hot-reloaded `SKILL.md` workflows.
- **MCP & Codemode**: Connect Stdio/HTTP MCP servers via `mcp.json`, discover large tool catalogs on demand with `tool_search`, and batch-call tools inside an isolated QuickJS (`rquickjs`) sandbox.
- **Providers, Compaction & Execution**: OpenAI Completions/Responses, Anthropic Messages, and custom gateways configured via `~/.mizar/models.json` or live `/settings`, backed by WAL SQLite sessions (`/tree` branching, single-stage checkpoint `/compact` + static `recall` with FTS5, `/trace` profiler, headless `mizar exec --trace-out`, streaming `--trace-jsonl` trajectory logs, and persistent `subagent`).

### Documentation & Development

- **Online Docs**: [English](https://mieluoxxx.github.io/Mizar/en/) · [中文文档](https://mieluoxxx.github.io/Mizar/) (deployed to GitHub Pages via [`.github/workflows/pages.yml`](.github/workflows/pages.yml))
- **Docs Site Source**: [`site/`](site/) (Chinese & English Astro + Pagefind site)
  ```sh
  pnpm --dir site install --frozen-lockfile
  pnpm --dir site dev
  ```
- **Architecture & Crates**: [`docs/plugins.md`](docs/plugins.md) (10 workspace crates assembled via static `MizarRuntime` and typed `mizar-types::hooks`)
- **Contributing**: [`CONTRIBUTING.md`](CONTRIBUTING.md) · **Repository Rules**: [`AGENTS.md`](AGENTS.md)

### License

[MIT](LICENSE)
