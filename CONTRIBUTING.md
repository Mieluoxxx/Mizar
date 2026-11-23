# Contributing to Mizar

English and Chinese contributions are welcome. 中文 Issue 和 PR 都欢迎。

- [Report a bug or request a feature](https://github.com/Mieluoxxx/Mizar/issues/new/choose). For bugs, include the Mizar version, platform, and a minimal reproduction. Redact secrets and private session data.
- For development setup, see the [developer guide](site/src/content/docs/en/dev/index.mdx). For repository constraints and architecture, see [AGENTS.md](AGENTS.md) and [docs/plugins.md](docs/plugins.md).
- Keep each PR focused, describe the problem and the change, and record the checks you ran. Prefer a `type(scope): summary` title that tells readers what changed.

Run the relevant checks with the Rust toolchain pinned in `rust-toolchain.toml`. The integration tests use `ripgrep` (`rg`):

```sh
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace --no-fail-fast
cargo check --locked -p mizar-core --no-default-features --all-targets
```

For changes to `site/`, use Node.js 24 and pnpm 12, as in the docs workflow:

```sh
pnpm --dir site install --frozen-lockfile
pnpm --dir site build
```
