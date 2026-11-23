## What

<!-- Describe what changes for users or contributors. Keep this PR focused on one change. -->

## Why

<!-- Explain the problem. Use `Fixes #123` if merging this PR should close an issue in this repo. -->

## Verified

<!-- List commands or manual checks you actually ran and their results. For fixes, say what failed before the change. If nothing was run, write "Not tested". -->

### Checklist

<!-- Check only the items that apply. -->

- [ ] Cross-crate API or type changes: ran `cargo check --locked --workspace`.
- [ ] `mizar-core` changes: ran `cargo check --locked -p mizar-core --no-default-features --all-targets`.
- [ ] Crate structure, module assembly, or service dependencies changed: updated `docs/plugins.md`.
- [ ] Reply emphasis changed: updated and tested both the prompt and renderer.
- [ ] Dependency changes preserve the crate boundaries documented in `AGENTS.md`; no files under `references/` are included.
- [ ] Docs site changes: ran `pnpm --dir site build`.
