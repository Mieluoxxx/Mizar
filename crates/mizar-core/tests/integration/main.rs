mod agent_core;
#[cfg(all(feature = "codemode", feature = "mcp"))]
mod coding_stack;
mod compaction_recall_e2e;
mod instructions_e2e;
mod session_persistence_e2e;
#[cfg(all(feature = "codemode", feature = "mcp"))]
mod spine_plugins;
