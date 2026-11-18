#[cfg(all(feature = "legacy", feature = "trajectory", feature = "router"))]
mod agent_core;
#[cfg(feature = "coding")]
mod coding_stack;
#[cfg(all(feature = "coding", feature = "sqlite"))]
mod instructions_e2e;
#[cfg(all(feature = "legacy", feature = "sqlite", feature = "trajectory"))]
mod session_persistence_e2e;
