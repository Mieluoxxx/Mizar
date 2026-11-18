//! 渐进式技能发现、分层注册表、热失效与按需加载工具（参考 `deepseek-harness/packages/skill`）。

pub mod filesystem;
pub mod frontmatter;
pub mod plugin;
pub mod registry;
pub mod types;

pub use filesystem::{
    CUSTOM_RANK, FilesystemSkillConfig, FilesystemSkillProvider, PROJECT_AGENTS_RANK,
    USER_AGENTS_RANK, find_project_root, is_potential_skill_path, resolve_agents_home,
};
pub use frontmatter::{ParsedSkillFile, parse_skill_markdown};
pub use plugin::{
    DEFAULT_CATALOG_DESCRIPTION_MAX_LENGTH, SKILL_CATALOG_SOURCE, SKILL_INVOCATION_SOURCE,
    SKILL_TOOL_NAME, SkillsConfig, SkillsPlugin, extract_skill_gestures, list_workspace_skills,
};
pub use registry::{
    EVENT_SKILLS_CHANGE, SkillProvider, SkillRegistry, SkillRegistryConfig, SkillsContextExt,
};
pub use types::{
    BUNDLED_SKILL_RANK, RUNTIME_SKILL_RANK, SkillCandidate, SkillCatalogSnapshot, SkillDefinition,
    SkillInvocationPolicy, SkillLookupOptions, SkillProviderObservation, SkillRegistration,
    SkillResourceBase, SkillSource, SkillSummary, is_skill_name, render_skill_content,
};
