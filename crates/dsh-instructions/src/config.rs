use std::path::PathBuf;

pub const AGENT_INSTRUCTIONS_SOURCE: &str = "agent-instructions";
pub const DEFAULT_MAX_BYTES: usize = 65_536;
pub const DEFAULT_MAX_SOURCE_BYTES: usize = 1_048_576;

pub fn default_instruction_candidates() -> Vec<String> {
    vec!["AGENTS.md".to_string(), "CLAUDE.md".to_string()]
}

pub fn default_local_instruction_candidates() -> Vec<String> {
    vec!["AGENTS.local.md".to_string(), "CLAUDE.local.md".to_string()]
}

pub fn default_project_root_markers() -> Vec<String> {
    vec![".git".to_string()]
}

#[derive(Debug, Clone)]
pub struct InstructionsConfig {
    pub workspace: PathBuf,
    pub enabled: bool,
    pub include_default_global: bool,
    pub dsh_home: Option<PathBuf>,
    pub agents_home: Option<PathBuf>,
    pub project_root_markers: Vec<String>,
    pub candidates: Vec<String>,
    pub local_candidates: Vec<String>,
    pub follow_subdirs: bool,
    pub max_bytes: usize,
    pub max_source_bytes: usize,
}

impl InstructionsConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            enabled: true,
            include_default_global: true,
            dsh_home: None,
            agents_home: None,
            project_root_markers: default_project_root_markers(),
            candidates: default_instruction_candidates(),
            local_candidates: default_local_instruction_candidates(),
            follow_subdirs: true,
            max_bytes: DEFAULT_MAX_BYTES,
            max_source_bytes: DEFAULT_MAX_SOURCE_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstructionScopeKind {
    UserGlobal,
    ProjectAncestor,
    Workspace,
    Subdir,
}

impl InstructionScopeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserGlobal => "global",
            Self::ProjectAncestor => "ancestor",
            Self::Workspace => "workspace",
            Self::Subdir => "subdir",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionScopeStatus {
    pub display_path: String,
    pub scope_kind: InstructionScopeKind,
    pub bytes: usize,
    pub truncated: bool,
    pub omitted: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionInstructionsReport {
    pub enabled: bool,
    pub project_root: Option<PathBuf>,
    pub scopes: Vec<InstructionScopeStatus>,
    pub warnings: Vec<String>,
}
