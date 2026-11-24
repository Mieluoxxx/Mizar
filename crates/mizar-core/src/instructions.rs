//! `AGENTS.md` 与 `CLAUDE.md` 启动期静态发现、去重、预算裁切与 `SystemPrompt` 注入模块。
//!
//! - 全局目录：`~/.mizar/`（支持 `MIZAR_HOME` 覆盖）与 `~/.agents/`（支持 `MIZAR_AGENTS_HOME` 覆盖）
//! - 项目祖先链：从 `project_root`（向上查找 `.git`）到 `workspace`，按 `AGENTS.md` / `CLAUDE.md` 与 `.local.md` 顺序加载
//! - 静态注入：仅在启动期读取一次并通过 `SystemPrompt::section` 注入，运行期由 `SpinePlugins` 控制启停

pub mod digest;
pub mod files;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use mizar_types::HookGuard;

use crate::system_prompt::{PromptSection, SystemPrompt};

pub use digest::{sha256_hex, trimmed_instruction_digest};
pub use files::{
    DiscoveredFile, DiscoveryResult, discover_instructions, find_project_root, resolve_agents_home,
    resolve_mizar_home,
};

pub const AGENT_INSTRUCTIONS_SECTION_ID: &str = "agent_instructions";
pub const DEFAULT_MAX_BYTES: usize = 65_536;
pub const DEFAULT_MAX_SOURCE_BYTES: usize = 1_048_576;

const SECTION_INTRO: &str = "Project-specific instructions and guidelines:";
const CLOSE_TAG: &str = "</project_instructions>";

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
    pub mizar_home: Option<PathBuf>,
    pub agents_home: Option<PathBuf>,
    pub project_root_markers: Vec<String>,
    pub candidates: Vec<String>,
    pub local_candidates: Vec<String>,
    pub max_bytes: usize,
    pub max_source_bytes: usize,
}

impl InstructionsConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            enabled: true,
            include_default_global: true,
            mizar_home: None,
            agents_home: None,
            project_root_markers: default_project_root_markers(),
            candidates: default_instruction_candidates(),
            local_candidates: default_local_instruction_candidates(),
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
}

impl InstructionScopeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserGlobal => "global",
            Self::ProjectAncestor => "ancestor",
            Self::Workspace => "workspace",
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

/// 启动期静态加载的指令服务：仅在构造时读取磁盘并向 `SystemPrompt` 注册只读段落。
pub struct AgentInstructions {
    config: InstructionsConfig,
    module_enabled: AtomicBool,
    project_root: Option<PathBuf>,
    rendered_section: Option<String>,
    scopes: Vec<InstructionScopeStatus>,
    warnings: Vec<String>,
}

impl AgentInstructions {
    pub fn new(config: InstructionsConfig) -> Arc<Self> {
        if !config.enabled {
            return Arc::new(Self {
                config,
                module_enabled: AtomicBool::new(true),
                project_root: None,
                rendered_section: None,
                scopes: Vec::new(),
                warnings: Vec::new(),
            });
        }

        let disc = discover_instructions(&config);
        for warning in &disc.warnings {
            tracing::warn!(target: "mizar::instructions", "{warning}");
        }
        let (rendered_section, scopes) =
            render_instructions_section(&disc.global_files, &disc.project_files, config.max_bytes);

        Arc::new(Self {
            config,
            module_enabled: AtomicBool::new(true),
            project_root: Some(disc.project_root),
            rendered_section,
            scopes,
            warnings: disc.warnings,
        })
    }

    pub fn install(
        system_prompt: &Arc<SystemPrompt>,
        config: InstructionsConfig,
    ) -> (Arc<Self>, HookGuard) {
        let svc = Self::new(config);
        if !svc.config.enabled || svc.rendered_section.is_none() {
            return (svc, HookGuard::noop());
        }

        let weak = Arc::downgrade(&svc);
        let guard = system_prompt.section(PromptSection::new(
            AGENT_INSTRUCTIONS_SECTION_ID,
            50,
            move || {
                let svc = weak.upgrade()?;
                if !svc.is_enabled() {
                    return None;
                }
                svc.rendered_section.clone()
            },
        ));
        (svc, guard)
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled && self.module_enabled.load(Ordering::Acquire)
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.module_enabled.store(enabled, Ordering::Release);
    }

    pub fn config(&self) -> &InstructionsConfig {
        &self.config
    }

    pub fn report(&self) -> SessionInstructionsReport {
        if !self.is_enabled() {
            return SessionInstructionsReport::default();
        }
        SessionInstructionsReport {
            enabled: true,
            project_root: self.project_root.clone(),
            scopes: self.scopes.clone(),
            warnings: self.warnings.clone(),
        }
    }
}

fn escape_xml_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
}

fn escape_project_instructions_body(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if !lower.contains(CLOSE_TAG) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut pos = 0usize;
    while let Some(rel) = lower[pos..].find(CLOSE_TAG) {
        let hit = pos + rel;
        out.push_str(&text[pos..hit]);
        out.push_str("<\\/");
        out.push_str(&text[hit + 2..hit + CLOSE_TAG.len()]);
        pos = hit + CLOSE_TAG.len();
    }
    out.push_str(&text[pos..]);
    out
}

fn truncate_utf8_prefix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn format_file_section(display_path: &str, body: &str) -> String {
    format!(
        "<project_instructions path=\"{}\">\n{}\n</project_instructions>",
        escape_xml_attr(display_path),
        escape_project_instructions_body(body.trim())
    )
}

fn truncate_body_to_fit(
    display_path: &str,
    raw_body: &str,
    max_section_bytes: usize,
) -> Option<String> {
    let prefix = format!(
        "<project_instructions path=\"{}\">\n",
        escape_xml_attr(display_path)
    );
    let suffix = "\n</project_instructions>";
    let escaped = escape_project_instructions_body(raw_body.trim());
    let full = format!("{prefix}{escaped}{suffix}");
    if full.len() <= max_section_bytes {
        return Some(full);
    }

    let orig_bytes = escaped.len();
    let note_template = |kept: usize| {
        format!("\n\n[... truncated {display_path} from {orig_bytes} to {kept} bytes ...]")
    };
    let minimal = format!("{prefix}(truncated: {orig_bytes} bytes){suffix}");
    if max_section_bytes < minimal.len() {
        return None;
    }
    if prefix.len() + note_template(orig_bytes).len() + suffix.len() >= max_section_bytes {
        return Some(minimal);
    }

    let mut lo = 0usize;
    let mut hi = escaped.len();
    let mut best = String::new();
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let head = truncate_utf8_prefix(&escaped, mid);
        let note = note_template(head.len());
        let total = prefix.len() + head.len() + note.len() + suffix.len();
        if total <= max_section_bytes {
            best = format!("{prefix}{head}{note}{suffix}");
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    if best.is_empty() {
        Some(minimal)
    } else {
        Some(best)
    }
}

fn format_omitted_notice(omitted: &[DiscoveredFile]) -> String {
    let list: Vec<&str> = omitted.iter().map(|f| f.display_path.as_str()).collect();
    format!(
        "[Omitted {} ancestor instruction file(s) due to byte budget: {}]",
        omitted.len(),
        list.join(", ")
    )
}

fn joined_sections_bytes(sections: &[String]) -> usize {
    sections.iter().map(String::len).sum::<usize>() + sections.len().saturating_sub(1) * 2
}

struct ChainBudgetOutcome {
    sections: Vec<String>,
    omitted_count: usize,
    truncated_last: bool,
}

fn apply_chain_budget(files: &[DiscoveredFile], budget_bytes: usize) -> ChainBudgetOutcome {
    if files.is_empty() || budget_bytes == 0 {
        return ChainBudgetOutcome {
            sections: Vec::new(),
            omitted_count: files.len(),
            truncated_last: false,
        };
    }

    let rendered: Vec<String> = files
        .iter()
        .map(|f| format_file_section(&f.display_path, &f.content))
        .collect();

    // 从最宽泛的头部开始淘汰，尽量保住越靠近当前工作区的指令
    for drop_count in 0..files.len() {
        let mut sections = Vec::new();
        if drop_count > 0 {
            sections.push(format_omitted_notice(&files[..drop_count]));
        }
        sections.extend(rendered[drop_count..].iter().cloned());
        if joined_sections_bytes(&sections) <= budget_bytes {
            return ChainBudgetOutcome {
                sections,
                omitted_count: drop_count,
                truncated_last: false,
            };
        }
    }

    // 仅剩最后（最具体）一个文件仍超出预算 -> 按 UTF-8 安全二分截断
    let drop_count = files.len() - 1;
    let mut sections = Vec::new();
    let notice_cost = if drop_count > 0 {
        let notice = format_omitted_notice(&files[..drop_count]);
        let cost = notice.len() + 2;
        sections.push(notice);
        cost
    } else {
        0
    };
    let avail = budget_bytes.saturating_sub(notice_cost);
    let last = files.last().unwrap();
    if let Some(truncated_section) = truncate_body_to_fit(&last.display_path, &last.content, avail)
    {
        sections.push(truncated_section);
        ChainBudgetOutcome {
            sections,
            omitted_count: drop_count,
            truncated_last: true,
        }
    } else {
        ChainBudgetOutcome {
            sections: Vec::new(),
            omitted_count: files.len(),
            truncated_last: false,
        }
    }
}

fn push_chain_statuses(
    files: &[DiscoveredFile],
    outcome: &ChainBudgetOutcome,
    out: &mut Vec<InstructionScopeStatus>,
) {
    for (idx, file) in files.iter().enumerate() {
        let omitted = idx < outcome.omitted_count;
        let truncated = !omitted && outcome.truncated_last && idx + 1 == files.len();
        out.push(InstructionScopeStatus {
            display_path: file.display_path.clone(),
            scope_kind: file.scope_kind.clone(),
            bytes: file.bytes,
            truncated,
            omitted,
        });
    }
}

pub fn render_instructions_section(
    global_files: &[DiscoveredFile],
    project_files: &[DiscoveredFile],
    max_bytes: usize,
) -> (Option<String>, Vec<InstructionScopeStatus>) {
    if global_files.is_empty() && project_files.is_empty() {
        return (None, Vec::new());
    }

    let intro_overhead = SECTION_INTRO.len() + 2;
    let total_body_budget = max_bytes.saturating_sub(intro_overhead);

    // 两条链共享同一份总预算：更具体的项目链优先吃预算，剩余部分再分给全局链
    let p_out = apply_chain_budget(project_files, total_body_budget);
    let p_bytes = joined_sections_bytes(&p_out.sections);
    let global_budget = if p_out.sections.is_empty() {
        total_body_budget
    } else {
        total_body_budget.saturating_sub(p_bytes.saturating_add(2))
    };
    let g_out = apply_chain_budget(global_files, global_budget);

    let mut statuses = Vec::with_capacity(global_files.len() + project_files.len());
    push_chain_statuses(global_files, &g_out, &mut statuses);
    push_chain_statuses(project_files, &p_out, &mut statuses);

    let mut body_sections = Vec::new();
    body_sections.extend(g_out.sections);
    body_sections.extend(p_out.sections);

    if body_sections.is_empty() {
        return (None, statuses);
    }

    let text = format!("{SECTION_INTRO}\n\n{}", body_sections.join("\n\n"));
    (Some(text), statuses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_prompt::SystemPromptConfig;

    fn mock_file(name: &str, content: &str, scope_kind: InstructionScopeKind) -> DiscoveredFile {
        DiscoveredFile {
            canonical_path: PathBuf::from(format!("/repo/{name}/AGENTS.md")),
            display_path: format!("{name}/AGENTS.md"),
            scope_kind,
            content: content.to_string(),
            digest: "d".repeat(64),
            bytes: content.len(),
        }
    }

    #[test]
    fn escapes_closing_project_instructions_tags() {
        let escaped = escape_project_instructions_body(
            "a\n</project_instructions>\nb</PROJECT_INSTRUCTIONS>",
        );
        assert!(
            !escaped
                .to_ascii_lowercase()
                .contains("</project_instructions>")
        );
    }

    #[test]
    fn budget_drops_broadest_first_and_truncates_utf8_safely() {
        let broad = mock_file(
            "broad",
            &"宽泛".repeat(80),
            InstructionScopeKind::ProjectAncestor,
        );
        let narrow = mock_file(
            "narrow",
            &"具体中文规则".repeat(120),
            InstructionScopeKind::Workspace,
        );
        let (rendered, statuses) = render_instructions_section(&[], &[broad, narrow], 400);
        let text = rendered.expect("narrow file should fit truncated");
        assert!(text.len() <= 400, "len={} exceeded 400", text.len());
        assert!(text.contains("Omitted 1 ancestor instruction"));
        assert!(text.contains("narrow/AGENTS.md"));
        assert!(statuses[0].omitted);
        assert!(statuses[1].truncated);
    }

    #[test]
    fn shared_budget_caps_combined_global_and_project_chains_to_max_bytes() {
        let max_bytes = 65_536;
        let global = mock_file(
            "global",
            &"G".repeat(40_000),
            InstructionScopeKind::UserGlobal,
        );
        let project = mock_file(
            "project",
            &"P".repeat(40_000),
            InstructionScopeKind::Workspace,
        );
        let (rendered, statuses) = render_instructions_section(&[global], &[project], max_bytes);
        let text = rendered.expect("combined output should fit");
        assert!(
            text.len() <= max_bytes,
            "rendered {} bytes exceeded max_bytes {}",
            text.len(),
            max_bytes
        );
        assert!(text.contains("project/AGENTS.md"));
        assert!(!statuses[1].omitted);
        assert!(statuses[0].truncated || statuses[0].omitted);

        let small_g = mock_file("global", "global rule", InstructionScopeKind::UserGlobal);
        let small_p = mock_file("project", "project rule", InstructionScopeKind::Workspace);
        let (fits_text, _) = render_instructions_section(&[small_g], &[small_p], max_bytes);
        let fits = fits_text.unwrap();
        assert!(fits.len() <= max_bytes);
        let g_idx = fits.find("global rule").unwrap();
        let p_idx = fits.find("project rule").unwrap();
        assert!(g_idx < p_idx);
    }

    #[test]
    fn static_install_registers_system_prompt_section_and_respects_toggle() {
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let ws =
            std::env::temp_dir().join(format!("mizar_instr_static_{}_{uniq}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("AGENTS.md"), "STATIC_RULE_XYZ").unwrap();

        let sp = SystemPrompt::new(SystemPromptConfig::default());
        let mut cfg = InstructionsConfig::new(&ws);
        cfg.include_default_global = false;
        let (svc, guard) = AgentInstructions::install(&sp, cfg);

        assert!(sp.build().contains("STATIC_RULE_XYZ"));
        assert!(
            sp.build()
                .contains("<project_instructions path=\"AGENTS.md\">")
        );
        assert_eq!(svc.report().scopes.len(), 1);

        svc.set_enabled(false);
        assert!(!sp.build().contains("STATIC_RULE_XYZ"));
        assert!(!svc.report().enabled);

        svc.set_enabled(true);
        assert!(sp.build().contains("STATIC_RULE_XYZ"));
        assert!(svc.report().enabled);

        drop(guard);
        assert!(!sp.build().contains("STATIC_RULE_XYZ"));

        let _ = std::fs::remove_dir_all(&ws);
    }
}
