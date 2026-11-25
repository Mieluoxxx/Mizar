//! `AGENTS.md` 与 `CLAUDE.md` 启动期静态发现、去重、预算裁切与 `SystemPrompt` 注入插件。
//!
//! - 全局目录：`~/.mizar/`（支持 `MIZAR_HOME` 覆盖）与 `~/.agents/`（支持 `MIZAR_AGENTS_HOME` 覆盖）
//! - 项目祖先链：从 `project_root`（向上查找 `.git`）到 `workspace`，按 `AGENTS.md` / `CLAUDE.md` 与 `.local.md` 顺序加载
//! - 静态注入：仅在启动期读取一次并通过 `SystemPrompt::section` 注入（`order = 80`），随插件卸载自动清理

pub mod config;
pub mod digest;
pub mod files;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cordis::{BoxFuture, Context, CordisError, Service};
use mizar_system_prompt::{PromptSection, SystemPrompt};

pub use config::{
    AGENT_INSTRUCTIONS_SECTION_ID, DEFAULT_MAX_BYTES, DEFAULT_MAX_SOURCE_BYTES,
    InstructionScopeKind, InstructionScopeStatus, InstructionsConfig, SessionInstructionsReport,
    default_instruction_candidates, default_local_instruction_candidates,
    default_project_root_markers,
};
pub use digest::{sha256_hex, trimmed_instruction_digest};
pub use files::{
    DiscoveredFile, DiscoveryResult, discover_instructions, find_project_root, resolve_agents_home,
    resolve_mizar_home,
};

const SECTION_INTRO: &str = "Project-specific instructions and guidelines:";
const CLOSE_TAG: &str = "</project_instructions>";

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
    pub fn is_enabled(&self) -> bool {
        self.config.enabled && self.module_enabled.load(Ordering::Acquire)
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.module_enabled.store(enabled, Ordering::Release);
    }

    pub fn config(&self) -> &InstructionsConfig {
        &self.config
    }

    pub fn rendered_section(&self) -> Option<&str> {
        self.rendered_section.as_deref()
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

impl Service for AgentInstructions {
    type Config = InstructionsConfig;

    fn service_name() -> &'static str {
        "agentInstructions"
    }

    fn plugin_name() -> &'static str {
        "agent-instructions"
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        if !config.enabled {
            return Ok(Arc::new(Self {
                config,
                module_enabled: AtomicBool::new(true),
                project_root: None,
                rendered_section: None,
                scopes: Vec::new(),
                warnings: Vec::new(),
            }));
        }

        let disc = discover_instructions(&config);
        if !disc.warnings.is_empty() {
            let logger = ctx.logger_named("instructions");
            for warning in &disc.warnings {
                logger.warn(warning.clone());
            }
        }
        let (rendered_section, scopes) =
            render_instructions_section(&disc.global_files, &disc.project_files, config.max_bytes);

        Ok(Arc::new(Self {
            config,
            module_enabled: AtomicBool::new(true),
            project_root: Some(disc.project_root),
            rendered_section,
            scopes,
            warnings: disc.warnings,
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let ctx_clone = ctx.clone();
        Box::pin(async move {
            if !self.config.enabled || self.rendered_section.is_none() {
                return Ok(());
            }

            let weak = Arc::downgrade(&self);
            let prompt_fiber = ctx_clone.try_inject(["systemPrompt"], move |c| {
                let weak = weak.clone();
                async move {
                    let sp = c.service::<SystemPrompt>("systemPrompt")?;
                    c.effect("instructions.system-prompt", move || {
                        Ok::<_, CordisError>(Some(sp.section(PromptSection::new(
                            AGENT_INSTRUCTIONS_SECTION_ID,
                            80,
                            move || {
                                let svc = weak.upgrade()?;
                                if !svc.is_enabled() {
                                    return None;
                                }
                                svc.rendered_section.clone()
                            },
                        ))))
                    })?;
                    Ok::<_, CordisError>(())
                }
            })?;
            if ctx_clone.get::<SystemPrompt>("systemPrompt").is_some() {
                prompt_fiber.wait_settled().await?;
            }

            Ok(())
        })
    }
}

pub trait InstructionsContextExt {
    fn agent_instructions(&self) -> Result<Arc<AgentInstructions>, CordisError>;
}

impl InstructionsContextExt for Context {
    fn agent_instructions(&self) -> Result<Arc<AgentInstructions>, CordisError> {
        self.service::<AgentInstructions>("agentInstructions")
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
