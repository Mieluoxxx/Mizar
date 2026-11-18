use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use dsh_agent_loop::{
    AgentPreStepEvent, AgentToolResultEvent, EVENT_AGENT_PRE_STEP, EVENT_AGENT_TOOL_RESULT,
};
use dsh_session::SessionEvent;
use dsh_tools::{
    ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolExposure, ToolOutput,
    ToolsContextExt,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::filesystem::{FilesystemSkillConfig, FilesystemSkillProvider};
use crate::registry::{SkillProvider, SkillRegistry, SkillsContextExt};
use crate::types::{
    SkillLookupOptions, SkillSummary, escape_xml_text, is_skill_name, render_skill_content,
};

pub const SKILL_TOOL_NAME: &str = "skill";
pub const SKILL_CATALOG_SOURCE: &str = "skill-catalog";
pub const SKILL_INVOCATION_SOURCE: &str = "skill-invocation";
pub const DEFAULT_CATALOG_DESCRIPTION_MAX_LENGTH: usize = 500;

#[derive(Debug, Clone)]
pub struct SkillsConfig {
    pub workspace: PathBuf,
    pub enabled: bool,
    pub include_default_roots: bool,
    pub agents_home: Option<PathBuf>,
    pub custom_skill_dirs: Vec<PathBuf>,
    pub bundled_skill_dir: Option<PathBuf>,
    pub catalog_description_max_length: usize,
}

impl SkillsConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            enabled: true,
            include_default_roots: true,
            agents_home: None,
            custom_skill_dirs: Vec::new(),
            bundled_skill_dir: None,
            catalog_description_max_length: DEFAULT_CATALOG_DESCRIPTION_MAX_LENGTH,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CatalogState {
    entries: Vec<(String, String)>,
    incomplete: bool,
}

#[derive(Debug, Clone)]
struct PublishedCatalogRecord {
    seq: u64,
    state: CatalogState,
}

fn normalize_catalog_description(raw: &str, max_len: usize) -> String {
    let max_len = max_len.max(3);
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let char_count = collapsed.chars().count();
    if char_count <= max_len {
        return collapsed;
    }
    let truncated: String = collapsed.chars().take(max_len - 3).collect();
    format!("{truncated}...")
}

fn build_catalog_state(
    skills: &[SkillSummary],
    max_desc_len: usize,
    incomplete: bool,
) -> CatalogState {
    let entries = skills
        .iter()
        .filter(|s| s.is_model_invocable())
        .map(|s| {
            (
                s.name.clone(),
                normalize_catalog_description(&s.description, max_desc_len),
            )
        })
        .collect();
    CatalogState {
        entries,
        incomplete,
    }
}

fn render_catalog_lines(entries: &[(String, String)]) -> Vec<String> {
    entries
        .iter()
        .map(|(name, desc)| format!("- `{name}`: {}", escape_xml_text(desc)))
        .collect()
}

fn render_catalog_message(state: &CatalogState, is_update: bool) -> String {
    let mut lines = vec!["<system-reminder>".to_string()];

    if is_update {
        lines.push(if state.incomplete {
            "Skill discovery is incomplete. This catalog of currently available skills replaces every earlier available-skills list in this session:".to_string()
        } else {
            "The available skill catalog changed. This complete catalog replaces every earlier available-skills list in this session:".to_string()
        });
    } else {
        lines.push(if state.entries.is_empty() {
            "No skills could be confirmed in this discovery attempt.".to_string()
        } else {
            "A skill is a reusable set of task-specific instructions. The following skills are available in this session:".to_string()
        });
    }

    lines.push(String::new());
    lines.push("<available_skills>".to_string());
    lines.extend(render_catalog_lines(&state.entries));
    lines.push("</available_skills>".to_string());
    lines.push(String::new());

    if state.incomplete {
        lines.push("Some skills could not be checked. Only confirmed skills are listed; a lower-priority definition of the same name may be selected when a preferred source is unavailable. Discovery will be retried on the next request.".to_string());
        lines.push(String::new());
    }

    if state.entries.is_empty() {
        lines.push(if is_update {
            if state.incomplete {
                "No skills could be confirmed in this discovery attempt. Do not use names from earlier skill catalogs.".to_string()
            } else {
                "No skills are currently available through the `skill` tool. Do not use names from earlier skill catalogs.".to_string()
            }
        } else {
            "Wait for an updated catalog before calling the `skill` tool. Do not infer skill instructions from an empty catalog.".to_string()
        });
    } else if is_update {
        lines.push("Use only names in this replacement catalog. If the user names a listed skill, or the task clearly matches its description, call the `skill` tool with the exact name before acting.".to_string());
    } else {
        lines.push("If the user names a skill, or the task clearly matches a skill's description, call the `skill` tool with the exact skill name before taking task actions. Load all applicable skills, then follow their full instructions. This catalog contains summaries only; do not infer or follow a skill's instructions until it has been loaded.".to_string());
    }

    lines.push("A user may also invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it, and do not call the `skill` tool again for that skill.".to_string());
    lines.push("</system-reminder>".to_string());
    lines.join("\n")
}

/// 从用户消息文本中提取空白定界的 `/<skill-name>` 手势（保持首次出现顺序，避免误匹配 `/usr/bin` 或 `5/8`）。
pub fn extract_skill_gestures(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let is_boundary = i == 0 || (bytes[i - 1] as char).is_whitespace();
        if is_boundary && bytes[i] == b'/' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && !(bytes[end] as char).is_whitespace() {
                end += 1;
            }
            if end > start {
                let token = &text[start..end];
                if is_skill_name(token) && !out.iter().any(|existing| existing == token) {
                    out.push(token.to_string());
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }

    out
}

fn create_skill_tool(registry: Arc<SkillRegistry>, workspace: PathBuf) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        SKILL_TOOL_NAME,
        "Load the full instructions for a skill. Call it before acting on a task that names or clearly matches a skill in the session skill catalog.",
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The exact skill name from the available skills list."
                }
            },
            "required": ["name"]
        }),
        move |args: Value, exec| {
            let reg = Arc::clone(&registry);
            let ws = workspace.clone();
            async move {
                let raw_name = args
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or("");
                if !is_skill_name(raw_name) {
                    return Ok(ToolOutput::error(format!(
                        "invalid skill name \"{raw_name}\""
                    )));
                }

                let opts = SkillLookupOptions {
                    cwd: Some(ws),
                    cancel: exec.cancel,
                };
                let summary = reg
                    .list(opts.clone())
                    .await
                    .into_iter()
                    .find(|s| s.name == raw_name);
                let Some(summary) = summary else {
                    return Ok(ToolOutput::error(format!(
                        "skill \"{raw_name}\" is unknown or no longer available"
                    )));
                };
                if !summary.is_model_invocable() {
                    return Ok(ToolOutput::error(format!(
                        "skill \"{raw_name}\" is not available for model invocation"
                    )));
                }

                let def = match reg.get(raw_name, opts).await {
                    Ok(Some(d)) => d,
                    Ok(None) => {
                        return Ok(ToolOutput::error(format!(
                            "skill \"{raw_name}\" is unknown or no longer available"
                        )));
                    }
                    Err(err) => return Ok(ToolOutput::error(err)),
                };

                if !def.summary.is_model_invocable() {
                    return Ok(ToolOutput::error(format!(
                        "skill \"{raw_name}\" is not available for model invocation"
                    )));
                }

                let rendered = render_skill_content(&def);
                Ok(ToolOutput::text(rendered).with_details(json!({
                    "name": def.summary.name,
                    "provider": def.summary.provider,
                    "source": def.summary.source,
                    "resourceBase": def.summary.resource_base,
                })))
            }
        },
    )
    .with_exposure(ToolExposure::Direct)
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_annotations(ToolAnnotations {
        read_only: true,
        destructive: false,
        idempotent: true,
        open_world: false,
    })
}

#[derive(Default, Clone, Copy)]
pub struct SkillsPlugin;

impl Plugin for SkillsPlugin {
    type Config = SkillsConfig;

    fn name(&self) -> Option<&str> {
        Some("skills-plugin")
    }

    fn inject(&self) -> Inject {
        Inject::from(["tools", "skills"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            if !config.enabled {
                return Ok(None);
            }

            let registry = ctx.skills()?;

            let fs_provider = Arc::new(FilesystemSkillProvider::new(
                ctx.clone(),
                FilesystemSkillConfig {
                    provider_name: "filesystem".to_string(),
                    include_default_roots: config.include_default_roots,
                    agents_home: config.agents_home.clone(),
                    custom_skill_dirs: config.custom_skill_dirs.clone(),
                    bundled_skill_dir: config.bundled_skill_dir.clone(),
                },
                Some(config.workspace.clone()),
            ));
            let reg_cleanup = registry
                .register_provider(Arc::clone(&fs_provider) as Arc<dyn SkillProvider>)
                .map_err(|e| -> BoxError { e.into() })?;
            ctx.effect("skills.filesystemProvider", move || {
                Ok::<_, cordis::CordisError>(Some(reg_cleanup))
            })?;

            let tools = ctx.tools()?;
            let skill_tool = create_skill_tool(Arc::clone(&registry), config.workspace.clone());
            ctx.effect("skills.skillTool", || {
                Ok::<_, dsh_tools::ToolError>(Some(tools.register(skill_tool)?))
            })?;

            // 监听 write / edit 工具写入完成，若命中任何技能目录则同步刷新缓存
            {
                let reg_for_mut = Arc::clone(&registry);
                let fs_for_mut = Arc::clone(&fs_provider);
                let ws_for_mut = config.workspace.clone();
                ctx.on(EVENT_AGENT_TOOL_RESULT, move |ev: &AgentToolResultEvent| {
                    if ev.is_error || !matches!(ev.name.as_str(), "write" | "edit") {
                        return;
                    }
                    let path_opt = ev
                        .details
                        .as_ref()
                        .and_then(|d| d.get("path"))
                        .and_then(Value::as_str);
                    if let Some(raw_path) = path_opt {
                        let target = if PathBuf::from(raw_path).is_absolute() {
                            PathBuf::from(raw_path)
                        } else {
                            ws_for_mut.join(raw_path)
                        };
                        if fs_for_mut.matches_mutation_path(&target, Some(&ws_for_mut)) {
                            reg_for_mut.invalidate_cache();
                        }
                    }
                })?;
            }

            // 在每步发送模型请求前：1）增量注入 <available_skills> 目录；2）展开用户 /<name> 技能手势
            let published_catalogs: Arc<Mutex<HashMap<String, PublishedCatalogRecord>>> =
                Arc::new(Mutex::new(HashMap::new()));
            let pre_step_ctx = ctx.clone();
            ctx.on_serial::<AgentPreStepEvent, (), _, _>(
                EVENT_AGENT_PRE_STEP,
                move |ev: AgentPreStepEvent| {
                    let reg = Arc::clone(&registry);
                    let ctx_inner = pre_step_ctx.clone();
                    let session = Arc::clone(&ev.session);
                    let cancel = ev.cancel.clone();
                    let ws = config.workspace.clone();
                    let max_desc_len = config.catalog_description_max_length.max(3);
                    let published_map = Arc::clone(&published_catalogs);

                    async move {
                        if cancel.is_cancelled() {
                            return None;
                        }
                        let lookup = SkillLookupOptions {
                            cwd: Some(ws),
                            cancel: cancel.clone(),
                        };

                        // 1. 若当前会话声明了 `skill` 工具，按需注入或更新 <available_skills> 目录
                        let skill_tool_visible = ctx_inner
                            .tools()
                            .map(|rt| {
                                rt.list_declared(Some(&session.id))
                                    .into_iter()
                                    .any(|t| t.name == SKILL_TOOL_NAME)
                            })
                            .unwrap_or(false);

                        if skill_tool_visible {
                            let snap = reg.snapshot(lookup.clone()).await;
                            let next_state =
                                build_catalog_state(&snap.skills, max_desc_len, !snap.complete);

                            let prev_record =
                                published_map.lock().unwrap().get(&session.id).cloned();
                            let (prev_still_visible, had_catalog_notice) =
                                session.with_visible_entries(|entries| {
                                    let still_vis = prev_record.as_ref().is_some_and(|rec| {
                                        entries.iter().any(|entry| {
                                            entry.seq == rec.seq
                                                && matches!(
                                                    &entry.event,
                                                    SessionEvent::ContextNotice { source, .. }
                                                        if source == SKILL_CATALOG_SOURCE
                                                )
                                        })
                                    });
                                    let has_any = entries.iter().any(|entry| {
                                        matches!(
                                            &entry.event,
                                            SessionEvent::ContextNotice { source, .. }
                                                if source == SKILL_CATALOG_SOURCE
                                        )
                                    });
                                    (still_vis, has_any)
                                });

                            let prev_state = if prev_still_visible {
                                prev_record.as_ref().map(|r| &r.state)
                            } else {
                                None
                            };

                            let should_emit = match prev_state {
                                Some(prev) => *prev != next_state,
                                None => !next_state.entries.is_empty() || next_state.incomplete,
                            };

                            if should_emit {
                                let is_update = prev_record.is_some() || had_catalog_notice;
                                let rendered = render_catalog_message(&next_state, is_update);
                                session.append(SessionEvent::ContextNotice {
                                    source: SKILL_CATALOG_SOURCE.to_string(),
                                    text: rendered,
                                });
                                if let Some(last_seq) =
                                    session.with_visible_entries(|e| e.last().map(|x| x.seq))
                                {
                                    published_map.lock().unwrap().insert(
                                        session.id.clone(),
                                        PublishedCatalogRecord {
                                            seq: last_seq,
                                            state: next_state,
                                        },
                                    );
                                }
                            }
                        }

                        // 2. 扫描最近一轮用户直输消息中的 `/<name>` 手势，仅展开 user_invocable 技能且仅处理本轮末尾新增的用户消息
                        let trailing_user_texts: Vec<String> =
                            session.with_visible_entries(|entries| {
                                let mut collected = Vec::new();
                                for entry in entries.iter().rev() {
                                    match &entry.event {
                                        SessionEvent::User { text } => {
                                            collected.push(text.clone())
                                        }
                                        SessionEvent::ContextNotice { source, .. }
                                            if source != SKILL_INVOCATION_SOURCE => {}
                                        SessionEvent::ContextNotice { .. }
                                        | SessionEvent::Assistant { .. }
                                        | SessionEvent::ToolResult { .. }
                                        | SessionEvent::Compaction { .. }
                                        | SessionEvent::Trace { .. } => break,
                                    }
                                }
                                collected.reverse();
                                collected
                            });

                        let mut gesture_names = Vec::new();
                        for text in trailing_user_texts {
                            for name in extract_skill_gestures(&text) {
                                if !gesture_names.contains(&name) {
                                    gesture_names.push(name);
                                }
                            }
                        }

                        for name in gesture_names {
                            if cancel.is_cancelled() {
                                break;
                            }
                            if let Ok(Some(def)) = reg.get(&name, lookup.clone()).await
                                && def.summary.is_user_invocable()
                            {
                                session.append(SessionEvent::ContextNotice {
                                    source: SKILL_INVOCATION_SOURCE.to_string(),
                                    text: render_skill_content(&def),
                                });
                            }
                        }

                        None
                    }
                },
            )?;

            Ok(None)
        })
    }
}

/// 辅助函数：读取工作区下可用技能摘要清单。
pub async fn list_workspace_skills(ctx: &Context, workspace: &std::path::Path) -> Vec<SkillSummary> {
    if let Ok(reg) = ctx.skills() {
        return reg
            .list(SkillLookupOptions {
                cwd: Some(workspace.to_path_buf()),
                cancel: CancellationToken::new(),
            })
            .await;
    }
    Vec::new()
}
