use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Next, Plugin};
use mizar_agent_loop::{
    AgentPreStepDecision, AgentPreStepInput, AgentToolResultEvent, EVENT_AGENT_PRE_STEP,
    EVENT_AGENT_TOOL_RESULT, TurnInput,
};
use mizar_session::SessionEvent;
use mizar_tools::{
    ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolExposure, ToolOutput, ToolsContextExt,
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
    /// 是否包含用户全局 `~/.agents/skills`（默认 true；`--isolated` 时置为 false）。
    pub include_user_roots: bool,
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
            include_user_roots: true,
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
    let mut iter = text.char_indices().peekable();
    let mut prev_is_whitespace = true;

    while let Some((idx, ch)) = iter.next() {
        if prev_is_whitespace && ch == '/' {
            let start = idx + ch.len_utf8();
            let mut end = start;
            while let Some(&(next_idx, next_ch)) = iter.peek() {
                if next_ch.is_whitespace() {
                    break;
                }
                end = next_idx + next_ch.len_utf8();
                iter.next();
            }
            if end > start {
                let token = &text[start..end];
                if is_skill_name(token) && !out.iter().any(|existing| existing == token) {
                    out.push(token.to_string());
                }
            }
            prev_is_whitespace = false;
        } else {
            prev_is_whitespace = ch.is_whitespace();
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
                    include_user_roots: config.include_user_roots,
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

            // 仅在发现至少 1 个模型可调用技能时注册 `skill` 工具；0 skills 时保持工具表精简零开销
            let initial_skills = registry
                .list(SkillLookupOptions {
                    cwd: Some(config.workspace.clone()),
                    cancel: CancellationToken::new(),
                })
                .await;
            if initial_skills.iter().any(|s| s.is_model_invocable()) {
                let tools = ctx.tools()?;
                let skill_tool = create_skill_tool(Arc::clone(&registry), config.workspace.clone());
                ctx.effect("skills.skillTool", || {
                    Ok::<_, mizar_tools::ToolError>(Some(tools.register(skill_tool)?))
                })?;
            }

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
            let processed_gesture_seqs: Arc<Mutex<HashMap<String, u64>>> =
                Arc::new(Mutex::new(HashMap::new()));
            let pre_step_ctx = ctx.clone();
            ctx.on_waterfall_with(
                EVENT_AGENT_PRE_STEP,
                move |input: AgentPreStepInput,
                      next: Next<
                    AgentPreStepInput,
                    BoxFuture<'static, AgentPreStepDecision>,
                >| {
                    let reg = Arc::clone(&registry);
                    let ctx_inner = pre_step_ctx.clone();
                    let session = Arc::clone(&input.session);
                    let cancel = input.cancel.clone();
                    let ws = config.workspace.clone();
                    let max_desc_len = config.catalog_description_max_length.max(3);
                    let published_map = Arc::clone(&published_catalogs);
                    let processed_gestures = Arc::clone(&processed_gesture_seqs);

                    Box::pin(async move {
                        if cancel.is_cancelled() {
                            return AgentPreStepDecision::Reject {
                                reason: "cancelled".to_string(),
                            };
                        }
                        let decision = next.call(input).await;
                        let AgentPreStepDecision::Enter { mut inputs } = decision else {
                            return decision;
                        };
                        if cancel.is_cancelled() {
                            return AgentPreStepDecision::Reject {
                                reason: "cancelled".to_string(),
                            };
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
                            if cancel.is_cancelled() {
                                return AgentPreStepDecision::Reject {
                                    reason: "cancelled".to_string(),
                                };
                            }
                            let next_state =
                                build_catalog_state(&snap.skills, max_desc_len, !snap.complete);

                            let prev_record =
                                published_map.lock().unwrap().get(&session.id).cloned();
                            let (prev_still_visible, latest_catalog) = session
                                .with_visible_entries(|entries| {
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
                                    let latest =
                                        entries.iter().rev().find_map(|entry| match &entry.event {
                                            SessionEvent::ContextNotice {
                                                source, text, ..
                                            } if source == SKILL_CATALOG_SOURCE => {
                                                Some((entry.seq, text.clone()))
                                            }
                                            _ => None,
                                        });
                                    (still_vis, latest)
                                });

                            let mut prev_state = if prev_still_visible {
                                prev_record.as_ref().map(|r| r.state.clone())
                            } else {
                                None
                            };

                            if prev_state.is_none()
                                && let Some((latest_seq, ref latest_text)) = latest_catalog
                            {
                                let matches_initial =
                                    *latest_text == render_catalog_message(&next_state, false);
                                let matches_update =
                                    *latest_text == render_catalog_message(&next_state, true);
                                if matches_initial || matches_update {
                                    published_map.lock().unwrap().insert(
                                        session.id.clone(),
                                        PublishedCatalogRecord {
                                            seq: latest_seq,
                                            state: next_state.clone(),
                                        },
                                    );
                                    prev_state = Some(next_state.clone());
                                }
                            }

                            let should_emit = match prev_state {
                                Some(ref prev) => *prev != next_state,
                                None => !next_state.entries.is_empty() || next_state.incomplete,
                            };

                            if should_emit {
                                let is_update = prev_record.is_some() || latest_catalog.is_some();
                                let rendered = render_catalog_message(&next_state, is_update);
                                session.append(SessionEvent::context_notice(
                                    SKILL_CATALOG_SOURCE,
                                    rendered,
                                ));
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

                        // 2. 扫描本步待入账的用户直输消息（或测试直接追加的未处理 User 条目）中的 `/<name>` 手势
                        let input_user_texts: Vec<String> = inputs
                            .iter()
                            .filter_map(|item| match item {
                                TurnInput::User { text, .. } => Some(text.clone()),
                                TurnInput::Notice { .. } => None,
                            })
                            .collect();
                        let from_turn_inputs = !input_user_texts.is_empty();

                        let (trailing_user_texts, max_user_seq) = if from_turn_inputs {
                            (input_user_texts, None)
                        } else {
                            let last_done_gesture_seq = processed_gestures
                                .lock()
                                .unwrap()
                                .get(&session.id)
                                .copied()
                                .unwrap_or(0);
                            session.with_visible_entries(|entries| {
                                let mut collected = Vec::new();
                                let mut max_seq: Option<u64> = None;
                                for entry in entries.iter().rev() {
                                    if entry.seq <= last_done_gesture_seq {
                                        break;
                                    }
                                    match &entry.event {
                                        SessionEvent::User { text, .. } => {
                                            max_seq = Some(
                                                max_seq.map_or(entry.seq, |m| m.max(entry.seq)),
                                            );
                                            collected.push(text.clone());
                                        }
                                        SessionEvent::ContextNotice { source, .. }
                                            if source != SKILL_INVOCATION_SOURCE => {}
                                        SessionEvent::ContextNotice { .. }
                                        | SessionEvent::Assistant { .. }
                                        | SessionEvent::ToolResult { .. }
                                        | SessionEvent::Compaction { .. }
                                        | SessionEvent::ToolResultPrune { .. }
                                        | SessionEvent::BashExecution { .. }
                                        | SessionEvent::Selection { .. }
                                        | SessionEvent::Usage { .. }
                                        | SessionEvent::Trace { .. } => break,
                                    }
                                }
                                collected.reverse();
                                (collected, max_seq)
                            })
                        };

                        let mut gesture_names = Vec::new();
                        for text in trailing_user_texts {
                            for name in extract_skill_gestures(&text) {
                                if !gesture_names.contains(&name) {
                                    gesture_names.push(name);
                                }
                            }
                        }

                        let mut resolved_defs = Vec::new();
                        let mut all_resolved = true;
                        for name in gesture_names {
                            if cancel.is_cancelled() {
                                all_resolved = false;
                                break;
                            }
                            match reg.get(&name, lookup.clone()).await {
                                Ok(Some(def)) if def.summary.is_user_invocable() => {
                                    resolved_defs.push(def);
                                }
                                Ok(_) => {}
                                Err(_) => {
                                    all_resolved = false;
                                    break;
                                }
                            }
                        }

                        if all_resolved && !cancel.is_cancelled() {
                            if from_turn_inputs {
                                for def in resolved_defs {
                                    inputs.push(TurnInput::notice(
                                        SKILL_INVOCATION_SOURCE,
                                        render_skill_content(&def),
                                    ));
                                }
                            } else {
                                for def in resolved_defs {
                                    session.append(SessionEvent::context_notice(
                                        SKILL_INVOCATION_SOURCE,
                                        render_skill_content(&def),
                                    ));
                                }
                                if let Some(max_seq) = max_user_seq {
                                    processed_gestures
                                        .lock()
                                        .unwrap()
                                        .entry(session.id.clone())
                                        .and_modify(|s| *s = (*s).max(max_seq))
                                        .or_insert(max_seq);
                                }
                            }
                        }

                        AgentPreStepDecision::Enter { inputs }
                    }) as BoxFuture<'static, AgentPreStepDecision>
                },
                cordis::EventOptions {
                    priority: 5,
                    ..Default::default()
                },
            )?;

            Ok(None)
        })
    }
}

/// 辅助函数：读取工作区下可用技能摘要清单。
pub async fn list_workspace_skills(
    ctx: &Context,
    workspace: &std::path::Path,
) -> Vec<SkillSummary> {
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
