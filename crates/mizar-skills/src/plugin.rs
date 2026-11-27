use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_agent_loop::{
    AgentPreStepEvent, AgentPrepareTurnEvent, AgentToolResultEvent, EVENT_AGENT_PRE_STEP,
    EVENT_AGENT_PREPARE_TURN, EVENT_AGENT_TOOL_RESULT,
};
use mizar_session::SessionEvent;
use mizar_tools::ToolsContextExt;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::filesystem::{FilesystemSkillConfig, FilesystemSkillProvider};
use crate::registry::{SkillProvider, SkillRegistry, SkillsContextExt};
use crate::types::{
    SkillLookupOptions, SkillSummary, escape_xml_text, is_skill_name, render_skill_content,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileReadMode {
    Read,
    Bash,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CatalogEntry {
    name: String,
    description: String,
    location: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CatalogState {
    entries: Vec<CatalogEntry>,
    incomplete: bool,
    read_mode: FileReadMode,
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
    read_mode: FileReadMode,
) -> CatalogState {
    let entries = skills
        .iter()
        .filter(|s| s.is_model_invocable())
        .filter_map(|s| {
            let path = s.path.as_ref()?;
            Some(CatalogEntry {
                name: s.name.clone(),
                description: normalize_catalog_description(&s.description, max_desc_len),
                location: path.display().to_string(),
            })
        })
        .collect();
    CatalogState {
        entries,
        incomplete,
        read_mode,
    }
}

fn render_catalog_lines(entries: &[CatalogEntry]) -> Vec<String> {
    let mut lines = Vec::with_capacity(entries.len() * 5);
    for entry in entries {
        lines.push("  <skill>".to_string());
        lines.push(format!("    <name>{}</name>", escape_xml_text(&entry.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml_text(&entry.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml_text(&entry.location)
        ));
        lines.push("  </skill>".to_string());
    }
    lines
}

fn read_instruction(mode: FileReadMode) -> &'static str {
    match mode {
        FileReadMode::Read => {
            "Use the read tool to load a skill's file when the task matches its description."
        }
        FileReadMode::Bash => {
            "Use bash to load a skill's file when the task matches its description."
        }
    }
}

fn render_catalog_message(state: &CatalogState, is_update: bool) -> String {
    let mut lines = vec!["<system-reminder>".to_string()];

    if is_update {
        lines.push(if state.incomplete {
            "Skill discovery is incomplete. This catalog of currently available skills replaces every earlier available-skills list in this session:".to_string()
        } else {
            "The available skill catalog changed. This complete catalog replaces every earlier available-skills list in this session:".to_string()
        });
    } else if state.entries.is_empty() {
        lines.push("No skills could be confirmed in this discovery attempt.".to_string());
    } else {
        lines.push(
            "The following skills provide specialized instructions for specific tasks.".to_string(),
        );
    }

    if !state.entries.is_empty() {
        lines.push(read_instruction(state.read_mode).to_string());
        lines.push(
            "Resolve relative paths against the skill's directory (parent of SKILL.md). Use absolute paths in tool calls."
                .to_string(),
        );
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
                "No skills could be confirmed in this discovery attempt. Do not use entries from earlier skill catalogs.".to_string()
            } else {
                "No skills are currently available. Do not use entries from earlier skill catalogs.".to_string()
            }
        } else {
            "Wait for an updated catalog before loading any skill file. Do not infer skill instructions from an empty catalog.".to_string()
        });
    } else if is_update {
        lines.push("Use only skills in this replacement catalog.".to_string());
    }

    lines.push("A user may also invoke a skill directly; its <skill_content> block then appears in this conversation. Follow it without re-reading the skill file.".to_string());
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

/// 为当前步生成待注入的 `<available_skills>` 目录消息（若目录无变化或未声明 `read`/`bash` 则返回 `None`）。
async fn collect_catalog_notice(
    ctx: &Context,
    reg: &SkillRegistry,
    session: &mizar_session::Session,
    lookup: &SkillLookupOptions,
    max_desc_len: usize,
    published_map: &Mutex<HashMap<String, PublishedCatalogRecord>>,
) -> Option<String> {
    let read_mode = ctx.tools().ok().and_then(|rt| {
        let declared = rt.list_declared(Some(&session.id));
        if declared.iter().any(|t| t.name == "read") {
            Some(FileReadMode::Read)
        } else if declared.iter().any(|t| t.name == "bash") {
            Some(FileReadMode::Bash)
        } else {
            None
        }
    })?;

    let snap = reg.snapshot(lookup.clone()).await;
    if lookup.cancel.is_cancelled() {
        return None;
    }
    let next_state = build_catalog_state(&snap.skills, max_desc_len, !snap.complete, read_mode);

    let prev_record = published_map.lock().unwrap().get(&session.id).cloned();
    let (prev_still_visible, latest_catalog) = session.with_visible_entries(|entries| {
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
        let latest = entries.iter().rev().find_map(|entry| match &entry.event {
            SessionEvent::ContextNotice { source, text, .. } if source == SKILL_CATALOG_SOURCE => {
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
        let matches_initial = *latest_text == render_catalog_message(&next_state, false);
        let matches_update = *latest_text == render_catalog_message(&next_state, true);
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
    if !should_emit {
        return None;
    }

    let is_update = prev_record.is_some() || latest_catalog.is_some();
    let rendered = render_catalog_message(&next_state, is_update);
    published_map.lock().unwrap().insert(
        session.id.clone(),
        PublishedCatalogRecord {
            seq: u64::MAX,
            state: next_state,
        },
    );
    Some(rendered)
}

/// 扫描最近一轮尚未处理的 `/<name>` 技能手势，返回按顺序解析后的 `<skill_content>` 列表与最大用户 seq。
async fn collect_gesture_notices(
    reg: &SkillRegistry,
    session: &mizar_session::Session,
    lookup: &SkillLookupOptions,
    processed_gestures: &Mutex<HashMap<String, u64>>,
) -> Option<(Vec<String>, Option<u64>)> {
    let last_done_gesture_seq = processed_gestures
        .lock()
        .unwrap()
        .get(&session.id)
        .copied()
        .unwrap_or(0);
    let (trailing_user_texts, max_user_seq) = session.with_visible_entries(|entries| {
        let mut collected = Vec::new();
        let mut max_seq: Option<u64> = None;
        for entry in entries.iter().rev() {
            if entry.seq <= last_done_gesture_seq {
                break;
            }
            match &entry.event {
                SessionEvent::User { text, .. } => {
                    max_seq = Some(max_seq.map_or(entry.seq, |m| m.max(entry.seq)));
                    collected.push(text.clone());
                }
                SessionEvent::ContextNotice { source, .. } if source != SKILL_INVOCATION_SOURCE => {
                }
                SessionEvent::ContextNotice { .. }
                | SessionEvent::Assistant { .. }
                | SessionEvent::ToolResult { .. }
                | SessionEvent::Compaction { .. }
                | SessionEvent::ToolResultPrune { .. }
                | SessionEvent::BashExecution { .. }
                | SessionEvent::Selection { .. }
                | SessionEvent::Trace { .. } => break,
            }
        }
        collected.reverse();
        (collected, max_seq)
    });

    let mut gesture_names = Vec::new();
    for text in trailing_user_texts {
        for name in extract_skill_gestures(&text) {
            if !gesture_names.contains(&name) {
                gesture_names.push(name);
            }
        }
    }

    let mut resolved_contents = Vec::new();
    for name in gesture_names {
        if lookup.cancel.is_cancelled() {
            return None;
        }
        match reg.get(&name, lookup.clone()).await {
            Ok(Some(def)) if def.summary.is_user_invocable() => {
                resolved_contents.push(render_skill_content(&def));
            }
            Ok(_) => {}
            Err(_) => return None,
        }
    }

    if lookup.cancel.is_cancelled() {
        return None;
    }
    Some((resolved_contents, max_user_seq))
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

            // 1）首轮用户消息落盘前通过 `agent/prepare-turn` 按严格顺序注入 `<available_skills>` 与 `/<name>` `<skill_content>`；
            // 2）每步模型请求前兜底检查目录增量（如压后重注）及用户中途 steer 携带的新手势。
            let published_catalogs: Arc<Mutex<HashMap<String, PublishedCatalogRecord>>> =
                Arc::new(Mutex::new(HashMap::new()));
            let processed_gesture_seqs: Arc<Mutex<HashMap<String, u64>>> =
                Arc::new(Mutex::new(HashMap::new()));
            let prepared_turns: Arc<Mutex<HashMap<String, u64>>> =
                Arc::new(Mutex::new(HashMap::new()));

            {
                let reg = Arc::clone(&registry);
                let prepare_ctx = ctx.clone();
                let ws = config.workspace.clone();
                let max_desc_len = config.catalog_description_max_length.max(3);
                let published_map = Arc::clone(&published_catalogs);
                let prepared_map = Arc::clone(&prepared_turns);
                ctx.on_serial::<AgentPrepareTurnEvent, (), _, _>(
                    EVENT_AGENT_PREPARE_TURN,
                    move |ev: AgentPrepareTurnEvent| {
                        let reg = Arc::clone(&reg);
                        let ctx_inner = prepare_ctx.clone();
                        let session = Arc::clone(&ev.session);
                        let cancel = ev.cancel.clone();
                        let input = ev.input.clone();
                        let turn = ev.turn;
                        let ws = ws.clone();
                        let published_map = Arc::clone(&published_map);
                        let prepared_map = Arc::clone(&prepared_map);

                        async move {
                            if cancel.is_cancelled() {
                                return None;
                            }
                            let lookup = SkillLookupOptions {
                                cwd: Some(ws),
                                cancel: cancel.clone(),
                            };

                            if let Some(rendered) = collect_catalog_notice(
                                &ctx_inner,
                                &reg,
                                &session,
                                &lookup,
                                max_desc_len,
                                &published_map,
                            )
                            .await
                            {
                                session.append(SessionEvent::context_notice(
                                    SKILL_CATALOG_SOURCE,
                                    rendered,
                                ));
                                if let Some(last_seq) =
                                    session.with_visible_entries(|e| e.last().map(|x| x.seq))
                                    && let Some(rec) =
                                        published_map.lock().unwrap().get_mut(&session.id)
                                {
                                    rec.seq = last_seq;
                                }
                            }

                            for name in extract_skill_gestures(&input) {
                                if cancel.is_cancelled() {
                                    return None;
                                }
                                if let Ok(Some(def)) = reg.get(&name, lookup.clone()).await
                                    && def.summary.is_user_invocable()
                                {
                                    session.append(SessionEvent::context_notice(
                                        SKILL_INVOCATION_SOURCE,
                                        render_skill_content(&def),
                                    ));
                                }
                            }

                            prepared_map
                                .lock()
                                .unwrap()
                                .insert(session.id.clone(), turn);
                            None
                        }
                    },
                )?;
            }

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
                    let processed_gestures = Arc::clone(&processed_gesture_seqs);
                    let prepared_map = Arc::clone(&prepared_turns);

                    async move {
                        if cancel.is_cancelled() {
                            return None;
                        }
                        let lookup = SkillLookupOptions {
                            cwd: Some(ws),
                            cancel: cancel.clone(),
                        };

                        // 1. 若当前会话声明了 `read` 或 `bash` 工具，按需注入或更新 <available_skills> 目录
                        if let Some(rendered) = collect_catalog_notice(
                            &ctx_inner,
                            &reg,
                            &session,
                            &lookup,
                            max_desc_len,
                            &published_map,
                        )
                        .await
                        {
                            session.append(SessionEvent::context_notice(
                                SKILL_CATALOG_SOURCE,
                                rendered,
                            ));
                            if let Some(last_seq) =
                                session.with_visible_entries(|e| e.last().map(|x| x.seq))
                                && let Some(rec) =
                                    published_map.lock().unwrap().get_mut(&session.id)
                            {
                                rec.seq = last_seq;
                            }
                        }

                        // 若本轮已在 `agent/prepare-turn` 展开过手势，则在 step 1 推进已处理水位，避免对本轮用户输入重复注入
                        let already_prepared_this_turn = ev.step == 1
                            && prepared_map
                                .lock()
                                .unwrap()
                                .get(&session.id)
                                .is_some_and(|&t| t == ev.turn);
                        if already_prepared_this_turn {
                            let latest_user_seq = session.with_visible_entries(|entries| {
                                entries.iter().rev().find_map(|e| {
                                    matches!(e.event, SessionEvent::User { .. }).then_some(e.seq)
                                })
                            });
                            if let Some(max_seq) = latest_user_seq {
                                processed_gestures
                                    .lock()
                                    .unwrap()
                                    .entry(session.id.clone())
                                    .and_modify(|s| *s = (*s).max(max_seq))
                                    .or_insert(max_seq);
                            }
                            return None;
                        }

                        // 2. 扫描最近一轮用户直输消息中的 `/<name>` 手势，仅展开 user_invocable 技能且仅处理尚未处理过的新增消息
                        if let Some((rendered_skills, max_user_seq)) =
                            collect_gesture_notices(&reg, &session, &lookup, &processed_gestures)
                                .await
                        {
                            for rendered in rendered_skills {
                                session.append(SessionEvent::context_notice(
                                    SKILL_INVOCATION_SOURCE,
                                    rendered,
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

                        None
                    }
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
