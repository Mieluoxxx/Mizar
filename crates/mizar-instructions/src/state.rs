use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cordis::{BoxFuture, Context, CordisError, Service};
use mizar_agent_loop::{AgentPreStepEvent, EVENT_AGENT_PRE_STEP};
use mizar_session::{Session, SessionEvent};
use mizar_tools::{EVENT_TOOLS_RESULT, ToolResultEvent};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::config::{
    AGENT_INSTRUCTIONS_SOURCE, InstructionScopeStatus, InstructionsConfig,
    SessionInstructionsReport,
};
use crate::digest::sha256_hex;
use crate::files::{
    DiscoveredFile, FsFingerprint, ScopeKey, ScopeRoot, ancestor_chain, canonical_or_abs,
    descendant_dirs_between, discover_instructions, find_project_root, read_fingerprint,
    resolve_agents_home, resolve_mizar_home, sanitize_candidates,
};
use crate::render::{
    DeltaChangeKind, DeltaItem, render_baseline_message, render_delta_message,
    render_delta_message_detailed,
};

#[derive(Debug, Clone)]
struct TrackedFile {
    display_path: String,
    scope_dir_display: String,
    digest: String,
}

#[derive(Debug, Default)]
struct SessionInstructionState {
    baseline_checked: bool,
    baseline_seq: Option<u64>,
    baseline_identity: String,
    represented: HashMap<ScopeKey, TrackedFile>,
    omitted: HashSet<ScopeKey>,
    truncated: HashSet<ScopeKey>,
    known_subdirs: Vec<PathBuf>,
    pending_subdirs: Vec<PathBuf>,
    watched_paths: HashMap<PathBuf, Option<FsFingerprint>>,
    report: SessionInstructionsReport,
}

pub struct AgentInstructions {
    config: InstructionsConfig,
    sessions: Mutex<HashMap<String, SessionInstructionState>>,
}

impl AgentInstructions {
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn config(&self) -> &InstructionsConfig {
        &self.config
    }

    pub fn session_report(&self, session_id: &str) -> Option<SessionInstructionsReport> {
        let guard = self.sessions.lock().unwrap();
        guard.get(session_id).map(|st| st.report.clone())
    }

    /// 若尚未运行过任何步，也可按当前配置做一次只读探测供 UI 查询。
    pub fn probe_report(&self, session_id: Option<&str>) -> SessionInstructionsReport {
        if let Some(sid) = session_id
            && let Some(existing) = self.session_report(sid)
        {
            return existing;
        }
        if !self.config.enabled {
            return SessionInstructionsReport {
                enabled: false,
                project_root: None,
                scopes: Vec::new(),
                warnings: Vec::new(),
            };
        }
        let disc = discover_instructions(&self.config, &[]);
        let statuses = render_baseline_message(
            &disc.global_files,
            &disc.project_files,
            self.config.max_bytes,
        )
        .map(|b| b.statuses)
        .unwrap_or_default();
        SessionInstructionsReport {
            enabled: true,
            project_root: Some(disc.project_root),
            scopes: statuses,
            warnings: disc.warnings,
        }
    }

    fn compute_baseline_identity(&self, project_root: &Path) -> String {
        let ws = canonical_or_abs(&self.config.workspace);
        let raw = format!(
            "ws={}|root={}|global={}|cands={}|local={}|max={}|src={}",
            ws.display(),
            project_root.display(),
            self.config.include_default_global,
            self.config.candidates.join(","),
            self.config.local_candidates.join(","),
            self.config.max_bytes,
            self.config.max_source_bytes
        );
        sha256_hex(raw.as_bytes())
    }

    fn collect_watched_paths(
        &self,
        known_subdirs: &[PathBuf],
    ) -> HashMap<PathBuf, Option<FsFingerprint>> {
        let mut paths = Vec::new();
        let base_cands = sanitize_candidates(&self.config.candidates);
        let local_cands = sanitize_candidates(&self.config.local_candidates);

        if self.config.include_default_global {
            let mizar_cands = if base_cands.is_empty() {
                vec!["AGENTS.md".to_string()]
            } else {
                base_cands.clone()
            };
            if let Some(mizar_home) = resolve_mizar_home(self.config.mizar_home.as_deref()) {
                for name in &mizar_cands {
                    paths.push(mizar_home.join(name));
                }
            }
            if let Some(agents_home) = resolve_agents_home(self.config.agents_home.as_deref()) {
                paths.push(agents_home.join("AGENTS.md"));
            }
        }

        let effective_ws = canonical_or_abs(&self.config.workspace);
        let project_root = find_project_root(&effective_ws, &self.config.project_root_markers);
        let mut dirs = ancestor_chain(&project_root, &effective_ws);
        for sub in known_subdirs {
            if !dirs.contains(sub) {
                dirs.push(sub.clone());
            }
        }
        for dir in dirs {
            for name in &base_cands {
                paths.push(dir.join(name));
            }
            for name in &local_cands {
                paths.push(dir.join(name));
            }
        }

        let mut map = HashMap::with_capacity(paths.len());
        for p in paths {
            let fp = read_fingerprint(&p).ok().flatten();
            map.insert(p, fp);
        }
        map
    }

    fn fingerprints_changed(prev: &HashMap<PathBuf, Option<FsFingerprint>>) -> bool {
        for (path, old_fp) in prev {
            let cur_fp = read_fingerprint(path).ok().flatten();
            if &cur_fp != old_fp {
                return true;
            }
        }
        false
    }

    fn record_tool_touch(&self, ev: &ToolResultEvent) {
        if !self.config.enabled || !self.config.follow_subdirs {
            return;
        }
        if ev.result.is_error || ev.result.blocked {
            return;
        }
        if !matches!(
            ev.exec.name.as_str(),
            "read" | "write" | "edit" | "read_file"
        ) {
            return;
        }
        let Some(raw_path) = ev
            .exec
            .args
            .get("path")
            .or_else(|| ev.exec.args.get("file_path"))
            .and_then(Value::as_str)
        else {
            return;
        };
        let dirs = descendant_dirs_between(&self.config.workspace, raw_path);
        if dirs.is_empty() {
            return;
        }
        let sid = ev
            .exec
            .session_id
            .as_deref()
            .unwrap_or("__default__")
            .to_string();
        let mut guard = self.sessions.lock().unwrap();
        let st = guard.entry(sid).or_default();
        for dir in dirs {
            if !st.known_subdirs.contains(&dir) && !st.pending_subdirs.contains(&dir) {
                st.pending_subdirs.push(dir);
            }
        }
    }

    fn log_warnings(ctx: &Context, warnings: &[String]) {
        if warnings.is_empty() {
            return;
        }
        let logger = ctx.logger_named("instructions");
        for w in warnings {
            logger.warn(w.clone());
        }
    }

    pub fn reconcile_pre_step(
        &self,
        ctx: &Context,
        session: &Arc<Session>,
        cancel: &CancellationToken,
    ) {
        if !self.config.enabled || self.config.max_bytes == 0 || cancel.is_cancelled() {
            return;
        }

        let sid = session.id.clone();
        let (
            mem_baseline_checked,
            mem_baseline_seq,
            mem_identity,
            pending_subdirs,
            known_subdirs,
            watched_empty_or_changed,
        ) = {
            let guard = self.sessions.lock().unwrap();
            match guard.get(&sid) {
                Some(st) => (
                    st.baseline_checked,
                    st.baseline_seq,
                    st.baseline_identity.clone(),
                    st.pending_subdirs.clone(),
                    st.known_subdirs.clone(),
                    (!st.watched_paths.is_empty() || !st.pending_subdirs.is_empty())
                        && Self::fingerprints_changed(&st.watched_paths),
                ),
                None => (false, None, String::new(), Vec::new(), Vec::new(), true),
            }
        };

        let visible_instruction_notices: Vec<(u64, String)> =
            session.with_visible_entries(|entries| {
                entries
                    .iter()
                    .filter_map(|e| match &e.event {
                        SessionEvent::ContextNotice { source, text, .. }
                            if source == AGENT_INSTRUCTIONS_SOURCE =>
                        {
                            Some((e.seq, text.clone()))
                        }
                        _ => None,
                    })
                    .collect()
            });

        let baseline_visible = mem_baseline_seq
            .is_some_and(|seq| visible_instruction_notices.iter().any(|(s, _)| *s == seq));

        if !mem_baseline_checked || (mem_baseline_seq.is_some() && !baseline_visible) {
            let mut merged_subdirs = known_subdirs;
            for sub in pending_subdirs {
                if !merged_subdirs.contains(&sub) {
                    merged_subdirs.push(sub);
                }
            }

            let disc = discover_instructions(&self.config, &merged_subdirs);
            Self::log_warnings(ctx, &disc.warnings);
            let identity = self.compute_baseline_identity(&disc.project_root);
            let rendered = render_baseline_message(
                &disc.global_files,
                &disc.project_files,
                self.config.max_bytes,
            );
            let watched = self.collect_watched_paths(&merged_subdirs);

            // 若刚从持久化恢复（内存态为空但日志里已存在相同基线），无需重复追加
            let hydrated_seq = if mem_baseline_seq.is_none() {
                rendered.as_ref().and_then(|r| {
                    visible_instruction_notices
                        .iter()
                        .rfind(|(_, text)| *text == r.text)
                        .map(|(seq, _)| *seq)
                })
            } else {
                None
            };
            let already_hydrated = hydrated_seq.is_some();

            let mut emitted_seq = hydrated_seq;

            if !already_hydrated {
                if let Some(ref r) = rendered {
                    session.append(SessionEvent::context_notice(
                        AGENT_INSTRUCTIONS_SOURCE,
                        &r.text,
                    ));
                    emitted_seq = session.with_visible_entries(|e| e.last().map(|x| x.seq));
                } else if !visible_instruction_notices.is_empty() {
                    // 以前加载过指令，现在文件全被删空且恢复后身份发生变化，发送清理通知
                    let items: Vec<DeltaItem> = {
                        let guard = self.sessions.lock().unwrap();
                        guard
                            .get(&sid)
                            .map(|st| {
                                st.represented
                                    .iter()
                                    .map(|(scope, tr)| DeltaItem {
                                        scope: scope.clone(),
                                        display_path: tr.display_path.clone(),
                                        scope_dir_display: tr.scope_dir_display.clone(),
                                        kind: DeltaChangeKind::Removed,
                                        content: String::new(),
                                    })
                                    .collect()
                            })
                            .unwrap_or_default()
                    };
                    if let Some(delta_text) = render_delta_message(&items, self.config.max_bytes) {
                        session.append(SessionEvent::context_notice(
                            AGENT_INSTRUCTIONS_SOURCE,
                            delta_text,
                        ));
                        emitted_seq = session.with_visible_entries(|e| e.last().map(|x| x.seq));
                    } else if mem_baseline_seq.is_none() {
                        // 离线恢复时此前有基线，但本地指令文件已全部删除
                        session.append(SessionEvent::context_notice(
                            AGENT_INSTRUCTIONS_SOURCE,
                            "<system-reminder>\nInstructions removed: previously loaded instruction files are no longer present.\n</system-reminder>",
                        ));
                        emitted_seq = session.with_visible_entries(|e| e.last().map(|x| x.seq));
                    }
                }
            }

            let mut next_rep = HashMap::new();
            let mut next_omitted = HashSet::new();
            let mut next_truncated = HashSet::new();
            let mut statuses = Vec::new();
            if let Some(r) = rendered {
                for f in r.represented {
                    next_rep.insert(
                        f.scope.clone(),
                        TrackedFile {
                            display_path: f.display_path,
                            scope_dir_display: f.scope_dir_display,
                            digest: f.digest,
                        },
                    );
                }
                for s in &r.statuses {
                    if let Some(f) = disc
                        .global_files
                        .iter()
                        .chain(disc.project_files.iter())
                        .find(|df| df.display_path == s.display_path)
                    {
                        if s.omitted {
                            next_omitted.insert(f.scope.clone());
                        }
                        if s.truncated {
                            next_truncated.insert(f.scope.clone());
                        }
                    }
                }
                statuses = r.statuses;
            }

            let mut guard = self.sessions.lock().unwrap();
            let st = guard.entry(sid).or_default();
            st.baseline_checked = true;
            st.baseline_seq = emitted_seq;
            st.baseline_identity = identity;
            st.represented = next_rep;
            st.omitted = next_omitted;
            st.truncated = next_truncated;
            st.known_subdirs = merged_subdirs;
            st.pending_subdirs
                .retain(|sub| !st.known_subdirs.contains(sub));
            st.watched_paths = watched;
            st.report = SessionInstructionsReport {
                enabled: true,
                project_root: Some(disc.project_root),
                scopes: statuses,
                warnings: disc.warnings,
            };
            return;
        }

        // 快速路径：无待扫子目录且任何探测文件元数据都未变化
        if pending_subdirs.is_empty() && !watched_empty_or_changed {
            return;
        }

        let mut merged_subdirs = known_subdirs;
        for sub in pending_subdirs {
            if !merged_subdirs.contains(&sub) {
                merged_subdirs.push(sub);
            }
        }

        let disc = discover_instructions(&self.config, &merged_subdirs);
        Self::log_warnings(ctx, &disc.warnings);
        let identity = self.compute_baseline_identity(&disc.project_root);
        let watched = self.collect_watched_paths(&merged_subdirs);

        if identity != mem_identity {
            // 极少见：workspace root/配置发生变化，重新生成完整基线
            if let Some(r) = render_baseline_message(
                &disc.global_files,
                &disc.project_files,
                self.config.max_bytes,
            ) {
                session.append(SessionEvent::context_notice(
                    AGENT_INSTRUCTIONS_SOURCE,
                    &r.text,
                ));
                let new_seq = session.with_visible_entries(|e| e.last().map(|x| x.seq));
                let mut next_rep = HashMap::new();
                let mut next_omitted = HashSet::new();
                let mut next_truncated = HashSet::new();
                for f in r.represented {
                    next_rep.insert(
                        f.scope.clone(),
                        TrackedFile {
                            display_path: f.display_path,
                            scope_dir_display: f.scope_dir_display,
                            digest: f.digest,
                        },
                    );
                }
                for s in &r.statuses {
                    if let Some(f) = disc
                        .global_files
                        .iter()
                        .chain(disc.project_files.iter())
                        .find(|df| df.display_path == s.display_path)
                    {
                        if s.omitted {
                            next_omitted.insert(f.scope.clone());
                        }
                        if s.truncated {
                            next_truncated.insert(f.scope.clone());
                        }
                    }
                }
                let mut guard = self.sessions.lock().unwrap();
                let st = guard.entry(sid).or_default();
                st.baseline_checked = true;
                st.baseline_seq = new_seq;
                st.baseline_identity = identity;
                st.represented = next_rep;
                st.omitted = next_omitted;
                st.truncated = next_truncated;
                st.known_subdirs = merged_subdirs;
                st.pending_subdirs
                    .retain(|sub| !st.known_subdirs.contains(sub));
                st.watched_paths = watched;
                st.report = SessionInstructionsReport {
                    enabled: true,
                    project_root: Some(disc.project_root),
                    scopes: r.statuses,
                    warnings: disc.warnings,
                };
            }
            return;
        }

        // 计算与上一次已加载集合的增量 diff
        let all_cur: Vec<&DiscoveredFile> = disc
            .global_files
            .iter()
            .chain(disc.project_files.iter())
            .collect();
        let cur_map: HashMap<ScopeKey, &DiscoveredFile> =
            all_cur.iter().map(|f| (f.scope.clone(), *f)).collect();

        let (prev_rep, prev_omitted, prev_truncated, had_baseline) = {
            let guard = self.sessions.lock().unwrap();
            let st = guard.get(&sid).unwrap();
            (
                st.represented.clone(),
                st.omitted.clone(),
                st.truncated.clone(),
                st.baseline_seq.is_some(),
            )
        };

        let mut delta_items = Vec::new();
        let mut delta_kinds = HashMap::new();

        for file in &all_cur {
            if prev_omitted.contains(&file.scope) {
                delta_kinds.insert(file.scope.clone(), DeltaChangeKind::Added);
                delta_items.push(DeltaItem {
                    scope: file.scope.clone(),
                    display_path: file.display_path.clone(),
                    scope_dir_display: file.scope_dir_display.clone(),
                    kind: DeltaChangeKind::Added,
                    content: file.content.clone(),
                });
                continue;
            }
            match prev_rep.get(&file.scope) {
                None => {
                    delta_kinds.insert(file.scope.clone(), DeltaChangeKind::Added);
                    delta_items.push(DeltaItem {
                        scope: file.scope.clone(),
                        display_path: file.display_path.clone(),
                        scope_dir_display: file.scope_dir_display.clone(),
                        kind: DeltaChangeKind::Added,
                        content: file.content.clone(),
                    });
                }
                Some(old) if old.digest != file.digest => {
                    delta_kinds.insert(file.scope.clone(), DeltaChangeKind::Updated);
                    delta_items.push(DeltaItem {
                        scope: file.scope.clone(),
                        display_path: file.display_path.clone(),
                        scope_dir_display: file.scope_dir_display.clone(),
                        kind: DeltaChangeKind::Updated,
                        content: file.content.clone(),
                    });
                }
                Some(_) => {}
            }
        }

        let mut removed_keys: Vec<ScopeKey> = prev_rep
            .keys()
            .filter(|k| !cur_map.contains_key(*k))
            .cloned()
            .collect();
        removed_keys.sort();
        for key in removed_keys {
            let old = &prev_rep[&key];
            delta_items.push(DeltaItem {
                scope: key,
                display_path: old.display_path.clone(),
                scope_dir_display: old.scope_dir_display.clone(),
                kind: DeltaChangeKind::Removed,
                content: String::new(),
            });
        }

        let rendered_delta = render_delta_message_detailed(&delta_items, self.config.max_bytes);
        let mut emitted_delta_seq = None;
        if let Some(ref d) = rendered_delta {
            session.append(SessionEvent::context_notice(
                AGENT_INSTRUCTIONS_SOURCE,
                &d.text,
            ));
            emitted_delta_seq = session.with_visible_entries(|e| e.last().map(|x| x.seq));
        }

        let mut next_rep = HashMap::new();
        let mut next_omitted = HashSet::new();
        let mut next_truncated = HashSet::new();
        for file in &all_cur {
            if delta_kinds.contains_key(&file.scope) {
                if let Some(ref d) = rendered_delta {
                    if d.omitted_scopes.contains(&file.scope) {
                        next_omitted.insert(file.scope.clone());
                        continue;
                    }
                    if d.represented_scopes.contains(&file.scope) {
                        if d.truncated_scopes.contains(&file.scope) {
                            next_truncated.insert(file.scope.clone());
                        }
                        next_rep.insert(
                            file.scope.clone(),
                            TrackedFile {
                                display_path: file.display_path.clone(),
                                scope_dir_display: file.scope_dir_display.clone(),
                                digest: file.digest.clone(),
                            },
                        );
                        continue;
                    }
                }
                next_omitted.insert(file.scope.clone());
            } else if prev_rep.contains_key(&file.scope) {
                if prev_truncated.contains(&file.scope) {
                    next_truncated.insert(file.scope.clone());
                }
                next_rep.insert(
                    file.scope.clone(),
                    TrackedFile {
                        display_path: file.display_path.clone(),
                        scope_dir_display: file.scope_dir_display.clone(),
                        digest: file.digest.clone(),
                    },
                );
            }
        }

        let next_statuses: Vec<InstructionScopeStatus> = all_cur
            .iter()
            .map(|f| InstructionScopeStatus {
                display_path: f.display_path.clone(),
                scope_kind: f.scope_kind.clone(),
                bytes: f.bytes,
                truncated: next_truncated.contains(&f.scope),
                omitted: next_omitted.contains(&f.scope),
            })
            .collect();

        let mut guard = self.sessions.lock().unwrap();
        let st = guard.entry(sid).or_default();
        if !had_baseline && emitted_delta_seq.is_some() {
            st.baseline_seq = emitted_delta_seq;
        }
        st.represented = next_rep;
        st.omitted = next_omitted;
        st.truncated = next_truncated;
        st.known_subdirs = merged_subdirs;
        st.pending_subdirs
            .retain(|sub| !st.known_subdirs.contains(sub));
        st.watched_paths = watched;
        st.report = SessionInstructionsReport {
            enabled: true,
            project_root: Some(disc.project_root),
            scopes: next_statuses,
            warnings: disc.warnings,
        };
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

    fn create(_ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        Ok(Arc::new(Self {
            config,
            sessions: Mutex::new(HashMap::new()),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let ctx_clone = ctx.clone();
        Box::pin(async move {
            if !self.config.enabled {
                return Ok(());
            }

            let for_touch = Arc::clone(&self);
            ctx_clone.on(EVENT_TOOLS_RESULT, move |ev: &ToolResultEvent| {
                for_touch.record_tool_touch(ev);
            })?;

            let for_step = Arc::clone(&self);
            let ctx_for_step = ctx_clone.clone();
            ctx_clone.on_serial::<AgentPreStepEvent, (), _, _>(
                EVENT_AGENT_PRE_STEP,
                move |ev: AgentPreStepEvent| {
                    let svc = Arc::clone(&for_step);
                    let ctx_inner = ctx_for_step.clone();
                    let session = Arc::clone(&ev.session);
                    let cancel = ev.cancel.clone();
                    async move {
                        svc.reconcile_pre_step(&ctx_inner, &session, &cancel);
                        None
                    }
                },
            )?;

            Ok(())
        })
    }
}

pub fn scope_root_kind(scope: &ScopeKey) -> &'static str {
    match &scope.root {
        ScopeRoot::MizarGlobal => "mizar-global",
        ScopeRoot::AgentsGlobal => "agents-global",
        ScopeRoot::Project(_) => "project",
    }
}
