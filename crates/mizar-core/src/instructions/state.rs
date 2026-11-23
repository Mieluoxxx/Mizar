use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::agent_loop::{AgentPreStepDecision, AgentPreStepInput, EVENT_AGENT_PRE_STEP, TurnInput};
use cordis::{BoxFuture, Context, CordisError, Next, Service};
use mizar_session::{Session, SessionEvent};
use mizar_tools::{EVENT_TOOLS_RESULT, ToolResultEvent};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::instructions::config::{
    AGENT_INSTRUCTIONS_SOURCE, InstructionScopeStatus, InstructionsConfig,
    SessionInstructionsReport,
};
use crate::instructions::digest::sha256_hex;
use crate::instructions::files::{
    DiscoveredFile, FsFingerprint, ScopeKey, ScopeRoot, ancestor_chain, canonical_or_abs,
    descendant_dirs_between, discover_instructions, find_project_root, read_fingerprint,
    resolve_agents_home, resolve_mizar_home, sanitize_candidates,
};
use crate::instructions::render::{
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
    baseline_text: Option<Arc<str>>,
    baseline_seq: Option<u64>,
    baseline_identity: String,
    represented: HashMap<ScopeKey, TrackedFile>,
    omitted: HashSet<ScopeKey>,
    truncated: HashSet<ScopeKey>,
    known_subdirs: Vec<PathBuf>,
    pending_subdirs: Vec<PathBuf>,
    watched_paths: Arc<HashMap<PathBuf, Option<FsFingerprint>>>,
    report: SessionInstructionsReport,
}

async fn run_blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context as TaskContext, Poll, Waker};

    struct BlockingFuture<T> {
        state: Arc<Mutex<(Option<T>, Option<Waker>)>>,
    }

    impl<T> Future for BlockingFuture<T> {
        type Output = T;

        fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<T> {
            let mut guard = self.state.lock().unwrap();
            if let Some(val) = guard.0.take() {
                Poll::Ready(val)
            } else {
                guard.1 = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }

    let state = Arc::new(Mutex::new((None, None::<Waker>)));
    let state_bg = Arc::clone(&state);
    std::thread::Builder::new()
        .name("mizar-instructions-io".to_string())
        .spawn(move || {
            let res = f();
            let waker = {
                let mut guard = state_bg.lock().unwrap();
                guard.0 = Some(res);
                guard.1.take()
            };
            if let Some(w) = waker {
                w.wake();
            }
        })
        .expect("spawn mizar-instructions-io blocking thread");
    BlockingFuture { state }.await
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

    fn log_warnings(warnings: &[String]) {
        for w in warnings {
            tracing::warn!(target: "mizar::instructions", "{w}");
        }
    }

    fn check_visible_baseline(
        session: &Session,
        baseline_seq: Option<u64>,
        baseline_text: Option<&str>,
    ) -> (bool, Option<u64>, bool) {
        session.with_visible_entries(|entries| {
            let mut has_any_notice = false;
            let mut matched_seq = None;
            for e in entries {
                if let SessionEvent::ContextNotice { source, text, .. } = &e.event
                    && source == AGENT_INSTRUCTIONS_SOURCE
                {
                    has_any_notice = true;
                    if let Some(seq) = baseline_seq {
                        if e.seq == seq {
                            matched_seq = Some(seq);
                        }
                    } else if let Some(bt) = baseline_text
                        && text == bt
                    {
                        matched_seq = Some(e.seq);
                    }
                }
            }
            (matched_seq.is_some(), matched_seq, has_any_notice)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_baseline_result(
        &self,
        session: &Session,
        sid: String,
        mem_baseline_text: Option<Arc<str>>,
        has_any_notice: bool,
        merged_subdirs: Vec<PathBuf>,
        disc: crate::instructions::files::DiscoveryResult,
        watched: HashMap<PathBuf, Option<FsFingerprint>>,
    ) -> Option<TurnInput> {
        Self::log_warnings(&disc.warnings);
        let identity = self.compute_baseline_identity(&disc.project_root);
        let rendered = render_baseline_message(
            &disc.global_files,
            &disc.project_files,
            self.config.max_bytes,
        );

        // 若刚从持久化恢复（内存态为空但日志里已存在相同基线），仅借用比较避免克隆 64KB 正文
        let hydrated = if mem_baseline_text.is_none() {
            rendered.as_ref().and_then(|r| {
                session.with_visible_entries(|entries| {
                    entries
                        .iter()
                        .rfind(|e| {
                            matches!(
                                &e.event,
                                SessionEvent::ContextNotice { source, text, .. }
                                    if source == AGENT_INSTRUCTIONS_SOURCE && text == &r.text
                            )
                        })
                        .map(|e| e.seq)
                })
            })
        } else {
            None
        };

        let mut emitted_baseline_text: Option<Arc<str>> = None;
        let mut emitted_baseline_seq: Option<u64> = hydrated;
        let mut emitted_notice: Option<String> = None;

        if let Some(seq) = hydrated {
            if let Some(ref r) = rendered {
                emitted_baseline_text = Some(Arc::from(r.text.as_str()));
            }
            emitted_baseline_seq = Some(seq);
        } else if let Some(ref r) = rendered {
            emitted_notice = Some(r.text.clone());
            emitted_baseline_text = Some(Arc::from(r.text.as_str()));
        } else if has_any_notice {
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
                emitted_baseline_text = Some(Arc::from(delta_text.as_str()));
                emitted_notice = Some(delta_text);
            } else if mem_baseline_text.is_none() {
                // 离线恢复时此前有基线，但本地指令文件已全部删除
                let msg = "<system-reminder>\nInstructions removed: previously loaded instruction files are no longer present.\n</system-reminder>".to_string();
                emitted_baseline_text = Some(Arc::from(msg.as_str()));
                emitted_notice = Some(msg);
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
        st.baseline_text = emitted_baseline_text;
        st.baseline_seq = emitted_baseline_seq;
        st.baseline_identity = identity;
        st.represented = next_rep;
        st.omitted = next_omitted;
        st.truncated = next_truncated;
        st.known_subdirs = merged_subdirs;
        st.pending_subdirs
            .retain(|sub| !st.known_subdirs.contains(sub));
        st.watched_paths = Arc::new(watched);
        st.report = SessionInstructionsReport {
            enabled: true,
            project_root: Some(disc.project_root),
            scopes: statuses,
            warnings: disc.warnings,
        };
        emitted_notice.map(|text| TurnInput::notice(AGENT_INSTRUCTIONS_SOURCE, text))
    }

    fn apply_delta_result(
        &self,
        sid: String,
        mem_identity: &str,
        merged_subdirs: Vec<PathBuf>,
        disc: crate::instructions::files::DiscoveryResult,
        watched: HashMap<PathBuf, Option<FsFingerprint>>,
    ) -> Option<TurnInput> {
        Self::log_warnings(&disc.warnings);
        let identity = self.compute_baseline_identity(&disc.project_root);

        if identity != mem_identity {
            // 极少见：workspace root/配置发生变化，重新生成完整基线
            let mut emitted_notice: Option<String> = None;
            if let Some(r) = render_baseline_message(
                &disc.global_files,
                &disc.project_files,
                self.config.max_bytes,
            ) {
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
                let baseline_arc: Arc<str> = Arc::from(r.text.as_str());
                emitted_notice = Some(r.text);
                let mut guard = self.sessions.lock().unwrap();
                let st = guard.entry(sid).or_default();
                st.baseline_checked = true;
                st.baseline_text = Some(baseline_arc);
                st.baseline_seq = None;
                st.baseline_identity = identity;
                st.represented = next_rep;
                st.omitted = next_omitted;
                st.truncated = next_truncated;
                st.known_subdirs = merged_subdirs;
                st.pending_subdirs
                    .retain(|sub| !st.known_subdirs.contains(sub));
                st.watched_paths = Arc::new(watched);
                st.report = SessionInstructionsReport {
                    enabled: true,
                    project_root: Some(disc.project_root),
                    scopes: r.statuses,
                    warnings: disc.warnings,
                };
            }
            return emitted_notice.map(|text| TurnInput::notice(AGENT_INSTRUCTIONS_SOURCE, text));
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
                st.baseline_text.is_some(),
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
        let emitted_notice = rendered_delta.as_ref().map(|d| d.text.clone());

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
        if !had_baseline && let Some(ref text) = emitted_notice {
            st.baseline_text = Some(Arc::from(text.as_str()));
            st.baseline_seq = None;
        }
        st.represented = next_rep;
        st.omitted = next_omitted;
        st.truncated = next_truncated;
        st.known_subdirs = merged_subdirs;
        st.pending_subdirs
            .retain(|sub| !st.known_subdirs.contains(sub));
        st.watched_paths = Arc::new(watched);
        st.report = SessionInstructionsReport {
            enabled: true,
            project_root: Some(disc.project_root),
            scopes: next_statuses,
            warnings: disc.warnings,
        };
        emitted_notice.map(|text| TurnInput::notice(AGENT_INSTRUCTIONS_SOURCE, text))
    }

    pub async fn reconcile_pre_step_async(
        self: &Arc<Self>,
        _ctx: &Context,
        session: &Arc<Session>,
        cancel: &CancellationToken,
    ) -> Option<TurnInput> {
        if !self.config.enabled || self.config.max_bytes == 0 || cancel.is_cancelled() {
            return None;
        }

        let sid = session.id.clone();
        // 锁内仅提取轻量引用/字段，避免在持锁期间做任何文件 stat
        let (
            mem_baseline_checked,
            mem_baseline_text,
            mem_baseline_seq,
            mem_identity,
            pending_subdirs,
            known_subdirs,
            watched_paths,
        ) = {
            let guard = self.sessions.lock().unwrap();
            match guard.get(&sid) {
                Some(st) => (
                    st.baseline_checked,
                    st.baseline_text.clone(),
                    st.baseline_seq,
                    st.baseline_identity.clone(),
                    st.pending_subdirs.clone(),
                    st.known_subdirs.clone(),
                    Arc::clone(&st.watched_paths),
                ),
                None => (
                    false,
                    None,
                    None,
                    String::new(),
                    Vec::new(),
                    Vec::new(),
                    Arc::new(HashMap::new()),
                ),
            }
        };

        let (baseline_visible, matched_seq, has_any_notice) =
            Self::check_visible_baseline(session, mem_baseline_seq, mem_baseline_text.as_deref());
        if mem_baseline_seq.is_none()
            && let Some(seq) = matched_seq
            && let Some(st) = self.sessions.lock().unwrap().get_mut(&sid)
        {
            st.baseline_seq = Some(seq);
        }

        if !mem_baseline_checked || (mem_baseline_text.is_some() && !baseline_visible) {
            let mut merged_subdirs = known_subdirs;
            for sub in pending_subdirs {
                if !merged_subdirs.contains(&sub) {
                    merged_subdirs.push(sub);
                }
            }
            let svc = Arc::clone(self);
            let subdirs_for_io = merged_subdirs.clone();
            let (disc, watched) = run_blocking(move || {
                let disc = discover_instructions(&svc.config, &subdirs_for_io);
                let watched = svc.collect_watched_paths(&subdirs_for_io);
                (disc, watched)
            })
            .await;
            return self.apply_baseline_result(
                session,
                sid,
                mem_baseline_text,
                has_any_notice,
                merged_subdirs,
                disc,
                watched,
            );
        }

        let watched_empty_or_changed = if watched_paths.is_empty() && pending_subdirs.is_empty() {
            false
        } else {
            let watched_c = Arc::clone(&watched_paths);
            run_blocking(move || Self::fingerprints_changed(&watched_c)).await
        };

        // 快速路径：无待扫子目录且任何探测文件元数据都未变化
        if pending_subdirs.is_empty() && !watched_empty_or_changed {
            return None;
        }

        let mut merged_subdirs = known_subdirs;
        for sub in pending_subdirs {
            if !merged_subdirs.contains(&sub) {
                merged_subdirs.push(sub);
            }
        }
        let svc = Arc::clone(self);
        let subdirs_for_io = merged_subdirs.clone();
        let (disc, watched) = run_blocking(move || {
            let disc = discover_instructions(&svc.config, &subdirs_for_io);
            let watched = svc.collect_watched_paths(&subdirs_for_io);
            (disc, watched)
        })
        .await;
        self.apply_delta_result(sid, &mem_identity, merged_subdirs, disc, watched)
    }

    pub fn reconcile_pre_step(
        &self,
        _ctx: &Context,
        session: &Arc<Session>,
        cancel: &CancellationToken,
    ) -> Option<TurnInput> {
        if !self.config.enabled || self.config.max_bytes == 0 || cancel.is_cancelled() {
            return None;
        }

        let sid = session.id.clone();
        let (
            mem_baseline_checked,
            mem_baseline_text,
            mem_baseline_seq,
            mem_identity,
            pending_subdirs,
            known_subdirs,
            watched_paths,
        ) = {
            let guard = self.sessions.lock().unwrap();
            match guard.get(&sid) {
                Some(st) => (
                    st.baseline_checked,
                    st.baseline_text.clone(),
                    st.baseline_seq,
                    st.baseline_identity.clone(),
                    st.pending_subdirs.clone(),
                    st.known_subdirs.clone(),
                    Arc::clone(&st.watched_paths),
                ),
                None => (
                    false,
                    None,
                    None,
                    String::new(),
                    Vec::new(),
                    Vec::new(),
                    Arc::new(HashMap::new()),
                ),
            }
        };

        let (baseline_visible, matched_seq, has_any_notice) =
            Self::check_visible_baseline(session, mem_baseline_seq, mem_baseline_text.as_deref());
        if mem_baseline_seq.is_none()
            && let Some(seq) = matched_seq
            && let Some(st) = self.sessions.lock().unwrap().get_mut(&sid)
        {
            st.baseline_seq = Some(seq);
        }

        if !mem_baseline_checked || (mem_baseline_text.is_some() && !baseline_visible) {
            let mut merged_subdirs = known_subdirs;
            for sub in pending_subdirs {
                if !merged_subdirs.contains(&sub) {
                    merged_subdirs.push(sub);
                }
            }
            let disc = discover_instructions(&self.config, &merged_subdirs);
            let watched = self.collect_watched_paths(&merged_subdirs);
            return self.apply_baseline_result(
                session,
                sid,
                mem_baseline_text,
                has_any_notice,
                merged_subdirs,
                disc,
                watched,
            );
        }

        let watched_empty_or_changed = (!watched_paths.is_empty() || !pending_subdirs.is_empty())
            && Self::fingerprints_changed(&watched_paths);

        // 快速路径：无待扫子目录且任何探测文件元数据都未变化
        if pending_subdirs.is_empty() && !watched_empty_or_changed {
            return None;
        }

        let mut merged_subdirs = known_subdirs;
        for sub in pending_subdirs {
            if !merged_subdirs.contains(&sub) {
                merged_subdirs.push(sub);
            }
        }
        let disc = discover_instructions(&self.config, &merged_subdirs);
        let watched = self.collect_watched_paths(&merged_subdirs);
        self.apply_delta_result(sid, &mem_identity, merged_subdirs, disc, watched)
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
            ctx_clone.on_waterfall(
                EVENT_AGENT_PRE_STEP,
                move |input: AgentPreStepInput,
                      next: Next<
                    AgentPreStepInput,
                    BoxFuture<'static, AgentPreStepDecision>,
                >| {
                    let svc = Arc::clone(&for_step);
                    let ctx_inner = ctx_for_step.clone();
                    let session = Arc::clone(&input.session);
                    let cancel = input.cancel.clone();
                    Box::pin(async move {
                        if cancel.is_cancelled() {
                            return AgentPreStepDecision::Reject {
                                reason: "cancelled".to_string(),
                            };
                        }
                        let AgentPreStepDecision::Enter { mut inputs } = next.call(input).await
                        else {
                            return AgentPreStepDecision::Reject {
                                reason: "rejected".to_string(),
                            };
                        };
                        if !cancel.is_cancelled()
                            && let Some(notice) =
                                svc.reconcile_pre_step_async(&ctx_inner, &session, &cancel)
                                    .await
                        {
                            inputs.insert(0, notice);
                        }
                        AgentPreStepDecision::Enter { inputs }
                    }) as BoxFuture<'static, AgentPreStepDecision>
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

#[cfg(test)]
mod tests {
    use super::*;
    use mizar_session::{SessionContextExt, SessionStore};

    #[tokio::test]
    async fn baseline_seq_latches_and_rebuilds_after_compaction() {
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let ws =
            std::env::temp_dir().join(format!("mizar_instr_seq_{}_{uniq}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("AGENTS.md"), "keep instruction").unwrap();

        let ctx = Context::new();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        let mut cfg = InstructionsConfig::new(ws.clone());
        cfg.include_default_global = false;
        let svc = AgentInstructions::create(&ctx, cfg).unwrap();
        let session = ctx.sessions().unwrap().create(None);
        let cancel = CancellationToken::new();

        let first = svc
            .reconcile_pre_step_async(&ctx, &session, &cancel)
            .await
            .expect("initial baseline");
        session.append(SessionEvent::ContextNotice {
            source: AGENT_INSTRUCTIONS_SOURCE.to_string(),
            text: first.text().to_string(),
            images: Vec::new(),
        });
        assert!(
            svc.sessions
                .lock()
                .unwrap()
                .get(&session.id)
                .unwrap()
                .baseline_seq
                .is_none()
        );

        // 下一步触发 seq 锁存，且文件未改动时无重复注入
        assert!(
            svc.reconcile_pre_step_async(&ctx, &session, &cancel)
                .await
                .is_none()
        );
        let latched = svc
            .sessions
            .lock()
            .unwrap()
            .get(&session.id)
            .unwrap()
            .baseline_seq;
        assert!(latched.is_some());

        // 压缩将原基线折叠出可见窗口后，应自动触发重注入并重置 seq
        session.append(SessionEvent::user("after"));
        session.compact("summary", 1, None);
        let rebuilt = svc
            .reconcile_pre_step_async(&ctx, &session, &cancel)
            .await
            .expect("rebuilt baseline after compaction");
        assert!(rebuilt.text().contains("keep instruction"));
        assert!(
            svc.sessions
                .lock()
                .unwrap()
                .get(&session.id)
                .unwrap()
                .baseline_seq
                .is_none()
        );

        let _ = std::fs::remove_dir_all(&ws);
    }
}
