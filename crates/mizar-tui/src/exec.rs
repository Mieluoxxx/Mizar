//! 无头调用方依赖标准输出解析结果，诊断信息必须走标准错误。

mod ndjson;

use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cordis::{Context, LogLevel, Service};
use mizar_agent_loop::{
    AgentError, AgentLoop, AgentRetryEvent, EVENT_AGENT_RETRY, RunOptions, TurnEndReason,
    TurnRunReport,
};
use mizar_agent_spine::{
    AgentSpineConfig, AgentSpinePlugin, AgentToolsMode, CliLlmOverride, EnvLlmOverride,
    load_mizar_config, load_mizar_config_isolated, resolve_default_sessions_dir,
};
use mizar_client::mentions::{FileMentionsConfig, FileMentionsPlugin};
use mizar_client_core::{ClientEvent, HostExtensionFlags, NoticeLevel};
use mizar_llm::{LlmError, LlmErrorCode, ReasoningEffort};
use mizar_llm_router::LlmRouter;
use mizar_mcp::McpManager;
use mizar_session::{Session, SessionContextExt, SessionListQuery, SessionPersistenceContextExt};
use mizar_tools_coding::CodingToolsPreset;
use mizar_trajectory::{
    DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS, TimelineMode, TraceExportMeta, TrajectoryExport,
    TrajectoryRuntime, write_trace_export_atomic,
};
use tokio_util::sync::CancellationToken;

use crate::runner::{ResumeTarget, wait_for_services};
use ndjson::{ExecStream, now_ms, write_event};

/// 无头输出格式。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecOutputFormat {
    #[default]
    Text,
    Json,
}

impl ExecOutputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
        }
    }
}

/// 一次性无头执行配置。
#[derive(Debug, Clone)]
pub struct ExecConfig {
    pub prompt: String,
    pub workspace: PathBuf,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub protocol: Option<String>,
    pub base_url: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub max_steps: Option<usize>,
    pub max_tokens: Option<u32>,
    pub context_window: Option<u64>,
    pub timeout_secs: Option<u64>,
    pub isolated: bool,
    pub no_persist: bool,
    pub no_instructions: bool,
    pub approve: bool,
    pub preset: Option<CodingToolsPreset>,
    pub include_tools: Option<Vec<String>>,
    pub exclude_tools: Vec<String>,
    pub resume_session: Option<ResumeTarget>,
    pub output_format: ExecOutputFormat,
    pub trace_out: Option<PathBuf>,
}

/// 解析无头轨迹导出路径：CLI `--trace-out` 优先级高于环境变量 `MIZAR_TRACE_OUT`。
pub fn resolve_trace_out_path(cli: Option<PathBuf>, env: Option<String>) -> Option<PathBuf> {
    cli.or_else(|| env.filter(|s| !s.trim().is_empty()).map(PathBuf::from))
}

/// 解析来自命令行参数、文件或标准输入的任务文本；拒绝空白与冲突来源。
pub fn resolve_exec_prompt(
    explicit_exec: bool,
    prompt_flag: Option<String>,
    prompt_file: Option<PathBuf>,
    positional: Option<String>,
) -> Result<String, String> {
    match (prompt_flag, prompt_file, positional) {
        (Some(_), Some(_), _)
        | (Some(_), _, Some(_))
        | (_, Some(_), Some(_)) => Err(
            "conflicting prompt sources: provide only one of positional <prompt>, -p/--prompt, or --prompt-file"
                .to_string(),
        ),
        (Some(text), None, None) => validate_non_blank_prompt(text, "-p/--prompt"),
        (None, Some(path), None) => {
            let text = std::fs::read_to_string(&path).map_err(|e| {
                format!("failed to read --prompt-file \"{}\": {e}", path.display())
            })?;
            validate_non_blank_prompt(text, "--prompt-file")
        }
        (None, None, Some(text)) => validate_non_blank_prompt(text, "positional prompt"),
        (None, None, None) => {
            if !explicit_exec {
                return Err("no prompt provided for headless execution".to_string());
            }
            if std::io::stdin().is_terminal() {
                return Err(
                    "exec requires a prompt via <prompt>, -p/--prompt, --prompt-file, or piped stdin"
                        .to_string(),
                );
            }
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("failed to read prompt from stdin: {e}"))?;
            validate_non_blank_prompt(buf, "stdin")
        }
    }
}

fn validate_non_blank_prompt(text: String, source: &str) -> Result<String, String> {
    if text.trim().is_empty() {
        return Err(format!("empty or whitespace-only prompt from {source}"));
    }
    Ok(text)
}

fn single_line(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_control() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

/// 对错误文本与警告日志进行凭据脱敏：先精确替换已知密钥，再匹配 Bearer / Header / Query 参数与常见 Key 前缀。
pub fn sanitize_diagnostic_text(input: &str, secrets: &[String]) -> String {
    let mut out = input.to_string();

    let mut deduped: Vec<&str> = secrets
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !s.trim().is_empty())
        .collect();
    deduped.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    deduped.dedup();
    for secret in deduped {
        out = out.replace(secret, "[REDACTED]");
    }

    out = redact_named_secrets(&single_line(&out));
    out = redact_bearer_tokens(&out);
    redact_prefixed_tokens(&out)
}

fn env_secrets() -> Vec<String> {
    [
        "API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    .into_iter()
    .filter_map(|key| std::env::var(key).ok())
    .collect()
}

fn is_token_delimiter(b: u8) -> bool {
    b.is_ascii_whitespace()
        || matches!(
            b,
            b'"' | b'\'' | b',' | b';' | b')' | b']' | b'}' | b'<' | b'>'
        )
}

fn is_secret_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase().replace(['-', '_'], "");
    normalized.ends_with("apikey")
        || normalized.ends_with("token")
        || normalized.ends_with("secret")
        || matches!(
            normalized.as_str(),
            "authorization" | "proxyauthorization" | "key" | "password"
        )
}

fn redact_named_secrets(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut copied = 0;
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        while i < bytes.len()
            && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'-' | b'_'))
        {
            i += 1;
        }
        if start == i {
            i += 1;
            continue;
        }
        let key = &input[start..i];
        let rest = input[i..].trim_start_matches(['\"', '\'']).trim_start();
        let Some(value) = rest.strip_prefix([':', '=']).map(str::trim_start) else {
            continue;
        };
        if !is_secret_key(key) {
            continue;
        }
        let mut pos = input.len() - value.len();
        let quoted = matches!(bytes.get(pos), Some(b'\"' | b'\''));
        if quoted {
            pos += 1;
        }
        let mut end = pos;
        if quoted {
            let quote = bytes[pos - 1];
            while end < bytes.len() && bytes[end] != quote {
                if bytes[end] == b'\\' && end + 1 < bytes.len() {
                    end += 1;
                }
                end += 1;
            }
        } else {
            // 认证方案和凭据之间可以有空格，不能只遮住方案名。
            if key.to_ascii_lowercase().ends_with("authorization")
                && let Some((scheme, token)) = value.split_once(' ')
                && (scheme.eq_ignore_ascii_case("bearer") || scheme.eq_ignore_ascii_case("basic"))
            {
                pos = input.len() - token.trim_start().len();
                end = pos;
            }
            if input[pos..].starts_with("[REDACTED]") {
                end += "[REDACTED]".len();
            }
            while end < bytes.len() && bytes[end] != b'&' && !is_token_delimiter(bytes[end]) {
                end += 1;
            }
        }
        out.push_str(&input[copied..pos]);
        out.push_str("[REDACTED]");
        copied = end;
        i = end;
    }
    out.push_str(&input[copied..]);
    out
}

fn redact_bearer_tokens(input: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let mut result = String::with_capacity(input.len());
    let mut cursor = 0usize;

    while let Some(rel) = lower[cursor..].find("bearer ") {
        let pos = cursor + rel;
        let boundary_ok = pos == 0 || !input.as_bytes()[pos - 1].is_ascii_alphanumeric();
        let after_prefix = pos + "bearer ".len();
        if !boundary_ok {
            result.push_str(&input[cursor..after_prefix]);
            cursor = after_prefix;
            continue;
        }
        result.push_str(&input[cursor..after_prefix]);
        let mut tok_start = after_prefix;
        while tok_start < input.len() && input.as_bytes()[tok_start] == b' ' {
            result.push(' ');
            tok_start += 1;
        }
        let mut tok_end = tok_start;
        let bytes = input.as_bytes();
        while tok_end < bytes.len() && !is_token_delimiter(bytes[tok_end]) {
            tok_end += 1;
        }
        let tok = &input[tok_start..tok_end];
        if !tok.is_empty() && !input[tok_start..].starts_with("[REDACTED]") {
            result.push_str("[REDACTED]");
        } else {
            result.push_str(tok);
        }
        cursor = tok_end;
    }
    result.push_str(&input[cursor..]);
    result
}

fn redact_prefixed_tokens(input: &str) -> String {
    let prefixes = ["sk-", "ghp_", "github_pat_"];
    let mut result = String::with_capacity(input.len());
    let mut cursor = 0usize;
    let bytes = input.as_bytes();

    while cursor < input.len() {
        let mut next_match: Option<(usize, &str)> = None;
        for &pat in &prefixes {
            if let Some(rel) = input[cursor..].find(pat) {
                let pos = cursor + rel;
                if next_match.is_none_or(|(best, _)| pos < best) {
                    next_match = Some((pos, pat));
                }
            }
        }
        let Some((pos, pat)) = next_match else {
            result.push_str(&input[cursor..]);
            break;
        };
        let boundary_ok =
            pos == 0 || (!bytes[pos - 1].is_ascii_alphanumeric() && bytes[pos - 1] != b'_');
        let body_start = pos + pat.len();
        let mut body_end = body_start;
        while body_end < bytes.len()
            && (bytes[body_end].is_ascii_alphanumeric()
                || matches!(bytes[body_end], b'-' | b'_' | b'.'))
        {
            body_end += 1;
        }
        let body_len = body_end - body_start;
        if boundary_ok && body_len >= 8 {
            result.push_str(&input[cursor..pos]);
            result.push_str("[REDACTED]");
            cursor = body_end;
        } else {
            result.push_str(&input[cursor..body_start]);
            cursor = body_start;
        }
    }
    result
}

/// 格式化单行诊断错误，保留 error code / HTTP status / provider message / retry_after_ms，
/// 并兼容 Harbor `BaseInstalledAgent.ERROR_PATTERNS`（末尾匹配优先 `LastMatchWins`）。
pub fn format_llm_error_diagnostic(err: &LlmError, secrets: &[String]) -> String {
    let sanitized_msg = sanitize_diagnostic_text(&err.message, secrets);
    let lower = sanitized_msg.to_ascii_lowercase();

    let mut meta = Vec::new();
    if let Some(status) = err.status {
        meta.push(format!("status={status}"));
    }
    if let Some(retry_after_ms) = err.retry_after_ms {
        meta.push(format!("retry_after_ms={retry_after_ms}"));
    }
    meta.push(format!("code={}", err.code.as_str()));

    match err.code {
        LlmErrorCode::ContextWindowExceeded => {
            meta.push("detail=input token count exceeds the maximum number of tokens".to_string());
        }
        LlmErrorCode::Auth | LlmErrorCode::InvalidCredential => {
            let already_matches_harbor = lower.contains("invalid_api_key")
                || lower.contains("invalid api key")
                || lower.contains("api key is invalid")
                || lower.contains("incorrect api key provided")
                || lower.contains("unexpected status 401 unauthorized");
            if !already_matches_harbor {
                if err.status == Some(401) || err.code == LlmErrorCode::Auth {
                    meta.push("detail=unexpected status 401 Unauthorized".to_string());
                } else {
                    meta.push("detail=Invalid API key".to_string());
                }
            }
        }
        LlmErrorCode::MissingCredential => {
            meta.push("detail=Not logged in: missing credential".to_string());
        }
        LlmErrorCode::Quota if !lower.contains("quota exceeded") => {
            meta.push("detail=Quota exceeded.".to_string());
        }
        _ => {}
    }

    if sanitized_msg.is_empty() {
        format!("({})", meta.join(", "))
    } else {
        format!("{sanitized_msg} ({})", meta.join(", "))
    }
}

fn configure_exec_logger(app: &Context, secrets: Arc<Mutex<Vec<String>>>) {
    app.logger().set_level(LogLevel::Warn);
    app.logger().set_handler(move |rec| {
        if matches!(rec.level, LogLevel::Warn | LogLevel::Error) {
            let line = format!("[{} {}] {}", rec.tag, rec.name, rec.message);
            eprintln!(
                "{}",
                sanitize_diagnostic_text(&line, &secrets.lock().unwrap())
            );
        }
    });
}

/// 运行无头一次性任务，返回进程应使用的退出码。
pub async fn run_exec_cli(cfg: ExecConfig) -> i32 {
    let exec_started = std::time::Instant::now();
    let log_secrets = Arc::new(Mutex::new(env_secrets()));
    let stream = Arc::new(Mutex::new(ExecStream::default()));
    let finish = |mut outcome: TurnExitOutcome,
                  app: Option<&Context>,
                  session: Option<(&Session, bool, u64)>,
                  report: Option<&TurnRunReport>,
                  provider: Option<String>,
                  model: Option<String>| {
        let json_meta = (cfg.output_format == ExecOutputFormat::Json
            && outcome.termination_reason != "invalid_request")
            .then(|| {
                let mut stream = stream.lock().unwrap();
                if let Some(err) = stream.write_error.take()
                    && outcome.exit_code == 0
                {
                    outcome =
                        TurnExitOutcome::failure("error", format!("failed to write stdout: {err}"));
                }
                let (session, resumed, duration) = match session {
                    Some((s, resumed, duration)) => (Some(s), resumed, duration),
                    None => (None, false, exec_started.elapsed().as_millis() as u64),
                };
                stream.result_fields(session, resumed, duration)
            });
        if let Some(err) = &mut outcome.error {
            *err = sanitize_diagnostic_text(err, &log_secrets.lock().unwrap());
            eprintln!("error: {err}");
        }
        if let Err(err) = outcome.write_output(cfg.output_format, report, json_meta) {
            let msg = sanitize_diagnostic_text(
                &format!("failed to write stdout: {err}"),
                &log_secrets.lock().unwrap(),
            );
            eprintln!("error: {msg}");
            if outcome.exit_code == 0 {
                outcome = TurnExitOutcome::failure("error", msg);
            }
        }
        let code = outcome.exit_code;
        export_exec_trace_best_effort(
            cfg.trace_out.as_deref(),
            app,
            session.map(|(session, _, _)| session.id.as_str()),
            TraceExportMeta {
                exit_code: code,
                termination_reason: outcome.termination_reason.to_owned(),
                error: outcome.error,
                duration_ms: exec_started.elapsed().as_millis() as u64,
                provider,
                model,
                isolated: cfg.isolated,
                output_format: Some(cfg.output_format.as_str().to_owned()),
            },
            &log_secrets,
        );
        code
    };

    if cfg.approve {
        let msg = "--approve is incompatible with non-interactive exec mode (code=invalid_request)";
        return finish(
            TurnExitOutcome::failure("invalid_request", msg),
            None,
            None,
            None,
            cfg.provider.clone(),
            cfg.model.clone(),
        );
    }
    if let Some(ref target) = cfg.resume_session {
        if matches!(target, ResumeTarget::Picker) {
            let msg = "--resume requires a session id in non-interactive exec mode (code=invalid_request)";
            return finish(
                TurnExitOutcome::failure("invalid_request", msg),
                None,
                None,
                None,
                cfg.provider.clone(),
                cfg.model.clone(),
            );
        }
        if cfg.no_persist || cfg.isolated {
            let msg = "--continue/--resume is incompatible with --no-persist/--isolated (code=invalid_request)";
            return finish(
                TurnExitOutcome::failure("invalid_request", msg),
                None,
                None,
                None,
                cfg.provider.clone(),
                cfg.model.clone(),
            );
        }
    }

    let loaded_cfg = if cfg.isolated {
        load_mizar_config_isolated(&cfg.workspace)
    } else {
        load_mizar_config(&cfg.workspace)
    };
    {
        let mut secrets = log_secrets.lock().unwrap();
        secrets.extend(
            loaded_cfg
                .router
                .providers
                .values()
                .filter_map(|r| r.api_key.clone()),
        );
    }

    if let Some(ui) = &loaded_cfg.settings.ui
        && let Some(ref mode) = ui.approval_mode
        && mode.trim().eq_ignore_ascii_case("ask")
    {
        let msg = "ui.approvalMode=\"ask\" is incompatible with non-interactive exec mode (code=invalid_request)";
        return finish(
            TurnExitOutcome::failure("invalid_request", msg),
            None,
            None,
            None,
            cfg.provider.clone(),
            cfg.model.clone(),
        );
    }

    let invalid_config = |err: String| {
        finish(
            TurnExitOutcome::failure("invalid_request", err),
            None,
            None,
            None,
            cfg.provider.clone(),
            cfg.model.clone(),
        )
    };
    if !loaded_cfg.errors.is_empty() {
        return invalid_config(loaded_cfg.errors.join(", "));
    }
    let cli_override = CliLlmOverride {
        provider: cfg.provider.clone(),
        model: cfg.model.clone(),
        protocol: cfg.protocol.clone(),
        base_url: cfg.base_url.clone(),
        reasoning_effort: cfg.reasoning_effort,
        context_window: cfg.context_window,
    };
    let env_override = match EnvLlmOverride::from_env(&cli_override) {
        Ok(env) => env,
        Err(err) => return invalid_config(err),
    };
    let mut spine_cfg = AgentSpineConfig::coding(cfg.workspace.clone());
    if !cfg.isolated && !cfg.no_persist {
        spine_cfg.sessions_dir = resolve_default_sessions_dir();
    }
    loaded_cfg.apply_to_spine(&mut spine_cfg);
    cli_override.apply_context_window(&env_override, &mut spine_cfg);
    let resolved_llm = match loaded_cfg.resolve_llm_router_with_cli(env_override, cli_override) {
        Ok(resolved) => resolved,
        Err(err) => return invalid_config(err.to_string()),
    };

    match resolved_llm.validate_active_route() {
        Ok(maybe_secret) => {
            if let Some(secret) = maybe_secret {
                let mut s = log_secrets.lock().unwrap();
                if !s.contains(&secret) {
                    s.push(secret);
                }
            }
        }
        Err(err) => {
            let diag = {
                let guard = log_secrets.lock().unwrap();
                format_llm_error_diagnostic(&err, &guard)
            };
            return finish(
                TurnExitOutcome::failure("invalid_request", diag),
                None,
                None,
                None,
                Some(resolved_llm.active_provider.clone()),
                resolved_llm.active_model.clone(),
            );
        }
    }

    if !spine_cfg
        .disabled_plugins
        .iter()
        .any(|id| id == "session-title")
    {
        spine_cfg.disabled_plugins.push("session-title".to_string());
    }
    if cfg.no_persist || (cfg.isolated && loaded_cfg.settings.sessions_dir.is_none()) {
        spine_cfg.sessions_dir = None;
    }
    if cfg.no_instructions
        && let AgentToolsMode::Coding {
            ref mut instructions,
            ..
        } = spine_cfg.tools_mode
    {
        instructions.enabled = false;
    }
    if let Some(preset) = cfg.preset {
        spine_cfg = spine_cfg.with_coding_preset(preset);
    }
    if cfg.include_tools.is_some() {
        spine_cfg.include_tools = cfg.include_tools.clone();
    }
    for pat in &cfg.exclude_tools {
        if !spine_cfg.exclude_tools.contains(pat) {
            spine_cfg.exclude_tools.push(pat.clone());
        }
    }

    spine_cfg.agent.provider = Some(resolved_llm.active_provider.clone());
    spine_cfg.agent.model = resolved_llm.active_model.clone();
    let effective_effort = cfg
        .reasoning_effort
        .or(loaded_cfg.settings.reasoning_effort);
    if let Some(effort) = effective_effort {
        spine_cfg.agent.reasoning_effort = Some(effort);
    }
    if let Some(limit) = cfg.max_steps {
        spine_cfg.agent.max_steps = Some(limit.max(1));
    }
    if let Some(mt) = cfg.max_tokens {
        spine_cfg.agent.max_tokens = Some(mt.max(1));
    }
    let configured_max_steps = spine_cfg.agent.max_steps;
    let configured_max_tokens = spine_cfg.agent.max_tokens;
    let has_sessions_dir = spine_cfg.sessions_dir.is_some();
    let active_provider = resolved_llm.active_provider.clone();
    let active_model = resolved_llm.active_model.clone();

    if cfg.resume_session.is_some() && !has_sessions_dir {
        let msg = "--continue/--resume requires session persistence directory configuration (code=invalid_request)";
        return finish(
            TurnExitOutcome::failure("invalid_request", msg),
            None,
            None,
            None,
            Some(active_provider.clone()),
            active_model.clone(),
        );
    }

    let app = Context::new();
    configure_exec_logger(&app, Arc::clone(&log_secrets));

    let ext_flags = HostExtensionFlags::default();
    let restrict_to_workspace = match &spine_cfg.tools_mode {
        AgentToolsMode::Coding { coding, .. } => coding.restrict_to_workspace,
        _ => false,
    };

    let include_tools = spine_cfg.include_tools.clone();
    app.plugin(AgentSpinePlugin, spine_cfg);
    let retry_secrets = Arc::clone(&log_secrets);
    let _retry_subscription = app
        .on(EVENT_AGENT_RETRY, move |ev: &AgentRetryEvent| {
            let reason = sanitize_diagnostic_text(&ev.reason, &retry_secrets.lock().unwrap());
            eprintln!(
                "[W retry] step={} attempt={}/{} wait={}ms reason={reason}",
                ev.step, ev.attempt, ev.max_attempts, ev.delay_ms
            );
        })
        .expect("active exec context");
    app.plugin(LlmRouter::plugin(), resolved_llm.router);
    if ext_flags.file_mentions {
        let mention_secrets = Arc::clone(&log_secrets);
        let mention_ws = cfg.workspace.clone();
        app.plugin(
            FileMentionsPlugin,
            FileMentionsConfig::with_callback(mention_ws, restrict_to_workspace, move |ev| {
                if let ClientEvent::Notice { level, text } = ev {
                    let tag = match level {
                        NoticeLevel::Info => "I",
                        NoticeLevel::Warn => "W",
                        NoticeLevel::Error => "E",
                    };
                    let guard = mention_secrets.lock().unwrap();
                    let clean = sanitize_diagnostic_text(&text, &guard);
                    eprintln!("[{tag} mentions] {clean}");
                }
            }),
        );
    }

    let cancel = CancellationToken::new();
    let timed_out = Arc::new(AtomicBool::new(false));
    let signal_code = Arc::new(AtomicI32::new(0));

    let signal_handle = spawn_signal_watcher(cancel.clone(), Arc::clone(&signal_code));
    let timeout_handle = cfg.timeout_secs.map(|secs| {
        let timeout_cancel = cancel.clone();
        let timeout_flag = Arc::clone(&timed_out);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(secs)).await;
            timeout_flag.store(true, Ordering::SeqCst);
            timeout_cancel.cancel();
        })
    });

    let init_res = tokio::select! {
        biased;
        () = cancel.cancelled() => Err("startup cancelled".to_string()),
        res = async {
            wait_for_services(&app).await.map_err(|e| e.to_string())?;
            if let Some(mcp) = app.get::<McpManager>("mcp") && mcp.needs_initial_catalog(include_tools.as_deref()) {
                mcp.wait_ready().await;
            }
            Ok::<(), String>(())
        } => res,
    };

    let complete =
        |outcome, session: Option<(&Session, bool, u64)>, report: Option<&TurnRunReport>| {
            finish(
                outcome,
                Some(&app),
                session,
                report,
                Some(active_provider.clone()),
                active_model.clone(),
            )
        };
    let interruption = || interrupted_exit(&timed_out, cfg.timeout_secs, &signal_code);
    let exit_code = if let Err(err_msg) = init_res {
        complete(
            interruption().unwrap_or_else(|| TurnExitOutcome::failure("error", err_msg)),
            None,
            None,
        )
    } else if has_sessions_dir && app.session_persistence().is_none() {
        complete(
            interruption().unwrap_or_else(|| {
                TurnExitOutcome::failure(
                    "storage",
                    "session persistence backend is unavailable (code=storage)",
                )
            }),
            None,
            None,
        )
    } else {
        match (app.sessions(), app.service::<AgentLoop>("agentLoop")) {
            (Ok(sessions), Ok(agent_loop)) => {
                let resolved_session = match &cfg.resume_session {
                    Some(ResumeTarget::Picker) => Err(
                        "--resume requires a session id in non-interactive exec mode (code=invalid_request)"
                            .to_string(),
                    ),
                    Some(ResumeTarget::Id(target_id)) => match sessions.get(target_id) {
                        Some(found) => Ok((found, true)),
                        None => Err(format!(
                            "session \"{target_id}\" not found (code=invalid_request)"
                        )),
                    },
                    Some(ResumeTarget::LatestInWorkspace) => {
                        let ws_str = cfg.workspace.display().to_string();
                        let latest_id = sessions
                            .list_headers(&SessionListQuery {
                                cwd: Some(ws_str),
                                limit: Some(1),
                            })
                            .ok()
                            .and_then(|mut list| list.drain(..).next())
                            .map(|h| h.id);
                        if let Some(id) = latest_id
                            && let Some(found) = sessions.get(&id)
                        {
                            Ok((found, true))
                        } else {
                            Ok((sessions.create(None), false))
                        }
                    }
                    None => Ok((sessions.create(None), false)),
                };

                match resolved_session {
                    Err(msg) => {
                        complete(TurnExitOutcome::failure("invalid_request", msg), None, None)
                    }
                    Ok((session, resumed)) => {
                        let turn_started = std::time::Instant::now();
                        if cfg.output_format == ExecOutputFormat::Json {
                            ndjson::subscribe(&app, &session.id, &stream, &log_secrets);
                            stream.lock().unwrap().emit(
                                serde_json::json!({
                                    "type": "session", "session_id": session.id, "resumed": resumed,
                                    "provider": active_provider, "model": active_model,
                                }),
                                now_ms(),
                            );
                        }
                        let opts = RunOptions {
                            provider: Some(active_provider.clone()),
                            model: active_model.clone(),
                            max_steps: configured_max_steps,
                            max_tokens: configured_max_tokens,
                            reasoning_effort: effective_effort,
                            ..Default::default()
                        };
                        let run_res = agent_loop
                            .run_with_report(Arc::clone(&session), cfg.prompt, cancel.clone(), opts)
                            .await;
                        let duration_ms = turn_started.elapsed().as_millis() as u64;

                        let flush_error = if has_sessions_dir {
                            sessions
                                .flush_async(Some(session.id.clone()))
                                .await
                                .err()
                                .map(|e| e.to_string())
                                .or_else(|| app.session_persistence().and_then(|b| b.last_error()))
                        } else {
                            None
                        };
                        let mut outcome = interruption().unwrap_or_else(|| {
                            classify_turn_result(&run_res, configured_max_steps, &log_secrets)
                        });
                        if let Some(err) = flush_error {
                            let msg = format!("failed to persist session: {err} (code=storage)");
                            if outcome.exit_code == 0 {
                                outcome = TurnExitOutcome::failure("storage", msg);
                            } else {
                                let clean =
                                    sanitize_diagnostic_text(&msg, &log_secrets.lock().unwrap());
                                eprintln!("[W storage] {clean}");
                            }
                        }
                        complete(
                            outcome,
                            Some((&session, resumed, duration_ms)),
                            run_res.as_ref().ok(),
                        )
                    }
                }
            }
            (Err(err), _) | (_, Err(err)) => complete(
                TurnExitOutcome::failure("error", err.to_string()),
                None,
                None,
            ),
        }
    };

    cleanup_exec(&app, signal_handle, timeout_handle).await;
    exit_code
}

struct TurnExitOutcome {
    exit_code: i32,
    termination_reason: &'static str,
    error: Option<String>,
}

fn export_exec_trace_best_effort(
    trace_out: Option<&Path>,
    app: Option<&Context>,
    session_id: Option<&str>,
    mut meta: TraceExportMeta,
    secrets: &Arc<Mutex<Vec<String>>>,
) {
    let Some(path) = trace_out else {
        return;
    };

    if let Some(ref err_msg) = meta.error {
        let guard = secrets.lock().unwrap();
        meta.error = Some(sanitize_diagnostic_text(err_msg, &guard));
    }

    let export = match (app, session_id) {
        (Some(ctx), Some(sid)) => {
            if let Some(traj) = ctx.get::<TrajectoryRuntime>("trajectory") {
                traj.build_trace_export(
                    sid,
                    TimelineMode::Sequence,
                    meta,
                    DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS,
                )
            } else {
                TrajectoryExport::empty(sid, meta)
            }
        }
        _ => TrajectoryExport::empty(session_id.unwrap_or(""), meta),
    };

    if let Err(err) = write_trace_export_atomic(path, &export) {
        let guard = secrets.lock().unwrap();
        let message = format!(
            "failed to write trajectory trace to \"{}\": {err}",
            path.display()
        );
        eprintln!("[W trace] {}", sanitize_diagnostic_text(&message, &guard));
    }
}

impl TurnExitOutcome {
    fn failure(reason: &'static str, message: impl Into<String>) -> Self {
        Self {
            exit_code: 1,
            termination_reason: reason,
            error: Some(message.into()),
        }
    }

    fn write_output(
        &self,
        format: ExecOutputFormat,
        report: Option<&TurnRunReport>,
        meta: Option<serde_json::Value>,
    ) -> io::Result<()> {
        let text = report
            .filter(|_| self.exit_code == 0)
            .map_or("", |r| r.text.as_str());
        if format == ExecOutputFormat::Json {
            let mut payload = meta.unwrap_or_else(|| serde_json::json!({}));
            payload["type"] = "result".into();
            payload["reason"] = self.termination_reason.into();
            payload["exit_code"] = self.exit_code.into();
            payload["text"] = text.into();
            if let Some(error) = &self.error {
                payload["error"] = error.as_str().into();
            }
            return write_event(payload, now_ms());
        }
        let mut out = io::stdout().lock();
        if !text.is_empty() {
            write!(out, "{text}")?;
            if !text.ends_with('\n') {
                writeln!(out)?;
            }
        }
        out.flush()
    }
}

pub fn report_exec_input_error(format: ExecOutputFormat, message: &str) {
    let msg = sanitize_diagnostic_text(message, &env_secrets());
    eprintln!("error: {msg}");
    if let Err(err) =
        TurnExitOutcome::failure("invalid_request", msg).write_output(format, None, None)
    {
        eprintln!(
            "error: failed to write stdout: {}",
            sanitize_diagnostic_text(&err.to_string(), &env_secrets())
        );
    }
}

fn interrupted_exit(
    timed_out: &AtomicBool,
    timeout_secs: Option<u64>,
    signal_code: &AtomicI32,
) -> Option<TurnExitOutcome> {
    if timed_out.load(Ordering::SeqCst) {
        let secs = timeout_secs.unwrap_or(0);
        return Some(TurnExitOutcome {
            exit_code: 124,
            termination_reason: "timeout",
            error: Some(format!("execution timed out after {secs}s (code=timeout)")),
        });
    }
    let sig = signal_code.load(Ordering::SeqCst);
    (sig != 0).then(|| TurnExitOutcome {
        exit_code: sig,
        termination_reason: "cancelled",
        error: Some("execution interrupted by signal (code=cancelled)".to_owned()),
    })
}

fn classify_turn_result(
    run_res: &Result<TurnRunReport, AgentError>,
    configured_max_steps: Option<usize>,
    secrets: &Mutex<Vec<String>>,
) -> TurnExitOutcome {
    match run_res {
        Ok(report) => match &report.reason {
            TurnEndReason::Completed => TurnExitOutcome {
                exit_code: 0,
                termination_reason: "completed",
                error: None,
            },
            TurnEndReason::MaxSteps => TurnExitOutcome::failure(
                "max_steps",
                format!(
                    "max steps exceeded (steps={}, max_steps={}, code=max_steps)",
                    report.steps,
                    configured_max_steps.unwrap_or(report.steps)
                ),
            ),
            TurnEndReason::Cancelled => {
                TurnExitOutcome::failure("cancelled", "execution cancelled (code=cancelled)")
            }
            TurnEndReason::Error { message } => TurnExitOutcome::failure("error", message),
            TurnEndReason::Truncated => TurnExitOutcome::failure(
                "truncated",
                "model output truncated: response exceeded model output token maximum (code=max_tokens)",
            ),
        },
        Err(AgentError::Llm(err)) if err.code == LlmErrorCode::Cancelled => {
            TurnExitOutcome::failure("cancelled", "execution cancelled (code=cancelled)")
        }
        Err(AgentError::Llm(err)) => TurnExitOutcome::failure(
            "error",
            format_llm_error_diagnostic(err, &secrets.lock().unwrap()),
        ),
        Err(err) => TurnExitOutcome::failure("error", err.to_string()),
    }
}

async fn cleanup_exec(
    app: &Context,
    signal_handle: tokio::task::JoinHandle<()>,
    timeout_handle: Option<tokio::task::JoinHandle<()>>,
) {
    signal_handle.abort();
    if let Some(handle) = timeout_handle {
        handle.abort();
    }
    let _ = tokio::time::timeout(Duration::from_secs(3), app.fiber().dispose()).await;
}

fn spawn_signal_watcher(
    cancel: CancellationToken,
    signal_code: Arc<AtomicI32>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let mut sigterm = signal(SignalKind::terminate()).ok();
            let mut sigint = signal(SignalKind::interrupt()).ok();
            tokio::select! {
                _ = async {
                    if let Some(ref mut s) = sigterm {
                        let _ = s.recv().await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    signal_code.store(143, Ordering::SeqCst);
                    cancel.cancel();
                }
                _ = async {
                    if let Some(ref mut s) = sigint {
                        let _ = s.recv().await;
                    } else {
                        let _ = tokio::signal::ctrl_c().await;
                    }
                } => {
                    signal_code.store(130, Ordering::SeqCst);
                    cancel.cancel();
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            signal_code.store(130, Ordering::SeqCst);
            cancel.cancel();
        }
    })
}
