//! 非交互一次性任务执行器（`mizar exec` / `mizar -p` / `mizar --prompt-file`）。
//!
//! 职责：
//! - 复用 `AgentSpinePlugin` + `LlmRouter` + `LlmRetryPlugin` 装配 Agent，但不启动 TUI 与版本检查，也不注入 TUI `reply_guidance` 段落；
//! - 支持 `--isolated` 隔离个人全局配置 / MCP / Skills / 指令与全局会话库；
//! - 将诊断与警告严格路由至 `stderr` 并完成敏感凭据脱敏，退出码仅在正常收敛（`Stop` 且未截断/未超步/未取消）时为 `0`；
//! - `stdout` 仅在退出码为 `0` 时输出最终回答全文，失败时 `stdout` 保持完全为空。

use std::io::{IsTerminal, Read};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cordis::{Context, LogLevel, Service};
use mizar_agent_loop::{AgentError, AgentLoop, RunOptions, StepOutcome, TurnEndReason};
use mizar_agent_spine::{
    AgentSpineConfig, AgentSpinePlugin, AgentToolsMode, CliLlmOverride, EnvLlmOverride,
    load_mizar_config, load_mizar_config_isolated, resolve_default_session_db_path,
};
use mizar_llm::{LlmError, LlmErrorCode, ReasoningEffort};
use mizar_llm_retry::{LlmRetryPlugin, RetryConfig};
use mizar_llm_router::LlmRouter;
use mizar_mcp::McpManager;
use mizar_session::{SessionContextExt, SessionListQuery, SessionPersistenceContextExt};
use mizar_tools_coding::CodingToolsPreset;
use mizar_trajectory::{TimelineMode, TrajectoryRuntime};
use tokio_util::sync::CancellationToken;

use crate::runner::{ResumeTarget, wait_for_services};

/// 无头输出格式。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecOutputFormat {
    #[default]
    Text,
    Json,
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
    let replaced = text.replace(['\r', '\n'], " ");
    let mut out = String::with_capacity(replaced.len());
    let mut prev_space = false;
    for ch in replaced.chars() {
        if ch.is_whitespace() {
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
        .filter(|s| {
            let trimmed = s.trim();
            trimmed.len() >= 6 || (trimmed.starts_with("sk-") && trimmed.len() >= 4)
        })
        .collect();
    deduped.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    deduped.dedup();
    for secret in deduped {
        out = out.replace(secret, "[REDACTED]");
    }

    out = redact_header_tokens(&out);
    out = redact_bearer_tokens(&out);
    out = redact_query_secrets(&out);
    redact_prefixed_tokens(&out)
}

fn is_token_delimiter(b: u8) -> bool {
    b.is_ascii_whitespace()
        || matches!(
            b,
            b'"' | b'\'' | b',' | b';' | b')' | b']' | b'}' | b'<' | b'>'
        )
}

fn redact_header_tokens(input: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let prefixes = ["authorization:", "x-api-key:", "api-key:"];
    let mut result = String::with_capacity(input.len());
    let mut cursor = 0usize;

    while cursor < input.len() {
        let mut next_match: Option<(usize, &str)> = None;
        for &pat in &prefixes {
            if let Some(rel) = lower[cursor..].find(pat) {
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

        let boundary_ok = pos == 0 || !input.as_bytes()[pos - 1].is_ascii_alphanumeric();
        if !boundary_ok {
            let advance = pos + pat.len();
            result.push_str(&input[cursor..advance]);
            cursor = advance;
            continue;
        }

        let after_pat = pos + pat.len();
        result.push_str(&input[cursor..after_pat]);
        let mut val_start = after_pat;
        while val_start < input.len() && input.as_bytes()[val_start] == b' ' {
            result.push(' ');
            val_start += 1;
        }
        if pat == "authorization:" && lower[val_start..].starts_with("bearer ") {
            let b_len = "bearer ".len();
            result.push_str(&input[val_start..val_start + b_len]);
            val_start += b_len;
            while val_start < input.len() && input.as_bytes()[val_start] == b' ' {
                result.push(' ');
                val_start += 1;
            }
        }
        let mut val_end = val_start;
        let bytes = input.as_bytes();
        while val_end < bytes.len() && !is_token_delimiter(bytes[val_end]) {
            val_end += 1;
        }
        if val_end > val_start {
            let tok = &input[val_start..val_end];
            if tok != "[REDACTED]" {
                result.push_str("[REDACTED]");
            } else {
                result.push_str(tok);
            }
        }
        cursor = val_end;
    }

    result
}

fn looks_like_bearer_secret(tok: &str) -> bool {
    if tok == "[REDACTED]" || tok.len() < 6 {
        return false;
    }
    let lower = tok.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "token"
            | "auth"
            | "authentication"
            | "header"
            | "headers"
            | "scheme"
            | "missing"
            | "invalid"
            | "expired"
            | "format"
            | "required"
            | "provided"
    ) {
        return false;
    }
    tok.len() >= 12
        || tok.starts_with("sk-")
        || tok.starts_with("rk-")
        || tok.starts_with("ghp_")
        || tok
            .bytes()
            .any(|b| b.is_ascii_digit() || b == b'-' || b == b'_' || b == b'.')
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
        if looks_like_bearer_secret(tok) {
            result.push_str("[REDACTED]");
        } else {
            result.push_str(tok);
        }
        cursor = tok_end;
    }
    result.push_str(&input[cursor..]);
    result
}

fn redact_query_secrets(input: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let keys = [
        "key=",
        "api_key=",
        "apikey=",
        "token=",
        "access_token=",
        "secret=",
    ];
    let mut result = String::with_capacity(input.len());
    let mut cursor = 0usize;

    while cursor < input.len() {
        let mut next_match: Option<(usize, &str)> = None;
        for &pat in &keys {
            if let Some(rel) = lower[cursor..].find(pat) {
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
        let is_query_param = pos > 0 && matches!(input.as_bytes()[pos - 1], b'?' | b'&');
        let val_start = pos + pat.len();
        if !is_query_param {
            result.push_str(&input[cursor..val_start]);
            cursor = val_start;
            continue;
        }
        result.push_str(&input[cursor..val_start]);
        let mut val_end = val_start;
        let bytes = input.as_bytes();
        while val_end < bytes.len() && bytes[val_end] != b'&' && !is_token_delimiter(bytes[val_end])
        {
            val_end += 1;
        }
        if val_end > val_start {
            result.push_str("[REDACTED]");
        }
        cursor = val_end;
    }
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
    let sanitized_msg = single_line(&sanitize_diagnostic_text(&err.message, secrets));
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
            let current_secrets = secrets.lock().unwrap().clone();
            let clean = single_line(&sanitize_diagnostic_text(&rec.message, &current_secrets));
            eprintln!("[{} {}] {}", rec.tag, rec.name, clean);
        }
    });
}

fn report_error_with_secrets(msg: &str, secrets: &Arc<Mutex<Vec<String>>>) {
    let guard = secrets.lock().unwrap();
    let clean = single_line(&sanitize_diagnostic_text(msg, &guard));
    eprintln!("error: {clean}");
}

/// 运行无头一次性任务，返回进程应使用的退出码。
pub async fn run_exec_cli(cfg: ExecConfig) -> i32 {
    if cfg.approve {
        eprintln!(
            "error: --approve is incompatible with non-interactive exec mode (code=invalid_request)"
        );
        return 1;
    }
    if let Some(ref target) = cfg.resume_session {
        if matches!(target, ResumeTarget::Picker) {
            eprintln!(
                "error: --resume requires a session id in non-interactive exec mode (code=invalid_request)"
            );
            return 1;
        }
        if cfg.no_persist || cfg.isolated {
            eprintln!(
                "error: --continue/--resume is incompatible with --no-persist/--isolated (code=invalid_request)"
            );
            return 1;
        }
    }

    let env_override = EnvLlmOverride::from_env();
    let log_secrets = Arc::new(Mutex::new(env_override.collect_secrets()));

    let loaded_cfg = if cfg.isolated {
        load_mizar_config_isolated(&cfg.workspace)
    } else {
        load_mizar_config(&cfg.workspace)
    };

    if let Some(ui) = &loaded_cfg.settings.ui
        && let Some(ref mode) = ui.approval_mode
        && mode.trim().eq_ignore_ascii_case("ask")
    {
        eprintln!(
            "error: ui.approvalMode=\"ask\" is incompatible with non-interactive exec mode (code=invalid_request)"
        );
        return 1;
    }

    for parse_err in &loaded_cfg.errors {
        let guard = log_secrets.lock().unwrap();
        let clean = single_line(&sanitize_diagnostic_text(parse_err, &guard));
        eprintln!("[W config] 配置解析失败：{clean}");
    }

    let cli_override = CliLlmOverride {
        provider: cfg.provider.clone(),
        model: cfg.model.clone(),
        protocol: cfg.protocol.clone(),
        base_url: cfg.base_url.clone(),
        reasoning_effort: cfg.reasoning_effort,
    };
    let resolved_llm = match loaded_cfg.resolve_llm_router_with_cli(env_override, cli_override) {
        Ok(resolved) => resolved,
        Err(err) => {
            report_error_with_secrets(&err.to_string(), &log_secrets);
            return 1;
        }
    };

    {
        let mut s = log_secrets.lock().unwrap();
        for route in resolved_llm.router.providers.values() {
            if let Some(ref key) = route.api_key
                && !key.trim().is_empty()
                && !s.contains(key)
            {
                s.push(key.clone());
            }
        }
    }

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
            let guard = log_secrets.lock().unwrap();
            eprintln!("error: {}", format_llm_error_diagnostic(&err, &guard));
            return 1;
        }
    }

    let mut spine_cfg = AgentSpineConfig::coding(cfg.workspace.clone());
    if !cfg.isolated && !cfg.no_persist {
        spine_cfg.session_db_path = resolve_default_session_db_path();
    }
    loaded_cfg.apply_to_spine(&mut spine_cfg);
    if !spine_cfg
        .disabled_plugins
        .iter()
        .any(|id| id == "session-title")
    {
        spine_cfg.disabled_plugins.push("session-title".to_string());
    }
    if cfg.no_persist || (cfg.isolated && loaded_cfg.settings.session_db.is_none()) {
        spine_cfg.session_db_path = None;
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
    let has_session_db = spine_cfg.session_db_path.is_some();

    if cfg.resume_session.is_some() && !has_session_db {
        eprintln!(
            "error: --continue/--resume requires session persistence database configuration (code=invalid_request)"
        );
        return 1;
    }

    let app = Context::new();
    configure_exec_logger(&app, Arc::clone(&log_secrets));

    app.plugin(AgentSpinePlugin, spine_cfg);
    app.plugin(LlmRetryPlugin, RetryConfig::default());
    app.plugin(LlmRouter::plugin(), resolved_llm.router);

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
            if has_session_db && app.session_persistence().is_none() {
                for _ in 0..25 {
                    if app.session_persistence().is_some() {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            }
            if let Some(mcp) = app.get::<McpManager>("mcp") {
                mcp.wait_ready().await;
            }
            Ok::<(), String>(())
        } => res,
    };

    if let Err(err_msg) = init_res {
        let code = resolve_cancel_or_error_exit(
            &timed_out,
            cfg.timeout_secs,
            &signal_code,
            &err_msg,
            &log_secrets,
        );
        cleanup_exec(&app, signal_handle, timeout_handle).await;
        return code;
    }

    let exit_code = match (app.sessions(), app.service::<AgentLoop>("agentLoop")) {
        (Ok(sessions), Ok(agent_loop)) => {
            if cfg.resume_session.is_some() && app.session_persistence().is_none() {
                eprintln!(
                    "error: session persistence backend is unavailable for --continue/--resume (code=storage)"
                );
                cleanup_exec(&app, signal_handle, timeout_handle).await;
                return 1;
            }

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

            let (session, resumed) = match resolved_session {
                Ok(pair) => pair,
                Err(msg) => {
                    eprintln!("error: {msg}");
                    cleanup_exec(&app, signal_handle, timeout_handle).await;
                    return 1;
                }
            };

            let turn_started = std::time::Instant::now();
            let opts = RunOptions {
                provider: Some(resolved_llm.active_provider),
                model: resolved_llm.active_model,
                max_steps: configured_max_steps,
                max_tokens: configured_max_tokens,
                reasoning_effort: effective_effort,
                ..Default::default()
            };
            let run_res = agent_loop
                .run_with_report(Arc::clone(&session), cfg.prompt, cancel.clone(), opts)
                .await;
            let duration_ms = turn_started.elapsed().as_millis() as u64;

            if has_session_db {
                let _ = sessions.flush_async(Some(session.id.clone())).await;
            }

            let json_meta = if cfg.output_format == ExecOutputFormat::Json {
                Some(build_exec_json_meta(
                    &app,
                    &session.id,
                    resumed,
                    duration_ms,
                ))
            } else {
                None
            };

            if timed_out.load(Ordering::SeqCst) {
                let secs = cfg.timeout_secs.unwrap_or(0);
                eprintln!("error: execution timed out after {secs}s (code=timeout)");
                124
            } else {
                let sig = signal_code.load(Ordering::SeqCst);
                if sig != 0 {
                    eprintln!("error: execution interrupted by signal (code=cancelled)");
                    sig
                } else {
                    classify_turn_result(
                        run_res,
                        configured_max_steps,
                        cfg.output_format,
                        json_meta.as_ref(),
                        &log_secrets,
                    )
                }
            }
        }
        (Err(e), _) | (_, Err(e)) => {
            report_error_with_secrets(&e.to_string(), &log_secrets);
            1
        }
    };

    cleanup_exec(&app, signal_handle, timeout_handle).await;
    exit_code
}

struct ExecJsonMeta {
    session_id: String,
    resumed: bool,
    duration_ms: u64,
    turn_llm_ms: u64,
    turn_ttft_ms: Option<u64>,
    turn_tool_calls: usize,
    session_turns: u64,
    session_steps: u64,
    session_tool_ms: u64,
}

fn build_exec_json_meta(
    app: &Context,
    session_id: &str,
    resumed: bool,
    duration_ms: u64,
) -> ExecJsonMeta {
    let report = app
        .get::<TrajectoryRuntime>("trajectory")
        .and_then(|traj| traj.inspect(session_id, TimelineMode::Sequence));
    if let Some(rep) = report {
        let last_turn = rep.snapshot.turns.last();
        let last_turn_num = last_turn.map(|t| t.turn);
        let turn_llm_ms = last_turn
            .map(|t| t.steps.iter().filter_map(|s| s.duration_ms).sum())
            .unwrap_or(rep.snapshot.stats.llm_ms);
        let turn_ttft_ms = last_turn.and_then(|t| t.steps.first().and_then(|s| s.ttft_ms));
        let turn_tool_calls = last_turn_num
            .map(|tnum| {
                rep.snapshot
                    .records
                    .iter()
                    .filter(|r| {
                        r.kind == mizar_trajectory::RecordKind::Tool && r.turn == Some(tnum)
                    })
                    .count()
            })
            .unwrap_or(rep.snapshot.stats.tool_calls as usize);
        ExecJsonMeta {
            session_id: session_id.to_string(),
            resumed,
            duration_ms,
            turn_llm_ms,
            turn_ttft_ms,
            turn_tool_calls,
            session_turns: rep.snapshot.stats.turns,
            session_steps: rep.snapshot.stats.steps,
            session_tool_ms: rep.snapshot.stats.tool_ms,
        }
    } else {
        ExecJsonMeta {
            session_id: session_id.to_string(),
            resumed,
            duration_ms,
            turn_llm_ms: 0,
            turn_ttft_ms: None,
            turn_tool_calls: 0,
            session_turns: 0,
            session_steps: 0,
            session_tool_ms: 0,
        }
    }
}

fn resolve_cancel_or_error_exit(
    timed_out: &AtomicBool,
    timeout_secs: Option<u64>,
    signal_code: &AtomicI32,
    fallback_msg: &str,
    secrets: &Arc<Mutex<Vec<String>>>,
) -> i32 {
    if timed_out.load(Ordering::SeqCst) {
        let secs = timeout_secs.unwrap_or(0);
        eprintln!("error: execution timed out after {secs}s (code=timeout)");
        return 124;
    }
    let sig = signal_code.load(Ordering::SeqCst);
    if sig != 0 {
        eprintln!("error: execution interrupted by signal (code=cancelled)");
        return sig;
    }
    report_error_with_secrets(fallback_msg, secrets);
    1
}

fn classify_turn_result(
    run_res: Result<mizar_agent_loop::TurnRunReport, AgentError>,
    configured_max_steps: Option<usize>,
    output_format: ExecOutputFormat,
    json_meta: Option<&ExecJsonMeta>,
    secrets: &Arc<Mutex<Vec<String>>>,
) -> i32 {
    match run_res {
        Ok(report) => {
            if report.reason == TurnEndReason::Truncated
                || matches!(report.last_step_outcome, Some(StepOutcome::MaxTokens))
            {
                eprintln!(
                    "error: model output truncated: response exceeded model output token maximum (code=max_tokens)"
                );
                return 1;
            }
            match report.reason {
                TurnEndReason::Completed => {
                    match output_format {
                        ExecOutputFormat::Text => {
                            if !report.text.is_empty() {
                                if report.text.ends_with('\n') {
                                    print!("{}", report.text);
                                } else {
                                    println!("{}", report.text);
                                }
                            }
                        }
                        ExecOutputFormat::Json => {
                            let payload = if let Some(meta) = json_meta {
                                serde_json::json!({
                                    "session_id": meta.session_id,
                                    "resumed": meta.resumed,
                                    "reason": "completed",
                                    "steps": report.steps,
                                    "duration_ms": meta.duration_ms,
                                    "llm_ms": meta.turn_llm_ms,
                                    "ttft_ms": meta.turn_ttft_ms,
                                    "tool_ms": meta.session_tool_ms,
                                    "tool_calls": meta.turn_tool_calls,
                                    "session_turns": meta.session_turns,
                                    "session_steps": meta.session_steps,
                                    "text": report.text,
                                })
                            } else {
                                serde_json::json!({
                                    "reason": "completed",
                                    "steps": report.steps,
                                    "text": report.text,
                                })
                            };
                            println!("{payload}");
                        }
                    }
                    0
                }
                TurnEndReason::MaxSteps => {
                    let limit = configured_max_steps.unwrap_or(report.steps);
                    eprintln!(
                        "error: max steps exceeded (steps={}, max_steps={limit}, code=max_steps)",
                        report.steps
                    );
                    1
                }
                TurnEndReason::Cancelled => {
                    eprintln!("error: execution cancelled (code=cancelled)");
                    1
                }
                TurnEndReason::Error { message } => {
                    report_error_with_secrets(&message, secrets);
                    1
                }
                TurnEndReason::Truncated => 1,
            }
        }
        Err(AgentError::Llm(llm_err)) => {
            if llm_err.code == LlmErrorCode::Cancelled {
                eprintln!("error: execution cancelled (code=cancelled)");
                1
            } else {
                let guard = secrets.lock().unwrap();
                eprintln!("error: {}", format_llm_error_diagnostic(&llm_err, &guard));
                1
            }
        }
        Err(err) => {
            report_error_with_secrets(&err.to_string(), secrets);
            1
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_explicit_secrets_headers_bearer_and_query_params() {
        let secrets = vec!["my-secret-api-key-999".to_string(), "sk-test".to_string()];
        let raw = "Authorization: Bearer my-secret-api-key-999, x-api-key: another-key-val, https://api.example.com/v1?api_key=qsecret&foo=bar, Bearer sk-proj-1234567890, invalid_api_key sk-test";
        let sanitized = sanitize_diagnostic_text(raw, &secrets);
        assert!(!sanitized.contains("my-secret-api-key-999"), "{sanitized}");
        assert!(!sanitized.contains("another-key-val"), "{sanitized}");
        assert!(!sanitized.contains("qsecret"), "{sanitized}");
        assert!(!sanitized.contains("sk-proj-1234567890"), "{sanitized}");
        assert!(!sanitized.contains("sk-test"), "{sanitized}");
        assert!(sanitized.contains("invalid_api_key"), "{sanitized}");
        assert!(sanitized.contains("foo=bar"), "{sanitized}");
        assert!(sanitized.contains("[REDACTED]"), "{sanitized}");

        // 不误伤普通英文错误短语中的 bearer token 描述
        let english = sanitize_diagnostic_text("Invalid bearer token provided", &[]);
        assert_eq!(english, "Invalid bearer token provided");
    }

    #[test]
    fn formats_harbor_compatible_diagnostics_for_rate_limit_context_and_auth() {
        let mut rate =
            LlmError::classify_http(429, Some(2500), r#"{"error":{"message":"slow down"}}"#);
        rate.retry_after_ms = Some(2500);
        let rate_diag = format_llm_error_diagnostic(&rate, &[]);
        assert!(rate_diag.contains("status=429"), "{rate_diag}");
        assert!(rate_diag.contains("retry_after_ms=2500"), "{rate_diag}");
        assert!(rate_diag.contains("code=rate_limit"), "{rate_diag}");

        let ctx = LlmError::classify_http(
            400,
            None,
            r#"{"error":{"message":"This model's maximum context length is 8192 tokens","code":"context_length_exceeded"}}"#,
        );
        let ctx_diag = format_llm_error_diagnostic(&ctx, &[]);
        assert!(
            ctx_diag.ends_with("detail=input token count exceeds the maximum number of tokens)"),
            "{ctx_diag}"
        );

        let auth = LlmError::classify_http(
            401,
            None,
            r#"{"error":{"message":"Invalid key sk-secret-12345"}}"#,
        );
        let auth_diag = format_llm_error_diagnostic(&auth, &["sk-secret-12345".to_string()]);
        assert!(!auth_diag.contains("sk-secret-12345"), "{auth_diag}");
        assert!(
            auth_diag.contains("unexpected status 401 Unauthorized"),
            "{auth_diag}"
        );
    }

    #[test]
    fn resolves_prompt_from_flag_file_or_positional_and_rejects_conflicts_or_blanks() {
        assert_eq!(
            resolve_exec_prompt(true, Some("hi".into()), None, None).unwrap(),
            "hi"
        );
        assert!(resolve_exec_prompt(true, Some("   \n".into()), None, None).is_err());
        assert!(resolve_exec_prompt(true, Some("a".into()), None, Some("b".into())).is_err());

        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("mizar_prompt_file_{}_{uniq}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("task.txt");
        std::fs::write(&file, "solve issue 18\n").unwrap();
        assert_eq!(
            resolve_exec_prompt(false, None, Some(file.clone()), None).unwrap(),
            "solve issue 18\n"
        );
        assert!(resolve_exec_prompt(true, Some("x".into()), Some(file), None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
