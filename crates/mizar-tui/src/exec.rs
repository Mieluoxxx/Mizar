//! 非交互一次性任务执行器（`mizar exec` / `mizar -p` / `mizar --prompt-file`）。
//!
//! 职责：
//! - 复用 `AgentSpinePlugin` + `LlmRouter` + `LlmRetryPlugin` 装配 Agent，但不启动 TUI 与版本检查，也不注入 TUI `reply_guidance` 段落；
//! - 支持 `--isolated` 隔离个人全局配置 / MCP / Skills / 指令与全局会话库；
//! - 将诊断与警告严格路由至 `stderr` 并完成敏感凭据脱敏，退出码仅在正常收敛（`Stop` 且未截断/未超步/未取消）时为 `0`；
//! - `stdout` 仅在退出码为 `0` 时输出最终回答全文，失败时 `stdout` 保持完全为空。

use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cordis::{Context, LogLevel, Service};
use mizar_agent_loop::{AgentError, AgentLoop, RunOptions, StepOutcome, TurnEndReason};
use mizar_agent_spine::{
    AgentSpineConfig, AgentSpinePlugin, AgentToolsMode, CliLlmOverride, EnvLlmOverride,
    load_mizar_config, load_mizar_config_isolated, resolve_default_session_db_path,
};
use mizar_client::{FileMentionsConfig, FileMentionsPlugin};
use mizar_client_core::{ClientEvent, HostExtensionFlags, NoticeLevel};
use mizar_llm::{LlmError, LlmErrorCode, ReasoningEffort};
use mizar_llm_retry::{LlmRetryPlugin, RetryConfig};
use mizar_llm_router::LlmRouter;
use mizar_mcp::McpManager;
use mizar_session::{SessionContextExt, SessionListQuery, SessionPersistenceContextExt};
use mizar_tools_coding::CodingToolsPreset;
use mizar_trajectory::{
    DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS, TimelineMode, TraceExportMeta, TrajectoryExport,
    TrajectoryRuntime, write_trace_export_atomic,
};
use tokio_util::sync::CancellationToken;

use crate::runner::{ResumeTarget, wait_for_services};

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

/// 一次性无头 `exec` 模式下关闭 `subagent`（并行加速与上下文减负统一交由 `codemode` 承担）。
pub(crate) fn disable_exec_subagent(spine_cfg: &mut AgentSpineConfig) {
    spine_cfg.subagent = None;
}

/// 运行无头一次性任务，返回进程应使用的退出码。
pub async fn run_exec_cli(cfg: ExecConfig) -> i32 {
    let exec_started = std::time::Instant::now();
    let env_override = EnvLlmOverride::from_env();
    let log_secrets = Arc::new(Mutex::new(env_override.collect_secrets()));
    let make_meta = |exit_code: i32,
                     termination_reason: &str,
                     error: Option<String>,
                     provider: Option<String>,
                     model: Option<String>| TraceExportMeta {
        exit_code,
        termination_reason: termination_reason.to_string(),
        error,
        duration_ms: exec_started.elapsed().as_millis() as u64,
        provider,
        model,
        isolated: cfg.isolated,
        output_format: Some(cfg.output_format.as_str().to_string()),
    };

    if cfg.approve {
        let msg = "--approve is incompatible with non-interactive exec mode (code=invalid_request)";
        eprintln!("error: {msg}");
        export_exec_trace_best_effort(
            cfg.trace_out.as_deref(),
            None,
            None,
            make_meta(
                1,
                "invalid_request",
                Some(msg.to_string()),
                cfg.provider.clone(),
                cfg.model.clone(),
            ),
            &log_secrets,
        );
        return 1;
    }
    if let Some(ref target) = cfg.resume_session {
        if matches!(target, ResumeTarget::Picker) {
            let msg = "--resume requires a session id in non-interactive exec mode (code=invalid_request)";
            eprintln!("error: {msg}");
            export_exec_trace_best_effort(
                cfg.trace_out.as_deref(),
                None,
                None,
                make_meta(
                    1,
                    "invalid_request",
                    Some(msg.to_string()),
                    cfg.provider.clone(),
                    cfg.model.clone(),
                ),
                &log_secrets,
            );
            return 1;
        }
        if cfg.no_persist || cfg.isolated {
            let msg = "--continue/--resume is incompatible with --no-persist/--isolated (code=invalid_request)";
            eprintln!("error: {msg}");
            export_exec_trace_best_effort(
                cfg.trace_out.as_deref(),
                None,
                None,
                make_meta(
                    1,
                    "invalid_request",
                    Some(msg.to_string()),
                    cfg.provider.clone(),
                    cfg.model.clone(),
                ),
                &log_secrets,
            );
            return 1;
        }
    }

    let loaded_cfg = if cfg.isolated {
        load_mizar_config_isolated(&cfg.workspace)
    } else {
        load_mizar_config(&cfg.workspace)
    };

    if let Some(ui) = &loaded_cfg.settings.ui
        && let Some(ref mode) = ui.approval_mode
        && mode.trim().eq_ignore_ascii_case("ask")
    {
        let msg = "ui.approvalMode=\"ask\" is incompatible with non-interactive exec mode (code=invalid_request)";
        eprintln!("error: {msg}");
        export_exec_trace_best_effort(
            cfg.trace_out.as_deref(),
            None,
            None,
            make_meta(
                1,
                "invalid_request",
                Some(msg.to_string()),
                cfg.provider.clone(),
                cfg.model.clone(),
            ),
            &log_secrets,
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
            let err_msg = err.to_string();
            report_error_with_secrets(&err_msg, &log_secrets);
            export_exec_trace_best_effort(
                cfg.trace_out.as_deref(),
                None,
                None,
                make_meta(
                    1,
                    "error",
                    Some(err_msg),
                    cfg.provider.clone(),
                    cfg.model.clone(),
                ),
                &log_secrets,
            );
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
            let diag = {
                let guard = log_secrets.lock().unwrap();
                format_llm_error_diagnostic(&err, &guard)
            };
            eprintln!("error: {diag}");
            export_exec_trace_best_effort(
                cfg.trace_out.as_deref(),
                None,
                None,
                make_meta(
                    1,
                    "error",
                    Some(diag),
                    Some(resolved_llm.active_provider.clone()),
                    resolved_llm.active_model.clone(),
                ),
                &log_secrets,
            );
            return 1;
        }
    }

    let mut spine_cfg = AgentSpineConfig::coding(cfg.workspace.clone());
    if !cfg.isolated && !cfg.no_persist {
        spine_cfg.session_db_path = resolve_default_session_db_path();
    }
    loaded_cfg.apply_to_spine(&mut spine_cfg);
    // 一次性无头 exec 由 codemode 承担确定性并行与上下文减负，关闭异步交互分身插件
    disable_exec_subagent(&mut spine_cfg);
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
        let msg = "--continue/--resume requires session persistence database configuration (code=invalid_request)";
        eprintln!("error: {msg}");
        export_exec_trace_best_effort(
            cfg.trace_out.as_deref(),
            None,
            None,
            make_meta(
                1,
                "invalid_request",
                Some(msg.to_string()),
                Some(resolved_llm.active_provider.clone()),
                resolved_llm.active_model.clone(),
            ),
            &log_secrets,
        );
        return 1;
    }

    let active_provider = resolved_llm.active_provider.clone();
    let active_model = resolved_llm.active_model.clone();
    let ext_flags = HostExtensionFlags::from_disabled_plugins(&spine_cfg.disabled_plugins);
    let restrict_to_workspace = loaded_cfg.settings.restrict_to_workspace.unwrap_or(false);

    let app = Context::new();
    configure_exec_logger(&app, Arc::clone(&log_secrets));

    app.plugin(AgentSpinePlugin, spine_cfg);
    app.plugin(LlmRetryPlugin, RetryConfig::default());
    app.plugin(LlmRouter::plugin(), resolved_llm.router);
    let mention_fiber = if ext_flags.file_mentions {
        let mention_logger = app.logger_named("mentions");
        Some(app.plugin(
            FileMentionsPlugin,
            FileMentionsConfig::with_callback(
                cfg.workspace.clone(),
                restrict_to_workspace,
                move |ev| {
                    if let ClientEvent::Notice { level, text } = ev {
                        match level {
                            NoticeLevel::Error => mention_logger.error(text),
                            NoticeLevel::Warn => mention_logger.warn(text),
                            NoticeLevel::Info => mention_logger.info(text),
                        }
                    }
                },
            ),
        ))
    } else {
        None
    };

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
            if let Some(f) = mention_fiber {
                f.wait_settled().await.map_err(|e| e.to_string())?;
            }
            if has_session_db && app.session_persistence().is_none() {
                let fibers: Vec<cordis::Fiber> = app
                    .registry()
                    .values()
                    .into_iter()
                    .flat_map(|rt| rt.fibers())
                    .collect();
                for f in fibers {
                    let _ = f.wait_settled().await;
                }
                if app.session_persistence().is_none() {
                    let has_pending = app.registry().values().into_iter().any(|rt| {
                        rt.fibers().into_iter().any(|f| {
                            matches!(
                                f.state(),
                                cordis::FiberState::Pending | cordis::FiberState::Loading
                            )
                        })
                    });
                    if has_pending {
                        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
                        let tx = Arc::new(Mutex::new(Some(tx)));
                        if let Ok(probe) = app.try_inject(
                            [mizar_session::SERVICE_SESSION_PERSISTENCE],
                            move |_| {
                                let tx = Arc::clone(&tx);
                                async move {
                                    if let Some(sender) = tx.lock().unwrap().take() {
                                        let _ = sender.send(());
                                    }
                                    Ok::<_, cordis::CordisError>(())
                                }
                            },
                        ) {
                            let _ = tokio::time::timeout(Duration::from_secs(5), rx).await;
                            probe.dispose().await;
                        }
                    }
                }
            }
            if let Some(mcp) = app.get::<McpManager>("mcp") {
                mcp.wait_ready().await;
            }
            Ok::<(), String>(())
        } => res,
    };

    if let Err(err_msg) = init_res {
        let outcome = resolve_cancel_or_error_exit(
            &timed_out,
            cfg.timeout_secs,
            &signal_code,
            &err_msg,
            &log_secrets,
        );
        export_exec_trace_best_effort(
            cfg.trace_out.as_deref(),
            Some(&app),
            None,
            make_meta(
                outcome.exit_code,
                outcome.termination_reason,
                outcome.error,
                Some(active_provider.clone()),
                active_model.clone(),
            ),
            &log_secrets,
        );
        cleanup_exec(&app, signal_handle, timeout_handle).await;
        return outcome.exit_code;
    }

    let exit_code = match (app.sessions(), app.service::<AgentLoop>("agentLoop")) {
        (Ok(sessions), Ok(agent_loop)) => {
            if cfg.resume_session.is_some() && app.session_persistence().is_none() {
                let msg = "session persistence backend is unavailable for --continue/--resume (code=storage)";
                eprintln!("error: {msg}");
                export_exec_trace_best_effort(
                    cfg.trace_out.as_deref(),
                    Some(&app),
                    None,
                    make_meta(
                        1,
                        "storage",
                        Some(msg.to_string()),
                        Some(active_provider.clone()),
                        active_model.clone(),
                    ),
                    &log_secrets,
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
                            exclude_subagents: true,
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
                    export_exec_trace_best_effort(
                        cfg.trace_out.as_deref(),
                        Some(&app),
                        None,
                        make_meta(
                            1,
                            "invalid_request",
                            Some(msg),
                            Some(active_provider.clone()),
                            active_model.clone(),
                        ),
                        &log_secrets,
                    );
                    cleanup_exec(&app, signal_handle, timeout_handle).await;
                    return 1;
                }
            };

            let turn_started = std::time::Instant::now();
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

            let outcome = if timed_out.load(Ordering::SeqCst) {
                let secs = cfg.timeout_secs.unwrap_or(0);
                let msg = format!("execution timed out after {secs}s (code=timeout)");
                eprintln!("error: {msg}");
                TurnExitOutcome {
                    exit_code: 124,
                    termination_reason: "timeout",
                    error: Some(msg),
                }
            } else {
                let sig = signal_code.load(Ordering::SeqCst);
                if sig != 0 {
                    let msg = "execution interrupted by signal (code=cancelled)".to_string();
                    eprintln!("error: {msg}");
                    TurnExitOutcome {
                        exit_code: sig,
                        termination_reason: "cancelled",
                        error: Some(msg),
                    }
                } else {
                    classify_turn_result(
                        run_res,
                        configured_max_steps,
                        cfg.output_format,
                        json_meta.as_ref(),
                        &log_secrets,
                    )
                }
            };

            export_exec_trace_best_effort(
                cfg.trace_out.as_deref(),
                Some(&app),
                Some(&session.id),
                make_meta(
                    outcome.exit_code,
                    outcome.termination_reason,
                    outcome.error,
                    Some(active_provider.clone()),
                    active_model.clone(),
                ),
                &log_secrets,
            );
            outcome.exit_code
        }
        (Err(e), _) | (_, Err(e)) => {
            let msg = e.to_string();
            report_error_with_secrets(&msg, &log_secrets);
            export_exec_trace_best_effort(
                cfg.trace_out.as_deref(),
                Some(&app),
                None,
                make_meta(1, "error", Some(msg), Some(active_provider), active_model),
                &log_secrets,
            );
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
        meta.error = Some(single_line(&sanitize_diagnostic_text(err_msg, &guard)));
    }

    let export = match (app, session_id) {
        (Some(ctx), Some(sid)) => {
            if let Ok(traj) = ctx.get::<TrajectoryRuntime>("trajectory").ok_or(()) {
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
        let clean = single_line(&sanitize_diagnostic_text(&err.to_string(), &guard));
        eprintln!(
            "[W trace] failed to write trajectory trace to \"{}\": {clean}",
            path.display()
        );
    }
}

fn resolve_cancel_or_error_exit(
    timed_out: &AtomicBool,
    timeout_secs: Option<u64>,
    signal_code: &AtomicI32,
    fallback_msg: &str,
    secrets: &Arc<Mutex<Vec<String>>>,
) -> TurnExitOutcome {
    if timed_out.load(Ordering::SeqCst) {
        let secs = timeout_secs.unwrap_or(0);
        let msg = format!("execution timed out after {secs}s (code=timeout)");
        eprintln!("error: {msg}");
        return TurnExitOutcome {
            exit_code: 124,
            termination_reason: "timeout",
            error: Some(msg),
        };
    }
    let sig = signal_code.load(Ordering::SeqCst);
    if sig != 0 {
        let msg = "execution interrupted by signal (code=cancelled)".to_string();
        eprintln!("error: {msg}");
        return TurnExitOutcome {
            exit_code: sig,
            termination_reason: "cancelled",
            error: Some(msg),
        };
    }
    report_error_with_secrets(fallback_msg, secrets);
    TurnExitOutcome {
        exit_code: 1,
        termination_reason: "error",
        error: Some(fallback_msg.to_string()),
    }
}

fn classify_turn_result(
    run_res: Result<mizar_agent_loop::TurnRunReport, AgentError>,
    configured_max_steps: Option<usize>,
    output_format: ExecOutputFormat,
    json_meta: Option<&ExecJsonMeta>,
    secrets: &Arc<Mutex<Vec<String>>>,
) -> TurnExitOutcome {
    match run_res {
        Ok(report) => {
            if report.reason == TurnEndReason::Truncated
                || matches!(report.last_step_outcome, Some(StepOutcome::MaxTokens))
            {
                let msg = "model output truncated: response exceeded model output token maximum (code=max_tokens)";
                eprintln!("error: {msg}");
                return TurnExitOutcome {
                    exit_code: 1,
                    termination_reason: "truncated",
                    error: Some(msg.to_string()),
                };
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
                    TurnExitOutcome {
                        exit_code: 0,
                        termination_reason: "completed",
                        error: None,
                    }
                }
                TurnEndReason::MaxSteps => {
                    let limit = configured_max_steps.unwrap_or(report.steps);
                    let msg = format!(
                        "max steps exceeded (steps={}, max_steps={limit}, code=max_steps)",
                        report.steps
                    );
                    eprintln!("error: {msg}");
                    TurnExitOutcome {
                        exit_code: 1,
                        termination_reason: "max_steps",
                        error: Some(msg),
                    }
                }
                TurnEndReason::Cancelled => {
                    let msg = "execution cancelled (code=cancelled)";
                    eprintln!("error: {msg}");
                    TurnExitOutcome {
                        exit_code: 1,
                        termination_reason: "cancelled",
                        error: Some(msg.to_string()),
                    }
                }
                TurnEndReason::Error { message } => {
                    report_error_with_secrets(&message, secrets);
                    TurnExitOutcome {
                        exit_code: 1,
                        termination_reason: "error",
                        error: Some(message),
                    }
                }
                TurnEndReason::Truncated => TurnExitOutcome {
                    exit_code: 1,
                    termination_reason: "truncated",
                    error: Some("model output truncated (code=max_tokens)".to_string()),
                },
            }
        }
        Err(AgentError::Llm(llm_err)) => {
            if llm_err.code == LlmErrorCode::Cancelled {
                let msg = "execution cancelled (code=cancelled)";
                eprintln!("error: {msg}");
                TurnExitOutcome {
                    exit_code: 1,
                    termination_reason: "cancelled",
                    error: Some(msg.to_string()),
                }
            } else {
                let diag = {
                    let guard = secrets.lock().unwrap();
                    format_llm_error_diagnostic(&llm_err, &guard)
                };
                eprintln!("error: {diag}");
                TurnExitOutcome {
                    exit_code: 1,
                    termination_reason: "error",
                    error: Some(diag),
                }
            }
        }
        Err(err) => {
            let msg = err.to_string();
            report_error_with_secrets(&msg, secrets);
            TurnExitOutcome {
                exit_code: 1,
                termination_reason: "error",
                error: Some(msg),
            }
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

    #[test]
    fn exec_mode_disables_subagent_and_leaves_session_title_off() {
        let mut spine = AgentSpineConfig::coding(PathBuf::from("/tmp"));
        assert!(spine.session_title.is_none());
        assert!(spine.subagent.is_some());
        disable_exec_subagent(&mut spine);
        assert!(spine.subagent.is_none());
    }

    #[test]
    fn exec_trace_out_prefers_cli_over_env_and_sanitizes_errors() {
        assert_eq!(
            resolve_trace_out_path(
                Some(PathBuf::from("cli/trace.json")),
                Some("env/trace.json".to_string())
            ),
            Some(PathBuf::from("cli/trace.json"))
        );
        assert_eq!(
            resolve_trace_out_path(None, Some("env/trace.json".to_string())),
            Some(PathBuf::from("env/trace.json"))
        );
        assert_eq!(resolve_trace_out_path(None, Some("   ".to_string())), None);
        assert_eq!(resolve_trace_out_path(None, None), None);

        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("mizar_exec_trace_{}_{uniq}", std::process::id()));
        let trace_path = dir.join("out/trajectory.json");
        let secrets = Arc::new(Mutex::new(vec!["sk-unit-secret-999".to_string()]));
        export_exec_trace_best_effort(
            Some(&trace_path),
            None,
            None,
            TraceExportMeta {
                exit_code: 1,
                termination_reason: "error".to_string(),
                error: Some("bad token sk-unit-secret-999".to_string()),
                duration_ms: 15,
                provider: Some("openai".to_string()),
                model: Some("gpt-4o".to_string()),
                isolated: true,
                output_format: Some("text".to_string()),
            },
            &secrets,
        );
        let raw = std::fs::read_to_string(&trace_path).unwrap();
        assert!(!raw.contains("sk-unit-secret-999"), "{raw}");
        assert!(raw.contains("[REDACTED]"), "{raw}");
        let parsed: TrajectoryExport = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed.exit_code, 1);
        assert_eq!(parsed.termination_reason, "error");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
