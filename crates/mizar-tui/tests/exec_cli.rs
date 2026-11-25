//! `mizar exec` 无头模式真实子进程集成测试（基于本地 `axum` HTTP 网关，不连接外网）。

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::post;
use mizar_trajectory::TrajectoryExport;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Notify;

static UNIQUE_SEQ: AtomicU64 = AtomicU64::new(0);

fn mizar_bin() -> &'static str {
    env!("CARGO_BIN_EXE_mizar")
}

struct TempRoot {
    path: PathBuf,
}

impl TempRoot {
    fn new(prefix: &str) -> Self {
        let seq = UNIQUE_SEQ.fetch_add(1, Ordering::Relaxed);
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mizar_{prefix}_{}_{}_{}",
            std::process::id(),
            uniq,
            seq
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[derive(Clone, Debug)]
struct RecordedRequest {
    path: String,
    authorization: Option<String>,
    x_api_key: Option<String>,
    session_id: Option<String>,
    x_client_request_id: Option<String>,
    x_session_id: Option<String>,
    body: Value,
}

#[derive(Clone)]
enum MockStep {
    CompletionsText(String),
    CompletionsTruncated(String),
    CompletionsReasoningOnly(String),
    CompletionsToolCall {
        id: String,
        name: String,
        args: Value,
    },
    CompletionsToolCallWithUsage {
        content_prefix: Option<String>,
        id: String,
        name: String,
        args: Value,
        usage: Value,
    },
    CompletionsTextWithUsage {
        text: String,
        usage: Value,
    },
    ResponsesText(String),
    MessagesText(String),
    HttpError {
        status: u16,
        retry_after: Option<String>,
        retry_after_ms: Option<String>,
        body: Value,
    },
    HangForever,
}

struct GatewayState {
    script: Mutex<Vec<MockStep>>,
    fallback: MockStep,
    requests: Mutex<Vec<RecordedRequest>>,
    first_req_notify: Arc<Notify>,
}

impl GatewayState {
    fn new(steps: Vec<MockStep>, fallback: MockStep) -> Arc<Self> {
        let mut reversed = steps;
        reversed.reverse();
        Arc::new(Self {
            script: Mutex::new(reversed),
            fallback,
            requests: Mutex::new(Vec::new()),
            first_req_notify: Arc::new(Notify::new()),
        })
    }

    fn set_script(&self, steps: Vec<MockStep>) {
        let mut reversed = steps;
        reversed.reverse();
        *self.script.lock().unwrap() = reversed;
    }

    fn next_step(&self) -> MockStep {
        let mut guard = self.script.lock().unwrap();
        guard.pop().unwrap_or_else(|| self.fallback.clone())
    }
}

async fn start_gateway(state: Arc<GatewayState>) -> String {
    async fn handle_req(
        axum::extract::State(st): axum::extract::State<Arc<GatewayState>>,
        uri: axum::http::Uri,
        headers: HeaderMap,
        axum::Json(body): axum::Json<Value>,
    ) -> Response {
        let header_str = |name: &str| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(ToString::to_string)
        };
        let rec = RecordedRequest {
            path: uri.path().to_string(),
            authorization: header_str("authorization"),
            x_api_key: header_str("x-api-key"),
            session_id: header_str("session_id"),
            x_client_request_id: header_str("x-client-request-id"),
            x_session_id: header_str("x-session-id"),
            body,
        };
        st.requests.lock().unwrap().push(rec);
        st.first_req_notify.notify_waiters();

        match st.next_step() {
            MockStep::CompletionsText(text) => {
                let delta1 = json!({"choices":[{"delta":{"content": text}}]}).to_string();
                let finish = json!({"choices":[{"delta":{},"finish_reason":"stop"}]}).to_string();
                let sse = format!("data: {delta1}\n\ndata: {finish}\n\ndata: [DONE]\n\n");
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }
            MockStep::CompletionsTruncated(text) => {
                let delta1 = json!({"choices":[{"delta":{"content": text}}]}).to_string();
                let finish = json!({"choices":[{"delta":{},"finish_reason":"length"}]}).to_string();
                let sse = format!("data: {delta1}\n\ndata: {finish}\n\ndata: [DONE]\n\n");
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }
            MockStep::CompletionsReasoningOnly(reasoning) => {
                let delta1 =
                    json!({"choices":[{"delta":{"reasoning_content": reasoning}}]}).to_string();
                let finish = json!({"choices":[{"delta":{},"finish_reason":"stop"}]}).to_string();
                let sse = format!("data: {delta1}\n\ndata: {finish}\n\ndata: [DONE]\n\n");
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }
            MockStep::CompletionsToolCall { id, name, args } => {
                let tc = json!({
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": args.to_string(),
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                })
                .to_string();
                let sse = format!("data: {tc}\n\ndata: [DONE]\n\n");
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }
            MockStep::CompletionsToolCallWithUsage {
                content_prefix,
                id,
                name,
                args,
                usage,
            } => {
                let mut sse = String::new();
                if let Some(prefix) = content_prefix {
                    let delta = json!({"choices":[{"delta":{"content": prefix}}]}).to_string();
                    sse.push_str(&format!("data: {delta}\n\n"));
                }
                let tc = json!({
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": args.to_string(),
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                })
                .to_string();
                let u = json!({"choices": [], "usage": usage}).to_string();
                sse.push_str(&format!("data: {tc}\n\ndata: {u}\n\ndata: [DONE]\n\n"));
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }
            MockStep::CompletionsTextWithUsage { text, usage } => {
                let delta1 = json!({"choices":[{"delta":{"content": text}}]}).to_string();
                let finish = json!({"choices":[{"delta":{},"finish_reason":"stop"}]}).to_string();
                let u = json!({"choices": [], "usage": usage}).to_string();
                let sse =
                    format!("data: {delta1}\n\ndata: {finish}\n\ndata: {u}\n\ndata: [DONE]\n\n");
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }
            MockStep::ResponsesText(text) => {
                let d = json!({"type":"response.output_text.delta","delta": text}).to_string();
                let done =
                    json!({"type":"response.completed","response":{"id":"resp_test"}}).to_string();
                let sse = format!("data: {d}\n\ndata: {done}\n\n");
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }
            MockStep::MessagesText(text) => {
                let ev1 = json!({"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":10,"output_tokens":0}}}).to_string();
                let ev2 = json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).to_string();
                let ev3 = json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text": text}}).to_string();
                let ev4 = json!({"type":"content_block_stop","index":0}).to_string();
                let ev5 = json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}).to_string();
                let ev6 = json!({"type":"message_stop"}).to_string();
                let sse = format!(
                    "event: message_start\ndata: {ev1}\n\n\
                     event: content_block_start\ndata: {ev2}\n\n\
                     event: content_block_delta\ndata: {ev3}\n\n\
                     event: content_block_stop\ndata: {ev4}\n\n\
                     event: message_delta\ndata: {ev5}\n\n\
                     event: message_stop\ndata: {ev6}\n\n"
                );
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }
            MockStep::HttpError {
                status,
                retry_after,
                retry_after_ms,
                body,
            } => {
                let mut builder = Response::builder()
                    .status(StatusCode::from_u16(status).unwrap())
                    .header("content-type", "application/json");
                if let Some(ra) = retry_after {
                    builder = builder.header("retry-after", ra);
                }
                if let Some(rams) = retry_after_ms {
                    builder = builder.header("retry-after-ms", rams);
                }
                builder.body(Body::from(body.to_string())).unwrap()
            }
            MockStep::HangForever => {
                tokio::time::sleep(Duration::from_secs(30)).await;
                Response::builder()
                    .status(StatusCode::REQUEST_TIMEOUT)
                    .body(Body::empty())
                    .unwrap()
            }
        }
    }

    let app = Router::new()
        .route("/chat/completions", post(handle_req))
        .route("/v1/chat/completions", post(handle_req))
        .route("/responses", post(handle_req))
        .route("/v1/responses", post(handle_req))
        .route("/v1/messages", post(handle_req))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}")
}

fn clean_env_cmd(home: &Path, ws: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::new(mizar_bin());
    cmd.env_clear();
    if let Some(path_val) = std::env::var_os("PATH") {
        cmd.env("PATH", path_val);
    }
    cmd.env("HOME", home);
    cmd.env("MIZAR_HOME", home.join(".mizar"));
    cmd.env("MIZAR_AGENTS_HOME", home.join(".agents"));
    cmd.env("MIZAR_SKIP_VERSION_CHECK", "1");
    cmd.env("MIZAR_OFFLINE", "1");
    cmd.current_dir(ws);
    cmd.args(args);
    cmd
}

async fn run_with_stdin(cmd: Command, stdin_data: Option<&str>) -> Output {
    let mut tcmd = tokio::process::Command::from(cmd);
    tcmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    if stdin_data.is_some() {
        tcmd.stdin(Stdio::piped());
    } else {
        tcmd.stdin(Stdio::null());
    }
    let mut child = tcmd.spawn().unwrap();
    if let Some(data) = stdin_data {
        use tokio::io::AsyncWriteExt;
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(data.as_bytes())
            .await
            .unwrap();
        let _ = child.stdin.take();
    }
    tokio::time::timeout(Duration::from_secs(20), child.wait_with_output())
        .await
        .expect("mizar subprocess timed out after 20s")
        .unwrap()
}

/// 完全复刻 `harbor-framework/harbor` `BaseInstalledAgent._classify_exec_error` 的精确 LastMatchWins 分类逻辑，
/// 用以在 CI 无 harbor Python 包时精准验证 `stdout + "\n" + stderr` 的错误归类结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
enum HarborErrorKind {
    AgentAuthenticationError,
    ApiRateLimitError,
    ApiUsageLimitError,
    ContextWindowExceededError,
    OutputTokenExceededError,
}

fn record_span(best: &mut Option<(usize, HarborErrorKind)>, end: usize, kind: HarborErrorKind) {
    if best.is_none_or(|(b, _)| end > b) {
        *best = Some((end, kind));
    }
}

fn classify_harbor_error(stdout: &str, stderr: &str) -> Option<HarborErrorKind> {
    let combined = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    let mut best: Option<(usize, HarborErrorKind)> = None;
    let record_needle =
        |best: &mut Option<(usize, HarborErrorKind)>, needle: &str, kind: HarborErrorKind| {
            for (idx, _) in combined.match_indices(needle) {
                record_span(best, idx + needle.len(), kind);
            }
        };

    // rate.?limit / too many requests -> ApiRateLimitError
    for (idx, _) in combined.match_indices("rate") {
        let rest = &combined[idx + 4..];
        if rest.starts_with("limit") {
            record_span(&mut best, idx + 9, HarborErrorKind::ApiRateLimitError);
        } else if let Some(ch) = rest.chars().next()
            && rest[ch.len_utf8()..].starts_with("limit")
        {
            record_span(
                &mut best,
                idx + 4 + ch.len_utf8() + 5,
                HarborErrorKind::ApiRateLimitError,
            );
        }
    }
    record_needle(
        &mut best,
        "too many requests",
        HarborErrorKind::ApiRateLimitError,
    );

    // ContextWindowExceededError
    record_needle(
        &mut best,
        "input token count exceeds the maximum number of tokens",
        HarborErrorKind::ContextWindowExceededError,
    );

    // OutputTokenExceededError: response exceeded .+ output token maximum
    for (idx, _) in combined.match_indices("response exceeded ") {
        let after = &combined[idx + "response exceeded ".len()..];
        if let Some(pos) = after.find(" output token maximum") {
            let candidate = &after[..pos];
            if !candidate.is_empty() && !candidate.contains('\n') {
                let end = idx + "response exceeded ".len() + pos + " output token maximum".len();
                record_span(&mut best, end, HarborErrorKind::OutputTokenExceededError);
            }
        }
    }

    // ApiUsageLimitError
    record_needle(
        &mut best,
        "quota exceeded.",
        HarborErrorKind::ApiUsageLimitError,
    );

    // AgentAuthenticationError
    record_needle(
        &mut best,
        "not logged in",
        HarborErrorKind::AgentAuthenticationError,
    );
    record_needle(
        &mut best,
        "unexpected status 401 unauthorized",
        HarborErrorKind::AgentAuthenticationError,
    );
    record_needle(
        &mut best,
        "incorrect api key provided",
        HarborErrorKind::AgentAuthenticationError,
    );
    record_needle(
        &mut best,
        "invalid api key",
        HarborErrorKind::AgentAuthenticationError,
    );
    record_needle(
        &mut best,
        "api key is invalid",
        HarborErrorKind::AgentAuthenticationError,
    );

    best.map(|(_, kind)| kind)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_supports_arg_flag_file_and_piped_stdin_and_keeps_stdout_clean() {
    let tmp = TempRoot::new("exec_inputs");
    let home = tmp.path().join("home");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&ws).unwrap();

    let state = GatewayState::new(
        vec![
            MockStep::CompletionsText("ANS_POS\n".into()),
            MockStep::CompletionsText("ANS_FLAG".into()),
            MockStep::CompletionsText("ANS_FILE".into()),
            MockStep::CompletionsText("ANS_PIPE".into()),
            MockStep::CompletionsText("ANS_NO_CONCAT".into()),
        ],
        MockStep::CompletionsText("FALLBACK".into()),
    );
    let url = start_gateway(Arc::clone(&state)).await;
    let base_v1 = format!("{url}/v1");

    // 1. 位置参数 mizar exec <prompt>
    let mut c1 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--no-persist",
            "--protocol",
            "completions",
            "--base-url",
            &base_v1,
            "--model",
            "eval-m",
            "task from pos",
        ],
    );
    c1.env("API_KEY", "sk-test-123456");
    let o1 = run_with_stdin(c1, None).await;
    assert!(
        o1.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&o1.stderr)
    );
    let out1 = String::from_utf8_lossy(&o1.stdout);
    assert_eq!(out1, "ANS_POS\n");
    assert!(!out1.contains('\x1b'));

    // 2. root -p <text>
    let mut c2 = clean_env_cmd(
        &home,
        &ws,
        &[
            "-p",
            "task from flag",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "eval-m",
        ],
    );
    c2.env("API_KEY", "sk-test-123456");
    let o2 = run_with_stdin(c2, None).await;
    assert!(o2.status.success());
    assert_eq!(String::from_utf8_lossy(&o2.stdout), "ANS_FLAG\n");

    // 3. --prompt-file <path>
    let prompt_file = tmp.path().join("task.md");
    std::fs::write(&prompt_file, "task from file\n").unwrap();
    let mut c3 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--prompt-file",
            prompt_file.to_str().unwrap(),
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "eval-m",
        ],
    );
    c3.env("API_KEY", "sk-test-123456");
    let o3 = run_with_stdin(c3, None).await;
    assert!(o3.status.success());
    assert_eq!(String::from_utf8_lossy(&o3.stdout), "ANS_FILE\n");

    // 4. piped stdin（无 explicit prompt 时读至 EOF）
    let mut c4 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "eval-m",
        ],
    );
    c4.env("API_KEY", "sk-test-123456");
    let o4 = run_with_stdin(c4, Some("task from pipe\n")).await;
    assert!(o4.status.success());
    assert_eq!(String::from_utf8_lossy(&o4.stdout), "ANS_PIPE\n");

    // 5. 传入显式 prompt 时，决不把管道 stdin 隐式拼接进来
    let mut c5 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "eval-m",
            "-p",
            "explicit only",
        ],
    );
    c5.env("API_KEY", "sk-test-123456");
    let o5 = run_with_stdin(c5, Some("SHOULD_NOT_BE_INCLUDED")).await;
    assert!(o5.status.success());
    assert_eq!(String::from_utf8_lossy(&o5.stdout), "ANS_NO_CONCAT\n");

    // 验证后端网关实际收到的请求内容与 system prompt（不得含 reply_guidance）
    let reqs = state.requests.lock().unwrap().clone();
    assert_eq!(reqs.len(), 5);
    assert!(reqs[0].body.to_string().contains("task from pos"));
    assert!(reqs[1].body.to_string().contains("task from flag"));
    assert!(reqs[2].body.to_string().contains("task from file"));
    assert!(reqs[3].body.to_string().contains("task from pipe"));
    let req5_str = reqs[4].body.to_string();
    assert!(req5_str.contains("explicit only"));
    assert!(!req5_str.contains("SHOULD_NOT_BE_INCLUDED"));
    for r in &reqs {
        let raw = r.body.to_string();
        assert!(
            !raw.contains("reply_guidance") && !raw.contains("pi-emphasis"),
            "exec mode must not inject TUI emphasis prompt: {raw}"
        );
    }

    // 6. 空白、冲突、缺失文件参数必须在发起网络请求前失败且 stdout 为空
    let bad_invocations: &[(&[&str], Option<&str>)] = &[
        (&["exec", "   \n"], None),
        (&["exec", "-p", "   "], None),
        (&["exec", "-p", "a", "b"], None),
        (
            &[
                "exec",
                "-p",
                "a",
                "--prompt-file",
                prompt_file.to_str().unwrap(),
            ],
            None,
        ),
        (
            &["exec", "--prompt-file", "/nonexistent/mizar_prompt.txt"],
            None,
        ),
        (&["exec", "one", "two"], None),
    ];
    for (args, pipe) in bad_invocations {
        let mut bad = clean_env_cmd(&home, &ws, args);
        bad.env("API_KEY", "sk-test-123456");
        let out = run_with_stdin(bad, *pipe).await;
        assert!(
            !out.status.success(),
            "expected failure for args {args:?}, got exit 0"
        );
        assert!(
            out.stdout.is_empty(),
            "expected empty stdout on failure for {args:?}, got: {}",
            String::from_utf8_lossy(&out.stdout)
        );
    }

    // 7. `exec` 在 stdin 为 TTY 且未提供任何 prompt 时立即非零失败（不挂起等待用户输入）
    let tty_out = Command::new("python3")
        .arg("-c")
        .arg(
            "import os, pty, select, subprocess, sys\n\
             master, slave = pty.openpty()\n\
             p = subprocess.Popen([sys.argv[1], 'exec'], stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n\
             os.close(slave)\n\
             out, err = p.communicate(timeout=5)\n\
             os.close(master)\n\
             assert p.returncode != 0, f'expected non-zero exit, got {p.returncode}'\n\
             assert len(out) == 0, f'expected empty stdout, got {out!r}'\n\
             assert len(err) > 0, 'expected stderr diagnostic'\n",
        )
        .arg(mizar_bin())
        .output()
        .unwrap();
    assert!(
        tty_out.status.success(),
        "TTY missing prompt check failed: {}",
        String::from_utf8_lossy(&tty_out.stderr)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_supports_completions_responses_messages_api_format_and_reasoning_effort() {
    let tmp = TempRoot::new("exec_protocols");
    let home = tmp.path().join("home");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(home.join(".mizar")).unwrap();
    std::fs::create_dir_all(ws.join(".mizar")).unwrap();

    // 在个人全局配置中放入低优先级默认值，验证 Env 与 CLI 优先级压过它
    std::fs::write(
        home.join(".mizar/models.json"),
        json!({
            "providers": {
                "gw": {
                    "api": "openai-completions",
                    "baseUrl": "http://127.0.0.1:9/stale",
                    "apiKey": "sk-config-secret-key",
                    "model": "cfg-model"
                }
            }
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        home.join(".mizar/settings.json"),
        r#"{"defaultProvider":"gw","defaultModel":"cfg-model","reasoningEffort":"low"}"#,
    )
    .unwrap();

    let state = GatewayState::new(
        vec![
            MockStep::CompletionsText("OK_COMP".into()),
            MockStep::ResponsesText("OK_RESP".into()),
            MockStep::MessagesText("OK_MSG".into()),
        ],
        MockStep::CompletionsText("UNUSED".into()),
    );
    let url = start_gateway(Arc::clone(&state)).await;
    let base_v1 = format!("{url}/v1");

    // 1. CLI > Env > Config（无需新建 models.json；--protocol completions + --reasoning-effort high）
    let mut c1 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--no-persist",
            "--provider",
            "gw",
            "--protocol",
            "completions",
            "--base-url",
            &base_v1,
            "--model",
            "cli-model-1",
            "--reasoning-effort",
            "high",
            "hello comp",
        ],
    );
    c1.env("API_KEY", "sk-env-override-111");
    c1.env("MODEL", "env-model-ignored");
    c1.env("PROTOCOL", "messages");
    let o1 = run_with_stdin(c1, None).await;
    assert!(
        o1.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&o1.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&o1.stdout), "OK_COMP\n");

    // 2. --protocol responses（无 models.json + --isolated）
    let mut c2 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--protocol",
            "responses",
            "--base-url",
            &base_v1,
            "--model",
            "resp-model-2",
            "--reasoning-effort",
            "medium",
            "hello resp",
        ],
    );
    c2.env("OPENAI_API_KEY", "sk-oai-222222");
    let o2 = run_with_stdin(c2, None).await;
    assert!(
        o2.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&o2.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&o2.stdout), "OK_RESP\n");

    // 3. --api-format messages（别名测试 + ANTHROPIC_API_KEY + 无 models.json）
    let mut c3 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--api-format=messages",
            "--base-url",
            &url,
            "--model",
            "claude-eval-3",
            "--reasoning-effort",
            "high",
            "hello msg",
        ],
    );
    c3.env("ANTHROPIC_API_KEY", "sk-ant-333333");
    let o3 = run_with_stdin(c3, None).await;
    assert!(
        o3.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&o3.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&o3.stdout), "OK_MSG\n");

    let reqs = state.requests.lock().unwrap().clone();
    assert_eq!(reqs.len(), 3);

    assert!(
        reqs[0].path.ends_with("/chat/completions"),
        "{}",
        reqs[0].path
    );
    assert_eq!(
        reqs[0].authorization.as_deref(),
        Some("Bearer sk-env-override-111")
    );
    assert_eq!(reqs[0].body["model"], "cli-model-1");
    assert_eq!(reqs[0].body["reasoning_effort"], "high");

    assert!(reqs[1].path.ends_with("/responses"), "{}", reqs[1].path);
    assert_eq!(
        reqs[1].authorization.as_deref(),
        Some("Bearer sk-oai-222222")
    );
    assert_eq!(reqs[1].body["model"], "resp-model-2");
    assert_eq!(reqs[1].body["reasoning"]["effort"], "medium");

    assert!(reqs[2].path.ends_with("/v1/messages"), "{}", reqs[2].path);
    assert_eq!(reqs[2].x_api_key.as_deref(), Some("sk-ant-333333"));
    assert_eq!(reqs[2].body["model"], "claude-eval-3");
    assert_eq!(reqs[2].body["thinking"]["type"], "enabled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_runs_auto_tool_loop_and_rejects_interactive_approval_modes() {
    let tmp = TempRoot::new("exec_tools");
    let home = tmp.path().join("home");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&ws).unwrap();

    let target_file = ws.join("out.txt");
    let state = GatewayState::new(
        vec![
            MockStep::CompletionsToolCall {
                id: "tc_1".into(),
                name: "write".into(),
                args: json!({
                    "path": "out.txt",
                    "content": "created_without_tty"
                }),
            },
            MockStep::CompletionsText("TOOL_LOOP_DONE".into()),
        ],
        MockStep::CompletionsText("UNUSED".into()),
    );
    let url = start_gateway(Arc::clone(&state)).await;
    let base_v1 = format!("{url}/v1");

    // 1. --approve 在非交互模式直接失败且不触网
    let mut cmd_approve = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--approve",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "write file",
        ],
    );
    cmd_approve.env("API_KEY", "sk-tool-secret-1");
    let out_approve = run_with_stdin(cmd_approve, None).await;
    assert!(!out_approve.status.success());
    assert!(out_approve.stdout.is_empty());
    assert!(state.requests.lock().unwrap().is_empty());

    // 2. settings.json ui.approvalMode="ask" 在无头模式同样直接失败且不静默自动批准
    std::fs::create_dir_all(ws.join(".mizar")).unwrap();
    std::fs::write(
        ws.join(".mizar/settings.json"),
        r#"{"ui":{"approvalMode":"ask"}}"#,
    )
    .unwrap();
    let mut cmd_ask = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "write file",
        ],
    );
    cmd_ask.env("API_KEY", "sk-tool-secret-1");
    let out_ask = run_with_stdin(cmd_ask, None).await;
    assert!(!out_ask.status.success());
    assert!(out_ask.stdout.is_empty());
    assert!(state.requests.lock().unwrap().is_empty());

    // 清理 ask 配置，运行自动工具执行环
    std::fs::write(ws.join(".mizar/settings.json"), "{}").unwrap();
    let mut cmd_run = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "write out.txt and confirm",
        ],
    );
    cmd_run.env("API_KEY", "sk-tool-secret-1");
    let out_run = run_with_stdin(cmd_run, None).await;
    assert!(
        out_run.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out_run.stdout), "TOOL_LOOP_DONE\n");
    assert_eq!(
        std::fs::read_to_string(&target_file).unwrap(),
        "created_without_tty"
    );
    assert_eq!(state.requests.lock().unwrap().len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_isolated_ignores_personal_configs_mcp_skills_instructions_and_sqlite() {
    let tmp = TempRoot::new("exec_isolated");
    let home = tmp.path().join("home");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(home.join(".mizar")).unwrap();
    std::fs::create_dir_all(home.join(".agents/skills/personal-skill")).unwrap();
    std::fs::create_dir_all(ws.join(".mizar")).unwrap();
    std::fs::create_dir_all(ws.join(".agents/skills/workspace-skill")).unwrap();

    let personal_mcp_marker = tmp.path().join("personal_mcp_spawned.marker");
    let custom_session_db = tmp.path().join("custom_global_sessions.sqlite");
    let default_session_db = home.join(".mizar/sessions.sqlite");

    // 个人全局污染配置（带 approvalMode=ask、坏模型路由、个人 MCP、个人 Skill、个人 AGENTS.md）
    std::fs::write(
        home.join(".mizar/settings.json"),
        r#"{"defaultProvider":"personal_bad","defaultModel":"bad-m","ui":{"approvalMode":"ask"}}"#,
    )
    .unwrap();
    std::fs::write(
        home.join(".mizar/models.json"),
        r#"{"providers":{"personal_bad":{"api":"openai-completions","baseUrl":"http://127.0.0.1:9/dead","apiKey":"sk-personal","model":"bad-m"}}}"#,
    )
    .unwrap();
    std::fs::write(
        home.join(".mizar/mcp.json"),
        json!({
            "mcpServers": {
                "personal_evil": {
                    "command": "python3",
                    "args": ["-c", format!("import pathlib; pathlib.Path({:?}).write_text('ran')", personal_mcp_marker.display().to_string())],
                    "exposure": "direct"
                }
            }
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        home.join(".agents/skills/personal-skill/SKILL.md"),
        "---\nname: personal-skill\ndescription: PERSONAL_SKILL_MARKER\n---\nPERSONAL_SKILL_BODY",
    )
    .unwrap();
    std::fs::write(
        home.join(".mizar/AGENTS.md"),
        "PERSONAL_GLOBAL_INSTRUCTION_MARKER",
    )
    .unwrap();

    // 工作区配置仍应生效
    let state = GatewayState::new(
        vec![MockStep::CompletionsText("ISO_OK".into())],
        MockStep::CompletionsText("ISO_OK".into()),
    );
    let url = start_gateway(Arc::clone(&state)).await;
    std::fs::write(
        ws.join(".mizar/models.json"),
        json!({
            "providers": {
                "ws_gw": {
                    "api": "openai-completions",
                    "baseUrl": format!("{url}/v1"),
                    "apiKey": "sk-ws-isolated-123",
                    "model": "ws-model"
                }
            }
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        ws.join(".mizar/settings.json"),
        r#"{"defaultProvider":"ws_gw","defaultModel":"ws-model"}"#,
    )
    .unwrap();
    std::fs::write(
        ws.join(".agents/skills/workspace-skill/SKILL.md"),
        "---\nname: workspace-skill\ndescription: WORKSPACE_SKILL_MARKER\n---\nWS_BODY",
    )
    .unwrap();
    std::fs::write(ws.join("AGENTS.md"), "WORKSPACE_INSTRUCTION_MARKER").unwrap();

    let mut cmd_iso = clean_env_cmd(&home, &ws, &["exec", "--isolated", "verify isolation"]);
    cmd_iso.env("MIZAR_SESSION_DB", &custom_session_db);
    let out_iso = run_with_stdin(cmd_iso, None).await;
    assert!(
        out_iso.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_iso.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out_iso.stdout), "ISO_OK\n");

    assert!(
        !personal_mcp_marker.exists(),
        "--isolated must never spawn personal mcp.json server"
    );
    assert!(
        !default_session_db.exists() && !custom_session_db.exists(),
        "--isolated must never write to global or $MIZAR_SESSION_DB sqlite"
    );

    let reqs = state.requests.lock().unwrap().clone();
    assert_eq!(reqs.len(), 1);
    let req_dump = reqs[0].body.to_string();
    assert!(
        req_dump.contains("WORKSPACE_INSTRUCTION_MARKER"),
        "{req_dump}"
    );
    assert!(req_dump.contains("WORKSPACE_SKILL_MARKER"), "{req_dump}");
    assert!(
        !req_dump.contains("PERSONAL_GLOBAL_INSTRUCTION_MARKER"),
        "{req_dump}"
    );
    assert!(!req_dump.contains("PERSONAL_SKILL_MARKER"), "{req_dump}");

    // 验证 --no-persist 在非 --isolated 模式下独立阻止落盘 SQLite
    let mut cmd_nopersist = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--no-persist",
            "--provider",
            "ws_gw",
            "--model",
            "ws-model",
            "verify no persist",
        ],
    );
    cmd_nopersist.env("MIZAR_SESSION_DB", &custom_session_db);
    // 先移走会阻断运行或干扰测试的个人 settings/mcp
    std::fs::remove_file(home.join(".mizar/settings.json")).unwrap();
    std::fs::remove_file(home.join(".mizar/mcp.json")).unwrap();
    let out_np = run_with_stdin(cmd_nopersist, None).await;
    assert!(
        out_np.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_np.stderr)
    );
    assert!(
        !default_session_db.exists() && !custom_session_db.exists(),
        "--no-persist alone must also prevent creating sqlite DB"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_failure_modes_exit_codes_sanitization_and_harbor_classification() {
    let tmp = TempRoot::new("exec_failures");
    let home = tmp.path().join("home");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&ws).unwrap();

    // A. 空配置且无环境变量 -> 非零退出，不回落 Mock
    let cmd_empty = clean_env_cmd(&home, &ws, &["exec", "--isolated", "hello"]);
    let out_empty = run_with_stdin(cmd_empty, None).await;
    assert!(!out_empty.status.success());
    assert!(out_empty.stdout.is_empty());
    let empty_err = String::from_utf8_lossy(&out_empty.stderr);
    assert!(!empty_err.trim().is_empty());

    // B. 缺失必需凭据（builtin openai 但未提供 OPENAI_API_KEY / API_KEY）-> Harbor AgentAuthenticationError
    let cmd_missing_cred = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--provider",
            "openai",
            "--model",
            "gpt-4o",
            "hello",
        ],
    );
    let out_mc = run_with_stdin(cmd_missing_cred, None).await;
    assert!(!out_mc.status.success());
    assert!(out_mc.stdout.is_empty());
    let mc_err = String::from_utf8_lossy(&out_mc.stderr);
    assert!(mc_err.contains("code=missing_credential"), "{mc_err}");
    assert_eq!(
        classify_harbor_error("", &mc_err),
        Some(HarborErrorKind::AgentAuthenticationError),
        "{mc_err}"
    );

    // 准备一个编排各种错误响应的本地网关
    let secret = "sk-eval-super-secret-987654";
    let err_401 = MockStep::HttpError {
        status: 401,
        retry_after: None,
        retry_after_ms: None,
        body: json!({"error":{"message": format!("bad token {secret}"), "type":"authentication_error"}}),
    };
    let err_404 = MockStep::HttpError {
        status: 404,
        retry_after: None,
        retry_after_ms: None,
        body: json!({"error":{"message": "The model 'missing-model-404' does not exist"}}),
    };
    let err_429 = MockStep::HttpError {
        status: 429,
        retry_after: Some("0.05".into()),
        retry_after_ms: None,
        body: json!({"error":{"message": "slow down"}}),
    };
    let err_ctx = MockStep::HttpError {
        status: 400,
        retry_after: None,
        retry_after_ms: None,
        body: json!({"error":{"message":"This model's maximum context length is 8192 tokens","code":"context_length_exceeded"}}),
    };

    let state = GatewayState::new(vec![], MockStep::HangForever);
    let url = start_gateway(Arc::clone(&state)).await;
    let base_v1 = format!("{url}/v1");

    // C. HTTP 401（必须对凭据脱敏 + 保留状态码 + 命中 Harbor Auth）
    state.set_script(vec![err_401]);
    let mut c_401 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "401 test",
        ],
    );
    c_401.env("API_KEY", secret);
    let o_401 = run_with_stdin(c_401, None).await;
    assert!(!o_401.status.success());
    assert!(o_401.stdout.is_empty());
    let s_401 = String::from_utf8_lossy(&o_401.stderr);
    assert!(!s_401.contains(secret), "secret leaked to stderr: {s_401}");
    assert!(s_401.contains("401"), "{s_401}");
    assert!(s_401.contains("code=auth"), "{s_401}");
    assert_eq!(
        classify_harbor_error("", &s_401),
        Some(HarborErrorKind::AgentAuthenticationError),
        "{s_401}"
    );

    // D. HTTP 404 模型不可用
    state.set_script(vec![err_404]);
    let mut c_404 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "missing-model-404",
            "404 test",
        ],
    );
    c_404.env("API_KEY", secret);
    let o_404 = run_with_stdin(c_404, None).await;
    assert!(!o_404.status.success());
    assert!(o_404.stdout.is_empty());
    let s_404 = String::from_utf8_lossy(&o_404.stderr);
    assert!(s_404.contains("404"), "{s_404}");
    assert!(s_404.contains("missing-model-404"), "{s_404}");

    // E. HTTP 429 + Retry-After -> 保留 retry_after_ms=50 与 Harbor ApiRateLimitError
    let reqs_before_429 = state.requests.lock().unwrap().len();
    state.set_script(vec![
        err_429.clone(),
        err_429.clone(),
        err_429.clone(),
        err_429,
    ]);
    let mut c_429 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "429 test",
        ],
    );
    c_429.env("API_KEY", secret);
    let o_429 = run_with_stdin(c_429, None).await;
    assert_eq!(state.requests.lock().unwrap().len() - reqs_before_429, 4);
    assert!(!o_429.status.success());
    assert!(o_429.stdout.is_empty());
    let s_429 = String::from_utf8_lossy(&o_429.stderr);
    assert!(s_429.contains("429"), "{s_429}");
    assert!(s_429.contains("retry_after_ms=50"), "{s_429}");
    assert!(s_429.contains("code=rate_limit"), "{s_429}");
    assert_eq!(
        classify_harbor_error("", &s_429),
        Some(HarborErrorKind::ApiRateLimitError),
        "{s_429}"
    );

    // F. 上下文窗口溢出 -> Harbor ContextWindowExceededError（涵盖溢出重试/压缩分支）
    state.set_script(vec![err_ctx.clone(), err_ctx.clone(), err_ctx]);
    let mut c_ctx = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "ctx test",
        ],
    );
    c_ctx.env("API_KEY", secret);
    let o_ctx = run_with_stdin(c_ctx, None).await;
    assert!(!o_ctx.status.success());
    assert!(o_ctx.stdout.is_empty());
    let s_ctx = String::from_utf8_lossy(&o_ctx.stderr);
    assert!(s_ctx.contains("code=context_window_exceeded"), "{s_ctx}");
    assert_eq!(
        classify_harbor_error("", &s_ctx),
        Some(HarborErrorKind::ContextWindowExceededError),
        "{s_ctx}"
    );

    // G. max-steps 耗尽 -> 必须正好在 2 步后停止并非零退出，stdout 为空
    let reqs_before_steps = state.requests.lock().unwrap().len();
    state.set_script(vec![
        MockStep::CompletionsToolCall {
            id: "s1".into(),
            name: "read".into(),
            args: json!({"path":"nope1.txt"}),
        },
        MockStep::CompletionsToolCall {
            id: "s2".into(),
            name: "read".into(),
            args: json!({"path":"nope2.txt"}),
        },
        MockStep::CompletionsText("SHOULD_NOT_REACH_STEP3".into()),
    ]);
    let mut c_steps = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "--max-steps",
            "2",
            "steps test",
        ],
    );
    c_steps.env("API_KEY", secret);
    let o_steps = run_with_stdin(c_steps, None).await;
    assert_eq!(state.requests.lock().unwrap().len() - reqs_before_steps, 2);
    assert!(!o_steps.status.success());
    assert!(o_steps.stdout.is_empty());
    let s_steps = String::from_utf8_lossy(&o_steps.stderr);
    assert!(s_steps.contains("code=max_steps"), "{s_steps}");

    // H. 模型输出被 length/MaxTokens 截断 -> 必须非零退出且不得把半截文本输出到 stdout
    state.set_script(vec![MockStep::CompletionsTruncated(
        "PARTIAL_LEAK_FORBIDDEN".into(),
    )]);
    let mut c_trunc = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "trunc test",
        ],
    );
    c_trunc.env("API_KEY", secret);
    let o_trunc = run_with_stdin(c_trunc, None).await;
    assert!(!o_trunc.status.success());
    assert!(o_trunc.stdout.is_empty());
    let s_trunc = String::from_utf8_lossy(&o_trunc.stderr);
    assert!(!s_trunc.contains("PARTIAL_LEAK_FORBIDDEN"));
    assert_eq!(
        classify_harbor_error("", &s_trunc),
        Some(HarborErrorKind::OutputTokenExceededError),
        "{s_trunc}"
    );

    // I. --timeout 1s 快速超时收敛并清理
    state.set_script(vec![MockStep::HangForever]);
    let mut c_timeout = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "--timeout",
            "1",
            "hang test",
        ],
    );
    c_timeout.env("API_KEY", secret);
    let t0 = Instant::now();
    let o_timeout = run_with_stdin(c_timeout, None).await;
    assert!(t0.elapsed() < Duration::from_secs(6));
    assert!(!o_timeout.status.success());
    assert!(o_timeout.stdout.is_empty());
    let s_timeout = String::from_utf8_lossy(&o_timeout.stderr);
    assert!(s_timeout.contains("code=timeout"), "{s_timeout}");

    // J. SIGTERM 中断：收到请求后发信号，进程必须快速非零退出且 stdout 为空
    state.set_script(vec![MockStep::HangForever]);
    let reqs_before = state.requests.lock().unwrap().len();
    let mut c_sig = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "sigterm test",
        ],
    );
    c_sig.env("API_KEY", secret);
    let mut t_sig = tokio::process::Command::from(c_sig);
    t_sig
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = t_sig.spawn().unwrap();
    let pid = child.id().expect("child pid");

    let wait_start = Instant::now();
    while state.requests.lock().unwrap().len() == reqs_before {
        assert!(
            wait_start.elapsed() < Duration::from_secs(5),
            "timed out waiting for child request"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let kill_status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .unwrap();
    assert!(kill_status.success());
    let o_sig = tokio::time::timeout(Duration::from_secs(10), child.wait_with_output())
        .await
        .expect("SIGTERM child wait timed out")
        .unwrap();
    assert!(!o_sig.status.success());
    assert!(o_sig.stdout.is_empty());
    let s_sig = String::from_utf8_lossy(&o_sig.stderr);
    assert!(s_sig.contains("code=cancelled"), "{s_sig}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_supports_multi_turn_continue_resume_json_and_max_tokens() {
    let tmp = TempRoot::new("exec_multiturn_json");
    let home = tmp.path().join("home");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&ws).unwrap();

    let state = GatewayState::new(vec![], MockStep::HangForever);
    let url = start_gateway(Arc::clone(&state)).await;
    let base_v1 = format!("{url}/v1");

    // 1. 第一轮：使用 --json 开启持久化会话并获取 session_id
    state.set_script(vec![MockStep::CompletionsText(
        "Turn 1 acknowledged ALPHA-42".into(),
    )]);
    let mut c1 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "--json",
            "Remember secret ALPHA-42",
        ],
    );
    c1.env("API_KEY", "sk-test");
    let o1 = run_with_stdin(c1, None).await;
    assert!(
        o1.status.success(),
        "turn 1 failed: {}",
        String::from_utf8_lossy(&o1.stderr)
    );
    let v1: Value = serde_json::from_slice(&o1.stdout).expect("turn 1 stdout must be valid JSON");
    let sid = v1["session_id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .expect("missing session_id")
        .to_string();
    assert_eq!(v1["resumed"].as_bool(), Some(false), "{v1}");
    assert_eq!(
        v1["text"].as_str(),
        Some("Turn 1 acknowledged ALPHA-42"),
        "{v1}"
    );
    assert!(v1["duration_ms"].is_number(), "{v1}");

    // 2. 第二轮：-c (--continue) + --output-format json 续接最近会话
    state.set_script(vec![MockStep::CompletionsText(
        "The code is ALPHA-42".into(),
    )]);
    let mut c2 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "-c",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "--output-format",
            "json",
            "What was the code?",
        ],
    );
    c2.env("API_KEY", "sk-test");
    let o2 = run_with_stdin(c2, None).await;
    assert!(
        o2.status.success(),
        "turn 2 failed: {}",
        String::from_utf8_lossy(&o2.stderr)
    );
    let v2: Value = serde_json::from_slice(&o2.stdout).expect("turn 2 stdout must be valid JSON");
    assert_eq!(v2["session_id"].as_str(), Some(sid.as_str()), "{v2}");
    assert_eq!(v2["resumed"].as_bool(), Some(true), "{v2}");
    assert_eq!(v2["session_turns"].as_u64(), Some(2), "{v2}");
    assert_eq!(v2["text"].as_str(), Some("The code is ALPHA-42"), "{v2}");

    // 3. 第三轮：--resume <id> + --max-tokens 256 + --json
    state.set_script(vec![MockStep::CompletionsText("Confirmed turn 3".into())]);
    let mut c3 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--resume",
            &sid,
            "--max-tokens",
            "256",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "--json",
            "Final verification turn",
        ],
    );
    c3.env("API_KEY", "sk-test");
    let o3 = run_with_stdin(c3, None).await;
    assert!(
        o3.status.success(),
        "turn 3 failed: {}",
        String::from_utf8_lossy(&o3.stderr)
    );
    let v3: Value = serde_json::from_slice(&o3.stdout).expect("turn 3 stdout must be valid JSON");
    assert_eq!(v3["session_id"].as_str(), Some(sid.as_str()), "{v3}");
    assert_eq!(v3["resumed"].as_bool(), Some(true), "{v3}");
    assert_eq!(v3["session_turns"].as_u64(), Some(3), "{v3}");
    assert_eq!(v3["text"].as_str(), Some("Confirmed turn 3"), "{v3}");

    {
        let reqs = state.requests.lock().unwrap();
        assert_eq!(reqs.len(), 3);
        let msgs2 = reqs[1].body["messages"].to_string();
        assert!(msgs2.contains("Remember secret ALPHA-42"), "{msgs2}");
        assert!(msgs2.contains("Turn 1 acknowledged ALPHA-42"), "{msgs2}");
        assert!(msgs2.contains("What was the code?"), "{msgs2}");

        let max_tok = reqs[2]
            .body
            .get("max_completion_tokens")
            .or_else(|| reqs[2].body.get("max_tokens"))
            .and_then(Value::as_u64);
        assert_eq!(max_tok, Some(256), "{}", reqs[2].body);
        let msgs3 = reqs[2].body["messages"].to_string();
        assert!(msgs3.contains("Remember secret ALPHA-42"), "{msgs3}");
        assert!(msgs3.contains("The code is ALPHA-42"), "{msgs3}");
        assert!(msgs3.contains("Final verification turn"), "{msgs3}");
    }

    // 4. 拒绝用例：exec 下的裸 -r、--no-persist 组合、--isolated 组合、不存在的会话 ID
    for bad_args in [
        vec!["exec", "-r", "--", "hello"],
        vec!["exec", "-c", "--no-persist", "hello"],
        vec!["exec", "-c", "--isolated", "hello"],
    ] {
        let out = run_with_stdin(clean_env_cmd(&home, &ws, &bad_args), None).await;
        assert!(
            !out.status.success(),
            "args {bad_args:?} unexpectedly succeeded"
        );
        assert!(out.stdout.is_empty(), "args {bad_args:?} leaked stdout");
    }

    let mut c_missing = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--resume",
            "missing-session-id-999",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "hello",
        ],
    );
    c_missing.env("API_KEY", "sk-test");
    let o_missing = run_with_stdin(c_missing, None).await;
    assert!(!o_missing.status.success());
    assert!(o_missing.stdout.is_empty());

    // 5. Reasoning-only 且无可见正文 -> 自动重试耗尽后绝不能静默成功，必须返回非零并带 empty_response 诊断
    state.set_script(vec![
        MockStep::CompletionsReasoningOnly(
            "internal chain of thought without answer".into()
        );
        4
    ]);
    let mut c_reason = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--no-persist",
            "--base-url",
            &base_v1,
            "--model",
            "m1",
            "answer concisely",
        ],
    );
    c_reason.env("API_KEY", "sk-test");
    let o_reason = run_with_stdin(c_reason, None).await;
    assert!(!o_reason.status.success());
    assert!(o_reason.stdout.is_empty());
    let reason_err = String::from_utf8_lossy(&o_reason.stderr);
    assert!(reason_err.contains("empty_response"), "{reason_err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_responses_multi_turn_preserves_cache_key_affinity_and_static_instructions() {
    let tmp = TempRoot::new("exec_resp_cache");
    let home = tmp.path().join("home");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&ws).unwrap();

    let state = GatewayState::new(
        vec![
            MockStep::ResponsesText("RESP_ONE".into()),
            MockStep::ResponsesText("RESP_TWO".into()),
        ],
        MockStep::ResponsesText("FALLBACK".into()),
    );
    let url = start_gateway(Arc::clone(&state)).await;
    let base_v1 = format!("{url}/v1");

    let mut c1 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--protocol",
            "responses",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-5",
            "--json",
            "first prompt",
        ],
    );
    c1.env("OPENAI_API_KEY", "sk-oai-cache");
    let o1 = run_with_stdin(c1, None).await;
    assert!(
        o1.status.success(),
        "turn 1 failed: {}",
        String::from_utf8_lossy(&o1.stderr)
    );
    let v1: Value = serde_json::from_slice(&o1.stdout).unwrap();
    let sid = v1["session_id"].as_str().unwrap().to_string();
    assert!(!sid.is_empty());

    let mut c2 = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "-c",
            "--protocol",
            "responses",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-5",
            "--json",
            "second prompt",
        ],
    );
    c2.env("OPENAI_API_KEY", "sk-oai-cache");
    let o2 = run_with_stdin(c2, None).await;
    assert!(
        o2.status.success(),
        "turn 2 failed: {}",
        String::from_utf8_lossy(&o2.stderr)
    );
    let v2: Value = serde_json::from_slice(&o2.stdout).unwrap();
    assert_eq!(v2["resumed"].as_bool(), Some(true));
    assert_eq!(v2["session_id"].as_str(), Some(sid.as_str()));

    let reqs = state.requests.lock().unwrap().clone();
    assert_eq!(reqs.len(), 2);

    for (idx, req) in reqs.iter().enumerate() {
        assert!(req.path.ends_with("/responses"), "req[{idx}] {}", req.path);
        assert_eq!(
            req.session_id.as_deref(),
            Some(sid.as_str()),
            "req[{idx}] missing session_id header"
        );
        assert_eq!(
            req.x_client_request_id.as_deref(),
            Some(sid.as_str()),
            "req[{idx}] missing x-client-request-id header"
        );
        assert_eq!(
            req.x_session_id, None,
            "req[{idx}] unexpected x-session-id header for openai responses"
        );
        let cache_key = req.body["prompt_cache_key"].as_str().unwrap_or("");
        assert!(!cache_key.is_empty() && cache_key.chars().count() <= 64);
        assert_eq!(cache_key, sid.chars().take(64).collect::<String>());
    }

    assert_eq!(
        reqs[0].body["prompt_cache_key"],
        reqs[1].body["prompt_cache_key"]
    );
    assert!(
        reqs[0].body.get("instructions").is_none(),
        "top-level instructions must not be sent by default: {}",
        reqs[0].body
    );
    assert!(
        reqs[1].body.get("instructions").is_none(),
        "top-level instructions must not be sent on turn 2: {}",
        reqs[1].body
    );
    assert_eq!(reqs[0].body["input"][0]["role"], "developer");
    assert_eq!(reqs[1].body["input"][0]["role"], "developer");
    let instr0 = reqs[0].body["input"][0]["content"].as_str().unwrap();
    let instr1 = reqs[1].body["input"][0]["content"].as_str().unwrap();
    assert_eq!(instr0, instr1, "developer prefix changed across turns");
    assert_eq!(
        reqs[0].body["tools"], reqs[1].body["tools"],
        "tools declaration changed across turns"
    );
    let ws_raw = ws.to_string_lossy().replace('\\', "/");
    let ws_display = std::fs::canonicalize(&ws)
        .unwrap_or_else(|_| ws.clone())
        .to_string_lossy()
        .replace('\\', "/");
    assert!(!instr0.contains(&ws_raw), "{instr0}");
    assert!(!instr0.contains(&ws_display), "{instr0}");
    assert!(!instr0.contains("当前工作目录："), "{instr0}");
    assert!(!instr0.contains("今天是 "), "{instr0}");

    let count_env = |body: &Value| -> usize {
        body["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| {
                item.get("role").and_then(Value::as_str) == Some("user")
                    && item
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(|c| c.contains("<environment_context>"))
            })
            .count()
    };
    assert_eq!(count_env(&reqs[0].body), 1, "{}", reqs[0].body["input"]);
    assert_eq!(count_env(&reqs[1].body), 1, "{}", reqs[1].body["input"]);
    assert_eq!(reqs[0].body["input"][1]["role"], "user");
    let env_text = reqs[0].body["input"][1]["content"].as_str().unwrap();
    assert!(
        env_text.contains(&format!("<cwd>{ws_display}</cwd>")),
        "{env_text}"
    );
    assert!(env_text.contains("<current_date>"), "{env_text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_trace_out_isolated_normal_text_json_and_env_precedence() {
    let tmp = TempRoot::new("exec_trace_normal");
    let home = tmp.path().join("home");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&ws).unwrap();

    // 构造超过默认摘要上限的大文件以验证截断摘要
    let big_content = format!(
        "HEADER\n{}\nFOOTER",
        "line-payload-1234567890\n".repeat(120)
    );
    std::fs::write(ws.join("big.txt"), &big_content).unwrap();

    let state = GatewayState::new(
        vec![
            MockStep::CompletionsToolCallWithUsage {
                content_prefix: Some("Checking file...".into()),
                id: "call_big_read".into(),
                name: "read".into(),
                args: json!({ "path": "big.txt" }),
                usage: json!({
                    "prompt_tokens": 120,
                    "completion_tokens": 18,
                    "prompt_tokens_details": { "cached_tokens": 40 }
                }),
            },
            MockStep::CompletionsTextWithUsage {
                text: "TRACE_NORMAL_OK".into(),
                usage: json!({
                    "prompt_tokens": 260,
                    "completion_tokens": 12,
                    "prompt_tokens_details": { "cached_tokens": 100 }
                }),
            },
        ],
        MockStep::CompletionsText("UNEXPECTED".into()),
    );
    let url = start_gateway(Arc::clone(&state)).await;
    let base_v1 = format!("{url}/v1");

    // A. --isolated + text 模式 + --trace-out 自动创建多层父目录
    let trace_a = tmp.path().join("nested/a/trajectory.json");
    let mut cmd_a = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-trace",
            "--trace-out",
            trace_a.to_str().unwrap(),
            "inspect big.txt",
        ],
    );
    cmd_a.env("API_KEY", "sk-trace-test");
    let out_a = run_with_stdin(cmd_a, None).await;
    assert!(
        out_a.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_a.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out_a.stdout), "TRACE_NORMAL_OK\n");
    assert!(!home.join(".mizar/sessions.sqlite").exists());

    let raw_a = std::fs::read_to_string(&trace_a).expect("trajectory.json must exist");
    let export_a: TrajectoryExport = serde_json::from_str(&raw_a).unwrap();
    assert_eq!(export_a.exit_code, 0);
    assert_eq!(export_a.termination_reason, "completed");
    assert!(export_a.error.is_none());
    assert!(export_a.isolated);
    assert_eq!(export_a.output_format.as_deref(), Some("text"));
    assert!(export_a.ttft_ms.is_some());
    assert_eq!(export_a.turns.len(), 1);
    assert_eq!(export_a.turns[0].steps.len(), 2);
    // step 1: 120 - 40 = 80; step 2: 260 - 100 = 160 -> total input = 240, cache_read = 140
    assert_eq!(export_a.usage.input_tokens, 240);
    assert_eq!(export_a.usage.output_tokens, 30);
    assert_eq!(export_a.usage.cache_read_tokens, 140);
    assert_eq!(export_a.turns[0].usage, export_a.usage);
    assert_eq!(export_a.tool_calls.len(), 1);
    assert_eq!(export_a.tool_calls[0].name, "read");
    assert!(export_a.tool_calls[0].arguments.contains("big.txt"));
    assert_eq!(export_a.tool_results.len(), 1);
    assert!(export_a.tool_results[0].truncated);
    assert!(export_a.tool_results[0].original_chars > 2000);
    assert!(
        export_a.tool_results[0].content_summary.chars().count()
            < export_a.tool_results[0].original_chars
    );
    assert!(!export_a.tool_results[0].is_error);
    assert_eq!(export_a.report.snapshot.stats.tool_calls, 1);

    // B. --isolated + --output-format json + MIZAR_TRACE_OUT 环境变量
    state.set_script(vec![MockStep::CompletionsTextWithUsage {
        text: "TRACE_JSON_OK".into(),
        usage: json!({ "prompt_tokens": 50, "completion_tokens": 8 }),
    }]);
    let trace_b = tmp.path().join("env_only/trajectory.json");
    let mut cmd_b = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--output-format",
            "json",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-trace",
            "json mode",
        ],
    );
    cmd_b.env("API_KEY", "sk-trace-test");
    cmd_b.env("MIZAR_TRACE_OUT", &trace_b);
    let out_b = run_with_stdin(cmd_b, None).await;
    assert!(
        out_b.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_b.stderr)
    );
    let stdout_json: Value = serde_json::from_slice(&out_b.stdout).unwrap();
    assert_eq!(stdout_json["reason"], "completed");
    let export_b: TrajectoryExport =
        serde_json::from_str(&std::fs::read_to_string(&trace_b).unwrap()).unwrap();
    assert_eq!(export_b.exit_code, 0);
    assert_eq!(export_b.output_format.as_deref(), Some("json"));
    assert_eq!(
        export_b.session_id,
        stdout_json["session_id"].as_str().unwrap()
    );

    // C. --trace-out 优先级高于 MIZAR_TRACE_OUT
    state.set_script(vec![MockStep::CompletionsText("CLI_WINS".into())]);
    let trace_cli = tmp.path().join("cli_wins/trajectory.json");
    let trace_env = tmp.path().join("env_ignored/trajectory.json");
    let mut cmd_c = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-trace",
            "--trace-out",
            trace_cli.to_str().unwrap(),
            "precedence",
        ],
    );
    cmd_c.env("API_KEY", "sk-trace-test");
    cmd_c.env("MIZAR_TRACE_OUT", &trace_env);
    let out_c = run_with_stdin(cmd_c, None).await;
    assert!(out_c.status.success());
    assert!(trace_cli.exists());
    assert!(!trace_env.exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_trace_out_isolated_error_timeout_signal_and_write_failure() {
    let tmp = TempRoot::new("exec_trace_failures");
    let home = tmp.path().join("home");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("note.txt"), "hello trace").unwrap();

    let state = GatewayState::new(vec![], MockStep::HangForever);
    let url = start_gateway(Arc::clone(&state)).await;
    let base_v1 = format!("{url}/v1");
    let secret = "sk-trace-secret-leak-check-123";

    // A. 运行期 HTTP 401 报错退出 -> 生成 trajectory.json 且错误信息脱敏
    state.set_script(vec![MockStep::HttpError {
        status: 401,
        retry_after: None,
        retry_after_ms: None,
        body: json!({"error":{"message": format!("bad key {secret}"), "type":"authentication_error"}}),
    }]);
    let trace_err = tmp.path().join("err_401/trajectory.json");
    let mut cmd_err = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-trace",
            "--trace-out",
            trace_err.to_str().unwrap(),
            "fail 401",
        ],
    );
    cmd_err.env("API_KEY", secret);
    let out_err = run_with_stdin(cmd_err, None).await;
    assert!(!out_err.status.success());
    assert!(out_err.stdout.is_empty());
    let raw_err = std::fs::read_to_string(&trace_err).unwrap();
    assert!(
        !raw_err.contains(secret),
        "secret leaked into trace: {raw_err}"
    );
    let export_err: TrajectoryExport = serde_json::from_str(&raw_err).unwrap();
    assert_eq!(export_err.exit_code, 1);
    assert_eq!(export_err.termination_reason, "error");
    assert!(export_err.error.is_some());
    assert_eq!(export_err.turns.len(), 1);

    // B. --max-steps 1 超步退出 -> 生成 trajectory.json 且保留已执行 tool_call / tool_result
    state.set_script(vec![
        MockStep::CompletionsToolCall {
            id: "call_step1".into(),
            name: "read".into(),
            args: json!({"path": "note.txt"}),
        },
        MockStep::CompletionsText("SHOULD_NOT_RUN".into()),
    ]);
    let trace_steps = tmp.path().join("max_steps/trajectory.json");
    let mut cmd_steps = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-trace",
            "--max-steps",
            "1",
            "--trace-out",
            trace_steps.to_str().unwrap(),
            "step cap",
        ],
    );
    cmd_steps.env("API_KEY", secret);
    let out_steps = run_with_stdin(cmd_steps, None).await;
    assert!(!out_steps.status.success());
    assert!(out_steps.stdout.is_empty());
    let export_steps: TrajectoryExport =
        serde_json::from_str(&std::fs::read_to_string(&trace_steps).unwrap()).unwrap();
    assert_eq!(export_steps.exit_code, 1);
    assert_eq!(export_steps.termination_reason, "max_steps");
    assert_eq!(export_steps.tool_calls.len(), 1);
    assert_eq!(export_steps.tool_results.len(), 1);
    assert!(!export_steps.tool_results[0].is_error);
    assert!(
        export_steps.tool_results[0]
            .content_summary
            .contains("hello trace")
    );

    // C. --timeout 1 超时退出（第 1 步执行工具调用，第 2 步挂起触发超时）
    state.set_script(vec![
        MockStep::CompletionsToolCallWithUsage {
            content_prefix: None,
            id: "call_before_timeout".into(),
            name: "read".into(),
            args: json!({"path": "note.txt"}),
            usage: json!({ "prompt_tokens": 42, "completion_tokens": 9 }),
        },
        MockStep::HangForever,
    ]);
    let trace_timeout = tmp.path().join("timeout/trajectory.json");
    let mut cmd_timeout = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-trace",
            "--timeout",
            "1",
            "--trace-out",
            trace_timeout.to_str().unwrap(),
            "timeout test",
        ],
    );
    cmd_timeout.env("API_KEY", secret);
    let out_timeout = run_with_stdin(cmd_timeout, None).await;
    assert_eq!(out_timeout.status.code(), Some(124));
    assert!(out_timeout.stdout.is_empty());
    let export_timeout: TrajectoryExport =
        serde_json::from_str(&std::fs::read_to_string(&trace_timeout).unwrap()).unwrap();
    assert_eq!(export_timeout.exit_code, 124);
    assert_eq!(export_timeout.termination_reason, "timeout");
    assert_eq!(export_timeout.tool_calls.len(), 1);
    assert_eq!(export_timeout.tool_results.len(), 1);
    assert_eq!(export_timeout.usage.input_tokens, 42);
    assert_eq!(export_timeout.usage.output_tokens, 9);

    // D. SIGTERM 中断退出（通过 MIZAR_TRACE_OUT 导出）
    state.set_script(vec![MockStep::HangForever]);
    let reqs_before = state.requests.lock().unwrap().len();
    let trace_sig = tmp.path().join("sigterm/trajectory.json");
    let mut cmd_sig = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-trace",
            "sigterm trace",
        ],
    );
    cmd_sig.env("API_KEY", secret);
    cmd_sig.env("MIZAR_TRACE_OUT", &trace_sig);
    let mut t_sig = tokio::process::Command::from(cmd_sig);
    t_sig
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = t_sig.spawn().unwrap();
    let pid = child.id().expect("child pid");
    let wait_start = Instant::now();
    while state.requests.lock().unwrap().len() == reqs_before {
        assert!(wait_start.elapsed() < Duration::from_secs(5));
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .unwrap()
            .success()
    );
    let out_sig = tokio::time::timeout(Duration::from_secs(10), child.wait_with_output())
        .await
        .unwrap()
        .unwrap();
    assert!(!out_sig.status.success());
    assert!(out_sig.stdout.is_empty());
    let export_sig: TrajectoryExport =
        serde_json::from_str(&std::fs::read_to_string(&trace_sig).unwrap()).unwrap();
    assert_eq!(export_sig.termination_reason, "cancelled");
    assert_ne!(export_sig.exit_code, 0);

    // E. 落盘目标为目录导致写入失败时，仅打告警并保持主退出码与 stdout 干净
    state.set_script(vec![MockStep::CompletionsText(
        "MAIN_OK_DESPITE_TRACE_ERR".into(),
    )]);
    let bad_trace_dir = tmp.path().join("existing_dir");
    std::fs::create_dir_all(&bad_trace_dir).unwrap();
    let mut cmd_bad_out = clean_env_cmd(
        &home,
        &ws,
        &[
            "exec",
            "--isolated",
            "--base-url",
            &base_v1,
            "--model",
            "gpt-trace",
            "--trace-out",
            bad_trace_dir.to_str().unwrap(),
            "best effort",
        ],
    );
    cmd_bad_out.env("API_KEY", secret);
    let out_bad = run_with_stdin(cmd_bad_out, None).await;
    assert!(out_bad.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out_bad.stdout),
        "MAIN_OK_DESPITE_TRACE_ERR\n"
    );
    let err_bad = String::from_utf8_lossy(&out_bad.stderr);
    assert!(err_bad.contains("[W trace]"), "{err_bad}");
}
