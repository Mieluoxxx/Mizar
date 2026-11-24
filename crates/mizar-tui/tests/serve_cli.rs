//! `mizar serve` HTTP / WebSocket JSON-RPC 2.0 网关端到端集成测试（自包含 RFC 6455 客户端，无新增外部依赖）。

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use futures::channel::mpsc as futures_mpsc;
use mizar_client::{
    ProcessMeta, ProcessRegistration, RpcServer, RpcServerConfig, SessionActor, SessionActorConfig,
    meta_path_for_pid, socket_path_for_pid,
};
use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
use mizar_llm::{
    ContentBlock, FinishReason, GenerateOptions, LlmAdapter, LlmError, ModelInfo, ProviderInfo,
    ReasoningEffort, StreamChunk, ToolCall,
};
use mizar_session::{SessionCreateOptions, SessionEvent, SessionOrigin};
use mizar_tools::{ToolAnnotations, ToolDefinition};
use mizar_types::rpc::{
    CapabilitiesListResult, InteractionResolveResult, JsonRpcId, JsonRpcNotification,
    JsonRpcRequest, JsonRpcResponse, QueuePopAllResult, QueueRemoveResult, RpcApprovalMode,
    RpcNotification, SessionConfigResult, SessionEventsResult, SessionForkResult,
    SessionListResult, SessionRuntimeState, SessionSubscribeSnapshot, SessionTreeResult,
    StreamDeltaKind, TurnCancelResult, TurnSubmitResult,
};
use mizar_types::{ApprovalMode, BoxError};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use mizar_tui::serve::{ServeConfig, start_serve};

static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(_tag: &str) -> Self {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("msv_{}_{}", std::process::id(), seq));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &PathBuf {
        &self.0
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct ChildProcessGuard(Option<Child>);

impl ChildProcessGuard {
    fn spawn_sleeper() -> Self {
        let child = Command::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleeper child for external PID simulation");
        Self(Some(child))
    }

    fn pid(&self) -> u32 {
        self.0.as_ref().expect("child active").id()
    }
}

impl Drop for ChildProcessGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// 自包含 FIPS 180-4 SHA-1 摘要实现（仅用于 RFC 6455 `Sec-WebSocket-Accept` 握手计算）。
fn sha1_digest(input: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x6745_2301;
    let mut h1: u32 = 0xEFCD_AB89;
    let mut h2: u32 = 0x98BA_DCFE;
    let mut h3: u32 = 0x1032_5476;
    let mut h4: u32 = 0xC3D2_E1F0;

    let bit_len = (input.len() as u64) * 8;
    let mut padded = Vec::with_capacity(input.len() + 72);
    padded.extend_from_slice(input);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 80];
        for (slot, bytes) in w[..16].iter_mut().zip(chunk.as_chunks::<4>().0) {
            *slot = u32::from_be_bytes(*bytes);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut out = [0u8; 20];
    for (i, word) in [h0, h1, h2, h3, h4].into_iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn compute_ws_accept(sec_ws_key: &str) -> String {
    let mut combined = String::with_capacity(sec_ws_key.len() + 36);
    combined.push_str(sec_ws_key.trim());
    combined.push_str("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let digest = sha1_digest(combined.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(digest)
}

/// 轻量自包含 RFC 6455 WebSocket 客户端，用于端到端 JSON-RPC 2.0 验证。
struct WsTestClient {
    stream: BufReader<TcpStream>,
    pending_notifs: VecDeque<RpcNotification>,
    next_id: i64,
    mask_counter: u32,
}

impl WsTestClient {
    async fn handshake(addr: SocketAddr, origin: Option<&str>) -> Result<Self, (u16, String)> {
        let mut tcp = TcpStream::connect(addr)
            .await
            .map_err(|e| (0u16, e.to_string()))?;

        let key_bytes: [u8; 16] = [
            0x74, 0x68, 0x65, 0x20, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x20, 0x6e, 0x6f, 0x6e,
            0x63, 0x65,
        ];
        let sec_key = base64::engine::general_purpose::STANDARD.encode(key_bytes);

        let mut req = format!(
            "GET /ws HTTP/1.1\r\n\
             Host: {addr}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: {sec_key}\r\n\
             Sec-WebSocket-Version: 13\r\n"
        );
        if let Some(orig) = origin {
            req.push_str(&format!("Origin: {orig}\r\n"));
        }
        req.push_str("\r\n");

        tcp.write_all(req.as_bytes())
            .await
            .map_err(|e| (0u16, e.to_string()))?;

        let mut reader = BufReader::new(tcp);
        let mut status_line = String::new();
        reader
            .read_line(&mut status_line)
            .await
            .map_err(|e| (0u16, e.to_string()))?;
        let status_code = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);

        let mut accept_header: Option<String> = None;
        loop {
            let mut line = String::new();
            let n = reader
                .read_line(&mut line)
                .await
                .map_err(|e| (status_code, e.to_string()))?;
            if n == 0 || line == "\r\n" || line == "\n" {
                break;
            }
            if let Some((name, val)) = line.split_once(':')
                && name.trim().eq_ignore_ascii_case("sec-websocket-accept")
            {
                accept_header = Some(val.trim().to_string());
            }
        }

        if status_code != 101 {
            return Err((status_code, status_line.trim().to_string()));
        }
        let expected_accept = compute_ws_accept(&sec_key);
        assert_eq!(
            accept_header.as_deref(),
            Some(expected_accept.as_str()),
            "Sec-WebSocket-Accept mismatch"
        );

        Ok(Self {
            stream: reader,
            pending_notifs: VecDeque::new(),
            next_id: 1,
            mask_counter: 1,
        })
    }

    async fn connect(addr: SocketAddr, origin: Option<&str>) -> Self {
        Self::handshake(addr, origin)
            .await
            .expect("WebSocket handshake should succeed with 101")
    }

    async fn send_frame(&mut self, opcode: u8, payload: &[u8]) {
        self.mask_counter = self.mask_counter.wrapping_add(0x0101_0101);
        let mask = self.mask_counter.to_be_bytes();

        let mut frame = Vec::with_capacity(14 + payload.len());
        frame.push(0x80 | (opcode & 0x0F));
        if payload.len() <= 125 {
            frame.push(0x80 | (payload.len() as u8));
        } else if payload.len() <= 65_535 {
            frame.push(0x80 | 126);
            frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        } else {
            frame.push(0x80 | 127);
            frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        }
        frame.extend_from_slice(&mask);
        for (i, &b) in payload.iter().enumerate() {
            frame.push(b ^ mask[i % 4]);
        }
        self.stream
            .get_mut()
            .write_all(&frame)
            .await
            .expect("write WebSocket frame");
    }

    async fn send_text(&mut self, text: &str) {
        self.send_frame(0x1, text.as_bytes()).await;
    }

    async fn read_text(&mut self) -> Option<String> {
        let mut msg_buf: Option<(u8, Vec<u8>)> = None;
        loop {
            let mut hdr = [0u8; 2];
            if self.stream.read_exact(&mut hdr).await.is_err() {
                return None;
            }
            let fin = (hdr[0] & 0x80) != 0;
            let opcode = hdr[0] & 0x0F;
            let masked = (hdr[1] & 0x80) != 0;
            let len_code = hdr[1] & 0x7F;

            let payload_len: usize = match len_code {
                126 => {
                    let mut ext = [0u8; 2];
                    self.stream.read_exact(&mut ext).await.ok()?;
                    usize::from(u16::from_be_bytes(ext))
                }
                127 => {
                    let mut ext = [0u8; 8];
                    self.stream.read_exact(&mut ext).await.ok()?;
                    usize::try_from(u64::from_be_bytes(ext)).ok()?
                }
                n => usize::from(n),
            };

            let mask = if masked {
                let mut m = [0u8; 4];
                self.stream.read_exact(&mut m).await.ok()?;
                Some(m)
            } else {
                None
            };

            let mut payload = vec![0u8; payload_len];
            self.stream.read_exact(&mut payload).await.ok()?;
            if let Some(m) = mask {
                for (i, b) in payload.iter_mut().enumerate() {
                    *b ^= m[i % 4];
                }
            }

            match opcode {
                0x8 => return None, // Close
                0x9 => {
                    // Ping -> Pong
                    self.send_frame(0xA, &payload).await;
                }
                0xA => {} // Pong
                0x1 | 0x2 => {
                    if fin {
                        if opcode == 0x1 {
                            return String::from_utf8(payload).ok();
                        }
                    } else {
                        msg_buf = Some((opcode, payload));
                    }
                }
                0x0 => {
                    if let Some((orig_op, mut acc)) = msg_buf.take() {
                        acc.extend_from_slice(&payload);
                        if fin {
                            if orig_op == 0x1 {
                                return String::from_utf8(acc).ok();
                            }
                        } else {
                            msg_buf = Some((orig_op, acc));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    async fn send_request(&mut self, method: &str, params: Option<Value>) -> JsonRpcId {
        let id = JsonRpcId::Number(self.next_id);
        self.next_id += 1;
        let req = JsonRpcRequest::new(id.clone(), method, params);
        let wire = serde_json::to_string(&req).unwrap();
        self.send_text(&wire).await;
        id
    }

    async fn read_response_for(&mut self, expected_id: &JsonRpcId) -> JsonRpcResponse {
        tokio::time::timeout(Duration::from_secs(6), async {
            loop {
                let text = self
                    .read_text()
                    .await
                    .expect("unexpected WebSocket close waiting for JSON-RPC response");
                let val: Value = serde_json::from_str(&text).expect("valid JSON frame");
                if val.get("id").is_some() {
                    let resp: JsonRpcResponse =
                        serde_json::from_value(val).expect("JsonRpcResponse");
                    if resp.id.as_ref() == Some(expected_id) {
                        return resp;
                    }
                } else {
                    let notif_rpc: JsonRpcNotification =
                        serde_json::from_value(val).expect("JsonRpcNotification");
                    let typed =
                        RpcNotification::from_jsonrpc(&notif_rpc).expect("typed RpcNotification");
                    self.pending_notifs.push_back(typed);
                }
            }
        })
        .await
        .expect("timed out waiting for JSON-RPC response")
    }

    async fn call_ok<T: serde::de::DeserializeOwned>(
        &mut self,
        method: &str,
        params: Option<Value>,
    ) -> T {
        let id = self.send_request(method, params).await;
        let resp = self.read_response_for(&id).await;
        assert!(
            resp.error.is_none(),
            "method `{method}` returned error: {:?}",
            resp.error
        );
        serde_json::from_value(resp.result.expect("result present")).expect("deserialize result")
    }

    async fn next_notification(&mut self) -> RpcNotification {
        if let Some(n) = self.pending_notifs.pop_front() {
            return n;
        }
        let text = tokio::time::timeout(Duration::from_secs(6), self.read_text())
            .await
            .expect("timed out waiting for notification")
            .expect("unexpected WebSocket close waiting for notification");
        let val: Value = serde_json::from_str(&text).expect("valid JSON notification");
        let notif_rpc: JsonRpcNotification =
            serde_json::from_value(val).expect("JsonRpcNotification");
        RpcNotification::from_jsonrpc(&notif_rpc).expect("typed RpcNotification")
    }

    async fn next_notification_matching<F>(&mut self, mut pred: F) -> RpcNotification
    where
        F: FnMut(&RpcNotification) -> bool,
    {
        tokio::time::timeout(Duration::from_secs(6), async {
            loop {
                let n = self.next_notification().await;
                if pred(&n) {
                    return n;
                }
            }
        })
        .await
        .expect("timed out waiting for matching notification")
    }

    async fn close(mut self) {
        self.send_frame(0x8, &[]).await;
    }
}

struct ScriptedAdapter {
    steps: Mutex<VecDeque<Vec<StreamChunk>>>,
}

impl ScriptedAdapter {
    fn new(steps: Vec<Vec<StreamChunk>>) -> Arc<Self> {
        Arc::new(Self {
            steps: Mutex::new(VecDeque::from(steps)),
        })
    }
}

impl LlmAdapter for ScriptedAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: "scripted".into(),
            name: "Scripted".into(),
            protocol: "mock".into(),
            models: vec![ModelInfo {
                id: "script-v1".into(),
                name: "Script V1".into(),
                context_window: Some(64_000),
                max_tokens: Some(4096),
                reasoning_efforts: vec![ReasoningEffort::Low, ReasoningEffort::High],
                input: Vec::new(),
                ..Default::default()
            }],
        })
    }

    fn stream(&self, options: GenerateOptions) -> mizar_llm::BoxChunkStream {
        let chunks = self.steps.lock().unwrap().pop_front().unwrap_or_else(|| {
            vec![
                StreamChunk::TextDelta {
                    index: 0,
                    text: "fallback-ok".into(),
                },
                StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                },
            ]
        });
        let (tx, rx) = futures_mpsc::unbounded();
        tokio::spawn(async move {
            for chunk in chunks {
                if options.cancel.is_cancelled() {
                    let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                    return;
                }
                let _ = tx.unbounded_send(Ok(chunk));
            }
        });
        Box::pin(rx)
    }
}

#[test]
fn rfc6455_sha1_test_vector_matches_spec() {
    assert_eq!(
        compute_ws_accept("dGhlIHNhbXBsZSBub25jZQ=="),
        "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    );
}

#[tokio::test]
async fn serve_rejects_non_loopback_host_0_0_0_0_via_cli_and_api() {
    let ws = TempDirGuard::new("non_loopback");
    let run_dir = ws.path().join("run");

    // 1. 直接调用 start_serve 应拒绝 0.0.0.0 并明确提示仅绑定回环与使用 Tailscale Serve
    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
        .await
        .unwrap();
    let err = match start_serve(
        ServeConfig::new(ws.path(), &run_dir)
            .with_host("0.0.0.0")
            .with_port(0),
        Arc::clone(&rt),
    )
    .await
    {
        Ok(_) => panic!("0.0.0.0 must be rejected"),
        Err(err) => err,
    };
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("0.0.0.0") && err_msg.contains("Tailscale Serve"),
        "unexpected error message: {err_msg}"
    );
    rt.shutdown().await;

    // 2. 真实子进程调用 `mizar serve --host 0.0.0.0` 应快速失败退出，且 `--help` 正常输出帮助
    let bin = env!("CARGO_BIN_EXE_mizar");
    let help_out = Command::new(bin)
        .args(["serve", "--help"])
        .output()
        .expect("spawn mizar serve --help");
    assert!(help_out.status.success(), "mizar serve --help must succeed");
    let help_stdout = String::from_utf8_lossy(&help_out.stdout);
    assert!(
        help_stdout.contains("--host")
            && help_stdout.contains("--port")
            && help_stdout.contains("--run-dir")
            && help_stdout.contains("--session-db"),
        "missing expected serve flags in --help: {help_stdout}"
    );

    let output = Command::new(bin)
        .args(["serve", "--host", "0.0.0.0", "--port", "0"])
        .arg("--workspace")
        .arg(ws.path())
        .arg("--run-dir")
        .arg(&run_dir)
        .output()
        .expect("spawn mizar serve");
    assert!(
        !output.status.success(),
        "mizar serve --host 0.0.0.0 must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Tailscale Serve") && stderr.contains("0.0.0.0"),
        "expected Tailscale Serve error on stderr, got: {stderr}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn serve_enforces_origin_and_exposes_secret_free_http_endpoints() {
    let ws = TempDirGuard::new("http_discovery");
    let run_dir = ws.path().join("run");
    let db_path = ws.path().join("sessions.sqlite");

    // 创建模拟 git 仓库，验证 /api/sessions 自动提取 branch
    std::fs::create_dir_all(ws.path().join(".git")).unwrap();
    std::fs::write(
        ws.path().join(".git/HEAD"),
        "ref: refs/heads/feat/serve-test\n",
    )
    .unwrap();

    // 先构建一个带持久化历史会话 + 活跃外部 RpcServer（含 rpc_token）的现场
    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()).with_session_db(&db_path))
        .await
        .unwrap();
    rt.register_mock_adapter().unwrap();

    let static_sess = rt.sessions.create_with(SessionCreateOptions {
        id: Some("sess-static-http".into()),
        cwd: Some(ws.path().display().to_string()),
        title: Some("Static History".into()),
        origin: Some(SessionOrigin::User),
        ..Default::default()
    });
    static_sess.append(SessionEvent::User {
        text: "hello static".into(),
        images: Vec::new(),
    });
    static_sess.flush_async().await.unwrap();

    let live_sess = rt.sessions.create_with(SessionCreateOptions {
        id: Some("sess-live-http".into()),
        cwd: Some(ws.path().display().to_string()),
        title: Some("Live Session".into()),
        origin: Some(SessionOrigin::User),
        ..Default::default()
    });
    live_sess.append(SessionEvent::User {
        text: "hello live".into(),
        images: Vec::new(),
    });
    live_sess.flush_async().await.unwrap();

    let ext_child = ChildProcessGuard::spawn_sleeper();
    let ext_pid = ext_child.pid();
    let mut ext_meta = ProcessMeta::for_current_process(
        &run_dir,
        &live_sess.id,
        ws.path().display().to_string(),
        Some("mock".into()),
        Some("mock-v1".into()),
    );
    ext_meta.pid = ext_pid;
    ext_meta.socket = socket_path_for_pid(&run_dir, ext_pid)
        .to_string_lossy()
        .into_owned();
    let ext_reg = ProcessRegistration::create(&run_dir, ext_meta).unwrap();
    let ext_actor = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&live_sess),
        SessionActorConfig::new(ws.path()).with_run_dir(&run_dir),
    );
    let rpc_srv = RpcServer::start(ext_actor.clone(), ext_reg, RpcServerConfig::default()).unwrap();
    let secret_token = rpc_srv
        .rpc_token()
        .expect("loopback tcp rpc_token present")
        .to_string();

    let serve_cfg = ServeConfig::new(ws.path(), &run_dir)
        .with_port(0)
        .with_session_db_path(&db_path);
    let serve = start_serve(serve_cfg, Arc::clone(&rt)).await.unwrap();
    let base_url = format!("http://{}", serve.addr());
    let http = reqwest::Client::new();

    // 1. GET /health
    let health = http.get(format!("{base_url}/health")).send().await.unwrap();
    assert_eq!(health.status(), reqwest::StatusCode::OK);
    let health_json: Value = health.json().await.unwrap();
    assert_eq!(health_json["ok"], true);
    assert_eq!(health_json["version"], env!("CARGO_PKG_VERSION"));

    // 2. Origin 校验（HTTP & WebSocket 握手）
    for bad_origin in [
        "https://evil.example",
        "https://ts.net.evil.com",
        "https://user@mizar.tail1234.ts.net",
        "http://mizar.tail1234.ts.net",
    ] {
        let resp = http
            .get(format!("{base_url}/api/processes"))
            .header("Origin", bad_origin)
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            reqwest::StatusCode::FORBIDDEN,
            "HTTP must reject Origin={bad_origin}"
        );

        let ws_res = WsTestClient::handshake(serve.addr(), Some(bad_origin)).await;
        assert!(
            matches!(ws_res, Err((403, _))),
            "WS must reject Origin={bad_origin} with 403"
        );
    }

    for good_origin in [
        None,
        Some("http://127.0.0.1:3000"),
        Some("http://localhost:5173"),
        Some("https://mizar.tail1234.ts.net"),
    ] {
        let mut req = http.get(format!("{base_url}/api/processes"));
        if let Some(orig) = good_origin {
            req = req.header("Origin", orig);
        }
        let resp = req.send().await.unwrap();
        assert_eq!(
            resp.status(),
            reqwest::StatusCode::OK,
            "HTTP must allow Origin={good_origin:?}"
        );

        let ws = WsTestClient::connect(serve.addr(), good_origin).await;
        ws.close().await;
    }

    // 3. GET /api/processes：必须省略 rpc_token，保留 socket 与 session_id
    let proc_text = http
        .get(format!("{base_url}/api/processes"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        !proc_text.contains("rpc_token") && !proc_text.contains(&secret_token),
        "GET /api/processes leaked rpc_token: {proc_text}"
    );
    let proc_json: Value = serde_json::from_str(&proc_text).unwrap();
    let procs = proc_json["processes"].as_array().unwrap();
    assert_eq!(procs.len(), 1);
    assert_eq!(procs[0]["pid"], ext_pid);
    assert_eq!(procs[0]["session_id"], live_sess.id);
    assert_eq!(
        procs[0]["socket"].as_str(),
        Some(rpc_srv.socket_path().to_string_lossy().as_ref())
    );
    assert!(procs[0].get("rpc_token").is_none());

    // 4. GET /api/sessions：合并 SQLite 会话头与活进程归属，零密钥泄露
    let sess_text = http
        .get(format!("{base_url}/api/sessions"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        !sess_text.contains("rpc_token") && !sess_text.contains(&secret_token),
        "GET /api/sessions leaked rpc_token: {sess_text}"
    );
    let sess_json: Value = serde_json::from_str(&sess_text).unwrap();
    let items = sess_json["sessions"].as_array().unwrap();
    let live_item = items
        .iter()
        .find(|s| s["id"] == live_sess.id)
        .expect("live session present");
    assert_eq!(live_item["live_pid"], ext_pid);
    assert_eq!(
        live_item["socket"].as_str(),
        Some(rpc_srv.socket_path().to_string_lossy().as_ref())
    );
    assert_eq!(live_item["title"], "Live Session");
    assert_eq!(live_item["branch"], "feat/serve-test");
    assert!(live_item["updated_at"].as_str().is_some_and(|s| s != "-"));

    let static_item = items
        .iter()
        .find(|s| s["id"] == static_sess.id)
        .expect("static session present");
    assert!(static_item.get("live_pid").is_none());
    assert!(static_item.get("socket").is_none());
    assert_eq!(static_item["title"], "Static History");
    assert_eq!(static_item["branch"], "feat/serve-test");

    serve.shutdown().await;
    rpc_srv.shutdown().await;
    ext_actor.shutdown().await;
    rt.shutdown().await;
}

#[cfg(unix)]
#[tokio::test]
async fn serve_ws_proxies_live_rpc_server_without_double_writer() {
    let ws = TempDirGuard::new("ws_proxy_live");
    let run_dir = ws.path().join("run");
    let ext_db = ws.path().join("ext.sqlite");
    let gw_db = ws.path().join("gw.sqlite");

    // 外部活进程侧运行时：挂载受审批保护的工具与脚本化模型
    let ext_rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()).with_session_db(&ext_db))
        .await
        .unwrap();
    let _tool_guard = ext_rt
        .tools
        .register(
            ToolDefinition::sync_tool(
                "serve_guarded_write",
                "guarded write tool for serve ws proxy test",
                json!({ "type": "object", "properties": {} }),
                |_, _| Ok::<_, BoxError>("guarded_exec_ok".to_string()),
            )
            .with_annotations(ToolAnnotations {
                read_only: false,
                ..Default::default()
            }),
        )
        .unwrap();
    ext_rt
        .register_adapter(
            "scripted",
            ScriptedAdapter::new(vec![
                vec![
                    StreamChunk::tool_call(
                        0,
                        ToolCall {
                            id: "call-ws-1".into(),
                            name: "serve_guarded_write".into(),
                            arguments: r#"{"ok":true}"#.into(),
                        },
                    ),
                    StreamChunk::Finish {
                        reason: FinishReason::ToolCalls,
                        replay: None,
                    },
                ],
                vec![
                    StreamChunk::TextDelta {
                        index: 0,
                        text: "网关代理".into(),
                    },
                    StreamChunk::TextDelta {
                        index: 0,
                        text: "成功".into(),
                    },
                    StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    },
                ],
            ]),
        )
        .unwrap();

    let live_sess = ext_rt.sessions.create_with(SessionCreateOptions {
        id: Some("sess-live-ws".into()),
        cwd: Some(ws.path().display().to_string()),
        title: Some("Live Proxy Target".into()),
        origin: Some(SessionOrigin::User),
        ..Default::default()
    });
    let ext_actor = SessionActor::spawn(
        Arc::clone(&ext_rt),
        Arc::clone(&live_sess),
        SessionActorConfig::new(ws.path())
            .with_run_dir(&run_dir)
            .with_approval_mode(ApprovalMode::Ask)
            .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
    );

    let ext_child = ChildProcessGuard::spawn_sleeper();
    let ext_pid = ext_child.pid();
    let mut ext_meta = ProcessMeta::for_current_process(
        &run_dir,
        &live_sess.id,
        ws.path().display().to_string(),
        Some("scripted".into()),
        Some("script-v1".into()),
    );
    ext_meta.pid = ext_pid;
    ext_meta.socket = socket_path_for_pid(&run_dir, ext_pid)
        .to_string_lossy()
        .into_owned();
    let ext_reg = ProcessRegistration::create(&run_dir, ext_meta).unwrap();
    let rpc_srv = RpcServer::start(ext_actor.clone(), ext_reg, RpcServerConfig::default()).unwrap();

    // 启动独立的 mizar serve 网关（连接同一 run_dir）
    let gw_rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()).with_session_db(&gw_db))
        .await
        .unwrap();
    let serve = start_serve(
        ServeConfig::new(ws.path(), &run_dir)
            .with_port(0)
            .with_session_db_path(&gw_db),
        Arc::clone(&gw_rt),
    )
    .await
    .unwrap();

    let mut ws1 = WsTestClient::connect(serve.addr(), Some("http://localhost:5173")).await;
    let mut ws2 = WsTestClient::connect(serve.addr(), Some("https://mizar.tail1234.ts.net")).await;

    // 1. session.list 聚合活进程归属
    let list_res: SessionListResult = ws1.call_ok("session.list", Some(json!({}))).await;
    let found = list_res
        .sessions
        .iter()
        .find(|s| s.id == live_sess.id)
        .expect("live session in session.list");
    assert_eq!(found.active_writer_pid, Some(ext_pid));

    // 2. session.subscribe -> 代理至 p-<ext_pid>.sock 并返回水合快照
    let snap: SessionSubscribeSnapshot = ws1
        .call_ok(
            "session.subscribe",
            Some(json!({ "session_id": live_sess.id })),
        )
        .await;
    assert_eq!(snap.session.id, live_sess.id);

    // 3. turn.submit -> 触发工具审批 interaction.requested 推送
    let submit_res: TurnSubmitResult = ws1
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": live_sess.id,
                "text": "run guarded write",
            })),
        )
        .await;
    assert!(submit_res.accepted);

    let req_notif = ws1
        .next_notification_matching(|n| matches!(n, RpcNotification::InteractionRequested(_)))
        .await;
    let request_id = match req_notif {
        RpcNotification::InteractionRequested(req) => {
            assert_eq!(req.tool, "serve_guarded_write");
            req.request_id
        }
        _ => unreachable!(),
    };

    // 4. 双客户端并发 interaction.resolve：仅首个胜出，另一个返回 already_resolved
    let (res1, res2): (InteractionResolveResult, InteractionResolveResult) = tokio::join!(
        ws1.call_ok(
            "interaction.resolve",
            Some(json!({
                "session_id": live_sess.id,
                "request_id": request_id,
                "decision": "allow_once",
            }))
        ),
        ws2.call_ok(
            "interaction.resolve",
            Some(json!({
                "session_id": live_sess.id,
                "request_id": request_id,
                "decision": "deny",
            }))
        )
    );
    assert_ne!(res1.ok, res2.ok, "exactly one resolver must win");
    let loser = if res1.ok { &res2 } else { &res1 };
    assert_eq!(loser.reason.as_deref(), Some("already_resolved"));

    // 5. 接收完整流式增量 stream.delta + 落盘 event.committed + 回合结束 turn.state=idle
    let mut streamed = String::new();
    let mut saw_resolved = false;
    let mut committed_count = 0usize;
    tokio::time::timeout(Duration::from_secs(6), async {
        loop {
            match ws1.next_notification().await {
                RpcNotification::InteractionResolved(_) => {
                    saw_resolved = true;
                }
                RpcNotification::StreamDelta(d) if d.kind == StreamDeltaKind::Text => {
                    streamed.push_str(&d.delta);
                }
                RpcNotification::EventCommitted(_) => {
                    committed_count += 1;
                }
                RpcNotification::TurnState(ts)
                    if ts.session_id == live_sess.id && ts.state == SessionRuntimeState::Idle =>
                {
                    break;
                }
                _ => {}
            }
        }
    })
    .await
    .expect("timed out waiting for live proxy turn completion");

    assert!(saw_resolved);
    assert_eq!(streamed, "网关代理成功");
    assert!(
        committed_count >= 3,
        "expected user/tool/assistant committed events"
    );

    // 网关自身进程不得错误抢占或注册 p-<self_pid>.json
    assert!(
        !meta_path_for_pid(&run_dir, std::process::id()).exists(),
        "gateway must not spawn local SessionActor when proxying live external session"
    );

    ws1.close().await;
    ws2.close().await;
    serve.shutdown().await;
    rpc_srv.shutdown().await;
    ext_actor.shutdown().await;
    ext_rt.shutdown().await;
    gw_rt.shutdown().await;
}

#[tokio::test]
async fn serve_ws_serves_static_offline_sqlite_session_via_lazy_headless_actor() {
    let ws = TempDirGuard::new("ws_static_sqlite");
    let run_dir = ws.path().join("run");
    let db_path = ws.path().join("sessions.sqlite");
    let static_sid = "sess-offline-static";

    // 1. 预先写入一条静态离线 SQLite 会话，随后关闭初始 runtime
    {
        let seed_rt =
            MizarRuntime::build(AgentSpineConfig::new(ws.path()).with_session_db(&db_path))
                .await
                .unwrap();
        let sess = seed_rt.sessions.create_with(SessionCreateOptions {
            id: Some(static_sid.into()),
            cwd: Some(ws.path().display().to_string()),
            title: Some("Offline Resume".into()),
            origin: Some(SessionOrigin::User),
            ..Default::default()
        });
        sess.append(SessionEvent::User {
            text: "seed question".into(),
            images: Vec::new(),
        });
        sess.append(SessionEvent::Assistant {
            blocks: vec![ContentBlock::Text {
                text: "seed answer".into(),
            }],
            replay: None,
            usage: None,
        });
        sess.flush_async().await.unwrap();
        seed_rt.shutdown().await;
    }

    // 2. 以全新 runtime 启动 mizar serve（此时无任何活进程持有 sess-offline-static）
    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()).with_session_db(&db_path))
        .await
        .unwrap();
    rt.register_adapter(
        "scripted",
        ScriptedAdapter::new(vec![vec![
            StreamChunk::TextDelta {
                index: 0,
                text: "离线恢复".into(),
            },
            StreamChunk::TextDelta {
                index: 0,
                text: "已续写".into(),
            },
            StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay: None,
            },
        ]]),
    )
    .unwrap();

    let serve = start_serve(
        ServeConfig::new(ws.path(), &run_dir)
            .with_port(0)
            .with_session_db_path(&db_path),
        Arc::clone(&rt),
    )
    .await
    .unwrap();

    let self_meta_path = meta_path_for_pid(&run_dir, std::process::id());
    assert!(
        !self_meta_path.exists(),
        "headless SessionActor must be instantiated lazily only when needed"
    );

    let mut ws_client =
        WsTestClient::connect(serve.addr(), Some("https://mizar.tail1234.ts.net")).await;

    // 3. session.list 直接聚合 SQLite 静态会话，且不触发懒加载
    let list_res: SessionListResult = ws_client.call_ok("session.list", Some(json!({}))).await;
    let listed = list_res
        .sessions
        .iter()
        .find(|s| s.id == static_sid)
        .expect("static session in session.list");
    assert_eq!(listed.active_writer_pid, None);
    assert!(!self_meta_path.exists());

    // 4. session.subscribe 懒加载无头 SessionActor + ProcessRegistration，并水合历史事件
    let snap: SessionSubscribeSnapshot = ws_client
        .call_ok(
            "session.subscribe",
            Some(json!({ "session_id": static_sid })),
        )
        .await;
    assert_eq!(snap.session.id, static_sid);
    assert_eq!(snap.events.len(), 2);
    assert!(
        self_meta_path.exists(),
        "lazy headless SessionActor must register p-<pid>.json for single-writer locking"
    );
    let reg_meta: ProcessMeta =
        serde_json::from_slice(&std::fs::read(&self_meta_path).unwrap()).unwrap();
    assert_eq!(reg_meta.session_id, static_sid);

    // 5. turn.submit 驱动无头 SessionActor 执行回合，推送流式增量与落盘事件
    let submit_res: TurnSubmitResult = ws_client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": static_sid,
                "text": "continue offline session",
            })),
        )
        .await;
    assert!(submit_res.accepted);

    let mut streamed = String::new();
    let mut committed_new = 0usize;
    tokio::time::timeout(Duration::from_secs(6), async {
        loop {
            match ws_client.next_notification().await {
                RpcNotification::StreamDelta(d) if d.kind == StreamDeltaKind::Text => {
                    streamed.push_str(&d.delta);
                }
                RpcNotification::EventCommitted(ev) if ev.seq > 2 => {
                    committed_new += 1;
                }
                RpcNotification::TurnState(ts)
                    if ts.session_id == static_sid && ts.state == SessionRuntimeState::Idle =>
                {
                    break;
                }
                _ => {}
            }
        }
    })
    .await
    .expect("timed out waiting for static headless actor turn completion");

    assert_eq!(streamed, "离线恢复已续写");
    assert!(committed_new >= 2, "expected new user + assistant events");

    let events_res: SessionEventsResult = ws_client
        .call_ok(
            "session.events",
            Some(json!({ "session_id": static_sid, "limit": 20 })),
        )
        .await;
    assert!(events_res.events.len() >= 4);

    // 6. 覆盖其余 JSON-RPC 控制面：session.config / capabilities.list / session.tree / queue.* / turn.cancel / session.fork
    let cfg_res: SessionConfigResult = ws_client
        .call_ok(
            "session.config",
            Some(json!({
                "session_id": static_sid,
                "reasoning_effort": "high",
                "approval_mode": "ask",
            })),
        )
        .await;
    assert_eq!(cfg_res.config.reasoning_effort, Some(ReasoningEffort::High));
    assert_eq!(cfg_res.config.approval_mode, RpcApprovalMode::Ask);

    let caps: CapabilitiesListResult = ws_client
        .call_ok(
            "capabilities.list",
            Some(json!({ "session_id": static_sid })),
        )
        .await;
    assert!(!caps.commands.is_empty());
    assert!(!caps.providers.is_empty());

    let tree: SessionTreeResult = ws_client
        .call_ok("session.tree", Some(json!({ "session_id": static_sid })))
        .await;
    assert!(!tree.entries.is_empty());

    let q1: TurnSubmitResult = ws_client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": static_sid,
                "text": "queued-one",
                "mode": "queue",
            })),
        )
        .await;
    let q2: TurnSubmitResult = ws_client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": static_sid,
                "text": "queued-two",
                "mode": "queue",
            })),
        )
        .await;
    let rm_res: QueueRemoveResult = ws_client
        .call_ok(
            "queue.remove",
            Some(json!({
                "session_id": static_sid,
                "id": q1.queue_id.unwrap(),
            })),
        )
        .await;
    assert!(rm_res.removed);
    let pop_res: QueuePopAllResult = ws_client
        .call_ok("queue.pop_all", Some(json!({ "session_id": static_sid })))
        .await;
    assert_eq!(pop_res.items.len(), 1);
    assert_eq!(pop_res.items[0].id, q2.queue_id.unwrap());

    let cancel_res: TurnCancelResult = ws_client
        .call_ok("turn.cancel", Some(json!({ "session_id": static_sid })))
        .await;
    assert!(!cancel_res.cancelled);

    let fork_res: SessionForkResult = ws_client
        .call_ok(
            "session.fork",
            Some(json!({ "session_id": static_sid, "at_seq": 2 })),
        )
        .await;
    let forked_sid = fork_res.session.id;
    let fork_snap: SessionSubscribeSnapshot = ws_client
        .call_ok(
            "session.subscribe",
            Some(json!({ "session_id": forked_sid })),
        )
        .await;
    assert_eq!(fork_snap.session.id, forked_sid);
    let switched_meta: ProcessMeta =
        serde_json::from_slice(&std::fs::read(&self_meta_path).unwrap()).unwrap();
    assert_eq!(switched_meta.session_id, forked_sid);

    ws_client.close().await;

    // 7. ServeHandle::shutdown() 清理懒加载 actor 与 p-<pid>.json
    serve.shutdown().await;
    assert!(
        !self_meta_path.exists(),
        "ServeHandle::shutdown must clean up p-<pid>.json ProcessRegistration"
    );
    rt.shutdown().await;
}
