//! `mizar-client` Unix Domain Socket 与回环 TCP（NDJSON / JSON-RPC 2.0）端到端集成测试。
//!
//! 覆盖 8 项核心契约：
//! 1. `0700` `run_dir`、`0600` `p-<pid>.sock` 与 `0600` `p-<pid>.json` 权限、原子元数据（含 `tcp_addr`/`rpc_token`）及停机清理；
//! 2. 流式半途 `session.subscribe` 水合快照（`active_turn` / `subagents` / `queued_inputs` / `after_seq`）；
//! 3. `stream.delta` 40ms/16 KiB UTF-8 聚合拆分及在 `event.committed` 前强制刷出；
//! 4. 双客户端并发 `interaction.resolve` 原子抢占 + `interaction.resolved` 广播 + SQLite `Approval` 落盘；
//! 5. `switch_session` 广播 `session.switched` 与 `p-<pid>.json` 热更新且保持 Socket 不断连；
//! 6. 慢客户端队列满（>256 帧）主动断连保护，不影响同实例健康客户端连续收帧；
//! 7. 真实 NDJSON Socket 覆盖 `session.config`、`session.list`、`session.events`、`session.fork`、`session.tree` 与 `capabilities.list`；
//! 8. 回环 TCP 旁路首帧 `auth.hello` 常量时间鉴权、未认证/错误 token 返回 `-32003` 即断连，以及认证后全链路流式与审批闭环。

#![cfg(unix)]

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener as StdUnixListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures::channel::mpsc as futures_mpsc;
use mizar_client::{
    ProcessMeta, ProcessRegistration, RpcServer, RpcServerConfig, SessionActor, SessionActorConfig,
    meta_path_for_pid, socket_path_for_pid,
};
use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
use mizar_llm::{
    FinishReason, GenerateOptions, LlmAdapter, LlmError, ModelInfo, ProviderInfo, ReasoningEffort,
    StreamChunk, ToolCall,
};
use mizar_session::{
    SessionEvent, SessionPersistence, SessionStore, SessionTraceEvent, SqliteSessionPersistence,
};
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolOutput};
use mizar_types::ApprovalMode;
use mizar_types::BoxError;
use mizar_types::rpc::{
    AuthHelloResult, CapabilitiesListResult, ERR_INVALID_PARAMS, ERR_METHOD_NOT_FOUND,
    ERR_PARSE_ERROR, ERR_SESSION_NOT_FOUND, ERR_UNAUTHORIZED, InteractionResolveResult, JsonRpcId,
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, METHOD_AUTH_HELLO, QueuePopAllResult,
    QueueRemoveResult, ResolvedBy, RpcApprovalDecision, RpcApprovalMode, RpcNotification,
    RpcSubagentStatus, SessionConfigResult, SessionEventsResult, SessionForkResult,
    SessionListResult, SessionRuntimeState, SessionSubscribeSnapshot, SessionTreeResult,
    StreamDeltaKind, SubmitMode, TurnCancelResult, TurnSubmitResult,
};
use serde_json::{Value, json};
use tokio::io::{
    AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader, ReadHalf, WriteHalf,
};
use tokio::net::{TcpStream, UnixStream};
use tokio::sync::watch;

/// 电平触发同步信号：保留常驻接收端，即使先触发后等待也不会丢失唤醒。
#[derive(Clone)]
struct StickySignal {
    tx: Arc<watch::Sender<bool>>,
    rx: watch::Receiver<bool>,
}

impl StickySignal {
    fn new() -> Self {
        let (tx, rx) = watch::channel(false);
        Self {
            tx: Arc::new(tx),
            rx,
        }
    }

    fn fire(&self) {
        let _ = self.tx.send(true);
    }

    async fn wait(&self) {
        let mut rx = self.rx.clone();
        let _ = rx.wait_for(|&fired| fired).await;
    }
}

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let seq = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("mizar_rpc_{tag}_{}_{}", std::process::id(), seq));
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

enum ScriptStep {
    Gated {
        before: Vec<StreamChunk>,
        gate: StickySignal,
        entered: StickySignal,
        after: Vec<StreamChunk>,
    },
    Immediate(Vec<StreamChunk>),
    EchoUser {
        prefix: String,
    },
}

struct ScriptedAdapter {
    steps: Mutex<VecDeque<ScriptStep>>,
}

impl ScriptedAdapter {
    fn new(steps: Vec<ScriptStep>) -> Arc<Self> {
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
            models: vec![
                ModelInfo {
                    id: "script-v1".into(),
                    name: "Script V1".into(),
                    context_window: Some(64_000),
                    max_tokens: Some(4096),
                    reasoning_efforts: vec![ReasoningEffort::Low, ReasoningEffort::High],
                    input: Vec::new(),
                    ..Default::default()
                },
                ModelInfo {
                    id: "script-v2".into(),
                    name: "Script V2".into(),
                    context_window: Some(128_000),
                    max_tokens: Some(8192),
                    reasoning_efforts: vec![ReasoningEffort::Medium, ReasoningEffort::High],
                    input: Vec::new(),
                    ..Default::default()
                },
            ],
        })
    }

    fn stream(&self, options: GenerateOptions) -> mizar_llm::BoxChunkStream {
        let step = self
            .steps
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| ScriptStep::EchoUser {
                prefix: "echo:".into(),
            });
        let (tx, rx) = futures_mpsc::unbounded();
        tokio::spawn(async move {
            match step {
                ScriptStep::Gated {
                    before,
                    gate,
                    entered,
                    after,
                } => {
                    for chunk in before {
                        let _ = tx.unbounded_send(Ok(chunk));
                    }
                    entered.fire();
                    tokio::select! {
                        _ = gate.wait() => {}
                        _ = options.cancel.cancelled() => {
                            let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                            return;
                        }
                    }
                    for chunk in after {
                        if options.cancel.is_cancelled() {
                            let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                            return;
                        }
                        let _ = tx.unbounded_send(Ok(chunk));
                    }
                }
                ScriptStep::Immediate(chunks) => {
                    for chunk in chunks {
                        if options.cancel.is_cancelled() {
                            let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                            return;
                        }
                        let _ = tx.unbounded_send(Ok(chunk));
                    }
                }
                ScriptStep::EchoUser { prefix } => {
                    let last_text = options
                        .messages
                        .last()
                        .map(|m| m.text_content())
                        .unwrap_or_default();
                    let _ = tx.unbounded_send(Ok(StreamChunk::TextDelta {
                        index: 0,
                        text: format!("{prefix}{last_text}"),
                    }));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
            }
        });
        Box::pin(rx)
    }
}

struct RpcTestClient<S> {
    reader: BufReader<ReadHalf<S>>,
    writer: WriteHalf<S>,
    pending_notifs: VecDeque<(String, RpcNotification)>,
    next_id: i64,
}

impl RpcTestClient<UnixStream> {
    async fn connect(socket_path: &std::path::Path) -> Self {
        let stream = UnixStream::connect(socket_path)
            .await
            .expect("connect unix rpc socket");
        Self::from_stream(stream)
    }
}

impl RpcTestClient<TcpStream> {
    async fn connect_tcp(addr: SocketAddr) -> Self {
        let stream = TcpStream::connect(addr)
            .await
            .expect("connect loopback tcp rpc");
        Self::from_stream(stream)
    }

    async fn auth_hello(&mut self, token: &str) -> AuthHelloResult {
        self.call_ok(METHOD_AUTH_HELLO, Some(json!({ "token": token })))
            .await
    }
}

impl<S> RpcTestClient<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn from_stream(stream: S) -> Self {
        let (read_half, write_half) = tokio::io::split(stream);
        Self {
            reader: BufReader::new(read_half),
            writer: write_half,
            pending_notifs: VecDeque::new(),
            next_id: 1,
        }
    }

    async fn send_raw_line(&mut self, raw: &str) {
        self.writer
            .write_all(raw.as_bytes())
            .await
            .expect("write raw line");
        if !raw.ends_with('\n') {
            self.writer.write_all(b"\n").await.expect("write newline");
        }
        self.writer.flush().await.expect("flush raw line");
    }

    async fn read_raw_line(&mut self) -> Option<String> {
        let mut line = String::new();
        let n = tokio::time::timeout(Duration::from_secs(5), self.reader.read_line(&mut line))
            .await
            .expect("timed out reading NDJSON line")
            .expect("read NDJSON line");
        if n == 0 { None } else { Some(line) }
    }

    async fn call_raw(&mut self, method: &str, params: Option<Value>) -> JsonRpcResponse {
        let id = self.next_id;
        self.next_id += 1;
        let req = JsonRpcRequest::new(JsonRpcId::Number(id), method, params);
        let wire = serde_json::to_string(&req).unwrap();
        self.send_raw_line(&wire).await;

        let expected_id = JsonRpcId::Number(id);
        loop {
            let line = self
                .read_raw_line()
                .await
                .expect("unexpected EOF waiting for response");
            let val: Value = serde_json::from_str(line.trim_end()).expect("valid JSON frame");
            if val.get("id").is_some() {
                let resp: JsonRpcResponse = serde_json::from_value(val).expect("JsonRpcResponse");
                if resp.id.as_ref() == Some(&expected_id) {
                    return resp;
                }
            } else {
                let notif_rpc: JsonRpcNotification =
                    serde_json::from_value(val).expect("JsonRpcNotification");
                let typed = RpcNotification::from_jsonrpc(&notif_rpc).expect("typed notification");
                self.pending_notifs.push_back((line, typed));
            }
        }
    }

    async fn call_ok<T: serde::de::DeserializeOwned>(
        &mut self,
        method: &str,
        params: Option<Value>,
    ) -> T {
        let resp = self.call_raw(method, params).await;
        assert!(
            resp.error.is_none(),
            "method `{method}` returned error: {:?}",
            resp.error
        );
        serde_json::from_value(resp.result.expect("result present")).expect("deserialize result")
    }

    async fn subscribe(
        &mut self,
        session_id: &str,
        after_seq: Option<u64>,
    ) -> SessionSubscribeSnapshot {
        self.call_ok(
            "session.subscribe",
            Some(json!({
                "session_id": session_id,
                "after_seq": after_seq,
            })),
        )
        .await
    }

    async fn next_notification(&mut self) -> (String, RpcNotification) {
        if let Some(item) = self.pending_notifs.pop_front() {
            return item;
        }
        let line = self
            .read_raw_line()
            .await
            .expect("unexpected EOF waiting for notification");
        let notif_rpc: JsonRpcNotification =
            serde_json::from_str(line.trim_end()).expect("JsonRpcNotification");
        let typed = RpcNotification::from_jsonrpc(&notif_rpc).expect("typed notification");
        (line, typed)
    }

    async fn next_notification_matching<F>(&mut self, mut pred: F) -> (String, RpcNotification)
    where
        F: FnMut(&RpcNotification) -> bool,
    {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let (raw, notif) = self.next_notification().await;
                if pred(&notif) {
                    return (raw, notif);
                }
            }
        })
        .await
        .expect("timed out waiting for matching notification")
    }
}

#[tokio::test]
async fn e2e_1_run_dir_socket_permissions_meta_errors_and_cleanup() {
    let ws = TempDirGuard::new("s1");
    let run_dir = ws.path().join("run");
    // 故意先以 0755 创建 run_dir，验证 RpcServer::start 会收敛为 0700
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::set_permissions(&run_dir, std::fs::Permissions::from_mode(0o755)).unwrap();

    // 故意留下一个已失活未监听的死 socket，验证启动时自动清理并重绑
    let stale_sock = socket_path_for_pid(&run_dir, std::process::id());
    {
        let _dead = StdUnixListener::bind(&stale_sock).unwrap();
    }
    assert!(stale_sock.exists());

    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
        .await
        .unwrap();
    rt.register_adapter("scripted", ScriptedAdapter::new(Vec::new()))
        .unwrap();
    let sess = rt.sessions.create(Some("sess-s1".into()));
    let actor = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess),
        SessionActorConfig::new(ws.path())
            .with_run_dir(&run_dir)
            .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
    );

    let meta = ProcessMeta::for_current_process(
        &run_dir,
        &sess.id,
        ws.path().display().to_string(),
        Some("scripted".into()),
        Some("script-v1".into()),
    );
    let registration = ProcessRegistration::create(&run_dir, meta).unwrap();
    let server = RpcServer::start(actor.clone(), registration, RpcServerConfig::default()).unwrap();

    let sock_path = server.socket_path().to_path_buf();
    let meta_path = server.meta_path().to_path_buf();
    assert_eq!(sock_path, stale_sock);
    assert_eq!(meta_path, meta_path_for_pid(&run_dir, std::process::id()));

    // 验证 0700 run_dir、0600 p-<pid>.sock 与 0600 p-<pid>.json 权限
    let run_mode = std::fs::metadata(&run_dir).unwrap().permissions().mode() & 0o777;
    let sock_mode = std::fs::metadata(&sock_path).unwrap().permissions().mode() & 0o777;
    let meta_mode = std::fs::metadata(&meta_path).unwrap().permissions().mode() & 0o777;
    assert_eq!(run_mode, 0o700);
    assert_eq!(sock_mode, 0o600);
    assert_eq!(meta_mode, 0o600);

    // 验证原子落盘 p-<pid>.json 元数据字段完整准确（含回环 TCP 地址与 64 位 hex rpc_token）
    let disk_meta: ProcessMeta =
        serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
    assert_eq!(disk_meta.pid, std::process::id());
    assert_eq!(disk_meta.session_id, sess.id);
    assert_eq!(disk_meta.cwd, ws.path().display().to_string());
    assert_eq!(disk_meta.socket, sock_path.to_string_lossy());
    let tcp_addr = server.tcp_addr().expect("loopback tcp addr");
    assert!(tcp_addr.ip().is_loopback());
    let rpc_token = server.rpc_token().expect("rpc token");
    assert_eq!(rpc_token.len(), 64);
    assert!(rpc_token.bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(
        disk_meta.tcp_addr.as_deref(),
        Some(tcp_addr.to_string().as_str())
    );
    assert_eq!(disk_meta.rpc_token.as_deref(), Some(rpc_token));
    let redacted = disk_meta.without_secret();
    assert_eq!(redacted.tcp_addr, disk_meta.tcp_addr);
    assert_eq!(redacted.rpc_token, None);
    assert_eq!(disk_meta.provider.as_deref(), Some("scripted"));
    assert_eq!(disk_meta.model.as_deref(), Some("script-v1"));
    assert!(disk_meta.started_at_ms > 0);

    // 验证无效 JSON (-32700)、未知方法 (-32601)、缺失参数 (-32602) 与会话不存在 (-32002)
    let mut client = RpcTestClient::connect(&sock_path).await;
    client.send_raw_line("{invalid_json").await;
    let parse_err_line = client.read_raw_line().await.unwrap();
    let parse_err: JsonRpcResponse = serde_json::from_str(parse_err_line.trim_end()).unwrap();
    assert_eq!(parse_err.error.unwrap().code, ERR_PARSE_ERROR);

    let unknown_resp = client.call_raw("unknown.method", Some(json!({}))).await;
    assert_eq!(unknown_resp.error.unwrap().code, ERR_METHOD_NOT_FOUND);

    let bad_params_resp = client.call_raw("turn.cancel", Some(json!({}))).await;
    assert_eq!(bad_params_resp.error.unwrap().code, ERR_INVALID_PARAMS);

    let missing_sess_resp = client
        .call_raw(
            "session.subscribe",
            Some(json!({ "session_id": "non-existent" })),
        )
        .await;
    assert_eq!(missing_sess_resp.error.unwrap().code, ERR_SESSION_NOT_FOUND);

    drop(client);
    server.shutdown().await;
    assert!(!sock_path.exists());
    assert!(!meta_path.exists());

    // 额外验证 Drop 自动清理路径（使用独立运行目录与 Actor，避免复用原 Actor 替换旧注册时提前删除同 PID 文件）
    let run_dir2 = ws.path().join("run2");
    let actor2 = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess),
        SessionActorConfig::new(ws.path()).with_run_dir(&run_dir2),
    );
    let meta2 = ProcessMeta::for_current_process(
        &run_dir2,
        &sess.id,
        ws.path().display().to_string(),
        None,
        None,
    );
    let reg2 = ProcessRegistration::create(&run_dir2, meta2).unwrap();
    let server2 = RpcServer::start(actor2.clone(), reg2, RpcServerConfig::default()).unwrap();
    let sock2 = server2.socket_path().to_path_buf();
    let meta_path2 = server2.meta_path().to_path_buf();
    assert!(sock2.exists() && meta_path2.exists());
    drop(server2);
    assert!(!sock2.exists());
    assert!(!meta_path2.exists());

    actor2.shutdown().await;
    actor.shutdown().await;
    rt.shutdown().await;
}

#[tokio::test]
async fn e2e_2_midstream_subscribe_snapshot_active_turn_subagents_queue_and_after_seq() {
    let ws = TempDirGuard::new("s2");
    let run_dir = ws.path().join("run");
    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
        .await
        .unwrap();

    let step1_entered = StickySignal::new();
    let step1_gate = StickySignal::new();
    let tool_entered = StickySignal::new();
    let tool_gate = StickySignal::new();
    let child_entered = StickySignal::new();
    let child_gate = StickySignal::new();

    let tool_entered_c = tool_entered.clone();
    let tool_gate_c = tool_gate.clone();
    let _slow_guard = rt
        .tools
        .register(
            ToolDefinition::async_rich_tool(
                "slow_inspect",
                "slow inspection tool",
                json!({ "type": "object", "properties": {} }),
                move |_args, ctx| {
                    let entered = tool_entered_c.clone();
                    let gate = tool_gate_c.clone();
                    async move {
                        ctx.emit_update("tool tail chunk", None);
                        entered.fire();
                        gate.wait().await;
                        Ok::<_, BoxError>(ToolOutput::text("slow_ok"))
                    }
                },
            )
            .with_annotations(ToolAnnotations {
                read_only: true,
                ..Default::default()
            }),
        )
        .unwrap();

    let adapter = ScriptedAdapter::new(vec![
        // 1. 先阻塞在 reasoning + text，放行后调用 slow_inspect
        ScriptStep::Gated {
            before: vec![
                StreamChunk::ReasoningDelta {
                    index: 0,
                    text: "思考现场".into(),
                },
                StreamChunk::TextDelta {
                    index: 1,
                    text: "半途正文".into(),
                },
            ],
            gate: step1_gate.clone(),
            entered: step1_entered.clone(),
            after: vec![
                StreamChunk::tool_call(
                    2,
                    ToolCall {
                        id: "call_slow_e2e".into(),
                        name: "slow_inspect".into(),
                        arguments: r#"{"scope":"full"}"#.into(),
                    },
                ),
                StreamChunk::Finish {
                    reason: FinishReason::ToolCalls,
                    replay: None,
                },
            ],
        },
        // 2. 启动后台子代理 worker-b
        ScriptStep::Immediate(vec![
            StreamChunk::tool_call(
                0,
                ToolCall {
                    id: "call_sub_e2e".into(),
                    name: "subagent".into(),
                    arguments: json!({
                        "action": "spawn",
                        "name": "worker-b",
                        "mode": "read_only",
                        "prompt": "inspect",
                    })
                    .to_string(),
                },
            ),
            StreamChunk::Finish {
                reason: FinishReason::ToolCalls,
                replay: None,
            },
        ]),
        // 3. 父会话收尾
        ScriptStep::Immediate(vec![
            StreamChunk::TextDelta {
                index: 0,
                text: "已启动子代理".into(),
            },
            StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay: None,
            },
        ]),
        // 4. 子代理阻塞在流式输出中
        ScriptStep::Gated {
            before: vec![StreamChunk::TextDelta {
                index: 0,
                text: "子代理流式中".into(),
            }],
            gate: child_gate.clone(),
            entered: child_entered.clone(),
            after: vec![StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay: None,
            }],
        },
        // 5. 队列任务执行与子代理汇总
        ScriptStep::EchoUser {
            prefix: "done:".into(),
        },
        ScriptStep::EchoUser {
            prefix: "done:".into(),
        },
    ]);
    rt.register_adapter("scripted", adapter).unwrap();

    let sess = rt.sessions.create(Some("sess-s2".into()));
    // 预置 2 条历史事件（seq=1, seq=2），便于测试 after_seq=1 过滤
    sess.append(SessionEvent::user("hist-1"));
    sess.append(SessionEvent::user("hist-2"));

    let actor = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess),
        SessionActorConfig::new(ws.path())
            .with_run_dir(&run_dir)
            .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
    );
    let meta = ProcessMeta::for_current_process(
        &run_dir,
        &sess.id,
        ws.path().display().to_string(),
        Some("scripted".into()),
        Some("script-v1".into()),
    );
    let reg = ProcessRegistration::create(&run_dir, meta).unwrap();
    let server = RpcServer::start(actor.clone(), reg, RpcServerConfig::default()).unwrap();

    let mut ctrl = RpcTestClient::connect(server.socket_path()).await;
    let submit_res: TurnSubmitResult = ctrl
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess.id,
                "text": "live-msg",
                "mode": "send",
            })),
        )
        .await;
    assert!(submit_res.accepted);
    assert_eq!(submit_res.turn, Some(1));

    step1_entered.wait().await;
    tokio::task::yield_now().await;

    // 趁忙碌期入队两条消息，测试 queued_inputs 快照与 after_seq 增量拉取
    let queued_res: TurnSubmitResult = ctrl
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess.id,
                "text": "queued-followup",
                "mode": "queue",
            })),
        )
        .await;
    assert_eq!(queued_res.mode, SubmitMode::Queue);
    assert!(queued_res.queue_id.is_some());

    let mut mid_client = RpcTestClient::connect(server.socket_path()).await;
    let mid_snap = mid_client.subscribe(&sess.id, Some(1)).await;
    assert_eq!(mid_snap.state, SessionRuntimeState::Streaming);
    assert!(
        mid_snap.events.iter().all(|e| e.seq > 1),
        "after_seq=1 必须仅返回 seq > 1 的事件: {:?}",
        mid_snap.events
    );
    assert!(mid_snap.events.iter().any(|e| e.seq == 2));
    let active = mid_snap.active_turn.expect("midstream active_turn");
    assert_eq!(active.reasoning, "思考现场");
    assert_eq!(active.text, "半途正文");
    assert!(active.running_tools.is_empty());
    assert_eq!(mid_snap.queued_inputs.len(), 1);
    assert_eq!(mid_snap.queued_inputs[0].text, "queued-followup");

    // 放行进入 slow_inspect，再连一个新客户端校验 running_tools 快照
    step1_gate.fire();
    tool_entered.wait().await;
    tokio::task::yield_now().await;

    let mut tool_client = RpcTestClient::connect(server.socket_path()).await;
    let tool_snap = tool_client.subscribe(&sess.id, None).await;
    let active_tool = tool_snap.active_turn.expect("tool active_turn");
    assert_eq!(active_tool.running_tools.len(), 1);
    assert_eq!(active_tool.running_tools[0].call_id, "call_slow_e2e");
    assert_eq!(active_tool.running_tools[0].name, "slow_inspect");
    assert_eq!(active_tool.running_tools[0].output_tail, "tool tail chunk");

    // 放行 slow_inspect，子代理 worker-b 进入阻塞中，校验 subagents 与子会话独立订阅
    tool_gate.fire();
    child_entered.wait().await;

    let (_, sub_notif) = mid_client
        .next_notification_matching(|n| {
            matches!(
                n,
                RpcNotification::SubagentUpdated(u)
                    if u.task.name == "worker-b" && u.task.status == RpcSubagentStatus::Running
            )
        })
        .await;
    let child_sid = match sub_notif {
        RpcNotification::SubagentUpdated(u) => u.task.child_session_id,
        _ => unreachable!(),
    };

    let mut sub_client = RpcTestClient::connect(server.socket_path()).await;
    let parent_with_sub = sub_client.subscribe(&sess.id, None).await;
    assert_eq!(parent_with_sub.subagents.len(), 1);
    assert_eq!(parent_with_sub.subagents[0].child_session_id, child_sid);
    assert_eq!(
        parent_with_sub.subagents[0].status,
        RpcSubagentStatus::Running
    );

    let child_snap = sub_client.subscribe(&child_sid, None).await;
    assert_eq!(child_snap.session.id, child_sid);
    assert_eq!(child_snap.state, SessionRuntimeState::Streaming);
    assert_eq!(
        child_snap.active_turn.expect("child active_turn").text,
        "子代理流式中"
    );

    // 放行子代理并等待收尾清理
    child_gate.fire();
    let _ = mid_client
        .next_notification_matching(|n| {
            matches!(
                n,
                RpcNotification::SubagentUpdated(u)
                    if u.task.name == "worker-b" && u.task.status == RpcSubagentStatus::Completed
            )
        })
        .await;

    drop(ctrl);
    drop(mid_client);
    drop(tool_client);
    drop(sub_client);
    server.shutdown().await;
    actor.shutdown().await;
    rt.shutdown().await;
}

#[tokio::test]
async fn e2e_3_stream_delta_coalescing_16kib_utf8_and_flush_before_event_committed() {
    let ws = TempDirGuard::new("s3");
    let run_dir = ws.path().join("run");
    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
        .await
        .unwrap();

    // 1. 第一步：快速连发 30 条短分片 + 超过 24 KiB 的多字节 UTF-8/转义字符分片，随后立刻 Finish 落盘
    let tiny_chunks: Vec<StreamChunk> = (0..30)
        .map(|i| StreamChunk::TextDelta {
            index: 0,
            text: format!("片段{i},"),
        })
        .collect();
    let tiny_expected: String = (0..30).map(|i| format!("片段{i},")).collect();

    let big_unit = "猫娘\"流式\"\n🚀";
    let big_payload = big_unit.repeat(2200);
    assert!(big_payload.len() > 24 * 1024);

    let mut step1_chunks = tiny_chunks;
    step1_chunks.push(StreamChunk::TextDelta {
        index: 0,
        text: big_payload.clone(),
    });
    step1_chunks.push(StreamChunk::Finish {
        reason: FinishReason::Stop,
        replay: None,
    });
    let full_expected = format!("{tiny_expected}{big_payload}");

    // 2. 第二步：在网关处停留，验证不来 EventCommitted 时 40ms 时间窗也能定时刷出聚合增量
    let gated_entered = StickySignal::new();
    let gated_gate = StickySignal::new();
    let adapter = ScriptedAdapter::new(vec![
        ScriptStep::Immediate(step1_chunks),
        ScriptStep::Gated {
            before: vec![
                StreamChunk::TextDelta {
                    index: 0,
                    text: "定时".into(),
                },
                StreamChunk::TextDelta {
                    index: 0,
                    text: "聚合".into(),
                },
                StreamChunk::TextDelta {
                    index: 0,
                    text: "窗口".into(),
                },
            ],
            gate: gated_gate.clone(),
            entered: gated_entered.clone(),
            after: vec![StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay: None,
            }],
        },
    ]);
    rt.register_adapter("scripted", adapter).unwrap();

    let sess = rt.sessions.create(Some("sess-s3".into()));
    let actor = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess),
        SessionActorConfig::new(ws.path())
            .with_run_dir(&run_dir)
            .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
    );
    let meta = ProcessMeta::for_current_process(
        &run_dir,
        &sess.id,
        ws.path().display().to_string(),
        Some("scripted".into()),
        Some("script-v1".into()),
    );
    let reg = ProcessRegistration::create(&run_dir, meta).unwrap();
    let server = RpcServer::start(actor.clone(), reg, RpcServerConfig::default()).unwrap();

    let mut client = RpcTestClient::connect(server.socket_path()).await;
    client.subscribe(&sess.id, None).await;

    let _: TurnSubmitResult = client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess.id,
                "text": "coalesce-test",
            })),
        )
        .await;

    let mut delta_frames = Vec::new();
    let mut saw_assistant_commit = false;
    loop {
        let (raw_line, notif) = client.next_notification().await;
        match notif {
            RpcNotification::StreamDelta(d) if d.kind == StreamDeltaKind::Text => {
                assert!(
                    !saw_assistant_commit,
                    "stream.delta 绝不能晚于其对应的 event.committed 到达！"
                );
                assert!(
                    raw_line.len() <= 16 * 1024,
                    "单帧字节数 {} 超过 16 KiB 上限",
                    raw_line.len()
                );
                delta_frames.push(d.delta);
            }
            RpcNotification::EventCommitted(ev) if ev.kind == "assistant" => {
                saw_assistant_commit = true;
            }
            RpcNotification::TurnState(ts) if ts.state == SessionRuntimeState::Idle => {
                break;
            }
            _ => {}
        }
    }

    assert!(saw_assistant_commit);
    // 31 条原始 chunk 至少应发生显著合并（< 10 帧，远小于 31 帧），且拆分还原后与原文逐字节一致
    assert!(
        delta_frames.len() >= 2 && delta_frames.len() < 10,
        "预期合并并拆成少量 <=16KiB 帧，实际收到 {} 帧",
        delta_frames.len()
    );
    let reassembled: String = delta_frames.concat();
    assert_eq!(reassembled, full_expected);

    // 验证 40ms 定时聚合窗（在 EventCommitted 尚未发生前即收到合并后的 "定时聚合窗口"）
    let _: TurnSubmitResult = client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess.id,
                "text": "timer-window",
            })),
        )
        .await;
    gated_entered.wait().await;
    let (_, timed_notif) = client
        .next_notification_matching(|n| matches!(n, RpcNotification::StreamDelta(_)))
        .await;
    match timed_notif {
        RpcNotification::StreamDelta(d) => {
            assert_eq!(d.delta, "定时聚合窗口");
        }
        _ => unreachable!(),
    }
    gated_gate.fire();
    let _ = client
        .next_notification_matching(|n| {
            matches!(
                n,
                RpcNotification::TurnState(ts) if ts.state == SessionRuntimeState::Idle
            )
        })
        .await;

    drop(client);
    server.shutdown().await;
    actor.shutdown().await;
    rt.shutdown().await;
}

#[tokio::test]
async fn e2e_4_dual_client_interaction_resolve_preemption_and_sqlite_approval_trace() {
    let ws = TempDirGuard::new("s4");
    let run_dir = ws.path().join("run");
    let db_path = ws.path().join("sessions.sqlite");
    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()).with_session_db(&db_path))
        .await
        .unwrap();

    let _danger_guard = rt
        .tools
        .register(
            ToolDefinition::sync_tool(
                "write_secret",
                "dangerous write",
                json!({ "type": "object", "properties": {} }),
                |_, _| Ok::<_, BoxError>("secret_ok".to_string()),
            )
            .with_annotations(ToolAnnotations {
                read_only: false,
                ..Default::default()
            }),
        )
        .unwrap();

    let adapter = ScriptedAdapter::new(vec![
        ScriptStep::Immediate(vec![
            StreamChunk::tool_call(
                0,
                ToolCall {
                    id: "call-apr-1".into(),
                    name: "write_secret".into(),
                    arguments: r#"{"target":"vault"}"#.into(),
                },
            ),
            StreamChunk::Finish {
                reason: FinishReason::ToolCalls,
                replay: None,
            },
        ]),
        ScriptStep::Immediate(vec![
            StreamChunk::TextDelta {
                index: 0,
                text: "审批执行完毕".into(),
            },
            StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay: None,
            },
        ]),
    ]);
    rt.register_adapter("scripted", adapter).unwrap();

    let sess = rt.sessions.create(Some("sess-s4".into()));
    let actor = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess),
        SessionActorConfig::new(ws.path())
            .with_run_dir(&run_dir)
            .with_approval_mode(ApprovalMode::Ask)
            .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
    );
    let meta = ProcessMeta::for_current_process(
        &run_dir,
        &sess.id,
        ws.path().display().to_string(),
        Some("scripted".into()),
        Some("script-v1".into()),
    );
    let reg = ProcessRegistration::create(&run_dir, meta).unwrap();
    let server = RpcServer::start(actor.clone(), reg, RpcServerConfig::default()).unwrap();

    let mut client_a = RpcTestClient::connect(server.socket_path()).await;
    let mut client_b = RpcTestClient::connect(server.socket_path()).await;
    client_a.subscribe(&sess.id, None).await;
    client_b.subscribe(&sess.id, None).await;

    let _: TurnSubmitResult = client_a
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess.id,
                "text": "run dangerous tool",
            })),
        )
        .await;

    let (_, req_a) = client_a
        .next_notification_matching(|n| matches!(n, RpcNotification::InteractionRequested(_)))
        .await;
    let (_, req_b) = client_b
        .next_notification_matching(|n| matches!(n, RpcNotification::InteractionRequested(_)))
        .await;

    let request_id = match (&req_a, &req_b) {
        (RpcNotification::InteractionRequested(a), RpcNotification::InteractionRequested(b)) => {
            assert_eq!(a.request_id, b.request_id);
            assert_eq!(a.tool, "write_secret");
            assert_eq!(a.call_id.as_deref(), Some("call-apr-1"));
            a.request_id.clone()
        }
        _ => unreachable!(),
    };

    // 双客户端真实并发经由各自 UnixStream 发送 interaction.resolve 抢占裁决
    let fut_a = client_a.call_ok::<InteractionResolveResult>(
        "interaction.resolve",
        Some(json!({
            "session_id": sess.id,
            "request_id": request_id,
            "decision": "allow_once",
        })),
    );
    let fut_b = client_b.call_ok::<InteractionResolveResult>(
        "interaction.resolve",
        Some(json!({
            "session_id": sess.id,
            "request_id": request_id,
            "decision": "deny",
            "note": "second client",
        })),
    );
    let (res_a, res_b) = tokio::join!(fut_a, fut_b);

    let outcomes = [res_a.clone(), res_b.clone()];
    let winners: Vec<_> = outcomes.iter().filter(|r| r.ok).collect();
    let losers: Vec<_> = outcomes.iter().filter(|r| !r.ok).collect();
    assert_eq!(winners.len(), 1);
    assert_eq!(losers.len(), 1);
    assert_eq!(losers[0].reason.as_deref(), Some("already_resolved"));

    // 两端均收到 interaction.resolved 广播且裁决内容一致
    let (_, resolved_a) = client_a
        .next_notification_matching(|n| matches!(n, RpcNotification::InteractionResolved(_)))
        .await;
    let (_, resolved_b) = client_b
        .next_notification_matching(|n| matches!(n, RpcNotification::InteractionResolved(_)))
        .await;
    let expected_decision = if res_a.ok {
        RpcApprovalDecision::AllowOnce
    } else {
        RpcApprovalDecision::Deny
    };
    for n in [resolved_a, resolved_b] {
        match n {
            RpcNotification::InteractionResolved(r) => {
                assert_eq!(r.request_id, request_id);
                assert_eq!(r.decision, expected_decision);
                assert_eq!(r.resolved_by, ResolvedBy::Rpc);
            }
            _ => unreachable!(),
        }
    }

    let _ = client_a
        .next_notification_matching(|n| {
            matches!(
                n,
                RpcNotification::TurnState(ts) if ts.state == SessionRuntimeState::Idle
            )
        })
        .await;

    let sid = sess.id.clone();
    let expected_msgs = sess.derive_messages();
    sess.flush_async().await.unwrap();

    drop(client_a);
    drop(client_b);
    server.shutdown().await;
    actor.shutdown().await;
    drop(sess);
    rt.shutdown().await;

    // 重开 SQLite 验证 SessionTraceEvent::Approval 落盘且 derive_messages() 完全不受影响
    let reopened = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let loaded = reopened.load_session(&sid).unwrap().expect("session saved");
    assert!(loaded.entries.iter().any(|entry| matches!(
        &entry.event,
        SessionEvent::Trace {
            detail: SessionTraceEvent::Approval {
                tool_call_id,
                tool,
                resolved_by,
                ..
            }
        } if tool_call_id == "call-apr-1"
            && tool == "write_secret"
            && resolved_by == "rpc"
    )));

    // 剥离 Approval Trace 构造镜像会话，确认 derive_messages() 逐条等价
    let plain_store = SessionStore::new();
    let plain = plain_store.create(None);
    for entry in &loaded.entries {
        if !matches!(
            &entry.event,
            SessionEvent::Trace {
                detail: SessionTraceEvent::Approval { .. }
            }
        ) {
            plain.append(entry.event.clone());
        }
    }
    let loaded_store = SessionStore::with_persistence(reopened);
    let reopened_sess = loaded_store.get(&sid).expect("reopen via store");
    assert_eq!(reopened_sess.derive_messages(), expected_msgs);
    assert_eq!(reopened_sess.derive_messages(), plain.derive_messages());
}

#[tokio::test]
async fn e2e_5_session_switched_broadcast_and_meta_update_without_disconnect() {
    let ws = TempDirGuard::new("s5");
    let run_dir = ws.path().join("run");
    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
        .await
        .unwrap();
    rt.register_adapter("scripted", ScriptedAdapter::new(Vec::new()))
        .unwrap();

    let sess_a = rt.sessions.create(Some("sess-a".into()));
    let sess_b = rt.sessions.create(Some("sess-b".into()));

    let actor = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess_a),
        SessionActorConfig::new(ws.path())
            .with_run_dir(&run_dir)
            .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
    );
    let meta = ProcessMeta::for_current_process(
        &run_dir,
        &sess_a.id,
        ws.path().display().to_string(),
        Some("scripted".into()),
        Some("script-v1".into()),
    );
    let reg = ProcessRegistration::create(&run_dir, meta).unwrap();
    let server = RpcServer::start(actor.clone(), reg, RpcServerConfig::default()).unwrap();

    let mut client = RpcTestClient::connect(server.socket_path()).await;
    let snap_a = client.subscribe(&sess_a.id, None).await;
    assert_eq!(snap_a.session.id, sess_a.id);

    // 切换活跃会话到 sess_b，验证不中断现有 Unix Socket 连接
    actor.switch_session(&sess_b.id).await.unwrap();

    let (_, switched_notif) = client
        .next_notification_matching(|n| matches!(n, RpcNotification::SessionSwitched(_)))
        .await;
    match switched_notif {
        RpcNotification::SessionSwitched(sw) => {
            assert_eq!(sw.old_session_id, sess_a.id);
            assert_eq!(sw.new_session_id, sess_b.id);
        }
        _ => unreachable!(),
    }

    // 验证 p-<pid>.json 已原子更新为新 session_id
    let disk_meta: ProcessMeta =
        serde_json::from_slice(&std::fs::read(server.meta_path()).unwrap()).unwrap();
    assert_eq!(disk_meta.session_id, sess_b.id);

    // 同一 Socket 无需重连即可直接订阅 sess_b 并提交新一轮对话
    let snap_b = client.subscribe(&sess_b.id, None).await;
    assert_eq!(snap_b.session.id, sess_b.id);
    let submit_b: TurnSubmitResult = client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess_b.id,
                "text": "hello b",
            })),
        )
        .await;
    assert!(submit_b.accepted);

    let _ = client
        .next_notification_matching(|n| {
            matches!(
                n,
                RpcNotification::TurnState(ts)
                    if ts.session_id == sess_b.id && ts.state == SessionRuntimeState::Idle
            )
        })
        .await;

    drop(client);
    server.shutdown().await;
    actor.shutdown().await;
    rt.shutdown().await;
}

#[tokio::test]
async fn e2e_6_slow_client_disconnect_protects_healthy_client() {
    let ws = TempDirGuard::new("s6");
    let run_dir = ws.path().join("run");
    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
        .await
        .unwrap();
    rt.register_adapter("scripted", ScriptedAdapter::new(Vec::new()))
        .unwrap();

    let sess = rt.sessions.create(Some("sess-s6".into()));
    let actor = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess),
        SessionActorConfig::new(ws.path())
            .with_run_dir(&run_dir)
            .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
    );
    let meta = ProcessMeta::for_current_process(
        &run_dir,
        &sess.id,
        ws.path().display().to_string(),
        Some("scripted".into()),
        Some("script-v1".into()),
    );
    let reg = ProcessRegistration::create(&run_dir, meta).unwrap();
    let server = RpcServer::start(actor.clone(), reg, RpcServerConfig::default()).unwrap();

    let mut slow_client = RpcTestClient::connect(server.socket_path()).await;
    let mut healthy_client = RpcTestClient::connect(server.socket_path()).await;

    slow_client.subscribe(&sess.id, None).await;
    healthy_client.subscribe(&sess.id, None).await;

    // 故意让 slow_client 停止读取 socket，同时让 healthy_client 后台持续读取，
    // 并发出 450 条 4 KiB 的 EventCommitted 通知：足以填满 slow_client 内核缓冲 + 256 帧出站队列。
    let total_events = 450usize;
    let healthy_task = tokio::spawn(async move {
        let mut healthy_received = 0usize;
        while healthy_received < total_events {
            let (_, notif) = healthy_client.next_notification().await;
            if matches!(notif, RpcNotification::EventCommitted(_)) {
                healthy_received += 1;
            }
        }
        (healthy_client, healthy_received)
    });

    let padding = "x".repeat(4096);
    for i in 0..total_events {
        sess.append(SessionEvent::user(format!("burst-{i:04}-{padding}")));
        if i % 16 == 15 {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    // 健康客户端必须毫发无损地连续收齐全部 450 条 event.committed
    let (healthy_client, healthy_received) = healthy_task.await.unwrap();
    assert_eq!(healthy_received, total_events);

    // 慢客户端稍后再尝试读至 EOF，必然在收齐前已被服务端断开（读到 EOF 且帧数严格少于 total_events）
    let mut slow_received = 0usize;
    while slow_client.read_raw_line().await.is_some() {
        slow_received += 1;
    }
    assert!(
        slow_received < total_events,
        "慢客户端应被队列满断连，实际竟收到了 {slow_received}/{total_events}"
    );

    drop(slow_client);
    drop(healthy_client);
    server.shutdown().await;
    actor.shutdown().await;
    rt.shutdown().await;
}

#[tokio::test]
async fn e2e_7_config_list_events_fork_tree_cancel_queue_and_capabilities() {
    let ws = TempDirGuard::new("s7");
    let run_dir = ws.path().join("run");
    let db_path = ws.path().join("sessions.sqlite");
    let other_ws = ws.path().join("other_project");
    std::fs::create_dir_all(&other_ws).unwrap();

    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()).with_session_db(&db_path))
        .await
        .unwrap();

    let cancel_entered = StickySignal::new();
    let cancel_gate = StickySignal::new();
    let adapter = ScriptedAdapter::new(vec![ScriptStep::Gated {
        before: vec![StreamChunk::TextDelta {
            index: 0,
            text: "即将取消".into(),
        }],
        gate: cancel_gate.clone(),
        entered: cancel_entered.clone(),
        after: vec![StreamChunk::Finish {
            reason: FinishReason::Stop,
            replay: None,
        }],
    }]);
    rt.register_adapter("scripted", adapter).unwrap();

    let main_cwd = ws.path().display().to_string();
    let sess_main = rt
        .sessions
        .create_with(mizar_session::SessionCreateOptions {
            cwd: Some(main_cwd.clone()),
            title: Some("Alpha Target Session".into()),
            ..Default::default()
        });
    for i in 1..=5 {
        sess_main.append(SessionEvent::user(format!("event-{i}")));
    }

    let sess_match_2 = rt
        .sessions
        .create_with(mizar_session::SessionCreateOptions {
            cwd: Some(main_cwd.clone()),
            title: Some("Alpha Second Match".into()),
            ..Default::default()
        });
    sess_match_2.append(SessionEvent::user("second"));

    let sess_other = rt
        .sessions
        .create_with(mizar_session::SessionCreateOptions {
            cwd: Some(other_ws.display().to_string()),
            title: Some("Alpha Other Cwd".into()),
            ..Default::default()
        });
    sess_other.append(SessionEvent::user("other"));

    let actor = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess_main),
        SessionActorConfig::new(ws.path())
            .with_run_dir(&run_dir)
            .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
    );
    let meta = ProcessMeta::for_current_process(
        &run_dir,
        &sess_main.id,
        main_cwd.clone(),
        Some("scripted".into()),
        Some("script-v1".into()),
    );
    let reg = ProcessRegistration::create(&run_dir, meta).unwrap();
    let server = RpcServer::start(actor.clone(), reg, RpcServerConfig::default()).unwrap();

    let mut client = RpcTestClient::connect(server.socket_path()).await;

    // 1. capabilities.list
    let caps: CapabilitiesListResult = client
        .call_ok(
            "capabilities.list",
            Some(json!({ "session_id": sess_main.id })),
        )
        .await;
    assert!(!caps.commands.is_empty());
    assert!(!caps.tools.is_empty());
    assert!(caps.providers.iter().any(|p| p.id == "scripted"));
    assert_eq!(caps.max_context_tokens, Some(64_000));

    // 2. session.config 热切换模型与审批模式，并验证 p-<pid>.json 同步刷新
    let cfg_res: SessionConfigResult = client
        .call_ok(
            "session.config",
            Some(json!({
                "session_id": sess_main.id,
                "model": "script-v2",
                "reasoning_effort": "high",
                "approval_mode": "ask",
            })),
        )
        .await;
    assert_eq!(cfg_res.config.provider.as_deref(), Some("scripted"));
    assert_eq!(cfg_res.config.model.as_deref(), Some("script-v2"));
    assert_eq!(cfg_res.config.reasoning_effort, Some(ReasoningEffort::High));
    assert_eq!(cfg_res.config.approval_mode, RpcApprovalMode::Ask);
    assert_eq!(cfg_res.context.max_tokens, Some(128_000));

    let disk_meta: ProcessMeta =
        serde_json::from_slice(&std::fs::read(server.meta_path()).unwrap()).unwrap();
    assert_eq!(disk_meta.model.as_deref(), Some("script-v2"));

    // 3. session.list 按 cwd / query / limit / cursor 分页
    let list_page1: SessionListResult = client
        .call_ok(
            "session.list",
            Some(json!({
                "cwd": main_cwd,
                "query": "alpha",
                "limit": 1,
            })),
        )
        .await;
    assert_eq!(list_page1.sessions.len(), 1);
    let cursor = list_page1
        .next_cursor
        .clone()
        .expect("should have next_cursor");
    let list_page2: SessionListResult = client
        .call_ok(
            "session.list",
            Some(json!({
                "cwd": main_cwd,
                "query": "alpha",
                "limit": 1,
                "cursor": cursor,
            })),
        )
        .await;
    assert_eq!(list_page2.sessions.len(), 1);
    assert!(list_page2.next_cursor.is_none());
    let listed_ids = [
        list_page1.sessions[0].id.as_str(),
        list_page2.sessions[0].id.as_str(),
    ];
    assert!(listed_ids.contains(&sess_main.id.as_str()));
    assert!(listed_ids.contains(&sess_match_2.id.as_str()));
    assert!(!listed_ids.contains(&sess_other.id.as_str()));

    // 4. session.events 双向游标分页
    let ev_head: SessionEventsResult = client
        .call_ok(
            "session.events",
            Some(json!({
                "session_id": sess_main.id,
                "after_seq": 0,
                "limit": 2,
            })),
        )
        .await;
    assert_eq!(
        ev_head.events.iter().map(|e| e.seq).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert!(ev_head.has_more);

    let ev_prev: SessionEventsResult = client
        .call_ok(
            "session.events",
            Some(json!({
                "session_id": sess_main.id,
                "before_seq": 5,
                "limit": 2,
            })),
        )
        .await;
    assert_eq!(
        ev_prev.events.iter().map(|e| e.seq).collect::<Vec<_>>(),
        vec![3, 4]
    );
    assert!(ev_prev.has_more);

    // 5. session.fork 与 session.tree
    let fork_res: SessionForkResult = client
        .call_ok(
            "session.fork",
            Some(json!({
                "session_id": sess_main.id,
                "at_seq": 3,
            })),
        )
        .await;
    assert_eq!(
        fork_res.session.parent_id.as_deref(),
        Some(sess_main.id.as_str())
    );
    assert_eq!(fork_res.session.fork_seq, Some(3));

    let tree_res: SessionTreeResult = client
        .call_ok("session.tree", Some(json!({ "session_id": sess_main.id })))
        .await;
    assert_eq!(tree_res.session_id, sess_main.id);
    assert_eq!(tree_res.entries.len(), 5);

    // 6. turn.submit + queue.remove + queue.pop_all + turn.cancel
    client.subscribe(&sess_main.id, None).await;
    let _: TurnSubmitResult = client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess_main.id,
                "text": "will cancel",
            })),
        )
        .await;
    cancel_entered.wait().await;

    let q1: TurnSubmitResult = client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess_main.id,
                "text": "draft-1",
                "mode": "queue",
            })),
        )
        .await;
    let _: TurnSubmitResult = client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess_main.id,
                "text": "draft-2",
                "mode": "queue",
            })),
        )
        .await;
    let rm_res: QueueRemoveResult = client
        .call_ok(
            "queue.remove",
            Some(json!({
                "session_id": sess_main.id,
                "id": q1.queue_id.unwrap(),
            })),
        )
        .await;
    assert!(rm_res.removed);
    assert_eq!(rm_res.item.unwrap().text, "draft-1");

    let pop_res: QueuePopAllResult = client
        .call_ok("queue.pop_all", Some(json!({ "session_id": sess_main.id })))
        .await;
    assert_eq!(pop_res.items.len(), 1);
    assert_eq!(pop_res.items[0].text, "draft-2");

    let cancel_res: TurnCancelResult = client
        .call_ok("turn.cancel", Some(json!({ "session_id": sess_main.id })))
        .await;
    assert!(cancel_res.cancelled);

    let _ = client
        .next_notification_matching(|n| {
            matches!(
                n,
                RpcNotification::TurnState(ts)
                    if ts.session_id == sess_main.id && ts.state == SessionRuntimeState::Idle
            )
        })
        .await;

    drop(client);
    server.shutdown().await;
    actor.shutdown().await;
    rt.shutdown().await;
}

#[tokio::test]
async fn e2e_8_loopback_tcp_auth_gate_and_live_rpc_roundtrip() {
    let ws = TempDirGuard::new("s8");
    let run_dir = ws.path().join("run");
    let db_path = ws.path().join("sessions.sqlite");
    let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()).with_session_db(&db_path))
        .await
        .unwrap();

    let _danger_guard = rt
        .tools
        .register(
            ToolDefinition::sync_tool(
                "tcp_guarded_write",
                "dangerous write for tcp e2e",
                json!({ "type": "object", "properties": {} }),
                |_, _| Ok::<_, BoxError>("tcp_tool_ok".to_string()),
            )
            .with_annotations(ToolAnnotations {
                read_only: false,
                ..Default::default()
            }),
        )
        .unwrap();

    let adapter = ScriptedAdapter::new(vec![
        ScriptStep::Immediate(vec![
            StreamChunk::tool_call(
                0,
                ToolCall {
                    id: "call-tcp-1".into(),
                    name: "tcp_guarded_write".into(),
                    arguments: r#"{"target":"tcp"}"#.into(),
                },
            ),
            StreamChunk::Finish {
                reason: FinishReason::ToolCalls,
                replay: None,
            },
        ]),
        ScriptStep::Immediate(vec![
            StreamChunk::TextDelta {
                index: 0,
                text: "你好，".into(),
            },
            StreamChunk::TextDelta {
                index: 0,
                text: "TCP".into(),
            },
            StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay: None,
            },
        ]),
    ]);
    rt.register_adapter("scripted", adapter).unwrap();

    let sess = rt.sessions.create(Some("sess-s8".into()));
    let actor = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess),
        SessionActorConfig::new(ws.path())
            .with_run_dir(&run_dir)
            .with_approval_mode(ApprovalMode::Ask)
            .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
    );
    let meta = ProcessMeta::for_current_process(
        &run_dir,
        &sess.id,
        ws.path().display().to_string(),
        Some("scripted".into()),
        Some("script-v1".into()),
    );
    let reg = ProcessRegistration::create(&run_dir, meta).unwrap();
    let server = RpcServer::start(actor.clone(), reg, RpcServerConfig::default()).unwrap();

    // 1. 验证 p-<pid>.json 为 0600 权限且含有效 tcp_addr 与 64 位 hex rpc_token
    let meta_mode = std::fs::metadata(server.meta_path())
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(meta_mode, 0o600);
    let disk_meta: ProcessMeta =
        serde_json::from_slice(&std::fs::read(server.meta_path()).unwrap()).unwrap();
    let tcp_addr = server.tcp_addr().expect("tcp_addr enabled by default");
    let token = server
        .rpc_token()
        .expect("rpc_token enabled by default")
        .to_string();
    assert_eq!(token.len(), 64);
    assert!(token.bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(
        disk_meta.tcp_addr.as_deref(),
        Some(tcp_addr.to_string().as_str())
    );
    assert_eq!(disk_meta.rpc_token.as_deref(), Some(token.as_str()));
    assert_eq!(disk_meta.without_secret().rpc_token, None);

    // 2. 未鉴权直接调用业务方法 -> 返回 -32003 并立即断开 TCP
    let mut unauth_client = RpcTestClient::connect_tcp(tcp_addr).await;
    let unauth_resp = unauth_client
        .call_raw("session.list", Some(json!({})))
        .await;
    assert_eq!(unauth_resp.error.unwrap().code, ERR_UNAUTHORIZED);
    assert!(
        unauth_client.read_raw_line().await.is_none(),
        "unauthenticated TCP connection must be closed immediately"
    );

    // 3. 错误 rpc_token（等长篡改末位与不等长）-> 返回 -32003 并立即断开 TCP
    let mut wrong_token = token.clone();
    let last_replacement = if wrong_token.ends_with('0') { '1' } else { '0' };
    wrong_token.pop();
    wrong_token.push(last_replacement);
    let mut bad_token_client = RpcTestClient::connect_tcp(tcp_addr).await;
    let bad_token_resp = bad_token_client
        .call_raw(METHOD_AUTH_HELLO, Some(json!({ "token": wrong_token })))
        .await;
    assert_eq!(bad_token_resp.error.unwrap().code, ERR_UNAUTHORIZED);
    assert!(bad_token_client.read_raw_line().await.is_none());

    let mut invalid_params_client = RpcTestClient::connect_tcp(tcp_addr).await;
    let invalid_params_resp = invalid_params_client
        .call_raw(METHOD_AUTH_HELLO, Some(json!({})))
        .await;
    assert_eq!(invalid_params_resp.error.unwrap().code, ERR_UNAUTHORIZED);
    assert!(invalid_params_client.read_raw_line().await.is_none());

    // 4. 首帧畸形 JSON -> 返回 -32700 并立即断开 TCP
    let mut bad_json_client = RpcTestClient::connect_tcp(tcp_addr).await;
    bad_json_client.send_raw_line("{bad_json").await;
    let bad_json_line = bad_json_client.read_raw_line().await.unwrap();
    let bad_json_resp: JsonRpcResponse = serde_json::from_str(bad_json_line.trim_end()).unwrap();
    assert_eq!(bad_json_resp.error.unwrap().code, ERR_PARSE_ERROR);
    assert!(bad_json_client.read_raw_line().await.is_none());

    // 5. 合法 auth.hello 鉴权后解锁 session.subscribe / turn.submit / live stream.delta / event.committed / interaction.resolve
    let mut client = RpcTestClient::connect_tcp(tcp_addr).await;
    let auth_res = client.auth_hello(&token).await;
    assert!(auth_res.ok);

    let snap = client.subscribe(&sess.id, None).await;
    assert_eq!(snap.session.id, sess.id);

    let submit_res: TurnSubmitResult = client
        .call_ok(
            "turn.submit",
            Some(json!({
                "session_id": sess.id,
                "text": "hello via tcp",
            })),
        )
        .await;
    assert!(submit_res.accepted);

    let (_, req_notif) = client
        .next_notification_matching(|n| matches!(n, RpcNotification::InteractionRequested(_)))
        .await;
    let request_id = match req_notif {
        RpcNotification::InteractionRequested(req) => {
            assert_eq!(req.tool, "tcp_guarded_write");
            req.request_id
        }
        _ => unreachable!(),
    };

    let resolve_res: InteractionResolveResult = client
        .call_ok(
            "interaction.resolve",
            Some(json!({
                "session_id": sess.id,
                "request_id": request_id,
                "decision": "allow_once",
            })),
        )
        .await;
    assert!(resolve_res.ok);

    let mut streamed_text = String::new();
    let mut saw_committed = false;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let (_, notif) = client.next_notification().await;
            match notif {
                RpcNotification::StreamDelta(d) if d.kind == StreamDeltaKind::Text => {
                    streamed_text.push_str(&d.delta);
                }
                RpcNotification::EventCommitted(_) => {
                    saw_committed = true;
                }
                RpcNotification::TurnState(ts)
                    if ts.session_id == sess.id && ts.state == SessionRuntimeState::Idle =>
                {
                    break;
                }
                _ => {}
            }
        }
    })
    .await
    .expect("timed out waiting for TCP turn completion");

    assert_eq!(streamed_text, "你好，TCP");
    assert!(saw_committed);

    drop(client);
    server.shutdown().await;
    actor.shutdown().await;

    // 6. 验证 enable_loopback_tcp = false 时仅绑定 Unix Socket，不暴露 TCP 或写入密钥
    let run_dir_no_tcp = ws.path().join("run_no_tcp");
    let actor_no_tcp = SessionActor::spawn(
        Arc::clone(&rt),
        Arc::clone(&sess),
        SessionActorConfig::new(ws.path()).with_run_dir(&run_dir_no_tcp),
    );
    let meta_no_tcp = ProcessMeta::for_current_process(
        &run_dir_no_tcp,
        &sess.id,
        ws.path().display().to_string(),
        None,
        None,
    );
    let reg_no_tcp = ProcessRegistration::create(&run_dir_no_tcp, meta_no_tcp).unwrap();
    let server_no_tcp = RpcServer::start(
        actor_no_tcp.clone(),
        reg_no_tcp,
        RpcServerConfig {
            enable_loopback_tcp: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(server_no_tcp.tcp_addr(), None);
    assert_eq!(server_no_tcp.rpc_token(), None);
    let disk_no_tcp: ProcessMeta =
        serde_json::from_slice(&std::fs::read(server_no_tcp.meta_path()).unwrap()).unwrap();
    assert_eq!(disk_no_tcp.tcp_addr, None);
    assert_eq!(disk_no_tcp.rpc_token, None);

    server_no_tcp.shutdown().await;
    actor_no_tcp.shutdown().await;
    rt.shutdown().await;
}
