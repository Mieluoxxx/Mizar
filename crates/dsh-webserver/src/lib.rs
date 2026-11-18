//! 网页服务（对应 `NamiDSH/packages/webserver`）。
//!
//! - `GET /` / `GET /index.html` 返回聊天与轨迹检测页面
//! - `POST /api/chat` 跑一次 Agent，用 SSE 把回答、工具调用与轨迹告警实时推给浏览器
//! - `GET /api/sessions` / `GET /api/sessions/{id}/trajectory` / `GET /api/sessions/{id}/export` / `POST /api/trajectory/inspect`

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use axum::extract::{Path, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cordis::{
    BoxError, BoxFuture, Cleanup, Context, DisposableHandle, Inject, Plugin,
};
use dsh_agent_loop::{
    AgentLoopContextExt, AgentReasoningDeltaEvent, AgentTextDeltaEvent, AgentToolCallEvent,
    AgentUsageEvent, EVENT_AGENT_REASONING_DELTA, EVENT_AGENT_TEXT_DELTA, EVENT_AGENT_TOOL_CALL,
    EVENT_AGENT_USAGE,
};
use dsh_session::{SessionContextExt, SessionListQuery};
use dsh_trajectory::{
    DetectorConfig, EVENT_TRAJECTORY_ANOMALY, TimelineMode, TraceEvent, TrajectoryAnomalyEvent,
    TrajectoryReport, TrajectoryRuntime, TrajectorySummary, derive_timeline, detect_anomalies,
    fold_trajectory,
};
use futures::Stream;
use serde::Deserialize;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub const EVENT_WEBSERVER_LISTENING: &str = "webserver/listening";

#[derive(Debug, Clone)]
pub struct WebServerConfig {
    pub port: u16,
    pub host: String,
    /// 静态页面所在目录（里面要有 `index.html`）。
    pub static_dir: PathBuf,
}

impl WebServerConfig {
    pub fn new(static_dir: impl Into<PathBuf>) -> Self {
        Self {
            port: 3000,
            host: "127.0.0.1".to_string(),
            static_dir: static_dir.into(),
        }
    }
}

#[derive(Clone)]
struct AppState {
    ctx: Context,
    static_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChatRequest {
    session_id: Option<String>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TrajectoryQuery {
    mode: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InspectRequest {
    session_id: Option<String>,
    mode: Option<String>,
    jsonl: Option<String>,
    events: Option<Vec<TraceEvent>>,
}

/// 在 SSE 流被浏览器断开或正常结束时，自动中止 Agent 请求并撤销事件监听。
struct SseStreamGuard {
    rx: mpsc::UnboundedReceiver<Result<Event, Infallible>>,
    cancel: CancellationToken,
    disposers: Arc<Mutex<Vec<DisposableHandle>>>,
}

impl Stream for SseStreamGuard {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for SseStreamGuard {
    fn drop(&mut self) {
        self.cancel.cancel();
        let handles: Vec<DisposableHandle> = self.disposers.lock().unwrap().drain(..).collect();
        for d in handles {
            drop(d.dispose());
        }
    }
}

async fn handle_index(State(state): State<AppState>) -> Response {
    let path = state.static_dir.join("index.html");
    match tokio::fs::read_to_string(&path).await {
        Ok(html) => Html(html).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to read {}: {err}", path.display()),
        )
            .into_response(),
    }
}

async fn handle_list_sessions(State(state): State<AppState>) -> Response {
    if let Some(traj) = state.ctx.get::<TrajectoryRuntime>("trajectory") {
        return Json(traj.list_sessions()).into_response();
    }
    let Ok(sessions) = state.ctx.sessions() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "sessions service unavailable").into_response();
    };
    let list: Vec<TrajectorySummary> = match sessions.list_headers(&SessionListQuery::default()) {
        Ok(headers) => headers
            .into_iter()
            .map(|h| TrajectorySummary {
                session_id: h.id,
                event_count: h.event_count as usize,
                turns: 0,
                steps: 0,
                tool_calls: 0,
                findings_count: 0,
                has_errors: false,
            })
            .collect(),
        Err(_) => sessions
            .list()
            .into_iter()
            .map(|s| TrajectorySummary {
                session_id: s.id.clone(),
                event_count: s.log().len(),
                turns: 0,
                steps: 0,
                tool_calls: 0,
                findings_count: 0,
                has_errors: false,
            })
            .collect(),
    };
    Json(list).into_response()
}

async fn handle_session_trajectory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<TrajectoryQuery>,
) -> Response {
    let Some(traj) = state.ctx.get::<TrajectoryRuntime>("trajectory") else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "trajectory service not mounted" })),
        )
            .into_response();
    };
    let mode = TimelineMode::parse(query.mode.as_deref());
    match traj.inspect(&id, mode) {
        Some(report) => Json(report).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("session \"{id}\" not found") })),
        )
            .into_response(),
    }
}

async fn handle_session_export(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    let Some(traj) = state.ctx.get::<TrajectoryRuntime>("trajectory") else {
        return (StatusCode::SERVICE_UNAVAILABLE, "trajectory service not mounted")
            .into_response();
    };
    let Some(jsonl) = traj.export_jsonl(&id) else {
        return (StatusCode::NOT_FOUND, format!("session \"{id}\" not found")).into_response();
    };
    let mut resp = jsonl.into_response();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
    );
    if let Ok(disposition) =
        HeaderValue::from_str(&format!("attachment; filename=\"trajectory-{id}.jsonl\""))
    {
        resp.headers_mut()
            .insert(header::CONTENT_DISPOSITION, disposition);
    }
    resp
}

async fn handle_inspect_trajectory(
    State(state): State<AppState>,
    Json(req): Json<InspectRequest>,
) -> Response {
    let mode = TimelineMode::parse(req.mode.as_deref());
    let events = if let Some(jsonl) = req.jsonl.as_deref() {
        match TrajectoryRuntime::parse_jsonl(jsonl) {
            Ok(ev) => ev,
            Err(err) => {
                return (StatusCode::BAD_REQUEST, Json(json!({ "error": err }))).into_response();
            }
        }
    } else if let Some(ev) = req.events {
        ev
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "missing `jsonl` or `events` field" })),
        )
            .into_response();
    };

    let sid = req
        .session_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .or_else(|| events.first().map(|e| e.session_id.as_str()))
        .unwrap_or("offline");

    if let Some(traj) = state.ctx.get::<TrajectoryRuntime>("trajectory") {
        return Json(traj.inspect_events(sid, &events, mode)).into_response();
    }

    let snapshot = fold_trajectory(sid, &events);
    let timeline = derive_timeline(&snapshot.records, mode);
    let findings = detect_anomalies(&events, &snapshot, &DetectorConfig::default());
    Json(TrajectoryReport {
        snapshot,
        timeline,
        findings,
    })
    .into_response()
}

async fn handle_chat(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let sessions = match state.ctx.sessions() {
        Ok(s) => s,
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response(),
    };
    let agent_loop = match state.ctx.agent_loop() {
        Ok(a) => a,
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response(),
    };

    let session = req
        .session_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .and_then(|id| sessions.get(id))
        .unwrap_or_else(|| sessions.create(None));

    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let send = |tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
                event: &str,
                data: serde_json::Value| {
        let _ = tx.send(Ok(Event::default().event(event).data(data.to_string())));
    };

    send(&tx, "session", json!({ "id": session.id }));

    let cancel = CancellationToken::new();
    let session_id = session.id.clone();

    let disposers = Arc::new(Mutex::new(Vec::new()));
    {
        let tx_delta = tx.clone();
        let sid = session_id.clone();
        if let Ok(d) = state
            .ctx
            .on(EVENT_AGENT_TEXT_DELTA, move |ev: &AgentTextDeltaEvent| {
                if ev.session.id == sid {
                    send(&tx_delta, "delta", json!({ "text": ev.text }));
                }
            })
        {
            disposers.lock().unwrap().push(d);
        }
    }
    {
        let tx_reason = tx.clone();
        let sid = session_id.clone();
        if let Ok(d) =
            state
                .ctx
                .on(EVENT_AGENT_REASONING_DELTA, move |ev: &AgentReasoningDeltaEvent| {
                    if ev.session.id == sid {
                        send(&tx_reason, "reasoning", json!({ "text": ev.text }));
                    }
                })
        {
            disposers.lock().unwrap().push(d);
        }
    }
    {
        let tx_tool = tx.clone();
        let sid = session_id.clone();
        if let Ok(d) = state
            .ctx
            .on(EVENT_AGENT_TOOL_CALL, move |ev: &AgentToolCallEvent| {
                if ev.session.id == sid {
                    send(&tx_tool, "tool", json!({ "name": ev.call.name }));
                }
            })
        {
            disposers.lock().unwrap().push(d);
        }
    }
    {
        let tx_usage = tx.clone();
        let sid = session_id.clone();
        if let Ok(d) = state.ctx.on(EVENT_AGENT_USAGE, move |ev: &AgentUsageEvent| {
            if ev.session.id == sid {
                send(&tx_usage, "usage", json!(ev.usage));
            }
        }) {
            disposers.lock().unwrap().push(d);
        }
    }
    {
        let tx_anomaly = tx.clone();
        let sid = session_id.clone();
        if let Ok(d) =
            state
                .ctx
                .on(EVENT_TRAJECTORY_ANOMALY, move |ev: &TrajectoryAnomalyEvent| {
                    if ev.session_id == sid {
                        send(&tx_anomaly, "anomaly", json!(ev.finding));
                    }
                })
        {
            disposers.lock().unwrap().push(d);
        }
    }

    let message = req.message.unwrap_or_default();
    let cancel_task = cancel.clone();
    let disposers_task = Arc::clone(&disposers);
    let tx_task = tx;
    tokio::spawn(async move {
        match agent_loop.run(session, message, cancel_task.clone()).await {
            Ok(_) => {
                send(&tx_task, "done", json!({}));
            }
            Err(err) => {
                if !cancel_task.is_cancelled() {
                    send(&tx_task, "error", json!({ "message": err.to_string() }));
                }
            }
        }
        // 请求结束时撤销监听（释放事件总线里持有的 tx_*），让 SSE 流自然 EOF
        let handles: Vec<DisposableHandle> = disposers_task.lock().unwrap().drain(..).collect();
        for d in handles {
            drop(d.dispose());
        }
    });

    let stream = SseStreamGuard {
        rx,
        cancel,
        disposers,
    };

    let mut resp = Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response();
    resp.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache"),
    );
    resp
}

#[derive(Default, Clone, Copy)]
pub struct WebServerPlugin;

impl Plugin for WebServerPlugin {
    type Config = WebServerConfig;

    fn name(&self) -> Option<&str> {
        Some("webserver")
    }

    fn inject(&self) -> Inject {
        Inject::from(["agentLoop", "sessions"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let logger = ctx.logger_named("webserver");
            let state = AppState {
                ctx: ctx.clone(),
                static_dir: config.static_dir,
            };

            let app = Router::new()
                .route("/", get(handle_index))
                .route("/index.html", get(handle_index))
                .route("/api/chat", post(handle_chat))
                .route("/api/sessions", get(handle_list_sessions))
                .route("/api/sessions/{id}/trajectory", get(handle_session_trajectory))
                .route("/api/sessions/{id}/export", get(handle_session_export))
                .route("/api/trajectory/inspect", post(handle_inspect_trajectory))
                .with_state(state);

            let bind_addr = format!("{}:{}", config.host, config.port);
            let listener = TcpListener::bind(&bind_addr).await?;
            let local_addr: SocketAddr = listener.local_addr()?;

            let ctx_effect = ctx.clone();
            ctx.effect("webserver.listen", move || {
                let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
                let server_handle = tokio::spawn(async move {
                    let _ = axum::serve(listener, app)
                        .with_graceful_shutdown(async move {
                            let _ = shutdown_rx.await;
                        })
                        .await;
                });

                logger.info(format!("网页端已启动：http://localhost:{}", local_addr.port()));
                ctx_effect.emit(EVENT_WEBSERVER_LISTENING, local_addr);

                Ok::<_, cordis::CordisError>(Some(Cleanup::async_fn(move || async move {
                    let _ = shutdown_tx.send(());
                    let _ = server_handle.await;
                })))
            })?;

            Ok(None)
        })
    }
}
