//! 仅监听回环地址的 HTTP / WebSocket JSON-RPC 2.0 网关（`mizar serve`）。
//!
//! 对齐 M4.2 网关安全边界与单写者聚合原则：
//! - 严格校验绑定地址仅限回环（`127.0.0.1` / `::1` / `[::1]` / `localhost`），远程访问交由 Tailscale Serve；
//! - 对 `/ws` 与 `/api/*` 强制执行 `Origin` 校验（放行缺失 `Origin`、本地回环 HTTP 与 `https://<host>.ts.net`）；
//! - `/api/processes` 与 `/api/sessions` 必须剥离 `rpc_token` 等本地密钥，绝不对外泄露；
//! - 对已由活外部进程持有的 `session_id` 通过持久化 `UnixStream` 代理至 `p-<pid>.sock`，
//!   对无活宿主的静态 SQLite 历史会话懒加载进程内无头 `SessionActor` 并持有 `ProcessRegistration` 参与单写者锁。

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use chrono::DateTime;
use futures::{SinkExt, StreamExt};
use mizar_client::actor::{ActorError, SessionActor, SessionActorConfig, SessionActorHandle};
use mizar_client::run_dir::{
    ProcessMeta, ProcessRegistration, check_session_writable, ensure_run_dir, is_pid_alive,
    resolve_run_dir_path, scan_live_processes,
};
use mizar_core::spine::{
    AgentSpineConfig, MizarRuntime, load_mizar_config, resolve_default_session_db_path,
};
use mizar_llm::retry::RetryConfig;
use mizar_session::{SessionCreateOptions, SessionListQuery, SessionOrigin};
use mizar_types::BoxError;
use mizar_types::rpc::{
    CapabilitiesListParams, InteractionResolveParams, JsonRpcError, JsonRpcId, JsonRpcRequest,
    JsonRpcResponse, METHOD_AUTH_HELLO, METHOD_CAPABILITIES_LIST, METHOD_INTERACTION_RESOLVE,
    METHOD_QUEUE_POP_ALL, METHOD_QUEUE_REMOVE, METHOD_SESSION_CONFIG, METHOD_SESSION_EVENTS,
    METHOD_SESSION_FORK, METHOD_SESSION_LIST, METHOD_SESSION_SUBSCRIBE, METHOD_SESSION_TREE,
    METHOD_TURN_CANCEL, METHOD_TURN_SUBMIT, QueuePopAllParams, QueueRemoveParams, ResolvedBy,
    RpcNotification, SessionConfigParams, SessionEventsParams, SessionForkParams,
    SessionListItemDto, SessionListParams, SessionListResult, SessionSubscribeParams,
    SessionTreeParams, StreamDeltaKind, StreamDeltaNotification, TurnCancelParams,
    TurnSubmitParams,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::{Mutex as AsyncMutex, broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

#[cfg(unix)]
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
#[cfg(unix)]
use tokio::net::UnixStream;

const DEFAULT_SERVE_HOST: &str = "127.0.0.1";
const DEFAULT_SERVE_PORT: u16 = 8787;
const DEFAULT_SESSION_LIST_LIMIT: usize = 50;
const DEFAULT_WS_OUTBOUND_CAPACITY: usize = 256;
const DEFAULT_COALESCE_WINDOW: Duration = Duration::from_millis(40);
const DEFAULT_MAX_DELTA_FRAME_BYTES: usize = 16 * 1024;

/// `mizar serve` 网关配置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub workspace: PathBuf,
    pub run_dir: PathBuf,
    pub session_db_path: Option<PathBuf>,
    pub allow_tailscale_origins: bool,
}

impl Default for ServeConfig {
    fn default() -> Self {
        let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let run_dir =
            resolve_run_dir_path().unwrap_or_else(|_| workspace.join(".mizar").join("run"));
        Self {
            host: DEFAULT_SERVE_HOST.to_string(),
            port: DEFAULT_SERVE_PORT,
            workspace,
            run_dir,
            session_db_path: resolve_default_session_db_path(),
            allow_tailscale_origins: true,
        }
    }
}

impl ServeConfig {
    pub fn new(workspace: impl Into<PathBuf>, run_dir: impl Into<PathBuf>) -> Self {
        Self {
            host: DEFAULT_SERVE_HOST.to_string(),
            port: DEFAULT_SERVE_PORT,
            workspace: workspace.into(),
            run_dir: run_dir.into(),
            session_db_path: None,
            allow_tailscale_origins: true,
        }
    }

    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_session_db_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.session_db_path = Some(path.into());
        self
    }

    pub fn with_allow_tailscale_origins(mut self, allow: bool) -> Self {
        self.allow_tailscale_origins = allow;
        self
    }
}

/// `mizar serve` 错误类型。
#[derive(Debug, Error)]
pub enum ServeError {
    #[error(
        "mizar serve 仅支持绑定回环地址（127.0.0.1 / ::1 / localhost），拒绝绑定 \"{host}\"；远程访问请通过 Tailscale Serve（loopback only; remote access must go through Tailscale Serve）"
    )]
    NonLoopbackHost { host: String },
    #[error("无效的 serve 参数: {0}")]
    InvalidArgs(String),
    #[error("运行目录初始化失败: {0}")]
    RunDir(#[from] std::io::Error),
    #[error("HTTP 服务监听绑定失败 ({addr}): {source}")]
    Bind {
        addr: String,
        source: std::io::Error,
    },
}

/// 严格校验绑定主机名：仅放行 `127.0.0.1`、`::1`、`[::1]` 与 `localhost`。
pub fn validate_loopback_host(host: &str) -> Result<IpAddr, ServeError> {
    let trimmed = host.trim();
    if trimmed == "127.0.0.1" || trimmed.eq_ignore_ascii_case("localhost") {
        return Ok(IpAddr::V4(Ipv4Addr::LOCALHOST));
    }
    if trimmed == "::1" || trimmed == "[::1]" {
        return Ok(IpAddr::V6(Ipv6Addr::LOCALHOST));
    }
    Err(ServeError::NonLoopbackHost {
        host: host.to_string(),
    })
}

/// 严格校验 HTTP `Origin` 请求头：
/// - 放行 `http://127.0.0.1[:port]`、`http://localhost[:port]`、`http://[::1][:port]`；
/// - 当 `allow_tailscale_origins == true` 时，放行 `https://<host>.ts.net[:port]`；
/// - 拒绝包含 `@` userinfo、路径/查询/锚点/空白、`http://*.ts.net` 或伪造后缀（如 `ts.net.evil.com`）的一切来源。
pub fn is_allowed_origin(origin: &str, allow_tailscale_origins: bool) -> bool {
    if origin.is_empty()
        || origin != origin.trim()
        || origin
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
    {
        return false;
    }

    let Some((scheme, authority)) = origin.split_once("://") else {
        return false;
    };
    if authority.is_empty()
        || authority.contains('@')
        || authority.contains('/')
        || authority.contains('?')
        || authority.contains('#')
        || authority.contains('\\')
    {
        return false;
    }

    let Some((host, _port)) = split_authority_host_port(authority) else {
        return false;
    };

    if scheme.eq_ignore_ascii_case("http") {
        return host == "127.0.0.1" || host.eq_ignore_ascii_case("localhost") || host == "[::1]";
    }
    if scheme.eq_ignore_ascii_case("https") && allow_tailscale_origins {
        return is_valid_tailscale_hostname(host);
    }
    false
}

fn split_authority_host_port(authority: &str) -> Option<(&str, Option<u16>)> {
    if authority.starts_with('[') {
        let close = authority.find(']')?;
        let host = &authority[..=close];
        let rest = &authority[close + 1..];
        if rest.is_empty() {
            return Some((host, None));
        }
        let port_str = rest.strip_prefix(':')?;
        let port = parse_valid_port(port_str)?;
        return Some((host, Some(port)));
    }
    if authority.contains(']') {
        return None;
    }
    if let Some((host, port_str)) = authority.split_once(':') {
        if host.is_empty() || host.contains(':') {
            return None;
        }
        let port = parse_valid_port(port_str)?;
        return Some((host, Some(port)));
    }
    Some((authority, None))
}

fn parse_valid_port(raw: &str) -> Option<u16> {
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    raw.parse::<u16>().ok().filter(|&p| p > 0)
}

fn is_valid_tailscale_hostname(host: &str) -> bool {
    let lower = host.to_ascii_lowercase();
    let Some(prefix) = lower.strip_suffix(".ts.net") else {
        return false;
    };
    if prefix.is_empty() || prefix.starts_with('.') || prefix.ends_with('.') {
        return false;
    }
    prefix.split('.').all(is_valid_dns_label)
}

fn is_valid_dns_label(label: &str) -> bool {
    if label.is_empty() || label.len() > 63 {
        return false;
    }
    let bytes = label.as_bytes();
    if bytes.first() == Some(&b'-') || bytes.last() == Some(&b'-') {
        return false;
    }
    bytes
        .iter()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-')
}

struct ServeState {
    config: ServeConfig,
    runtime: Arc<MizarRuntime>,
    headless_actor: AsyncMutex<Option<SessionActorHandle>>,
    shutdown_cancel: CancellationToken,
}

impl ServeState {
    async fn shutdown_headless_actor(&self) {
        let actor = self.headless_actor.lock().await.take();
        if let Some(handle) = actor {
            handle.shutdown().await;
        }
    }

    /// 懒加载或切换到目标静态会话对应的进程内无头 `SessionActor`（附带 `ProcessRegistration` 参与单写者锁）。
    async fn ensure_headless_actor_for(
        &self,
        target_sid: Option<&str>,
        need_active: bool,
        allow_child_without_switch: bool,
    ) -> Result<SessionActorHandle, JsonRpcError> {
        let mut guard = self.headless_actor.lock().await;
        if let Some(actor) = guard.as_ref() {
            if need_active && let Some(sid) = target_sid {
                let active_sid = actor.active_session_id();
                if active_sid != sid {
                    if allow_child_without_switch
                        && is_subagent_child_of(&self.runtime, sid, &active_sid)
                    {
                        return Ok(actor.clone());
                    }
                    actor
                        .switch_session(sid)
                        .await
                        .map_err(|e| e.to_jsonrpc_error())?;
                }
            } else if let Some(sid) = target_sid
                && self.runtime.sessions.get(sid).is_none()
            {
                return Err(JsonRpcError::session_not_found(sid));
            }
            return Ok(actor.clone());
        }

        let session = match target_sid {
            Some(sid) => self
                .runtime
                .sessions
                .get(sid)
                .ok_or_else(|| JsonRpcError::session_not_found(sid))?,
            None => self.runtime.sessions.create_with(SessionCreateOptions {
                cwd: Some(self.config.workspace.display().to_string()),
                origin: Some(SessionOrigin::User),
                ..Default::default()
            }),
        };

        check_session_writable(&self.config.run_dir, &session.id, std::process::id())
            .map_err(|e| ActorError::SessionLocked(e).to_jsonrpc_error())?;

        let (provider, model, reasoning_effort) =
            resolve_initial_actor_model(&self.runtime, &session);

        let meta = ProcessMeta::for_current_process(
            &self.config.run_dir,
            &session.id,
            self.config.workspace.display().to_string(),
            provider.clone(),
            model.clone(),
        );
        let registration = ProcessRegistration::create(&self.config.run_dir, meta)
            .map_err(|e| JsonRpcError::internal_error(format!("进程元数据注册失败: {e}")))?;

        let mut actor_cfg = SessionActorConfig::new(&self.config.workspace)
            .with_run_dir(&self.config.run_dir)
            .with_provider_model(provider, model, reasoning_effort)
            .with_registration(registration);
        actor_cfg.session_db_path = self.config.session_db_path.clone();

        let actor = SessionActor::spawn(Arc::clone(&self.runtime), session, actor_cfg);
        *guard = Some(actor.clone());
        Ok(actor)
    }
}

fn is_subagent_child_of(runtime: &MizarRuntime, child_sid: &str, parent_sid: &str) -> bool {
    runtime
        .sessions
        .inspect_header(child_sid)
        .is_some_and(|h| h.is_subagent_child() && h.parent_id.as_deref() == Some(parent_sid))
}

fn resolve_initial_actor_model(
    runtime: &MizarRuntime,
    session: &mizar_session::Session,
) -> (
    Option<String>,
    Option<String>,
    Option<mizar_llm::ReasoningEffort>,
) {
    if let Some(sel) = session.latest_selection() {
        let mut provider = sel.provider;
        let mut model = sel.model;
        if provider.is_none() {
            let (def_p, def_m) = default_provider_and_model(runtime);
            provider = def_p;
            if model.is_none() {
                model = def_m;
            }
        }
        return (provider, model, sel.reasoning_effort);
    }
    let (provider, model) = default_provider_and_model(runtime);
    (provider, model, None)
}

fn default_provider_and_model(runtime: &MizarRuntime) -> (Option<String>, Option<String>) {
    let providers = runtime.llm.provider_infos();
    let provider = runtime
        .llm
        .default_provider()
        .or_else(|| providers.first().map(|p| p.id.clone()));
    let model = provider.as_deref().and_then(|pid| {
        providers
            .iter()
            .find(|p| p.id == pid)
            .and_then(|p| p.models.first().map(|m| m.id.clone()))
    });
    (provider, model)
}

/// 已启动的 `mizar serve` 服务句柄，提供实际绑定地址查询与优雅停机。
pub struct ServeHandle {
    addr: SocketAddr,
    shutdown_cancel: CancellationToken,
    server_task: Option<JoinHandle<()>>,
    state: Arc<ServeState>,
}

impl ServeHandle {
    /// 返回 HTTP/WebSocket 网关实际绑定的本地回环地址（传入 `port: 0` 时可获取动态端口）。
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// 优雅停止网关、关闭活跃 WebSocket 连接并清理懒加载的 `SessionActor` 与 `ProcessRegistration`。
    pub async fn shutdown(mut self) {
        self.shutdown_cancel.cancel();
        if let Some(task) = self.server_task.take() {
            let _ = task.await;
        }
        self.state.shutdown_headless_actor().await;
    }
}

impl Drop for ServeHandle {
    fn drop(&mut self) {
        self.shutdown_cancel.cancel();
        if let Some(task) = self.server_task.take() {
            task.abort();
        }
    }
}

/// 启动 `mizar serve` HTTP/WebSocket 网关，返回持有运行状态与优雅关闭能力的服务句柄。
pub async fn start_serve(
    config: ServeConfig,
    runtime: Arc<MizarRuntime>,
) -> Result<ServeHandle, BoxError> {
    let ip = validate_loopback_host(&config.host)?;
    ensure_run_dir(&config.run_dir)?;

    let bind_addr = SocketAddr::new(ip, config.port);
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .map_err(|source| ServeError::Bind {
            addr: bind_addr.to_string(),
            source,
        })?;
    let addr = listener.local_addr()?;

    let shutdown_cancel = CancellationToken::new();
    let state = Arc::new(ServeState {
        config,
        runtime,
        headless_actor: AsyncMutex::new(None),
        shutdown_cancel: shutdown_cancel.clone(),
    });

    let protected_routes = Router::new()
        .route("/api/processes", get(api_processes_handler))
        .route("/api/sessions", get(api_sessions_handler))
        .route("/ws", get(ws_handler))
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            validate_origin_middleware,
        ));

    let app = Router::new()
        .route("/health", get(health_handler))
        .merge(protected_routes)
        .with_state(Arc::clone(&state));

    let server_cancel = shutdown_cancel.clone();
    let server_task = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app)
            .with_graceful_shutdown(server_cancel.cancelled_owned())
            .await
        {
            tracing::warn!(error = %err, "mizar serve HTTP 网关退出异常");
        }
    });

    Ok(ServeHandle {
        addr,
        shutdown_cancel,
        server_task: Some(server_task),
        state,
    })
}

async fn validate_origin_middleware(
    State(state): State<Arc<ServeState>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    let mut origin_values = headers.get_all(header::ORIGIN).iter();
    let Some(first) = origin_values.next() else {
        return next.run(request).await;
    };
    if origin_values.next().is_some() {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({ "error": "forbidden: duplicate Origin header" })),
        )
            .into_response();
    }
    let allowed = first
        .to_str()
        .is_ok_and(|raw| is_allowed_origin(raw, state.config.allow_tailscale_origins));
    if !allowed {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({ "error": "forbidden origin" })),
        )
            .into_response();
    }
    next.run(request).await
}

async fn health_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({
            "ok": true,
            "version": env!("CARGO_PKG_VERSION"),
        })),
    )
}

#[derive(Debug, Serialize)]
struct ApiProcessesResponse {
    processes: Vec<ProcessMeta>,
}

async fn api_processes_handler(State(state): State<Arc<ServeState>>) -> Response {
    match scan_live_processes(&state.config.run_dir) {
        Ok(list) => {
            let processes: Vec<ProcessMeta> =
                list.into_iter().map(|m| m.without_secret()).collect();
            (StatusCode::OK, Json(ApiProcessesResponse { processes })).into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("扫描活进程失败: {err}") })),
        )
            .into_response(),
    }
}

#[derive(Debug, Default, Deserialize)]
struct ApiSessionsQuery {
    cwd: Option<String>,
    limit: Option<usize>,
    exclude_subagents: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct ApiSessionSummary {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fork_seq: Option<u64>,
    event_count: u64,
    created_at_ms: u64,
    updated_at_ms: u64,
    updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    live_pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_writer_pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    socket: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

#[derive(Debug, Serialize)]
struct ApiSessionsResponse {
    sessions: Vec<ApiSessionSummary>,
}

async fn api_sessions_handler(
    State(state): State<Arc<ServeState>>,
    Query(query): Query<ApiSessionsQuery>,
) -> Response {
    let headers = match state
        .runtime
        .sessions
        .list_headers_async(SessionListQuery {
            cwd: query.cwd.clone(),
            limit: None,
            exclude_subagents: query.exclude_subagents.unwrap_or(true),
        })
        .await
    {
        Ok(headers) => headers,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("读取会话列表失败: {err}") })),
            )
                .into_response();
        }
    };

    let live_procs = scan_live_processes(&state.config.run_dir).unwrap_or_default();
    let mut live_by_sid: HashMap<String, ProcessMeta> = live_procs
        .into_iter()
        .map(|p| (p.session_id.clone(), p.without_secret()))
        .collect();

    let mut branch_cache: HashMap<String, Option<String>> = HashMap::new();
    let mut sessions = Vec::with_capacity(headers.len() + live_by_sid.len());

    for header in headers {
        let live = live_by_sid.remove(&header.id);
        let cwd = header
            .cwd
            .clone()
            .or_else(|| live.as_ref().map(|p| p.cwd.clone()));
        let branch = resolve_branch_cached(&mut branch_cache, cwd.as_deref());
        let created_at_ms = u64::try_from(header.created_at_ms).unwrap_or(u64::MAX);
        let updated_at_ms = u64::try_from(header.updated_at_ms).unwrap_or(u64::MAX);
        sessions.push(ApiSessionSummary {
            id: header.id,
            cwd,
            title: header.title,
            parent_id: header.parent_id,
            fork_seq: header.fork_seq,
            event_count: header.event_count,
            created_at_ms,
            updated_at_ms,
            updated_at: format_updated_at_iso(updated_at_ms),
            branch,
            live_pid: live.as_ref().map(|p| p.pid),
            active_writer_pid: live.as_ref().map(|p| p.pid),
            socket: live.as_ref().map(|p| p.socket.clone()),
            provider: live.as_ref().and_then(|p| p.provider.clone()),
            model: live.and_then(|p| p.model),
        });
    }

    for (_, live) in live_by_sid {
        if query
            .cwd
            .as_ref()
            .is_some_and(|c| live.cwd.as_str() != c.as_str())
        {
            continue;
        }
        let branch = resolve_branch_cached(&mut branch_cache, Some(&live.cwd));
        sessions.push(ApiSessionSummary {
            id: live.session_id,
            cwd: Some(live.cwd),
            title: None,
            parent_id: None,
            fork_seq: None,
            event_count: 0,
            created_at_ms: live.started_at_ms,
            updated_at_ms: live.started_at_ms,
            updated_at: format_updated_at_iso(live.started_at_ms),
            branch,
            live_pid: Some(live.pid),
            active_writer_pid: Some(live.pid),
            socket: Some(live.socket),
            provider: live.provider,
            model: live.model,
        });
    }

    sessions.sort_by(|a, b| {
        b.updated_at_ms
            .cmp(&a.updated_at_ms)
            .then_with(|| a.id.cmp(&b.id))
    });
    if let Some(limit) = query.limit {
        sessions.truncate(limit);
    }

    (StatusCode::OK, Json(ApiSessionsResponse { sessions })).into_response()
}

fn resolve_branch_cached(
    cache: &mut HashMap<String, Option<String>>,
    cwd: Option<&str>,
) -> Option<String> {
    let cwd = cwd?.trim();
    if cwd.is_empty() {
        return None;
    }
    if let Some(cached) = cache.get(cwd) {
        return cached.clone();
    }
    let branch = resolve_git_branch(Path::new(cwd));
    cache.insert(cwd.to_string(), branch.clone());
    branch
}

fn resolve_git_branch(dir: &Path) -> Option<String> {
    let mut cur = Some(dir);
    while let Some(path) = cur {
        let dot_git = path.join(".git");
        if dot_git.is_dir() {
            return parse_git_head_file(&dot_git.join("HEAD"));
        }
        if dot_git.is_file() {
            let content = std::fs::read_to_string(&dot_git).ok()?;
            let gitdir_raw = content.strip_prefix("gitdir:")?.trim();
            let gitdir = if Path::new(gitdir_raw).is_absolute() {
                PathBuf::from(gitdir_raw)
            } else {
                path.join(gitdir_raw)
            };
            return parse_git_head_file(&gitdir.join("HEAD"));
        }
        cur = path.parent();
    }
    None
}

fn parse_git_head_file(head_path: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(head_path).ok()?;
    let trimmed = raw.trim();
    if let Some(branch) = trimmed.strip_prefix("ref: refs/heads/") {
        let name = branch.trim();
        return (!name.is_empty()).then(|| name.to_string());
    }
    None
}

fn format_updated_at_iso(ms: u64) -> String {
    i64::try_from(ms)
        .ok()
        .filter(|&v| v > 0)
        .and_then(DateTime::from_timestamp_millis)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_else(|| "-".to_string())
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServeState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_connection(socket, state))
}

#[cfg(unix)]
struct ExternalUnixProxy {
    socket_path: String,
    writer_tx: mpsc::UnboundedSender<String>,
    proxy_cancel: CancellationToken,
    task: JoinHandle<()>,
}

#[cfg(unix)]
impl Drop for ExternalUnixProxy {
    fn drop(&mut self) {
        self.proxy_cancel.cancel();
        self.task.abort();
    }
}

struct InprocSubActivation {
    session_id: String,
    dedup_seq: Option<u64>,
    notif_rx: broadcast::Receiver<RpcNotification>,
}

enum InprocSubCmd {
    Activate(InprocSubActivation),
    Clear,
}

struct WsConnState {
    bound_session_id: Option<String>,
    subscribed_pid: Option<u32>,
    #[cfg(unix)]
    proxies: HashMap<u32, ExternalUnixProxy>,
}

impl WsConnState {
    fn new() -> Self {
        Self {
            bound_session_id: None,
            subscribed_pid: None,
            #[cfg(unix)]
            proxies: HashMap::new(),
        }
    }

    fn clear_subscribed_proxy(&mut self, keep_pid: Option<u32>) {
        #[cfg(unix)]
        if let Some(prev_pid) = self.subscribed_pid.take()
            && Some(prev_pid) != keep_pid
        {
            self.proxies.remove(&prev_pid);
        }
        #[cfg(not(unix))]
        let _ = keep_pid;
    }
}

async fn handle_ws_connection(socket: WebSocket, state: Arc<ServeState>) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let (outbound_tx, mut outbound_rx) = mpsc::channel::<Message>(DEFAULT_WS_OUTBOUND_CAPACITY);
    let (inproc_sub_tx, inproc_sub_rx) = mpsc::unbounded_channel::<InprocSubCmd>();
    let conn_cancel = CancellationToken::new();

    let writer_cancel = conn_cancel.clone();
    let shutdown_cancel_w = state.shutdown_cancel.clone();
    let writer_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = shutdown_cancel_w.cancelled() => break,
                _ = writer_cancel.cancelled() => break,
                maybe_msg = outbound_rx.recv() => {
                    let Some(msg) = maybe_msg else {
                        let _ = ws_tx.close().await;
                        break;
                    };
                    if ws_tx.send(msg).await.is_err() {
                        writer_cancel.cancel();
                        break;
                    }
                }
            }
        }
    });

    let notifier_task = tokio::spawn(run_ws_inproc_notifier_loop(
        inproc_sub_rx,
        outbound_tx.clone(),
        conn_cancel.clone(),
    ));

    let mut conn_state = WsConnState::new();

    loop {
        let frame = tokio::select! {
            biased;
            _ = state.shutdown_cancel.cancelled() => break,
            _ = conn_cancel.cancelled() => break,
            next = ws_rx.next() => next,
        };
        let Some(frame_res) = frame else {
            break;
        };
        let Ok(msg) = frame_res else {
            break;
        };

        match msg {
            Message::Text(text) => {
                if !handle_ws_text_frame(
                    text.as_str(),
                    &state,
                    &mut conn_state,
                    &outbound_tx,
                    &inproc_sub_tx,
                    &conn_cancel,
                )
                .await
                {
                    break;
                }
            }
            Message::Binary(_) => {
                let resp = JsonRpcResponse::err(
                    None,
                    JsonRpcError::invalid_request(
                        "binary WebSocket frames are not supported; send JSON-RPC text frames",
                    ),
                );
                if !send_ws_response(&outbound_tx, &resp).await {
                    break;
                }
            }
            Message::Ping(_) | Message::Pong(_) => {}
            Message::Close(_) => break,
        }
    }

    conn_cancel.cancel();
    drop(inproc_sub_tx);
    drop(conn_state);
    let _ = notifier_task.await;
    let _ = writer_task.await;
}

async fn handle_ws_text_frame(
    raw_text: &str,
    state: &Arc<ServeState>,
    conn_state: &mut WsConnState,
    outbound_tx: &mpsc::Sender<Message>,
    inproc_sub_tx: &mpsc::UnboundedSender<InprocSubCmd>,
    conn_cancel: &CancellationToken,
) -> bool {
    let trimmed = raw_text.trim();
    if trimmed.is_empty() {
        return true;
    }

    let raw_value: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(err) => {
            let resp = JsonRpcResponse::err(None, JsonRpcError::parse_error(err.to_string()));
            return send_ws_response(outbound_tx, &resp).await;
        }
    };

    let extracted_id = raw_value
        .get("id")
        .and_then(|v| serde_json::from_value::<JsonRpcId>(v.clone()).ok());
    let req: JsonRpcRequest = match serde_json::from_value(raw_value.clone()) {
        Ok(r) => r,
        Err(err) => {
            let resp =
                JsonRpcResponse::err(extracted_id, JsonRpcError::invalid_request(err.to_string()));
            return send_ws_response(outbound_tx, &resp).await;
        }
    };

    if !req.is_valid_version() {
        let resp = JsonRpcResponse::err(
            req.id,
            JsonRpcError::invalid_request(format!("unsupported jsonrpc version: {}", req.jsonrpc)),
        );
        return send_ws_response(outbound_tx, &resp).await;
    }

    if req.method == METHOD_AUTH_HELLO {
        if let Some(id) = req.id {
            let resp =
                JsonRpcResponse::err(Some(id), JsonRpcError::method_not_found(METHOD_AUTH_HELLO));
            return send_ws_response(outbound_tx, &resp).await;
        }
        return true;
    }

    if req.method == METHOD_SESSION_LIST {
        let outcome = handle_ws_session_list(state, req.params).await;
        return emit_rpc_outcome(outbound_tx, req.id, outcome).await;
    }

    let explicit_sid = req
        .params
        .as_ref()
        .and_then(|p| p.get("session_id"))
        .and_then(|v| v.as_str())
        .map(str::to_string);

    #[cfg(unix)]
    if explicit_sid.is_none()
        && matches!(
            req.method.as_str(),
            METHOD_INTERACTION_RESOLVE | METHOD_CAPABILITIES_LIST
        )
        && let Some(active_pid) = conn_state.subscribed_pid
        && let Some(proxy) = conn_state.proxies.get(&active_pid)
        && !proxy.writer_tx.is_closed()
        && is_pid_alive(active_pid)
        && let Ok(mut line) = serde_json::to_string(&raw_value)
    {
        line.push('\n');
        if proxy.writer_tx.send(line).is_ok() {
            return true;
        }
        conn_state.proxies.remove(&active_pid);
    }

    let effective_sid = explicit_sid.clone().or_else(|| {
        if matches!(
            req.method.as_str(),
            METHOD_INTERACTION_RESOLVE | METHOD_CAPABILITIES_LIST
        ) {
            conn_state.bound_session_id.clone()
        } else {
            None
        }
    });

    if let Some(ref sid) = effective_sid
        && let Some(ext_meta) = find_external_live_owner(&state.config.run_dir, sid)
    {
        return forward_to_external_process(
            state,
            conn_state,
            outbound_tx,
            inproc_sub_tx,
            conn_cancel,
            &ext_meta,
            &req,
            &raw_value,
            sid,
        )
        .await;
    }

    dispatch_via_headless_actor(
        state,
        conn_state,
        outbound_tx,
        inproc_sub_tx,
        req,
        effective_sid,
    )
    .await
}

fn find_external_live_owner(run_dir: &Path, session_id: &str) -> Option<ProcessMeta> {
    let self_pid = std::process::id();
    scan_live_processes(run_dir)
        .ok()?
        .into_iter()
        .find(|m| m.session_id == session_id && m.pid != self_pid)
}

#[allow(clippy::too_many_arguments)]
async fn forward_to_external_process(
    state: &Arc<ServeState>,
    conn_state: &mut WsConnState,
    outbound_tx: &mpsc::Sender<Message>,
    inproc_sub_tx: &mpsc::UnboundedSender<InprocSubCmd>,
    conn_cancel: &CancellationToken,
    ext_meta: &ProcessMeta,
    req: &JsonRpcRequest,
    raw_value: &Value,
    target_sid: &str,
) -> bool {
    #[cfg(unix)]
    {
        let _ = inproc_sub_tx.send(InprocSubCmd::Clear);
        if req.method == METHOD_SESSION_SUBSCRIBE {
            conn_state.clear_subscribed_proxy(Some(ext_meta.pid));
            conn_state.bound_session_id = Some(target_sid.to_string());
            conn_state.subscribed_pid = Some(ext_meta.pid);
        }

        let need_new = conn_state
            .proxies
            .get(&ext_meta.pid)
            .is_none_or(|p| p.socket_path != ext_meta.socket || p.writer_tx.is_closed());

        if need_new {
            match UnixStream::connect(&ext_meta.socket).await {
                Ok(stream) => {
                    let proxy = spawn_external_unix_proxy(
                        ext_meta.socket.clone(),
                        stream,
                        outbound_tx.clone(),
                        conn_cancel.clone(),
                    );
                    conn_state.proxies.insert(ext_meta.pid, proxy);
                }
                Err(err) => {
                    if !is_pid_alive(ext_meta.pid) {
                        let _ = scan_live_processes(&state.config.run_dir);
                        return dispatch_via_headless_actor(
                            state,
                            conn_state,
                            outbound_tx,
                            inproc_sub_tx,
                            req.clone(),
                            Some(target_sid.to_string()),
                        )
                        .await;
                    }
                    if let Some(id) = req.id.clone() {
                        let resp = JsonRpcResponse::err(
                            Some(id),
                            JsonRpcError::internal_error(format!(
                                "连接活跃进程 Unix Socket 失败 ({}): {err}",
                                ext_meta.socket
                            )),
                        );
                        return send_ws_response(outbound_tx, &resp).await;
                    }
                    return true;
                }
            }
        }

        let line = match serde_json::to_string(raw_value) {
            Ok(mut s) => {
                s.push('\n');
                s
            }
            Err(err) => {
                if let Some(id) = req.id.clone() {
                    let resp = JsonRpcResponse::err(
                        Some(id),
                        JsonRpcError::internal_error(err.to_string()),
                    );
                    return send_ws_response(outbound_tx, &resp).await;
                }
                return true;
            }
        };

        if let Some(proxy) = conn_state.proxies.get(&ext_meta.pid)
            && proxy.writer_tx.send(line).is_ok()
        {
            return true;
        }
        conn_state.proxies.remove(&ext_meta.pid);
        if let Some(id) = req.id.clone() {
            let resp = JsonRpcResponse::err(
                Some(id),
                JsonRpcError::internal_error("活跃进程代理连接已断开"),
            );
            return send_ws_response(outbound_tx, &resp).await;
        }
        true
    }
    #[cfg(not(unix))]
    {
        let _ = (
            state,
            conn_state,
            inproc_sub_tx,
            conn_cancel,
            ext_meta,
            raw_value,
            target_sid,
        );
        if let Some(id) = req.id.clone() {
            let resp = JsonRpcResponse::err(
                Some(id),
                JsonRpcError::internal_error("当前平台不支持 Unix Socket 进程代理"),
            );
            return send_ws_response(outbound_tx, &resp).await;
        }
        true
    }
}

#[cfg(unix)]
fn spawn_external_unix_proxy(
    socket_path: String,
    stream: UnixStream,
    ws_outbound_tx: mpsc::Sender<Message>,
    conn_cancel: CancellationToken,
) -> ExternalUnixProxy {
    let (read_half, mut write_half) = stream.into_split();
    let (writer_tx, mut writer_rx) = mpsc::unbounded_channel::<String>();
    let proxy_cancel = conn_cancel.child_token();

    let read_cancel = proxy_cancel.clone();
    let write_cancel = proxy_cancel.clone();

    let task = tokio::spawn(async move {
        let reader_fut = async {
            let mut reader = BufReader::new(read_half);
            let mut line = String::new();
            loop {
                line.clear();
                let res = tokio::select! {
                    biased;
                    _ = read_cancel.cancelled() => break,
                    res = reader.read_line(&mut line) => res,
                };
                match res {
                    Ok(0) | Err(_) => {
                        read_cancel.cancel();
                        break;
                    }
                    Ok(_) => {
                        let clean = line.trim_end_matches(['\r', '\n']);
                        if clean.is_empty() {
                            continue;
                        }
                        if ws_outbound_tx
                            .send(Message::Text(clean.to_string().into()))
                            .await
                            .is_err()
                        {
                            read_cancel.cancel();
                            break;
                        }
                    }
                }
            }
        };

        let writer_fut = async {
            loop {
                let maybe_line = tokio::select! {
                    biased;
                    _ = write_cancel.cancelled() => break,
                    msg = writer_rx.recv() => msg,
                };
                let Some(line) = maybe_line else {
                    let _ = write_half.shutdown().await;
                    break;
                };
                let write_res = tokio::select! {
                    biased;
                    _ = write_cancel.cancelled() => break,
                    res = write_half.write_all(line.as_bytes()) => res,
                };
                if write_res.is_err() {
                    write_cancel.cancel();
                    break;
                }
            }
        };

        tokio::join!(reader_fut, writer_fut);
    });

    ExternalUnixProxy {
        socket_path,
        writer_tx,
        proxy_cancel,
        task,
    }
}

async fn dispatch_via_headless_actor(
    state: &Arc<ServeState>,
    conn_state: &mut WsConnState,
    outbound_tx: &mpsc::Sender<Message>,
    inproc_sub_tx: &mpsc::UnboundedSender<InprocSubCmd>,
    req: JsonRpcRequest,
    effective_sid: Option<String>,
) -> bool {
    if req.method == METHOD_SESSION_SUBSCRIBE {
        let params: SessionSubscribeParams = match parse_required_params(req.params) {
            Ok(p) => p,
            Err(err) => return emit_rpc_outcome(outbound_tx, req.id, Err(err)).await,
        };
        conn_state.clear_subscribed_proxy(None);
        let actor = match state
            .ensure_headless_actor_for(Some(&params.session_id), true, true)
            .await
        {
            Ok(actor) => actor,
            Err(err) => return emit_rpc_outcome(outbound_tx, req.id, Err(err)).await,
        };

        // 先建立广播订阅再取水合快照，且先发响应帧再激活推送循环，保证无漏帧且响应先于通知
        let notif_rx = actor.subscribe_notifications();
        match actor.subscribe_snapshot_with(params.clone()).await {
            Ok(snapshot) => {
                let dedup_seq = snapshot.events.last().map(|e| e.seq).or(params.after_seq);
                let sub_sid = snapshot.session.id.clone();
                conn_state.bound_session_id = Some(sub_sid.clone());
                if let Some(id) = req.id {
                    let resp = ok_response(id, &snapshot);
                    if !send_ws_response(outbound_tx, &resp).await {
                        return false;
                    }
                }
                let _ = inproc_sub_tx.send(InprocSubCmd::Activate(InprocSubActivation {
                    session_id: sub_sid,
                    dedup_seq,
                    notif_rx,
                }));
                return true;
            }
            Err(err) => {
                return emit_rpc_outcome(outbound_tx, req.id, Err(err.to_jsonrpc_error())).await;
            }
        }
    }

    let need_active = matches!(
        req.method.as_str(),
        METHOD_TURN_SUBMIT | METHOD_SESSION_CONFIG
    );
    let actor = match state
        .ensure_headless_actor_for(effective_sid.as_deref(), need_active, false)
        .await
    {
        Ok(actor) => actor,
        Err(err) => return emit_rpc_outcome(outbound_tx, req.id, Err(err)).await,
    };

    if need_active && let Some(sid) = effective_sid {
        conn_state.bound_session_id = Some(sid);
    }

    let outcome = dispatch_actor_method(&actor, &req.method, req.params).await;
    emit_rpc_outcome(outbound_tx, req.id, outcome).await
}

async fn dispatch_actor_method(
    handle: &SessionActorHandle,
    method: &str,
    params: Option<Value>,
) -> Result<Value, JsonRpcError> {
    match method {
        METHOD_TURN_SUBMIT => {
            let p: TurnSubmitParams = parse_required_params(params)?;
            to_value_result(handle.submit_turn(p).await)
        }
        METHOD_TURN_CANCEL => {
            let p: TurnCancelParams = parse_required_params(params)?;
            to_value_result(handle.cancel_turn(&p.session_id).await)
        }
        METHOD_QUEUE_REMOVE => {
            let p: QueueRemoveParams = parse_required_params(params)?;
            to_value_result(handle.queue_remove(&p.session_id, p.id).await)
        }
        METHOD_QUEUE_POP_ALL => {
            let p: QueuePopAllParams = parse_required_params(params)?;
            to_value_result(handle.queue_pop_all(&p.session_id).await)
        }
        METHOD_INTERACTION_RESOLVE => {
            let p: InteractionResolveParams = parse_required_params(params)?;
            to_value_result(handle.resolve_interaction(p, ResolvedBy::Rpc).await)
        }
        METHOD_SESSION_CONFIG => {
            let p: SessionConfigParams = parse_required_params(params)?;
            to_value_result(handle.update_config(p).await)
        }
        METHOD_SESSION_EVENTS => {
            let p: SessionEventsParams = parse_required_params(params)?;
            to_value_result(handle.list_events(p).await)
        }
        METHOD_SESSION_FORK => {
            let p: SessionForkParams = parse_required_params(params)?;
            to_value_result(handle.fork_session(p).await)
        }
        METHOD_SESSION_TREE => {
            let p: SessionTreeParams = parse_required_params(params)?;
            to_value_result(handle.session_tree(&p.session_id).await)
        }
        METHOD_CAPABILITIES_LIST => {
            let p: CapabilitiesListParams = parse_optional_params(params)?;
            to_value_result(handle.list_capabilities(p.session_id.as_deref()).await)
        }
        other => Err(JsonRpcError::method_not_found(other)),
    }
}

async fn handle_ws_session_list(
    state: &Arc<ServeState>,
    params: Option<Value>,
) -> Result<Value, JsonRpcError> {
    let params: SessionListParams = parse_optional_params(params)?;
    let query_norm = params
        .query
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase);

    let parsed_cursor = if let Some(ref raw) = params.cursor {
        Some(parse_session_cursor(raw)?)
    } else {
        None
    };

    let headers = state
        .runtime
        .sessions
        .list_headers_async(SessionListQuery {
            cwd: params.cwd.clone(),
            limit: None,
            exclude_subagents: params.exclude_subagents.unwrap_or(true),
        })
        .await
        .map_err(|e| JsonRpcError::internal_error(e.to_string()))?;

    let live_procs = scan_live_processes(&state.config.run_dir).unwrap_or_default();
    let writer_pids: HashMap<String, u32> = live_procs
        .iter()
        .map(|m| (m.session_id.clone(), m.pid))
        .collect();
    let seen_ids: HashSet<String> = headers.iter().map(|h| h.id.clone()).collect();

    let mut merged: Vec<SessionListItemDto> = headers
        .into_iter()
        .map(|h| {
            let active_writer_pid = writer_pids.get(&h.id).copied();
            SessionListItemDto {
                id: h.id,
                cwd: h.cwd,
                title: h.title,
                parent_id: h.parent_id,
                fork_seq: h.fork_seq,
                event_count: h.event_count,
                created_at_ms: u64::try_from(h.created_at_ms).unwrap_or(u64::MAX),
                updated_at_ms: u64::try_from(h.updated_at_ms).unwrap_or(u64::MAX),
                active_writer_pid,
            }
        })
        .collect();

    for live in live_procs {
        if seen_ids.contains(&live.session_id) {
            continue;
        }
        if params
            .cwd
            .as_ref()
            .is_some_and(|c| live.cwd.as_str() != c.as_str())
        {
            continue;
        }
        merged.push(SessionListItemDto {
            id: live.session_id,
            cwd: Some(live.cwd),
            title: None,
            parent_id: None,
            fork_seq: None,
            event_count: 0,
            created_at_ms: live.started_at_ms,
            updated_at_ms: live.started_at_ms,
            active_writer_pid: Some(live.pid),
        });
    }

    merged.sort_by(|a, b| {
        b.updated_at_ms
            .cmp(&a.updated_at_ms)
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut filtered: Vec<SessionListItemDto> = merged
        .into_iter()
        .filter(|item| {
            if let Some(ref q) = query_norm {
                let id_hit = item.id.to_ascii_lowercase().contains(q);
                let title_hit = item
                    .title
                    .as_deref()
                    .is_some_and(|t| t.to_ascii_lowercase().contains(q));
                let cwd_hit = item
                    .cwd
                    .as_deref()
                    .is_some_and(|c| c.to_ascii_lowercase().contains(q));
                if !(id_hit || title_hit || cwd_hit) {
                    return false;
                }
            }
            if let Some((cursor_ts, ref cursor_id)) = parsed_cursor {
                let ts = u128::from(item.updated_at_ms);
                return ts < cursor_ts
                    || (ts == cursor_ts && item.id.as_str() > cursor_id.as_str());
            }
            true
        })
        .collect();

    let limit = params.limit.unwrap_or(DEFAULT_SESSION_LIST_LIMIT);
    let next_cursor = if limit == 0 {
        filtered.clear();
        None
    } else if filtered.len() > limit {
        filtered.truncate(limit);
        filtered
            .last()
            .map(|last| format!("{}:{}", last.updated_at_ms, last.id))
    } else {
        None
    };

    serde_json::to_value(SessionListResult {
        sessions: filtered,
        next_cursor,
    })
    .map_err(|e| JsonRpcError::internal_error(e.to_string()))
}

fn parse_session_cursor(raw: &str) -> Result<(u128, String), JsonRpcError> {
    let Some((ts_str, id)) = raw.split_once(':') else {
        return Err(JsonRpcError::invalid_params(format!(
            "无效的会话分页游标 `{raw}`"
        )));
    };
    let ts = ts_str
        .parse::<u128>()
        .map_err(|_| JsonRpcError::invalid_params(format!("无效的游标时间戳 `{ts_str}`")))?;
    if id.is_empty() {
        return Err(JsonRpcError::invalid_params(format!(
            "无效的会话分页游标 `{raw}`"
        )));
    }
    Ok((ts, id.to_string()))
}

fn ok_response<T: Serialize>(id: JsonRpcId, value: &T) -> JsonRpcResponse {
    match JsonRpcResponse::ok_typed(id.clone(), value) {
        Ok(resp) => resp,
        Err(err) => JsonRpcResponse::err(Some(id), JsonRpcError::internal_error(err.to_string())),
    }
}

fn parse_required_params<T: DeserializeOwned>(params: Option<Value>) -> Result<T, JsonRpcError> {
    let value = params.ok_or_else(|| JsonRpcError::invalid_params("missing params"))?;
    serde_json::from_value(value).map_err(|err| JsonRpcError::invalid_params(err.to_string()))
}

fn parse_optional_params<T: DeserializeOwned + Default>(
    params: Option<Value>,
) -> Result<T, JsonRpcError> {
    match params {
        None | Some(Value::Null) => Ok(T::default()),
        Some(value) => serde_json::from_value(value)
            .map_err(|err| JsonRpcError::invalid_params(err.to_string())),
    }
}

fn to_value_result<T: Serialize>(res: Result<T, ActorError>) -> Result<Value, JsonRpcError> {
    match res {
        Ok(val) => {
            serde_json::to_value(val).map_err(|e| JsonRpcError::internal_error(e.to_string()))
        }
        Err(err) => Err(err.to_jsonrpc_error()),
    }
}

async fn emit_rpc_outcome(
    outbound_tx: &mpsc::Sender<Message>,
    id: Option<JsonRpcId>,
    outcome: Result<Value, JsonRpcError>,
) -> bool {
    let Some(id) = id else {
        return true;
    };
    let resp = match outcome {
        Ok(result) => JsonRpcResponse::ok(id, result),
        Err(err) => JsonRpcResponse::err(Some(id), err),
    };
    send_ws_response(outbound_tx, &resp).await
}

async fn send_ws_response(outbound_tx: &mpsc::Sender<Message>, resp: &JsonRpcResponse) -> bool {
    let Ok(json) = serde_json::to_string(resp) else {
        return true;
    };
    outbound_tx.send(Message::Text(json.into())).await.is_ok()
}

async fn recv_inproc_notification(
    sub: Option<&mut InprocSubActivation>,
) -> Result<RpcNotification, broadcast::error::RecvError> {
    match sub {
        Some(active) => active.notif_rx.recv().await,
        None => std::future::pending().await,
    }
}

async fn run_ws_inproc_notifier_loop(
    mut sub_cmd_rx: mpsc::UnboundedReceiver<InprocSubCmd>,
    outbound_tx: mpsc::Sender<Message>,
    conn_cancel: CancellationToken,
) {
    let mut active_sub: Option<InprocSubActivation> = None;
    let mut coalescer =
        WsDeltaCoalescer::new(DEFAULT_COALESCE_WINDOW, DEFAULT_MAX_DELTA_FRAME_BYTES);

    loop {
        let sleep_fut = async {
            match coalescer.deadline() {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending().await,
            }
        };

        tokio::select! {
            biased;
            _ = conn_cancel.cancelled() => break,
            cmd = sub_cmd_rx.recv() => {
                match cmd {
                    Some(InprocSubCmd::Activate(next_sub)) => {
                        coalescer.clear();
                        active_sub = Some(next_sub);
                    }
                    Some(InprocSubCmd::Clear) => {
                        coalescer.clear();
                        active_sub = None;
                    }
                    None => break,
                }
            }
            _ = sleep_fut => {
                if !coalescer.flush_all(&outbound_tx, &conn_cancel) {
                    break;
                }
            }
            recv_res = recv_inproc_notification(active_sub.as_mut()) => {
                match recv_res {
                    Ok(notif) => {
                        let Some(sub) = active_sub.as_mut() else {
                            continue;
                        };
                        if !matches_subscription(&notif, &sub.session_id, sub.dedup_seq) {
                            continue;
                        }
                        if let RpcNotification::EventCommitted(ref ev) = notif {
                            sub.dedup_seq = Some(sub.dedup_seq.map_or(ev.seq, |m| m.max(ev.seq)));
                        }
                        match notif {
                            RpcNotification::StreamDelta(delta) => {
                                if !coalescer.push(delta, &outbound_tx, &conn_cancel) {
                                    break;
                                }
                            }
                            other => {
                                if !coalescer.flush_all(&outbound_tx, &conn_cancel) {
                                    break;
                                }
                                if !enqueue_ws_notification(&other, &outbound_tx, &conn_cancel) {
                                    break;
                                }
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(skipped, "WebSocket 内置会话广播接收滞后，断开连接");
                        conn_cancel.cancel();
                        break;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        let _ = coalescer.flush_all(&outbound_tx, &conn_cancel);
                        active_sub = None;
                    }
                }
            }
        }
    }
}

fn matches_subscription(
    notif: &RpcNotification,
    subscribed_sid: &str,
    dedup_seq: Option<u64>,
) -> bool {
    match notif {
        RpcNotification::TurnState(n) => n.session_id == subscribed_sid,
        RpcNotification::StreamDelta(n) => n.session_id == subscribed_sid,
        RpcNotification::EventCommitted(n) => {
            n.session_id == subscribed_sid && dedup_seq.is_none_or(|max_seq| n.seq > max_seq)
        }
        RpcNotification::InteractionRequested(n) => {
            n.session_id.as_deref().is_none_or(|s| s == subscribed_sid)
        }
        RpcNotification::InteractionResolved(n) => {
            n.session_id.as_deref().is_none_or(|s| s == subscribed_sid)
        }
        RpcNotification::QueueUpdated(n) => n.session_id == subscribed_sid,
        RpcNotification::SubagentUpdated(n) => {
            n.session_id == subscribed_sid || n.task.child_session_id == subscribed_sid
        }
        RpcNotification::SessionSwitched(_) => true,
    }
}

fn enqueue_ws_notification(
    notif: &RpcNotification,
    tx: &mpsc::Sender<Message>,
    conn_cancel: &CancellationToken,
) -> bool {
    let rpc = notif.to_jsonrpc();
    let Ok(json) = serde_json::to_string(&rpc) else {
        return true;
    };
    try_enqueue_ws_text(tx, json, conn_cancel)
}

fn try_enqueue_ws_text(
    tx: &mpsc::Sender<Message>,
    text: String,
    conn_cancel: &CancellationToken,
) -> bool {
    match tx.try_send(Message::Text(text.into())) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(_)) | Err(mpsc::error::TrySendError::Closed(_)) => {
            conn_cancel.cancel();
            false
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeltaKey {
    session_id: String,
    turn: u64,
    step: usize,
    kind: StreamDeltaKind,
    call_id: Option<String>,
}

struct PendingDeltaBucket {
    key: DeltaKey,
    delta: String,
}

struct WsDeltaCoalescer {
    coalesce_window: Duration,
    max_frame_bytes: usize,
    pending: Vec<PendingDeltaBucket>,
    deadline: Option<Instant>,
}

impl WsDeltaCoalescer {
    fn new(coalesce_window: Duration, max_frame_bytes: usize) -> Self {
        Self {
            coalesce_window,
            max_frame_bytes: max_frame_bytes.max(1),
            pending: Vec::new(),
            deadline: None,
        }
    }

    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    fn clear(&mut self) {
        self.pending.clear();
        self.deadline = None;
    }

    fn push(
        &mut self,
        delta: StreamDeltaNotification,
        outbound_tx: &mpsc::Sender<Message>,
        conn_cancel: &CancellationToken,
    ) -> bool {
        if delta.delta.is_empty() {
            return true;
        }
        let key = DeltaKey {
            session_id: delta.session_id,
            turn: delta.turn,
            step: delta.step,
            kind: delta.kind,
            call_id: delta.call_id,
        };
        if let Some(bucket) = self.pending.iter_mut().find(|b| b.key == key) {
            bucket.delta.push_str(&delta.delta);
        } else {
            self.pending.push(PendingDeltaBucket {
                key,
                delta: delta.delta,
            });
            if self.deadline.is_none() {
                self.deadline = Some(Instant::now() + self.coalesce_window);
            }
        }
        let total_approx: usize = self.pending.iter().map(|b| b.delta.len()).sum();
        if total_approx >= self.max_frame_bytes {
            return self.flush_all(outbound_tx, conn_cancel);
        }
        true
    }

    fn flush_all(
        &mut self,
        outbound_tx: &mpsc::Sender<Message>,
        conn_cancel: &CancellationToken,
    ) -> bool {
        self.deadline = None;
        if self.pending.is_empty() {
            return true;
        }
        let buckets = std::mem::take(&mut self.pending);
        for mut bucket in buckets {
            while !bucket.delta.is_empty() {
                let split = find_delta_split(&bucket.delta, self.max_frame_bytes);
                let chunk = bucket.delta[..split].to_string();
                bucket.delta.drain(..split);
                let notif = RpcNotification::StreamDelta(StreamDeltaNotification {
                    session_id: bucket.key.session_id.clone(),
                    turn: bucket.key.turn,
                    step: bucket.key.step,
                    kind: bucket.key.kind,
                    call_id: bucket.key.call_id.clone(),
                    delta: chunk,
                });
                if !enqueue_ws_notification(&notif, outbound_tx, conn_cancel) {
                    return false;
                }
            }
        }
        true
    }
}

fn find_delta_split(text: &str, max_bytes: usize) -> usize {
    if text.len() <= max_bytes {
        return text.len();
    }
    let mut i = max_bytes.max(1).min(text.len());
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    if i == 0 {
        text.chars().next().map_or(0, char::len_utf8)
    } else {
        i
    }
}

#[derive(Debug, Default)]
struct ServeCliParsed {
    host: Option<String>,
    port: Option<u16>,
    run_dir: Option<PathBuf>,
    workspace: Option<PathBuf>,
    session_db: Option<PathBuf>,
    help: bool,
}

fn format_serve_help_text() -> &'static str {
    "\
Usage:
  mizar serve [OPTIONS]

Options:
      --host <host>           Loopback bind address (127.0.0.1, ::1, localhost; default: 127.0.0.1)
      --port <port>           TCP listen port (default: 8787)
  -C, --workspace <dir>       Workspace directory (defaults to current directory)
      --run-dir <dir>         Runtime directory for p-<pid>.json/.sock (default: ~/.mizar/run)
      --session-db <path>     SQLite session database path override
  -h, --help                  Print help"
}

fn split_serve_flag_value(raw: &str) -> (&str, Option<String>) {
    if raw.starts_with("--")
        && let Some((name, val)) = raw.split_once('=')
    {
        return (name, Some(val.to_string()));
    }
    if let Some(val) = raw.strip_prefix("-C=") {
        return ("-C", Some(val.to_string()));
    }
    (raw, None)
}

fn take_serve_value<I>(flag: &str, inline: Option<String>, args: &mut I) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    match inline {
        Some(v) => Ok(v),
        None => args.next().ok_or_else(|| format!("缺少 {flag} 的参数值")),
    }
}

fn parse_serve_cli_args(args: &[String]) -> Result<ServeCliParsed, ServeError> {
    let mut out = ServeCliParsed::default();
    let mut iter = args.iter().cloned();
    while let Some(raw) = iter.next() {
        let (flag, inline) = split_serve_flag_value(&raw);
        match flag {
            "--help" | "-h" => {
                if inline.is_some() {
                    return Err(ServeError::InvalidArgs(format!("参数 {flag} 不接受值")));
                }
                out.help = true;
            }
            "--host" => {
                let val =
                    take_serve_value(flag, inline, &mut iter).map_err(ServeError::InvalidArgs)?;
                out.host = Some(val);
            }
            "--port" => {
                let val =
                    take_serve_value(flag, inline, &mut iter).map_err(ServeError::InvalidArgs)?;
                let port = val.trim().parse::<u16>().map_err(|_| {
                    ServeError::InvalidArgs(format!("无效端口号 \"{val}\"（范围 0..=65535）"))
                })?;
                out.port = Some(port);
            }
            "--run-dir" => {
                let val =
                    take_serve_value(flag, inline, &mut iter).map_err(ServeError::InvalidArgs)?;
                if val.trim().is_empty() {
                    return Err(ServeError::InvalidArgs("--run-dir 不能为空".to_string()));
                }
                out.run_dir = Some(PathBuf::from(val));
            }
            "--workspace" | "-C" => {
                let val =
                    take_serve_value(flag, inline, &mut iter).map_err(ServeError::InvalidArgs)?;
                if val.trim().is_empty() {
                    return Err(ServeError::InvalidArgs(format!("{flag} 不能为空")));
                }
                out.workspace = Some(PathBuf::from(val));
            }
            "--session-db" => {
                let val =
                    take_serve_value(flag, inline, &mut iter).map_err(ServeError::InvalidArgs)?;
                if val.trim().is_empty() {
                    return Err(ServeError::InvalidArgs("--session-db 不能为空".to_string()));
                }
                out.session_db = Some(PathBuf::from(val));
            }
            other => {
                return Err(ServeError::InvalidArgs(format!(
                    "未知参数：{other}（运行 `mizar serve --help` 查看可用参数）"
                )));
            }
        }
    }
    Ok(out)
}

/// `mizar serve` 子命令入口：校验回环参数、构建运行时并启动 HTTP/WebSocket 网关，监听 Ctrl-C/SIGTERM 优雅退出。
pub async fn run_serve_cli(args: &[String]) -> Result<(), BoxError> {
    let parsed = parse_serve_cli_args(args)?;
    if parsed.help {
        println!("{}", format_serve_help_text());
        return Ok(());
    }

    let host = parsed
        .host
        .unwrap_or_else(|| DEFAULT_SERVE_HOST.to_string());
    validate_loopback_host(&host)?;

    let workspace = parsed
        .workspace
        .or_else(|| {
            std::env::var("WORKSPACE")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| std::env::current_dir().expect("failed to resolve cwd"));
    let run_dir = match parsed.run_dir {
        Some(dir) => dir,
        None => resolve_run_dir_path()?,
    };
    let port = parsed.port.unwrap_or(DEFAULT_SERVE_PORT);

    let loaded_cfg = load_mizar_config(&workspace);
    let mut spine_cfg = AgentSpineConfig::coding(workspace.clone());
    spine_cfg.session_db_path = resolve_default_session_db_path();
    loaded_cfg.apply_to_spine(&mut spine_cfg);
    if let Some(ref explicit_db) = parsed.session_db {
        spine_cfg.session_db_path = Some(explicit_db.clone());
    }

    let resolved_llm = loaded_cfg.resolve_llm_router();
    if let Some(ref resolved) = resolved_llm {
        spine_cfg.agent.provider = Some(resolved.active_provider.clone());
        spine_cfg.agent.model = resolved.active_model.clone();
    }

    let session_db_path = spine_cfg.session_db_path.clone();
    let runtime = MizarRuntime::build(spine_cfg).await?;
    runtime.install_retry(RetryConfig::default());
    if let Some(resolved) = resolved_llm {
        runtime.install_router(resolved.router);
    } else {
        runtime.register_mock_adapter()?;
    }

    let mut cfg = ServeConfig::new(workspace, run_dir)
        .with_host(host)
        .with_port(port);
    cfg.session_db_path = session_db_path;

    let handle = start_serve(cfg, Arc::clone(&runtime)).await?;
    println!("mizar serve listening on http://{}", handle.addr());

    wait_for_shutdown_signal().await;
    handle.shutdown().await;
    runtime.shutdown().await;
    Ok(())
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = async {
                if let Some(ref mut s) = sigterm {
                    s.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_host_validator_accepts_only_loopback() {
        assert_eq!(
            validate_loopback_host("127.0.0.1").unwrap(),
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        );
        assert_eq!(
            validate_loopback_host("::1").unwrap(),
            IpAddr::V6(Ipv6Addr::LOCALHOST)
        );
        assert_eq!(
            validate_loopback_host("[::1]").unwrap(),
            IpAddr::V6(Ipv6Addr::LOCALHOST)
        );
        assert_eq!(
            validate_loopback_host("localhost").unwrap(),
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        );
        assert_eq!(
            validate_loopback_host("LOCALHOST").unwrap(),
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        );

        for rejected in [
            "0.0.0.0",
            "::",
            "[::]",
            "192.168.1.20",
            "10.0.0.5",
            "127.0.0.2",
            "example.com",
            "",
        ] {
            let err = validate_loopback_host(rejected).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("Tailscale Serve"),
                "expected Tailscale Serve guidance for {rejected}: {msg}"
            );
        }
    }

    #[test]
    fn origin_validation_blocks_spoofed_hosts_and_userinfo() {
        assert!(is_allowed_origin("http://127.0.0.1", true));
        assert!(is_allowed_origin("http://127.0.0.1:3000", true));
        assert!(is_allowed_origin("http://localhost", true));
        assert!(is_allowed_origin("http://localhost:5173", true));
        assert!(is_allowed_origin("http://[::1]", true));
        assert!(is_allowed_origin("http://[::1]:8787", true));
        assert!(is_allowed_origin("https://mizar.tail1234.ts.net", true));
        assert!(is_allowed_origin("https://my-host.ts.net:8443", true));

        assert!(!is_allowed_origin("https://mizar.tail1234.ts.net", false));
        assert!(!is_allowed_origin("https://evil.example", true));
        assert!(!is_allowed_origin("https://ts.net.evil.com", true));
        assert!(!is_allowed_origin("https://ts.net", true));
        assert!(!is_allowed_origin(
            "https://user@mizar.tail1234.ts.net",
            true
        ));
        assert!(!is_allowed_origin("https://mizar.ts.net@evil.com", true));
        assert!(!is_allowed_origin("http://mizar.tail1234.ts.net", true));
        assert!(!is_allowed_origin("http://localhost/evil", true));
        assert!(!is_allowed_origin("http://localhost?x=1", true));
        assert!(!is_allowed_origin("http://localhost:0", true));
        assert!(!is_allowed_origin("http://localhost:99999", true));
        assert!(!is_allowed_origin("null", true));
        assert!(!is_allowed_origin("", true));
        assert!(!is_allowed_origin(" https://mizar.ts.net", true));
    }

    #[test]
    fn serve_cli_parser_handles_flags_and_errors() {
        let args: Vec<String> = [
            "--host=::1",
            "--port",
            "9090",
            "-C",
            "/tmp/ws",
            "--run-dir=/tmp/run",
            "--session-db",
            "/tmp/sess.sqlite",
        ]
        .into_iter()
        .map(ToString::to_string)
        .collect();
        let parsed = parse_serve_cli_args(&args).unwrap();
        assert_eq!(parsed.host.as_deref(), Some("::1"));
        assert_eq!(parsed.port, Some(9090));
        assert_eq!(parsed.workspace.as_deref(), Some(Path::new("/tmp/ws")));
        assert_eq!(parsed.run_dir.as_deref(), Some(Path::new("/tmp/run")));
        assert_eq!(
            parsed.session_db.as_deref(),
            Some(Path::new("/tmp/sess.sqlite"))
        );
        assert!(!parsed.help);

        assert!(parse_serve_cli_args(&["--help".into()]).unwrap().help);
        assert!(parse_serve_cli_args(&["--port".into(), "abc".into()]).is_err());
        assert!(parse_serve_cli_args(&["--unknown".into()]).is_err());
    }

    #[test]
    fn resolve_git_branch_reads_standard_and_worktree_git_dirs() {
        let base = std::env::temp_dir().join(format!("mizar_serve_git_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let repo = base.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::write(repo.join(".git/HEAD"), "ref: refs/heads/feat/serve\n").unwrap();
        assert_eq!(resolve_git_branch(&repo).as_deref(), Some("feat/serve"));

        let wt = base.join("wt");
        let real_git = base.join("real_git");
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::create_dir_all(&real_git).unwrap();
        std::fs::write(real_git.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", real_git.display())).unwrap();
        assert_eq!(resolve_git_branch(&wt).as_deref(), Some("main"));

        let _ = std::fs::remove_dir_all(&base);
    }
}
