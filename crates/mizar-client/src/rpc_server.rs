//! 基于 Unix Domain Socket 与受鉴权回环 TCP（NDJSON / JSON-RPC 2.0）的 Mizar 本地进程间 RPC 服务端。
//!
//! 职责：
//! - 绑定 `<run_dir>/p-<pid>.sock` 并收敛为 `0600` 权限，同时清理已失活残留的 `.sock` 文件；
//! - 默认同步绑定 `127.0.0.1:0` 回环 TCP 旁路、生成 256-bit 随机 `rpc_token` 并以 `0600` 写入 `p-<pid>.json`；
//! - TCP 连接首帧必须通过 `auth.hello` 常量时间校验 `rpc_token`，否则返回 `-32003`（畸形 JSON 返回 `-32700`）并立即断连；
//! - 向已认证客户端提供全量 12 个 JSON-RPC 2.0 方法并映射 [`ActorError`] 错误码；
//! - 为每个已订阅连接独立维护 40ms / 16 KiB 的 `stream.delta` 聚合器与 256 帧有界出站队列，
//!   在非增量通知（如 `event.committed` / `turn.state`）前强制刷出待发增量，并在慢客户端队列溢出时主动断开以保护主循环。

use std::io;
#[cfg(unix)]
use std::net::Ipv4Addr;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use mizar_types::rpc::{
    AuthHelloParams, AuthHelloResult, CapabilitiesListParams, InteractionResolveParams,
    JsonRpcError, JsonRpcId, JsonRpcRequest, JsonRpcResponse, METHOD_AUTH_HELLO,
    METHOD_CAPABILITIES_LIST, METHOD_INTERACTION_RESOLVE, METHOD_QUEUE_POP_ALL,
    METHOD_QUEUE_REMOVE, METHOD_SESSION_CONFIG, METHOD_SESSION_EVENTS, METHOD_SESSION_FORK,
    METHOD_SESSION_LIST, METHOD_SESSION_SUBSCRIBE, METHOD_SESSION_TREE, METHOD_TURN_CANCEL,
    METHOD_TURN_SUBMIT, QueuePopAllParams, QueueRemoveParams, ResolvedBy, RpcNotification,
    SessionConfigParams, SessionEventsParams, SessionForkParams, SessionListParams,
    SessionSubscribeParams, SessionTreeParams, StreamDeltaKind, StreamDeltaNotification,
    TurnCancelParams, TurnSubmitParams,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
#[cfg(unix)]
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
#[cfg(unix)]
use tokio::task::JoinSet;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
#[cfg(unix)]
use uuid::Uuid;

use crate::actor::SessionActorHandle;
use crate::run_dir::{self, ProcessRegistration};

/// 默认 `stream.delta` 聚合时间窗（40ms，处于 30~50ms 推荐区间）。
pub const DEFAULT_COALESCE_WINDOW: Duration = Duration::from_millis(40);
/// 默认单条 `stream.delta` NDJSON 帧字节上限（16 KiB）。
pub const DEFAULT_MAX_DELTA_FRAME_BYTES: usize = 16 * 1024;
/// 默认单客户端出站有界队列容量（256 帧）。
pub const DEFAULT_CLIENT_QUEUE_CAPACITY: usize = 256;

/// `RpcServer` 运行配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RpcServerConfig {
    /// `stream.delta` 聚合时间窗（默认 40ms）。
    pub coalesce_window: Duration,
    /// 单条 `stream.delta` NDJSON 帧字节上限（默认 16 KiB）。
    pub max_delta_frame_bytes: usize,
    /// 单客户端出站有界队列容量（默认 256；满则立即断开慢客户端）。
    pub client_queue_capacity: usize,
    /// 是否启用 `127.0.0.1:0` 回环 TCP 旁路（需首帧 `auth.hello` 鉴权，默认开启）。
    pub enable_loopback_tcp: bool,
}

impl Default for RpcServerConfig {
    fn default() -> Self {
        Self {
            coalesce_window: DEFAULT_COALESCE_WINDOW,
            max_delta_frame_bytes: DEFAULT_MAX_DELTA_FRAME_BYTES,
            client_queue_capacity: DEFAULT_CLIENT_QUEUE_CAPACITY,
            enable_loopback_tcp: true,
        }
    }
}

/// Mizar Unix Socket / 回环 TCP JSON-RPC 2.0 服务端构建入口。
#[derive(Debug)]
pub struct RpcServer;

/// 已启动的 RPC 服务端句柄，支持优雅停机并在 `Drop` 时兜底清理文件。
#[derive(Debug)]
pub struct RpcServerHandle {
    socket_path: PathBuf,
    meta_path: PathBuf,
    tcp_addr: Option<SocketAddr>,
    rpc_token: Option<String>,
    cancel: CancellationToken,
    accept_task: Option<JoinHandle<()>>,
    tcp_accept_task: Option<JoinHandle<()>>,
    cleaned_up: Arc<AtomicBool>,
}

impl RpcServerHandle {
    /// 返回服务端监听的 Unix Socket 文件路径（`<run_dir>/p-<pid>.sock`）。
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// 返回关联的进程元信息文件路径（`<run_dir>/p-<pid>.json`）。
    pub fn meta_path(&self) -> &Path {
        &self.meta_path
    }

    /// 返回已启用的回环 TCP 监听地址（禁用回环 TCP 时为 `None`）。
    pub fn tcp_addr(&self) -> Option<SocketAddr> {
        self.tcp_addr
    }

    /// 返回当前进程回环 TCP 鉴权密钥（禁用回环 TCP 时为 `None`）。
    pub fn rpc_token(&self) -> Option<&str> {
        self.rpc_token.as_deref()
    }

    /// 优雅停止监听器与全部活跃客户端任务，并删除 `.sock` 与 `.json` 运行态文件。
    pub async fn shutdown(mut self) {
        self.cancel.cancel();
        if let Some(task) = self.accept_task.take() {
            let _ = task.await;
        }
        if let Some(task) = self.tcp_accept_task.take() {
            let _ = task.await;
        }
        self.cleanup_files();
    }

    fn cleanup_files(&self) {
        if !self.cleaned_up.swap(true, Ordering::AcqRel) {
            let _ = std::fs::remove_file(&self.socket_path);
            let _ = std::fs::remove_file(&self.meta_path);
        }
    }
}

impl Drop for RpcServerHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.accept_task.take() {
            task.abort();
        }
        if let Some(task) = self.tcp_accept_task.take() {
            task.abort();
        }
        self.cleanup_files();
    }
}

impl RpcServer {
    /// 启动 RPC 服务：确保运行目录 `0700`、绑定并设为 `0600`、按需启用回环 TCP 旁路、挂接进程注册句柄并开始监听。
    #[cfg(unix)]
    pub fn start(
        handle: SessionActorHandle,
        mut registration: ProcessRegistration,
        config: RpcServerConfig,
    ) -> io::Result<RpcServerHandle> {
        let socket_path = registration.socket_path().to_path_buf();
        let meta_path = registration.meta_path().to_path_buf();

        if let Some(parent) = socket_path.parent()
            && !parent.as_os_str().is_empty()
        {
            run_dir::ensure_run_dir(parent)?;
        }
        if let Some(meta_parent) = meta_path.parent()
            && !meta_parent.as_os_str().is_empty()
            && Some(meta_parent) != socket_path.parent()
        {
            run_dir::ensure_run_dir(meta_parent)?;
        }

        remove_stale_socket_if_unattached(&socket_path)?;

        let listener = UnixListener::bind(&socket_path)?;
        if let Err(err) = run_dir::set_socket_permissions_0600(&socket_path) {
            let _ = std::fs::remove_file(&socket_path);
            return Err(err);
        }

        let (tcp_listener, tcp_addr, rpc_token) = if config.enable_loopback_tcp {
            match bind_loopback_tcp_and_token(&mut registration) {
                Ok((tcp_listener, addr, token)) => (Some(tcp_listener), Some(addr), Some(token)),
                Err(err) => {
                    let _ = std::fs::remove_file(&socket_path);
                    return Err(err);
                }
            }
        } else {
            if (registration.meta().tcp_addr.is_some() || registration.meta().rpc_token.is_some())
                && let Err(err) = registration.update_tcp_endpoint(None, None)
            {
                let _ = std::fs::remove_file(&socket_path);
                return Err(err);
            }
            (None, None, None)
        };

        handle.attach_registration(registration);

        let cancel = CancellationToken::new();
        let accept_task = tokio::spawn(run_unix_accept_loop(
            listener,
            handle.clone(),
            config,
            cancel.clone(),
        ));
        let tcp_accept_task = match (tcp_listener, rpc_token.as_deref()) {
            (Some(tcp_listener), Some(token)) => Some(tokio::spawn(run_tcp_accept_loop(
                tcp_listener,
                handle,
                config,
                Arc::<str>::from(token),
                cancel.clone(),
            ))),
            _ => None,
        };

        Ok(RpcServerHandle {
            socket_path,
            meta_path,
            tcp_addr,
            rpc_token,
            cancel,
            accept_task: Some(accept_task),
            tcp_accept_task,
            cleaned_up: Arc::new(AtomicBool::new(false)),
        })
    }

    /// 非 Unix 平台暂不支持 Unix Domain Socket RPC 服务。
    #[cfg(not(unix))]
    pub fn start(
        _handle: SessionActorHandle,
        _registration: ProcessRegistration,
        _config: RpcServerConfig,
    ) -> io::Result<RpcServerHandle> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Unix Domain Socket RPC 仅支持 Unix 平台",
        ))
    }
}

#[cfg(unix)]
fn generate_rpc_token() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

#[cfg(unix)]
fn bind_loopback_tcp_and_token(
    registration: &mut ProcessRegistration,
) -> io::Result<(TcpListener, SocketAddr, String)> {
    let std_tcp = std::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))?;
    std_tcp.set_nonblocking(true)?;
    let tcp_listener = TcpListener::from_std(std_tcp)?;
    let addr = tcp_listener.local_addr()?;
    let token = generate_rpc_token();
    registration.update_tcp_endpoint(Some(addr.to_string()), Some(token.clone()))?;
    Ok((tcp_listener, addr, token))
}

#[cfg(unix)]
fn remove_stale_socket_if_unattached(socket_path: &Path) -> io::Result<()> {
    use std::os::unix::fs::FileTypeExt;

    let Ok(meta) = std::fs::symlink_metadata(socket_path) else {
        return Ok(());
    };
    if !meta.file_type().is_socket() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!(
                "Socket 路径已存在且非 socket 文件: {}",
                socket_path.display()
            ),
        ));
    }
    match std::os::unix::net::UnixStream::connect(socket_path) {
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::AddrInUse,
            format!("Unix socket 已被活跃进程占用: {}", socket_path.display()),
        )),
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::ConnectionRefused | io::ErrorKind::NotFound
            ) =>
        {
            match std::fs::remove_file(socket_path) {
                Ok(()) => Ok(()),
                Err(rm_err) if rm_err.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(rm_err) => Err(rm_err),
            }
        }
        Err(_) => match std::fs::remove_file(socket_path) {
            Ok(()) => Ok(()),
            Err(rm_err) if rm_err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(rm_err) => Err(rm_err),
        },
    }
}

#[cfg(unix)]
async fn run_unix_accept_loop(
    listener: UnixListener,
    handle: SessionActorHandle,
    config: RpcServerConfig,
    cancel: CancellationToken,
) {
    let mut clients = JoinSet::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                break;
            }
            accept_res = listener.accept() => {
                match accept_res {
                    Ok((stream, _addr)) => {
                        let conn_handle = handle.clone();
                        let conn_cancel = cancel.child_token();
                        clients.spawn(run_connection(stream, conn_handle, config, None, conn_cancel));
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "RPC UnixListener accept 失败");
                        tokio::select! {
                            _ = cancel.cancelled() => break,
                            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
                        }
                    }
                }
            }
            Some(_) = clients.join_next(), if !clients.is_empty() => {}
        }
    }
    clients.shutdown().await;
}

#[cfg(unix)]
async fn run_tcp_accept_loop(
    listener: TcpListener,
    handle: SessionActorHandle,
    config: RpcServerConfig,
    auth_token: Arc<str>,
    cancel: CancellationToken,
) {
    let mut clients = JoinSet::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                break;
            }
            accept_res = listener.accept() => {
                match accept_res {
                    Ok((stream, _addr)) => {
                        let _ = stream.set_nodelay(true);
                        let conn_handle = handle.clone();
                        let conn_cancel = cancel.child_token();
                        clients.spawn(run_connection(
                            stream,
                            conn_handle,
                            config,
                            Some(Arc::clone(&auth_token)),
                            conn_cancel,
                        ));
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "RPC TcpListener accept 失败");
                        tokio::select! {
                            _ = cancel.cancelled() => break,
                            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
                        }
                    }
                }
            }
            Some(_) = clients.join_next(), if !clients.is_empty() => {}
        }
    }
    clients.shutdown().await;
}

struct SubscriptionActivation {
    session_id: String,
    dedup_seq: Option<u64>,
    notif_rx: broadcast::Receiver<RpcNotification>,
}

async fn run_connection<S>(
    stream: S,
    handle: SessionActorHandle,
    config: RpcServerConfig,
    auth_token: Option<Arc<str>>,
    conn_cancel: CancellationToken,
) where
    S: AsyncRead + AsyncWrite + Send + 'static,
{
    let (read_half, write_half) = tokio::io::split(stream);
    let queue_cap = config.client_queue_capacity.max(1);
    let (outbound_tx, outbound_rx) = mpsc::channel::<String>(queue_cap);
    let (sub_cmd_tx, sub_cmd_rx) = mpsc::unbounded_channel::<SubscriptionActivation>();

    let writer_fut = run_writer_loop(write_half, outbound_rx, conn_cancel.clone());
    let notifier_fut =
        run_notifier_loop(sub_cmd_rx, outbound_tx.clone(), config, conn_cancel.clone());
    let reader_fut = run_reader_loop(
        read_half,
        handle,
        outbound_tx,
        sub_cmd_tx,
        auth_token,
        conn_cancel.clone(),
    );

    tokio::join!(writer_fut, notifier_fut, reader_fut);
}

async fn run_writer_loop<W>(
    mut write_half: W,
    mut outbound_rx: mpsc::Receiver<String>,
    conn_cancel: CancellationToken,
) where
    W: AsyncWrite + Unpin,
{
    loop {
        tokio::select! {
            biased;
            _ = conn_cancel.cancelled() => break,
            maybe_line = outbound_rx.recv() => {
                let Some(line) = maybe_line else {
                    let _ = write_half.shutdown().await;
                    break;
                };
                let write_res = tokio::select! {
                    biased;
                    _ = conn_cancel.cancelled() => break,
                    res = write_half.write_all(line.as_bytes()) => res,
                };
                if write_res.is_err() {
                    conn_cancel.cancel();
                    break;
                }
            }
        }
    }
}

async fn recv_next_notification(
    sub: Option<&mut SubscriptionActivation>,
) -> Result<RpcNotification, broadcast::error::RecvError> {
    match sub {
        Some(active) => active.notif_rx.recv().await,
        None => std::future::pending().await,
    }
}

async fn run_notifier_loop(
    mut sub_cmd_rx: mpsc::UnboundedReceiver<SubscriptionActivation>,
    outbound_tx: mpsc::Sender<String>,
    config: RpcServerConfig,
    conn_cancel: CancellationToken,
) {
    let mut active_sub: Option<SubscriptionActivation> = None;
    let mut coalescer =
        DeltaCoalescer::new(config.coalesce_window, config.max_delta_frame_bytes.max(1));

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
                    Some(next_sub) => {
                        // 重新订阅时新快照已包含最新文本现场，清空旧订阅遗留的瞬态增量缓冲
                        coalescer.clear();
                        active_sub = Some(next_sub);
                    }
                    None => {
                        // 仅当从未订阅过（如单次请求后半关闭）时随请求读取端自然退出
                        if active_sub.is_none() {
                            break;
                        }
                        std::future::pending::<()>().await;
                    }
                }
            }
            _ = sleep_fut => {
                if !coalescer.flush_all(&outbound_tx, &conn_cancel) {
                    break;
                }
            }
            recv_res = recv_next_notification(active_sub.as_mut()) => {
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
                                // 非增量事件前强制刷出所有挂起增量，确保流式增量绝不晚于其落盘事件到达
                                if !coalescer.flush_all(&outbound_tx, &conn_cancel) {
                                    break;
                                }
                                if !enqueue_notification(&other, &outbound_tx, &conn_cancel) {
                                    break;
                                }
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(skipped, "RPC 客户端广播接收滞后，主动断开连接");
                        conn_cancel.cancel();
                        break;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        let _ = coalescer.flush_all(&outbound_tx, &conn_cancel);
                        break;
                    }
                }
            }
        }
    }
}

/// 常量时间字节序列相等比较（覆盖内容与长度，避免时序侧信道泄露 `rpc_token`）。
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let max_len = a.len().max(b.len());
    let mut diff = a.len() ^ b.len();
    for i in 0..max_len {
        let ba = a.get(i).copied().unwrap_or(0);
        let bb = b.get(i).copied().unwrap_or(0);
        diff |= usize::from(ba ^ bb);
    }
    std::hint::black_box(diff) == 0
}

async fn run_reader_loop<R>(
    read_half: R,
    handle: SessionActorHandle,
    outbound_tx: mpsc::Sender<String>,
    sub_cmd_tx: mpsc::UnboundedSender<SubscriptionActivation>,
    auth_token: Option<Arc<str>>,
    conn_cancel: CancellationToken,
) where
    R: AsyncRead + Unpin,
{
    let mut reader = BufReader::new(read_half);
    let mut raw_buf = Vec::with_capacity(1024);
    let mut has_subscribed = false;
    let mut authenticated = auth_token.is_none();

    loop {
        raw_buf.clear();
        let read_res = tokio::select! {
            biased;
            _ = conn_cancel.cancelled() => break,
            res = reader.read_until(b'\n', &mut raw_buf) => res,
        };
        let bytes_read = match read_res {
            Ok(0) => {
                if has_subscribed {
                    conn_cancel.cancel();
                }
                break;
            }
            Ok(n) => n,
            Err(_) => {
                conn_cancel.cancel();
                break;
            }
        };

        let Ok(line_str) = std::str::from_utf8(&raw_buf[..bytes_read]) else {
            let resp = JsonRpcResponse::err(
                None,
                JsonRpcError::parse_error("invalid UTF-8 in JSON-RPC request"),
            );
            if !enqueue_response(&outbound_tx, &resp, &conn_cancel) || !authenticated {
                break;
            }
            continue;
        };
        let trimmed = line_str.trim();
        if trimmed.is_empty() {
            continue;
        }

        let raw_value: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(err) => {
                let resp = JsonRpcResponse::err(None, JsonRpcError::parse_error(err.to_string()));
                if !enqueue_response(&outbound_tx, &resp, &conn_cancel) || !authenticated {
                    break;
                }
                continue;
            }
        };

        let extracted_id = raw_value
            .get("id")
            .and_then(|v| serde_json::from_value::<JsonRpcId>(v.clone()).ok());
        let req: JsonRpcRequest = match serde_json::from_value(raw_value) {
            Ok(r) => r,
            Err(err) => {
                let err_obj = if authenticated {
                    JsonRpcError::invalid_request(err.to_string())
                } else {
                    JsonRpcError::unauthorized(
                        "authentication required: invalid auth.hello request",
                    )
                };
                let resp = JsonRpcResponse::err(extracted_id, err_obj);
                if !enqueue_response(&outbound_tx, &resp, &conn_cancel) || !authenticated {
                    break;
                }
                continue;
            }
        };
        if !req.is_valid_version() {
            let err_obj = if authenticated {
                JsonRpcError::invalid_request(format!(
                    "unsupported jsonrpc version: {}",
                    req.jsonrpc
                ))
            } else {
                JsonRpcError::unauthorized("authentication required: invalid jsonrpc version")
            };
            let resp = JsonRpcResponse::err(req.id, err_obj);
            if !enqueue_response(&outbound_tx, &resp, &conn_cancel) || !authenticated {
                break;
            }
            continue;
        }

        if !authenticated {
            let Some(expected) = auth_token.as_deref() else {
                break;
            };
            if req.method != METHOD_AUTH_HELLO {
                let resp = JsonRpcResponse::err(
                    req.id,
                    JsonRpcError::unauthorized(
                        "authentication required: first frame must be auth.hello",
                    ),
                );
                let _ = enqueue_response(&outbound_tx, &resp, &conn_cancel);
                break;
            }
            let Ok(params) = parse_required_params::<AuthHelloParams>(req.params) else {
                let resp = JsonRpcResponse::err(
                    req.id,
                    JsonRpcError::unauthorized("invalid auth.hello params"),
                );
                let _ = enqueue_response(&outbound_tx, &resp, &conn_cancel);
                break;
            };
            if expected.is_empty() || !ct_eq(params.token.as_bytes(), expected.as_bytes()) {
                let resp =
                    JsonRpcResponse::err(req.id, JsonRpcError::unauthorized("invalid rpc token"));
                let _ = enqueue_response(&outbound_tx, &resp, &conn_cancel);
                break;
            }
            authenticated = true;
            if let Some(id) = req.id {
                let resp = ok_response(id, &AuthHelloResult { ok: true });
                if !enqueue_response(&outbound_tx, &resp, &conn_cancel) {
                    break;
                }
            }
            continue;
        }

        if req.method == METHOD_SESSION_SUBSCRIBE {
            match parse_required_params::<SessionSubscribeParams>(req.params) {
                Ok(params) => {
                    // 先建立广播订阅再读取快照，并先投递快照响应再激活推送任务，保证无漏帧且响应先于推送
                    let notif_rx = handle.subscribe_notifications();
                    match handle.subscribe_snapshot_with(params.clone()).await {
                        Ok(snapshot) => {
                            let dedup_seq =
                                snapshot.events.last().map(|e| e.seq).or(params.after_seq);
                            let sub_sid = snapshot.session.id.clone();
                            if let Some(id) = req.id {
                                let resp = ok_response(id, &snapshot);
                                if !enqueue_response(&outbound_tx, &resp, &conn_cancel) {
                                    break;
                                }
                            }
                            let _ = sub_cmd_tx.send(SubscriptionActivation {
                                session_id: sub_sid,
                                dedup_seq,
                                notif_rx,
                            });
                            has_subscribed = true;
                        }
                        Err(err) => {
                            if let Some(id) = req.id {
                                let resp = JsonRpcResponse::err(Some(id), err.to_jsonrpc_error());
                                if !enqueue_response(&outbound_tx, &resp, &conn_cancel) {
                                    break;
                                }
                            }
                        }
                    }
                }
                Err(err) => {
                    if let Some(id) = req.id {
                        let resp = JsonRpcResponse::err(Some(id), err);
                        if !enqueue_response(&outbound_tx, &resp, &conn_cancel) {
                            break;
                        }
                    }
                }
            }
            continue;
        }

        let outcome = dispatch_non_subscribe_method(&handle, &req.method, req.params).await;
        if let Some(id) = req.id {
            let resp = match outcome {
                Ok(result) => JsonRpcResponse::ok(id, result),
                Err(err) => JsonRpcResponse::err(Some(id), err),
            };
            if !enqueue_response(&outbound_tx, &resp, &conn_cancel) {
                break;
            }
        }
    }
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

fn to_value_result<T: Serialize>(
    res: Result<T, crate::actor::ActorError>,
) -> Result<Value, JsonRpcError> {
    match res {
        Ok(val) => {
            serde_json::to_value(val).map_err(|e| JsonRpcError::internal_error(e.to_string()))
        }
        Err(err) => Err(err.to_jsonrpc_error()),
    }
}

async fn dispatch_non_subscribe_method(
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
        METHOD_SESSION_LIST => {
            let p: SessionListParams = parse_optional_params(params)?;
            to_value_result(handle.list_sessions(p).await)
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

fn enqueue_response(
    tx: &mpsc::Sender<String>,
    resp: &JsonRpcResponse,
    conn_cancel: &CancellationToken,
) -> bool {
    let Ok(mut line) = serde_json::to_string(resp) else {
        return true;
    };
    line.push('\n');
    try_enqueue_frame(tx, line, conn_cancel)
}

fn enqueue_notification(
    notif: &RpcNotification,
    tx: &mpsc::Sender<String>,
    conn_cancel: &CancellationToken,
) -> bool {
    let rpc = notif.to_jsonrpc();
    let Ok(mut line) = serde_json::to_string(&rpc) else {
        return true;
    };
    line.push('\n');
    try_enqueue_frame(tx, line, conn_cancel)
}

fn try_enqueue_frame(
    tx: &mpsc::Sender<String>,
    line: String,
    conn_cancel: &CancellationToken,
) -> bool {
    match tx.try_send(line) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(_)) => {
            tracing::warn!("RPC 客户端出站队列已满（慢客户端），主动断开连接");
            conn_cancel.cancel();
            false
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            conn_cancel.cancel();
            false
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

/// 按 `(session_id, turn, step, kind, call_id)` 分组聚合的 `stream.delta` 缓冲器。
struct DeltaCoalescer {
    coalesce_window: Duration,
    max_delta_frame_bytes: usize,
    pending: Vec<PendingDeltaBucket>,
    deadline: Option<Instant>,
}

impl DeltaCoalescer {
    fn new(coalesce_window: Duration, max_delta_frame_bytes: usize) -> Self {
        Self {
            coalesce_window,
            max_delta_frame_bytes: max_delta_frame_bytes.max(1),
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

    /// 推入一条增量分片；若某组缓冲序列化后超出单帧字节上限，则立即按 UTF-8 边界拆帧发送。
    fn push(
        &mut self,
        delta: StreamDeltaNotification,
        outbound_tx: &mpsc::Sender<String>,
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

        let idx = if let Some(pos) = self.pending.iter().position(|b| b.key == key) {
            self.pending[pos].delta.push_str(&delta.delta);
            pos
        } else {
            self.pending.push(PendingDeltaBucket {
                key,
                delta: delta.delta,
            });
            if self.deadline.is_none() {
                self.deadline = Some(Instant::now() + self.coalesce_window);
            }
            self.pending.len() - 1
        };

        let mut emitted_overflow = false;
        while !self.pending[idx].delta.is_empty()
            && build_delta_line(&self.pending[idx].key, &self.pending[idx].delta).len()
                > self.max_delta_frame_bytes
        {
            let split = find_delta_split_index(
                &self.pending[idx].key,
                &self.pending[idx].delta,
                self.max_delta_frame_bytes,
            );
            if split >= self.pending[idx].delta.len() {
                // 当外部把 max_delta_frame_bytes 设得比单字符 JSON 包裹还小时，直接刷出确保前进
                let line = build_delta_line(&self.pending[idx].key, &self.pending[idx].delta);
                self.pending[idx].delta.clear();
                if !try_enqueue_frame(outbound_tx, line, conn_cancel) {
                    return false;
                }
                emitted_overflow = true;
                break;
            }
            let line = build_delta_line(&self.pending[idx].key, &self.pending[idx].delta[..split]);
            self.pending[idx].delta.drain(..split);
            if !try_enqueue_frame(outbound_tx, line, conn_cancel) {
                return false;
            }
            emitted_overflow = true;
        }

        if self.pending[idx].delta.is_empty() {
            self.pending.remove(idx);
        }
        if self.pending.is_empty() {
            self.deadline = None;
        } else if emitted_overflow {
            self.deadline = Some(Instant::now() + self.coalesce_window);
        }
        true
    }

    /// 按先入先出顺序刷出全部挂起增量帧。
    fn flush_all(
        &mut self,
        outbound_tx: &mpsc::Sender<String>,
        conn_cancel: &CancellationToken,
    ) -> bool {
        self.deadline = None;
        if self.pending.is_empty() {
            return true;
        }
        let buckets = std::mem::take(&mut self.pending);
        for mut bucket in buckets {
            while !bucket.delta.is_empty() {
                let split =
                    find_delta_split_index(&bucket.key, &bucket.delta, self.max_delta_frame_bytes);
                let line = build_delta_line(&bucket.key, &bucket.delta[..split]);
                bucket.delta.drain(..split);
                if !try_enqueue_frame(outbound_tx, line, conn_cancel) {
                    return false;
                }
            }
        }
        true
    }
}

fn build_delta_line(key: &DeltaKey, delta_slice: &str) -> String {
    let notif = RpcNotification::StreamDelta(StreamDeltaNotification {
        session_id: key.session_id.clone(),
        turn: key.turn,
        step: key.step,
        kind: key.kind,
        call_id: key.call_id.clone(),
        delta: delta_slice.to_string(),
    });
    let rpc = notif.to_jsonrpc();
    let mut line = serde_json::to_string(&rpc).expect("StreamDeltaNotification serialization");
    line.push('\n');
    line
}

/// 在不破坏 UTF-8 字符边界的前提下，找出满足 `build_delta_line(key, &text[..idx]).len() <= max_frame_bytes` 的最大切分点。
fn find_delta_split_index(key: &DeltaKey, text: &str, max_frame_bytes: usize) -> usize {
    if text.is_empty() {
        return 0;
    }
    if build_delta_line(key, text).len() <= max_frame_bytes {
        return text.len();
    }

    let envelope_bytes = build_delta_line(key, "").len();
    let raw_budget = max_frame_bytes.saturating_sub(envelope_bytes);
    if raw_budget == 0 {
        return text.chars().next().map_or(0, char::len_utf8);
    }

    let hi_bound = floor_char_boundary(text, raw_budget.min(text.len()));
    if hi_bound == 0 {
        return text.chars().next().map_or(0, char::len_utf8);
    }
    if build_delta_line(key, &text[..hi_bound]).len() <= max_frame_bytes {
        return hi_bound;
    }

    // 仅当包含大量 JSON 转义字符（如 `"` / `\n`）时走二分精确定位最大前缀
    let mut best = 0usize;
    let mut lo = 1usize;
    let mut hi = hi_bound;
    while lo <= hi {
        let mid_raw = lo + (hi - lo) / 2;
        let mut mid = floor_char_boundary(text, mid_raw);
        if mid < lo {
            mid = ceil_char_boundary(text, mid_raw);
            if mid > hi {
                break;
            }
        }
        if build_delta_line(key, &text[..mid]).len() <= max_frame_bytes {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid.saturating_sub(1);
        }
    }

    if best == 0 {
        text.chars().next().map_or(0, char::len_utf8)
    } else {
        best
    }
}

fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use mizar_types::rpc::JsonRpcNotification;

    use super::*;

    fn make_delta(
        kind: StreamDeltaKind,
        call_id: Option<&str>,
        text: &str,
    ) -> StreamDeltaNotification {
        StreamDeltaNotification {
            session_id: "sess-unit".to_string(),
            turn: 1,
            step: 1,
            kind,
            call_id: call_id.map(str::to_string),
            delta: text.to_string(),
        }
    }

    fn drain_deltas(rx: &mut mpsc::Receiver<String>) -> Vec<(String, StreamDeltaNotification)> {
        let mut out = Vec::new();
        while let Ok(line) = rx.try_recv() {
            let rpc: JsonRpcNotification = serde_json::from_str(line.trim_end()).unwrap();
            let RpcNotification::StreamDelta(delta) = RpcNotification::from_jsonrpc(&rpc).unwrap()
            else {
                panic!("expected stream.delta");
            };
            out.push((line, delta));
        }
        out
    }

    #[test]
    fn coalescer_groups_same_key_and_preserves_distinct_keys() {
        let mut coalescer = DeltaCoalescer::new(Duration::from_millis(40), 16 * 1024);
        let (tx, mut rx) = mpsc::channel(32);
        let cancel = CancellationToken::new();

        assert!(coalescer.push(
            make_delta(StreamDeltaKind::Text, None, "你好"),
            &tx,
            &cancel
        ));
        assert!(coalescer.push(
            make_delta(StreamDeltaKind::ToolOutput, Some("c1"), "out-1\n"),
            &tx,
            &cancel
        ));
        assert!(coalescer.push(
            make_delta(StreamDeltaKind::Text, None, "世界"),
            &tx,
            &cancel
        ));
        assert!(coalescer.push(
            make_delta(StreamDeltaKind::ToolOutput, Some("c1"), "out-2"),
            &tx,
            &cancel
        ));
        assert!(rx.try_recv().is_err());

        assert!(coalescer.flush_all(&tx, &cancel));
        let frames = drain_deltas(&mut rx);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].1.kind, StreamDeltaKind::Text);
        assert_eq!(frames[0].1.delta, "你好世界");
        assert_eq!(frames[1].1.kind, StreamDeltaKind::ToolOutput);
        assert_eq!(frames[1].1.call_id.as_deref(), Some("c1"));
        assert_eq!(frames[1].1.delta, "out-1\nout-2");
    }

    #[test]
    fn coalescer_splits_multibyte_utf8_and_escaped_text_within_16kib_frame_cap() {
        let cap = 16 * 1024;
        let mut coalescer = DeltaCoalescer::new(Duration::from_millis(40), cap);
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();

        // 混合中文（3B）、emoji（4B）与 JSON 需转义的引号/换行，总长度远超 16 KiB
        let unit = "猫娘\"日志\"\n🚀";
        let payload = unit.repeat(2400);
        assert!(payload.len() > cap * 2);

        assert!(coalescer.push(
            make_delta(StreamDeltaKind::Text, None, &payload),
            &tx,
            &cancel
        ));
        assert!(coalescer.flush_all(&tx, &cancel));

        let frames = drain_deltas(&mut rx);
        assert!(frames.len() >= 3);

        let mut reassembled = String::new();
        for (raw_line, parsed) in &frames {
            assert!(
                raw_line.len() <= cap,
                "frame len {} exceeded cap {cap}",
                raw_line.len()
            );
            assert!(raw_line.ends_with('\n'));
            reassembled.push_str(&parsed.delta);
        }
        assert_eq!(reassembled, payload);
    }

    #[test]
    fn constant_time_token_equality_covers_content_and_length() {
        assert!(ct_eq(b"abcdef0123456789", b"abcdef0123456789"));
        assert!(ct_eq(b"", b""));
        assert!(!ct_eq(b"abcdef0123456789", b"Xbcdef0123456789"));
        assert!(!ct_eq(b"abcdef0123456789", b"abcdef01X3456789"));
        assert!(!ct_eq(b"abcdef0123456789", b"abcdef012345678X"));
        assert!(!ct_eq(b"abcdef0123456789", b"abcdef012345678"));
        assert!(!ct_eq(b"abcdef012345678", b"abcdef0123456789"));
        assert!(!ct_eq(b"", b"a"));
    }

    #[cfg(unix)]
    #[test]
    fn generated_rpc_token_is_64_lowercase_hex_chars() {
        let t1 = generate_rpc_token();
        let t2 = generate_rpc_token();
        assert_eq!(t1.len(), 64);
        assert_eq!(t2.len(), 64);
        assert_ne!(t1, t2);
        assert!(t1.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')));
    }
}
