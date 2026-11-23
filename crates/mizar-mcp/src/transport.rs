//! MCP JSON-RPC 2.0 传输层：Stdio 子进程传输 + Streamable HTTP (JSON / SSE) 传输。

use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mizar_types::llm::sse::SseParser;
use serde_json::{Value, json};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex as AsyncMutex, oneshot};
use tokio_util::sync::CancellationToken;

use crate::config::{McpServerConfig, expand_env_vars};

#[derive(Debug, Error)]
pub enum McpError {
    #[error("MCP transport error: {0}")]
    Transport(String),
    #[error("MCP protocol error ({code}): {message}")]
    Rpc { code: i64, message: String },
    #[error("MCP request timed out after {0}ms")]
    Timeout(u64),
    #[error("MCP request aborted")]
    Cancelled,
}

#[derive(Clone)]
pub enum McpTransport {
    Stdio(Arc<StdioTransport>),
    Http(Arc<HttpTransport>),
}

impl McpTransport {
    pub async fn connect(
        server_name: &str,
        config: &McpServerConfig,
        workspace: &Path,
    ) -> Result<Self, McpError> {
        if let Some(cmd) = &config.command {
            let stdio = StdioTransport::spawn(server_name, cmd, config, workspace).await?;
            Ok(Self::Stdio(Arc::new(stdio)))
        } else if let Some(url) = &config.url {
            let http = HttpTransport::new(url, config)?;
            Ok(Self::Http(Arc::new(http)))
        } else {
            Err(McpError::Transport(format!(
                "MCP server `{server_name}` has neither command nor url"
            )))
        }
    }

    pub async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> Result<Value, McpError> {
        match self {
            Self::Stdio(t) => t.request(method, params, timeout, cancel).await,
            Self::Http(t) => t.request(method, params, timeout, cancel).await,
        }
    }

    pub async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
        match self {
            Self::Stdio(t) => t.notify(method, params).await,
            Self::Http(t) => t.notify(method, params).await,
        }
    }

    pub async fn close(&self) {
        match self {
            Self::Stdio(t) => t.close().await,
            Self::Http(_) => {}
        }
    }
}

enum StdioRpcState {
    Open {
        pending: HashMap<i64, oneshot::Sender<Result<Value, McpError>>>,
    },
    Closed,
}

impl StdioRpcState {
    fn open() -> Self {
        Self::Open {
            pending: HashMap::new(),
        }
    }

    fn insert_pending(
        &mut self,
        id: i64,
        tx: oneshot::Sender<Result<Value, McpError>>,
    ) -> Result<(), McpError> {
        match self {
            Self::Open { pending } => {
                pending.insert(id, tx);
                Ok(())
            }
            Self::Closed => Err(McpError::Transport("MCP stdio transport closed".into())),
        }
    }

    fn take_pending(&mut self, id: i64) -> Option<oneshot::Sender<Result<Value, McpError>>> {
        match self {
            Self::Open { pending } => pending.remove(&id),
            Self::Closed => None,
        }
    }

    fn is_closed(&self) -> bool {
        matches!(self, Self::Closed)
    }

    fn transition_closed(&mut self) -> Vec<oneshot::Sender<Result<Value, McpError>>> {
        match std::mem::replace(self, Self::Closed) {
            Self::Open { mut pending } => pending.drain().map(|(_, tx)| tx).collect(),
            Self::Closed => Vec::new(),
        }
    }
}

fn close_stdio_rpc(state: &Mutex<StdioRpcState>, reason: &str) {
    let waiters = state.lock().unwrap().transition_closed();
    for tx in waiters {
        let _ = tx.send(Err(McpError::Transport(reason.to_string())));
    }
}

pub struct StdioTransport {
    next_id: AtomicI64,
    rpc: Arc<Mutex<StdioRpcState>>,
    writer: AsyncMutex<Option<tokio::process::ChildStdin>>,
    child: AsyncMutex<Option<Child>>,
}

impl StdioTransport {
    async fn spawn(
        server_name: &str,
        command: &str,
        config: &McpServerConfig,
        workspace: &Path,
    ) -> Result<Self, McpError> {
        let expanded_cmd = expand_env_vars(command);
        let mut cmd = Command::new(&expanded_cmd);
        for arg in &config.args {
            cmd.arg(expand_env_vars(arg));
        }
        let cwd = config.cwd.as_deref().unwrap_or(workspace);
        cmd.current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (k, v) in &config.env {
            cmd.env(k, expand_env_vars(v));
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Transport(format!("failed to spawn `{expanded_cmd}`: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("failed to capture stdio stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("failed to capture stdio stdout".into()))?;
        let stderr = child.stderr.take();

        let rpc = Arc::new(Mutex::new(StdioRpcState::open()));

        let rpc_reader = Arc::clone(&rpc);
        let srv_name = server_name.to_string();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => break,
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let Ok(msg) = serde_json::from_str::<Value>(trimmed) else {
                            tracing::debug!(
                                target: "mizar::mcp",
                                "mcp[{srv_name}] non-json line: {trimmed}"
                            );
                            continue;
                        };
                        if let Some(id) = msg.get("id").and_then(Value::as_i64)
                            && (msg.get("result").is_some() || msg.get("error").is_some())
                        {
                            let tx = rpc_reader.lock().unwrap().take_pending(id);
                            if let Some(tx) = tx {
                                let _ = tx.send(parse_rpc_response(msg));
                            }
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            target: "mizar::mcp",
                            "mcp[{srv_name}] stdout read error: {err}"
                        );
                        break;
                    }
                }
            }
            close_stdio_rpc(&rpc_reader, "MCP stdio connection closed");
        });

        if let Some(stderr) = stderr {
            let srv = server_name.to_string();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                while let Ok(n) = reader.read_line(&mut line).await {
                    if n == 0 {
                        break;
                    }
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        tracing::debug!(target: "mizar::mcp", "mcp[{srv}] stderr: {trimmed}");
                    }
                    line.clear();
                }
            });
        }

        Ok(Self {
            next_id: AtomicI64::new(1),
            rpc,
            writer: AsyncMutex::new(Some(stdin)),
            child: AsyncMutex::new(Some(child)),
        })
    }

    async fn write_line(&self, payload: &Value) -> Result<(), McpError> {
        if self.rpc.lock().unwrap().is_closed() {
            return Err(McpError::Transport("MCP stdio transport closed".into()));
        }
        let mut line = serde_json::to_string(payload)
            .map_err(|e| McpError::Transport(format!("json encode failed: {e}")))?;
        line.push('\n');

        let mut guard = self.writer.lock().await;
        let Some(stdin) = guard.as_mut() else {
            return Err(McpError::Transport("MCP stdio stdin closed".into()));
        };
        stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| McpError::Transport(format!("stdin write failed: {e}")))?;
        stdin
            .flush()
            .await
            .map_err(|e| McpError::Transport(format!("stdin flush failed: {e}")))?;
        Ok(())
    }

    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.rpc.lock().unwrap().insert_pending(id, tx)?;

        let mut req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
        });
        if let Some(p) = params {
            req.as_object_mut().unwrap().insert("params".into(), p);
        }

        if let Err(err) = self.write_line(&req).await {
            let _ = self.rpc.lock().unwrap().take_pending(id);
            return Err(err);
        }

        tokio::select! {
            _ = cancel.cancelled() => {
                let _ = self.rpc.lock().unwrap().take_pending(id);
                let _ = tokio::time::timeout(
                    Duration::from_millis(200),
                    self.notify("notifications/cancelled", Some(json!({ "requestId": id }))),
                )
                .await;
                Err(McpError::Cancelled)
            }
            _ = tokio::time::sleep(timeout) => {
                let _ = self.rpc.lock().unwrap().take_pending(id);
                Err(McpError::Timeout(timeout.as_millis() as u64))
            }
            awaited = rx => {
                match awaited {
                    Ok(inner) => inner,
                    Err(_) => Err(McpError::Transport("MCP response channel dropped".into())),
                }
            }
        }
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
        let mut msg = json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(p) = params {
            msg.as_object_mut().unwrap().insert("params".into(), p);
        }
        self.write_line(&msg).await
    }

    async fn close(&self) {
        close_stdio_rpc(&self.rpc, "MCP stdio transport closed");
        drop(self.writer.lock().await.take());
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
        }
    }
}

#[derive(Debug, Clone, Default)]
enum HttpSessionState {
    #[default]
    Handshake,
    Established(String),
}

pub struct HttpTransport {
    client: reqwest::Client,
    url: String,
    headers: HashMap<String, String>,
    session: Mutex<HttpSessionState>,
    next_id: AtomicI64,
}

impl HttpTransport {
    fn new(url: &str, config: &McpServerConfig) -> Result<Self, McpError> {
        let client = mizar_llm::http::default_http_client();
        let mut headers = HashMap::new();
        for (k, v) in &config.headers {
            headers.insert(k.clone(), expand_env_vars(v));
        }
        Ok(Self {
            client,
            url: expand_env_vars(url),
            headers,
            session: Mutex::new(HttpSessionState::Handshake),
            next_id: AtomicI64::new(1),
        })
    }

    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let mut body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
        });
        if let Some(p) = params {
            body.as_object_mut().unwrap().insert("params".into(), p);
        }
        self.post_json(&body, Some(id), timeout, cancel)
            .await?
            .ok_or_else(|| {
                McpError::Transport("empty HTTP response body for JSON-RPC request".into())
            })
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
        let mut body = json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(p) = params {
            body.as_object_mut().unwrap().insert("params".into(), p);
        }
        let _ = self
            .post_json(
                &body,
                None,
                Duration::from_secs(10),
                &CancellationToken::new(),
            )
            .await?;
        Ok(())
    }

    async fn post_json(
        &self,
        body: &Value,
        expect_id: Option<i64>,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> Result<Option<Value>, McpError> {
        let mut req = self
            .client
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .timeout(timeout);
        if let HttpSessionState::Established(ref sid) = *self.session.lock().unwrap() {
            req = req.header("Mcp-Session-Id", sid);
        }
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }

        let fut = async {
            let mut resp = req
                .json(body)
                .send()
                .await
                .map_err(|e| McpError::Transport(format!("HTTP error: {e}")))?;

            if let Some(sid) = resp
                .headers()
                .get("mcp-session-id")
                .and_then(|v| v.to_str().ok())
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                *self.session.lock().unwrap() = HttpSessionState::Established(sid.to_string());
            }

            let status = resp.status();
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_ascii_lowercase();

            if !status.is_success() {
                let text = resp
                    .text()
                    .await
                    .map_err(|e| McpError::Transport(format!("failed reading HTTP body: {e}")))?;
                return Err(McpError::Transport(format!(
                    "HTTP {}: {}",
                    status.as_u16(),
                    text.trim()
                )));
            }

            if expect_id.is_none() {
                return Ok(None);
            }

            if content_type.contains("text/event-stream") {
                let mut dec = SseParser::new();
                let mut matched: Option<Result<Value, McpError>> = None;
                let mut consume_events = |events: Vec<mizar_types::llm::sse::SseEvent>| -> bool {
                    for ev in events {
                        let data = ev.data.trim();
                        if data.is_empty() || data == "[DONE]" {
                            continue;
                        }
                        if let Ok(v) = serde_json::from_str::<Value>(data)
                            && v.get("id").and_then(Value::as_i64) == expect_id
                            && (v.get("result").is_some() || v.get("error").is_some())
                        {
                            matched = Some(parse_rpc_response(v));
                            return true;
                        }
                    }
                    false
                };

                while let Some(chunk) = resp.chunk().await.map_err(|e| {
                    McpError::Transport(format!("failed reading HTTP SSE chunk: {e}"))
                })? {
                    if consume_events(dec.push(&chunk, || {})) {
                        return matched.transpose();
                    }
                }
                if consume_events(dec.finish(|| {})) {
                    return matched.transpose();
                }
                return Err(McpError::Transport(
                    "SSE stream finished without matching JSON-RPC response".into(),
                ));
            }

            let text = resp
                .text()
                .await
                .map_err(|e| McpError::Transport(format!("failed reading HTTP body: {e}")))?;
            if text.trim().is_empty() {
                return Ok(None);
            }
            let val: Value = serde_json::from_str(text.trim())
                .map_err(|e| McpError::Transport(format!("invalid JSON-RPC response: {e}")))?;
            parse_rpc_response(val).map(Some)
        };

        tokio::select! {
            _ = cancel.cancelled() => Err(McpError::Cancelled),
            _ = tokio::time::sleep(timeout) => Err(McpError::Timeout(timeout.as_millis() as u64)),
            res = fut => res,
        }
    }
}

fn parse_rpc_response(msg: Value) -> Result<Value, McpError> {
    if let Some(err) = msg.get("error") {
        let code = err.get("code").and_then(Value::as_i64).unwrap_or(-32000);
        let message = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown MCP error")
            .to_string();
        return Err(McpError::Rpc { code, message });
    }
    Ok(msg.get("result").cloned().unwrap_or(Value::Null))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn stdio_rpc_state_rejects_after_close_and_drains_waiters() {
        let state = Mutex::new(StdioRpcState::open());
        let (tx1, mut rx1) = oneshot::channel();
        state.lock().unwrap().insert_pending(1, tx1).unwrap();

        close_stdio_rpc(&state, "closed");
        assert!(matches!(
            rx1.try_recv(),
            Ok(Err(McpError::Transport(msg))) if msg == "closed"
        ));

        let (tx2, _rx2) = oneshot::channel();
        let err = state.lock().unwrap().insert_pending(2, tx2).unwrap_err();
        assert!(matches!(err, McpError::Transport(_)));
    }

    #[tokio::test]
    async fn http_transport_streams_sse_chunk_without_waiting_for_eof() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let (keepalive_tx, mut keepalive_rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf).await;

            let headers = concat!(
                "HTTP/1.1 200 OK\r\n",
                "Content-Type: text/event-stream\r\n",
                "Mcp-Session-Id: sess-42\r\n",
                "Transfer-Encoding: chunked\r\n",
                "\r\n"
            );
            sock.write_all(headers.as_bytes()).await.unwrap();

            let ev = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n\n";
            let chunk = format!("{:x}\r\n{ev}\r\n", ev.len());
            sock.write_all(chunk.as_bytes()).await.unwrap();
            sock.flush().await.unwrap();

            // Deliberately keep TCP stream open until client finishes parsing chunk
            let _ = tokio::time::timeout(Duration::from_secs(5), &mut keepalive_rx).await;
        });

        let cfg = McpServerConfig::http(format!("http://{addr}/mcp"));
        let transport = HttpTransport::new(&cfg.url.clone().unwrap(), &cfg).unwrap();
        let res = transport
            .request(
                "tools/list",
                None,
                Duration::from_millis(800),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(res["ok"], json!(true));
        assert!(matches!(
            *transport.session.lock().unwrap(),
            HttpSessionState::Established(ref s) if s == "sess-42"
        ));
        let _ = keepalive_tx.send(());
    }
}
