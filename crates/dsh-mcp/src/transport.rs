//! MCP JSON-RPC 2.0 传输层：Stdio 子进程传输 + Streamable HTTP (JSON / SSE) 传输。

use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cordis::Logger;
use dsh_llm::sse::SseParser;
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
        logger: Logger,
    ) -> Result<Self, McpError> {
        if let Some(cmd) = &config.command {
            let stdio = StdioTransport::spawn(server_name, cmd, config, workspace, logger).await?;
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

type PendingRpcMap = Arc<Mutex<HashMap<i64, oneshot::Sender<Result<Value, McpError>>>>>;

pub struct StdioTransport {
    next_id: AtomicI64,
    pending: PendingRpcMap,
    writer: AsyncMutex<Option<tokio::process::ChildStdin>>,
    child: AsyncMutex<Option<Child>>,
    closed: Arc<Mutex<bool>>,
}

impl StdioTransport {
    async fn spawn(
        server_name: &str,
        command: &str,
        config: &McpServerConfig,
        workspace: &Path,
        logger: Logger,
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

        let pending: PendingRpcMap = Arc::new(Mutex::new(HashMap::new()));
        let closed = Arc::new(Mutex::new(false));

        let pending_reader = Arc::clone(&pending);
        let closed_reader = Arc::clone(&closed);
        let log_reader = logger.clone();
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
                            log_reader.debug(format!("mcp[{srv_name}] non-json line: {trimmed}"));
                            continue;
                        };
                        if let Some(id) = msg.get("id").and_then(Value::as_i64)
                            && (msg.get("result").is_some() || msg.get("error").is_some())
                        {
                            let tx = pending_reader.lock().unwrap().remove(&id);
                            if let Some(tx) = tx {
                                let _ = tx.send(parse_rpc_response(msg));
                            }
                        }
                    }
                    Err(err) => {
                        log_reader.warn(format!("mcp[{srv_name}] stdout read error: {err}"));
                        break;
                    }
                }
            }
            *closed_reader.lock().unwrap() = true;
            let mut waiters = pending_reader.lock().unwrap();
            for (_, tx) in waiters.drain() {
                let _ = tx.send(Err(McpError::Transport(
                    "MCP stdio connection closed".into(),
                )));
            }
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
                        logger.debug(format!("mcp[{srv}] stderr: {trimmed}"));
                    }
                    line.clear();
                }
            });
        }

        Ok(Self {
            next_id: AtomicI64::new(1),
            pending,
            writer: AsyncMutex::new(Some(stdin)),
            child: AsyncMutex::new(Some(child)),
            closed,
        })
    }

    async fn write_line(&self, payload: &Value) -> Result<(), McpError> {
        if *self.closed.lock().unwrap() {
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
        self.pending.lock().unwrap().insert(id, tx);

        let mut req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
        });
        if let Some(p) = params {
            req.as_object_mut().unwrap().insert("params".into(), p);
        }

        if let Err(err) = self.write_line(&req).await {
            self.pending.lock().unwrap().remove(&id);
            return Err(err);
        }

        let res = tokio::select! {
            _ = cancel.cancelled() => {
                self.pending.lock().unwrap().remove(&id);
                let _ = self.notify("notifications/cancelled", Some(json!({ "requestId": id }))).await;
                Err(McpError::Cancelled)
            }
            _ = tokio::time::sleep(timeout) => {
                self.pending.lock().unwrap().remove(&id);
                Err(McpError::Timeout(timeout.as_millis() as u64))
            }
            awaited = rx => {
                match awaited {
                    Ok(inner) => inner,
                    Err(_) => Err(McpError::Transport("MCP response channel dropped".into())),
                }
            }
        };
        res
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
        *self.closed.lock().unwrap() = true;
        drop(self.writer.lock().await.take());
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
        }
    }
}

pub struct HttpTransport {
    client: reqwest::Client,
    url: String,
    headers: HashMap<String, String>,
    session_id: Mutex<Option<String>>,
    next_id: AtomicI64,
}

impl HttpTransport {
    fn new(url: &str, config: &McpServerConfig) -> Result<Self, McpError> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| McpError::Transport(format!("failed to build http client: {e}")))?;
        let mut headers = HashMap::new();
        for (k, v) in &config.headers {
            headers.insert(k.clone(), expand_env_vars(v));
        }
        Ok(Self {
            client,
            url: expand_env_vars(url),
            headers,
            session_id: Mutex::new(None),
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
            .ok_or_else(|| McpError::Transport("empty HTTP response body for JSON-RPC request".into()))
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
            .post_json(&body, None, Duration::from_secs(10), &CancellationToken::new())
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
        if let Some(sid) = self.session_id.lock().unwrap().as_deref() {
            req = req.header("Mcp-Session-Id", sid);
        }
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }

        let fut = async {
            let resp = req
                .json(body)
                .send()
                .await
                .map_err(|e| McpError::Transport(format!("HTTP error: {e}")))?;

            if let Some(sid) = resp
                .headers()
                .get("mcp-session-id")
                .and_then(|v| v.to_str().ok())
            {
                *self.session_id.lock().unwrap() = Some(sid.to_string());
            }

            let status = resp.status();
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_ascii_lowercase();
            let text = resp
                .text()
                .await
                .map_err(|e| McpError::Transport(format!("failed reading HTTP body: {e}")))?;

            if !status.is_success() {
                return Err(McpError::Transport(format!(
                    "HTTP {}: {}",
                    status.as_u16(),
                    text.trim()
                )));
            }
            if expect_id.is_none() || text.trim().is_empty() {
                return Ok(None);
            }

            if content_type.contains("text/event-stream") {
                let mut dec = SseParser::new();
                let mut events = dec.push(text.as_bytes(), || {});
                events.extend(dec.finish(|| {}));
                for ev in events {
                    let data = ev.data.trim();
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(data)
                        && (v.get("id").and_then(Value::as_i64) == expect_id || expect_id.is_none())
                        && (v.get("result").is_some() || v.get("error").is_some())
                    {
                        return parse_rpc_response(v).map(Some);
                    }
                }
                return Err(McpError::Transport(
                    "SSE stream finished without matching JSON-RPC response".into(),
                ));
            }

            let val: Value = serde_json::from_str(text.trim())
                .map_err(|e| McpError::Transport(format!("invalid JSON-RPC response: {e}")))?;
            parse_rpc_response(val).map(Some)
        };

        tokio::select! {
            _ = cancel.cancelled() => Err(McpError::Cancelled),
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
