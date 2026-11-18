//! MCP 协议客户端会话（`initialize`、`tools/list`、`tools/call`、资源列举与读取）。

use std::path::Path;
use std::time::Duration;

use cordis::Logger;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::config::McpServerConfig;
use crate::transport::{McpError, McpTransport};

const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const MAX_LIST_PAGES: usize = 200;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpRemoteTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Value>,
    #[serde(default)]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub annotations: Option<McpRemoteAnnotations>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct McpRemoteAnnotations {
    #[serde(default)]
    pub read_only_hint: Option<bool>,
    #[serde(default)]
    pub destructive_hint: Option<bool>,
    #[serde(default)]
    pub idempotent_hint: Option<bool>,
    #[serde(default)]
    pub open_world_hint: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResourceTemplate {
    pub uri_template: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
}

#[derive(Clone)]
pub struct McpClient {
    pub server_name: String,
    pub instructions: Option<String>,
    pub has_tools: bool,
    pub has_resources: bool,
    timeout: Duration,
    transport: McpTransport,
}

impl McpClient {
    pub async fn connect(
        server_name: &str,
        config: &McpServerConfig,
        workspace: &Path,
        logger: Logger,
        cancel: &CancellationToken,
    ) -> Result<Self, McpError> {
        let timeout = Duration::from_millis(config.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS));
        let transport = McpTransport::connect(server_name, config, workspace, logger).await?;

        let init_res = transport
            .request(
                "initialize",
                Some(json!({
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "mizar",
                        "version": env!("CARGO_PKG_VERSION"),
                    }
                })),
                timeout,
                cancel,
            )
            .await;

        let init = match init_res {
            Ok(val) => val,
            Err(err) => {
                transport.close().await;
                return Err(err);
            }
        };

        let _ = transport.notify("notifications/initialized", None).await;

        let instructions = init
            .get("instructions")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let caps = init.get("capabilities");
        let has_tools = caps.and_then(|c| c.get("tools")).is_some();
        let has_resources = caps.and_then(|c| c.get("resources")).is_some();

        Ok(Self {
            server_name: server_name.to_string(),
            instructions,
            has_tools,
            has_resources,
            timeout,
            transport,
        })
    }

    pub async fn list_all_tools(&self, cancel: &CancellationToken) -> Result<Vec<McpRemoteTool>, McpError> {
        let mut all = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let params = cursor.as_ref().map(|c| json!({ "cursor": c }));
            let res = self
                .transport
                .request("tools/list", params, self.timeout, cancel)
                .await?;
            if let Some(arr) = res.get("tools").and_then(Value::as_array) {
                for item in arr {
                    if let Ok(tool) = serde_json::from_value::<McpRemoteTool>(item.clone()) {
                        all.push(tool);
                    }
                }
            }
            match res.get("nextCursor").and_then(Value::as_str) {
                Some(next) if !next.is_empty() => cursor = Some(next.to_string()),
                _ => break,
            }
        }
        Ok(all)
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
        cancel: &CancellationToken,
    ) -> Result<Value, McpError> {
        self.transport
            .request(
                "tools/call",
                Some(json!({
                    "name": name,
                    "arguments": arguments,
                })),
                self.timeout,
                cancel,
            )
            .await
    }

    pub async fn list_resources_page(
        &self,
        cursor: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<Value, McpError> {
        let params = cursor.map(|c| json!({ "cursor": c }));
        self.transport
            .request("resources/list", params, self.timeout, cancel)
            .await
    }

    pub async fn list_resource_templates_page(
        &self,
        cursor: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<Value, McpError> {
        let params = cursor.map(|c| json!({ "cursor": c }));
        self.transport
            .request("resources/templates/list", params, self.timeout, cancel)
            .await
    }

    pub async fn read_resource(&self, uri: &str, cancel: &CancellationToken) -> Result<Value, McpError> {
        self.transport
            .request(
                "resources/read",
                Some(json!({ "uri": uri })),
                self.timeout,
                cancel,
            )
            .await
    }

    pub async fn close(&self) {
        self.transport.close().await;
    }
}
