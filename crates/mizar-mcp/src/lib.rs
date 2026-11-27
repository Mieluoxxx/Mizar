//! MCP 插件：连接配置文件与程序注册的 MCP Server，将远程工具以 `mcp__<server>__<tool>` 注册进 `ToolRuntime`，
//! 并支持 `Direct` / `Codemode` / `Deferred` / `Hidden` 分级暴露及 `list_mcp_resources` 资源工具。

pub mod client;
pub mod config;
pub mod transport;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use cordis::{BoxError, BoxFuture, Cleanup, Context, CordisError, Inject, Logger, Service};
use mizar_image::{MAX_IMAGE_INPUT_BYTES, base64_encoded_len, process_image_bytes};
use mizar_system_prompt::{PromptSection, SystemPrompt};
use mizar_tools::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head, write_temp_output,
};
use mizar_tools::{
    MAX_IMAGES_PER_MESSAGE, ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolExposure,
    ToolNamespace, ToolOutput, ToolsContextExt,
};
use serde_json::{Value, json};
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

pub use client::{McpClient, McpRemoteTool};
pub use config::{
    CONFIG_DIR_NAME, LoadedMcpConfig, LoadedMcpEntry, McpExposure, McpServerConfig,
    load_mcp_config, load_mcp_config_with_global, mcp_namespace, mcp_tool_name,
    resolve_global_config_dir,
};

pub const LIST_MCP_RESOURCES_TOOL: &str = "list_mcp_resources";
pub const LIST_MCP_RESOURCE_TEMPLATES_TOOL: &str = "list_mcp_resource_templates";
pub const READ_MCP_RESOURCE_TOOL: &str = "read_mcp_resource";
const MAX_SERVER_DESCRIPTION_CHARS: usize = 250;
const MAX_SERVERS_SECTION_CHARS: usize = 4096;

#[derive(Debug, Clone)]
pub struct McpPluginConfig {
    pub workspace: PathBuf,
    /// 可选覆盖全局配置目录（默认 `~/.mizar`，或 `MIZAR_HOME`）。
    pub global_config_dir: Option<PathBuf>,
    /// 显式跳过全局 `mcp.json`（用于 `--isolated` 隔离模式），仅加载工作区 `.mizar/mcp.json`。
    pub skip_global_config: bool,
    /// 是否从 `~/.mizar/mcp.json` 与 `<workspace>/.mizar/mcp.json` 自动加载配置（默认 true）。
    pub load_config_files: bool,
    /// 代码内显式追加/覆盖的 MCP Server 配置。
    pub servers: HashMap<String, McpServerConfig>,
    pub auto_enable_codemode: bool,
    /// 连接时是否异步后台连接（默认 true，不阻塞首屏启动；测试可设 false 并调用 `wait_ready`）。
    pub background_connect: bool,
}

impl McpPluginConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            load_config_files: true,
            background_connect: true,
            ..Self::default()
        }
    }

    fn load_servers(&mut self) -> LoadedMcpConfig {
        let mut loaded = if self.load_config_files {
            let global = if self.skip_global_config {
                None
            } else {
                self.global_config_dir
                    .clone()
                    .or_else(resolve_global_config_dir)
            };
            load_mcp_config_with_global(&self.workspace, global.as_deref())
        } else {
            LoadedMcpConfig {
                auto_enable_codemode: true,
                ..Default::default()
            }
        };
        loaded.auto_enable_codemode &= self.auto_enable_codemode;
        loaded
            .entries
            .retain(|entry| !self.servers.contains_key(&entry.name));
        loaded
            .entries
            .extend(self.servers.drain().map(|(name, config)| LoadedMcpEntry {
                name,
                config,
                source: "programmatic".into(),
            }));
        loaded.entries.sort_by(|a, b| a.name.cmp(&b.name));
        loaded
    }
}

impl Default for McpPluginConfig {
    fn default() -> Self {
        Self {
            workspace: PathBuf::new(),
            global_config_dir: None,
            skip_global_config: false,
            load_config_files: false,
            servers: HashMap::new(),
            auto_enable_codemode: true,
            background_connect: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum McpConnectionState {
    Connecting,
    Connected {
        tools_count: usize,
        has_resources: bool,
    },
    Disabled,
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum McpManagerPhase {
    Initializing,
    Ready,
    Closing,
    Closed,
}

impl McpManagerPhase {
    fn is_closing_or_closed(self) -> bool {
        matches!(self, Self::Closing | Self::Closed)
    }
}

#[derive(Debug, Clone)]
pub struct McpServerStatus {
    pub name: String,
    pub exposure: McpExposure,
    pub source: String,
    pub state: McpConnectionState,
    pub registered_tools: Vec<(String, ToolExposure)>,
    pub description: Option<String>,
    pub instructions: Option<String>,
}

struct ConnectedServer {
    entry: LoadedMcpEntry,
    client: Arc<McpClient>,
    cleanups: Vec<Cleanup>,
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    if max_chars <= 1 {
        return String::new();
    }
    let head: String = text.chars().take(max_chars - 1).collect();
    format!("{}…", head.trim_end())
}

pub struct McpManager {
    ctx: Context,
    config: LoadedMcpConfig,
    workspace: PathBuf,
    background_connect: bool,
    servers: AsyncMutex<HashMap<String, ConnectedServer>>,
    statuses: Mutex<HashMap<String, McpServerStatus>>,
    resource_cleanups: AsyncMutex<Vec<Cleanup>>,
    connect_lock: AsyncMutex<()>,
    connect_cancel: CancellationToken,
    phase: Mutex<McpManagerPhase>,
    ready_notify: Arc<tokio::sync::Notify>,
}

impl McpManager {
    fn logger(&self) -> Logger {
        self.ctx.logger_named("mcp")
    }

    fn is_closing_or_closed(&self) -> bool {
        self.phase.lock().unwrap().is_closing_or_closed() || self.connect_cancel.is_cancelled()
    }

    pub async fn wait_ready(&self) {
        loop {
            let notified = self.ready_notify.notified();
            if *self.phase.lock().unwrap() != McpManagerPhase::Initializing {
                return;
            }
            notified.await;
        }
    }

    pub fn needs_initial_catalog(&self, include: Option<&[String]>) -> bool {
        let mut enabled = self
            .config
            .entries
            .iter()
            .filter(|s| s.config.enabled)
            .peekable();
        if enabled.peek().is_none() {
            return false;
        }
        if let Some(patterns) = include {
            return patterns.iter().any(|pattern| {
                let prefix = pattern.split('*').next().unwrap_or(pattern);
                pattern.starts_with("mcp__")
                    || (pattern.contains('*') && "mcp__".starts_with(prefix))
                    || [
                        "tool_search",
                        LIST_MCP_RESOURCES_TOOL,
                        LIST_MCP_RESOURCE_TEMPLATES_TOOL,
                        READ_MCP_RESOURCE_TOOL,
                    ]
                    .iter()
                    .any(|name| mizar_tools::wildcard_matches(pattern, name))
            });
        }
        // 纯直连配置的首步需要真实声明；间接工具可在后台就绪后发现。
        enabled.all(|s| {
            s.config.exposure == McpExposure::Direct
                && !s.config.uses_exposure(McpExposure::Codemode)
                && !s.config.uses_exposure(McpExposure::Deferred)
        })
    }

    pub fn statuses(&self) -> Vec<McpServerStatus> {
        let mut list: Vec<McpServerStatus> =
            self.statuses.lock().unwrap().values().cloned().collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }

    pub fn render_servers_section(&self) -> Option<String> {
        let indirect: Vec<_> = self
            .config
            .entries
            .iter()
            .filter(|s| {
                s.config.uses_exposure(McpExposure::Codemode)
                    || s.config.uses_exposure(McpExposure::Deferred)
            })
            .collect();
        if indirect.is_empty() {
            return None;
        }

        let intro = "Configured MCP namespaces with indirect tools:";
        let heads: Vec<_> = indirect
            .iter()
            .map(|s| format!("- {}", mcp_namespace(&s.name)))
            .collect();

        let omitted = |count: usize| -> Option<String> {
            (count > 0).then(|| {
                let suffix = if count == 1 { "" } else { "s" };
                format!("- … {count} more server{suffix}")
            })
        };
        let size_for = |kept: usize| -> usize {
            let mut len = intro.chars().count();
            for head in heads.iter().take(kept) {
                len += 1 + head.chars().count();
            }
            if let Some(om) = omitted(indirect.len() - kept) {
                len += 1 + om.chars().count();
            }
            len
        };

        let mut kept = indirect.len();
        while kept > 0 && size_for(kept) > MAX_SERVERS_SECTION_CHARS {
            kept -= 1;
        }
        let per_server = if kept == 0 {
            0
        } else {
            let avail = MAX_SERVERS_SECTION_CHARS.saturating_sub(size_for(kept));
            (avail / kept)
                .saturating_sub(2)
                .min(MAX_SERVER_DESCRIPTION_CHARS)
        };

        let mut parts = vec![intro.to_string()];
        for (idx, s) in indirect.iter().take(kept).enumerate() {
            let summary = if per_server > 0 {
                let first_line = s
                    .config
                    .description
                    .as_deref()
                    .unwrap_or_default()
                    .trim()
                    .lines()
                    .next()
                    .unwrap_or_default();
                truncate_chars(first_line, per_server)
            } else {
                String::new()
            };
            if summary.is_empty() {
                parts.push(heads[idx].clone());
            } else {
                parts.push(format!("{}: {summary}", heads[idx]));
            }
        }
        if let Some(om) = omitted(indirect.len() - kept) {
            parts.push(om);
        }
        Some(parts.join("\n"))
    }

    pub async fn connect_configured(&self) {
        let _guard = self.connect_lock.lock().await;
        if self.is_closing_or_closed() {
            self.ready_notify.notify_waiters();
            return;
        }
        let entries = &self.config.entries;

        {
            let mut st = self.statuses.lock().unwrap();
            st.clear();
            for entry in entries {
                let state = if entry.config.enabled {
                    McpConnectionState::Connecting
                } else {
                    McpConnectionState::Disabled
                };
                st.insert(
                    entry.name.clone(),
                    McpServerStatus {
                        name: entry.name.clone(),
                        exposure: entry.config.exposure,
                        source: entry.source.clone(),
                        state,
                        registered_tools: Vec::new(),
                        description: entry.config.description.clone(),
                        instructions: None,
                    },
                );
            }
        }

        for entry in entries {
            if self.is_closing_or_closed() {
                break;
            }
            if !entry.config.enabled {
                continue;
            }
            if let Err(err) = self.connect_one(entry.clone()).await {
                if self.is_closing_or_closed() {
                    break;
                }
                self.logger()
                    .warn(format!("MCP server `{}` failed: {err}", entry.name));
                if let Some(st) = self.statuses.lock().unwrap().get_mut(&entry.name) {
                    st.state = McpConnectionState::Failed(err);
                }
            }
        }

        if !self.is_closing_or_closed() {
            self.sync_resource_tools().await;
        }
        {
            let mut phase = self.phase.lock().unwrap();
            if *phase == McpManagerPhase::Initializing {
                *phase = McpManagerPhase::Ready;
            }
        }
        self.ready_notify.notify_waiters();
    }

    async fn connect_one(&self, entry: LoadedMcpEntry) -> Result<(), String> {
        let cancel = self.connect_cancel.clone();
        let client = Arc::new(
            McpClient::connect(
                &entry.name,
                &entry.config,
                &self.workspace,
                self.logger(),
                &cancel,
            )
            .await
            .map_err(|e| e.to_string())?,
        );

        let remote_tools = match client.list_all_tools(&cancel).await {
            Ok(tools) => tools,
            Err(err) => {
                client.close().await;
                return Err(err.to_string());
            }
        };

        if self.is_closing_or_closed() {
            client.close().await;
            return Err("MCP manager closed".to_string());
        }

        let tools_rt = self.ctx.tools().map_err(|e| e.to_string())?;
        let ns_name = mcp_namespace(&entry.name);
        let ns_desc = entry
            .config
            .description
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string);
        let ns_instructions = client
            .instructions
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string);
        let namespace = ToolNamespace {
            name: ns_name,
            aliases: vec![entry.name.clone()],
            description: ns_desc.clone(),
            instructions: ns_instructions.clone(),
        };

        let mut cleanups = Vec::new();
        let mut registered_tools = Vec::new();

        for remote in &remote_tools {
            let full_name = mcp_tool_name(&entry.name, &remote.name);
            let mcp_exp = entry.config.resolve_tool_exposure(&remote.name);
            let exposure: ToolExposure = mcp_exp.into();

            let mut def = make_mcp_tool_def(
                &full_name,
                &remote.name,
                remote,
                namespace.clone(),
                exposure,
                Arc::clone(&client),
            );
            def.default_active = Some(exposure == ToolExposure::Direct);

            match tools_rt.register(def) {
                Ok(c) => {
                    cleanups.push(c);
                    registered_tools.push((full_name, exposure));
                }
                Err(err) => {
                    self.logger()
                        .warn(format!("skipping MCP tool `{full_name}`: {err}"));
                }
            }
        }

        let state = McpConnectionState::Connected {
            tools_count: remote_tools.len(),
            has_resources: client.has_resources,
        };
        if let Some(st) = self.statuses.lock().unwrap().get_mut(&entry.name) {
            st.state = state;
            st.registered_tools = registered_tools;
            if st.description.is_none() {
                st.description = ns_desc;
            }
            st.instructions = ns_instructions;
        }

        self.servers.lock().await.insert(
            entry.name.clone(),
            ConnectedServer {
                entry,
                client,
                cleanups,
            },
        );
        Ok(())
    }

    async fn sync_resource_tools(&self) {
        let mut res_cleanups = self.resource_cleanups.lock().await;
        if !res_cleanups.is_empty() {
            return;
        }
        let any_resources = {
            let srvs = self.servers.lock().await;
            srvs.values().any(|s| s.client.has_resources)
        };
        if !any_resources {
            return;
        }
        let Ok(tools_rt) = self.ctx.tools() else {
            return;
        };

        let defs = [
            make_list_resources_def(false),
            make_list_resources_def(true),
            make_read_resource_def(),
        ];
        for def in defs {
            if let Ok(c) = tools_rt.register(def) {
                res_cleanups.push(c);
            }
        }
    }

    pub async fn close_all(&self) {
        {
            let mut phase = self.phase.lock().unwrap();
            if *phase == McpManagerPhase::Closed {
                return;
            }
            *phase = McpManagerPhase::Closing;
        }
        self.connect_cancel.cancel();
        if let Ok(tools) = self.ctx.tools() {
            tools.set_auto_default_active("codemode", None);
        }
        self.ready_notify.notify_waiters();
        let _connect_guard = self.connect_lock.lock().await;

        let mut res = self.resource_cleanups.lock().await;
        for c in res.drain(..) {
            run_cleanup(c).await;
        }
        drop(res);

        let mut srvs = self.servers.lock().await;
        let drained: Vec<ConnectedServer> = srvs.drain().map(|(_, v)| v).collect();
        drop(srvs);

        for srv in drained {
            for c in srv.cleanups {
                run_cleanup(c).await;
            }
            srv.client.close().await;
        }

        {
            let mut st = self.statuses.lock().unwrap();
            for s in st.values_mut() {
                s.registered_tools.clear();
                if !matches!(s.state, McpConnectionState::Failed(_)) {
                    s.state = McpConnectionState::Disabled;
                }
            }
        }

        *self.phase.lock().unwrap() = McpManagerPhase::Closed;
        self.ready_notify.notify_waiters();
    }

    async fn resource_clients(&self, filter_server: Option<&str>) -> Vec<(String, Arc<McpClient>)> {
        let srvs = self.servers.lock().await;
        let mut list: Vec<(String, Arc<McpClient>)> = srvs
            .values()
            .filter(|s| s.client.has_resources && filter_server.is_none_or(|f| f == s.entry.name))
            .map(|s| (s.entry.name.clone(), Arc::clone(&s.client)))
            .collect();
        list.sort_by(|a, b| a.0.cmp(&b.0));
        list
    }
}

async fn run_cleanup(c: Cleanup) {
    match c {
        Cleanup::Sync(f) => f(),
        Cleanup::Async(f) => f().await,
    }
}

fn make_mcp_tool_def(
    full_name: &str,
    raw_name: &str,
    remote: &McpRemoteTool,
    namespace: ToolNamespace,
    exposure: ToolExposure,
    client: Arc<McpClient>,
) -> ToolDefinition {
    let params = remote
        .input_schema
        .clone()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    let ann = remote.annotations.clone().unwrap_or_default();
    let annotations = ToolAnnotations {
        read_only: ann.read_only_hint.unwrap_or(false),
        destructive: ann.destructive_hint.unwrap_or(false),
        idempotent: ann.idempotent_hint.unwrap_or(false),
        open_world: ann.open_world_hint.unwrap_or(false),
    };
    let exec_mode = if annotations.read_only {
        ToolExecutionMode::Parallel
    } else {
        ToolExecutionMode::Sequential
    };
    let remote_name = raw_name.to_string();
    let desc = remote
        .description
        .clone()
        .unwrap_or_else(|| format!("MCP tool `{raw_name}`"));

    let mut def =
        ToolDefinition::async_rich_tool(full_name, desc, params, move |args, call_ctx| {
            let client = Arc::clone(&client);
            let remote_name = remote_name.clone();
            async move {
                let res = client
                    .call_tool(&remote_name, args, &call_ctx.cancel)
                    .await?;
                format_mcp_tool_result(res).await
            }
        })
        .with_exposure(exposure)
        .with_execution_mode(exec_mode)
        .with_annotations(annotations)
        .with_namespace(namespace);

    if let Some(out_schema) = remote.output_schema.clone() {
        def = def.with_output_schema(out_schema);
    }
    def
}

async fn format_mcp_tool_result(mut res: Value) -> Result<ToolOutput, BoxError> {
    let is_error = res.get("isError").and_then(Value::as_bool).unwrap_or(false);
    let mut parts = Vec::new();
    let mut images = Vec::new();
    let mut skipped_images = 0;
    let mut structured = res.get_mut("structuredContent").map(Value::take);

    if let Some(blocks) = res.get_mut("content").and_then(Value::as_array_mut) {
        if let Some(value) = &mut structured {
            let payloads: Vec<_> = blocks
                .iter()
                .filter(|block| block["type"] == "image")
                .filter_map(|block| block["data"].as_str())
                .collect();
            if !payloads.is_empty() {
                mizar_tools::omit_image_data(value, &payloads);
            }
        }
        for block in blocks {
            let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
            match kind {
                "text" => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        parts.push(text.to_string());
                    }
                }
                "image" => {
                    if images.len() == MAX_IMAGES_PER_MESSAGE {
                        skipped_images += 1;
                        continue;
                    }
                    let Value::String(data) = block["data"].take() else {
                        parts.push("[Image omitted: missing base64 data]".into());
                        continue;
                    };
                    if data.len() > base64_encoded_len(MAX_IMAGE_INPUT_BYTES) {
                        parts.push("[Image omitted: exceeds 32 MiB input limit]".into());
                        continue;
                    }
                    let decoded = tokio::task::spawn_blocking(move || -> Result<_, BoxError> {
                        let bytes = STANDARD.decode(data)?;
                        if bytes.len() > MAX_IMAGE_INPUT_BYTES {
                            return Err("exceeds 32 MiB input limit".into());
                        }
                        Ok(process_image_bytes(&bytes)?)
                    })
                    .await?;
                    match decoded {
                        Ok(image) => {
                            parts.push(format!(
                                "[Image {}: {}×{}, {}]",
                                images.len() + 1,
                                image.original_width,
                                image.original_height,
                                image.attachment.media_type
                            ));
                            images.push(image.attachment);
                        }
                        Err(error) => parts.push(format!("[Image omitted: {error}]")),
                    }
                }
                "resource" => {
                    if let Some(text) = block
                        .get("resource")
                        .and_then(|r| r.get("text"))
                        .and_then(Value::as_str)
                    {
                        parts.push(text.to_string());
                    } else {
                        parts.push(block.to_string());
                    }
                }
                _ => parts.push(block.to_string()),
            }
        }
    }

    let raw_text = if parts.is_empty() {
        structured
            .as_ref()
            .map(|v| serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()))
            .unwrap_or_else(|| "(empty result)".to_string())
    } else {
        parts.join("\n")
    };

    let trunc = truncate_head(&raw_text, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let mut content = if trunc.truncated {
        let spilled = write_temp_output("mizar-mcp-", &raw_text)
            .await
            .ok()
            .map(|p| format!(" Full output saved to: {}", p.display()))
            .unwrap_or_default();
        format!(
            "{}\n\n[Truncated to {} lines / {}.{}]",
            trunc.content,
            trunc.output_lines,
            format_size(DEFAULT_MAX_BYTES),
            spilled
        )
    } else {
        trunc.content
    };

    if skipped_images > 0 {
        content.push_str(&format!("\n[Image limit reached: kept the first {MAX_IMAGES_PER_MESSAGE}, skipped {skipped_images}.]"));
    }
    if !images.is_empty()
        && let Some(Value::Object(fields)) = &mut structured
    {
        fields
            .entry("text")
            .or_insert_with(|| Value::String(content.clone()));
    }
    Ok(ToolOutput {
        content,
        images,
        is_error,
        structured_content: structured,
        details: None,
    })
}

fn make_list_resources_def(templates: bool) -> ToolDefinition {
    let (name, desc) = if templates {
        (
            LIST_MCP_RESOURCE_TEMPLATES_TOOL,
            "List MCP resource URI templates exposed by connected servers.",
        )
    } else {
        (
            LIST_MCP_RESOURCES_TOOL,
            "List MCP resources exposed by connected servers.",
        )
    };
    ToolDefinition::async_rich_tool(
        name,
        desc,
        json!({
            "type": "object",
            "properties": {
                "server": { "type": "string", "description": "Optional MCP server name filter" },
                "cursor": { "type": "string", "description": "Optional pagination cursor when filtering one server" }
            }
        }),
        move |args, call_ctx| async move {
            let mgr = call_ctx
                .ctx
                .get::<McpManager>("mcp")
                .ok_or("MCP 服务未就绪")?;
            let server_filter = args.get("server").and_then(Value::as_str);
            let cursor = args.get("cursor").and_then(Value::as_str);
            let clients = mgr.resource_clients(server_filter).await;

            let key = if templates {
                "resourceTemplates"
            } else {
                "resources"
            };
            let mut items = Vec::new();
            let mut next_cursor: Option<String> = None;

            for (srv_name, client) in clients {
                let page = if templates {
                    client
                        .list_resource_templates_page(cursor, &call_ctx.cancel)
                        .await?
                } else {
                    client.list_resources_page(cursor, &call_ctx.cancel).await?
                };
                if let Some(arr) = page.get(key).and_then(Value::as_array) {
                    for item in arr {
                        let mut obj = item.as_object().cloned().unwrap_or_default();
                        obj.remove("_meta");
                        obj.remove("icons");
                        obj.insert("server".into(), Value::String(srv_name.clone()));
                        items.push(Value::Object(obj));
                    }
                }
                if server_filter.is_some() {
                    next_cursor = page
                        .get("nextCursor")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
            }

            let mut payload = json!({ key: items });
            if let Some(srv) = server_filter {
                payload["server"] = Value::String(srv.to_string());
            }
            if let Some(nc) = next_cursor {
                payload["nextCursor"] = Value::String(nc);
            }

            let text = serde_json::to_string_pretty(&payload)?;
            Ok(ToolOutput::text(text).with_structured(payload))
        },
    )
    .with_exposure(ToolExposure::Direct)
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_annotations(ToolAnnotations {
        read_only: true,
        ..Default::default()
    })
}

fn make_read_resource_def() -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        READ_MCP_RESOURCE_TOOL,
        "Read an MCP resource by server name and URI.",
        json!({
            "type": "object",
            "properties": {
                "server": { "type": "string", "description": "MCP server name" },
                "uri": { "type": "string", "description": "Resource URI to read" }
            },
            "required": ["server", "uri"]
        }),
        move |args, call_ctx| async move {
            let server = args
                .get("server")
                .and_then(Value::as_str)
                .ok_or("缺少 server 参数")?;
            let uri = args
                .get("uri")
                .and_then(Value::as_str)
                .ok_or("缺少 uri 参数")?;
            let mgr = call_ctx
                .ctx
                .get::<McpManager>("mcp")
                .ok_or("MCP 服务未就绪")?;
            let clients = mgr.resource_clients(Some(server)).await;
            let Some((_, client)) = clients.into_iter().next() else {
                return Err(format!("Unknown or resource-less MCP server: {server}").into());
            };
            let res = client.read_resource(uri, &call_ctx.cancel).await?;
            let mut texts = Vec::new();
            if let Some(contents) = res.get("contents").and_then(Value::as_array) {
                for item in contents {
                    if let Some(t) = item.get("text").and_then(Value::as_str) {
                        texts.push(t.to_string());
                    } else if let Some(blob) = item.get("blob").and_then(Value::as_str) {
                        let path = write_temp_output("mizar-mcp-blob-", blob).await?;
                        texts.push(format!("[Binary resource saved to: {}]", path.display()));
                    }
                }
            }
            let text = if texts.is_empty() {
                res.to_string()
            } else {
                texts.join("\n")
            };
            Ok(ToolOutput::text(text).with_structured(res))
        },
    )
    .with_exposure(ToolExposure::Direct)
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_annotations(ToolAnnotations {
        read_only: true,
        ..Default::default()
    })
}

impl Service for McpManager {
    type Config = McpPluginConfig;

    fn service_name() -> &'static str {
        "mcp"
    }

    fn plugin_name() -> &'static str {
        "McpManager"
    }

    fn inject() -> Inject {
        Inject::from(["tools"])
    }

    fn create(ctx: &Context, mut config: Self::Config) -> Result<Arc<Self>, CordisError> {
        let mut loaded = config.load_servers();
        for error in loaded.errors.drain(..) {
            ctx.logger_named("mcp").warn(error);
        }
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            config: loaded,
            workspace: config.workspace,
            background_connect: config.background_connect,
            servers: AsyncMutex::new(HashMap::new()),
            statuses: Mutex::new(HashMap::new()),
            resource_cleanups: AsyncMutex::new(Vec::new()),
            connect_lock: AsyncMutex::new(()),
            connect_cancel: CancellationToken::new(),
            phase: Mutex::new(McpManagerPhase::Initializing),
            ready_notify: Arc::new(tokio::sync::Notify::new()),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let mgr_connect = Arc::clone(&self);
        let mgr_close = Arc::clone(&self);
        let ctx = ctx.clone();

        Box::pin(async move {
            ctx.effect("mcp.close-all", move || {
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    mgr_close.close_all().await;
                })))
            })?;

            let auto = self.config.auto_enable_codemode
                && self
                    .config
                    .entries
                    .iter()
                    .any(|s| s.config.uses_exposure(McpExposure::Codemode));
            ctx.tools()?
                .set_auto_default_active("codemode", auto.then_some(true));
            if let Some(summary) = self.render_servers_section() {
                let section = format!("<mcp_servers>\n{summary}\n</mcp_servers>");
                let fiber = ctx.try_inject(["systemPrompt"], move |c| {
                    let section = section.clone();
                    async move {
                        let sp = c.service::<SystemPrompt>("systemPrompt")?;
                        c.effect("mcp.servers-section", move || {
                            Ok::<_, CordisError>(Some(sp.section(PromptSection::new(
                                "mcp_servers",
                                35,
                                move || Some(section.clone()),
                            ))))
                        })?;
                        Ok::<_, CordisError>(())
                    }
                })?;
                if ctx.get::<SystemPrompt>("systemPrompt").is_some() {
                    fiber.wait_settled().await?;
                }
            }
            if mgr_connect.background_connect {
                tokio::spawn(async move {
                    mgr_connect.connect_configured().await;
                });
            } else {
                mgr_connect.connect_configured().await;
            }

            Ok(())
        })
    }
}

pub type McpPlugin = cordis::ServicePlugin<McpManager>;

pub trait McpContextExt {
    fn mcp(&self) -> Result<Arc<McpManager>, CordisError>;
}

impl McpContextExt for Context {
    fn mcp(&self) -> Result<Arc<McpManager>, CordisError> {
        self.service::<McpManager>("mcp")
    }
}
