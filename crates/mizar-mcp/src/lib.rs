//! MCP 插件：连接配置文件与程序注册的 MCP Server，将远程工具以 `mcp__<server>__<tool>` 注册进 `ToolRuntime`，
//! 并支持 `Direct` / `Codemode` / `Deferred` / `Hidden` 分级暴露、`list_mcp_resources` 资源工具及 `<mcp_servers>` 提示词。

pub mod client;
pub mod config;
pub mod transport;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cordis::{BoxError, BoxFuture, Cleanup, Context, CordisError, Inject, Logger, Service};
use mizar_system_prompt::{PromptSection, SystemPrompt};
use mizar_tools::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head, write_temp_output,
};
use mizar_tools::{
    ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolExposure, ToolNamespace, ToolOutput,
    ToolsContextExt,
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

#[derive(Debug, Clone, Default)]
pub struct McpPluginConfig {
    pub workspace: PathBuf,
    /// 可选覆盖全局配置目录（默认 `~/.mizar`，或 `MIZAR_HOME`）。
    pub global_config_dir: Option<PathBuf>,
    /// 是否从 `~/.mizar/mcp.json` 与 `<workspace>/.mizar/mcp.json` 自动加载配置（默认 true）。
    pub load_config_files: bool,
    /// 代码内显式追加/覆盖的 MCP Server 配置。
    pub servers: HashMap<String, McpServerConfig>,
    /// 连接时是否异步后台连接（默认 true，不阻塞首屏启动；测试可设 false 并调用 `wait_ready`）。
    pub background_connect: bool,
}

impl McpPluginConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            global_config_dir: None,
            load_config_files: true,
            servers: HashMap::new(),
            background_connect: true,
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
    state: McpConnectionState,
    cleanups: Vec<Cleanup>,
}

fn mcp_server_summary(description: Option<&str>, instructions: Option<&str>) -> String {
    let text = description
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| instructions.map(str::trim).filter(|s| !s.is_empty()))
        .unwrap_or("");
    text.lines().next().unwrap_or("").trim().to_string()
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
    config: McpPluginConfig,
    servers: AsyncMutex<HashMap<String, ConnectedServer>>,
    statuses: Mutex<HashMap<String, McpServerStatus>>,
    resource_cleanups: AsyncMutex<Vec<Cleanup>>,
    connect_lock: AsyncMutex<()>,
    ready_notify: Arc<tokio::sync::Notify>,
    initial_done: Mutex<bool>,
}

impl McpManager {
    fn logger(&self) -> Logger {
        self.ctx.logger_named("mcp")
    }

    pub async fn wait_ready(&self) {
        loop {
            let notified = self.ready_notify.notified();
            if *self.initial_done.lock().unwrap() {
                return;
            }
            notified.await;
        }
    }

    pub fn statuses(&self) -> Vec<McpServerStatus> {
        let mut list: Vec<McpServerStatus> =
            self.statuses.lock().unwrap().values().cloned().collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }

    pub fn render_servers_section(&self) -> Option<String> {
        let statuses = self.statuses();
        let mut indirect: Vec<&McpServerStatus> = statuses
            .iter()
            .filter(|s| {
                if matches!(s.state, McpConnectionState::Disabled) {
                    return false;
                }
                s.exposure == McpExposure::Codemode
                    || s.exposure == McpExposure::Deferred
                    || s.registered_tools.iter().any(|(_, exp)| {
                        matches!(exp, ToolExposure::Codemode | ToolExposure::Deferred)
                    })
            })
            .collect();
        if indirect.is_empty() {
            return None;
        }
        indirect.sort_by(|a, b| a.name.cmp(&b.name));

        let mut has_codemode = false;
        let mut has_deferred = false;
        let mut heads = Vec::with_capacity(indirect.len());
        for s in &indirect {
            let uses_codemode = s.exposure == McpExposure::Codemode
                || s.registered_tools
                    .iter()
                    .any(|(_, e)| *e == ToolExposure::Codemode);
            if uses_codemode {
                has_codemode = true;
            } else {
                has_deferred = true;
            }
            let reach = if uses_codemode {
                "codemode"
            } else {
                "tool_search"
            };
            heads.push(format!("- {} ({reach})", mcp_namespace(&s.name)));
        }

        let mut intro = String::from("MCP servers whose tools are not declared to you.");
        if has_codemode {
            intro.push_str(" Call the tools of `codemode` servers from codemode scripts.");
        }
        if has_deferred {
            intro.push_str(" Load the tools of `tool_search` servers with `tool_search`.");
        }

        let omitted = |count: usize| -> Option<String> {
            (count > 0).then(|| {
                let suffix = if count == 1 { "" } else { "s" };
                format!("- … {count} more server{suffix}; find their tools with searchTools()")
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

        let mut parts = vec![intro];
        for (idx, s) in indirect.iter().take(kept).enumerate() {
            let summary = if per_server > 0 {
                let first_line = mcp_server_summary(
                    s.description.as_deref(),
                    s.instructions.as_deref(),
                );
                truncate_chars(&first_line, per_server)
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
        let mut entries_map: HashMap<String, LoadedMcpEntry> = HashMap::new();

        if self.config.load_config_files {
            let global_dir = self
                .config
                .global_config_dir
                .clone()
                .or_else(resolve_global_config_dir);
            let loaded = load_mcp_config_with_global(&self.config.workspace, global_dir.as_deref());
            for err in loaded.errors {
                self.logger().warn(err);
            }
            for entry in loaded.entries {
                entries_map.insert(entry.name.clone(), entry);
            }
        }
        for (name, cfg) in &self.config.servers {
            entries_map.insert(
                name.clone(),
                LoadedMcpEntry {
                    name: name.clone(),
                    config: cfg.clone(),
                    source: "programmatic".to_string(),
                },
            );
        }

        let mut entries: Vec<LoadedMcpEntry> = entries_map.into_values().collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));

        {
            let mut st = self.statuses.lock().unwrap();
            st.clear();
            for entry in &entries {
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
            if !entry.config.enabled {
                continue;
            }
            if let Err(err) = self.connect_one(entry.clone()).await {
                self.logger()
                    .warn(format!("MCP server `{}` failed: {err}", entry.name));
                if let Some(st) = self.statuses.lock().unwrap().get_mut(&entry.name) {
                    st.state = McpConnectionState::Failed(err);
                }
            }
        }

        self.sync_resource_tools().await;
        *self.initial_done.lock().unwrap() = true;
        self.ready_notify.notify_waiters();
    }

    async fn connect_one(&self, entry: LoadedMcpEntry) -> Result<(), String> {
        let cancel = CancellationToken::new();
        let client = Arc::new(
            McpClient::connect(
                &entry.name,
                &entry.config,
                &self.config.workspace,
                self.logger(),
                &cancel,
            )
            .await
            .map_err(|e| e.to_string())?,
        );

        let remote_tools = client
            .list_all_tools(&cancel)
            .await
            .map_err(|e| e.to_string())?;

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
            st.state = state.clone();
            st.registered_tools = registered_tools.clone();
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
                state,
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
        let mut res = self.resource_cleanups.lock().await;
        for c in res.drain(..) {
            run_cleanup(c).await;
        }
        drop(res);

        let mut srvs = self.servers.lock().await;
        let drained: Vec<ConnectedServer> = srvs.drain().map(|(_, v)| v).collect();
        drop(srvs);

        for srv in drained {
            let _ = srv.state;
            for c in srv.cleanups {
                run_cleanup(c).await;
            }
            srv.client.close().await;
        }
    }

    async fn resource_clients(&self, filter_server: Option<&str>) -> Vec<(String, Arc<McpClient>)> {
        let srvs = self.servers.lock().await;
        let mut list: Vec<(String, Arc<McpClient>)> = srvs
            .values()
            .filter(|s| {
                s.client.has_resources
                    && filter_server.is_none_or(|f| f == s.entry.name)
            })
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

    let mut def = ToolDefinition::async_rich_tool(
        full_name,
        desc,
        params,
        move |args, call_ctx| {
            let client = Arc::clone(&client);
            let remote_name = remote_name.clone();
            async move {
                let res = client
                    .call_tool(&remote_name, args, &call_ctx.cancel)
                    .await?;
                format_mcp_tool_result(&res).await
            }
        },
    )
    .with_exposure(exposure)
    .with_execution_mode(exec_mode)
    .with_annotations(annotations)
    .with_namespace(namespace);

    if let Some(out_schema) = remote.output_schema.clone() {
        def = def.with_output_schema(out_schema);
    }
    def
}

async fn format_mcp_tool_result(res: &Value) -> Result<ToolOutput, BoxError> {
    let is_error = res.get("isError").and_then(Value::as_bool).unwrap_or(false);
    let structured = res.get("structuredContent").cloned();
    let mut parts = Vec::new();

    if let Some(blocks) = res.get("content").and_then(Value::as_array) {
        for block in blocks {
            let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
            match kind {
                "text" => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        parts.push(text.to_string());
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
    let content = if trunc.truncated {
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

    Ok(ToolOutput {
        content,
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

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            config,
            servers: AsyncMutex::new(HashMap::new()),
            statuses: Mutex::new(HashMap::new()),
            resource_cleanups: AsyncMutex::new(Vec::new()),
            connect_lock: AsyncMutex::new(()),
            ready_notify: Arc::new(tokio::sync::Notify::new()),
            initial_done: Mutex::new(false),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let mgr_connect = Arc::clone(&self);
        let mgr_close = Arc::clone(&self);
        let mgr_prompt = Arc::clone(&self);
        let ctx = ctx.clone();

        Box::pin(async move {
            if let Some(sp) = ctx.get::<SystemPrompt>("systemPrompt") {
                ctx.effect("mcp.system-prompt", move || {
                    let mgr = mgr_prompt;
                    Ok::<_, CordisError>(Some(sp.section(PromptSection::new(
                        "mcp_servers",
                        30,
                        move || mgr.render_servers_section(),
                    ))))
                })?;
            }

            ctx.effect("mcp.close-all", move || {
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    mgr_close.close_all().await;
                })))
            })?;

            if mgr_connect.config.background_connect {
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
