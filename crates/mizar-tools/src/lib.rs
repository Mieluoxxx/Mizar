//! 工具注册表、分层暴露策略与三段执行管线（对齐 `NamiDSH` + `pi` 工具装载机制）。
//!
//! 每次工具调用经过三道 `waterfall`：
//! - `tools/pre-execute`  → 执行前：拦截 / 检查（返回 `allow: false` 就不执行）
//! - `tools/execute`      → 执行：可包一层超时、监控
//! - `tools/post-execute` → 执行后：可改写结果或追加上下文提醒

pub mod mutation_queue;
pub mod truncate;

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::{Arc, Mutex};

use cordis::{BoxError, BoxFuture, Cleanup, Context, CordisError, Service};
use mizar_llm::ToolSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

pub const EVENT_TOOLS_PRE_EXECUTE: &str = "tools/pre-execute";
pub const EVENT_TOOLS_EXECUTE: &str = "tools/execute";
pub const EVENT_TOOLS_POST_EXECUTE: &str = "tools/post-execute";
pub const EVENT_TOOLS_CHANGE: &str = "tools/change";
pub const EVENT_TOOLS_RESULT: &str = "tools/result";

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool \"{0}\" already registered")]
    DuplicateTool(String),
}

/// 工具暴露级别：决定工具是否声明给模型、能否被 `codemode` 脚本调用、能否被 `tool_search` 检索激活。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ToolExposure {
    /// 激活时既声明给模型，也可被 `codemode` 脚本调用（内置核心工具默认模式）。
    #[default]
    Direct,
    /// 激活时声明给模型，但禁止被 `codemode` 脚本嵌套调用（适用于 `codemode`、`tool_search` 等编排工具）。
    ModelOnly,
    /// 已注册且可被 `codemode` 脚本调用；不直接声明给模型（除非被 `tool_search` 等激活）；会列在 `codemode` 描述中。
    Codemode,
    /// 与 `Codemode` 类似，但不会自动列在 `codemode` 描述中，主要通过 `tool_search` / `searchTools()` 发现激活。
    Deferred,
    /// 已注册但彻底不可达，无法激活，也不被脚本或模型调用。
    Hidden,
}

/// 工具并发执行模式。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ToolExecutionMode {
    #[default]
    Sequential,
    Parallel,
}

/// 语义提示注解（兼容 MCP ToolAnnotations）。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ToolAnnotations {
    pub read_only: bool,
    pub destructive: bool,
    pub idempotent: bool,
    pub open_world: bool,
}

/// 工具所属命名空间（例如 MCP Server）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolNamespace {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
}

impl ToolNamespace {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            aliases: Vec::new(),
            description: None,
            instructions: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecution {
    pub name: String,
    pub args: Value,
    pub session_id: Option<String>,
}

impl ToolExecution {
    pub fn new(name: impl Into<String>, args: Value, session_id: Option<String>) -> Self {
        Self {
            name: name.into(),
            args,
            session_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolContextNote {
    pub source: String,
    pub text: String,
}

fn is_false(v: &bool) -> bool {
    !*v
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub blocked: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_contexts: Vec<ToolContextNote>,
    /// 面向脚本/编排层的结构化返回值（如 `codemode` 直接消费 JSON）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    /// 面向 UI 展示的附加细节（例如 diff、exit_code、loaded_tools，不拼进发给模型的纯文本）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            blocked: false,
            additional_contexts: Vec::new(),
            structured_content: None,
            details: None,
        }
    }

    pub fn err(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            blocked: false,
            additional_contexts: Vec::new(),
            structured_content: None,
            details: None,
        }
    }

    pub fn denied(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            blocked: true,
            additional_contexts: Vec::new(),
            structured_content: None,
            details: None,
        }
    }

    pub fn from_output(output: ToolOutput) -> Self {
        Self {
            content: output.content,
            is_error: output.is_error,
            blocked: false,
            additional_contexts: Vec::new(),
            structured_content: output.structured_content,
            details: output.details,
        }
    }

    pub fn with_structured(mut self, structured: Value) -> Self {
        self.structured_content = Some(structured);
        self
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_context(mut self, source: impl Into<String>, text: impl Into<String>) -> Self {
        self.additional_contexts.push(ToolContextNote {
            source: source.into(),
            text: text.into(),
        });
        self
    }

    pub fn prepend_context(mut self, source: impl Into<String>, text: impl Into<String>) -> Self {
        self.additional_contexts.insert(
            0,
            ToolContextNote {
                source: source.into(),
                text: text.into(),
            },
        );
        self
    }
}

/// 工具执行返回的富输出结构。
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub structured_content: Option<Value>,
    pub details: Option<Value>,
}

impl ToolOutput {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            structured_content: None,
            details: None,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            structured_content: None,
            details: None,
        }
    }

    pub fn with_structured(mut self, value: Value) -> Self {
        self.structured_content = Some(value);
        self
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }
}

impl From<String> for ToolOutput {
    fn from(content: String) -> Self {
        Self::text(content)
    }
}

impl From<&str> for ToolOutput {
    fn from(content: &str) -> Self {
        Self::text(content)
    }
}

/// 工具执行期间流式上报给 UI 的增量进度。
#[derive(Debug, Clone)]
pub struct ToolUpdate {
    pub preview: String,
    pub details: Option<Value>,
}

pub type ToolUpdateCallback = Arc<dyn Fn(ToolUpdate) + Send + Sync>;

/// 传给富工具执行函数的上下文。
#[derive(Clone)]
pub struct ToolCallContext {
    pub exec: ToolExecution,
    pub cancel: CancellationToken,
    pub on_update: Option<ToolUpdateCallback>,
    pub ctx: Context,
}

impl ToolCallContext {
    pub fn session_id(&self) -> Option<&str> {
        self.exec.session_id.as_deref()
    }

    pub fn emit_update(&self, preview: impl Into<String>, details: Option<Value>) {
        if let Some(cb) = &self.on_update {
            cb(ToolUpdate {
                preview: preview.into(),
                details,
            });
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreExecuteDecision {
    pub allow: bool,
    pub reason: Option<String>,
}

impl PreExecuteDecision {
    pub fn allow() -> Self {
        Self {
            allow: true,
            reason: None,
        }
    }

    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            allow: false,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PostExecuteInput {
    pub exec: ToolExecution,
    pub result: ToolResult,
}

#[derive(Debug, Clone)]
pub struct ToolResultEvent {
    pub exec: ToolExecution,
    pub result: ToolResult,
}

/// 已注册工具的只读元数据视图（供 `tool_search`、`codemode`、`prepare_loadout` 与系统提示词读取）。
#[derive(Debug, Clone)]
pub struct ToolRuntimeInfo {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub output_schema: Option<Value>,
    pub exposure: ToolExposure,
    pub execution_mode: ToolExecutionMode,
    pub annotations: ToolAnnotations,
    pub namespace: Option<ToolNamespace>,
    pub default_active: Option<bool>,
    pub prompt_snippet: Option<String>,
    pub prompt_guidelines: Vec<String>,
}

impl ToolRuntimeInfo {
    pub fn to_schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

/// 传给 `prepare_loadout` 钩子的当前会话工具装配视图。
#[derive(Debug, Clone)]
pub struct ToolLoadout {
    pub declared: Vec<ToolRuntimeInfo>,
    pub callable: Vec<ToolRuntimeInfo>,
    pub registered: Vec<ToolRuntimeInfo>,
}

#[derive(Debug, Clone, Default)]
pub struct ToolLoadoutChanges {
    pub descriptions: HashMap<String, String>,
    pub hidden_declarations: Vec<String>,
}

pub type PrepareLoadoutFn =
    Arc<dyn Fn(&ToolLoadout) -> Option<ToolLoadoutChanges> + Send + Sync>;

pub type ToolRichExecFn =
    Arc<dyn Fn(Value, ToolCallContext) -> BoxFuture<'static, Result<ToolOutput, BoxError>> + Send + Sync>;

#[derive(Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub output_schema: Option<Value>,
    pub exposure: ToolExposure,
    pub execution_mode: ToolExecutionMode,
    pub annotations: ToolAnnotations,
    pub namespace: Option<ToolNamespace>,
    pub default_active: Option<bool>,
    pub prompt_snippet: Option<String>,
    pub prompt_guidelines: Vec<String>,
    pub prepare_loadout: Option<PrepareLoadoutFn>,
    pub execute: ToolRichExecFn,
}

impl ToolDefinition {
    pub fn async_tool<F, Fut, E>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        execute: F,
    ) -> Self
    where
        F: Fn(Value, ToolExecution) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, E>> + Send + 'static,
        E: Into<BoxError>,
    {
        Self::async_rich_tool(name, description, parameters, move |args, call_ctx| {
            let fut = execute(args, call_ctx.exec);
            async move { fut.await.map(ToolOutput::text).map_err(Into::into) }
        })
    }

    pub fn sync_tool<F, E>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        execute: F,
    ) -> Self
    where
        F: Fn(Value, ToolExecution) -> Result<String, E> + Send + Sync + 'static,
        E: Into<BoxError>,
    {
        Self::async_rich_tool(name, description, parameters, move |args, call_ctx| {
            let res: Result<ToolOutput, BoxError> =
                execute(args, call_ctx.exec).map(ToolOutput::text).map_err(Into::into);
            async move { res }
        })
    }

    pub fn async_rich_tool<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        execute: F,
    ) -> Self
    where
        F: Fn(Value, ToolCallContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutput, BoxError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            output_schema: None,
            exposure: ToolExposure::Direct,
            execution_mode: ToolExecutionMode::Sequential,
            annotations: ToolAnnotations::default(),
            namespace: None,
            default_active: None,
            prompt_snippet: None,
            prompt_guidelines: Vec::new(),
            prepare_loadout: None,
            execute: Arc::new(move |args, call_ctx| Box::pin(execute(args, call_ctx))),
        }
    }

    pub fn with_exposure(mut self, exposure: ToolExposure) -> Self {
        self.exposure = exposure;
        self
    }

    pub fn with_default_active(mut self, default_active: bool) -> Self {
        self.default_active = Some(default_active);
        self
    }

    pub fn with_execution_mode(mut self, mode: ToolExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    pub fn with_annotations(mut self, annotations: ToolAnnotations) -> Self {
        self.annotations = annotations;
        self
    }

    pub fn with_namespace(mut self, namespace: ToolNamespace) -> Self {
        self.namespace = Some(namespace);
        self
    }

    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    pub fn with_prompt_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.prompt_snippet = Some(snippet.into());
        self
    }

    pub fn with_prompt_guidelines(mut self, guidelines: Vec<String>) -> Self {
        self.prompt_guidelines = guidelines;
        self
    }

    pub fn with_prepare_loadout<F>(mut self, hook: F) -> Self
    where
        F: Fn(&ToolLoadout) -> Option<ToolLoadoutChanges> + Send + Sync + 'static,
    {
        self.prepare_loadout = Some(Arc::new(hook));
        self
    }

    fn info(&self) -> ToolRuntimeInfo {
        ToolRuntimeInfo {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            output_schema: self.output_schema.clone(),
            exposure: self.exposure,
            execution_mode: self.execution_mode,
            annotations: self.annotations,
            namespace: self.namespace.clone(),
            default_active: self.default_active,
            prompt_snippet: self.prompt_snippet.clone(),
            prompt_guidelines: self.prompt_guidelines.clone(),
        }
    }
}

/// 通配符模式匹配（支持 `*` 通配任意字符）。
pub fn wildcard_matches(pattern: &str, candidate: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == candidate;
    }
    let mut rest = candidate;
    for (idx, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if idx == 0 {
            if let Some(after) = rest.strip_prefix(part) {
                rest = after;
            } else {
                return false;
            }
        } else if idx == parts.len() - 1 && !pattern.ends_with('*') {
            return rest.ends_with(part);
        } else if let Some(pos) = rest.find(part) {
            rest = &rest[pos + part.len()..];
        } else {
            return false;
        }
    }
    true
}

#[derive(Default)]
struct ToolState {
    tools: Vec<ToolDefinition>,
    /// 全局默认激活覆盖（例如连接了 MCP server 后自动开启 `codemode` / `tool_search`）。
    global_active_overrides: HashMap<String, bool>,
    /// 若设置了 `Some(patterns)`，默认只激活匹配这些模式且未关闭的工具。
    include_patterns: Option<Vec<String>>,
    /// 屏蔽规则（即使注册或尝试激活也不对外暴露）。
    exclude_patterns: Vec<String>,
    /// 按会话增量激活的工具列表（如 `tool_search` 动态加载）。
    session_added: HashMap<String, Vec<String>>,
    /// 按会话显式替换过的激活集合。
    session_explicit: HashMap<String, Vec<String>>,
}

impl ToolState {
    fn is_excluded(&self, name: &str) -> bool {
        self.exclude_patterns
            .iter()
            .any(|p| wildcard_matches(p, name))
    }

    fn is_activatable(tool: &ToolDefinition, excluded: bool) -> bool {
        !excluded && tool.exposure != ToolExposure::Hidden
    }

    fn is_default_active(&self, tool: &ToolDefinition) -> bool {
        if !Self::is_activatable(tool, self.is_excluded(&tool.name)) {
            return false;
        }
        if let Some(&overridden) = self.global_active_overrides.get(&tool.name) {
            return overridden;
        }
        if let Some(include) = &self.include_patterns {
            return include.iter().any(|p| wildcard_matches(p, &tool.name));
        }
        tool.default_active
            .unwrap_or(matches!(tool.exposure, ToolExposure::Direct | ToolExposure::ModelOnly))
    }

    fn resolve_active_defs(&self, session_id: Option<&str>) -> Vec<ToolDefinition> {
        let mut by_name: HashMap<&str, &ToolDefinition> = HashMap::with_capacity(self.tools.len());
        for t in &self.tools {
            by_name.insert(t.name.as_str(), t);
        }

        if let Some(sid) = session_id
            && let Some(explicit) = self.session_explicit.get(sid)
        {
            let mut out = Vec::new();
            let mut seen = HashSet::new();
            for name in explicit {
                if seen.insert(name.as_str())
                    && let Some(def) = by_name.get(name.as_str())
                    && Self::is_activatable(def, self.is_excluded(name))
                {
                    out.push((*def).clone());
                }
            }
            return out;
        }

        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for t in &self.tools {
            if self.is_default_active(t) && seen.insert(t.name.as_str()) {
                out.push(t.clone());
            }
        }
        if let Some(sid) = session_id
            && let Some(added) = self.session_added.get(sid)
        {
            for name in added {
                if seen.insert(name.as_str())
                    && let Some(def) = by_name.get(name.as_str())
                    && Self::is_activatable(def, self.is_excluded(name))
                {
                    out.push((*def).clone());
                }
            }
        }
        out
    }

    fn resolve_callable_infos(&self, active_defs: &[ToolDefinition]) -> Vec<ToolRuntimeInfo> {
        let active_names: HashSet<&str> = active_defs.iter().map(|t| t.name.as_str()).collect();
        self.tools
            .iter()
            .filter(|t| {
                if self.is_excluded(&t.name) {
                    return false;
                }
                match t.exposure {
                    ToolExposure::Direct => active_names.contains(t.name.as_str()),
                    ToolExposure::Codemode | ToolExposure::Deferred => true,
                    ToolExposure::ModelOnly | ToolExposure::Hidden => false,
                }
            })
            .map(ToolDefinition::info)
            .collect()
    }

    fn is_callable_in_sandbox(&self, session_id: Option<&str>, tool_name: &str) -> bool {
        let active_defs = self.resolve_active_defs(session_id);
        let Some(tool) = self.tools.iter().find(|t| t.name == tool_name) else {
            return false;
        };
        if self.is_excluded(tool_name) {
            return false;
        }
        match tool.exposure {
            ToolExposure::Direct => active_defs.iter().any(|t| t.name == tool_name),
            ToolExposure::Codemode | ToolExposure::Deferred => true,
            ToolExposure::ModelOnly | ToolExposure::Hidden => false,
        }
    }
}

pub struct ToolRuntime {
    ctx: Context,
    state: Mutex<ToolState>,
}

impl ToolRuntime {
    /// 工具插件调用它登记自己。返回的 `Cleanup` 放入 `ctx.effect` 即可随插件自动注销。
    pub fn register(self: &Arc<Self>, tool: ToolDefinition) -> Result<Cleanup, ToolError> {
        let tool_name = tool.name.clone();
        {
            let mut st = self.state.lock().unwrap();
            if st.tools.iter().any(|t| t.name == tool_name) {
                return Err(ToolError::DuplicateTool(tool_name));
            }
            st.tools.push(tool);
        }
        self.ctx.emit(EVENT_TOOLS_CHANGE, ());

        let runtime = Arc::downgrade(self);
        Ok(Cleanup::sync(move || {
            if let Some(rt) = runtime.upgrade() {
                {
                    let mut st = rt.state.lock().unwrap();
                    if let Some(idx) = st.tools.iter().position(|t| t.name == tool_name) {
                        st.tools.remove(idx);
                    }
                    st.global_active_overrides.remove(&tool_name);
                    for list in st.session_added.values_mut() {
                        list.retain(|n| n != &tool_name);
                    }
                    for list in st.session_explicit.values_mut() {
                        list.retain(|n| n != &tool_name);
                    }
                }
                rt.ctx.emit(EVENT_TOOLS_CHANGE, ());
            }
        }))
    }

    /// 配置全局工具选择（白名单模式列表 + 黑名单屏蔽模式列表）。
    pub fn set_tool_selection(&self, include: Option<Vec<String>>, exclude: Vec<String>) {
        {
            let mut st = self.state.lock().unwrap();
            st.include_patterns = include;
            st.exclude_patterns = exclude;
        }
        self.ctx.emit(EVENT_TOOLS_CHANGE, ());
    }

    /// 全局覆盖某个工具的默认激活状态（例如接入 MCP server 后自动打开 `codemode` / `tool_search`）。
    pub fn set_global_active_override(&self, name: &str, active: bool) {
        {
            let mut st = self.state.lock().unwrap();
            st.global_active_overrides.insert(name.to_string(), active);
        }
        self.ctx.emit(EVENT_TOOLS_CHANGE, ());
    }

    /// 获取某会话当前激活的工具名列表（含默认激活工具与动态激活工具）。
    pub fn get_active_tools(&self, session_id: &str) -> Vec<String> {
        let st = self.state.lock().unwrap();
        st.resolve_active_defs(Some(session_id))
            .into_iter()
            .map(|t| t.name)
            .collect()
    }

    /// 设置会话级激活工具集合：保留仍处于激活集内的原顺序，新加入的工具追加到末尾，保证 Prompt Cache 稳定。
    pub fn set_active_tools(&self, session_id: &str, names: &[String]) -> Vec<String> {
        let result = {
            let mut st = self.state.lock().unwrap();
            let current: Vec<String> = st
                .resolve_active_defs(Some(session_id))
                .into_iter()
                .map(|t| t.name)
                .collect();
            let requested: HashSet<&str> = names.iter().map(String::as_str).collect();
            let mut merged: Vec<String> = current
                .into_iter()
                .filter(|n| requested.contains(n.as_str()))
                .collect();
            let mut seen: HashSet<String> = merged.iter().cloned().collect();
            for name in names {
                if seen.insert(name.clone()) {
                    merged.push(name.clone());
                }
            }
            let valid: Vec<String> = merged
                .into_iter()
                .filter(|n| {
                    st.tools
                        .iter()
                        .find(|t| &t.name == n)
                        .is_some_and(|t| ToolState::is_activatable(t, st.is_excluded(n)))
                })
                .collect();
            st.session_explicit
                .insert(session_id.to_string(), valid.clone());
            st.session_added.remove(session_id);
            valid
        };
        self.ctx.emit(EVENT_TOOLS_CHANGE, ());
        result
    }

    /// 在不破坏现有激活集合的前提下，增量激活一批工具（常用于 `tool_search`）。返回本次新增的工具名。
    pub fn activate_tools(&self, session_id: &str, names: &[String]) -> Vec<String> {
        let added = {
            let mut st = self.state.lock().unwrap();
            let current_set: HashSet<String> = st
                .resolve_active_defs(Some(session_id))
                .into_iter()
                .map(|t| t.name)
                .collect();
            let mut newly_added = Vec::new();
            for name in names {
                if current_set.contains(name) {
                    continue;
                }
                let ok = st
                    .tools
                    .iter()
                    .find(|t| &t.name == name)
                    .is_some_and(|t| ToolState::is_activatable(t, st.is_excluded(name)));
                if ok && !newly_added.contains(name) {
                    newly_added.push(name.clone());
                }
            }
            if !newly_added.is_empty() {
                if let Some(explicit) = st.session_explicit.get_mut(session_id) {
                    explicit.extend(newly_added.iter().cloned());
                } else {
                    st.session_added
                        .entry(session_id.to_string())
                        .or_default()
                        .extend(newly_added.iter().cloned());
                }
            }
            newly_added
        };
        if !added.is_empty() {
            self.ctx.emit(EVENT_TOOLS_CHANGE, ());
        }
        added
    }

    /// 给模型看的工具清单（无特定会话上下文时返回默认声明集）。
    pub fn list(&self) -> Vec<ToolSchema> {
        self.list_declared(None)
    }

    /// 给模型看的会话级声明工具清单（执行 `prepare_loadout`、动态替换描述并过滤 `hidden_declarations`）。
    pub fn list_declared(&self, session_id: Option<&str>) -> Vec<ToolSchema> {
        self.list_declared_infos(session_id)
            .into_iter()
            .map(|info| info.to_schema())
            .collect()
    }

    pub fn list_declared_infos(&self, session_id: Option<&str>) -> Vec<ToolRuntimeInfo> {
        let (active_defs, callable, registered) = {
            let st = self.state.lock().unwrap();
            let active_defs = st.resolve_active_defs(session_id);
            let callable = st.resolve_callable_infos(&active_defs);
            let registered: Vec<ToolRuntimeInfo> = st.tools.iter().map(ToolDefinition::info).collect();
            (active_defs, callable, registered)
        };

        let declared_infos: Vec<ToolRuntimeInfo> =
            active_defs.iter().map(ToolDefinition::info).collect();
        let hooks: Vec<PrepareLoadoutFn> = active_defs
            .iter()
            .filter_map(|t| t.prepare_loadout.clone())
            .collect();

        if hooks.is_empty() {
            return declared_infos;
        }

        let loadout = ToolLoadout {
            declared: declared_infos.clone(),
            callable,
            registered,
        };
        let mut description_overrides: HashMap<String, String> = HashMap::new();
        let mut hidden: HashSet<String> = HashSet::new();

        for hook in hooks {
            if let Some(changes) = hook(&loadout) {
                for (name, desc) in changes.descriptions {
                    description_overrides.insert(name, desc);
                }
                for name in changes.hidden_declarations {
                    hidden.insert(name);
                }
            }
        }

        declared_infos
            .into_iter()
            .filter(|info| !hidden.contains(&info.name))
            .map(|mut info| {
                if let Some(desc) = description_overrides.remove(&info.name) {
                    info.description = desc;
                }
                info
            })
            .collect()
    }

    /// 供 `codemode` 沙箱使用的可调用工具列表（包含当前活跃 `Direct` 工具以及所有 `Codemode`/`Deferred` 工具）。
    pub fn list_callable(&self, session_id: Option<&str>) -> Vec<ToolRuntimeInfo> {
        let st = self.state.lock().unwrap();
        let active_defs = st.resolve_active_defs(session_id);
        st.resolve_callable_infos(&active_defs)
    }

    /// 全局已注册且未被 `exclude` 屏蔽的全部工具元数据（供 `tool_search` 建立检索索引）。
    pub fn list_all(&self) -> Vec<ToolRuntimeInfo> {
        let st = self.state.lock().unwrap();
        st.tools
            .iter()
            .filter(|t| !st.is_excluded(&t.name))
            .map(ToolDefinition::info)
            .collect()
    }

    pub fn execution_mode(&self, name: &str) -> ToolExecutionMode {
        let st = self.state.lock().unwrap();
        st.tools
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.execution_mode)
            .unwrap_or_default()
    }

    fn find_tool(&self, name: &str) -> Option<ToolDefinition> {
        let st = self.state.lock().unwrap();
        if st.is_excluded(name) {
            return None;
        }
        st.tools.iter().find(|t| t.name == name).cloned()
    }

    pub async fn execute(self: &Arc<Self>, exec: ToolExecution) -> ToolResult {
        self.execute_with(exec, CancellationToken::new(), None).await
    }

    pub async fn execute_with(
        self: &Arc<Self>,
        exec: ToolExecution,
        cancel: CancellationToken,
        on_update: Option<ToolUpdateCallback>,
    ) -> ToolResult {
        self.execute_internal(exec, cancel, on_update, false).await
    }

    /// 供 `codemode` 沙箱发出的嵌套工具调用：禁止调用 `ModelOnly`/`Hidden`/未激活的工具，仍完整走三段管线。
    pub async fn execute_nested(
        self: &Arc<Self>,
        exec: ToolExecution,
        cancel: CancellationToken,
        on_update: Option<ToolUpdateCallback>,
    ) -> ToolResult {
        self.execute_internal(exec, cancel, on_update, true).await
    }

    async fn execute_internal(
        self: &Arc<Self>,
        exec: ToolExecution,
        cancel: CancellationToken,
        on_update: Option<ToolUpdateCallback>,
        nested: bool,
    ) -> ToolResult {
        let decision: PreExecuteDecision = self
            .ctx
            .waterfall(
                EVENT_TOOLS_PRE_EXECUTE,
                exec.clone(),
                |_| -> BoxFuture<'static, PreExecuteDecision> {
                    Box::pin(async { PreExecuteDecision::allow() })
                },
            )
            .await;

        let result = if !decision.allow {
            ToolResult::denied(format!(
                "已拦截：{} {}",
                decision.reason.as_deref().unwrap_or("不允许调用"),
                exec.name
            ))
        } else if nested && !self.can_call_nested(exec.session_id.as_deref(), &exec.name) {
            ToolResult::err(format!("沙箱脚本无权调用工具：{}", exec.name))
        } else {
            let runtime = Arc::clone(self);
            let cancel_for_exec = cancel.clone();
            let update_for_exec = on_update.clone();
            self.ctx
                .waterfall(
                    EVENT_TOOLS_EXECUTE,
                    exec.clone(),
                    move |exec| -> BoxFuture<'static, ToolResult> {
                        let runtime = Arc::clone(&runtime);
                        let cancel = cancel_for_exec.clone();
                        let on_update = update_for_exec.clone();
                        Box::pin(async move {
                            let Some(tool) = runtime.find_tool(&exec.name) else {
                                return ToolResult::err(format!("没有这个工具：{}", exec.name));
                            };
                            if tool.exposure == ToolExposure::Hidden {
                                return ToolResult::err(format!("没有这个工具：{}", exec.name));
                            }
                            let args = exec.args.clone();
                            let call_ctx = ToolCallContext {
                                exec,
                                cancel,
                                on_update,
                                ctx: runtime.ctx.clone(),
                            };
                            match (tool.execute)(args, call_ctx).await {
                                Ok(output) => ToolResult::from_output(output),
                                Err(err) => ToolResult::err(format!("工具出错：{err}")),
                            }
                        })
                    },
                )
                .await
        };

        let result = self
            .ctx
            .waterfall(
                EVENT_TOOLS_POST_EXECUTE,
                PostExecuteInput {
                    exec: exec.clone(),
                    result,
                },
                |input| -> BoxFuture<'static, ToolResult> { Box::pin(async move { input.result }) },
            )
            .await;

        self.ctx.emit(
            EVENT_TOOLS_RESULT,
            ToolResultEvent {
                exec,
                result: result.clone(),
            },
        );
        result
    }

    fn can_call_nested(&self, session_id: Option<&str>, name: &str) -> bool {
        let st = self.state.lock().unwrap();
        st.is_callable_in_sandbox(session_id, name)
    }
}

impl Service for ToolRuntime {
    type Config = ();

    fn service_name() -> &'static str {
        "tools"
    }

    fn plugin_name() -> &'static str {
        "ToolRuntime"
    }

    fn create(ctx: &Context, _config: Self::Config) -> Result<Arc<Self>, CordisError> {
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            state: Mutex::new(ToolState::default()),
        }))
    }
}

pub trait ToolsContextExt {
    fn tools(&self) -> Result<Arc<ToolRuntime>, CordisError>;
}

impl ToolsContextExt for Context {
    fn tools(&self) -> Result<Arc<ToolRuntime>, CordisError> {
        self.service::<ToolRuntime>("tools")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wildcard_matches_patterns() {
        assert!(wildcard_matches("*", "anything"));
        assert!(wildcard_matches("mcp__fff__*", "mcp__fff__grep"));
        assert!(wildcard_matches("*_grep", "multi_grep"));
        assert!(!wildcard_matches("mcp__fff__*", "grep"));
    }

    #[tokio::test]
    async fn manages_exposure_activation_loadout_and_nested_guards() {
        let app = Context::new();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        let tools = app.tools().unwrap();

        let _c1 = tools
            .register(
                ToolDefinition::sync_tool("read", "read file", json!({}), |_, _| Ok::<_, BoxError>("text".into()))
                    .with_execution_mode(ToolExecutionMode::Parallel),
            )
            .unwrap();
        let _c2 = tools
            .register(
                ToolDefinition::sync_tool("mcp_secret", "secret op", json!({}), |_, _| {
                    Ok::<_, BoxError>("secret".into())
                })
                .with_exposure(ToolExposure::Deferred),
            )
            .unwrap();
        let _c3 = tools
            .register(
                ToolDefinition::sync_tool("codemode", "run js", json!({}), |_, _| {
                    Ok::<_, BoxError>("done".into())
                })
                .with_exposure(ToolExposure::ModelOnly)
                .with_prepare_loadout(|loadout| {
                    let mut descriptions = HashMap::new();
                    let names: Vec<&str> = loadout.callable.iter().map(|t| t.name.as_str()).collect();
                    descriptions.insert("codemode".to_string(), format!("callable: {}", names.join(",")));
                    Some(ToolLoadoutChanges {
                        descriptions,
                        hidden_declarations: Vec::new(),
                    })
                }),
            )
            .unwrap();

        let declared = tools.list_declared(Some("s1"));
        assert_eq!(declared.len(), 2);
        assert_eq!(declared[0].name, "read");
        assert_eq!(declared[1].name, "codemode");
        assert_eq!(declared[1].description, "callable: read,mcp_secret");

        let newly = tools.activate_tools("s1", &["mcp_secret".to_string()]);
        assert_eq!(newly, vec!["mcp_secret".to_string()]);
        let declared_after = tools.list_declared(Some("s1"));
        assert_eq!(
            declared_after.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["read", "codemode", "mcp_secret"]
        );

        let nested_ok = tools
            .execute_nested(
                ToolExecution::new("mcp_secret", json!({}), Some("s1".into())),
                CancellationToken::new(),
                None,
            )
            .await;
        assert!(!nested_ok.is_error);

        let nested_denied = tools
            .execute_nested(
                ToolExecution::new("codemode", json!({}), Some("s1".into())),
                CancellationToken::new(),
                None,
            )
            .await;
        assert!(nested_denied.is_error);
    }
}
