//! JSON-RPC 2.0 跨端通信协议契约（NDJSON over Unix Socket / WebSocket）。
//!
//! 严格保持纯数据结构（仅依赖 `serde` / `serde_json` 与同 crate 纯契约），
//! 不引入任何异步运行时、`mizar-session` 或终端依赖，供 `mizar-client`、`mizar-tui` 与外部前端共用。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::{
    ApprovalDecision, ApprovalMode, McpSummaryRow, SessionTreeEntry, SkillPickerItem,
    ToolSummaryRow,
};
use crate::llm::{ProviderInfo, ReasoningEffort};

pub const JSONRPC_VERSION: &str = "2.0";

pub const ERR_PARSE_ERROR: i64 = -32700;
pub const ERR_INVALID_REQUEST: i64 = -32600;
pub const ERR_METHOD_NOT_FOUND: i64 = -32601;
pub const ERR_INVALID_PARAMS: i64 = -32602;
pub const ERR_INTERNAL_ERROR: i64 = -32603;

pub const ERR_SESSION_LOCKED: i64 = -32001;
pub const ERR_SESSION_NOT_FOUND: i64 = -32002;

pub const METHOD_SESSION_SUBSCRIBE: &str = "session.subscribe";
pub const METHOD_TURN_SUBMIT: &str = "turn.submit";
pub const METHOD_TURN_CANCEL: &str = "turn.cancel";
pub const METHOD_QUEUE_REMOVE: &str = "queue.remove";
pub const METHOD_QUEUE_POP_ALL: &str = "queue.pop_all";
pub const METHOD_INTERACTION_RESOLVE: &str = "interaction.resolve";
pub const METHOD_SESSION_CONFIG: &str = "session.config";
pub const METHOD_SESSION_LIST: &str = "session.list";
pub const METHOD_SESSION_EVENTS: &str = "session.events";
pub const METHOD_SESSION_FORK: &str = "session.fork";
pub const METHOD_SESSION_TREE: &str = "session.tree";
pub const METHOD_CAPABILITIES_LIST: &str = "capabilities.list";

pub const NOTIF_TURN_STATE: &str = "turn.state";
pub const NOTIF_STREAM_DELTA: &str = "stream.delta";
pub const NOTIF_EVENT_COMMITTED: &str = "event.committed";
pub const NOTIF_INTERACTION_REQUESTED: &str = "interaction.requested";
pub const NOTIF_INTERACTION_RESOLVED: &str = "interaction.resolved";
pub const NOTIF_QUEUE_UPDATED: &str = "queue.updated";
pub const NOTIF_SUBAGENT_UPDATED: &str = "subagent.updated";
pub const NOTIF_SESSION_SWITCHED: &str = "session.switched";

/// JSON-RPC 2.0 请求/响应关联标识符。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcId {
    String(String),
    Number(i64),
}

impl From<i64> for JsonRpcId {
    fn from(value: i64) -> Self {
        Self::Number(value)
    }
}

impl From<String> for JsonRpcId {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for JsonRpcId {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

/// JSON-RPC 2.0 错误载荷。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    pub fn with_data(code: i64, message: impl Into<String>, data: Value) -> Self {
        Self {
            code,
            message: message.into(),
            data: Some(data),
        }
    }

    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::new(ERR_PARSE_ERROR, message)
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ERR_INVALID_REQUEST, message)
    }

    pub fn method_not_found(method: &str) -> Self {
        Self::new(ERR_METHOD_NOT_FOUND, format!("Method not found: {method}"))
    }

    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::new(ERR_INVALID_PARAMS, message)
    }

    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(ERR_INTERNAL_ERROR, message)
    }

    pub fn session_locked(message: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            code: ERR_SESSION_LOCKED,
            message: message.into(),
            data,
        }
    }

    pub fn session_not_found(session_id: &str) -> Self {
        Self::new(
            ERR_SESSION_NOT_FOUND,
            format!("Session not found: {session_id}"),
        )
    }
}

/// JSON-RPC 2.0 请求帧（NDJSON 单行）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<JsonRpcId>,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    pub fn new(id: JsonRpcId, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(id),
            method: method.into(),
            params,
        }
    }

    pub fn is_valid_version(&self) -> bool {
        self.jsonrpc == JSONRPC_VERSION
    }
}

/// JSON-RPC 2.0 响应帧（`result` 与 `error` 互斥）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<JsonRpcId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    pub fn ok(id: JsonRpcId, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(id),
            result: Some(result),
            error: None,
        }
    }

    pub fn ok_typed<T: Serialize>(id: JsonRpcId, result: &T) -> Result<Self, serde_json::Error> {
        Ok(Self::ok(id, serde_json::to_value(result)?))
    }

    pub fn err(id: Option<JsonRpcId>, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// JSON-RPC 2.0 服务端主动推送通知帧（无 `id`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcNotification {
    pub fn new(method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
        }
    }
}

/// 会话运行态指示（驱动多端顶栏/状态灯）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionRuntimeState {
    #[default]
    Idle,
    Streaming,
    WaitingApproval,
}

impl SessionRuntimeState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Streaming => "streaming",
            Self::WaitingApproval => "waiting_approval",
        }
    }
}

/// RPC 线协议审批模式（固定序列化为 `"auto" | "ask"`）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RpcApprovalMode {
    #[default]
    Auto,
    Ask,
}

impl RpcApprovalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ask => "ask",
        }
    }
}

impl From<ApprovalMode> for RpcApprovalMode {
    fn from(mode: ApprovalMode) -> Self {
        match mode {
            ApprovalMode::Auto => Self::Auto,
            ApprovalMode::Ask => Self::Ask,
        }
    }
}

impl From<RpcApprovalMode> for ApprovalMode {
    fn from(mode: RpcApprovalMode) -> Self {
        match mode {
            RpcApprovalMode::Auto => Self::Auto,
            RpcApprovalMode::Ask => Self::Ask,
        }
    }
}

/// RPC 线协议审批决策枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RpcApprovalDecision {
    AllowOnce,
    #[serde(alias = "allow_always_for_tool")]
    AllowAlways,
    Deny,
}

impl RpcApprovalDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AllowOnce => "allow_once",
            Self::AllowAlways => "allow_always",
            Self::Deny => "deny",
        }
    }

    pub fn to_approval_decision(self, note: Option<String>) -> ApprovalDecision {
        match self {
            Self::AllowOnce => ApprovalDecision::AllowOnce,
            Self::AllowAlways => ApprovalDecision::AllowAlwaysForTool,
            Self::Deny => match note.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()) {
                Some(reason) => ApprovalDecision::DenyWithReason(reason),
                None => ApprovalDecision::Deny,
            },
        }
    }

    pub fn from_approval_decision(decision: &ApprovalDecision) -> (Self, Option<String>) {
        match decision {
            ApprovalDecision::AllowOnce => (Self::AllowOnce, None),
            ApprovalDecision::AllowAlwaysForTool => (Self::AllowAlways, None),
            ApprovalDecision::Deny => (Self::Deny, None),
            ApprovalDecision::DenyWithReason(reason) => (Self::Deny, Some(reason.clone())),
        }
    }
}

/// 已落盘事件可见性（纯协议镜像，不耦合 `mizar-session`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RpcEventVisibility {
    Surface,
    #[serde(alias = "log_only")]
    LogOnly,
}

impl RpcEventVisibility {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::LogOnly => "log-only",
        }
    }
}

/// `session.subscribe` 请求参数。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSubscribeParams {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_seq: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// 会话基础元信息快照。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMetaDto {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork_seq: Option<u64>,
}

/// 当前会话运行配置快照。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfigDto {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub approval_mode: RpcApprovalMode,
}

/// 会话上下文窗口使用量快照。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionContextUsageDto {
    pub used_tokens: u64,
    #[serde(default)]
    pub max_tokens: Option<u64>,
}

/// 已提交落盘的规范化事件 DTO（零 `mizar-session` 依赖）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommittedEventDto {
    pub session_id: String,
    pub seq: u64,
    pub ts_ms: u64,
    pub visibility: RpcEventVisibility,
    pub kind: String,
    pub payload: Value,
}

/// 正在执行中的工具调用瞬态视图。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunningToolDto {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub output_tail: String,
}

/// 内存中尚未落盘的当前回合生成现场。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveTurnDto {
    pub turn: u64,
    pub step: usize,
    #[serde(default)]
    pub reasoning: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub running_tools: Vec<RunningToolDto>,
}

/// 等待人工交互决策（如审批确认）的挂起条目。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingInteractionDto {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub request_id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub tool: String,
    pub summary: String,
    #[serde(default)]
    pub args: Value,
    #[serde(default)]
    pub options: Vec<String>,
}

/// 消息附件引用或内联数据。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RpcAttachmentDto {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

/// 内核统一输入队列条目。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedInputDto {
    pub id: u64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<RpcAttachmentDto>,
    pub created_at_ms: u64,
}

/// 子代理权限模式镜像。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RpcSubagentMode {
    #[default]
    ReadOnly,
    Edit,
}

/// 子代理运行状态镜像。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RpcSubagentStatus {
    Running,
    Completed,
    MaxSteps,
    Truncated,
    Failed,
    TimedOut,
    Stopped,
}

/// 后台子代理状态快照 DTO（纯数据镜像，不耦合 `mizar-core`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentSnapshotDto {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    pub child_session_id: String,
    pub mode: RpcSubagentMode,
    #[serde(default)]
    pub prompt_preview: String,
    pub status: RpcSubagentStatus,
    #[serde(default)]
    pub steps: usize,
    #[serde(default)]
    pub current_step: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_tool: Option<String>,
    #[serde(default)]
    pub idle_secs: u64,
    #[serde(default)]
    pub watchdog_alerts: usize,
    #[serde(default)]
    pub elapsed_secs: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
}

/// `session.subscribe` 响应的完整会话水合快照。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSubscribeSnapshot {
    pub session: SessionMetaDto,
    pub state: SessionRuntimeState,
    pub config: SessionConfigDto,
    pub context: SessionContextUsageDto,
    #[serde(default)]
    pub events: Vec<CommittedEventDto>,
    #[serde(default)]
    pub active_turn: Option<ActiveTurnDto>,
    #[serde(default)]
    pub pending_interactions: Vec<PendingInteractionDto>,
    #[serde(default)]
    pub queued_inputs: Vec<QueuedInputDto>,
    #[serde(default)]
    pub subagents: Vec<SubagentSnapshotDto>,
}

/// `turn.state` 推送载荷。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnStateNotification {
    pub session_id: String,
    pub turn: u64,
    pub step: usize,
    pub state: SessionRuntimeState,
}

/// `stream.delta` 增量分片类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamDeltaKind {
    Text,
    Reasoning,
    ToolArgs,
    ToolOutput,
}

/// `stream.delta` 瞬态流式推送载荷（仅存内存，不写 SQLite）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamDeltaNotification {
    pub session_id: String,
    pub turn: u64,
    pub step: usize,
    pub kind: StreamDeltaKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub delta: String,
}

/// 交互裁决来源端标识。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedBy {
    Tui,
    Rpc,
}

impl ResolvedBy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tui => "tui",
            Self::Rpc => "rpc",
        }
    }
}

/// `interaction.resolved` 广播载荷。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractionResolvedNotification {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub request_id: String,
    pub decision: RpcApprovalDecision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub resolved_by: ResolvedBy,
    pub ts_ms: u64,
}

/// `queue.updated` 广播载荷。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueUpdatedNotification {
    pub session_id: String,
    pub items: Vec<QueuedInputDto>,
}

/// `subagent.updated` 广播载荷。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentUpdatedNotification {
    pub session_id: String,
    pub task: SubagentSnapshotDto,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notice: Option<String>,
}

/// `session.switched` 广播载荷（TUI 执行 `/resume` 或 `/new` 切换活跃会话时发出）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSwitchedNotification {
    pub old_session_id: String,
    pub new_session_id: String,
}

/// 服务端实时推送的 8 类强类型通知集合。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum RpcNotification {
    #[serde(rename = "turn.state")]
    TurnState(TurnStateNotification),
    #[serde(rename = "stream.delta")]
    StreamDelta(StreamDeltaNotification),
    #[serde(rename = "event.committed")]
    EventCommitted(CommittedEventDto),
    #[serde(rename = "interaction.requested")]
    InteractionRequested(PendingInteractionDto),
    #[serde(rename = "interaction.resolved")]
    InteractionResolved(InteractionResolvedNotification),
    #[serde(rename = "queue.updated")]
    QueueUpdated(QueueUpdatedNotification),
    #[serde(rename = "subagent.updated")]
    SubagentUpdated(SubagentUpdatedNotification),
    #[serde(rename = "session.switched")]
    SessionSwitched(SessionSwitchedNotification),
}

impl RpcNotification {
    pub fn method(&self) -> &'static str {
        match self {
            Self::TurnState(_) => NOTIF_TURN_STATE,
            Self::StreamDelta(_) => NOTIF_STREAM_DELTA,
            Self::EventCommitted(_) => NOTIF_EVENT_COMMITTED,
            Self::InteractionRequested(_) => NOTIF_INTERACTION_REQUESTED,
            Self::InteractionResolved(_) => NOTIF_INTERACTION_RESOLVED,
            Self::QueueUpdated(_) => NOTIF_QUEUE_UPDATED,
            Self::SubagentUpdated(_) => NOTIF_SUBAGENT_UPDATED,
            Self::SessionSwitched(_) => NOTIF_SESSION_SWITCHED,
        }
    }

    pub fn to_jsonrpc(&self) -> JsonRpcNotification {
        let params = match self {
            Self::TurnState(p) => serde_json::to_value(p),
            Self::StreamDelta(p) => serde_json::to_value(p),
            Self::EventCommitted(p) => serde_json::to_value(p),
            Self::InteractionRequested(p) => serde_json::to_value(p),
            Self::InteractionResolved(p) => serde_json::to_value(p),
            Self::QueueUpdated(p) => serde_json::to_value(p),
            Self::SubagentUpdated(p) => serde_json::to_value(p),
            Self::SessionSwitched(p) => serde_json::to_value(p),
        }
        .expect("RpcNotification payload serialization must not fail");
        JsonRpcNotification::new(self.method(), Some(params))
    }

    pub fn from_jsonrpc(notif: &JsonRpcNotification) -> Result<Self, serde_json::Error> {
        let params = notif.params.clone().unwrap_or(Value::Null);
        match notif.method.as_str() {
            NOTIF_TURN_STATE => Ok(Self::TurnState(serde_json::from_value(params)?)),
            NOTIF_STREAM_DELTA => Ok(Self::StreamDelta(serde_json::from_value(params)?)),
            NOTIF_EVENT_COMMITTED => Ok(Self::EventCommitted(serde_json::from_value(params)?)),
            NOTIF_INTERACTION_REQUESTED => {
                Ok(Self::InteractionRequested(serde_json::from_value(params)?))
            }
            NOTIF_INTERACTION_RESOLVED => {
                Ok(Self::InteractionResolved(serde_json::from_value(params)?))
            }
            NOTIF_QUEUE_UPDATED => Ok(Self::QueueUpdated(serde_json::from_value(params)?)),
            NOTIF_SUBAGENT_UPDATED => Ok(Self::SubagentUpdated(serde_json::from_value(params)?)),
            NOTIF_SESSION_SWITCHED => Ok(Self::SessionSwitched(serde_json::from_value(params)?)),
            unknown => Err(serde::de::Error::custom(format!(
                "unknown notification method: {unknown}"
            ))),
        }
    }
}

/// `turn.submit` 发送模式。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmitMode {
    #[default]
    Send,
    Queue,
    Steer,
}

/// `turn.submit` 请求参数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnSubmitParams {
    pub session_id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Vec<RpcAttachmentDto>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<SubmitMode>,
}

impl TurnSubmitParams {
    pub fn effective_mode(&self) -> SubmitMode {
        self.mode.unwrap_or_default()
    }
}

/// `turn.submit` 响应结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnSubmitResult {
    pub accepted: bool,
    pub mode: SubmitMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<u64>,
}

/// `turn.cancel` 请求参数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnCancelParams {
    pub session_id: String,
}

/// `turn.cancel` 响应结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnCancelResult {
    pub cancelled: bool,
}

/// `queue.remove` 请求参数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueRemoveParams {
    pub session_id: String,
    pub id: u64,
}

/// `queue.remove` 响应结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueRemoveResult {
    pub removed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item: Option<QueuedInputDto>,
}

/// `queue.pop_all` 请求参数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuePopAllParams {
    pub session_id: String,
}

/// `queue.pop_all` 响应结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuePopAllResult {
    pub items: Vec<QueuedInputDto>,
}

/// `interaction.resolve` 请求参数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractionResolveParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub request_id: String,
    pub decision: RpcApprovalDecision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// `interaction.resolve` 响应结果（支持并发抢占幂等应答）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractionResolveResult {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl InteractionResolveResult {
    pub fn ok() -> Self {
        Self {
            ok: true,
            reason: None,
        }
    }

    pub fn already_resolved() -> Self {
        Self {
            ok: false,
            reason: Some("already_resolved".to_string()),
        }
    }

    pub fn rejected(reason: impl Into<String>) -> Self {
        Self {
            ok: false,
            reason: Some(reason.into()),
        }
    }
}

/// `session.config` 请求参数（仅传入需修改的字段）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfigParams {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_mode: Option<RpcApprovalMode>,
}

/// `session.config` 响应结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfigResult {
    pub config: SessionConfigDto,
    pub context: SessionContextUsageDto,
}

/// `session.list` 请求参数（轻量按 `cwd`/关键字过滤，不走 FTS5）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionListParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_subagents: Option<bool>,
}

/// `session.list` 单条会话摘要。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionListItemDto {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork_seq: Option<u64>,
    pub event_count: u64,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_writer_pid: Option<u32>,
}

/// `session.list` 响应结果。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionListResult {
    pub sessions: Vec<SessionListItemDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// `session.events` 双向分页请求参数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEventsParams {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_seq: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_seq: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// `session.events` 响应结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEventsResult {
    pub session_id: String,
    pub events: Vec<CommittedEventDto>,
    pub has_more: bool,
}

/// `session.fork` 请求参数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionForkParams {
    pub session_id: String,
    pub at_seq: u64,
}

/// `session.fork` 响应结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionForkResult {
    pub session: SessionMetaDto,
}

/// `session.tree` 请求参数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTreeParams {
    pub session_id: String,
}

/// `session.tree` 响应结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTreeResult {
    pub session_id: String,
    pub entries: Vec<SessionTreeEntry>,
}

/// `capabilities.list` 请求参数。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitiesListParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// 可用 `/` 斜杠命令描述。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlashCommandDto {
    pub command: String,
    pub description: String,
    pub takes_arg: bool,
}

/// `capabilities.list` 响应结果。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilitiesListResult {
    #[serde(default)]
    pub commands: Vec<SlashCommandDto>,
    #[serde(default)]
    pub skills: Vec<SkillPickerItem>,
    #[serde(default)]
    pub tools: Vec<ToolSummaryRow>,
    #[serde(default)]
    pub mcps: Vec<McpSummaryRow>,
    #[serde(default)]
    pub providers: Vec<ProviderInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ModelInfo;
    use serde_json::json;

    #[test]
    fn jsonrpc_wire_frames_roundtrip_and_error_codes() {
        assert_eq!(ERR_PARSE_ERROR, -32700);
        assert_eq!(ERR_INVALID_REQUEST, -32600);
        assert_eq!(ERR_METHOD_NOT_FOUND, -32601);
        assert_eq!(ERR_INVALID_PARAMS, -32602);
        assert_eq!(ERR_INTERNAL_ERROR, -32603);
        assert_eq!(ERR_SESSION_LOCKED, -32001);
        assert_eq!(ERR_SESSION_NOT_FOUND, -32002);

        for id in [JsonRpcId::from(42), JsonRpcId::from("req-abc")] {
            let req = JsonRpcRequest::new(
                id.clone(),
                METHOD_SESSION_SUBSCRIBE,
                Some(json!({ "session_id": "01J", "after_seq": 10 })),
            );
            assert!(req.is_valid_version());
            let raw = serde_json::to_string(&req).unwrap();
            let decoded: JsonRpcRequest = serde_json::from_str(&raw).unwrap();
            assert_eq!(decoded, req);

            let ok_resp = JsonRpcResponse::ok(id, json!({ "ok": true }));
            let ok_val = serde_json::to_value(&ok_resp).unwrap();
            assert!(ok_val.get("error").is_none());
            assert_eq!(
                serde_json::from_value::<JsonRpcResponse>(ok_val).unwrap(),
                ok_resp
            );
        }

        let err_resp = JsonRpcResponse::err(
            None,
            JsonRpcError::session_locked("locked", Some(json!({ "owner_pid": 123 }))),
        );
        let err_val = serde_json::to_value(&err_resp).unwrap();
        assert_eq!(err_val["id"], Value::Null);
        assert!(err_val.get("result").is_none());
        assert_eq!(err_val["error"]["code"], ERR_SESSION_LOCKED);
        assert_eq!(
            serde_json::from_value::<JsonRpcResponse>(err_val).unwrap(),
            err_resp
        );
    }

    #[test]
    fn session_subscribe_params_and_snapshot_roundtrip() {
        let params: SessionSubscribeParams =
            serde_json::from_str(r#"{"session_id":"01J999","after_seq":120}"#).unwrap();
        assert_eq!(params.session_id, "01J999");
        assert_eq!(params.after_seq, Some(120));
        assert_eq!(params.limit, None);

        let snapshot = SessionSubscribeSnapshot {
            session: SessionMetaDto {
                id: "01J999".into(),
                cwd: Some("/repo".into()),
                title: Some("Fix RPC".into()),
                parent_id: None,
                fork_seq: None,
            },
            state: SessionRuntimeState::WaitingApproval,
            config: SessionConfigDto {
                provider: Some("magpie".into()),
                model: Some("isok/gpt-6.2-sol".into()),
                reasoning_effort: Some(ReasoningEffort::High),
                approval_mode: RpcApprovalMode::Ask,
            },
            context: SessionContextUsageDto {
                used_tokens: 12_345,
                max_tokens: Some(200_000),
            },
            events: vec![CommittedEventDto {
                session_id: "01J999".into(),
                seq: 121,
                ts_ms: 1_763_971_200_000,
                visibility: RpcEventVisibility::Surface,
                kind: "user".into(),
                payload: json!({ "type": "user", "text": "hello" }),
            }],
            active_turn: Some(ActiveTurnDto {
                turn: 4,
                step: 2,
                reasoning: "thinking...".into(),
                text: "working".into(),
                running_tools: vec![RunningToolDto {
                    call_id: "call_1".into(),
                    name: "bash".into(),
                    arguments: r#"{"command":"cargo test"}"#.into(),
                    output_tail: "running".into(),
                }],
            }),
            pending_interactions: vec![PendingInteractionDto {
                session_id: Some("01J999".into()),
                request_id: "req_1".into(),
                kind: "approval".into(),
                call_id: Some("call_1".into()),
                tool: "bash".into(),
                summary: "Run cargo test".into(),
                args: json!({ "command": "cargo test" }),
                options: vec!["allow_once".into(), "allow_always".into(), "deny".into()],
            }],
            queued_inputs: vec![QueuedInputDto {
                id: 7,
                text: "next task".into(),
                attachments: Vec::new(),
                created_at_ms: 1_763_971_201_000,
            }],
            subagents: vec![SubagentSnapshotDto {
                name: "verifier".into(),
                parent_session_id: Some("01J999".into()),
                child_session_id: "01JCHILD".into(),
                mode: RpcSubagentMode::ReadOnly,
                prompt_preview: "Verify diff".into(),
                status: RpcSubagentStatus::Running,
                steps: 3,
                current_step: 2,
                last_tool: Some("read".into()),
                idle_secs: 1,
                watchdog_alerts: 0,
                elapsed_secs: 5,
                summary: None,
                output_path: None,
            }],
        };

        let val = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(val["state"], "waiting_approval");
        assert_eq!(val["config"]["approval_mode"], "ask");
        assert_eq!(val["events"][0]["visibility"], "surface");
        assert_eq!(val["subagents"][0]["mode"], "read_only");
        assert_eq!(val["subagents"][0]["status"], "running");

        let back: SessionSubscribeSnapshot = serde_json::from_value(val).unwrap();
        assert_eq!(back, snapshot);
    }

    #[test]
    fn all_eight_notifications_roundtrip_via_jsonrpc() {
        let cases = vec![
            (
                NOTIF_TURN_STATE,
                RpcNotification::TurnState(TurnStateNotification {
                    session_id: "s1".into(),
                    turn: 1,
                    step: 0,
                    state: SessionRuntimeState::Streaming,
                }),
            ),
            (
                NOTIF_STREAM_DELTA,
                RpcNotification::StreamDelta(StreamDeltaNotification {
                    session_id: "s1".into(),
                    turn: 1,
                    step: 0,
                    kind: StreamDeltaKind::ToolArgs,
                    call_id: Some("c1".into()),
                    delta: "{\"path\":".into(),
                }),
            ),
            (
                NOTIF_EVENT_COMMITTED,
                RpcNotification::EventCommitted(CommittedEventDto {
                    session_id: "s1".into(),
                    seq: 2,
                    ts_ms: 1_763_971_200_123,
                    visibility: RpcEventVisibility::LogOnly,
                    kind: "approval".into(),
                    payload: json!({ "tool": "bash", "decision": "allow_once" }),
                }),
            ),
            (
                NOTIF_INTERACTION_REQUESTED,
                RpcNotification::InteractionRequested(PendingInteractionDto {
                    session_id: None,
                    request_id: "ir1".into(),
                    kind: "approval".into(),
                    call_id: Some("c1".into()),
                    tool: "bash".into(),
                    summary: "rm -rf".into(),
                    args: json!({ "command": "echo hi" }),
                    options: vec!["allow_once".into(), "deny".into()],
                }),
            ),
            (
                NOTIF_INTERACTION_RESOLVED,
                RpcNotification::InteractionResolved(InteractionResolvedNotification {
                    session_id: Some("s1".into()),
                    request_id: "ir1".into(),
                    decision: RpcApprovalDecision::AllowOnce,
                    note: None,
                    resolved_by: ResolvedBy::Rpc,
                    ts_ms: 1_763_971_200_500,
                }),
            ),
            (
                NOTIF_QUEUE_UPDATED,
                RpcNotification::QueueUpdated(QueueUpdatedNotification {
                    session_id: "s1".into(),
                    items: vec![QueuedInputDto {
                        id: 1,
                        text: "queued".into(),
                        attachments: vec![],
                        created_at_ms: 100,
                    }],
                }),
            ),
            (
                NOTIF_SUBAGENT_UPDATED,
                RpcNotification::SubagentUpdated(SubagentUpdatedNotification {
                    session_id: "s1".into(),
                    task: SubagentSnapshotDto {
                        name: "explorer".into(),
                        parent_session_id: Some("s1".into()),
                        child_session_id: "s2".into(),
                        mode: RpcSubagentMode::Edit,
                        prompt_preview: "inspect".into(),
                        status: RpcSubagentStatus::Completed,
                        steps: 4,
                        current_step: 4,
                        last_tool: None,
                        idle_secs: 0,
                        watchdog_alerts: 0,
                        elapsed_secs: 9,
                        summary: Some("done".into()),
                        output_path: None,
                    },
                    notice: Some("finished".into()),
                }),
            ),
            (
                NOTIF_SESSION_SWITCHED,
                RpcNotification::SessionSwitched(SessionSwitchedNotification {
                    old_session_id: "s1".into(),
                    new_session_id: "s3".into(),
                }),
            ),
        ];

        assert_eq!(cases.len(), 8);
        for (expected_method, notif) in cases {
            assert_eq!(notif.method(), expected_method);
            let wire = notif.to_jsonrpc();
            assert_eq!(wire.jsonrpc, JSONRPC_VERSION);
            assert_eq!(wire.method, expected_method);
            let line = serde_json::to_string(&wire).unwrap();
            let decoded_wire: JsonRpcNotification = serde_json::from_str(&line).unwrap();
            let roundtripped = RpcNotification::from_jsonrpc(&decoded_wire).unwrap();
            assert_eq!(roundtripped, notif);
        }

        let delta_val = serde_json::to_value(StreamDeltaKind::ToolOutput).unwrap();
        assert_eq!(delta_val, "tool_output");
        let resolved_val = serde_json::to_value(ResolvedBy::Tui).unwrap();
        assert_eq!(resolved_val, "tui");
    }

    #[test]
    fn request_and_result_dtos_roundtrip() {
        let submit: TurnSubmitParams = serde_json::from_str(
            r#"{"session_id":"s1","text":"hi","mode":"steer","attachments":[{"path":"a.png","mime":"image/png"}]}"#,
        )
        .unwrap();
        assert_eq!(submit.effective_mode(), SubmitMode::Steer);
        assert_eq!(submit.attachments.as_ref().unwrap().len(), 1);
        assert_eq!(
            serde_json::from_str::<TurnSubmitParams>(&serde_json::to_string(&submit).unwrap())
                .unwrap(),
            submit
        );

        let submit_res = TurnSubmitResult {
            accepted: true,
            mode: SubmitMode::Queue,
            queue_id: Some(9),
            turn: None,
        };
        assert_eq!(
            serde_json::from_str::<TurnSubmitResult>(&serde_json::to_string(&submit_res).unwrap())
                .unwrap(),
            submit_res
        );

        let cancel_p = TurnCancelParams {
            session_id: "s1".into(),
        };
        let cancel_r = TurnCancelResult { cancelled: true };
        assert_eq!(
            serde_json::from_str::<TurnCancelParams>(&serde_json::to_string(&cancel_p).unwrap())
                .unwrap(),
            cancel_p
        );
        assert_eq!(
            serde_json::from_str::<TurnCancelResult>(&serde_json::to_string(&cancel_r).unwrap())
                .unwrap(),
            cancel_r
        );

        let q_rm = QueueRemoveParams {
            session_id: "s1".into(),
            id: 3,
        };
        let q_rm_r = QueueRemoveResult {
            removed: true,
            item: Some(QueuedInputDto {
                id: 3,
                text: "later".into(),
                attachments: vec![],
                created_at_ms: 55,
            }),
        };
        assert_eq!(
            serde_json::from_str::<QueueRemoveParams>(&serde_json::to_string(&q_rm).unwrap())
                .unwrap(),
            q_rm
        );
        assert_eq!(
            serde_json::from_str::<QueueRemoveResult>(&serde_json::to_string(&q_rm_r).unwrap())
                .unwrap(),
            q_rm_r
        );

        let q_pop = QueuePopAllParams {
            session_id: "s1".into(),
        };
        let q_pop_r = QueuePopAllResult { items: vec![] };
        assert_eq!(
            serde_json::from_str::<QueuePopAllParams>(&serde_json::to_string(&q_pop).unwrap())
                .unwrap(),
            q_pop
        );
        assert_eq!(
            serde_json::from_str::<QueuePopAllResult>(&serde_json::to_string(&q_pop_r).unwrap())
                .unwrap(),
            q_pop_r
        );

        let resolve_p = InteractionResolveParams {
            session_id: Some("s1".into()),
            request_id: "r1".into(),
            decision: RpcApprovalDecision::Deny,
            note: Some("use rg instead".into()),
        };
        assert_eq!(
            resolve_p
                .decision
                .to_approval_decision(resolve_p.note.clone()),
            ApprovalDecision::DenyWithReason("use rg instead".into())
        );
        assert_eq!(
            serde_json::from_str::<InteractionResolveParams>(
                &serde_json::to_string(&resolve_p).unwrap()
            )
            .unwrap(),
            resolve_p
        );

        let already = InteractionResolveResult::already_resolved();
        assert!(!already.ok);
        assert_eq!(already.reason.as_deref(), Some("already_resolved"));
        assert_eq!(
            serde_json::from_str::<InteractionResolveResult>(
                &serde_json::to_string(&already).unwrap()
            )
            .unwrap(),
            already
        );

        let cfg_p = SessionConfigParams {
            session_id: "s1".into(),
            provider: Some("deepseek".into()),
            model: Some("deepseek-chat".into()),
            reasoning_effort: Some(ReasoningEffort::Medium),
            approval_mode: Some(RpcApprovalMode::Auto),
        };
        let cfg_r = SessionConfigResult {
            config: SessionConfigDto {
                provider: Some("deepseek".into()),
                model: Some("deepseek-chat".into()),
                reasoning_effort: Some(ReasoningEffort::Medium),
                approval_mode: RpcApprovalMode::Auto,
            },
            context: SessionContextUsageDto {
                used_tokens: 500,
                max_tokens: Some(128_000),
            },
        };
        assert_eq!(
            serde_json::from_str::<SessionConfigParams>(&serde_json::to_string(&cfg_p).unwrap())
                .unwrap(),
            cfg_p
        );
        assert_eq!(
            serde_json::from_str::<SessionConfigResult>(&serde_json::to_string(&cfg_r).unwrap())
                .unwrap(),
            cfg_r
        );

        let list_p = SessionListParams {
            cwd: Some("/ws".into()),
            query: Some("fix".into()),
            limit: Some(20),
            cursor: None,
            exclude_subagents: Some(true),
        };
        let list_r = SessionListResult {
            sessions: vec![SessionListItemDto {
                id: "s1".into(),
                cwd: Some("/ws".into()),
                title: Some("fix bug".into()),
                parent_id: None,
                fork_seq: None,
                event_count: 10,
                created_at_ms: 1000,
                updated_at_ms: 2000,
                active_writer_pid: Some(48210),
            }],
            next_cursor: Some("cur_1".into()),
        };
        assert_eq!(
            serde_json::from_str::<SessionListParams>(&serde_json::to_string(&list_p).unwrap())
                .unwrap(),
            list_p
        );
        assert_eq!(
            serde_json::from_str::<SessionListResult>(&serde_json::to_string(&list_r).unwrap())
                .unwrap(),
            list_r
        );

        let events_p = SessionEventsParams {
            session_id: "s1".into(),
            before_seq: Some(50),
            after_seq: None,
            limit: Some(25),
        };
        let events_r = SessionEventsResult {
            session_id: "s1".into(),
            events: vec![],
            has_more: false,
        };
        assert_eq!(
            serde_json::from_str::<SessionEventsParams>(&serde_json::to_string(&events_p).unwrap())
                .unwrap(),
            events_p
        );
        assert_eq!(
            serde_json::from_str::<SessionEventsResult>(&serde_json::to_string(&events_r).unwrap())
                .unwrap(),
            events_r
        );

        let fork_p = SessionForkParams {
            session_id: "s1".into(),
            at_seq: 12,
        };
        let fork_r = SessionForkResult {
            session: SessionMetaDto {
                id: "s2".into(),
                cwd: Some("/ws".into()),
                title: Some("forked".into()),
                parent_id: Some("s1".into()),
                fork_seq: Some(12),
            },
        };
        assert_eq!(
            serde_json::from_str::<SessionForkParams>(&serde_json::to_string(&fork_p).unwrap())
                .unwrap(),
            fork_p
        );
        assert_eq!(
            serde_json::from_str::<SessionForkResult>(&serde_json::to_string(&fork_r).unwrap())
                .unwrap(),
            fork_r
        );

        let tree_p = SessionTreeParams {
            session_id: "s1".into(),
        };
        let tree_r = SessionTreeResult {
            session_id: "s1".into(),
            entries: vec![SessionTreeEntry {
                seq: 1,
                kind_tag: "USER".into(),
                summary: "hi".into(),
                depth: 0,
                can_fork: true,
            }],
        };
        assert_eq!(
            serde_json::from_str::<SessionTreeParams>(&serde_json::to_string(&tree_p).unwrap())
                .unwrap(),
            tree_p
        );
        assert_eq!(
            serde_json::from_str::<SessionTreeResult>(&serde_json::to_string(&tree_r).unwrap())
                .unwrap(),
            tree_r
        );

        let cap_p = CapabilitiesListParams {
            session_id: Some("s1".into()),
        };
        let cap_r = CapabilitiesListResult {
            commands: vec![SlashCommandDto {
                command: "resume".into(),
                description: "Resume session".into(),
                takes_arg: true,
            }],
            skills: vec![SkillPickerItem {
                name: "rust".into(),
                description: "Rust skill".into(),
                source: "workspace".into(),
            }],
            tools: vec![ToolSummaryRow {
                name: "read".into(),
                exposure: "direct".into(),
                declared: true,
            }],
            mcps: vec![McpSummaryRow {
                name: "git".into(),
                status: "connected".into(),
                ok: true,
            }],
            providers: vec![ProviderInfo {
                id: "magpie".into(),
                name: "Magpie".into(),
                protocol: "openai-responses".into(),
                models: vec![ModelInfo {
                    id: "gpt-6".into(),
                    name: "GPT 6".into(),
                    context_window: Some(200_000),
                    ..Default::default()
                }],
            }],
            max_context_tokens: Some(200_000),
        };
        assert_eq!(
            serde_json::from_str::<CapabilitiesListParams>(&serde_json::to_string(&cap_p).unwrap())
                .unwrap(),
            cap_p
        );
        assert_eq!(
            serde_json::from_str::<CapabilitiesListResult>(&serde_json::to_string(&cap_r).unwrap())
                .unwrap(),
            cap_r
        );
    }
}
