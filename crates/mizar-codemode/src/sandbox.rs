//! QuickJS (`rquickjs`) 沙箱执行器：安全运行模型编写的多工具 JS 编排脚本。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mizar_tool_search::{Bm25Ranker, ToolSearchDocument, create_tool_search_document};
use mizar_tools::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head, write_temp_output,
};
use mizar_tools::{ToolCallContext, ToolExecution, ToolOutput, ToolRuntime, ToolRuntimeInfo};
use rquickjs::context::EvalOptions;
use rquickjs::prelude::{Async, Func};
use rquickjs::{AsyncContext, AsyncRuntime, CatchResultExt, Function, Promise, Value as JsValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Semaphore;

use crate::declarations::{CODEMODE_TOOL_NAME, to_codemode_identifier};

pub const CODEMODE_MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
pub const CODEMODE_STACK_SIZE_BYTES: usize = 1024 * 1024;
pub const DEFAULT_CODEMODE_TIMEOUT_MS: u64 = 120_000;
const ARGS_PREVIEW_CHARS: usize = 160;

#[derive(Debug, Clone, Default)]
enum ScriptExitSignal {
    #[default]
    Running,
    Requested(Option<String>),
}

enum ScriptOutcome {
    Completed { returned_json: Option<String> },
    Exited { message: Option<String> },
    Aborted,
    TimedOut { timeout_ms: u64 },
    Failed { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodemodeNestedCall {
    pub id: String,
    pub name: String,
    pub args_preview: String,
    pub status: String,
    pub duration_ms: u64,
}

fn preview_json(v: &Value, max_chars: usize) -> String {
    let raw = serde_json::to_string(v).unwrap_or_default();
    let count = raw.chars().count();
    if count <= max_chars {
        raw
    } else {
        let head: String = raw.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{head}...")
    }
}

fn format_preview_calls(calls: &[CodemodeNestedCall]) -> String {
    calls
        .iter()
        .map(|c| {
            format!(
                "{} {}({}) [{}ms]",
                c.status, c.name, c.args_preview, c.duration_ms
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_timeout_override(code: &str, default_ms: u64) -> u64 {
    let Some(first_line) = code.lines().next() else {
        return default_ms;
    };
    let Some(rest) = first_line.trim().strip_prefix("// @options:") else {
        return default_ms;
    };
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(rest.trim())
        && let Some(ms) = map.get("timeoutMs").and_then(Value::as_u64)
    {
        return ms.clamp(100, 600_000);
    }
    default_ms
}

const PRELUDE_JS: &str = r#"
const __idToRaw = JSON.parse(__ID_TO_RAW_JSON);
const ALL_TOOLS = JSON.parse(__ALL_TOOLS_JSON);

function __formatArg(v) {
    if (typeof v === "string") return v;
    if (v === undefined) return "undefined";
    try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}

const console = {
    log: (...args) => __host_log("log", args.map(__formatArg).join(" ")),
    info: (...args) => __host_log("info", args.map(__formatArg).join(" ")),
    warn: (...args) => __host_log("warn", args.map(__formatArg).join(" ")),
    error: (...args) => __host_log("error", args.map(__formatArg).join(" ")),
};

function text(v) {
    __host_log("text", typeof v === "string" ? v : __formatArg(v));
}

function exit(msg) {
    __host_exit(msg === undefined ? null : String(msg));
    const err = new Error("__codemode_exit__");
    err.__codemodeExit = true;
    throw err;
}

function store(key, value) {
    if (value === undefined) {
        __host_store_set(String(key), null);
    } else {
        __host_store_set(String(key), JSON.stringify(value));
    }
}

function load(key) {
    const raw = __host_store_get(String(key));
    return raw == null ? undefined : JSON.parse(raw);
}

async function searchTools(query, opts) {
    const limit = opts && typeof opts.limit === "number" ? opts.limit : 8;
    const ns = opts && typeof opts.namespace === "string" ? opts.namespace : null;
    return JSON.parse(__host_search_tools(String(query || ""), limit, ns));
}

async function describeTool(name) {
    const raw = __host_describe_tool(String(name || ""));
    return raw ? JSON.parse(raw) : null;
}

async function describeNamespace(name) {
    const raw = __host_describe_namespace(String(name || ""));
    return raw ? JSON.parse(raw) : null;
}

async function __callTool(prop, args) {
    const rawName = __idToRaw[prop] || prop;
    const jsonArgs = JSON.stringify(args === undefined ? {} : args);
    const respRaw = await __host_call_tool(rawName, jsonArgs);
    const resp = JSON.parse(respRaw);
    if (!resp.ok) {
        throw new Error(resp.error || ("Tool call failed: " + rawName));
    }
    return resp.value;
}

const tools = new Proxy({}, {
    get(_target, prop) {
        if (typeof prop !== "string") return undefined;
        return (args) => __callTool(prop, args);
    }
});
"#;

pub async fn execute_codemode_script(
    code: String,
    call_ctx: ToolCallContext,
    tools_rt: Arc<ToolRuntime>,
    session_store: Arc<Mutex<HashMap<String, HashMap<String, Value>>>>,
    worker_sem: Arc<Semaphore>,
    default_timeout_ms: u64,
) -> Result<ToolOutput, cordis::BoxError> {
    let permit = tokio::select! {
        _ = call_ctx.cancel.cancelled() => return Err("Operation aborted".into()),
        res = worker_sem.acquire_owned() => res.map_err(|_| "codemode worker pool closed")?,
    };
    let _permit = permit;

    let session_key = call_ctx.session_id().unwrap_or("__default__").to_string();

    let callable: Vec<ToolRuntimeInfo> = tools_rt
        .list_callable(call_ctx.session_id())
        .into_iter()
        .filter(|t| t.name != CODEMODE_TOOL_NAME)
        .collect();

    let mut id_to_raw = HashMap::<String, String>::new();
    let mut all_tools_meta = Vec::new();
    for info in &callable {
        let js_id = to_codemode_identifier(&info.name);
        id_to_raw.insert(js_id.clone(), info.name.clone());
        id_to_raw.insert(info.name.clone(), info.name.clone());
        all_tools_meta.push(json!({
            "name": info.name,
            "id": js_id,
            "description": info.description,
            "inputSchema": info.parameters,
            "outputSchema": info.output_schema,
            "namespace": info.namespace.as_ref().map(|n| n.name.clone()),
        }));
    }

    let id_to_raw_json = serde_json::to_string(&id_to_raw)?;
    let all_tools_json = serde_json::to_string(&all_tools_meta)?;

    let staging_store: HashMap<String, Value> = {
        let guard = session_store.lock().unwrap();
        guard.get(&session_key).cloned().unwrap_or_default()
    };
    let store_state = Arc::new(Mutex::new(staging_store));
    let logs = Arc::new(Mutex::new(Vec::<String>::new()));
    let nested_calls = Arc::new(Mutex::new(Vec::<CodemodeNestedCall>::new()));
    let nested_seq = Arc::new(AtomicUsize::new(0));
    let exit_signal = Arc::new(Mutex::new(ScriptExitSignal::Running));
    let script_cancel = call_ctx.cancel.child_token();
    let timed_out_hit = Arc::new(AtomicBool::new(false));

    let timeout_ms = parse_timeout_override(&code, default_timeout_ms);
    let timeout = Duration::from_millis(timeout_ms);
    let started = Instant::now();

    let host_handle = tokio::runtime::Handle::current();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel::<Result<Option<String>, String>>();

    let logs_host = Arc::clone(&logs);
    let store_get = Arc::clone(&store_state);
    let store_set = Arc::clone(&store_state);
    let nested_host = Arc::clone(&nested_calls);
    let seq_host = Arc::clone(&nested_seq);
    let tools_host = Arc::clone(&tools_rt);
    let call_ctx_host = call_ctx.clone();
    let exit_host = Arc::clone(&exit_signal);
    let parent_cancel_thread = call_ctx.cancel.clone();
    let script_cancel_thread = script_cancel.clone();
    let timed_out_thread = Arc::clone(&timed_out_hit);
    let callable_thread = callable.clone();

    // QuickJS 同步执行与 GC 隔离到独立 OS 线程，防止阻塞 Tokio worker 线程
    std::thread::Builder::new()
        .name("mizar-codemode-js".to_string())
        .spawn(move || {
            let local_rt = match tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
            {
                Ok(rt) => rt,
                Err(err) => {
                    let _ = result_tx.send(Err(format!("Failed to init JS thread runtime: {err}")));
                    return;
                }
            };

            local_rt.block_on(async move {
                let rt = match AsyncRuntime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = result_tx.send(Err(format!("Failed to init QuickJS runtime: {e}")));
                        return;
                    }
                };
                rt.set_memory_limit(CODEMODE_MEMORY_LIMIT_BYTES).await;
                rt.set_max_stack_size(CODEMODE_STACK_SIZE_BYTES).await;

                let cancel_interrupt = script_cancel_thread.clone();
                rt.set_interrupt_handler(Some(Box::new(move || {
                    cancel_interrupt.is_cancelled() || started.elapsed() >= timeout
                })))
                .await;

                let qctx = match AsyncContext::full(&rt).await {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        let _ = result_tx.send(Err(format!("Failed to init QuickJS context: {e}")));
                        return;
                    }
                };

                let callable_for_search = callable_thread.clone();
                let callable_for_desc = callable_thread.clone();
                let callable_for_ns = callable_thread;
                let nested_cancel = script_cancel_thread.clone();

                let deadline = tokio::time::Instant::from_std(started) + timeout;
                let script_outcome: Result<Option<String>, String> = {
                    let mut eval_fut = Box::pin(qctx.async_with(async |ctx| {
                        let globals = ctx.globals();
                        globals
                            .set("__ID_TO_RAW_JSON", id_to_raw_json.as_str())
                            .map_err(|e| e.to_string())?;
                        globals
                            .set("__ALL_TOOLS_JSON", all_tools_json.as_str())
                            .map_err(|e| e.to_string())?;

                        let exit_cb = Arc::clone(&exit_host);
                        globals
                            .set(
                                "__host_exit",
                                Func::from(move |msg: Option<String>| {
                                    *exit_cb.lock().unwrap() = ScriptExitSignal::Requested(msg);
                                }),
                            )
                            .map_err(|e| e.to_string())?;

                        let logs_cb = Arc::clone(&logs_host);
                        globals
                            .set(
                                "__host_log",
                                Func::from(move |kind: String, text: String| {
                                    let line = if kind == "warn" {
                                        format!("[warn] {text}")
                                    } else if kind == "error" {
                                        format!("[error] {text}")
                                    } else {
                                        text
                                    };
                                    logs_cb.lock().unwrap().push(line);
                                }),
                            )
                            .map_err(|e| e.to_string())?;

                        globals
                            .set(
                                "__host_store_get",
                                Func::from(move |key: String| -> Option<String> {
                                    store_get
                                        .lock()
                                        .unwrap()
                                        .get(&key)
                                        .and_then(|v| serde_json::to_string(v).ok())
                                }),
                            )
                            .map_err(|e| e.to_string())?;

                        globals
                            .set(
                                "__host_store_set",
                                Func::from(move |key: String, json_opt: Option<String>| {
                                    let mut map = store_set.lock().unwrap();
                                    match json_opt {
                                        None => {
                                            map.remove(&key);
                                        }
                                        Some(raw) => {
                                            if let Ok(v) = serde_json::from_str::<Value>(&raw) {
                                                map.insert(key, v);
                                            }
                                        }
                                    }
                                }),
                            )
                            .map_err(|e| e.to_string())?;

                        globals
                            .set(
                                "__host_search_tools",
                                Func::from(
                                    move |query: String, limit: usize, ns: Option<String>| -> String {
                                        let filtered: Vec<&ToolRuntimeInfo> = callable_for_search
                                            .iter()
                                            .filter(|t| {
                                                ns.as_deref().is_none_or(|want| {
                                                    t.namespace.as_ref().is_some_and(|n| {
                                                        n.name == want
                                                            || n.aliases.iter().any(|a| a == want)
                                                    })
                                                })
                                            })
                                            .collect();
                                        let docs: Vec<ToolSearchDocument> = filtered
                                            .iter()
                                            .map(|t| create_tool_search_document(t))
                                            .collect();
                                        let matches = Bm25Ranker::default()
                                            .rank(&query, &docs, limit.clamp(1, 32));
                                        let out: Vec<Value> = matches
                                            .into_iter()
                                            .filter_map(|m| {
                                                filtered.iter().find(|t| t.name == m.name)
                                            })
                                            .map(|t| {
                                                json!({
                                                    "name": t.name,
                                                    "id": to_codemode_identifier(&t.name),
                                                    "description": t.description,
                                                    "inputSchema": t.parameters,
                                                    "outputSchema": t.output_schema,
                                                })
                                            })
                                            .collect();
                                        serde_json::to_string(&out).unwrap_or_else(|_| "[]".into())
                                    },
                                ),
                            )
                            .map_err(|e| e.to_string())?;

                        globals
                            .set(
                                "__host_describe_tool",
                                Func::from(move |name: String| -> Option<String> {
                                    callable_for_desc
                                        .iter()
                                        .find(|t| {
                                            t.name == name
                                                || to_codemode_identifier(&t.name) == name
                                        })
                                        .and_then(|t| {
                                            serde_json::to_string(&json!({
                                                "name": t.name,
                                                "id": to_codemode_identifier(&t.name),
                                                "description": t.description,
                                                "inputSchema": t.parameters,
                                                "outputSchema": t.output_schema,
                                                "guidelines": t.prompt_guidelines,
                                            }))
                                            .ok()
                                        })
                                }),
                            )
                            .map_err(|e| e.to_string())?;

                        globals
                            .set(
                                "__host_describe_namespace",
                                Func::from(move |name: String| -> Option<String> {
                                    let matching: Vec<&ToolRuntimeInfo> = callable_for_ns
                                        .iter()
                                        .filter(|t| {
                                            t.namespace.as_ref().is_some_and(|n| {
                                                n.name == name || n.aliases.iter().any(|a| a == &name)
                                            })
                                        })
                                        .collect();
                                    if matching.is_empty() {
                                        return None;
                                    }
                                    let ns_info = matching[0].namespace.as_ref().unwrap();
                                    serde_json::to_string(&json!({
                                        "name": ns_info.name,
                                        "description": ns_info.description,
                                        "instructions": ns_info.instructions,
                                        "tools": matching.iter().map(|t| json!({
                                            "name": t.name,
                                            "id": to_codemode_identifier(&t.name),
                                            "description": t.description,
                                        })).collect::<Vec<_>>(),
                                    }))
                                    .ok()
                                }),
                            )
                            .map_err(|e| e.to_string())?;

                        let host_rt = host_handle.clone();
                        let call_fn = Function::new(
                            ctx.clone(),
                            Async(move |raw_name: String, args_json: String| {
                                let tools = Arc::clone(&tools_host);
                                let parent_ctx = call_ctx_host.clone();
                                let child_cancel = nested_cancel.clone();
                                let nested = Arc::clone(&nested_host);
                                let seq = Arc::clone(&seq_host);
                                let host_rt = host_rt.clone();
                                async move {
                                    let n = seq.fetch_add(1, Ordering::SeqCst) + 1;
                                    let call_id = format!("cm/{n}");
                                    let parsed_args: Value = serde_json::from_str(&args_json)
                                        .unwrap_or_else(|_| json!({}));
                                    let args_preview =
                                        preview_json(&parsed_args, ARGS_PREVIEW_CHARS);

                                    {
                                        let mut list = nested.lock().unwrap();
                                        list.push(CodemodeNestedCall {
                                            id: call_id.clone(),
                                            name: raw_name.clone(),
                                            args_preview: args_preview.clone(),
                                            status: "⏳".into(),
                                            duration_ms: 0,
                                        });
                                        parent_ctx.emit_update(format_preview_calls(&list), None);
                                    }

                                    let t0 = Instant::now();
                                    let exec_name = raw_name.clone();
                                    let session_id = parent_ctx.exec.session_id.clone();
                                    let res = match host_rt
                                        .spawn(async move {
                                            tools
                                                .execute_nested(
                                                    ToolExecution::new(
                                                        exec_name,
                                                        parsed_args,
                                                        session_id,
                                                    ),
                                                    child_cancel,
                                                    None,
                                                )
                                                .await
                                        })
                                        .await
                                    {
                                        Ok(res) => res,
                                        Err(join_err) => mizar_tools::ToolResult::err(format!(
                                            "Nested tool execution failed: {join_err}"
                                        )),
                                    };
                                    let elapsed = t0.elapsed().as_millis() as u64;

                                    {
                                        let mut list = nested.lock().unwrap();
                                        if let Some(entry) = list.iter_mut().find(|c| c.id == call_id)
                                        {
                                            entry.status = if res.is_error {
                                                "✗".into()
                                            } else {
                                                "✓".into()
                                            };
                                            entry.duration_ms = elapsed;
                                        }
                                        parent_ctx.emit_update(format_preview_calls(&list), None);
                                    }

                                    if res.is_error {
                                        Ok::<_, rquickjs::Error>(
                                            json!({ "ok": false, "error": res.content }).to_string(),
                                        )
                                    } else {
                                        let val = res
                                            .structured_content
                                            .unwrap_or(Value::String(res.content));
                                        Ok::<_, rquickjs::Error>(
                                            json!({ "ok": true, "value": val }).to_string(),
                                        )
                                    }
                                }
                            }),
                        )
                        .map_err(|e| e.to_string())?;
                        globals
                            .set("__host_call_tool", call_fn)
                            .map_err(|e| e.to_string())?;

                        ctx.eval::<(), _>(PRELUDE_JS)
                            .catch(&ctx)
                            .map_err(|e| format!("Prelude error: {e}"))?;

                        let wrapped = format!(
                            "(async () => {{\n{}\n}})().then(__v => __v === undefined ? null : JSON.stringify(__v))",
                            code
                        );
                        let mut opts = EvalOptions::default();
                        opts.promise = false;
                        opts.strict = false;
                        let promise: Promise = ctx
                            .eval_with_options(wrapped, opts)
                            .catch(&ctx)
                            .map_err(|e| format!("{e}"))?;

                        let awaited: JsValue = promise
                            .into_future::<JsValue>()
                            .await
                            .catch(&ctx)
                            .map_err(|e| format!("{e}"))?;

                        let ret_str = if awaited.is_null() || awaited.is_undefined() {
                            None
                        } else if let Some(s) = awaited.as_string() {
                            s.to_string().ok()
                        } else {
                            None
                        };
                        Ok(ret_str)
                    }));
                    tokio::select! {
                        biased;
                        res = &mut eval_fut => res,
                        _ = parent_cancel_thread.cancelled() => {
                            script_cancel_thread.cancel();
                            Err("Script aborted".to_string())
                        }
                        _ = tokio::time::sleep_until(deadline) => {
                            timed_out_thread.store(true, Ordering::SeqCst);
                            script_cancel_thread.cancel();
                            Err(format!("Script timed out after {timeout_ms}ms"))
                        }
                    }
                };

                // 超时/取消后限制 idle 等待上限为 200ms，防止未决 Promise 永久挂死沙箱线程
                let _ = tokio::time::timeout(Duration::from_millis(200), rt.idle()).await;
                drop(qctx);
                drop(rt);
                let _ = result_tx.send(script_outcome);
            });
        })
        .map_err(|e| format!("Failed to spawn QuickJS thread: {e}"))?;

    let script_outcome = result_rx
        .await
        .unwrap_or_else(|_| Err("QuickJS worker thread terminated unexpectedly".to_string()));

    let calls_snapshot = nested_calls.lock().unwrap().clone();
    let mut logs_snapshot = logs.lock().unwrap().clone();
    let recorded_exit = exit_signal.lock().unwrap().clone();

    let outcome = if call_ctx.cancel.is_cancelled() {
        ScriptOutcome::Aborted
    } else if let ScriptExitSignal::Requested(message) = recorded_exit {
        ScriptOutcome::Exited { message }
    } else {
        match script_outcome {
            Ok(returned_json) => ScriptOutcome::Completed { returned_json },
            Err(_) if timed_out_hit.load(Ordering::SeqCst) || started.elapsed() >= timeout => {
                ScriptOutcome::TimedOut { timeout_ms }
            }
            Err(message) => ScriptOutcome::Failed { message },
        }
    };

    let (is_error, returned_val, error_msg) = match outcome {
        ScriptOutcome::Completed { returned_json } => {
            let mut guard = session_store.lock().unwrap();
            guard.insert(session_key, store_state.lock().unwrap().clone());
            (false, returned_json, None)
        }
        ScriptOutcome::Exited { message } => {
            let mut guard = session_store.lock().unwrap();
            guard.insert(session_key, store_state.lock().unwrap().clone());
            if let Some(msg) = message.filter(|m| !m.trim().is_empty()) {
                logs_snapshot.push(msg);
            }
            (false, None, None)
        }
        ScriptOutcome::Aborted => (true, None, Some("Script aborted".to_string())),
        ScriptOutcome::TimedOut { timeout_ms } => (
            true,
            None,
            Some(format!("Script timed out after {timeout_ms}ms")),
        ),
        ScriptOutcome::Failed { message } => (true, None, Some(message)),
    };

    let mut sections = Vec::new();
    sections.push(if is_error {
        "Script failed".to_string()
    } else {
        "Script completed".to_string()
    });
    if !logs_snapshot.is_empty() {
        sections.push(logs_snapshot.join("\n"));
    }
    if let Some(ret_json) = &returned_val {
        let pretty = serde_json::from_str::<Value>(ret_json)
            .ok()
            .and_then(|v| match v {
                Value::String(s) => Some(s),
                other => serde_json::to_string_pretty(&other).ok(),
            })
            .unwrap_or_else(|| ret_json.clone());
        sections.push(format!("Return value:\n{pretty}"));
    }
    if let Some(err) = &error_msg {
        sections.push(format!("Script error:\n{err}"));
    }

    let raw_text = sections.join("\n\n");
    let trunc = truncate_head(&raw_text, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let final_text = if trunc.truncated {
        let spilled = write_temp_output("mizar-codemode-", &raw_text)
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

    let details = json!({
        "nested_calls": calls_snapshot,
        "duration_ms": started.elapsed().as_millis() as u64,
        "truncated": trunc.truncated,
    });
    let structured = returned_val
        .as_deref()
        .and_then(|s| serde_json::from_str::<Value>(s).ok());

    Ok(ToolOutput {
        content: final_text,
        is_error,
        structured_content: structured,
        details: Some(details),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cordis::{BoxError, Context, Service};
    use mizar_tools::{ToolDefinition, ToolExposure, ToolsContextExt};
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn sandbox_runs_on_dedicated_os_thread_and_bounds_idle_after_timeout() {
        let app = Context::new();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        let tools = app.tools().unwrap();

        let caller_thread_id = std::thread::current().id();
        let observed_name = Arc::new(Mutex::new(None::<String>));
        let observed_name_c = Arc::clone(&observed_name);

        let _probe = tools
            .register(
                ToolDefinition::sync_tool("thread_probe", "probe", json!({}), move |_, _| {
                    *observed_name_c.lock().unwrap() =
                        std::thread::current().name().map(str::to_string);
                    Ok::<_, BoxError>("ok".into())
                })
                .with_exposure(ToolExposure::Codemode),
            )
            .unwrap();

        let session_store = Arc::new(Mutex::new(HashMap::new()));
        let sem = Arc::new(Semaphore::new(2));

        // 1. 工具调用本身走主 runtime，而同步 JS 忙等跑在独立 OS 线程（不阻塞 Tokio 异步定时器）
        let ticks = Arc::new(AtomicUsize::new(0));
        let ticks_bg = Arc::clone(&ticks);
        let ticker_cancel = CancellationToken::new();
        let ticker_cancel_bg = ticker_cancel.clone();
        let ticker = tokio::spawn(async move {
            while !ticker_cancel_bg.is_cancelled() {
                ticks_bg.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
        });

        let call_ctx = ToolCallContext {
            ctx: app.clone(),
            exec: ToolExecution::new("codemode", json!({}), Some("s1".into())),
            cancel: CancellationToken::new(),
            on_update: None,
        };
        let ok_out = execute_codemode_script(
            "const end = Date.now() + 90; while (Date.now() < end) {} await tools.thread_probe({}); return 42;".into(),
            call_ctx,
            tools.clone(),
            Arc::clone(&session_store),
            Arc::clone(&sem),
            5_000,
        )
        .await
        .unwrap();
        ticker_cancel.cancel();
        let _ = ticker.await;
        assert!(!ok_out.is_error);
        assert!(ok_out.content.contains("42"));
        assert!(observed_name.lock().unwrap().is_some());
        assert_ne!(
            observed_name.lock().unwrap().as_deref(),
            Some("mizar-codemode-js")
        );
        assert!(
            ticks.load(Ordering::SeqCst) >= 3,
            "Tokio runtime should keep ticking while sync JS runs on dedicated thread"
        );

        // 2. 死循环或未决 Promise 超时后，rt.idle() 受 200ms 上限保护，迅速返回超时错误
        let t0 = Instant::now();
        let timeout_ctx = ToolCallContext {
            ctx: app.clone(),
            exec: ToolExecution::new("codemode", json!({}), Some("s1".into())),
            cancel: CancellationToken::new(),
            on_update: None,
        };
        let timeout_out = execute_codemode_script(
            "// @options: {\"timeoutMs\": 120}\nawait new Promise(() => {});".into(),
            timeout_ctx,
            tools,
            session_store,
            sem,
            5_000,
        )
        .await
        .unwrap();
        assert!(timeout_out.is_error);
        assert!(timeout_out.content.contains("timed out"));
        assert!(t0.elapsed() < Duration::from_secs(2));
        assert_eq!(std::thread::current().id(), caller_thread_id);

        app.fiber().dispose().await;
    }
}
