//! `codemode` 插件：把 `codemode` 工具注册到 `ToolRuntime`，支持 QuickJS 沙箱执行、
//! `prepare_loadout` 动态目录生成与检测到 `Codemode` 工具时自动激活。

pub mod declarations;
pub mod sandbox;
pub mod schema_ts;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_tools::{
    EVENT_TOOLS_CHANGE, ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolExposure,
    ToolRuntime, ToolsContextExt,
};
use serde_json::{Value, json};
use tokio::sync::Semaphore;

pub use declarations::{
    CODEMODE_TOOL_NAME, DEFAULT_CODEMODE_INLINE_BUDGET, create_codemode_description,
    prepare_codemode_loadout, to_codemode_identifier,
};
pub use sandbox::{CodemodeNestedCall, DEFAULT_CODEMODE_TIMEOUT_MS, execute_codemode_script};

#[derive(Debug, Clone)]
pub struct CodemodeConfig {
    /// 检测到存在 `ToolExposure::Codemode` 工具时自动激活 `codemode`（默认 true）。
    pub auto_enable: bool,
    /// 为 `true` 时类似 `pi` 的 `only` 模式：隐藏 `Direct` 工具的直接声明，改由脚本统一调度（默认 false）。
    pub only_mode: bool,
    pub inline_budget: usize,
    pub max_workers: usize,
    pub timeout_ms: u64,
}

impl Default for CodemodeConfig {
    fn default() -> Self {
        Self {
            auto_enable: true,
            only_mode: false,
            inline_budget: DEFAULT_CODEMODE_INLINE_BUDGET,
            max_workers: 2,
            timeout_ms: DEFAULT_CODEMODE_TIMEOUT_MS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoEnableState {
    Inactive,
    Active,
}

fn sync_auto_enable(tools: &ToolRuntime, state: &Mutex<AutoEnableState>) {
    let has_codemode_tools = tools
        .list_all()
        .into_iter()
        .any(|t| t.exposure == ToolExposure::Codemode);
    let next = if has_codemode_tools {
        AutoEnableState::Active
    } else {
        AutoEnableState::Inactive
    };
    let changed = {
        let mut cur = state.lock().unwrap();
        if *cur == next {
            false
        } else {
            *cur = next;
            true
        }
    };
    if changed {
        tools.set_auto_default_active(CODEMODE_TOOL_NAME, Some(has_codemode_tools));
    }
}

#[derive(Default, Clone, Copy)]
pub struct CodemodePlugin;

impl Plugin for CodemodePlugin {
    type Config = CodemodeConfig;

    fn name(&self) -> Option<&str> {
        Some("codemode")
    }

    fn inject(&self) -> Inject {
        Inject::from(["tools"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let tools = ctx.tools()?;
            let session_store: Arc<Mutex<HashMap<String, HashMap<String, Value>>>> =
                Arc::new(Mutex::new(HashMap::new()));
            let worker_sem = Arc::new(Semaphore::new(config.max_workers.max(1)));
            let only_mode = config.only_mode;
            let inline_budget = config.inline_budget;
            let timeout_ms = config.timeout_ms;

            let store_for_exec = Arc::clone(&session_store);
            let sem_for_exec = Arc::clone(&worker_sem);

            let reg = ctx.effect("codemode.register", || {
                Ok::<_, mizar_tools::ToolError>(Some(tools.register(
                    ToolDefinition::async_rich_tool(
                        CODEMODE_TOOL_NAME,
                        create_codemode_description(&[], inline_budget),
                        json!({
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "Async JavaScript code to run in the QuickJS sandbox."
                                }
                            },
                            "required": ["code"]
                        }),
                        move |args, call_ctx| {
                            let store = Arc::clone(&store_for_exec);
                            let sem = Arc::clone(&sem_for_exec);
                            async move {
                                let code = args
                                    .get("code")
                                    .and_then(Value::as_str)
                                    .ok_or("Missing required `code` parameter.")?
                                    .to_string();
                                let rt = call_ctx.ctx.tools()?;
                                execute_codemode_script(
                                    code,
                                    call_ctx,
                                    rt,
                                    store,
                                    sem,
                                    timeout_ms,
                                )
                                .await
                            }
                        },
                    )
                    .with_exposure(ToolExposure::ModelOnly)
                    .with_default_active(false)
                    .with_execution_mode(ToolExecutionMode::Sequential)
                    .with_annotations(ToolAnnotations {
                        read_only: false,
                        destructive: false,
                        idempotent: false,
                        open_world: false,
                    })
                    .with_prepare_loadout(move |loadout| {
                        Some(prepare_codemode_loadout(loadout, only_mode, inline_budget))
                    })
                    .with_prompt_snippet("Execute sandboxed JavaScript to compose multiple tools in one turn"),
                )?))
            })?;

            let mut handles = vec![reg];
            if config.auto_enable {
                let last_state = Arc::new(Mutex::new(AutoEnableState::Inactive));
                sync_auto_enable(&tools, &last_state);
                let tools_sub = tools;
                let last_sub = Arc::clone(&last_state);
                let sub = ctx.on(EVENT_TOOLS_CHANGE, move |_: &()| {
                    sync_auto_enable(&tools_sub, &last_sub);
                })?;
                handles.push(sub);
            }

            Ok(Some(Cleanup::async_fn(move || async move {
                for h in handles {
                    h.dispose().await;
                }
            })))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cordis::{Context, Service};
    use mizar_tools::{ToolExecution, ToolNamespace, ToolOutput};
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn executes_async_js_orchestrating_tools_store_search_and_isolation() {
        let app = Context::new();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app.plugin(CodemodePlugin, CodemodeConfig::default())
            .await
            .unwrap();

        let tools = app.tools().unwrap();
        assert!(
            tools.list_declared(Some("s1")).is_empty(),
            "codemode should stay inactive until Codemode exposure tools register"
        );

        let _c1 = tools
            .register(
                ToolDefinition::async_rich_tool(
                    "mcp__demo__get_item",
                    "Fetch an item by numeric id",
                    json!({
                        "type": "object",
                        "properties": { "id": { "type": "integer" } },
                        "required": ["id"]
                    }),
                    |args, _| async move {
                        let id = args["id"].as_i64().unwrap_or(0);
                        Ok(ToolOutput::text(format!("item {id}"))
                            .with_structured(json!({ "id": id, "name": format!("row-{id}") })))
                    },
                )
                .with_exposure(ToolExposure::Codemode)
                .with_execution_mode(ToolExecutionMode::Parallel)
                .with_namespace(ToolNamespace::new("mcp__demo")),
            )
            .unwrap();

        let declared = tools.list_declared(Some("s1"));
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].name, "codemode");
        assert!(declared[0].description.contains("mcp__demo__get_item"));

        // 1. 组合 Promise.all + searchTools + store()
        let res = tools
            .execute(ToolExecution::new(
                "codemode",
                json!({
                    "code": r#"
                        const found = await searchTools("numeric id");
                        const [a, b] = await Promise.all([
                            tools.mcp__demo__get_item({ id: 10 }),
                            tools.mcp__demo__get_item({ id: 20 })
                        ]);
                        store("last_sum", a.id + b.id);
                        console.log("loaded:", found[0].id);
                        return { names: [a.name, b.name], sum: load("last_sum") };
                    "#
                }),
                Some("s1".into()),
            ))
            .await;
        assert!(!res.is_error, "codemode err: {}", res.content);
        assert!(res.content.contains("loaded: mcp__demo__get_item"));
        assert!(res.content.contains("\"sum\": 30"));
        assert_eq!(res.structured_content.unwrap()["sum"], 30);

        // 2. 验证第二次运行能读到 store("last_sum")
        let res2 = tools
            .execute(ToolExecution::new(
                "codemode",
                json!({ "code": "return load('last_sum');" }),
                Some("s1".into()),
            ))
            .await;
        assert!(!res2.is_error, "{}", res2.content);
        assert_eq!(res2.structured_content, Some(json!(30)));

        // 3. 验证禁止沙箱内套娃调用 ModelOnly 工具
        let res_blocked = tools
            .execute(ToolExecution::new(
                "codemode",
                json!({ "code": "await tools.codemode({ code: 'return 1' });" }),
                Some("s1".into()),
            ))
            .await;
        assert!(res_blocked.is_error);
        assert!(res_blocked.content.contains("无权调用工具"));

        // 4. 验证死循环超时中断机制
        let res_timeout = tools
            .execute_with(
                ToolExecution::new(
                    "codemode",
                    json!({ "code": "// @options: {\"timeoutMs\": 120}\nwhile (true) {}" }),
                    Some("s1".into()),
                ),
                CancellationToken::new(),
                None,
            )
            .await;
        assert!(res_timeout.is_error);
        assert!(
            res_timeout.content.contains("timed out")
                || res_timeout.content.contains("interrupted"),
            "unexpected timeout msg: {}",
            res_timeout.content
        );

        app.fiber().dispose().await;
    }

    #[tokio::test]
    async fn handles_exit_with_message_unspoofable_errors_and_async_host_timeout() {
        let app = Context::new();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app.plugin(CodemodePlugin, CodemodeConfig::default())
            .await
            .unwrap();

        let tools = app.tools().unwrap();
        let nested_cancelled = Arc::new(Mutex::new(false));
        let flag = Arc::clone(&nested_cancelled);
        let _slow = tools
            .register(
                ToolDefinition::async_rich_tool(
                    "mcp__demo__slow",
                    "Sleeps until cancelled",
                    json!({"type": "object", "properties": {}}),
                    move |_, call_ctx| {
                        let cancelled = Arc::clone(&flag);
                        async move {
                            tokio::select! {
                                _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => Ok(ToolOutput::text("done")),
                                _ = call_ctx.cancel.cancelled() => {
                                    *cancelled.lock().unwrap() = true;
                                    Err("cancelled".into())
                                }
                            }
                        }
                    },
                )
                .with_exposure(ToolExposure::Codemode),
            )
            .unwrap();

        // 1. exit("early done") 应正常成功退出并持久化 store、保留消息
        let exit_ok = tools
            .execute(ToolExecution::new(
                "codemode",
                json!({ "code": "store('k', 99); exit('early done');" }),
                Some("s_exit".into()),
            ))
            .await;
        assert!(!exit_ok.is_error, "exit failed: {}", exit_ok.content);
        assert!(
            exit_ok.content.contains("early done"),
            "{}",
            exit_ok.content
        );

        let verify_store = tools
            .execute(ToolExecution::new(
                "codemode",
                json!({ "code": "return load('k');" }),
                Some("s_exit".into()),
            ))
            .await;
        assert_eq!(verify_store.structured_content, Some(json!(99)));

        // 2. 手动抛出含哨兵字符串的普通 Error 仍必须按失败处理，不可伪造 exit
        let spoof_err = tools
            .execute(ToolExecution::new(
                "codemode",
                json!({ "code": "throw new Error('__CODEMODE_EXIT__');" }),
                Some("s_exit".into()),
            ))
            .await;
        assert!(spoof_err.is_error);

        // 3. await 慢速宿主异步工具超出 timeoutMs 时必须及时超时并向子调用传播 cancel
        let async_timeout = tools
            .execute(ToolExecution::new(
                "codemode",
                json!({
                    "code": "// @options: {\"timeoutMs\": 120}\nawait tools.mcp__demo__slow({});"
                }),
                Some("s_exit".into()),
            ))
            .await;
        assert!(async_timeout.is_error);
        assert!(
            async_timeout.content.contains("timed out"),
            "{}",
            async_timeout.content
        );
        assert!(
            *nested_cancelled.lock().unwrap(),
            "script timeout must cancel in-flight nested host tool call"
        );

        app.fiber().dispose().await;
    }

    #[tokio::test]
    async fn codemode_stays_hidden_without_codemode_tools_and_activates_when_registered() {
        let app = Context::new();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app.plugin(CodemodePlugin, CodemodeConfig::default())
            .await
            .unwrap();

        let tools = app.tools().unwrap();
        let _direct = tools
            .register(
                ToolDefinition::sync_tool("read", "Read file", json!({}), |_, _| {
                    Ok::<_, BoxError>("ok".into())
                })
                .with_exposure(ToolExposure::Direct),
            )
            .unwrap();
        let _deferred = tools
            .register(
                ToolDefinition::sync_tool("mcp__x__deferred", "Deferred", json!({}), |_, _| {
                    Ok::<_, BoxError>("ok".into())
                })
                .with_exposure(ToolExposure::Deferred),
            )
            .unwrap();

        assert!(
            !tools
                .list_declared(Some("s1"))
                .iter()
                .any(|t| t.name == CODEMODE_TOOL_NAME),
            "codemode must remain hidden when 0 Codemode tools are registered"
        );

        let cm_guard = tools
            .register(
                ToolDefinition::sync_tool("mcp__x__run", "Codemode tool", json!({}), |_, _| {
                    Ok::<_, BoxError>("ok".into())
                })
                .with_exposure(ToolExposure::Codemode),
            )
            .unwrap();
        assert!(
            tools
                .list_declared(Some("s1"))
                .iter()
                .any(|t| t.name == CODEMODE_TOOL_NAME),
            "codemode must activate once a ToolExposure::Codemode tool is registered"
        );

        match cm_guard {
            Cleanup::Sync(f) => f(),
            Cleanup::Async(f) => f().await,
        }
        assert!(
            !tools
                .list_declared(Some("s1"))
                .iter()
                .any(|t| t.name == CODEMODE_TOOL_NAME),
            "codemode must deactivate again after Codemode tools unregister"
        );

        app.fiber().dispose().await;
    }
}
