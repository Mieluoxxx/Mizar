//! `codemode` 插件：把 `codemode` 工具常驻注册到 `ToolRuntime`，支持 QuickJS 沙箱执行、
//! `prepare_loadout` 动态目录生成与声明工具反向引导。

pub mod declarations;
pub mod sandbox;
pub mod schema_ts;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_tools::{
    ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolExposure, ToolsContextExt,
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
    /// 为 `true` 时类似 `pi` 的 `only` 模式：隐藏 `Direct` 工具的直接声明，改由脚本统一调度（默认 false）。
    pub only_mode: bool,
    pub inline_budget: usize,
    pub max_workers: usize,
    pub timeout_ms: u64,
}

impl Default for CodemodeConfig {
    fn default() -> Self {
        Self {
            only_mode: false,
            inline_budget: DEFAULT_CODEMODE_INLINE_BUDGET,
            max_workers: 2,
            timeout_ms: DEFAULT_CODEMODE_TIMEOUT_MS,
        }
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

            ctx.effect("codemode.register", || {
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
                                    .ok_or("缺少 code 参数")?
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

            Ok(None)
        })
    }
}
