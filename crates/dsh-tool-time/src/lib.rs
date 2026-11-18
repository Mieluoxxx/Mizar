//! 最小工具插件：依赖 `tools` 服务，用 `effect` 登记，插件卸载时工具自动消失
//!（对应 `NamiDSH/packages/tool-time`）。

use chrono::Local;
use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use dsh_tools::{ToolDefinition, ToolsContextExt};
use serde_json::json;

#[derive(Default, Clone, Copy)]
pub struct ToolTimePlugin;

impl Plugin for ToolTimePlugin {
    type Config = ();

    fn name(&self) -> Option<&str> {
        Some("tool-time")
    }

    fn inject(&self) -> Inject {
        Inject::from(["tools"])
    }

    fn apply(
        &self,
        ctx: Context,
        _config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let tools = ctx.tools()?;
            ctx.effect("tool-time.register", || {
                Ok::<_, dsh_tools::ToolError>(Some(tools.register(ToolDefinition::sync_tool(
                    "get_time",
                    "获取当前的日期和时间",
                    json!({ "type": "object", "properties": {} }),
                    |_args, _exec| Ok::<_, BoxError>(Local::now().format("%Y/%m/%d %H:%M:%S").to_string()),
                ))?))
            })?;
            Ok(None)
        })
    }
}
