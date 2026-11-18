//! 网页端入口（对应 `NamiDSH/apps/web/src/main.ts`）。
//!
//! 环境变量：
//! - `API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY`（不设就用离线假模型）
//! - `PROTOCOL`：`completions`（默认）/ `responses` / `messages`
//! - `BASE_URL`、`MODEL`、`PORT`、`HOST`、`WORKSPACE`、`STATIC_DIR`

use std::path::PathBuf;

use cordis::{Context, Service};
use dsh_agent_spine::{
    AgentSpineConfig, AgentSpinePlugin, load_dsh_config, resolve_default_session_db_path,
};
use dsh_llm_mock::MockPlugin;
use dsh_llm_retry::{LlmRetryPlugin, RetryConfig};
use dsh_llm_router::LlmRouter;
use dsh_webserver::{WebServerConfig, WebServerPlugin};

fn resolve_static_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("STATIC_DIR") {
        return PathBuf::from(dir);
    }
    let local = PathBuf::from("public");
    if local.join("index.html").exists() {
        return local;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("public")
}

fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = Context::new();

    // 1. 核心套件：llm、sessions、tools、systemPrompt、agentLoop，外加两个工具插件
    let workspace = env_non_empty("WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("failed to resolve cwd"));
    let loaded_cfg = load_dsh_config(&workspace);
    let mut spine_cfg = AgentSpineConfig::new(workspace);
    spine_cfg.session_db_path = resolve_default_session_db_path();
    loaded_cfg.apply_to_spine(&mut spine_cfg);

    let resolved_llm = loaded_cfg.resolve_llm_router();
    if let Some(ref resolved) = resolved_llm {
        spine_cfg.agent.provider = Some(resolved.active_provider.clone());
        spine_cfg.agent.model = resolved.active_model.clone();
    }

    app.plugin(AgentSpinePlugin, spine_cfg);
    for err in &loaded_cfg.errors {
        app.logger_named("config").warn(err);
    }

    // 2. 挂载瞬时故障自动重试中间件
    app.plugin(LlmRetryPlugin, RetryConfig::default());

    // 3. 模型路由：从 ~/.dsh-rs/models.json 与环境变量接入三种协议，或回退到离线假模型
    if let Some(resolved) = resolved_llm {
        app.plugin(LlmRouter::plugin(), resolved.router);
    } else {
        app.plugin(MockPlugin, ());
        app.logger_named("web")
            .warn("没有设置 API_KEY 且未配置 ~/.dsh-rs/models.json，使用离线假模型");
    }

    // 4. 入口：网页服务
    let port = env_non_empty("PORT")
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(3000);
    let host = env_non_empty("HOST").unwrap_or_else(|| "127.0.0.1".to_string());

    app.plugin(
        WebServerPlugin,
        WebServerConfig {
            port,
            host,
            static_dir: resolve_static_dir(),
        },
    );

    tokio::signal::ctrl_c().await?;
    app.fiber().dispose().await;
    Ok(())
}
