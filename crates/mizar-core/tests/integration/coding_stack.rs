//! 全链路集成测试：验证 `AgentSpineConfig::coding` 一次性装配的
//! `read + bash + edit + write + grep + find + mcp + tool_search + codemode`
//! 以及 `EmptyResponse` 自动重试在真实 `AgentLoop` 中协同工作。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cordis::{Context, CordisError, PluginDef};
use futures::channel::mpsc;
use mizar_core::agent_loop::{AgentEvent, AgentLoopContextExt};
use mizar_core::spine::{AgentSpineConfig, AgentSpinePlugin, AgentToolsMode};
use mizar_llm::retry::{LlmRetryPlugin, RetryConfig};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmError,
    StreamChunk, ToolCall,
};
use mizar_mcp::{McpExposure, McpManager, McpServerConfig};
use mizar_session::SessionContextExt;
use serde_json::json;
use tokio_util::sync::CancellationToken;

const MOCK_FFF_PY: &str = r#"
import json, sys

TOOLS = [
    {
        "name": "find_files",
        "description": "Fuzzy file lookup via fff index",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "annotations": {"readOnlyHint": True}
    },
    {
        "name": "grep",
        "description": "Indexed content search via fff",
        "inputSchema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        "outputSchema": {"type": "object", "properties": {"hits": {"type": "array"}}},
        "annotations": {"readOnlyHint": True}
    },
    {
        "name": "multi_grep",
        "description": "Legacy multi pattern OR search",
        "inputSchema": {"type": "object", "properties": {"patterns": {"type": "array"}}, "required": ["patterns"]}
    }
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    rid = msg.get("id")
    m = msg.get("method")
    if m == "notifications/initialized":
        continue
    if m == "initialize":
        res = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "instructions": "fff index server"}
    elif m == "tools/list":
        res = {"tools": TOOLS}
    elif m == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments", {})
        if name == "grep":
            pat = args.get("pattern", "")
            res = {
                "content": [{"type": "text", "text": f"hit:{pat}"}],
                "structuredContent": {"hits": [{"path": "src/app.rs", "pattern": pat}]},
                "isError": False
            }
        else:
            res = {"content": [{"type": "text", "text": f"fff:{name}"}], "isError": False}
    else:
        res = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}) + "\n")
    sys.stdout.flush()
"#;

#[tokio::test]
async fn full_coding_stack_end_to_end_with_mcp_tool_search_codemode_and_empty_retry() {
    let ws = std::env::temp_dir().join(format!("mizar_full_stack_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(ws.join(".mizar")).unwrap();
    let mock_py = ws.join("fff_mock.py");
    std::fs::write(&mock_py, MOCK_FFF_PY).unwrap();

    // 在工作区配置 fff MCP：默认 codemode 暴露，但强制把 multi_grep 屏蔽为 hidden
    let mut cfg = AgentSpineConfig::coding(&ws).with_disabled_plugins(["session-title"]);
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = cfg.tools_mode
    {
        mcp.load_config_files = false;
        mcp.background_connect = false;
        let mut fff = McpServerConfig::stdio("python3", vec![mock_py.to_string_lossy().into()]);
        fff.exposure = McpExposure::Codemode;
        fff.tool_exposure
            .insert("multi_grep".into(), McpExposure::Hidden);
        mcp.servers.insert("fff".into(), fff);

        skills.include_user_roots = false;
        skills.agents_home = Some(ws.join("isolated-agents"));
        instructions.include_default_global = false;
    }

    struct ScriptedAdapter {
        attempt: AtomicUsize,
        seen_tools_per_step: Arc<Mutex<Vec<Vec<String>>>>,
    }
    impl LlmAdapter for ScriptedAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let n = self.attempt.fetch_add(1, Ordering::SeqCst);
            let tool_names: Vec<String> = options.tools.iter().map(|t| t.name.clone()).collect();
            self.seen_tools_per_step.lock().unwrap().push(tool_names);
            let (tx, rx) = mpsc::unbounded();

            tokio::spawn(async move {
                match n {
                    // 第 0 次故意模拟上游返回瞬时 EmptyResponse 文案，验证自动重试恢复
                    0 => {
                        let _ = tx.unbounded_send(Err(LlmError::classify_provider_error(
                            Some(400),
                            None,
                            None,
                            "model output error: model output must contain either output text or tool calls, these cannot both be empty, please try again",
                        )));
                    }
                    // 第 1 次重试成功：先建文件并按需激活延迟 MCP 工具（write/edit 已支持跨文件 Parallel，有依赖步骤分轮发出）
                    1 => {
                        let calls = vec![
                            ToolCall {
                                id: "t1".into(),
                                name: "write".into(),
                                arguments: json!({"path": "src/main.rs", "content": "fn main() {\n    println!(\"v1\");\n}\n"}).to_string(),
                            },
                            ToolCall {
                                id: "t8".into(),
                                name: "tool_search".into(),
                                arguments: json!({"query": "fff find_files"}).to_string(),
                            },
                        ];
                        for (idx, c) in calls.into_iter().enumerate() {
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(idx, c)));
                        }
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    }
                    // 第 2 步：此时 mcp__fff__find_files 已激活；对刚写入的文件执行 edit
                    2 => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: "t2".into(),
                                name: "edit".into(),
                                arguments: json!({"path": "src/main.rs", "edits": [{"oldText": "v1", "newText": "v2"}]}).to_string(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    }
                    // 第 3 步：单轮并发调用 read + grep + find + bash + codemode + mcp__fff__find_files
                    3 => {
                        let calls = vec![
                            ToolCall {
                                id: "t3".into(),
                                name: "read".into(),
                                arguments: json!({"path": "src/main.rs"}).to_string(),
                            },
                            ToolCall {
                                id: "t4".into(),
                                name: "grep".into(),
                                arguments: json!({"pattern": "v2", "literal": true}).to_string(),
                            },
                            ToolCall {
                                id: "t5".into(),
                                name: "find".into(),
                                arguments: json!({"pattern": "*.rs"}).to_string(),
                            },
                            ToolCall {
                                id: "t6".into(),
                                name: "bash".into(),
                                arguments: json!({"command": "echo ok_bash"}).to_string(),
                            },
                            ToolCall {
                                id: "t7".into(),
                                name: "codemode".into(),
                                arguments: json!({
                                    "code": "const g = await tools.mcp__fff__grep({ pattern: 'v2' }); return g.hits[0];"
                                }).to_string(),
                            },
                            ToolCall {
                                id: "t9".into(),
                                name: "mcp__fff__find_files".into(),
                                arguments: json!({"query": "main.rs"}).to_string(),
                            },
                        ];
                        for (idx, c) in calls.into_iter().enumerate() {
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(idx, c)));
                        }
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    }
                    _ => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("all_mechanisms_verified")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                }
            });
            Box::pin(rx)
        }
    }

    let seen_tools = Arc::new(Mutex::new(Vec::new()));
    let results_log = Arc::new(Mutex::new(Vec::<(String, bool, String)>::new()));

    let app = Context::new();
    app.plugin(AgentSpinePlugin, cfg);
    app.plugin(LlmRetryPlugin, RetryConfig::default());

    let adapter = Arc::new(ScriptedAdapter {
        attempt: AtomicUsize::new(0),
        seen_tools_per_step: Arc::clone(&seen_tools),
    });
    app.plugin(
        PluginDef::sync("scripted", move |ctx, ()| {
            let llm = ctx.llm()?;
            let a = Arc::clone(&adapter);
            ctx.effect("scripted.reg", move || {
                Ok::<_, LlmError>(Some(llm.register_adapter("scripted", a)?))
            })?;
            Ok::<_, CordisError>(())
        })
        .inject(["llm"]),
        (),
    );

    // 等待全部依赖和 McpManager 就绪
    for rt in app.registry().values() {
        for f in rt.fibers() {
            let _ = f.wait_settled().await;
        }
    }
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    for rt in app.registry().values() {
        for f in rt.fibers() {
            let _ = f.wait_settled().await;
        }
    }
    if let Some(mgr) = app.get::<McpManager>("mcp") {
        mgr.wait_ready().await;
    }

    let results_sub = Arc::clone(&results_log);
    let _tool_result_guard = app.agent_loop().unwrap().on_event(move |ev| {
        if let AgentEvent::ToolResult(ev) = ev {
            results_sub
                .lock()
                .unwrap()
                .push((ev.name.clone(), ev.is_error, ev.content.clone()));
        }
    });

    // MCP 不应向静态 SystemPrompt 注入动态 mcp_servers 段落（保持前缀字节冻结）
    let prompt = mizar_core::system_prompt::SystemPromptContextExt::system_prompt(&app)
        .unwrap()
        .build();
    assert!(!prompt.contains("mcp_servers"), "prompt:\n{prompt}");
    assert!(!prompt.contains("mcp__fff"), "prompt:\n{prompt}");

    let session = app.sessions().unwrap().create(None);
    let reply = app
        .agent_loop()
        .unwrap()
        .run(
            Arc::clone(&session),
            "verify full stack",
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(reply, "all_mechanisms_verified");

    // 校验所有工具全部执行成功且无报错
    {
        let recorded = results_log.lock().unwrap();
        assert_eq!(
            recorded.len(),
            9,
            "expected 9 tool executions: {recorded:?}"
        );
        for (name, is_err, content) in recorded.iter() {
            assert!(!*is_err, "tool `{name}` returned error: {content}");
        }
        assert!(
            recorded
                .iter()
                .any(|(n, _, c)| n == "codemode" && c.contains("src/app.rs"))
        );
        assert!(
            recorded
                .iter()
                .any(|(n, _, c)| n == "mcp__fff__find_files" && c.contains("fff:find_files"))
        );
    }

    // 校验工具可见性演化：step1 有 tool_search/codemode/recall 且无 skill（0 skills）与 mcp__fff__*；step2 多出 mcp__fff__find_files 且始终没有 multi_grep
    {
        let steps = seen_tools.lock().unwrap();
        assert!(steps[1].contains(&"tool_search".to_string()));
        assert!(steps[1].contains(&"codemode".to_string()));
        assert!(steps[1].contains(&"recall".to_string()));
        assert!(!steps[1].contains(&"skill".to_string()));
        assert!(!steps[1].contains(&"mcp__fff__find_files".to_string()));
        assert!(!steps[1].contains(&"mcp__fff__multi_grep".to_string()));

        assert!(steps[2].contains(&"mcp__fff__find_files".to_string()));
        assert!(!steps[2].contains(&"mcp__fff__multi_grep".to_string()));
    }

    let _ = std::fs::remove_dir_all(&ws);
    app.fiber().dispose().await;
}
