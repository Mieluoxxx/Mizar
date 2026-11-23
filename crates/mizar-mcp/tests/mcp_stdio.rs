//! MCP stdio 客户端端到端测试：通过本地 Python stdio mock server 模拟 `fff-mcp`，
//! 验证 `Direct` / `Hidden`（屏蔽 `multi_grep`） / `Codemode` 分级暴露、工具执行、资源读取及 SystemPrompt 段落。

use std::sync::Arc;

use cordis::{Context, Service};
use mizar_mcp::{McpContextExt, McpExposure, McpManager, McpPluginConfig, McpServerConfig};
use mizar_tools::{ToolExecution, ToolRuntime, ToolsContextExt};
use serde_json::json;

const MOCK_MCP_SERVER_PY: &str = r#"
import json, sys

TOOLS = [
    {
        "name": "find_files",
        "description": "Fuzzy search files by name",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "annotations": {"readOnlyHint": True}
    },
    {
        "name": "grep",
        "description": "Fast indexed content grep",
        "inputSchema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        "outputSchema": {"type": "object", "properties": {"matches": {"type": "array"}}},
        "annotations": {"readOnlyHint": True}
    },
    {
        "name": "multi_grep",
        "description": "Multi pattern OR search",
        "inputSchema": {"type": "object", "properties": {"patterns": {"type": "array"}}, "required": ["patterns"]}
    }
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    req_id = msg.get("id")
    method = msg.get("method")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}, "resources": {}},
                "instructions": "Indexed repo search server"
            }
        }
    elif method == "tools/list":
        resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments", {})
        if name == "grep":
            pat = args.get("pattern", "")
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"src/lib.rs:1: matched {pat}"}],
                    "structuredContent": {"matches": [{"file": "src/lib.rs", "line": 1, "pattern": pat}]},
                    "isError": False
                }
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": f"called {name}"}], "isError": False}
            }
    elif method == "resources/list":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "resources": [
                    {"uri": "fff://status", "name": "index_status", "mimeType": "text/plain"}
                ]
            }
        }
    elif method == "resources/templates/list":
        resp = {"jsonrpc": "2.0", "id": req_id, "result": {"resourceTemplates": []}}
    elif method == "resources/read":
        uri = msg["params"]["uri"]
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "contents": [{"uri": uri, "text": "indexed=42 files", "mimeType": "text/plain"}]
            }
        }
    else:
        resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "method not found"}}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"#;

#[tokio::test]
async fn connects_stdio_mcp_applies_hidden_and_codemode_exposures_and_reads_resources() {
    let dir = std::env::temp_dir().join(format!("mizar_mcp_stdio_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let script_path = dir.join("mock_mcp.py");
    std::fs::write(&script_path, MOCK_MCP_SERVER_PY).unwrap();

    let mut fff_cfg = McpServerConfig::stdio("python3", vec![script_path.to_string_lossy().into()]);
    fff_cfg.exposure = McpExposure::Direct;
    fff_cfg
        .tool_exposure
        .insert("multi_grep".into(), McpExposure::Hidden);
    fff_cfg
        .tool_exposure
        .insert("find_files".into(), McpExposure::Codemode);

    let mut cfg = McpPluginConfig::new(&dir);
    cfg.load_config_files = false;
    cfg.background_connect = false;
    cfg.servers.insert("fff".into(), fff_cfg);

    let app = Context::new();
    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app.plugin(McpManager::plugin(), cfg).await.unwrap();

    let mgr: Arc<McpManager> = app.get("mcp").unwrap();
    mgr.wait_ready().await;

    let tools = app.tools().unwrap();
    let declared = tools.list_declared(Some("s1"));
    let declared_names: Vec<&str> = declared.iter().map(|t| t.name.as_str()).collect();

    assert!(
        declared_names.contains(&"mcp__fff__grep"),
        "{declared_names:?}"
    );
    assert!(declared_names.contains(&"list_mcp_resources"));
    assert!(declared_names.contains(&"read_mcp_resource"));
    assert!(
        !declared_names.contains(&"mcp__fff__multi_grep"),
        "multi_grep must stay hidden"
    );
    assert!(
        !declared_names.contains(&"mcp__fff__find_files"),
        "codemode-only tool should not be declared directly"
    );

    // callable 里应当包含 find_files（codemode）和 grep（direct），不含 multi_grep（hidden）
    let callable = tools.list_callable(Some("s1"));
    let callable_names: Vec<&str> = callable.iter().map(|t| t.name.as_str()).collect();
    assert!(callable_names.contains(&"mcp__fff__find_files"));
    assert!(callable_names.contains(&"mcp__fff__grep"));
    assert!(!callable_names.contains(&"mcp__fff__multi_grep"));

    let find_info = callable
        .iter()
        .find(|t| t.name == "mcp__fff__find_files")
        .unwrap();
    let ns = find_info.namespace.as_ref().unwrap();
    assert_eq!(ns.description, None);
    assert_eq!(
        ns.instructions.as_deref(),
        Some("Indexed repo search server")
    );

    // 调用 mcp__fff__grep 并保留 structured_content
    let grep_res = tools
        .execute(ToolExecution::new(
            "mcp__fff__grep",
            json!({ "pattern": "hello" }),
            Some("s1".into()),
        ))
        .await;
    assert!(!grep_res.is_error, "{}", grep_res.content);
    assert!(grep_res.content.contains("matched hello"));
    assert_eq!(
        grep_res.structured_content.as_ref().unwrap()["matches"][0]["file"],
        "src/lib.rs"
    );

    // 被设为 Hidden 的 multi_grep 拒绝调用
    let hidden_res = tools
        .execute(ToolExecution::new(
            "mcp__fff__multi_grep",
            json!({ "patterns": ["a"] }),
            Some("s1".into()),
        ))
        .await;
    assert!(hidden_res.is_error);

    // 资源读取工具
    let list_res = tools
        .execute(ToolExecution::new("list_mcp_resources", json!({}), None))
        .await;
    assert!(!list_res.is_error, "{}", list_res.content);
    assert!(list_res.content.contains("fff://status"));

    let read_res = tools
        .execute(ToolExecution::new(
            "read_mcp_resource",
            json!({ "server": "fff", "uri": "fff://status" }),
            None,
        ))
        .await;
    assert!(!read_res.is_error, "{}", read_res.content);
    assert!(read_res.content.contains("indexed=42 files"));

    // render_servers_section 辅助方法本身仍可按需渲染 MCP server 摘要
    let section = app
        .mcp()
        .unwrap()
        .render_servers_section()
        .expect("render_servers_section should summarize codemode servers");
    assert!(
        section.contains("Call the tools of `codemode` servers from codemode scripts."),
        "section:\n{section}"
    );
    assert!(
        section.contains("mcp__fff (codemode): Indexed repo search server"),
        "section:\n{section}"
    );

    let _ = std::fs::remove_dir_all(&dir);
    app.fiber().dispose().await;
}

#[tokio::test]
async fn skip_global_config_ignores_global_mcp_but_loads_project_mcp() {
    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "mizar_mcp_skip_global_{}_{uniq}",
        std::process::id()
    ));
    let global_dir = dir.join("global_home");
    let workspace = dir.join("workspace");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&global_dir).unwrap();
    std::fs::create_dir_all(workspace.join(".mizar")).unwrap();

    let script = dir.join("server.py");
    std::fs::write(&script, MOCK_MCP_SERVER_PY).unwrap();
    let marker = dir.join("global_spawned.marker");

    let global_cfg = json!({
        "mcpServers": {
            "global_srv": {
                "command": "python3",
                "args": ["-c", format!("import pathlib; pathlib.Path({:?}).write_text('ran')", marker.display().to_string())],
                "exposure": "direct"
            }
        }
    });
    std::fs::write(
        global_dir.join("mcp.json"),
        serde_json::to_string_pretty(&global_cfg).unwrap(),
    )
    .unwrap();

    let project_cfg = json!({
        "mcpServers": {
            "proj_srv": {
                "command": "python3",
                "args": [&script],
                "exposure": "direct"
            }
        }
    });
    std::fs::write(
        workspace.join(".mizar").join("mcp.json"),
        serde_json::to_string_pretty(&project_cfg).unwrap(),
    )
    .unwrap();

    let app = Context::new();
    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app.plugin(
        McpManager::plugin(),
        McpPluginConfig {
            workspace: workspace.clone(),
            global_config_dir: Some(global_dir.clone()),
            skip_global_config: true,
            load_config_files: true,
            servers: Default::default(),
            background_connect: false,
        },
    )
    .await
    .unwrap();
    for rt in app.registry().values() {
        for f in rt.fibers() {
            let _ = f.wait_settled().await;
        }
    }

    let mcp = app.service::<McpManager>("mcp").unwrap();
    mcp.wait_ready().await;

    let statuses = mcp.statuses();
    assert!(
        statuses.iter().any(|s| s.name == "proj_srv"),
        "workspace MCP should still load: {statuses:?}"
    );
    assert!(
        !statuses.iter().any(|s| s.name == "global_srv"),
        "global MCP must be skipped: {statuses:?}"
    );
    assert!(
        !marker.exists(),
        "global MCP server process must never be spawned"
    );

    let tools = app.tools().unwrap();
    let names: Vec<String> = tools
        .list_callable(None)
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(names.contains(&"mcp__proj_srv__grep".to_string()));
    assert!(!names.iter().any(|n| n.contains("global_srv")));

    app.fiber().dispose().await;
    let _ = std::fs::remove_dir_all(&dir);
}
