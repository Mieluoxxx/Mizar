//! 工作区与运行期观测信息汇总（工具、MCP、指令文件、Git 变更、用户技能与轨迹异常）。

use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use mizar_core::skills::SkillLookupOptions;
use mizar_core::spine::MizarRuntime;
use mizar_mcp::{McpConnectionState, McpManager};
use mizar_tools::ToolExposure;
use mizar_types::client::{
    FileTouchRow, InstructionSummaryRow, McpSummaryRow, SidePanelData, ToolSummaryRow,
};
use tokio_util::sync::CancellationToken;

/// 工作区与后端服务观测数据的纯数据快照，供各端（TUI / Merak）渲染侧栏或仪表盘。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkspacePanelSnapshot {
    pub instructions: Vec<InstructionSummaryRow>,
    pub git_changes: Vec<FileTouchRow>,
    pub git_diff_stat: String,
    pub tools: Vec<ToolSummaryRow>,
    pub mcps: Vec<McpSummaryRow>,
    pub findings: Vec<String>,
}

impl WorkspacePanelSnapshot {
    /// 一次性采集工作区及全部后端观测维度。
    pub async fn collect(
        runtime: &MizarRuntime,
        workspace: &Path,
        session_id: Option<&str>,
    ) -> Self {
        collect_workspace_panel_snapshot(runtime, workspace, session_id).await
    }

    /// 将工作区观测数据合并进前端侧栏模型（保留前端自己维护的上下文用量与已触达文件列表）。
    pub fn apply_to_side_panel(&self, panel: &mut SidePanelData, runtime_anomalies: &[String]) {
        panel.instructions = self.instructions.clone();
        panel.git_changes = self.git_changes.clone();
        panel.git_diff_stat = self.git_diff_stat.clone();
        panel.tools = self.tools.clone();
        panel.mcps = self.mcps.clone();

        let mut merged = runtime_anomalies.to_vec();
        for line in &self.findings {
            if !merged.contains(line) {
                merged.push(line.clone());
            }
        }
        panel.findings = merged;
    }
}

/// 收集完整工作区面板快照（同步聚合服务状态与异步采集 Git 摘要）。
pub async fn collect_workspace_panel_snapshot(
    runtime: &MizarRuntime,
    workspace: &Path,
    session_id: Option<&str>,
) -> WorkspacePanelSnapshot {
    let instructions = collect_instructions_summary(runtime);
    let ((git_changes, git_diff_stat), mcp_mgr) =
        tokio::join!(collect_git_summary(workspace), runtime.mcp());
    let tools = collect_tools_summary(runtime, session_id);
    let mcps = format_mcps_summary(runtime.modules.is_enabled("mcp"), mcp_mgr.as_deref());
    let findings = session_id
        .map(|sid| collect_trajectory_findings(runtime, sid))
        .unwrap_or_default();

    WorkspacePanelSnapshot {
        instructions,
        git_changes,
        git_diff_stat,
        tools,
        mcps,
        findings,
    }
}

/// 汇总全局已注册工具及其在当前会话下的可见性（exposure 与声明状态）。
pub fn collect_tools_summary(
    runtime: &MizarRuntime,
    session_id: Option<&str>,
) -> Vec<ToolSummaryRow> {
    let rt = &runtime.tools;
    let declared: HashSet<String> = rt
        .list_declared(session_id)
        .into_iter()
        .map(|t| t.name)
        .collect();
    rt.list_all()
        .into_iter()
        .map(|info| {
            let exposure = match info.exposure {
                ToolExposure::Direct => "direct",
                ToolExposure::ModelOnly => "model-only",
                ToolExposure::Codemode => "codemode",
                ToolExposure::Deferred => "deferred",
                ToolExposure::Hidden => "hidden",
            }
            .to_string();
            let is_decl = declared.contains(&info.name);
            ToolSummaryRow {
                name: info.name,
                exposure,
                declared: is_decl,
            }
        })
        .collect()
}

fn format_mcps_summary(enabled: bool, mgr: Option<&McpManager>) -> Vec<McpSummaryRow> {
    if !enabled {
        return Vec::new();
    }
    let Some(mgr) = mgr else {
        return Vec::new();
    };
    mgr.statuses()
        .into_iter()
        .map(|s| {
            let (status, ok) = match s.state {
                McpConnectionState::Connecting => ("连接中…".to_string(), false),
                McpConnectionState::Connected {
                    tools_count,
                    has_resources,
                } => (
                    format!(
                        "已连接 ({tools_count} 工具{})",
                        if has_resources { " + 资源" } else { "" }
                    ),
                    true,
                ),
                McpConnectionState::Disabled => ("已禁用".to_string(), false),
                McpConnectionState::Failed(err) => (format!("失败: {err}"), false),
            };
            McpSummaryRow {
                name: s.name,
                status,
                ok,
            }
        })
        .collect()
}

/// 汇总所有已配置的 MCP Server 连接状态与可用能力。
pub fn collect_mcps_summary(runtime: &MizarRuntime) -> Vec<McpSummaryRow> {
    format_mcps_summary(
        runtime.modules.is_enabled("mcp"),
        runtime.try_mcp().as_deref(),
    )
}

/// 返回当前工作区启动期加载的 `AGENTS.md` / `CLAUDE.md` 指令作用域摘要。
pub fn collect_instructions_summary(runtime: &MizarRuntime) -> Vec<InstructionSummaryRow> {
    let Some(inst) = runtime.instructions.as_ref() else {
        return Vec::new();
    };
    let report = inst.report();
    report
        .scopes
        .into_iter()
        .map(|s| InstructionSummaryRow {
            path: s.display_path,
            scope: s.scope_kind.as_str().to_string(),
            bytes: s.bytes,
            truncated: s.truncated,
            omitted: s.omitted,
        })
        .collect()
}

/// 带超时采集工作区的 `git status --porcelain`（最多 24 条）与 `git diff --shortstat`。
pub async fn collect_git_summary(workspace: &Path) -> (Vec<FileTouchRow>, String) {
    let mut status_cmd = tokio::process::Command::new("git");
    status_cmd
        .arg("status")
        .arg("--porcelain")
        .current_dir(workspace)
        .kill_on_drop(true);
    let status_fut = tokio::time::timeout(Duration::from_millis(300), status_cmd.output());

    let mut stat_cmd = tokio::process::Command::new("git");
    stat_cmd
        .arg("diff")
        .arg("--shortstat")
        .current_dir(workspace)
        .kill_on_drop(true);
    let stat_fut = tokio::time::timeout(Duration::from_millis(300), stat_cmd.output());

    let (status_res, stat_res) = tokio::join!(status_fut, stat_fut);

    let mut changes = Vec::new();
    if let Ok(Ok(out)) = status_res
        && out.status.success()
    {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines().take(24) {
            if line.len() < 4 {
                continue;
            }
            let code = line[..2].trim();
            let path = line[3..].trim().to_string();
            if !path.is_empty() {
                changes.push(FileTouchRow {
                    path,
                    op: if code.is_empty() {
                        "?".to_string()
                    } else {
                        code.to_string()
                    },
                });
            }
        }
    }

    let mut stat = String::new();
    if let Ok(Ok(out)) = stat_res
        && out.status.success()
    {
        stat = String::from_utf8_lossy(&out.stdout).trim().to_string();
    }
    (changes, stat)
}

/// 列举当前工作区下可由用户通过 `/<skill>` 直接调用的技能名称列表。
pub async fn collect_user_skill_names(runtime: &MizarRuntime, workspace: &Path) -> Vec<String> {
    let Some(reg) = runtime.skills.as_ref() else {
        return Vec::new();
    };
    reg.list(SkillLookupOptions {
        cwd: Some(workspace.to_path_buf()),
        cancel: CancellationToken::new(),
    })
    .await
    .into_iter()
    .filter(|s| s.is_user_invocable())
    .map(|s| s.name)
    .collect()
}

/// 采集指定会话的轨迹异常发现（去重格式化为 `[Severity] title: summary`）。
pub fn collect_trajectory_findings(runtime: &MizarRuntime, session_id: &str) -> Vec<String> {
    let Some(traj) = runtime.trajectory.as_ref() else {
        return Vec::new();
    };
    let raw_findings = traj.findings_only(session_id);
    let mut findings = Vec::new();
    for f in raw_findings {
        let line = format!("[{:?}] {}: {}", f.severity, f.title, f.summary);
        if !findings.contains(&line) {
            findings.push(line);
        }
    }
    findings
}
