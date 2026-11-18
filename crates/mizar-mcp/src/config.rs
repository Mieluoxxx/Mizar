//! MCP 服务器配置解析（兼容标准 `mcpServers` JSON 结构及 `pi` 的 `exposure` / `toolExposure` 分级控制）。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mizar_tools::{ToolExposure, wildcard_matches};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum McpExposure {
    #[default]
    Codemode,
    #[serde(alias = "codemode-deferred")]
    Deferred,
    Direct,
    Hidden,
}

impl From<McpExposure> for ToolExposure {
    fn from(value: McpExposure) -> Self {
        match value {
            McpExposure::Codemode => ToolExposure::Codemode,
            McpExposure::Deferred => ToolExposure::Deferred,
            McpExposure::Direct => ToolExposure::Direct,
            McpExposure::Hidden => ToolExposure::Hidden,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub struct McpServerConfig {
    /// Stdio 启动命令。
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,

    /// Streamable HTTP MCP 服务端地址。
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,

    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub exposure: McpExposure,
    #[serde(default, alias = "tool_exposure")]
    pub tool_exposure: HashMap<String, McpExposure>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, alias = "timeout_ms")]
    pub timeout_ms: Option<u64>,
}

fn default_true() -> bool {
    true
}

impl McpServerConfig {
    pub fn stdio(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: Some(command.into()),
            args,
            enabled: true,
            ..Default::default()
        }
    }

    pub fn http(url: impl Into<String>) -> Self {
        Self {
            url: Some(url.into()),
            enabled: true,
            ..Default::default()
        }
    }

    pub fn is_override_only(&self) -> bool {
        self.command.is_none() && self.url.is_none()
    }

    pub fn resolve_tool_exposure(&self, raw_tool_name: &str) -> McpExposure {
        if let Some(&exact) = self.tool_exposure.get(raw_tool_name) {
            return exact;
        }
        // 最长具体前缀/模式优先匹配通配符规则
        let mut best: Option<(usize, McpExposure)> = None;
        for (pattern, &exp) in &self.tool_exposure {
            if pattern.contains('*') && wildcard_matches(pattern, raw_tool_name) {
                let specificity = pattern.bytes().filter(|&b| b != b'*').count();
                if best.is_none_or(|(best_spec, _)| specificity > best_spec) {
                    best = Some((specificity, exp));
                }
            }
        }
        best.map(|(_, exp)| exp).unwrap_or(self.exposure)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawMcpFile {
    #[serde(default)]
    mcp_servers: HashMap<String, RawMcpEntry>,
    #[serde(default)]
    auto_enable_codemode: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct RawMcpEntry {
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Option<Vec<String>>,
    #[serde(default)]
    env: Option<HashMap<String, String>>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    headers: Option<HashMap<String, String>>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    exposure: Option<McpExposure>,
    #[serde(default, alias = "tool_exposure")]
    tool_exposure: Option<HashMap<String, McpExposure>>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default, alias = "timeout_ms", alias = "timeout")]
    timeout_ms: Option<u64>,
}

impl RawMcpEntry {
    fn is_override_only(&self) -> bool {
        self.command.is_none() && self.url.is_none()
    }

    fn into_full_config(self) -> McpServerConfig {
        McpServerConfig {
            command: self.command,
            args: self.args.unwrap_or_default(),
            env: self.env.unwrap_or_default(),
            cwd: self.cwd,
            url: self.url,
            headers: self.headers.unwrap_or_default(),
            enabled: self.enabled.unwrap_or(true),
            exposure: self.exposure.unwrap_or_default(),
            tool_exposure: self.tool_exposure.unwrap_or_default(),
            description: self.description,
            timeout_ms: self.timeout_ms,
        }
    }

    fn apply_override_onto(&self, target: &mut McpServerConfig) {
        if let Some(enabled) = self.enabled {
            target.enabled = enabled;
        }
        if let Some(exposure) = self.exposure {
            target.exposure = exposure;
        }
        if let Some(tool_exp) = &self.tool_exposure {
            for (k, v) in tool_exp {
                target.tool_exposure.insert(k.clone(), *v);
            }
        }
        if let Some(desc) = &self.description {
            target.description = Some(desc.clone());
        }
        if let Some(t) = self.timeout_ms {
            target.timeout_ms = Some(t);
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedMcpEntry {
    pub name: String,
    pub config: McpServerConfig,
    pub source: String,
}

#[derive(Debug, Clone, Default)]
pub struct LoadedMcpConfig {
    pub entries: Vec<LoadedMcpEntry>,
    pub auto_enable_codemode: bool,
    pub errors: Vec<String>,
}

/// 解析 `${VAR}` 环境变量占位符。
pub fn expand_env_vars(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        if let Some(end) = after.find('}') {
            let var_name = &after[..end];
            if let Ok(val) = std::env::var(var_name) {
                out.push_str(&val);
            }
            rest = &after[end + 1..];
        } else {
            out.push_str("${");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

pub fn sanitize_mcp_identifier(raw: &str) -> String {
    let out: String = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        "server".to_string()
    } else {
        out
    }
}

pub fn mcp_namespace(server_name: &str) -> String {
    format!("mcp__{}", sanitize_mcp_identifier(server_name))
}

pub fn mcp_tool_name(server_name: &str, tool_name: &str) -> String {
    format!("{}__{}", mcp_namespace(server_name), tool_name)
}

fn read_mcp_file(path: &Path) -> Result<Option<RawMcpFile>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let no_bom = raw.strip_prefix('\u{FEFF}').unwrap_or(&raw);
    let parsed: RawMcpFile =
        serde_json::from_str(no_bom).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Some(parsed))
}

pub const CONFIG_DIR_NAME: &str = ".mizar";

/// 全局配置目录（支持 `MIZAR_HOME` 环境变量覆盖，默认 `~/.mizar`）。
pub fn resolve_global_config_dir() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("MIZAR_HOME") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    dirs::home_dir().map(|home| home.join(CONFIG_DIR_NAME))
}

pub fn load_mcp_config(workspace: &Path) -> LoadedMcpConfig {
    load_mcp_config_with_global(workspace, resolve_global_config_dir().as_deref())
}

pub fn load_mcp_config_with_global(
    workspace: &Path,
    global_dir: Option<&Path>,
) -> LoadedMcpConfig {
    let mut map: HashMap<String, LoadedMcpEntry> = HashMap::new();
    let mut errors = Vec::new();
    let mut auto_enable_codemode = true;

    let mut files = Vec::new();
    if let Some(dir) = global_dir {
        files.push(("global", dir.join("mcp.json")));
    }
    files.push(("project", workspace.join(CONFIG_DIR_NAME).join("mcp.json")));

    for (scope, path) in files {
        match read_mcp_file(&path) {
            Ok(Some(file)) => {
                if let Some(auto) = file.auto_enable_codemode {
                    auto_enable_codemode = auto;
                }
                for (name, raw_entry) in file.mcp_servers {
                    if scope == "project"
                        && raw_entry.is_override_only()
                        && let Some(existing) = map.get_mut(&name)
                    {
                        raw_entry.apply_override_onto(&mut existing.config);
                        continue;
                    }
                    let cfg = raw_entry.into_full_config();
                    if cfg.command.is_none() && cfg.url.is_none() {
                        errors.push(format!(
                            "{}: MCP server `{name}` missing command or url",
                            path.display()
                        ));
                        continue;
                    }
                    map.insert(
                        name.clone(),
                        LoadedMcpEntry {
                            name,
                            config: cfg,
                            source: path.display().to_string(),
                        },
                    );
                }
            }
            Ok(None) => {}
            Err(err) => errors.push(err),
        }
    }

    let mut entries: Vec<LoadedMcpEntry> = map.into_values().collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));

    LoadedMcpConfig {
        entries,
        auto_enable_codemode,
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_exact_and_wildcard_tool_exposure() {
        let mut cfg = McpServerConfig::stdio("fff-mcp", vec![]);
        cfg.exposure = McpExposure::Direct;
        cfg.tool_exposure
            .insert("multi_grep".into(), McpExposure::Hidden);
        cfg.tool_exposure
            .insert("debug_*".into(), McpExposure::Deferred);

        assert_eq!(cfg.resolve_tool_exposure("grep"), McpExposure::Direct);
        assert_eq!(cfg.resolve_tool_exposure("multi_grep"), McpExposure::Hidden);
        assert_eq!(
            cfg.resolve_tool_exposure("debug_index"),
            McpExposure::Deferred
        );
    }

    #[test]
    fn merges_global_and_project_override_mcp_json() {
        let tmp = std::env::temp_dir().join(format!("mizar_mcp_cfg_{}", std::process::id()));
        let global_dir = tmp.join("global-mizar");
        let proj_dir = tmp.join("project");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&global_dir).unwrap();
        std::fs::create_dir_all(proj_dir.join(".mizar")).unwrap();

        std::fs::write(
            global_dir.join("mcp.json"),
            r#"{
                "mcpServers": {
                    "fff": {
                        "command": "fff-mcp",
                        "exposure": "codemode",
                        "toolExposure": { "multi_grep": "hidden" }
                    }
                }
            }"#,
        )
        .unwrap();
        std::fs::write(
            proj_dir.join(".mizar/mcp.json"),
            r#"{
                "mcpServers": {
                    "fff": {
                        "exposure": "direct"
                    }
                }
            }"#,
        )
        .unwrap();

        let loaded = load_mcp_config_with_global(&proj_dir, Some(&global_dir));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        let fff = loaded.entries.iter().find(|e| e.name == "fff").unwrap();
        assert_eq!(fff.config.command.as_deref(), Some("fff-mcp"));
        assert_eq!(fff.config.exposure, McpExposure::Direct);
        assert_eq!(
            fff.config.resolve_tool_exposure("multi_grep"),
            McpExposure::Hidden
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
