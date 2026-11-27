use std::any::Any;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

/// 随包与内嵌技能提供方的标准优先级（与 `deepseek-harness` 的 `BUNDLED_SKILL_RANK` 对齐）。
pub const BUNDLED_SKILL_RANK: u32 = 600;
pub const RUNTIME_SKILL_RANK: u32 = 250;

/// 校验技能名是否符合严格 kebab-case 命名规范：`^[a-z0-9]+(?:-[a-z0-9]+)*$`。
pub fn is_skill_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let bytes = name.as_bytes();
    let mut prev_hyphen = true; // 禁止以 '-' 开头
    for &b in bytes {
        let is_lower_alnum = b.is_ascii_lowercase() || b.is_ascii_digit();
        if is_lower_alnum {
            prev_hyphen = false;
        } else if b == b'-' {
            if prev_hyphen {
                return false;
            }
            prev_hyphen = true;
        } else {
            return false;
        }
    }
    !prev_hyphen
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SkillSource {
    ProjectAgents,
    Custom,
    UserAgents,
    Bundled,
    Runtime,
    Other(String),
}

impl SkillSource {
    pub fn as_str(&self) -> &str {
        match self {
            Self::ProjectAgents => "project-agents",
            Self::Custom => "custom",
            Self::UserAgents => "user-agents",
            Self::Bundled => "bundled",
            Self::Runtime => "runtime",
            Self::Other(s) => s.as_str(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SkillResourceBase {
    Directory { path: PathBuf },
    Url { url: String },
    Opaque { description: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillInvocationPolicy {
    pub model_invocable: bool,
    pub user_invocable: bool,
}

impl Default for SkillInvocationPolicy {
    fn default() -> Self {
        Self {
            model_invocable: true,
            user_invocable: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillSummary {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when_to_use: Option<String>,
    pub invocation: SkillInvocationPolicy,
    pub source: SkillSource,
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_base: Option<SkillResourceBase>,
}

impl SkillSummary {
    pub fn is_model_invocable(&self) -> bool {
        self.invocation.model_invocable
    }

    pub fn is_user_invocable(&self) -> bool {
        self.invocation.user_invocable
    }
}

#[derive(Clone)]
pub struct SkillCandidate {
    pub summary: SkillSummary,
    pub rank: u32,
    pub locator: Arc<dyn Any + Send + Sync>,
    pub metadata: Option<BTreeMap<String, Value>>,
}

#[derive(Debug, Clone)]
pub struct SkillDefinition {
    pub summary: SkillSummary,
    pub content: String,
    pub metadata: Option<BTreeMap<String, Value>>,
}

#[derive(Debug, Clone)]
pub struct SkillRegistration {
    pub name: String,
    pub description: String,
    pub when_to_use: Option<String>,
    pub invocation: Option<SkillInvocationPolicy>,
    pub source: SkillSource,
    pub path: Option<PathBuf>,
    pub resource_base: Option<SkillResourceBase>,
    pub content: String,
    pub metadata: Option<BTreeMap<String, Value>>,
}

#[derive(Debug, Clone, Default)]
pub struct SkillLookupOptions {
    pub cwd: Option<PathBuf>,
    pub cancel: CancellationToken,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCatalogSnapshot {
    pub skills: Vec<SkillSummary>,
    pub complete: bool,
}

pub struct SkillProviderObservation {
    pub candidates: Vec<SkillCandidate>,
    pub complete: bool,
}

pub fn escape_xml_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
}

pub fn escape_xml_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn render_resource_hint(summary: &SkillSummary) -> Vec<String> {
    match &summary.resource_base {
        None => vec![
            format!(
                "Resources for this skill are managed by provider \"{}\".",
                escape_xml_text(&summary.provider)
            ),
            "Load referenced resources only as needed.".to_string(),
        ],
        Some(SkillResourceBase::Directory { path }) => vec![
            format!(
                "Base directory for this skill: {}",
                escape_xml_text(&path.display().to_string())
            ),
            "Resolve relative paths mentioned by this skill against the base directory before using them. Load referenced resources only as needed.".to_string(),
        ],
        Some(SkillResourceBase::Url { url }) => vec![
            format!("Base URL for this skill: {}", escape_xml_text(url)),
            "Resolve relative URLs mentioned by this skill against the base URL before using them. Load referenced resources only as needed.".to_string(),
        ],
        Some(SkillResourceBase::Opaque { description }) => vec![
            format!(
                "Resources for this skill: {}",
                escape_xml_text(description)
            ),
            "Load referenced resources only as needed.".to_string(),
        ],
    }
}

/// 统一渲染用户 `/<name>` 手势加载后的 `<skill_content>` 块。
pub fn render_skill_content(def: &SkillDefinition) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "<skill_content name=\"{}\">",
        escape_xml_attr(&def.summary.name)
    ));
    lines.push("<skill_resources>".to_string());
    lines.extend(render_resource_hint(&def.summary));
    lines.push("</skill_resources>".to_string());
    lines.push(String::new());
    lines.push("<skill_instructions>".to_string());
    lines.push(def.content.clone());
    lines.push("</skill_instructions>".to_string());
    lines.push("</skill_content>".to_string());
    lines.join("\n")
}
