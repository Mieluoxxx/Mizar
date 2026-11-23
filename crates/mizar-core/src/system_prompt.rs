//! 系统提示词（对应 `NamiDSH/packages/system-prompt`）。
//!
//! 系统提示词不是写死的字符串，而是由各模块各自贡献「段落」按 `order` 拼起来的。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mizar_types::HookGuard;

/// 默认编码 Agent 人设（`order = 0`，纯英文静态前缀）。
pub const DEFAULT_PERSONA: &str = "\
You are Mizar, a coding agent working in the user's workspace. Respond in the same language as the user unless asked otherwise.

Verify changes with targeted checks or tests when available; inspect command exit codes and investigate failures before moving on. Never fabricate tool output, test results, or file contents, and avoid edits outside the task scope.

Engineering principles:
- Choose the simplest modular design for the task and make long-term structural decisions over quick hacks.
- Inspect existing code, workspace dependencies, and proven patterns before adding new dependencies or abstractions.
- Grow changes incrementally from a minimal working end-to-end slice.
- Remove obsolete code directly without leaving shims, while preserving task requirements and existing contracts.";

/// 默认运行时上下文标签协议说明（`order = 40`，纯英文静态前缀）。
pub const DEFAULT_RUNTIME_PROTOCOL: &str = "\
Runtime context blocks in conversation messages:
- `<environment_context>`: current workspace path and date.
- `<project_instructions>` (or `<system-reminder>`): workspace instruction files (e.g. AGENTS.md) and skill catalog updates; follow them alongside the user's request.
- `<skill_content>`: loaded skill instructions; follow them without re-calling `skill`.
- `<mizar-compaction>`: compacted summary of earlier turns; trust newer messages or current files on conflict and use `recall` for verbatim history.";

#[derive(Debug, Clone, Default)]
pub struct SystemPromptConfig {
    /// 部署方人设，放在最前面（`order = 0`）。
    pub persona: Option<String>,
    /// 可选的静态运行时上下文标签协议说明（`order = 40`）。
    pub runtime_protocol: Option<String>,
}

type RenderSectionFn = dyn Fn(Option<&str>) -> Option<String> + Send + Sync;

#[derive(Clone)]
pub struct PromptSection {
    pub id: String,
    /// 越小越靠前。
    pub order: i32,
    pub render: Arc<RenderSectionFn>,
}

impl PromptSection {
    pub fn new(
        id: impl Into<String>,
        order: i32,
        render: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self::new_for_session(id, order, move |_| render())
    }

    /// 注册可感知 `session_id` 的提示词段落（用于按会话隔离工具或增量段）。
    pub fn new_for_session(
        id: impl Into<String>,
        order: i32,
        render: impl Fn(Option<&str>) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            order,
            render: Arc::new(render),
        }
    }
}

pub struct SystemPrompt {
    sections: Mutex<HashMap<String, PromptSection>>,
}

impl SystemPrompt {
    /// 构造系统提示词服务，并预装人设（`order = 0`）与可选运行时上下文协议说明（`order = 40`）。
    pub fn new(config: SystemPromptConfig) -> Arc<Self> {
        let persona = config
            .persona
            .unwrap_or_else(|| DEFAULT_PERSONA.to_string());
        let runtime_protocol = config
            .runtime_protocol
            .filter(|text| !text.trim().is_empty());

        let mut sections = HashMap::new();
        sections.insert(
            "persona".to_string(),
            PromptSection::new("persona", 0, move || Some(persona.clone())),
        );
        if let Some(protocol) = runtime_protocol {
            sections.insert(
                "runtime_protocol".to_string(),
                PromptSection::new("runtime_protocol", 40, move || Some(protocol.clone())),
            );
        }

        Arc::new(Self {
            sections: Mutex::new(sections),
        })
    }

    /// 其他模块调用它贡献一段提示词，返回 `HookGuard` 在释放时自动移除对应段落。
    pub fn section(self: &Arc<Self>, section: PromptSection) -> HookGuard {
        let id = section.id.clone();
        self.sections.lock().unwrap().insert(id.clone(), section);
        let weak = Arc::downgrade(self);
        HookGuard::new(move || {
            if let Some(sp) = weak.upgrade() {
                sp.sections.lock().unwrap().remove(&id);
            }
        })
    }

    pub fn build(&self) -> String {
        self.build_for_session(None)
    }

    /// 按会话上下文拼装静态系统提示词（将 `session_id` 透传给需要会话隔离的段落）。
    pub fn build_for_session(&self, session_id: Option<&str>) -> String {
        let mut sections: Vec<PromptSection> =
            self.sections.lock().unwrap().values().cloned().collect();
        sections.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.id.cmp(&b.id)));
        sections
            .into_iter()
            .filter_map(|s| (s.render)(session_id))
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_sections_ordered_deterministically_without_dynamic_date() {
        let sp = SystemPrompt::new(SystemPromptConfig {
            persona: Some("测试人设".into()),
            ..Default::default()
        });

        let h_mid_b = sp.section(PromptSection::new("b_same_order", 50, || {
            Some("段落 B".into())
        }));
        let h_mid_a = sp.section(PromptSection::new("a_same_order", 50, || {
            Some("段落 A".into())
        }));

        let prompt = sp.build();
        assert!(!prompt.contains("今天是 "));
        let lines: Vec<&str> = prompt.split("\n\n").collect();
        assert_eq!(lines, vec!["测试人设", "段落 A", "段落 B"]);

        drop(h_mid_a);
        drop(h_mid_b);
        let prompt_after = sp.build();
        assert_eq!(prompt_after, "测试人设");
    }

    #[test]
    fn default_persona_and_runtime_protocol_are_english_and_session_aware() {
        let sp = SystemPrompt::new(SystemPromptConfig {
            persona: None,
            runtime_protocol: Some(DEFAULT_RUNTIME_PROTOCOL.to_string()),
        });

        let _h = sp.section(PromptSection::new_for_session("session_probe", 20, |sid| {
            sid.map(|id| format!("Session section: {id}"))
        }));

        let default_built = sp.build();
        assert!(default_built.contains("Respond in the same language as the user"));
        assert!(default_built.contains("Verify changes with targeted checks"));
        assert!(default_built.contains("Engineering principles:"));
        assert!(default_built.contains("<environment_context>"));
        assert!(default_built.contains("<mizar-compaction>"));
        assert!(!default_built.contains("Session section:"));
        assert!(
            !default_built
                .chars()
                .any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)),
            "default system prompt must be pure English: {default_built}"
        );

        let with_sid = sp.build_for_session(Some("s-123"));
        assert!(with_sid.contains("Session section: s-123"));
        let idx_persona = with_sid.find("Engineering principles:").unwrap();
        let idx_tools = with_sid.find("Session section: s-123").unwrap();
        let idx_proto = with_sid.find("Runtime context blocks").unwrap();
        assert!(idx_persona < idx_tools && idx_tools < idx_proto);
    }
}
