//! 系统提示词（对应 `NamiDSH/packages/system-prompt`）。
//!
//! 系统提示词不是写死的字符串，而是由多个插件各自贡献「段落」按 `order` 拼起来的。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::Local;
use cordis::{BoxFuture, Cleanup, Context, CordisError, Service};

#[derive(Debug, Clone, Default)]
pub struct SystemPromptConfig {
    /// 部署方人设，放在最前面。
    pub persona: Option<String>,
}

#[derive(Clone)]
pub struct PromptSection {
    pub id: String,
    /// 越小越靠前。
    pub order: i32,
    pub render: Arc<dyn Fn() -> Option<String> + Send + Sync>,
}

impl PromptSection {
    pub fn new(
        id: impl Into<String>,
        order: i32,
        render: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            order,
            render: Arc::new(render),
        }
    }
}

pub struct SystemPrompt {
    persona: String,
    sections: Mutex<HashMap<String, PromptSection>>,
}

impl SystemPrompt {
    /// 其他插件调用它贡献一段提示词，返回 `Cleanup` 放入 `ctx.effect` 即可随插件卸载移除。
    pub fn section(self: &Arc<Self>, section: PromptSection) -> Cleanup {
        let id = section.id.clone();
        self.sections.lock().unwrap().insert(id.clone(), section);
        let weak = Arc::downgrade(self);
        Cleanup::sync(move || {
            if let Some(sp) = weak.upgrade() {
                sp.sections.lock().unwrap().remove(&id);
            }
        })
    }

    pub fn build(&self) -> String {
        let mut sections: Vec<PromptSection> =
            self.sections.lock().unwrap().values().cloned().collect();
        sections.sort_by_key(|s| s.order);
        sections
            .into_iter()
            .filter_map(|s| (s.render)())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl Service for SystemPrompt {
    type Config = SystemPromptConfig;

    fn service_name() -> &'static str {
        "systemPrompt"
    }

    fn plugin_name() -> &'static str {
        "SystemPrompt"
    }

    fn create(_ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        let persona = config.persona.unwrap_or_else(|| {
            "你是一个乐于助人的 AI 助手，回答简洁准确。需要时调用工具获取信息。".to_string()
        });
        Ok(Arc::new(Self {
            persona,
            sections: Mutex::new(HashMap::new()),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let persona = self.persona.clone();
        let sp1 = Arc::clone(&self);
        let sp2 = Arc::clone(&self);
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.effect("systemPrompt.persona", move || {
                Ok::<_, CordisError>(Some(sp1.section(PromptSection::new(
                    "persona",
                    0,
                    move || Some(persona.clone()),
                ))))
            })?;
            ctx.effect("systemPrompt.time", move || {
                Ok::<_, CordisError>(Some(sp2.section(PromptSection::new("time", 100, || {
                    Some(format!("今天是 {}。", Local::now().format("%Y/%-m/%-d")))
                }))))
            })?;
            Ok(())
        })
    }
}

pub trait SystemPromptContextExt {
    fn system_prompt(&self) -> Result<Arc<SystemPrompt>, CordisError>;
}

impl SystemPromptContextExt for Context {
    fn system_prompt(&self) -> Result<Arc<SystemPrompt>, CordisError> {
        self.service::<SystemPrompt>("systemPrompt")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn builds_sections_ordered_and_cleans_up() {
        let app = Context::new();
        app.plugin(
            SystemPrompt::plugin(),
            SystemPromptConfig {
                persona: Some("测试人设".into()),
            },
        )
        .await
        .unwrap();

        let sp = app.system_prompt().unwrap();
        let handle = app
            .effect("extra.section", || {
                Ok::<_, CordisError>(Some(
                    sp.section(PromptSection::new("mid", 50, || Some("中间段落".into()))),
                ))
            })
            .unwrap();

        let prompt = sp.build();
        let lines: Vec<&str> = prompt.split("\n\n").collect();
        assert_eq!(lines[0], "测试人设");
        assert_eq!(lines[1], "中间段落");
        assert!(lines[2].starts_with("今天是 "));

        handle.dispose().await;
        let prompt_after = sp.build();
        assert!(!prompt_after.contains("中间段落"));
    }
}
