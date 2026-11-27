//! 系统提示词（对应 `NamiDSH/packages/system-prompt`）。
//!
//! 系统提示词不是写死的字符串，而是由多个插件各自贡献「段落」按 `order` 拼起来的。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cordis::{BoxFuture, Cleanup, Context, CordisError, Service};

/// 默认身份定位与语言对齐（`order = 0`，纯英文静态前缀）。
pub const DEFAULT_IDENTITY: &str = "\
You are a coding assistant running inside Mizar, a coding agent harness. Help users inspect files, run commands, and modify code. Respond in the same language as the user unless asked otherwise.";

/// 默认运行时上下文标签协议说明（`order = 40`，纯英文静态前缀）。
pub const DEFAULT_RUNTIME_PROTOCOL: &str = "\
`<environment_context>` gives the workspace and date. Follow workspace instructions in `<project_instructions>`, updates in `<system-reminder>`, and loaded instructions in `<skill_content>`.
`<mizar-compaction>` summarizes earlier turns. Prefer newer messages or current files on conflict.";

#[derive(Debug, Clone, Default)]
pub struct SystemPromptConfig {
    /// 身份定位（`order = 0`）；`None` 使用 `DEFAULT_IDENTITY`，`Some("")` 关闭。
    pub identity: Option<String>,
    pub personality: Option<String>,
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
    identity: Option<String>,
    personality: Option<String>,
    runtime_protocol: Option<String>,
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

impl Service for SystemPrompt {
    type Config = SystemPromptConfig;

    fn service_name() -> &'static str {
        "systemPrompt"
    }

    fn plugin_name() -> &'static str {
        "SystemPrompt"
    }

    fn create(_ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        let identity = config
            .identity
            .or_else(|| Some(DEFAULT_IDENTITY.to_owned()))
            .filter(|text| !text.trim().is_empty());
        let personality = config.personality.filter(|text| !text.trim().is_empty());
        let runtime_protocol = config
            .runtime_protocol
            .filter(|text| !text.trim().is_empty());
        Ok(Arc::new(Self {
            identity,
            personality,
            runtime_protocol,
            sections: Mutex::new(HashMap::new()),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let identity = self.identity.clone();
        let personality = self.personality.clone();
        let runtime_protocol = self.runtime_protocol.clone();
        let sp = Arc::clone(&self);
        let ctx = ctx.clone();
        Box::pin(async move {
            if let Some(identity) = identity {
                let sp_identity = Arc::clone(&sp);
                ctx.effect("systemPrompt.identity", move || {
                    Ok::<_, CordisError>(Some(sp_identity.section(PromptSection::new(
                        "identity",
                        0,
                        move || Some(identity.clone()),
                    ))))
                })?;
            }
            if let Some(personality) = personality {
                let sp_personality = Arc::clone(&sp);
                ctx.effect("systemPrompt.personality", move || {
                    Ok::<_, CordisError>(Some(sp_personality.section(PromptSection::new(
                        "personality",
                        20,
                        move || Some(personality.clone()),
                    ))))
                })?;
            }
            if let Some(protocol) = runtime_protocol {
                ctx.effect("systemPrompt.runtimeProtocol", move || {
                    Ok::<_, CordisError>(Some(sp.section(PromptSection::new(
                        "runtime_protocol",
                        40,
                        move || Some(protocol.clone()),
                    ))))
                })?;
            }
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
