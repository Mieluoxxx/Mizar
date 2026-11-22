//! 离线假模型（对应 `NamiDSH/packages/llm-mock`）。
//!
//! 不需要 API Key，用几条规则模拟「决定调工具 → 拿到结果 → 回答」。

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use futures::channel::mpsc;
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmError, Message,
    ModelInfo, ProviderInfo, StreamChunk, ToolCall,
};

pub struct MockAdapter;

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn contains_any(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| text.contains(p))
}

impl LlmAdapter for MockAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: "mock".to_string(),
            name: "Offline Mock".to_string(),
            protocol: "mock".to_string(),
            models: vec![ModelInfo {
                id: "mock-v1".to_string(),
                name: "Mock V1".to_string(),
                context_window: Some(32_768),
                max_tokens: Some(4096),
                reasoning_efforts: Vec::new(),
                input: Vec::new(),
                ..Default::default()
            }],
        })
    }

    fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
        let (tx, rx) = mpsc::unbounded();
        tokio::spawn(async move {
            let has = |name: &str| options.tools.iter().any(|t| t.name == name);
            let reply = match options.messages.last() {
                Some(Message::Tool { content, .. }) => format!(
                    "工具返回了结果：\n\n{content}\n\n（这是离线假模型的回答。设置环境变量 API_KEY 后会改用真实模型。）"
                ),
                Some(msg) => {
                    let image_count: usize = options
                        .messages
                        .iter()
                        .rev()
                        .take_while(|m| matches!(m, Message::User { .. }))
                        .map(|m| m.images().len())
                        .sum();
                    let content = options
                        .messages
                        .iter()
                        .rev()
                        .find_map(|m| match m {
                            Message::User { content, .. }
                                if !content.trim_start().starts_with("<system-reminder>")
                                    && !content.trim_start().starts_with("<skill_content")
                                    && !content.trim_start().starts_with("<attached-") =>
                            {
                                Some(content.clone())
                            }
                            _ => None,
                        })
                        .unwrap_or_else(|| msg.text_content());
                    if contains_any(&content, &["时间", "几点", "日期"]) && has("get_time") {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: format!("call_{}", now_millis()),
                                name: "get_time".to_string(),
                                arguments: "{}".to_string(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                        return;
                    }
                    if contains_any(&content, &["文件", "目录", "项目"]) && has("list_dir") {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: format!("call_{}", now_millis()),
                                name: "list_dir".to_string(),
                                arguments: r#"{"path":"."}"#.to_string(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                        return;
                    }
                    if contains_any(&content, &["文件", "目录", "项目"]) && has("find") {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: format!("call_{}", now_millis()),
                                name: "find".to_string(),
                                arguments: r#"{"pattern":"*"}"#.to_string(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                        return;
                    }
                    if contains_any(&content, &["搜索", "查找", "grep"]) && has("grep") {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: format!("call_{}", now_millis()),
                                name: "grep".to_string(),
                                arguments: r#"{"pattern":"Cargo","literal":true}"#.to_string(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                        return;
                    }
                    if image_count > 0 {
                        format!(
                            "已收到 {image_count} 张图片（当前为离线假模型，无法真实识图；设置环境变量 API_KEY 后会改用真实模型）。"
                        )
                    } else if has("find") || has("grep") || has("read") {
                        "你好，我是离线假模型。当前环境已挂载编码工具，可以试试：\n- 问我「项目里有哪些文件」（触发 `find`）\n- 问我「搜索 Cargo」（触发 `grep`）\n\n设置环境变量 API_KEY 后，就会换成真正的模型。".to_string()
                    } else {
                        "你好，我是离线假模型，只会两件事：\n- 问我「现在几点」，我会调用 get_time\n- 问我「项目里有哪些文件」，我会调用 list_dir\n\n设置环境变量 API_KEY 后，就会换成真正的模型。".to_string()
                    }
                }
                None => String::new(),
            };

            // 模拟逐字流式输出
            for ch in reply.chars() {
                if options.cancel.is_cancelled() {
                    let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                    return;
                }
                if tx
                    .unbounded_send(Ok(StreamChunk::text(ch.to_string())))
                    .is_err()
                {
                    return;
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_millis(8)) => {}
                    _ = options.cancel.cancelled() => {
                        let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                        return;
                    }
                }
            }
            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay: None,
            }));
        });
        Box::pin(rx)
    }
}

#[derive(Default, Clone, Copy)]
pub struct MockPlugin;

impl Plugin for MockPlugin {
    type Config = ();

    fn name(&self) -> Option<&str> {
        Some("llm-mock")
    }

    fn inject(&self) -> Inject {
        Inject::from(["llm"])
    }

    fn apply(
        &self,
        ctx: Context,
        _config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let llm = ctx.llm()?;
            ctx.effect("llm-mock.registerAdapter", || {
                Ok::<_, mizar_llm::LlmError>(Some(
                    llm.register_adapter("mock", Arc::new(MockAdapter))?,
                ))
            })?;
            Ok(None)
        })
    }
}
