use std::collections::HashMap;

use crate::types::{
    ContentBlock, ContentBlockKind, FinishReason, Message, ReplayEnvelope, StreamChunk, TokenUsage,
    ToolCall,
};

struct PartialBlock {
    kind: ContentBlockKind,
    text: String,
    tool_call_id: Option<String>,
    tool_call_name: Option<String>,
    tool_call_arguments: String,
    closed: Option<ContentBlock>,
}

/// 增量流块组装器：把乱序/分片的 `StreamChunk` 组装为有序的 `ContentBlock` 列表与回放状态。
#[derive(Default)]
pub struct BlockAssembler {
    partials: HashMap<usize, PartialBlock>,
    order: Vec<usize>,
    usage: Option<TokenUsage>,
    finish: Option<FinishReason>,
    replay: Option<ReplayEnvelope>,
}

impl BlockAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: StreamChunk) {
        match chunk {
            StreamChunk::BlockStart { index, kind } => {
                self.ensure(index, kind);
            }
            StreamChunk::TextDelta { index, text } => {
                let p = self.ensure(index, ContentBlockKind::Text);
                if p.closed.is_none() {
                    p.text.push_str(&text);
                }
            }
            StreamChunk::ReasoningDelta { index, text } => {
                let p = self.ensure(index, ContentBlockKind::Reasoning);
                if p.closed.is_none() {
                    p.text.push_str(&text);
                }
            }
            StreamChunk::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                let p = self.ensure(index, ContentBlockKind::ToolCall);
                if p.closed.is_none() {
                    if let Some(id) = id
                        && !id.is_empty()
                    {
                        p.tool_call_id = Some(id);
                    }
                    if let Some(name) = name
                        && !name.is_empty()
                    {
                        p.tool_call_name = Some(name);
                    }
                    p.tool_call_arguments.push_str(&arguments_delta);
                }
            }
            StreamChunk::BlockEnd { index, block } => {
                let p = self.ensure(index, block.kind());
                if p.closed.is_none() {
                    p.closed = Some(block);
                }
            }
            StreamChunk::Usage { usage } => {
                self.usage = Some(usage);
            }
            StreamChunk::Finish { reason, replay } => {
                self.finish = Some(reason);
                self.replay = replay;
            }
        }
    }

    fn ensure(&mut self, index: usize, kind: ContentBlockKind) -> &mut PartialBlock {
        if !self.partials.contains_key(&index) {
            self.order.push(index);
            self.partials.insert(
                index,
                PartialBlock {
                    kind,
                    text: String::new(),
                    tool_call_id: None,
                    tool_call_name: None,
                    tool_call_arguments: String::new(),
                    closed: None,
                },
            );
        }
        self.partials.get_mut(&index).expect("partial inserted")
    }

    fn finalize_partial(index: usize, partial: &PartialBlock) -> ContentBlock {
        if let Some(closed) = &partial.closed {
            return closed.clone();
        }
        match partial.kind {
            ContentBlockKind::Text => ContentBlock::Text {
                text: partial.text.clone(),
            },
            ContentBlockKind::Reasoning => ContentBlock::Reasoning {
                text: partial.text.clone(),
            },
            ContentBlockKind::ToolCall => ContentBlock::ToolCall {
                id: partial
                    .tool_call_id
                    .clone()
                    .unwrap_or_else(|| format!("call_{index}")),
                name: partial.tool_call_name.clone().unwrap_or_default(),
                arguments: partial.tool_call_arguments.clone(),
            },
        }
    }

    fn assembled(&self) -> (Vec<ContentBlock>, Option<ReplayEnvelope>) {
        let all: Vec<ContentBlock> = self
            .order
            .iter()
            .filter_map(|idx| {
                self.partials
                    .get(idx)
                    .map(|p| Self::finalize_partial(*idx, p))
            })
            .collect();

        // 若因达到 max_tokens 截断，丢弃可能不完整的 tool_call，并同步裁剪 replay.blocks。
        let drop_tool_calls = matches!(self.finish, Some(FinishReason::MaxTokens));
        let keep_mask: Vec<bool> = all
            .iter()
            .map(|b| !drop_tool_calls || !matches!(b, ContentBlock::ToolCall { .. }))
            .collect();

        let blocks: Vec<ContentBlock> = all
            .iter()
            .zip(keep_mask.iter())
            .filter(|&(_, keep)| *keep)
            .map(|(b, _)| b.clone())
            .collect();

        let replay = self.replay.as_ref().and_then(|env| {
            if env.blocks.is_empty() {
                return Some(env.clone());
            }
            if env.blocks.len() != all.len() {
                return None;
            }
            let filtered_blocks: Vec<serde_json::Value> = env
                .blocks
                .iter()
                .zip(keep_mask.iter())
                .filter(|&(_, keep)| *keep)
                .map(|(v, _)| v.clone())
                .collect();
            Some(ReplayEnvelope {
                protocol: env.protocol.clone(),
                provider: env.provider.clone(),
                model: env.model.clone(),
                response: env.response.clone(),
                blocks: filtered_blocks,
            })
        });

        (blocks, replay)
    }

    pub fn blocks(&self) -> Vec<ContentBlock> {
        self.assembled().0
    }

    /// 中断时仅保留已有非空白文本与思考内容，丢弃尚未派发的工具调用。
    pub fn interrupted_blocks(&self) -> Vec<ContentBlock> {
        self.order
            .iter()
            .filter_map(|idx| {
                let p = self.partials.get(idx)?;
                let kind = p.closed.as_ref().map(ContentBlock::kind).unwrap_or(p.kind);
                if kind == ContentBlockKind::ToolCall {
                    return None;
                }
                let block = Self::finalize_partial(*idx, p);
                let non_empty = match &block {
                    ContentBlock::Text { text } | ContentBlock::Reasoning { text } => {
                        !text.trim().is_empty()
                    }
                    ContentBlock::ToolCall { .. } => false,
                };
                non_empty.then_some(block)
            })
            .collect()
    }

    pub fn text(&self) -> String {
        self.blocks()
            .iter()
            .filter_map(ContentBlock::as_text)
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn reasoning(&self) -> String {
        self.blocks()
            .iter()
            .filter_map(ContentBlock::as_reasoning)
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.blocks()
            .iter()
            .filter_map(ContentBlock::to_tool_call)
            .collect()
    }

    pub fn usage(&self) -> Option<TokenUsage> {
        self.usage
    }

    pub fn finish(&self) -> FinishReason {
        self.finish.clone().unwrap_or(FinishReason::Stop)
    }

    pub fn replay(&self) -> Option<ReplayEnvelope> {
        self.assembled().1
    }

    pub fn into_assistant_message(self) -> Message {
        let (blocks, replay) = self.assembled();
        Message::Assistant { blocks, replay }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_interleaved_blocks_and_prunes_truncated_tool_calls() {
        let mut asm = BlockAssembler::new();
        asm.push(StreamChunk::BlockStart {
            index: 0,
            kind: ContentBlockKind::Reasoning,
        });
        asm.push(StreamChunk::ReasoningDelta {
            index: 0,
            text: "先思考".into(),
        });
        asm.push(StreamChunk::TextDelta {
            index: 1,
            text: "答案".into(),
        });
        asm.push(StreamChunk::ToolCallDelta {
            index: 2,
            id: Some("call_1".into()),
            name: Some("read_file".into()),
            arguments_delta: r#"{"path":"#.into(),
        });
        asm.push(StreamChunk::Finish {
            reason: FinishReason::MaxTokens,
            replay: Some(ReplayEnvelope {
                protocol: "anthropic-messages".into(),
                provider: "anthropic".into(),
                model: "claude".into(),
                response: serde_json::json!({ "id": "msg_1" }),
                blocks: vec![
                    serde_json::json!({ "signature": "sig_0" }),
                    serde_json::json!({}),
                    serde_json::json!({}),
                ],
            }),
        });

        let blocks = asm.blocks();
        assert_eq!(
            blocks,
            vec![
                ContentBlock::Reasoning {
                    text: "先思考".into()
                },
                ContentBlock::Text {
                    text: "答案".into()
                }
            ]
        );
        let replay = asm.replay().unwrap();
        assert_eq!(replay.blocks.len(), 2);
        assert!(asm.tool_calls().is_empty());
    }
}
