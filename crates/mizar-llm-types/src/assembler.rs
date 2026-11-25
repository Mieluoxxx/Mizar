use std::collections::HashMap;

use crate::types::{
    ContentBlock, ContentBlockKind, FinishReason, Message, ReplayEnvelope, StreamChunk, TokenUsage,
    ToolCall,
};

enum OpenBlockBuffer {
    Text(String),
    Reasoning(String),
    ToolCall {
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
}

impl OpenBlockBuffer {
    fn from_kind(kind: ContentBlockKind) -> Self {
        match kind {
            ContentBlockKind::Text => Self::Text(String::new()),
            ContentBlockKind::Reasoning => Self::Reasoning(String::new()),
            ContentBlockKind::ToolCall => Self::ToolCall {
                id: None,
                name: None,
                arguments: String::new(),
            },
        }
    }

    fn kind(&self) -> ContentBlockKind {
        match self {
            Self::Text(_) => ContentBlockKind::Text,
            Self::Reasoning(_) => ContentBlockKind::Reasoning,
            Self::ToolCall { .. } => ContentBlockKind::ToolCall,
        }
    }
}

enum PartialBlock {
    Open(OpenBlockBuffer),
    Closed(ContentBlock),
}

impl PartialBlock {
    fn kind(&self) -> ContentBlockKind {
        match self {
            Self::Open(buf) => buf.kind(),
            Self::Closed(block) => block.kind(),
        }
    }
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
                if let PartialBlock::Open(OpenBlockBuffer::Text(buf)) =
                    self.ensure(index, ContentBlockKind::Text)
                {
                    buf.push_str(&text);
                }
            }
            StreamChunk::ReasoningDelta { index, text } => {
                if let PartialBlock::Open(OpenBlockBuffer::Reasoning(buf)) =
                    self.ensure(index, ContentBlockKind::Reasoning)
                {
                    buf.push_str(&text);
                }
            }
            StreamChunk::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                if let PartialBlock::Open(OpenBlockBuffer::ToolCall {
                    id: cur_id,
                    name: cur_name,
                    arguments,
                }) = self.ensure(index, ContentBlockKind::ToolCall)
                {
                    if let Some(id) = id
                        && !id.is_empty()
                    {
                        *cur_id = Some(id);
                    }
                    if let Some(name) = name
                        && !name.is_empty()
                    {
                        *cur_name = Some(name);
                    }
                    arguments.push_str(&arguments_delta);
                }
            }
            StreamChunk::BlockEnd { index, block } => {
                let entry = self.ensure(index, block.kind());
                if matches!(entry, PartialBlock::Open(_)) {
                    *entry = PartialBlock::Closed(block);
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
            self.partials
                .insert(index, PartialBlock::Open(OpenBlockBuffer::from_kind(kind)));
        }
        self.partials.get_mut(&index).expect("partial inserted")
    }

    fn finalize_partial(index: usize, partial: &PartialBlock) -> ContentBlock {
        match partial {
            PartialBlock::Closed(closed) => closed.clone(),
            PartialBlock::Open(OpenBlockBuffer::Text(text)) => {
                ContentBlock::Text { text: text.clone() }
            }
            PartialBlock::Open(OpenBlockBuffer::Reasoning(text)) => {
                ContentBlock::Reasoning { text: text.clone() }
            }
            PartialBlock::Open(OpenBlockBuffer::ToolCall {
                id,
                name,
                arguments,
            }) => ContentBlock::ToolCall {
                id: id.clone().unwrap_or_else(|| format!("call_{index}")),
                name: name.clone().unwrap_or_default(),
                arguments: arguments.clone(),
            },
        }
    }

    fn sorted_indices(&self) -> Vec<usize> {
        let mut indices = self.order.clone();
        indices.sort_unstable();
        indices
    }

    fn assembled(&self) -> (Vec<ContentBlock>, Option<ReplayEnvelope>) {
        let sorted_indices = self.sorted_indices();
        let all: Vec<(usize, ContentBlock)> = sorted_indices
            .iter()
            .filter_map(|&idx| {
                self.partials
                    .get(&idx)
                    .map(|p| (idx, Self::finalize_partial(idx, p)))
            })
            .collect();

        // 若因达到 max_tokens 截断，丢弃可能不完整的 tool_call，并同步裁剪 replay.blocks。
        let drop_tool_calls = matches!(self.finish, Some(FinishReason::MaxTokens));
        let keep_mask: Vec<bool> = all
            .iter()
            .map(|(_, b)| !drop_tool_calls || !matches!(b, ContentBlock::ToolCall { .. }))
            .collect();

        let blocks: Vec<ContentBlock> = all
            .iter()
            .zip(keep_mask.iter())
            .filter(|&(_, keep)| *keep)
            .map(|((_, b), _)| b.clone())
            .collect();

        let replay = self.replay.as_ref().and_then(|env| {
            if env.blocks.is_empty() {
                return Some(env.clone());
            }
            let filtered_blocks = if env.blocks.len() == all.len() {
                env.blocks
                    .iter()
                    .zip(keep_mask.iter())
                    .filter(|&(_, keep)| *keep)
                    .map(|(v, _)| v.clone())
                    .collect()
            } else if all
                .last()
                .is_some_and(|(max_idx, _)| *max_idx < env.blocks.len())
            {
                all.iter()
                    .zip(keep_mask.iter())
                    .filter(|&(_, keep)| *keep)
                    .map(|((idx, _), _)| env.blocks[*idx].clone())
                    .collect()
            } else {
                return None;
            };
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
        self.sorted_indices()
            .into_iter()
            .filter_map(|idx| {
                let p = self.partials.get(&idx)?;
                if p.kind() == ContentBlockKind::ToolCall {
                    return None;
                }
                let block = Self::finalize_partial(idx, p);
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
