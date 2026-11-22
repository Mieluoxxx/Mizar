//! 提交时 `@` 文件/目录/图片引用展开、`agent/pre-step` 钩子与 `FileMentionsPlugin`。
//!
//! 保留用户输入原文在 `SessionEvent::User` 中（气泡只渲染用户所写），
//! 同时把解析得到的 `<attached-file ...>`、`<attached-directory ...>` 与图片附件
//! 以 `SessionEvent::ContextNotice { source: "file-mentions", .. }` 注入本轮模型上下文。

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cordis::{
    BoxError, BoxFuture, Cleanup, Context, CordisError, DisposableHandle, Inject, Next, Plugin,
};
use mizar_agent_loop::{AgentPreStepDecision, AgentPreStepInput, EVENT_AGENT_PRE_STEP, TurnInput};
use mizar_client_core::{
    ClientEvent, FILE_MENTIONS_SOURCE, NoticeLevel, PLUGIN_FILE_MENTIONS, extract_mentions,
};
use mizar_llm::{ImageAttachment, LlmContextExt, MAX_IMAGES_PER_MESSAGE};
use mizar_tools_coding::path_utils::resolve_read_path;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// 单文件最大附着字符数（防止误 `@` 大文件挤爆上下文）。
pub const MENTION_MAX_FILE_CHARS: usize = 50_000;
/// 单条消息所有 `@` 附件共享的总字符预算。
pub const MENTION_MAX_TOTAL_CHARS: usize = 200_000;
/// 目录列举最大条目数。
pub const MENTION_MAX_DIR_ENTRIES: usize = 200;
/// 单文本文件读取上限（16 MiB），避免超大二进制文件引发内存膨胀。
const MENTION_MAX_READ_BYTES: u64 = 16 * 1024 * 1024;
/// 单张待解码 `@` 图片文件读取上限（32 MiB）。
const MENTION_MAX_IMAGE_READ_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MentionExpansionResult {
    pub blocks: Vec<String>,
    pub images: Vec<ImageAttachment>,
    pub image_paths: Vec<String>,
    pub attached: Vec<String>,
    pub missing: Vec<String>,
    pub errors: Vec<String>,
}

/// `FileMentionsPlugin` 插件配置：绑定工作区边界与客户端事件投递通道。
#[derive(Clone)]
pub struct FileMentionsConfig {
    pub workspace: PathBuf,
    pub restrict_to_workspace: bool,
    pub emit: Arc<dyn Fn(ClientEvent) + Send + Sync>,
}

impl fmt::Debug for FileMentionsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileMentionsConfig")
            .field("workspace", &self.workspace)
            .field("restrict_to_workspace", &self.restrict_to_workspace)
            .finish_non_exhaustive()
    }
}

impl FileMentionsConfig {
    pub fn new(
        workspace: impl Into<PathBuf>,
        restrict_to_workspace: bool,
        tx: mpsc::UnboundedSender<ClientEvent>,
    ) -> Self {
        Self::with_callback(workspace, restrict_to_workspace, move |ev| {
            let _ = tx.send(ev);
        })
    }

    pub fn with_callback<F>(
        workspace: impl Into<PathBuf>,
        restrict_to_workspace: bool,
        emit: F,
    ) -> Self
    where
        F: Fn(ClientEvent) + Send + Sync + 'static,
    {
        Self {
            workspace: workspace.into(),
            restrict_to_workspace,
            emit: Arc::new(emit),
        }
    }
}

/// 官方客户端文件/目录/图片 `@` 引用展开 Cordis 插件（依赖 `llm` 探测视觉能力）。
#[derive(Default, Clone, Copy)]
pub struct FileMentionsPlugin;

impl Plugin for FileMentionsPlugin {
    type Config = FileMentionsConfig;

    fn name(&self) -> Option<&str> {
        Some(PLUGIN_FILE_MENTIONS)
    }

    fn inject(&self) -> Inject {
        Inject::from(["llm"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let _ = ctx.llm()?;
            let c = ctx.clone();
            ctx.effect("mizar-client-file-mentions", move || {
                let emit = Arc::clone(&config.emit);
                let handle = register_mention_hook_with(
                    &c,
                    config.workspace,
                    config.restrict_to_workspace,
                    move |ev| emit(ev),
                )?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    handle.dispose().await;
                })))
            })?;
            Ok(None)
        })
    }
}

fn escape_xml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn truncate_chars(text: &str, max_chars: usize) -> (&str, bool) {
    if let Some((byte_idx, _)) = text.char_indices().nth(max_chars) {
        (&text[..byte_idx], true)
    } else {
        (text, false)
    }
}

async fn try_resolve_mention(
    workspace: &Path,
    raw: &str,
    restrict_to_workspace: bool,
) -> Option<(PathBuf, std::fs::Metadata)> {
    let path = resolve_read_path(workspace, raw, restrict_to_workspace)
        .await
        .ok()?;
    let meta = tokio::fs::metadata(&path).await.ok()?;
    Some((path, meta))
}

/// 快速探测文本中的 `@` 引用是否包含受支持的魔数图片文件（用于在不支持视觉的模型上预检拦截）。
pub async fn mentions_resolve_any_image(
    workspace: &Path,
    restrict_to_workspace: bool,
    text: &str,
) -> bool {
    for mention in extract_mentions(text) {
        let mut resolved =
            try_resolve_mention(workspace, &mention.path, restrict_to_workspace).await;
        if resolved.is_none()
            && let Some(lit) = &mention.literal
        {
            resolved = try_resolve_mention(workspace, lit, restrict_to_workspace).await;
        }
        let Some((target, meta)) = resolved else {
            continue;
        };
        if meta.is_file() {
            let Ok(mut file) = tokio::fs::File::open(&target).await else {
                continue;
            };
            let mut head = [0u8; 16];
            if let Ok(n) = file.read(&mut head).await
                && crate::images::sniff_image_format(&head[..n]).is_some()
            {
                return true;
            }
        }
    }
    false
}

/// 解析用户文本中的所有 `@` 引用，按出现顺序生成 `<attached-file>` / `<attached-directory>` 块及图片附件。
pub async fn expand_text_mentions(
    workspace: &Path,
    restrict_to_workspace: bool,
    text: &str,
    cancel: &CancellationToken,
) -> MentionExpansionResult {
    let mentions = extract_mentions(text);
    let mut out = MentionExpansionResult::default();
    if mentions.is_empty() {
        return out;
    }

    let mut budget = MENTION_MAX_TOTAL_CHARS;

    for mention in mentions {
        if cancel.is_cancelled() || budget == 0 {
            break;
        }
        let display = mention
            .literal
            .as_deref()
            .unwrap_or(&mention.path)
            .to_string();

        // strip-first + literal-fallback（兼容文件名本身含 `#L...` 的文件）
        let mut resolved =
            try_resolve_mention(workspace, &mention.path, restrict_to_workspace).await;
        let mut literal_fallback = false;
        if resolved.is_none()
            && let Some(lit) = &mention.literal
        {
            resolved = try_resolve_mention(workspace, lit, restrict_to_workspace).await;
            literal_fallback = resolved.is_some();
        }

        let Some((target, meta)) = resolved else {
            out.missing.push(display);
            continue;
        };

        let shown_path = if literal_fallback {
            display.clone()
        } else if mention.path.starts_with('~') {
            target.display().to_string()
        } else {
            mention.path.clone()
        };
        let shown_attr = escape_xml_attr(&shown_path);

        if meta.is_file() {
            if meta.len() > MENTION_MAX_IMAGE_READ_BYTES {
                // 先读 16 字节区分是超大图片还是超大普通文件
                let is_img_header = match tokio::fs::File::open(&target).await {
                    Ok(mut f) => {
                        let mut head = [0u8; 16];
                        matches!(f.read(&mut head).await, Ok(n) if crate::images::sniff_image_format(&head[..n]).is_some())
                    }
                    Err(_) => false,
                };
                if is_img_header {
                    out.errors
                        .push(format!("{display}（图片文件超过 32 MiB 读取上限）"));
                } else {
                    out.missing.push(display);
                }
                continue;
            }

            let Ok(bytes) = tokio::fs::read(&target).await else {
                out.missing.push(display);
                continue;
            };

            // 在 String::from_utf8 之前按魔数识别 PNG/JPEG/WebP/GIF，命中的走图片缩放与重编管线，不再落 <attached-file>
            if crate::images::sniff_image_format(&bytes).is_some() {
                match crate::images::process_image_bytes(&bytes) {
                    Ok(processed) => {
                        out.images.push(processed.attachment);
                        out.image_paths.push(shown_path);
                        out.attached.push(display);
                    }
                    Err(err) => {
                        out.errors.push(format!("{display}（{err}）"));
                    }
                }
                continue;
            }

            if (bytes.len() as u64) > MENTION_MAX_READ_BYTES {
                out.missing.push(display);
                continue;
            }
            let Ok(raw_text) = String::from_utf8(bytes) else {
                out.missing.push(display);
                continue;
            };
            let clean_text = raw_text.strip_prefix('\u{FEFF}').unwrap_or(&raw_text);

            let mut header = format!("<attached-file path=\"{shown_attr}\">");
            let body_owned: String;
            let body_slice: &str = if let (Some(start_line), Some(end_line)) =
                (mention.start_line, mention.end_line)
                && !literal_fallback
            {
                let lines: Vec<&str> = clean_text
                    .split('\n')
                    .map(|l| l.strip_suffix('\r').unwrap_or(l))
                    .collect();
                if start_line > lines.len() {
                    header = format!(
                        "<attached-file path=\"{shown_attr}\" lines=\"{start_line}-{end_line}\" note=\"requested lines beyond EOF (file has {} line{}); whole file attached\">",
                        lines.len(),
                        if lines.len() == 1 { "" } else { "s" }
                    );
                    clean_text
                } else {
                    let clamped_end = end_line.min(lines.len());
                    let range_attr = if clamped_end == start_line {
                        format!("{start_line}")
                    } else {
                        format!("{start_line}-{clamped_end}")
                    };
                    header =
                        format!("<attached-file path=\"{shown_attr}\" lines=\"{range_attr}\">");
                    body_owned = lines[start_line - 1..clamped_end].join("\n");
                    &body_owned
                }
            } else {
                clean_text
            };

            let cap = MENTION_MAX_FILE_CHARS.min(budget);
            let (clipped, truncated) = truncate_chars(body_slice, cap);
            let used = clipped.chars().count();
            budget = budget.saturating_sub(used);
            let suffix = if truncated { "\n[… truncated]" } else { "" };
            out.blocks
                .push(format!("{header}\n{clipped}{suffix}\n</attached-file>"));
            out.attached.push(display);
            continue;
        }

        if meta.is_dir() {
            let Ok(mut rd) = tokio::fs::read_dir(&target).await else {
                out.missing.push(display);
                continue;
            };
            let mut entries = Vec::new();
            while let Ok(Some(entry)) = rd.next_entry().await {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name == ".git" || name == ".DS_Store" {
                    continue;
                }
                let is_dir = entry
                    .file_type()
                    .await
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false);
                entries.push(if is_dir { format!("{name}/") } else { name });
            }
            entries.sort();
            let total = entries.len();
            let mut listing: Vec<String> =
                entries.into_iter().take(MENTION_MAX_DIR_ENTRIES).collect();
            if total > MENTION_MAX_DIR_ENTRIES {
                listing.push(format!("… ({} more)", total - MENTION_MAX_DIR_ENTRIES));
            }
            let raw_listing = listing.join("\n");
            let (clipped, truncated) = truncate_chars(&raw_listing, budget);
            let used = clipped.chars().count();
            budget = budget.saturating_sub(used);
            let suffix = if truncated { "\n[… truncated]" } else { "" };
            out.blocks.push(format!(
                "<attached-directory path=\"{shown_attr}\">\n{clipped}{suffix}\n</attached-directory>"
            ));
            out.attached.push(display);
            continue;
        }

        out.missing.push(display);
    }

    out
}

fn model_supports_images(ctx: &Context, provider: Option<&str>, model: Option<&str>) -> bool {
    let Ok(llm) = ctx.llm() else {
        return true;
    };
    for prov in llm.provider_infos() {
        if provider.is_some_and(|pid| pid != prov.id) {
            continue;
        }
        for info in &prov.models {
            if model.is_none() || model == Some(info.id.as_str()) {
                return info.supports_images();
            }
        }
    }
    true
}

/// 挂载 `agent/pre-step` 监听器：在每轮首次请求模型前，自动展开最新用户消息中的 `@` 引用。
pub fn register_mention_hook(
    ctx: &Context,
    workspace: PathBuf,
    restrict_to_workspace: bool,
    tx: mpsc::UnboundedSender<ClientEvent>,
) -> Result<DisposableHandle, CordisError> {
    register_mention_hook_with(ctx, workspace, restrict_to_workspace, move |ev| {
        let _ = tx.send(ev);
    })
}

/// 挂载 `agent/pre-step` 监听器并通过自定义闭包投递 `ClientEvent`（便于上层包装成 `LoopClientMsg`）。
pub fn register_mention_hook_with<F>(
    ctx: &Context,
    workspace: PathBuf,
    restrict_to_workspace: bool,
    emit: F,
) -> Result<DisposableHandle, CordisError>
where
    F: Fn(ClientEvent) + Clone + Send + Sync + 'static,
{
    let ctx_hook = ctx.clone();

    ctx.on_waterfall_with(
        EVENT_AGENT_PRE_STEP,
        move |mut input: AgentPreStepInput,
              next: Next<AgentPreStepInput, BoxFuture<'static, AgentPreStepDecision>>| {
            let ws = workspace.clone();
            let emit = emit.clone();
            let ctx_hook = ctx_hook.clone();
            Box::pin(async move {
                if input.cancel.is_cancelled() {
                    return AgentPreStepDecision::Reject {
                        reason: "cancelled".to_string(),
                    };
                }

                let mut user_texts = Vec::new();
                let mut existing_images = 0usize;
                for item in &input.inputs {
                    if let TurnInput::User { text, images } = item {
                        user_texts.push(text.as_str());
                        existing_images += images.len();
                    }
                }

                if user_texts.is_empty() {
                    return next.call(input).await;
                }

                let combined_text = user_texts.join("\n");
                let mut result =
                    expand_text_mentions(&ws, restrict_to_workspace, &combined_text, &input.cancel)
                        .await;

                if input.cancel.is_cancelled() {
                    return AgentPreStepDecision::Reject {
                        reason: "cancelled".to_string(),
                    };
                }

                if !result.images.is_empty()
                    && !model_supports_images(
                        &ctx_hook,
                        input.provider.as_deref(),
                        input.model.as_deref(),
                    )
                {
                    let model_name = input.model.as_deref().unwrap_or("当前模型");
                    let reason = format!(
                        "当前模型 {model_name} 不支持图片输入，已取消 `@` 图片发送（{}）",
                        result.image_paths.join(", ")
                    );
                    emit(ClientEvent::Notice {
                        level: NoticeLevel::Error,
                        text: reason.clone(),
                    });
                    return AgentPreStepDecision::Reject { reason };
                }

                if existing_images + result.images.len() > MAX_IMAGES_PER_MESSAGE {
                    let allowed = MAX_IMAGES_PER_MESSAGE.saturating_sub(existing_images);
                    let skipped = result.images.len() - allowed;
                    result.images.truncate(allowed);
                    result.image_paths.truncate(allowed);
                    emit(ClientEvent::Notice {
                        level: NoticeLevel::Warn,
                        text: format!(
                            "单条消息最多附加 {MAX_IMAGES_PER_MESSAGE} 张图片，已跳过超出上限的 {skipped} 张 `@` 图片"
                        ),
                    });
                }

                let mut notice_sections = result.blocks;
                for path in &result.image_paths {
                    notice_sections.push(format!(
                        "<attached-image path=\"{}\" />",
                        escape_xml_attr(path)
                    ));
                }

                if !notice_sections.is_empty() || !result.images.is_empty() {
                    input.inputs.push(TurnInput::notice_with_images(
                        FILE_MENTIONS_SOURCE,
                        notice_sections.join("\n\n"),
                        result.images,
                    ));
                }

                if !result.attached.is_empty() {
                    emit(ClientEvent::Flash(format!(
                        "✓ 已附加 {} 个 @ 引用",
                        result.attached.len()
                    )));
                }
                if !result.errors.is_empty() {
                    emit(ClientEvent::Notice {
                        level: NoticeLevel::Error,
                        text: format!("@ 图片处理失败：{}", result.errors.join("; ")),
                    });
                }
                if !result.missing.is_empty() {
                    emit(ClientEvent::Notice {
                        level: NoticeLevel::Warn,
                        text: format!("未找到或无法读取 @ 引用：{}", result.missing.join(", ")),
                    });
                }

                next.call(input).await
            }) as BoxFuture<'static, AgentPreStepDecision>
        },
        cordis::EventOptions {
            priority: -50,
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tiny_png(width: u32, height: u32) -> Vec<u8> {
        let img = image::DynamicImage::ImageRgba8(image::ImageBuffer::from_pixel(
            width,
            height,
            image::Rgba([30, 144, 255, 255]),
        ));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    #[tokio::test]
    async fn expands_files_line_ranges_directories_and_missing_mentions() {
        let tmp = std::env::temp_dir().join(format!(
            "mizar_client_mention_expand_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("src/sub")).unwrap();
        std::fs::write(tmp.join("src/a.rs"), "line1\nline2\nline3\nline4\nline5\n").unwrap();
        std::fs::write(tmp.join("src/sub/b.rs"), "pub fn b() {}\n").unwrap();
        std::fs::write(tmp.join("spaced file.txt"), "hello spaces").unwrap();
        std::fs::write(tmp.join("literal#L2.txt"), "literal file").unwrap();

        let res = expand_text_mentions(
            &tmp,
            true,
            "看 @src/a.rs#L2-4 和 @\"spaced file.txt\" 以及 @src/sub 和 @literal#L2.txt 和 @missing.rs",
            &CancellationToken::new(),
        )
        .await;

        assert_eq!(res.attached.len(), 4);
        assert_eq!(res.missing, vec!["missing.rs".to_string()]);
        assert!(res.blocks[0].contains("lines=\"2-4\""));
        assert!(res.blocks[0].contains("line2\nline3\nline4"));
        assert!(!res.blocks[0].contains("line1"));
        assert!(res.blocks[1].contains("hello spaces"));
        assert!(res.blocks[2].contains("<attached-directory path=\"src/sub\">"));
        assert!(res.blocks[2].contains("b.rs"));
        assert!(res.blocks[3].contains("literal file"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn expands_image_mentions_into_attachments_without_attached_file_text() {
        let tmp = std::env::temp_dir().join(format!(
            "mizar_client_mention_images_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("shot.png"), tiny_png(3000, 1500)).unwrap();
        std::fs::write(tmp.join("note.txt"), "plain text").unwrap();
        // 坏掉的图片魔数文件走 error，不静默成 missing
        std::fs::write(
            tmp.join("broken.png"),
            [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0],
        )
        .unwrap();

        let res = expand_text_mentions(
            &tmp,
            true,
            "检查 @shot.png @note.txt @broken.png",
            &CancellationToken::new(),
        )
        .await;

        assert_eq!(res.images.len(), 1);
        assert_eq!(res.images[0].media_type, "image/png");
        assert_eq!(res.images[0].width, Some(2048));
        assert_eq!(res.images[0].height, Some(1024));
        assert_eq!(res.image_paths, vec!["shot.png".to_string()]);
        // 图片绝不落进 <attached-file>
        assert_eq!(res.blocks.len(), 1);
        assert!(res.blocks[0].contains("plain text"));
        assert!(!res.blocks.iter().any(|b| b.contains("shot.png")));
        assert!(res.missing.is_empty());
        assert_eq!(res.errors.len(), 1);
        assert!(res.errors[0].contains("broken.png"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn cancelled_check_does_not_latch_processed_seq() {
        use cordis::Service;
        use mizar_llm::{
            BoxChunkStream, GenerateOptions, LlmAdapter, LlmRuntime, ModelInfo, ProviderInfo,
        };
        use mizar_session::{SessionContextExt, SessionStore};
        use tokio::sync::mpsc;

        struct TextOnlyAdapter;
        impl LlmAdapter for TextOnlyAdapter {
            fn info(&self) -> Option<ProviderInfo> {
                Some(ProviderInfo {
                    id: "text-only-p".into(),
                    name: "Text Only".into(),
                    protocol: "mock".into(),
                    models: vec![
                        ModelInfo {
                            id: "text-only-m".into(),
                            name: "text-only-m".into(),
                            context_window: Some(32_000),
                            max_tokens: None,
                            reasoning_efforts: Vec::new(),
                            input: vec!["text".into()],
                        },
                        ModelInfo {
                            id: "vision-m".into(),
                            name: "vision-m".into(),
                            context_window: Some(32_000),
                            max_tokens: None,
                            reasoning_efforts: Vec::new(),
                            input: vec!["text".into(), "image".into()],
                        },
                    ],
                })
            }

            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                unreachable!("pre-step cancel test should not invoke stream")
            }
        }

        let tmp = std::env::temp_dir().join(format!(
            "mizar_client_mention_cancel_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("shot.png"), tiny_png(100, 100)).unwrap();

        let ctx = cordis::Context::new();
        ctx.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        let _adapter_guard = ctx
            .llm()
            .unwrap()
            .register_adapter("text-only-p", Arc::new(TextOnlyAdapter))
            .unwrap();

        let (tx, _rx) = mpsc::unbounded_channel();
        let _hook = register_mention_hook(&ctx, tmp.clone(), true, tx).unwrap();
        let session = ctx.sessions().unwrap().create(None);

        // 1. 不支持图片的模型在 pre-step 阶段直接 Reject，不入账用户消息
        let cancel1 = CancellationToken::new();
        let ev1 = AgentPreStepInput {
            session: Arc::clone(&session),
            turn: 1,
            step: 1,
            provider: Some("text-only-p".into()),
            model: Some("text-only-m".into()),
            cancel: cancel1.clone(),
            inputs: vec![TurnInput::user("请看 @shot.png")],
        };
        let decision1 = ctx
            .events()
            .waterfall(
                EVENT_AGENT_PRE_STEP,
                ev1,
                |inp| -> cordis::BoxFuture<'static, AgentPreStepDecision> {
                    Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
                },
            )
            .await;
        assert!(matches!(decision1, AgentPreStepDecision::Reject { .. }));

        // 2. 换用支持图片的模型重放同输入时顺利 Enter 并附加 file-mentions 通知
        let cancel2 = CancellationToken::new();
        let ev2 = AgentPreStepInput {
            session: Arc::clone(&session),
            turn: 1,
            step: 1,
            provider: Some("text-only-p".into()),
            model: Some("vision-m".into()),
            cancel: cancel2.clone(),
            inputs: vec![TurnInput::user("请看 @shot.png")],
        };
        let decision2 = ctx
            .events()
            .waterfall(
                EVENT_AGENT_PRE_STEP,
                ev2,
                |inp| -> cordis::BoxFuture<'static, AgentPreStepDecision> {
                    Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
                },
            )
            .await;
        assert!(!cancel2.is_cancelled());
        match decision2 {
            AgentPreStepDecision::Enter { inputs } => {
                assert!(inputs.iter().any(|item| {
                    matches!(
                        item,
                        TurnInput::Notice { source, images, .. }
                        if source == FILE_MENTIONS_SOURCE && images.len() == 1
                    )
                }));
            }
            other => panic!("expected Enter, got {other:?}"),
        }

        ctx.fiber().dispose().await;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn plugin_mounts_and_cleans_up_on_dispose() {
        use cordis::Service;
        use mizar_llm::LlmRuntime;
        use mizar_session::{SessionContextExt, SessionStore};

        let tmp = std::env::temp_dir().join(format!(
            "mizar_client_mention_plugin_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("hello.txt"), "world").unwrap();

        let ctx = Context::new();
        ctx.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let plugin_fiber = ctx
            .plugin(
                FileMentionsPlugin,
                FileMentionsConfig::new(tmp.clone(), true, tx),
            )
            .await
            .unwrap();

        let session = ctx.sessions().unwrap().create(None);

        let ev = AgentPreStepInput {
            session: Arc::clone(&session),
            turn: 1,
            step: 1,
            provider: None,
            model: None,
            cancel: CancellationToken::new(),
            inputs: vec![TurnInput::user("看 @hello.txt")],
        };
        let decision1 = ctx
            .events()
            .waterfall(
                EVENT_AGENT_PRE_STEP,
                ev,
                |inp| -> cordis::BoxFuture<'static, AgentPreStepDecision> {
                    Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
                },
            )
            .await;
        match decision1 {
            AgentPreStepDecision::Enter { inputs } => {
                assert!(inputs.iter().any(|item| {
                    matches!(
                        item,
                        TurnInput::Notice { source, .. } if source == FILE_MENTIONS_SOURCE
                    )
                }));
            }
            other => panic!("expected Enter, got {other:?}"),
        }
        assert!(matches!(rx.try_recv(), Ok(ClientEvent::Flash(_))));

        // 卸载插件后钩子应随之撤销，不再处理新消息
        plugin_fiber.dispose().await;
        let ev2 = AgentPreStepInput {
            session: Arc::clone(&session),
            turn: 2,
            step: 1,
            provider: None,
            model: None,
            cancel: CancellationToken::new(),
            inputs: vec![TurnInput::user("再看 @hello.txt")],
        };
        let decision2 = ctx
            .events()
            .waterfall(
                EVENT_AGENT_PRE_STEP,
                ev2,
                |inp| -> cordis::BoxFuture<'static, AgentPreStepDecision> {
                    Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
                },
            )
            .await;
        match decision2 {
            AgentPreStepDecision::Enter { inputs } => {
                assert_eq!(inputs.len(), 1);
            }
            other => panic!("expected Enter, got {other:?}"),
        }

        ctx.fiber().dispose().await;
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
