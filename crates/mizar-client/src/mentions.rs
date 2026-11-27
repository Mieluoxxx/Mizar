//! 提交时 `@` 文件/目录/图片引用展开、`agent/pre-step` 钩子与 `FileMentionsPlugin`。
//!
//! 保留用户输入原文在 `SessionEvent::User` 中（气泡只渲染用户所写），
//! 同时把解析得到的 `<attached-file ...>`、`<attached-directory ...>` 与图片附件
//! 以 `SessionEvent::ContextNotice { source: "file-mentions", .. }` 注入本轮模型上下文。

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cordis::{
    BoxError, BoxFuture, Cleanup, Context, CordisError, DisposableHandle, Inject, Plugin,
};
use mizar_agent_loop::{AgentPreStepEvent, EVENT_AGENT_PRE_STEP};
use mizar_client_core::{
    ClientEvent, FILE_MENTIONS_SOURCE, NoticeLevel, PLUGIN_FILE_MENTIONS, extract_mentions,
};
use mizar_llm::{ImageAttachment, LlmContextExt, MAX_IMAGES_PER_MESSAGE};
use mizar_session::SessionEvent;
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
            if meta.len() > mizar_image::MAX_IMAGE_INPUT_BYTES as u64 {
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
    let processed_seq_by_session: Arc<Mutex<HashMap<String, u64>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let ctx_hook = ctx.clone();

    ctx.on_serial::<AgentPreStepEvent, (), _, _>(
        EVENT_AGENT_PRE_STEP,
        move |ev: AgentPreStepEvent| {
            let ws = workspace.clone();
            let emit = emit.clone();
            let processed = Arc::clone(&processed_seq_by_session);
            let ctx_hook = ctx_hook.clone();
            async move {
                if ev.cancel.is_cancelled() {
                    return None;
                }

                let sid = ev.session.id.clone();
                let last_done_seq = processed.lock().unwrap().get(&sid).copied().unwrap_or(0);

                let trailing_users: Vec<(u64, String, usize)> =
                    ev.session.with_visible_entries(|entries| {
                        let mut collected = Vec::new();
                        for entry in entries.iter().rev() {
                            if entry.seq <= last_done_seq {
                                break;
                            }
                            match &entry.event {
                                SessionEvent::User { text, images } => {
                                    collected.push((entry.seq, text.clone(), images.len()))
                                }
                                SessionEvent::ContextNotice { source, .. }
                                    if source != FILE_MENTIONS_SOURCE => {}
                                SessionEvent::ContextNotice { .. }
                                | SessionEvent::Assistant { .. }
                                | SessionEvent::ToolResult { .. }
                                | SessionEvent::Compaction { .. }
                                | SessionEvent::ToolResultPrune { .. }
                                | SessionEvent::BashExecution { .. }
                                | SessionEvent::Selection { .. }
                                | SessionEvent::Trace { .. } => break,
                            }
                        }
                        collected.reverse();
                        collected
                    });

                if trailing_users.is_empty() {
                    return None;
                }

                let max_seq = trailing_users.iter().map(|(s, _, _)| *s).max().unwrap_or(0);
                let existing_images: usize = trailing_users.iter().map(|(_, _, n)| *n).sum();
                let combined_text = trailing_users
                    .iter()
                    .map(|(_, t, _)| t.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                let mut result =
                    expand_text_mentions(&ws, restrict_to_workspace, &combined_text, &ev.cancel)
                        .await;

                if ev.cancel.is_cancelled() {
                    return None;
                }

                if !result.images.is_empty()
                    && !model_supports_images(
                        &ctx_hook,
                        ev.provider.as_deref(),
                        ev.model.as_deref(),
                    )
                {
                    let model_name = ev.model.as_deref().unwrap_or("当前模型");
                    emit(ClientEvent::Notice {
                        level: NoticeLevel::Error,
                        text: format!(
                            "当前模型 {model_name} 不支持图片输入，已取消 `@` 图片发送（{}）",
                            result.image_paths.join(", ")
                        ),
                    });
                    ev.cancel.cancel();
                    return None;
                }

                processed.lock().unwrap().insert(sid, max_seq);

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
                    ev.session.append(SessionEvent::ContextNotice {
                        source: FILE_MENTIONS_SOURCE.to_string(),
                        text: notice_sections.join("\n\n"),
                        images: result.images,
                    });
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

                None
            }
        },
    )
}
