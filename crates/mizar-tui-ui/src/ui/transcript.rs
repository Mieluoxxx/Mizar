//! 对话流虚拟化渲染、Inline scrollback 行构建与流式 Markdown 增量前缀缓存。

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::BRAILLE_SPINNER;
use super::chrome::render_inline_pills;
use super::tool_card::{
    ToolGroupCtx, build_tool_lines, compute_tool_group_ctx, compute_tool_group_ctx_cached,
    is_tool_expandable, tool_base_status_code, tool_extra_len,
};
use super::welcome::build_welcome_section;
use crate::app::{
    AppState, ChatItem, ItemRenderKey, NoticeLevel, StreamingMarkdownPrefixCache, ToolExtraMeta,
    ToolState, TranscriptRenderCache, truncate_display,
};
use crate::emphasis::EmphasisConfig;
use crate::markdown::{
    RenderedLine, parse_inline_rich, render_markdown_rich, wrap_styled_spans_rich,
};
use crate::reveal::{RevealController, RevealSlot};
use crate::selection::SelectionPane;
use crate::terminal::ScreenMode;
use crate::theme::Theme;
use crate::timeline_rail::{
    RAIL_MIN_TERMINAL_WIDTH, RAIL_MIN_TURNS, RAIL_WIDTH, TimelineTurn, build_timeline_snapshot,
    compute_rail_geometry, render_timeline_rail,
};

const REASONING_CACHE_CAP: usize = 8;

#[derive(Clone)]
struct ReasoningCacheEntry {
    width: usize,
    theme_id: &'static str,
    emphasis_enabled: bool,
    round_caps: bool,
    source: String,
    lines: Arc<[RenderedLine]>,
}

struct ReasoningPrefixCache {
    width: usize,
    theme_id: &'static str,
    emphasis_enabled: bool,
    round_caps: bool,
    prefix_text: String,
    lines: Vec<RenderedLine>,
}

#[derive(Default)]
struct ReasoningRenderCache {
    entries: Vec<ReasoningCacheEntry>,
    streaming_prefix: Option<ReasoningPrefixCache>,
}

#[derive(Default)]
struct LazyResizeHeights {
    session_id: String,
    heights: HashMap<usize, (ItemRenderKey, usize)>,
}

thread_local! {
    static REASONING_RENDER_CACHE: RefCell<ReasoningRenderCache> =
        RefCell::new(ReasoningRenderCache::default());
    static LAZY_RESIZE_HEIGHTS: RefCell<LazyResizeHeights> =
        RefCell::new(LazyResizeHeights::default());
}

fn same_content_ignoring_width(a: &ItemRenderKey, b: &ItemRenderKey) -> bool {
    a.theme_id == b.theme_id
        && a.emphasis_enabled == b.emphasis_enabled
        && a.round_caps == b.round_caps
        && a.expanded == b.expanded
        && a.status_code == b.status_code
        && a.reasoning_len == b.reasoning_len
        && a.text_len == b.text_len
        && a.preview_len == b.preview_len
}

fn decorate_reasoning_lines(raw_lines: Vec<RenderedLine>, theme: &Theme) -> Vec<RenderedLine> {
    let mut out = Vec::with_capacity(raw_lines.len());
    for rl in raw_lines {
        let mut spans = vec![Span::styled("┆ ", Style::default().fg(theme.text_dim))];
        for s in rl.line.spans {
            spans.push(Span::styled(
                s.content,
                s.style.fg(theme.text_dim).add_modifier(Modifier::ITALIC),
            ));
        }
        out.push(RenderedLine {
            line: Line::from(spans),
            soft_wrapped: rl.soft_wrapped,
        });
    }
    out
}

fn render_reasoning_lines_cached(
    vis_reasoning: &str,
    quote_w: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> Arc<[RenderedLine]> {
    if vis_reasoning.is_empty() {
        return Arc::from([]);
    }

    let hit = REASONING_RENDER_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(pos) = cache.entries.iter().position(|e| {
            e.width == quote_w
                && e.theme_id == theme.id
                && e.emphasis_enabled == emphasis.enabled
                && e.round_caps == emphasis.round_caps
                && e.source.len() == vis_reasoning.len()
                && e.source == vis_reasoning
        }) {
            let entry = cache.entries.remove(pos);
            let lines = Arc::clone(&entry.lines);
            cache.entries.push(entry);
            Some(lines)
        } else {
            None
        }
    });
    if let Some(lines) = hit {
        return lines;
    }

    let built_vec: Vec<RenderedLine> = if vis_reasoning.len() > 1500
        && let Some(split_idx) = find_streaming_markdown_split(vis_reasoning)
    {
        let new_prefix = &vis_reasoning[..split_idx];
        let tail = &vis_reasoning[split_idx + 1..];
        let mut out = REASONING_RENDER_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(cached) = cache.streaming_prefix.as_mut()
                && cached.width == quote_w
                && cached.theme_id == theme.id
                && cached.emphasis_enabled == emphasis.enabled
                && cached.round_caps == emphasis.round_caps
            {
                if new_prefix == cached.prefix_text {
                    return Some(cached.lines.clone());
                }
                if new_prefix.len() > cached.prefix_text.len()
                    && new_prefix.starts_with(&cached.prefix_text)
                    && new_prefix.as_bytes().get(cached.prefix_text.len()) == Some(&b'\n')
                {
                    let delta_prefix = &new_prefix[cached.prefix_text.len() + 1..];
                    let delta_lines = decorate_reasoning_lines(
                        render_markdown_rich(delta_prefix, quote_w, theme, emphasis),
                        theme,
                    );
                    cached.lines.extend(delta_lines);
                    cached.prefix_text = new_prefix.to_string();
                    return Some(cached.lines.clone());
                }
            }
            None
        });
        if out.is_none() {
            let prefix_lines = decorate_reasoning_lines(
                render_markdown_rich(new_prefix, quote_w, theme, emphasis),
                theme,
            );
            REASONING_RENDER_CACHE.with(|cell| {
                cell.borrow_mut().streaming_prefix = Some(ReasoningPrefixCache {
                    width: quote_w,
                    theme_id: theme.id,
                    emphasis_enabled: emphasis.enabled,
                    round_caps: emphasis.round_caps,
                    prefix_text: new_prefix.to_string(),
                    lines: prefix_lines.clone(),
                });
            });
            out = Some(prefix_lines);
        }
        let mut combined = out.unwrap_or_default();
        combined.extend(decorate_reasoning_lines(
            render_markdown_rich(tail, quote_w, theme, emphasis),
            theme,
        ));
        combined
    } else {
        decorate_reasoning_lines(
            render_markdown_rich(vis_reasoning, quote_w, theme, emphasis),
            theme,
        )
    };

    let shared: Arc<[RenderedLine]> = Arc::from(built_vec);
    REASONING_RENDER_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(pos) = cache.entries.iter().position(|e| {
            e.width == quote_w
                && e.theme_id == theme.id
                && e.emphasis_enabled == emphasis.enabled
                && e.round_caps == emphasis.round_caps
                && vis_reasoning.starts_with(&e.source)
        }) {
            cache.entries.remove(pos);
        } else if cache.entries.len() >= REASONING_CACHE_CAP {
            cache.entries.remove(0);
        }
        cache.entries.push(ReasoningCacheEntry {
            width: quote_w,
            theme_id: theme.id,
            emphasis_enabled: emphasis.enabled,
            round_caps: emphasis.round_caps,
            source: vis_reasoning.to_string(),
            lines: Arc::clone(&shared),
        });
    });
    shared
}

/// 将单条已稳定完成的对话项渲染为可推送至 Inline 原生 scrollback 的行序列。
pub fn build_item_lines_for_scrollback(
    item: &ChatItem,
    extra: Option<&ToolExtraMeta>,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    show_thinking: bool,
    verbose_expanded: bool,
) -> Vec<Line<'static>> {
    let w = width.max(12);
    let rendered = match item {
        ChatItem::User { text } => build_user_lines(text, w, theme),
        ChatItem::Assistant { reasoning, text } => build_assistant_lines(
            reasoning,
            if show_thinking {
                reasoning.as_str()
            } else {
                ""
            },
            text.as_str(),
            show_thinking,
            w,
            theme,
            emphasis,
        ),
        ChatItem::ToolCall {
            name,
            args_summary,
            status,
            output_preview,
            ..
        } => {
            let show_expand_hint = !verbose_expanded && is_tool_expandable(item, extra, w, theme);
            build_tool_lines(
                name,
                args_summary,
                *status,
                output_preview,
                extra,
                verbose_expanded,
                ToolGroupCtx {
                    is_group_end: true,
                    show_expand_hint,
                },
                0,
                w,
                theme,
            )
        }
        ChatItem::Notice { level, text } => build_notice_lines(*level, text, w, theme, emphasis),
    };
    rendered.into_iter().map(|rl| rl.line).collect()
}

/// 在完整条目切片上下文中将第 `idx` 条渲染为 Inline scrollback 行序列，保持连续工具卡片分组去空行一致。
#[allow(clippy::too_many_arguments)]
pub fn build_item_lines_for_scrollback_at(
    items: &[ChatItem],
    idx: usize,
    tool_extras: &HashMap<String, ToolExtraMeta>,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    show_thinking: bool,
    verbose_expanded: bool,
) -> Vec<Line<'static>> {
    let Some(item) = items.get(idx) else {
        return Vec::new();
    };
    let w = width.max(12);
    let rendered = match item {
        ChatItem::User { text } => build_user_lines(text, w, theme),
        ChatItem::Assistant { reasoning, text } => build_assistant_lines(
            reasoning,
            if show_thinking {
                reasoning.as_str()
            } else {
                ""
            },
            text.as_str(),
            show_thinking,
            w,
            theme,
            emphasis,
        ),
        ChatItem::ToolCall {
            id,
            name,
            args_summary,
            status,
            output_preview,
        } => {
            let extra = tool_extras.get(id);
            let group_ctx =
                compute_tool_group_ctx(items, 0, idx, tool_extras, verbose_expanded, w, theme);
            build_tool_lines(
                name,
                args_summary,
                *status,
                output_preview,
                extra,
                verbose_expanded,
                group_ctx,
                0,
                w,
                theme,
            )
        }
        ChatItem::Notice { level, text } => build_notice_lines(*level, text, w, theme, emphasis),
    };
    rendered.into_iter().map(|rl| rl.line).collect()
}

#[allow(clippy::too_many_arguments)]
fn build_transcript_item_lines(
    items: &[ChatItem],
    start_item: usize,
    idx: usize,
    tool_extras: &HashMap<String, ToolExtraMeta>,
    reveal: &RevealController,
    cache: &mut TranscriptRenderCache,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    is_busy: bool,
    show_thinking: bool,
    verbose_expanded: bool,
    spinner_tick: usize,
    width: usize,
) -> Vec<RenderedLine> {
    let is_last_active = is_busy && idx + 1 == items.len();
    match &items[idx] {
        ChatItem::User { text } => build_user_lines(text, width, theme),
        ChatItem::Assistant { reasoning, text } => {
            let vis_reasoning = if show_thinking {
                reveal.visible_slice(RevealSlot::AssistantReasoning(idx), reasoning)
            } else {
                ""
            };
            let vis_text = reveal.visible_slice(RevealSlot::AssistantText(idx), text);
            build_assistant_lines_cached(
                reasoning,
                vis_reasoning,
                vis_text,
                show_thinking,
                width,
                theme,
                emphasis,
                cache,
                idx,
                is_last_active,
            )
        }
        ChatItem::ToolCall {
            id,
            name,
            args_summary,
            status,
            output_preview,
        } => {
            let extra = tool_extras.get(id);
            let group_ctx = compute_tool_group_ctx_cached(
                items,
                start_item,
                idx,
                tool_extras,
                cache,
                verbose_expanded,
                width,
                theme,
            );
            let spin_key = if matches!(status, ToolState::Running) {
                spinner_tick % BRAILLE_SPINNER.len()
            } else {
                0
            };
            build_tool_lines(
                name,
                args_summary,
                *status,
                output_preview,
                extra,
                verbose_expanded,
                group_ctx,
                spin_key,
                width,
                theme,
            )
        }
        ChatItem::Notice { level, text } => {
            build_notice_lines(*level, text, width, theme, emphasis)
        }
    }
}

pub(super) fn render_transcript(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    let total_items = app.items.len();
    let start_item = if app.screen_mode == ScreenMode::Inline {
        app.inline_committed_items.min(total_items)
    } else {
        0
    };
    let user_turn_count = app.items[start_item..]
        .iter()
        .filter(|i| matches!(i, ChatItem::User { .. }))
        .count();
    let reserve_rail = app.features.timeline_rail
        && app.screen_mode == ScreenMode::Fullscreen
        && area.width >= RAIL_MIN_TERMINAL_WIDTH
        && area.height >= 3
        && user_turn_count >= RAIL_MIN_TURNS;

    let content_w = if reserve_rail {
        area.width.saturating_sub(RAIL_WIDTH)
    } else {
        area.width
    };
    let content_area = Rect::new(area.x, area.y, content_w, area.height);
    app.selection_geom.chat_rect = Some(content_area);

    let width_usize = content_area.width.max(1) as usize;

    let emphasis = app.emphasis;
    let is_busy = app.is_busy();
    let show_thinking = app.show_thinking;
    let verbose_expanded = app.verbose_expanded;
    let spinner_tick = app.spinner_tick;

    if app.screen_mode == ScreenMode::Inline && !app.items.is_empty() && start_item >= total_items {
        let hint = Line::from(vec![
            Span::styled(
                " ↑ 历史回复已归档至终端原生滚动区 ",
                Style::default().fg(app.theme.text_dim),
            ),
            Span::styled("│ F3 切回全屏", Style::default().fg(app.theme.accent)),
        ]);
        frame.buffer_mut().set_line(
            content_area.x,
            content_area.y + content_area.height.saturating_sub(1),
            &hint,
            content_area.width,
        );
        return;
    }

    // 全屏模式始终在顶部保留欢迎页序言；Inline 模式仅在空会话时于视口内渲染欢迎页
    let (welcome_lines, welcome_chips, welcome_total_lines) =
        if app.screen_mode == ScreenMode::Fullscreen || app.items.is_empty() {
            let welcome = build_welcome_section(
                app,
                content_area.width as usize,
                content_area.height as usize,
            );
            let chips = welcome
                .chips_line_idx
                .map(|chip_line_idx| (chip_line_idx, welcome.chip_spans));
            let mut total = welcome.lines.len();
            if start_item < total_items {
                total += 1;
            }
            (welcome.lines, chips, total)
        } else {
            (Vec::new(), None, 0)
        };

    if app.tool_extras.len() > 64 {
        app.enforce_tool_extras_budget();
    }

    // Pass 1：利用轻量高度缓存 + 局部失效只对必要条目求行数并收集时间轴转折点
    let mut turns: Vec<TimelineTurn> = Vec::with_capacity(user_turn_count);
    let mut spans = std::mem::take(&mut app.item_cache.span_scratch);
    spans.clear();
    let item_count = total_items - start_item;
    if spans.capacity() < item_count {
        spans.reserve(item_count - spans.capacity());
    }

    let mut cur_line = welcome_total_lines;
    let auto_scroll = app.auto_scroll;
    let fixed_scroll_bottom = app
        .scroll_from_top
        .saturating_add(content_area.height as usize + 32);
    let scroll_top_hint = app.scroll_from_top;
    let eager_tail_start = total_items.saturating_sub((content_area.height as usize).max(24));

    LAZY_RESIZE_HEIGHTS.with(|cell| {
        let mut lazy = cell.borrow_mut();
        if lazy.session_id != app.session_id || app.item_cache.is_empty() {
            lazy.session_id.clone_from(&app.session_id);
            lazy.heights.clear();
        }
    });

    {
        let items = &app.items;
        let tool_extras = &app.tool_extras;
        let reveal = &mut app.reveal;
        let cache = &mut app.item_cache;
        let theme = &app.theme;

        for idx in start_item..total_items {
            let start_line = cur_line;
            let is_last_active = is_busy && idx + 1 == total_items;

            let key = match &items[idx] {
                ChatItem::User { text } => {
                    let preview = text
                        .lines()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or(text.as_str())
                        .to_string();
                    turns.push(TimelineTurn {
                        item_index: idx,
                        top_line: start_line,
                        preview,
                    });
                    ItemRenderKey {
                        width: width_usize,
                        theme_id: theme.id,
                        emphasis_enabled: emphasis.enabled,
                        round_caps: emphasis.round_caps,
                        expanded: verbose_expanded,
                        status_code: 0,
                        reasoning_len: 0,
                        text_len: text.len(),
                        preview_len: 0,
                    }
                }
                ChatItem::Assistant { reasoning, text } => {
                    reveal.observe(
                        RevealSlot::AssistantReasoning(idx),
                        reasoning.len(),
                        is_last_active,
                    );
                    reveal.observe(RevealSlot::AssistantText(idx), text.len(), is_last_active);

                    let vis_reasoning_len = if show_thinking {
                        reveal
                            .visible_slice(RevealSlot::AssistantReasoning(idx), reasoning)
                            .len()
                    } else {
                        0
                    };
                    let vis_text_len = reveal
                        .visible_slice(RevealSlot::AssistantText(idx), text)
                        .len();
                    let collapsed_marker = if !show_thinking && !reasoning.is_empty() {
                        1
                    } else {
                        0
                    };
                    ItemRenderKey {
                        width: width_usize,
                        theme_id: theme.id,
                        emphasis_enabled: emphasis.enabled,
                        round_caps: emphasis.round_caps,
                        expanded: verbose_expanded,
                        status_code: collapsed_marker,
                        reasoning_len: vis_reasoning_len,
                        text_len: vis_text_len,
                        preview_len: if show_thinking { 0 } else { reasoning.len() },
                    }
                }
                ChatItem::ToolCall {
                    id,
                    name,
                    args_summary,
                    status,
                    output_preview,
                } => {
                    let extra = tool_extras.get(id);
                    let group_ctx = compute_tool_group_ctx_cached(
                        items,
                        start_item,
                        idx,
                        tool_extras,
                        cache,
                        verbose_expanded,
                        width_usize,
                        theme,
                    );
                    let extra_len = tool_extra_len(extra);
                    let status_code = tool_base_status_code(*status)
                        | if group_ctx.is_group_end { 4 } else { 0 }
                        | if group_ctx.show_expand_hint { 8 } else { 0 };
                    let spin_key = if matches!(status, ToolState::Running) {
                        spinner_tick % BRAILLE_SPINNER.len()
                    } else {
                        0
                    };
                    ItemRenderKey {
                        width: width_usize,
                        theme_id: theme.id,
                        emphasis_enabled: emphasis.enabled,
                        round_caps: emphasis.round_caps,
                        expanded: verbose_expanded,
                        status_code,
                        reasoning_len: spin_key,
                        text_len: extra_len,
                        preview_len: args_summary.len() + output_preview.len() + name.len(),
                    }
                }
                ChatItem::Notice { level, text } => {
                    let level_code = match level {
                        NoticeLevel::Info => 0,
                        NoticeLevel::Warn => 1,
                        NoticeLevel::Error => 2,
                    };
                    ItemRenderKey {
                        width: width_usize,
                        theme_id: theme.id,
                        emphasis_enabled: emphasis.enabled,
                        round_caps: emphasis.round_caps,
                        expanded: verbose_expanded,
                        status_code: level_code,
                        reasoning_len: 0,
                        text_len: text.len(),
                        preview_len: 0,
                    }
                }
            };

            let line_count = if let Some(h) = cache.get_height(idx, &key) {
                LAZY_RESIZE_HEIGHTS.with(|cell| {
                    let mut lazy = cell.borrow_mut();
                    let up_to_date = lazy
                        .heights
                        .get(&idx)
                        .is_some_and(|(k, lh)| k == &key && *lh == h);
                    if !up_to_date {
                        lazy.heights.insert(idx, (key.clone(), h));
                    }
                });
                h
            } else {
                // Resize 期间离屏条目保留旧高度占位，仅视口附近条目立即重排
                let stale_offscreen_h = LAZY_RESIZE_HEIGHTS.with(|cell| {
                    let lazy = cell.borrow();
                    let (old_key, old_h) = lazy.heights.get(&idx)?;
                    if old_key.width == key.width || !same_content_ignoring_width(old_key, &key) {
                        return None;
                    }
                    let is_offscreen = if auto_scroll {
                        idx < eager_tail_start
                    } else {
                        start_line + *old_h <= scroll_top_hint.saturating_sub(16)
                            || start_line >= fixed_scroll_bottom
                    };
                    is_offscreen.then_some(*old_h)
                });
                if let Some(old_h) = stale_offscreen_h {
                    old_h
                } else {
                    let built = Arc::new(build_transcript_item_lines(
                        items,
                        start_item,
                        idx,
                        tool_extras,
                        reveal,
                        cache,
                        theme,
                        &emphasis,
                        is_busy,
                        show_thinking,
                        verbose_expanded,
                        spinner_tick,
                        width_usize,
                    ));
                    let len = built.len();
                    LAZY_RESIZE_HEIGHTS.with(|cell| {
                        cell.borrow_mut().heights.insert(idx, (key.clone(), len));
                    });
                    let far_below_fixed_viewport =
                        !auto_scroll && start_line >= fixed_scroll_bottom;
                    if far_below_fixed_viewport {
                        cache.record_height_only(idx, key.clone(), len);
                    } else {
                        cache.insert_lines(idx, key.clone(), built);
                    }
                    len
                }
            };

            spans.push((start_line, line_count, key));
            cur_line += line_count;
        }
    }

    let total_lines = cur_line;
    let scroll_top = if app.items.is_empty() && app.auto_scroll {
        app.last_viewport_height = content_area.height;
        app.last_total_lines = total_lines;
        app.scroll_from_top = 0;
        0
    } else {
        app.effective_scroll(content_area.height, total_lines)
    };
    let max_scroll = total_lines.saturating_sub(content_area.height as usize);
    if app.features.timeline_rail {
        app.timeline_snap = build_timeline_snapshot(turns, scroll_top, max_scroll);
    }

    // 顶部吸顶轮次条（仅当活动 turn 首行已滚出视口上方且视口足够高时显示）
    let show_sticky = content_area.height >= 6
        && app.timeline_snap.pinned_idx.is_some()
        && total_lines > content_area.height as usize;

    let buf = frame.buffer_mut();
    let sticky_rows: usize = if show_sticky {
        if let Some(pinned_idx) = app.timeline_snap.pinned_idx
            && let Some(turn) = app.timeline_snap.turns.get(pinned_idx)
        {
            let sticky_rect = Rect::new(content_area.x, content_area.y, content_area.width, 1);
            let sticky_bg = if app.theme.card_bg != Color::Reset {
                app.theme.card_bg
            } else {
                Color::DarkGray
            };
            buf.set_style(sticky_rect, Style::default().bg(sticky_bg));
            let label = format!(
                " ↳ Turn #{}: {} ",
                pinned_idx + 1,
                truncate_display(
                    &turn.preview,
                    content_area.width.saturating_sub(14) as usize
                )
            );
            buf.set_stringn(
                content_area.x,
                content_area.y,
                label,
                content_area.width as usize,
                Style::default()
                    .fg(app.theme.accent)
                    .bg(sticky_bg)
                    .add_modifier(Modifier::BOLD),
            );
        }
        1
    } else {
        0
    };

    // Pass 2：二分定位仅取样/物化命中当前可见视口 `[vis_line_start .. vis_line_end)` 的条目
    let view_h = content_area.height as usize;
    let vis_line_start = scroll_top + sticky_rows;
    let vis_line_end = scroll_top + view_h;
    let mut row_off = sticky_rows;

    if vis_line_start < welcome_total_lines {
        let empty_rl = RenderedLine::hard(Line::raw(""));
        for line_idx in vis_line_start..welcome_total_lines.min(vis_line_end) {
            if row_off >= view_h {
                break;
            }
            let y = content_area.y + row_off as u16;
            let rl = welcome_lines.get(line_idx).unwrap_or(&empty_rl);
            buf.set_line(content_area.x, y, &rl.line, content_area.width);
            let lw = (rl.line.width() as u16).min(content_area.width);
            let col_end = content_area.x + lw.max(1).min(content_area.width);
            app.selection_geom.push_row(
                SelectionPane::Chat,
                y,
                content_area.x,
                col_end,
                rl.soft_wrapped,
            );
            row_off += 1;
        }
    }

    let first_span_idx =
        spans.partition_point(|&(start, count, _)| start.saturating_add(count) <= vis_line_start);

    {
        let items = &app.items;
        let tool_extras = &app.tool_extras;
        let reveal = &app.reveal;
        let cache = &mut app.item_cache;
        let theme = &app.theme;
        let selection_geom = &mut app.selection_geom;

        for (k, &(item_start, _item_len, ref key)) in spans.iter().enumerate().skip(first_span_idx)
        {
            if item_start >= vis_line_end || row_off >= view_h {
                break;
            }
            let item_idx = start_item + k;
            let lines = if let Some(cached) = cache.get_lines(item_idx, key) {
                cached
            } else {
                let built = Arc::new(build_transcript_item_lines(
                    items,
                    start_item,
                    item_idx,
                    tool_extras,
                    reveal,
                    cache,
                    theme,
                    &emphasis,
                    is_busy,
                    show_thinking,
                    verbose_expanded,
                    spinner_tick,
                    width_usize,
                ));
                let built_len = built.len();
                LAZY_RESIZE_HEIGHTS.with(|cell| {
                    cell.borrow_mut()
                        .heights
                        .insert(item_idx, (key.clone(), built_len));
                });
                cache.insert_lines(item_idx, key.clone(), Arc::clone(&built));
                built
            };

            let local_start = vis_line_start.saturating_sub(item_start);
            let local_end = (vis_line_end.saturating_sub(item_start)).min(lines.len());
            for rl in &lines[local_start..local_end] {
                if row_off >= view_h {
                    break;
                }
                let y = content_area.y + row_off as u16;
                buf.set_line(content_area.x, y, &rl.line, content_area.width);
                let lw = (rl.line.width() as u16).min(content_area.width);
                let col_end = content_area.x + lw.max(1).min(content_area.width);
                selection_geom.push_row(
                    SelectionPane::Chat,
                    y,
                    content_area.x,
                    col_end,
                    rl.soft_wrapped,
                );
                row_off += 1;
            }
        }
    }
    app.item_cache.span_scratch = spans;

    // 仅当快捷操作行落在可视区间 `[sticky_rows..view_h)` 内时动态投影命中矩形
    if let Some((chip_line_idx, chip_spans)) = welcome_chips
        && chip_line_idx >= scroll_top + sticky_rows
        && chip_line_idx - scroll_top < view_h
    {
        let y = content_area.y + (chip_line_idx - scroll_top) as u16;
        for chip in chip_spans {
            app.launchpad_hit_rects.push((
                chip.action,
                Rect::new(content_area.x + chip.x_offset, y, chip.width, 1),
            ));
        }
    }

    // 2 列时间轴导轨绘制（使用同一份 timeline_snap 快照）
    if reserve_rail {
        let rail_area = Rect::new(
            content_area.x + content_area.width,
            area.y,
            RAIL_WIDTH,
            area.height,
        );
        let at_bottom = app.auto_scroll || scroll_top >= max_scroll;
        if let Some(geo) = compute_rail_geometry(rail_area, &app.timeline_snap, at_bottom) {
            render_timeline_rail(
                frame.buffer_mut(),
                &geo,
                &app.timeline_snap,
                app.hovered_rail_row,
                &app.theme,
            );
            app.rail_geo = Some(geo);
        }
    }
}

fn build_user_lines(text: &str, width: usize, theme: &Theme) -> Vec<RenderedLine> {
    let mut out = Vec::new();
    let user_style = Style::default().fg(theme.text).bg(theme.user_bg);
    let image_pill_style = Style::default()
        .fg(theme.info)
        .bg(theme.user_bg)
        .add_modifier(Modifier::BOLD);
    let prefix_first = Span::styled(
        "❯ ",
        Style::default()
            .fg(theme.accent)
            .bg(theme.user_bg)
            .add_modifier(Modifier::BOLD),
    );
    let prefix_cont = Span::styled("  ", Style::default().bg(theme.user_bg));

    let raw_lines: Vec<&str> = if text.is_empty() {
        vec![""]
    } else {
        text.split('\n').collect()
    };

    for (i, line) in raw_lines.iter().enumerate() {
        let pf = if i == 0 {
            Some(prefix_first.clone())
        } else {
            Some(prefix_cont.clone())
        };
        let spans = render_inline_pills(line, user_style, image_pill_style, false);
        let wrapped = wrap_styled_spans_rich(spans, pf, Some(prefix_cont.clone()), width);
        out.extend(wrapped);
    }
    out.push(RenderedLine::hard(Line::raw("")));
    out
}

fn find_streaming_markdown_split(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut fence: Option<(char, usize)> = None;
    let mut tree_possibly_active = false;
    let mut pos = 0usize;
    let mut last_safe: Option<usize> = None;

    while pos < bytes.len() {
        let rel_nl = bytes[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|r| pos + r);
        let line_end = rel_nl.unwrap_or(bytes.len());
        let line = &text[pos..line_end];
        let trimmed = line.trim_start();

        let in_fence_line = if let Some((fc, flen)) = fence {
            let close_len = trimmed.bytes().take_while(|&b| b == fc as u8).count();
            if close_len >= flen && trimmed[close_len..].trim().is_empty() {
                fence = None;
            }
            true
        } else if trimmed.starts_with("```") {
            let flen = trimmed.bytes().take_while(|&b| b == b'`').count();
            fence = Some(('`', flen));
            true
        } else if trimmed.starts_with("~~~") {
            let flen = trimmed.bytes().take_while(|&b| b == b'~').count();
            fence = Some(('~', flen));
            true
        } else {
            false
        };

        if in_fence_line {
            tree_possibly_active = false;
        } else if line.is_empty() && rel_nl.is_some() {
            let tail_start = line_end + 1;
            let next_end = bytes[tail_start..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(bytes.len(), |r| tail_start + r);
            let next_line = &text[tail_start..next_end];
            let nt = next_line.trim_start();
            let indent = next_line.len() - nt.len();
            let is_ordered = nt.find(". ").is_some_and(|dot| {
                (1..=4).contains(&dot) && nt[..dot].bytes().all(|b| b.is_ascii_digit())
            });
            let safe_next = !nt.is_empty()
                && indent < 2
                && !nt.starts_with("- ")
                && !nt.starts_with("* ")
                && !is_ordered
                && !next_line.contains('├')
                && !next_line.contains('└')
                && !next_line.contains('│');
            if !tree_possibly_active || safe_next {
                tree_possibly_active = false;
                // 在空白行结束的 `\n` 处切分：前段含尾部单个 `\n`，后段从下一行起点开始
                last_safe = Some(line_end);
            }
        } else if line.contains('├') || line.contains('└') || line.contains('│') {
            tree_possibly_active = true;
        } else {
            let indent = line.len() - trimmed.len();
            let is_ordered = trimmed.find(". ").is_some_and(|dot| {
                (1..=4).contains(&dot) && trimmed[..dot].bytes().all(|b| b.is_ascii_digit())
            });
            if indent < 2
                && !trimmed.is_empty()
                && !trimmed.starts_with("- ")
                && !trimmed.starts_with("* ")
                && !is_ordered
            {
                tree_possibly_active = false;
            }
        }

        match rel_nl {
            Some(nl) => pos = nl + 1,
            None => break,
        }
    }

    last_safe
}

fn render_markdown_streaming_cached(
    cache: &mut TranscriptRenderCache,
    item_idx: usize,
    vis_text: &str,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    allow_cache: bool,
) -> Vec<RenderedLine> {
    if !allow_cache || vis_text.len() <= 1500 {
        return render_markdown_rich(vis_text, width, theme, emphasis);
    }
    let Some(split_idx) = find_streaming_markdown_split(vis_text) else {
        return render_markdown_rich(vis_text, width, theme, emphasis);
    };
    let new_prefix = &vis_text[..split_idx];
    let tail = &vis_text[split_idx + 1..];

    if let Some(cached) = cache.streaming_prefix.as_mut()
        && cached.item_idx == item_idx
        && cached.width == width
        && cached.theme_id == theme.id
        && cached.emphasis_enabled == emphasis.enabled
        && cached.round_caps == emphasis.round_caps
    {
        if new_prefix == cached.prefix_text {
            let mut out = cached.lines.clone();
            out.extend(render_markdown_rich(tail, width, theme, emphasis));
            return out;
        }
        if new_prefix.len() > cached.prefix_text.len()
            && new_prefix.starts_with(&cached.prefix_text)
            && new_prefix.as_bytes().get(cached.prefix_text.len()) == Some(&b'\n')
        {
            let delta_prefix = &new_prefix[cached.prefix_text.len() + 1..];
            cached
                .lines
                .extend(render_markdown_rich(delta_prefix, width, theme, emphasis));
            cached.prefix_text = new_prefix.to_string();
            let mut out = cached.lines.clone();
            out.extend(render_markdown_rich(tail, width, theme, emphasis));
            return out;
        }
    }

    let prefix_lines = render_markdown_rich(new_prefix, width, theme, emphasis);
    cache.streaming_prefix = Some(StreamingMarkdownPrefixCache {
        item_idx,
        width,
        theme_id: theme.id,
        emphasis_enabled: emphasis.enabled,
        round_caps: emphasis.round_caps,
        prefix_text: new_prefix.to_string(),
        lines: prefix_lines.clone(),
    });
    let mut out = prefix_lines;
    out.extend(render_markdown_rich(tail, width, theme, emphasis));
    out
}

fn build_assistant_lines(
    full_reasoning: &str,
    vis_reasoning: &str,
    vis_text: &str,
    show_thinking: bool,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> Vec<RenderedLine> {
    build_assistant_lines_with_cache(
        full_reasoning,
        vis_reasoning,
        vis_text,
        show_thinking,
        width,
        theme,
        emphasis,
        None,
        0,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_assistant_lines_cached(
    full_reasoning: &str,
    vis_reasoning: &str,
    vis_text: &str,
    show_thinking: bool,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    cache: &mut TranscriptRenderCache,
    item_idx: usize,
    is_last_active: bool,
) -> Vec<RenderedLine> {
    build_assistant_lines_with_cache(
        full_reasoning,
        vis_reasoning,
        vis_text,
        show_thinking,
        width,
        theme,
        emphasis,
        Some(cache),
        item_idx,
        is_last_active,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_assistant_lines_with_cache(
    full_reasoning: &str,
    vis_reasoning: &str,
    vis_text: &str,
    show_thinking: bool,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    cache: Option<&mut TranscriptRenderCache>,
    item_idx: usize,
    is_last_active: bool,
) -> Vec<RenderedLine> {
    let mut out = Vec::new();

    let mut emitted_reasoning = false;
    if !full_reasoning.is_empty() {
        if show_thinking {
            if !vis_reasoning.is_empty() {
                let quote_w = width.saturating_sub(2).max(2);
                let r_lines =
                    render_reasoning_lines_cached(vis_reasoning, quote_w, theme, emphasis);
                if !r_lines.is_empty() {
                    out.extend(r_lines.iter().cloned());
                    emitted_reasoning = true;
                }
            }
        } else {
            let line_count = full_reasoning.lines().count().max(1);
            let collapsed = format!("┆ 思考过程已折叠 ({line_count} 行，按 Ctrl+O 展开)");
            out.extend(wrap_styled_spans_rich(
                vec![Span::styled(
                    collapsed,
                    Style::default()
                        .fg(theme.text_dim)
                        .add_modifier(Modifier::ITALIC),
                )],
                None,
                None,
                width,
            ));
            emitted_reasoning = true;
        }
    }

    if !vis_text.is_empty() {
        if emitted_reasoning {
            out.push(RenderedLine::hard(Line::raw("")));
        }
        let md_lines = if let Some(c) = cache {
            render_markdown_streaming_cached(
                c,
                item_idx,
                vis_text,
                width,
                theme,
                emphasis,
                is_last_active,
            )
        } else {
            render_markdown_rich(vis_text, width, theme, emphasis)
        };
        out.extend(md_lines);
    } else if full_reasoning.is_empty() {
        out.push(RenderedLine::hard(Line::from(Span::styled(
            "● …",
            Style::default().fg(theme.text_dim),
        ))));
    }

    out.push(RenderedLine::hard(Line::raw("")));
    out
}

fn build_notice_lines(
    level: NoticeLevel,
    text: &str,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> Vec<RenderedLine> {
    let (prefix, color) = match level {
        NoticeLevel::Info => ("! ", theme.info),
        NoticeLevel::Warn => ("⚠ ", theme.warning),
        NoticeLevel::Error => ("✗ ", theme.error),
    };
    let mut out = Vec::new();
    let prefix_span = Span::styled(
        prefix,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    );
    let cont_span = Span::raw("  ");
    let raw_lines: Vec<&str> = text.split('\n').collect();
    for (idx, line) in raw_lines.iter().enumerate() {
        let spans = parse_inline_rich(line, Style::default().fg(color), theme, emphasis, false);
        let pf = if idx == 0 {
            Some(prefix_span.clone())
        } else {
            Some(cont_span.clone())
        };
        out.extend(wrap_styled_spans_rich(
            spans,
            pf,
            Some(cont_span.clone()),
            width,
        ));
    }
    out.push(RenderedLine::hard(Line::raw("")));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_markdown_prefix_cache_matches_full_render() {
        let theme = Theme::mist_dark();
        let emphasis = EmphasisConfig::default();
        let long_prefix: String = (0..30)
            .map(|i| {
                format!(
                    "## Section {i}\n\nParagraph {i} with **bold** and `code_{i}`.\n\n```rust\nlet x_{i} = {i};\n```\n\n"
                )
            })
            .collect();
        let stages = [
            format!("{long_prefix}Tail stage one"),
            format!("{long_prefix}Tail stage one\n\nTail stage two with - list"),
            format!("{long_prefix}Tail stage one\n\nTail stage two\n\nFinal paragraph."),
        ];

        let mut cache = TranscriptRenderCache::default();
        for stage in &stages {
            let cached_lines =
                render_markdown_streaming_cached(&mut cache, 7, stage, 84, &theme, &emphasis, true);
            let full_lines = render_markdown_rich(stage, 84, &theme, &emphasis);
            let cached_cmp: Vec<(bool, &_)> = cached_lines
                .iter()
                .map(|l| (l.soft_wrapped, &l.line))
                .collect();
            let full_cmp: Vec<(bool, &_)> = full_lines
                .iter()
                .map(|l| (l.soft_wrapped, &l.line))
                .collect();
            assert_eq!(cached_cmp, full_cmp);
        }
        assert!(cache.streaming_prefix.is_some());
    }

    #[test]
    fn vis_reasoning_cache_reuses_arc_slice_across_text_streaming_ticks() {
        let theme = Theme::mist_dark();
        let emphasis = EmphasisConfig::default();
        let reasoning =
            "Step 1: analyze requirements.\n\nStep 2: check `render_transcript` CJK 渲染性能。";

        let first = render_reasoning_lines_cached(reasoning, 76, &theme, &emphasis);
        let second = render_reasoning_lines_cached(reasoning, 76, &theme, &emphasis);
        assert!(!first.is_empty());
        assert!(Arc::ptr_eq(&first, &second));

        // 宽度变化应未命中并重排，不返回同一份 Arc
        let narrower = render_reasoning_lines_cached(reasoning, 32, &theme, &emphasis);
        assert!(!Arc::ptr_eq(&first, &narrower));
    }
}
