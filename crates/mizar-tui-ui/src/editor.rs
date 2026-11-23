//! 支持 Unicode / 中文宽度、词级编辑、原子粘贴块与撤销栈的现代终端编辑器
//!（融合 `pi` `editor.ts` + `dsh-TUI` `PromptEditor.tsx`）。

use std::collections::HashMap;
use std::sync::Arc;

use mizar_llm_types::ImageAttachment;
use unicode_width::UnicodeWidthChar;

use crate::app::BashInputMode;
use crate::mentions::{MentionCaret, mention_at_caret};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputLayout {
    pub lines: Vec<String>,
    pub cursor_row: u16,
    pub cursor_col: u16,
}

#[derive(Debug, Clone)]
struct UndoEntry {
    text: String,
    cursor: usize,
    pastes: HashMap<String, Arc<PasteEntry>>,
    paste_seq: usize,
    image_seq: usize,
}

#[derive(Debug, Clone)]
struct PasteEntry {
    token: String,
    raw: String,
    image: Option<ImageAttachment>,
}

const PASTE_COLLAPSE_MIN_LINES: usize = 8;
const PASTE_COLLAPSE_MIN_CHARS: usize = 600;
const MAX_UNDO_STACK: usize = 100;

#[derive(Debug, Clone)]
struct CachedInputLayout {
    text: String,
    cursor: usize,
    max_width: u16,
    layout: InputLayout,
}

#[derive(Debug, Clone, Default)]
enum HistoryNavState {
    #[default]
    Inactive,
    Recalling {
        index: usize,
        draft_text: String,
        draft_cursor: usize,
        draft_pastes: HashMap<String, Arc<PasteEntry>>,
        draft_paste_seq: usize,
        draft_image_seq: usize,
    },
}

/// 多行文本编辑器：按 UTF-8 字符边界编辑、原子粘贴块（长文本与图片）、词级导航/撤销，并按终端显示列宽折行定位光标。
#[derive(Debug, Default, Clone)]
pub struct TextInput {
    text: String,
    /// 当前光标在 `text` 中的 UTF-8 字节偏移。
    cursor: usize,
    history: Vec<String>,
    history_nav: HistoryNavState,
    undo_stack: Vec<UndoEntry>,
    kill_ring: String,
    paste_seq: usize,
    image_seq: usize,
    pastes: HashMap<String, Arc<PasteEntry>>,
    layout_cache: Option<CachedInputLayout>,
}

impl TextInput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor_byte(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn history(&self) -> &[String] {
        &self.history
    }

    pub fn has_collapsed_pastes(&self) -> bool {
        !self.pastes.is_empty()
            && self
                .pastes
                .values()
                .any(|entry| entry.image.is_none() && self.text.contains(&entry.token))
    }

    pub fn active_images(&self) -> Vec<(String, ImageAttachment)> {
        let mut positioned: Vec<(usize, String, ImageAttachment)> = Vec::new();
        for entry in self.pastes.values() {
            let Some(img) = entry.image.as_ref() else {
                continue;
            };
            for (pos, _) in self.text.match_indices(&entry.token) {
                positioned.push((pos, entry.token.clone(), img.clone()));
            }
        }
        positioned.sort_by_key(|(pos, _, _)| *pos);
        positioned
            .into_iter()
            .map(|(_, token, img)| (token, img))
            .collect()
    }

    pub fn image_count(&self) -> usize {
        if self.pastes.is_empty() {
            return 0;
        }
        self.pastes
            .values()
            .filter(|entry| entry.image.is_some())
            .map(|entry| self.text.match_indices(&entry.token).count())
            .sum()
    }

    pub fn has_images(&self) -> bool {
        if self.pastes.is_empty() {
            return false;
        }
        self.pastes
            .values()
            .any(|entry| entry.image.is_some() && self.text.contains(&entry.token))
    }

    fn push_undo(&mut self) {
        if let Some(last) = self.undo_stack.last()
            && last.text == self.text
            && last.cursor == self.cursor
            && last.pastes.len() == self.pastes.len()
            && last.paste_seq == self.paste_seq
            && last.image_seq == self.image_seq
        {
            return;
        }
        if self.undo_stack.len() >= MAX_UNDO_STACK {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(UndoEntry {
            text: self.text.clone(),
            cursor: self.cursor,
            pastes: self.pastes.clone(),
            paste_seq: self.paste_seq,
            image_seq: self.image_seq,
        });
    }

    pub fn undo(&mut self) -> bool {
        let Some(prev) = self.undo_stack.pop() else {
            return false;
        };
        self.text = prev.text;
        self.cursor = prev.cursor.min(self.text.len());
        self.pastes = prev.pastes;
        self.paste_seq = self.paste_seq.max(prev.paste_seq);
        self.image_seq = self.image_seq.max(prev.image_seq);
        self.history_nav = HistoryNavState::Inactive;
        true
    }

    pub fn clear(&mut self) {
        if !self.text.is_empty() || !self.pastes.is_empty() {
            self.push_undo();
        }
        self.text.clear();
        self.cursor = 0;
        self.history_nav = HistoryNavState::Inactive;
        self.pastes.clear();
        self.image_seq = 0;
    }

    /// 展开所有原子文本粘贴令牌为原始文本（图片令牌保留在正文中）。
    pub fn expand_text(&self) -> String {
        let mut out = self.text.clone();
        for entry in self.pastes.values() {
            if entry.image.is_none() {
                out = out.replace(&entry.token, &entry.raw);
            }
        }
        out
    }

    /// 将编辑器内的折叠文本粘贴块就地展开为可编辑文本（保留图片原子块）。
    pub fn expand_pastes_in_place(&mut self) -> bool {
        if !self.has_collapsed_pastes() {
            return false;
        }
        self.push_undo();
        self.text = self.expand_text();
        self.cursor = self.text.len();
        self.pastes.retain(|_, entry| entry.image.is_some());
        true
    }

    pub fn take(&mut self) -> String {
        self.take_with_images().0
    }

    /// 提取展开后的文本与当前编辑器内按顺序排列的所有有效图片附件，随后清空草稿状态。
    pub fn take_with_images(&mut self) -> (String, Vec<(String, ImageAttachment)>) {
        let images = self.active_images();
        let out = self.expand_text();
        self.text.clear();
        self.cursor = 0;
        self.history_nav = HistoryNavState::Inactive;
        self.undo_stack.clear();
        self.pastes.clear();
        self.image_seq = 0;
        (out, images)
    }

    pub fn push_history(&mut self, entry: &str) {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.history.last().map(String::as_str) != Some(trimmed) {
            self.history.push(trimmed.to_string());
        }
    }

    pub fn set_text_end(&mut self, value: &str) {
        self.push_undo();
        self.text = value.to_string();
        self.cursor = self.text.len();
        self.history_nav = HistoryNavState::Inactive;
        self.pastes.clear();
        self.image_seq = 0;
    }

    /// 用队列内容替换草稿，并清空撤销栈：被换下的草稿已经停放到队列尾部，
    /// 留着撤销点会让 `Ctrl+Z` 把同一段文字同时还原进输入框，造成重复发送。
    pub fn replace_draft_from_queue(&mut self, value: &str) {
        self.replace_draft_from_queue_with_images(value, &[]);
    }

    /// 用包含图片的队列内容替换草稿，重新注册图片原子令牌以便继续整块编辑或删除。
    pub fn replace_draft_from_queue_with_images(
        &mut self,
        value: &str,
        images: &[(String, ImageAttachment)],
    ) {
        self.text = value.to_string();
        self.cursor = self.text.len();
        self.history_nav = HistoryNavState::Inactive;
        self.undo_stack.clear();
        self.pastes.clear();
        self.image_seq = images.len();
        for (token, image) in images {
            self.pastes.insert(
                token.clone(),
                Arc::new(PasteEntry {
                    token: token.clone(),
                    raw: String::new(),
                    image: Some(image.clone()),
                }),
            );
        }
    }

    pub fn insert_char(&mut self, ch: char) {
        let normalized = if ch == '\r' { '\n' } else { ch };
        // 在词边界或换行前记录撤销点，避免单字母打字塞满撤销栈
        if normalized.is_whitespace() || self.undo_stack.is_empty() {
            self.push_undo();
        }
        self.history_nav = HistoryNavState::Inactive;
        self.text.insert(self.cursor, normalized);
        self.cursor += normalized.len_utf8();
    }

    pub fn insert_str(&mut self, raw: &str) {
        if raw.is_empty() {
            return;
        }
        self.push_undo();
        self.history_nav = HistoryNavState::Inactive;
        let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
        self.text.insert_str(self.cursor, &normalized);
        self.cursor += normalized.len();
    }

    /// 插入剪贴板或 `@` 图片原子占位符（如 `[Image #1]` 或 `[Image #1 · 4032×3024 → 2048×1536]`）。
    pub fn insert_image(&mut self, summary_suffix: Option<&str>, image: ImageAttachment) -> String {
        self.push_undo();
        self.history_nav = HistoryNavState::Inactive;
        let token = loop {
            self.image_seq += 1;
            let candidate = match summary_suffix.map(str::trim).filter(|s| !s.is_empty()) {
                Some(s) => format!("[Image #{} · {s}]", self.image_seq),
                None => format!("[Image #{}]", self.image_seq),
            };
            if !self.pastes.contains_key(&candidate) {
                break candidate;
            }
        };
        self.pastes.insert(
            token.clone(),
            Arc::new(PasteEntry {
                token: token.clone(),
                raw: String::new(),
                image: Some(image),
            }),
        );
        self.text.insert_str(self.cursor, &token);
        self.cursor += token.len();
        token
    }

    /// 处理终端 Bracketed Paste：超过阈值自动折叠为 `[paste #N +X lines]` 原子令牌。
    pub fn insert_paste(&mut self, raw: &str) {
        if raw.is_empty() {
            return;
        }
        let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
        let line_count = normalized.lines().count();
        let char_count = normalized.chars().count();
        if line_count > PASTE_COLLAPSE_MIN_LINES || char_count > PASTE_COLLAPSE_MIN_CHARS {
            self.push_undo();
            self.history_nav = HistoryNavState::Inactive;
            self.paste_seq += 1;
            let token = if line_count > 1 {
                format!("[paste #{} +{} lines]", self.paste_seq, line_count)
            } else {
                format!("[paste #{} {} chars]", self.paste_seq, char_count)
            };
            self.pastes.insert(
                token.clone(),
                Arc::new(PasteEntry {
                    token: token.clone(),
                    raw: normalized,
                    image: None,
                }),
            );
            self.text.insert_str(self.cursor, &token);
            self.cursor += token.len();
        } else {
            self.insert_str(&normalized);
        }
    }

    /// 在当前词内替换为补全内容，并把光标定位到 `replacement` 内指定字节偏移处。
    pub fn replace_byte_range_with_cursor(
        &mut self,
        start: usize,
        end: usize,
        replacement: &str,
        cursor_offset: usize,
    ) {
        if start > end || end > self.text.len() {
            return;
        }
        self.push_undo();
        self.history_nav = HistoryNavState::Inactive;
        self.text.replace_range(start..end, replacement);
        self.cursor = start + cursor_offset.min(replacement.len());
    }

    /// 在当前词内替换为补全内容（例如 `/` 命令或 `@` 文件路径补全）。
    pub fn replace_byte_range(&mut self, start: usize, end: usize, replacement: &str) {
        self.replace_byte_range_with_cursor(start, end, replacement, replacement.len());
    }

    fn paste_span_containing(&self, byte_idx: usize) -> Option<(usize, usize)> {
        for token in self.pastes.keys() {
            let mut offset = 0usize;
            while let Some(rel) = self.text[offset..].find(token) {
                let start = offset + rel;
                let end = start + token.len();
                if byte_idx > start && byte_idx <= end {
                    return Some((start, end));
                }
                offset = end;
            }
        }
        None
    }

    fn paste_span_starting_at(&self, byte_idx: usize) -> Option<(usize, usize)> {
        for token in self.pastes.keys() {
            if self.text[byte_idx..].starts_with(token) {
                return Some((byte_idx, byte_idx + token.len()));
            }
        }
        None
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.push_undo();
        self.history_nav = HistoryNavState::Inactive;
        if let Some((start, end)) = self.paste_span_containing(self.cursor) {
            self.text.replace_range(start..end, "");
            self.cursor = start;
            return;
        }
        if let Some((prev_idx, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.text.replace_range(prev_idx..self.cursor, "");
            self.cursor = prev_idx;
        }
    }

    pub fn delete(&mut self) {
        if self.cursor >= self.text.len() {
            return;
        }
        self.push_undo();
        self.history_nav = HistoryNavState::Inactive;
        if let Some((start, end)) = self.paste_span_starting_at(self.cursor) {
            self.text.replace_range(start..end, "");
            return;
        }
        if let Some(ch) = self.text[self.cursor..].chars().next() {
            let end = self.cursor + ch.len_utf8();
            self.text.replace_range(self.cursor..end, "");
        }
    }

    pub fn delete_word_backward(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.push_undo();
        self.history_nav = HistoryNavState::Inactive;
        let target = self.find_word_left();
        self.kill_ring = self.text[target..self.cursor].to_string();
        self.text.replace_range(target..self.cursor, "");
        self.cursor = target;
    }

    pub fn delete_to_line_start(&mut self) {
        let line_start = self.text[..self.cursor]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        if line_start < self.cursor {
            self.push_undo();
            self.history_nav = HistoryNavState::Inactive;
            self.kill_ring = self.text[line_start..self.cursor].to_string();
            self.text.replace_range(line_start..self.cursor, "");
            self.cursor = line_start;
        }
    }

    pub fn delete_to_line_end(&mut self) {
        let line_end = self.text[self.cursor..]
            .find('\n')
            .map(|p| self.cursor + p)
            .unwrap_or(self.text.len());
        if self.cursor < line_end {
            self.push_undo();
            self.history_nav = HistoryNavState::Inactive;
            self.kill_ring = self.text[self.cursor..line_end].to_string();
            self.text.replace_range(self.cursor..line_end, "");
        }
    }

    /// 粘回最近一次剪切（Ctrl+W / Ctrl+U / Ctrl+K）的内容（Emacs KillRing）。
    pub fn yank(&mut self) -> bool {
        if self.kill_ring.is_empty() {
            return false;
        }
        let text = self.kill_ring.clone();
        self.insert_str(&text);
        true
    }

    pub fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        if let Some((start, _)) = self.paste_span_containing(self.cursor) {
            self.cursor = start;
            return;
        }
        if let Some((prev_idx, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.cursor = prev_idx;
        }
    }

    pub fn move_right(&mut self) {
        if self.cursor >= self.text.len() {
            return;
        }
        if let Some((_, end)) = self.paste_span_starting_at(self.cursor) {
            self.cursor = end;
            return;
        }
        if let Some(ch) = self.text[self.cursor..].chars().next() {
            self.cursor += ch.len_utf8();
        }
    }

    fn find_word_left(&self) -> usize {
        if self.cursor == 0 {
            return 0;
        }
        if let Some((start, _)) = self.paste_span_containing(self.cursor) {
            return start;
        }
        let prefix = &self.text[..self.cursor];
        let mut iter = prefix.char_indices().rev().peekable();
        while let Some(&(_, ch)) = iter.peek() {
            if ch.is_whitespace() {
                iter.next();
            } else {
                break;
            }
        }
        let mut target = 0usize;
        for (idx, ch) in iter {
            if ch.is_whitespace() {
                break;
            }
            target = idx;
        }
        if let Some((start, _)) = self.paste_span_containing(target) {
            return start;
        }
        target
    }

    fn find_word_right(&self) -> usize {
        if self.cursor >= self.text.len() {
            return self.text.len();
        }
        if let Some((_, end)) = self.paste_span_starting_at(self.cursor) {
            return end;
        }
        let suffix = &self.text[self.cursor..];
        let mut iter = suffix.char_indices().peekable();
        while let Some(&(_, ch)) = iter.peek() {
            if !ch.is_whitespace() {
                iter.next();
            } else {
                break;
            }
        }
        while let Some(&(idx, ch)) = iter.peek() {
            if ch.is_whitespace() {
                iter.next();
            } else {
                let target = self.cursor + idx;
                if let Some((_, end)) = self.paste_span_containing(target)
                    && target < end
                {
                    return end;
                }
                return target;
            }
        }
        self.text.len()
    }

    pub fn move_word_left(&mut self) {
        self.cursor = self.find_word_left();
    }

    pub fn move_word_right(&mut self) {
        self.cursor = self.find_word_right();
    }

    pub fn move_home(&mut self) {
        let line_start = self.text[..self.cursor]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        self.cursor = line_start;
    }

    pub fn move_end(&mut self) {
        let line_end = self.text[self.cursor..]
            .find('\n')
            .map(|p| self.cursor + p)
            .unwrap_or(self.text.len());
        self.cursor = line_end;
    }

    /// 向上移动一行；若已在首行则向上翻找历史输入。
    pub fn move_up(&mut self) {
        let line_start = self.text[..self.cursor]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        if line_start == 0 {
            self.recall_prev_history();
            return;
        }
        let target_col = display_width(&self.text[line_start..self.cursor]);
        let prev_end = line_start - 1;
        let prev_start = self.text[..prev_end]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        let mut next = prev_start + offset_at_width(&self.text[prev_start..prev_end], target_col);
        if let Some((start, end)) = self.paste_span_containing(next)
            && next < end
        {
            next = start;
        }
        self.cursor = next;
    }

    /// 向下移动一行；若已在末行且处于历史回溯中则向下恢复。
    pub fn move_down(&mut self) {
        let Some(nl_rel) = self.text[self.cursor..].find('\n') else {
            self.recall_next_history();
            return;
        };
        let line_start = self.text[..self.cursor]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        let target_col = display_width(&self.text[line_start..self.cursor]);
        let next_start = self.cursor + nl_rel + 1;
        let next_end = self.text[next_start..]
            .find('\n')
            .map(|p| next_start + p)
            .unwrap_or(self.text.len());
        let mut next = next_start + offset_at_width(&self.text[next_start..next_end], target_col);
        if let Some((start, end)) = self.paste_span_containing(next)
            && next < end
        {
            next = start;
        }
        self.cursor = next;
    }

    fn recall_prev_history(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next_idx = match &mut self.history_nav {
            HistoryNavState::Inactive => {
                let idx = self.history.len() - 1;
                self.history_nav = HistoryNavState::Recalling {
                    index: idx,
                    draft_text: std::mem::take(&mut self.text),
                    draft_cursor: self.cursor,
                    draft_pastes: std::mem::take(&mut self.pastes),
                    draft_paste_seq: self.paste_seq,
                    draft_image_seq: self.image_seq,
                };
                self.image_seq = 0;
                idx
            }
            HistoryNavState::Recalling { index, .. } => {
                *index = index.saturating_sub(1);
                *index
            }
        };
        self.text = self.history[next_idx].clone();
        self.cursor = self.text.len();
    }

    fn recall_next_history(&mut self) {
        let len = self.history.len();
        match &self.history_nav {
            HistoryNavState::Inactive => return,
            HistoryNavState::Recalling { index, .. } if index + 1 < len => {
                let next_idx = index + 1;
                if let HistoryNavState::Recalling { index, .. } = &mut self.history_nav {
                    *index = next_idx;
                }
                self.text = self.history[next_idx].clone();
                self.cursor = self.text.len();
                return;
            }
            HistoryNavState::Recalling { .. } => {}
        }
        if let HistoryNavState::Recalling {
            draft_text,
            draft_cursor,
            draft_pastes,
            draft_paste_seq,
            draft_image_seq,
            ..
        } = std::mem::take(&mut self.history_nav)
        {
            self.text = draft_text;
            self.cursor = draft_cursor.min(self.text.len());
            self.pastes = draft_pastes;
            self.paste_seq = self.paste_seq.max(draft_paste_seq);
            self.image_seq = draft_image_seq;
        }
    }

    pub fn bash_mode(&self) -> Option<BashInputMode> {
        BashInputMode::from_input(&self.text)
    }

    /// 检测光标当前是否处于 `/` 命令首行前缀或 `@` 文件引用片段中，返回 `(token_start_byte, query)`。
    pub fn active_slash_trigger(&self) -> Option<(usize, String)> {
        if self.bash_mode().is_some() || !self.text.starts_with('/') {
            return None;
        }
        let up_to_cursor = &self.text[..self.cursor];
        if up_to_cursor.chars().any(char::is_whitespace) {
            return None;
        }
        let token = &up_to_cursor[1..];
        // 对齐 pi / dsh-TUI：包含 `/` 时代表输入文件系统路径（如 `/Users/...`），不弹出命令补全
        if token.contains('/') {
            return None;
        }
        Some((0, token.to_string()))
    }

    pub fn active_mention_caret(&self) -> Option<MentionCaret> {
        mention_at_caret(&self.text, self.cursor)
    }

    pub fn active_mention_trigger(&self) -> Option<(usize, String)> {
        self.active_mention_caret().map(|m| (m.start, m.query))
    }

    /// 按终端显示列宽对输入文本折行，并计算物理光标位置 `(row, col)`。
    pub fn layout(&self, max_width: u16) -> InputLayout {
        let width = max_width.max(1) as usize;
        let mut lines = Vec::new();
        let mut current = String::new();
        let mut col = 0usize;
        let mut cursor_pos: Option<(u16, u16)> = None;

        for (idx, ch) in self.text.char_indices() {
            if idx == self.cursor && cursor_pos.is_none() {
                if col >= width {
                    lines.push(std::mem::take(&mut current));
                    col = 0;
                }
                cursor_pos = Some((lines.len() as u16, col as u16));
            }

            if ch == '\n' {
                lines.push(std::mem::take(&mut current));
                col = 0;
                continue;
            }

            let ch_w = char_width(ch);
            if ch_w > 0 && col + ch_w > width && col > 0 {
                lines.push(std::mem::take(&mut current));
                col = 0;
            }
            current.push(ch);
            col += ch_w;
        }

        if cursor_pos.is_none() {
            if col >= width {
                lines.push(std::mem::take(&mut current));
                col = 0;
            }
            cursor_pos = Some((lines.len() as u16, col as u16));
        }

        lines.push(current);
        let (cursor_row, cursor_col) = cursor_pos.unwrap_or((0, 0));
        InputLayout {
            lines,
            cursor_row,
            cursor_col,
        }
    }

    /// 缓存版输入框排版：仅当草稿文本、光标偏移或视口宽度变化时才重新计算换行。
    pub fn layout_cached(&mut self, max_width: u16) -> &InputLayout {
        let hit = self.layout_cache.as_ref().is_some_and(|c| {
            c.max_width == max_width && c.cursor == self.cursor && c.text == self.text
        });
        if !hit {
            let layout = self.layout(max_width);
            self.layout_cache = Some(CachedInputLayout {
                text: self.text.clone(),
                cursor: self.cursor,
                max_width,
                layout,
            });
        }
        &self.layout_cache.as_ref().unwrap().layout
    }
}

pub fn char_width(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

pub fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

fn offset_at_width(line: &str, target_col: usize) -> usize {
    let mut col = 0usize;
    for (idx, ch) in line.char_indices() {
        let w = char_width(ch);
        if col + w > target_col {
            return idx;
        }
        col += w;
    }
    line.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_cjk_cursor_and_wrapping() {
        let mut input = TextInput::new();
        input.insert_str("你好ab");
        // 每个中文占 2 列，总宽度 6；宽度 4 应折成两行："你好" (4) + "ab" (2)
        let layout = input.layout(4);
        assert_eq!(layout.lines, vec!["你好".to_string(), "ab".to_string()]);
        assert_eq!((layout.cursor_row, layout.cursor_col), (1, 2));

        input.move_left();
        let layout2 = input.layout(4);
        assert_eq!((layout2.cursor_row, layout2.cursor_col), (1, 1));
    }

    #[test]
    fn handles_multiline_paste_and_history() {
        let mut input = TextInput::new();
        input.insert_str("first");
        let sent = input.take();
        input.push_history(&sent);

        input.insert_str("line1\r\nline2");
        assert_eq!(input.text(), "line1\nline2");
        input.move_up();
        input.move_up();
        assert_eq!(input.text(), "first");
        input.move_down();
        assert_eq!(input.text(), "line1\nline2");
    }

    #[test]
    fn collapses_large_paste_as_atomic_token_and_expands_on_take() {
        let mut input = TextInput::new();
        input.insert_str("check ");
        let big = (1..=12)
            .map(|i| format!("line-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        input.insert_paste(&big);
        assert!(input.text().contains("[paste #1 +12 lines]"));
        assert!(input.has_collapsed_pastes());

        // 单次 move_left 跳过整个原子粘贴块
        input.move_left();
        assert_eq!(input.cursor_byte(), "check ".len());
        input.move_right();
        assert_eq!(input.cursor_byte(), input.text().len());

        // take 自动还原原文
        let expanded = input.take();
        assert_eq!(expanded, format!("check {big}"));
    }

    #[test]
    fn supports_word_edit_undo_and_triggers() {
        let mut input = TextInput::new();
        input.insert_str("/mod");
        assert_eq!(input.active_slash_trigger(), Some((0, "mod".to_string())));

        input.clear();
        input.insert_str("/Users/alice/docs/plugins.md");
        assert_eq!(input.active_slash_trigger(), None);

        input.clear();
        input.insert_str("read @src/ap");
        assert_eq!(
            input.active_mention_trigger(),
            Some((5, "src/ap".to_string()))
        );

        input.delete_word_backward();
        assert_eq!(input.text(), "read ");
        assert!(input.yank());
        assert_eq!(input.text(), "read @src/ap");
        assert!(input.undo());
        assert_eq!(input.text(), "read ");
        assert!(input.undo());
        assert_eq!(input.text(), "read @src/ap");
    }

    #[test]
    fn image_markers_are_atomic_across_cursor_delete_and_expand() {
        let img1 = ImageAttachment::new("image/png", "QUFB").with_dimensions(2048, 1536);
        let img2 = ImageAttachment::new("image/jpeg", "QkJC").with_dimensions(800, 600);
        let mut input = TextInput::new();
        input.insert_str("看下 ");
        let tok1 = input.insert_image(Some("4032×3024 → 2048×1536"), img1.clone());
        assert_eq!(tok1, "[Image #1 · 4032×3024 → 2048×1536]");
        assert_eq!(input.image_count(), 1);
        assert!(!input.has_collapsed_pastes());

        // 单次 Left / Right 必须跳过整个图片占位符，不能进入内部字符
        let after_prefix = "看下 ".len();
        input.move_left();
        assert_eq!(input.cursor_byte(), after_prefix);
        input.move_right();
        assert_eq!(input.cursor_byte(), input.text().len());

        // Ctrl+W（delete_word_backward）也不能将 `[Image #1 · ...]` 从空格处撕碎
        input.delete_word_backward();
        assert_eq!(input.text(), "看下 ");
        assert_eq!(input.image_count(), 0);
        assert!(input.undo());
        assert_eq!(input.text(), format!("看下 {tok1}"));
        assert_eq!(input.image_count(), 1);

        // 单次 Backspace 整块删除图片占位符
        input.backspace();
        assert_eq!(input.text(), "看下 ");
        assert_eq!(input.image_count(), 0);
        assert!(input.undo());
        assert_eq!(input.image_count(), 1);

        // 跳到占位符起始位置按 Delete 也整块删除
        input.move_left();
        assert_eq!(input.cursor_byte(), after_prefix);
        input.delete();
        assert_eq!(input.text(), "看下 ");
        assert_eq!(input.image_count(), 0);
        assert!(input.undo());

        // 混用折叠文本 paste + Ctrl+E 就地展开：只展开文本粘贴，图片占位符保持原样和有效附件
        input.move_end();
        input.insert_str(" ");
        let big = (1..=10)
            .map(|i| format!("l{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        input.insert_paste(&big);
        let tok2 = input.insert_image(None, img2.clone());
        assert!(input.has_collapsed_pastes());
        assert_eq!(input.image_count(), 2);
        assert!(input.expand_pastes_in_place());
        assert!(!input.has_collapsed_pastes());
        assert_eq!(input.image_count(), 2);
        assert!(input.text().contains(&tok1));
        assert!(input.text().contains(&tok2));

        let (sent_text, sent_images) = input.take_with_images();
        assert!(sent_text.contains("l10"));
        assert!(sent_text.contains(&tok1));
        assert!(sent_text.contains(&tok2));
        assert_eq!(sent_images, vec![(tok1, img1), (tok2, img2)]);
        assert!(input.is_empty());
        assert_eq!(input.image_count(), 0);
    }

    #[test]
    fn undo_and_history_navigation_preserve_pastes_and_images() {
        let mut input = TextInput::new();
        let big = (1..=10)
            .map(|i| format!("line-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let img = ImageAttachment::new("image/png", "AAAA").with_dimensions(640, 480);

        // 1. 折叠粘贴 + 图片后 clear / set_text_end，Ctrl+Z 必须完整还原 pastes 与 image_seq
        input.insert_paste(&big);
        input.insert_str(" ");
        let img_tok = input.insert_image(Some("640×480"), img.clone());
        assert!(input.has_collapsed_pastes());
        assert_eq!(input.image_count(), 1);

        input.clear();
        assert!(input.is_empty());
        assert_eq!(input.image_count(), 0);

        assert!(input.undo());
        assert!(input.has_collapsed_pastes());
        assert_eq!(input.image_count(), 1);
        assert!(input.expand_text().contains("line-10"));

        input.set_text_end("replaced draft");
        assert_eq!(input.image_count(), 0);
        assert!(input.undo());
        assert_eq!(input.image_count(), 1);
        assert!(input.text().contains(&img_tok));

        // 2. 历史翻阅期间原草稿中的粘贴/图片与光标位置完全隔离保存，下翻恢复时精确还原
        input.push_history("older prompt");
        input.push_history("latest prompt");
        input.move_left();
        let saved_cursor = input.cursor_byte();

        input.move_up();
        assert_eq!(input.text(), "latest prompt");
        assert_eq!(input.image_count(), 0);
        assert!(!input.has_collapsed_pastes());

        input.move_up();
        assert_eq!(input.text(), "older prompt");

        input.move_down();
        assert_eq!(input.text(), "latest prompt");
        input.move_down();
        assert!(input.has_collapsed_pastes());
        assert_eq!(input.image_count(), 1);
        assert_eq!(input.cursor_byte(), saved_cursor);
    }

    #[test]
    fn push_undo_shares_paste_entries_and_counters_avoid_cloning() {
        let mut input = TextInput::new();
        let big = (1..=12)
            .map(|i| format!("paste-line-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        input.insert_paste(&big);
        let img = ImageAttachment::new("image/png", "A".repeat(1024)).with_dimensions(320, 240);
        let tok = input.insert_image(None, img);

        assert!(input.has_images());
        assert_eq!(input.image_count(), input.active_images().len());

        // 敲空格触发 push_undo()：undo 快照与当前 pastes 应共享同一份 Arc<PasteEntry>
        input.insert_str(" hello ");
        let active_arc = input.pastes.get(&tok).expect("image entry");
        let undo_arc = input
            .undo_stack
            .last()
            .and_then(|u| u.pastes.get(&tok))
            .expect("undo image entry");
        assert!(Arc::ptr_eq(active_arc, undo_arc));

        // 删掉图片占位符后计数器清零，撤销后恢复
        for _ in 0.." hello ".len() {
            input.backspace();
        }
        input.backspace();
        assert!(!input.has_images());
        assert_eq!(input.image_count(), 0);
        assert!(input.undo());
        assert!(input.has_images());
        assert_eq!(input.image_count(), 1);
    }
}
