//! 支持 Unicode / 中文宽度、词级编辑、原子粘贴块与撤销栈的现代终端编辑器
//!（融合 `pi` `editor.ts` + `dsh-TUI` `PromptEditor.tsx`）。

use std::collections::HashMap;

use unicode_width::UnicodeWidthChar;

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
}

#[derive(Debug, Clone)]
struct PasteEntry {
    token: String,
    raw: String,
}

const PASTE_COLLAPSE_MIN_LINES: usize = 8;
const PASTE_COLLAPSE_MIN_CHARS: usize = 600;
const MAX_UNDO_STACK: usize = 100;

/// 多行文本编辑器：按 UTF-8 字符边界编辑、原子粘贴块、词级导航/撤销，并按终端显示列宽折行定位光标。
#[derive(Debug, Default, Clone)]
pub struct TextInput {
    text: String,
    /// 当前光标在 `text` 中的 UTF-8 字节偏移。
    cursor: usize,
    history: Vec<String>,
    history_idx: Option<usize>,
    draft_before_history: String,
    undo_stack: Vec<UndoEntry>,
    kill_ring: String,
    paste_seq: usize,
    pastes: HashMap<String, PasteEntry>,
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
        !self.pastes.is_empty() && self.pastes.keys().any(|t| self.text.contains(t))
    }

    fn push_undo(&mut self) {
        if let Some(last) = self.undo_stack.last()
            && last.text == self.text
            && last.cursor == self.cursor
        {
            return;
        }
        if self.undo_stack.len() >= MAX_UNDO_STACK {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(UndoEntry {
            text: self.text.clone(),
            cursor: self.cursor,
        });
    }

    pub fn undo(&mut self) -> bool {
        let Some(prev) = self.undo_stack.pop() else {
            return false;
        };
        self.text = prev.text;
        self.cursor = prev.cursor.min(self.text.len());
        self.history_idx = None;
        true
    }

    pub fn clear(&mut self) {
        if !self.text.is_empty() {
            self.push_undo();
        }
        self.text.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.pastes.clear();
    }

    /// 展开所有原子粘贴令牌为原始文本（在提交或展开草稿编辑器时调用）。
    pub fn expand_text(&self) -> String {
        let mut out = self.text.clone();
        for entry in self.pastes.values() {
            out = out.replace(&entry.token, &entry.raw);
        }
        out
    }

    /// 将编辑器内的折叠粘贴块就地展开为可编辑文本。
    pub fn expand_pastes_in_place(&mut self) -> bool {
        if !self.has_collapsed_pastes() {
            return false;
        }
        self.push_undo();
        self.text = self.expand_text();
        self.cursor = self.text.len();
        self.pastes.clear();
        true
    }

    pub fn take(&mut self) -> String {
        let out = self.expand_text();
        self.text.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.draft_before_history.clear();
        self.undo_stack.clear();
        self.pastes.clear();
        out
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
        self.history_idx = None;
        self.pastes.clear();
    }

    pub fn insert_char(&mut self, ch: char) {
        let normalized = if ch == '\r' { '\n' } else { ch };
        // 在词边界或换行前记录撤销点，避免单字母打字塞满撤销栈
        if normalized.is_whitespace() || self.undo_stack.is_empty() {
            self.push_undo();
        }
        self.history_idx = None;
        self.text.insert(self.cursor, normalized);
        self.cursor += normalized.len_utf8();
    }

    pub fn insert_str(&mut self, raw: &str) {
        if raw.is_empty() {
            return;
        }
        self.push_undo();
        self.history_idx = None;
        let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
        self.text.insert_str(self.cursor, &normalized);
        self.cursor += normalized.len();
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
            self.history_idx = None;
            self.paste_seq += 1;
            let token = if line_count > 1 {
                format!("[paste #{} +{} lines]", self.paste_seq, line_count)
            } else {
                format!("[paste #{} {} chars]", self.paste_seq, char_count)
            };
            self.pastes.insert(
                token.clone(),
                PasteEntry {
                    token: token.clone(),
                    raw: normalized,
                },
            );
            self.text.insert_str(self.cursor, &token);
            self.cursor += token.len();
        } else {
            self.insert_str(&normalized);
        }
    }

    /// 在当前词内替换为补全内容（例如 `/` 命令或 `@` 文件路径补全）。
    pub fn replace_byte_range(&mut self, start: usize, end: usize, replacement: &str) {
        if start > end || end > self.text.len() {
            return;
        }
        self.push_undo();
        self.history_idx = None;
        self.text.replace_range(start..end, replacement);
        self.cursor = start + replacement.len();
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
        self.history_idx = None;
        if let Some((start, end)) = self.paste_span_containing(self.cursor)
            && self.cursor == end
        {
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
        self.history_idx = None;
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
        if let Some((start, end)) = self.paste_span_containing(self.cursor)
            && self.cursor == end
        {
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
        target
    }

    fn find_word_right(&self) -> usize {
        if self.cursor >= self.text.len() {
            return self.text.len();
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
                return self.cursor + idx;
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
        self.cursor = prev_start + offset_at_width(&self.text[prev_start..prev_end], target_col);
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
        self.cursor = next_start + offset_at_width(&self.text[next_start..next_end], target_col);
    }

    fn recall_prev_history(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next_idx = match self.history_idx {
            None => {
                self.draft_before_history = self.text.clone();
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_idx = Some(next_idx);
        self.text = self.history[next_idx].clone();
        self.cursor = self.text.len();
    }

    fn recall_next_history(&mut self) {
        let Some(idx) = self.history_idx else {
            return;
        };
        if idx + 1 < self.history.len() {
            let next_idx = idx + 1;
            self.history_idx = Some(next_idx);
            self.text = self.history[next_idx].clone();
        } else {
            self.history_idx = None;
            self.text = std::mem::take(&mut self.draft_before_history);
        }
        self.cursor = self.text.len();
    }

    /// 检测光标当前是否处于 `/` 命令首行前缀或 `@` 文件引用片段中，返回 `(token_start_byte, query)`。
    pub fn active_slash_trigger(&self) -> Option<(usize, String)> {
        if !self.text.starts_with('/') {
            return None;
        }
        let up_to_cursor = &self.text[..self.cursor];
        if up_to_cursor.contains('\n') || up_to_cursor.contains(' ') {
            return None;
        }
        Some((0, up_to_cursor[1..].to_string()))
    }

    pub fn active_mention_trigger(&self) -> Option<(usize, String)> {
        let prefix = &self.text[..self.cursor];
        let at_pos = prefix.rfind('@')?;
        if at_pos > 0 {
            let prev = prefix[..at_pos].chars().next_back()?;
            if !prev.is_whitespace() && !matches!(prev, '(' | '[' | '"' | '\'') {
                return None;
            }
        }
        let token = &prefix[at_pos + 1..];
        if token.contains(char::is_whitespace) {
            return None;
        }
        Some((at_pos, token.to_string()))
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
}
