//! 键盘、鼠标、粘贴、`/` 斜杠命令与浮层动作分派，受 `AppState.features` 插件开关约束。

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use mizar_llm_types::{ImageAttachment, MAX_IMAGES_PER_MESSAGE, ReasoningEffort};

use super::{
    AppAction, AppState, ApprovalMode, BashInputMode, LaunchpadQuickAction, NoticeLevel,
    PendingSettingsModelPick,
};
use crate::mentions::format_mention_replacement;
use crate::overlay::{ApprovalDecision, OverlayAction, OverlayKind, SettingItemKind};
use crate::selection::{copy_to_clipboard, extract_selected_text};
use crate::terminal::{ProgramStatusState, ScreenMode};
use crate::theme::ThemePreset;
use crate::timeline_rail::{RailHit, hit_test_rail};
use crate::trajectory_scene::TrajectoryKeyOutcome;

impl AppState {
    /// 根据光标上下文同步 `/` 或 `@` 补全浮层。
    pub fn sync_input_completion_overlay(&mut self) {
        if self.overlay.is_modal() {
            return;
        }
        // Bash 模式下禁止触发 `/` 斜杠命令补全
        if self.bash_mode().is_none()
            && let Some((start, query)) = self.input.active_slash_trigger()
        {
            self.dismissed_mention = None;
            let selected = match &self.overlay.active {
                Some(OverlayKind::SlashComplete {
                    query: prev_query,
                    selected,
                    ..
                }) if prev_query == &query => *selected,
                _ => 0,
            };
            self.overlay.active = Some(OverlayKind::SlashComplete {
                token_start: start,
                query,
                selected,
            });
            return;
        }
        if let Some(caret) = self.input.active_mention_caret() {
            if self
                .dismissed_mention
                .as_ref()
                .is_some_and(|(s, q)| *s == caret.start && q == &caret.query)
            {
                if matches!(self.overlay.active, Some(OverlayKind::FileMention { .. })) {
                    self.overlay.active = None;
                }
                return;
            }
            self.dismissed_mention = None;
            let matches = self.file_cache.match_query(&self.workspace, &caret.query);
            let selected = match &self.overlay.active {
                Some(OverlayKind::FileMention {
                    matches: prev_matches,
                    selected: prev_sel,
                    ..
                }) => prev_matches
                    .get(*prev_sel)
                    .and_then(|prev| matches.iter().position(|m| m.path == prev.path))
                    .unwrap_or((*prev_sel).min(matches.len().saturating_sub(1))),
                _ => 0,
            };
            self.overlay.active = Some(OverlayKind::FileMention {
                token_start: caret.start,
                token_end: caret.end,
                path_end: caret.path_end,
                is_quoted: caret.is_quoted,
                query: caret.query,
                matches,
                selected,
            });
            return;
        }
        self.dismissed_mention = None;
        if matches!(
            self.overlay.active,
            Some(OverlayKind::SlashComplete { .. } | OverlayKind::FileMention { .. })
        ) {
            self.overlay.active = None;
        }
    }

    fn apply_overlay_action(&mut self, action: OverlayAction) -> AppAction {
        let out = match action {
            OverlayAction::None => AppAction::None,
            OverlayAction::Close => {
                if let Some(closed) = self.overlay.active.take() {
                    match closed {
                        OverlayKind::FileMention {
                            token_start, query, ..
                        } => {
                            self.dismissed_mention = Some((token_start, query));
                        }
                        OverlayKind::ModelPicker { .. } => {
                            if let Some(pending) = self.pending_settings_model_pick.take() {
                                self.overlay.active = Some(OverlayKind::SettingsModal {
                                    items: pending.items,
                                    query: pending.query,
                                    selected: pending.selected,
                                    save_path_hint: pending.save_path_hint,
                                });
                            }
                        }
                        _ => {}
                    }
                }
                AppAction::None
            }
            OverlayAction::CompleteSlash {
                token_start,
                command,
                execute_now,
            } => {
                self.overlay.active = None;
                let end = self.input.cursor_byte();
                if execute_now {
                    self.input.clear();
                    self.handle_slash_command(&command)
                } else {
                    self.input
                        .replace_byte_range(token_start, end, &format!("/{command} "));
                    AppAction::None
                }
            }
            OverlayAction::CompleteMention {
                token_start,
                token_end,
                path_end,
                is_quoted,
                candidate,
            } => {
                self.overlay.active = None;
                self.dismissed_mention = None;
                let len = self.input.text().len();
                let safe_end = token_end.min(len);
                let safe_path_end = path_end.min(safe_end);
                let suffix = self.input.text()[safe_path_end..safe_end].to_string();
                let (replacement, cursor_offset) =
                    format_mention_replacement(&candidate, &suffix, is_quoted);
                self.input.replace_byte_range_with_cursor(
                    token_start,
                    safe_end,
                    &replacement,
                    cursor_offset,
                );
                self.sync_input_completion_overlay();
                AppAction::None
            }
            OverlayAction::SelectModel { provider, model } => {
                if let Some(mut pending) = self.pending_settings_model_pick.take() {
                    let (next_p, next_m, raw_val, flash_val) = if model.trim().is_empty() {
                        (None, None, String::new(), "跟随会话模型".to_string())
                    } else if provider.trim().is_empty() {
                        (None, Some(model.clone()), model.clone(), model)
                    } else {
                        let combined = format!("{provider}/{model}");
                        (Some(provider), Some(model), combined.clone(), combined)
                    };
                    if let Some(item) = pending
                        .items
                        .iter_mut()
                        .find(|it| it.key == pending.setting_key)
                        && let SettingItemKind::Model {
                            provider: ref mut p_slot,
                            model: ref mut m_slot,
                        } = item.kind
                    {
                        *p_slot = next_p;
                        *m_slot = next_m;
                    }
                    self.overlay.active = Some(OverlayKind::SettingsModal {
                        items: pending.items,
                        query: pending.query,
                        selected: pending.selected,
                        save_path_hint: pending.save_path_hint,
                    });
                    self.set_flash(format!("✓ 已更新 {} = {flash_val}", pending.setting_key));
                    AppAction::ApplySetting {
                        key: pending.setting_key,
                        value: raw_val,
                    }
                } else {
                    self.overlay.active = None;
                    self.provider = Some(provider);
                    self.model = Some(model);
                    self.push_notice(
                        NoticeLevel::Info,
                        format!("已切换后续请求模型为：{}", self.active_model_label()),
                    );
                    AppAction::None
                }
            }
            OverlayAction::SelectEffort(effort) => {
                self.overlay.active = None;
                self.set_effort(effort);
                AppAction::None
            }
            OverlayAction::PreviewTheme(preset) => {
                self.set_theme_preset(preset);
                AppAction::None
            }
            OverlayAction::CommitTheme(preset) => {
                self.overlay.active = None;
                self.set_theme_preset(preset);
                self.set_flash(format!("✓ 主题已切换为 {}", preset.name()));
                AppAction::None
            }
            OverlayAction::RevertTheme(orig) => {
                self.overlay.active = None;
                self.set_theme_preset(orig);
                AppAction::None
            }
            OverlayAction::ApplyHistory(text) => {
                self.overlay.active = None;
                self.input.set_text_end(&text);
                AppAction::None
            }
            OverlayAction::InsertSkillSlash(skill_name) => {
                self.overlay.active = None;
                self.input.set_text_end(&format!("/{skill_name} "));
                AppAction::None
            }
            OverlayAction::SelectSession(session_id) => {
                self.overlay.active = None;
                if self.is_running_activity() {
                    self.push_notice(NoticeLevel::Warn, "请先等待或中断当前任务后再切换会话");
                    AppAction::None
                } else {
                    AppAction::ResumeSession(session_id)
                }
            }
            OverlayAction::ForkSessionAtSeq(seq) => {
                self.overlay.active = None;
                if self.is_running_activity() {
                    self.push_notice(NoticeLevel::Warn, "请先等待或中断当前任务后再分支回溯");
                    AppAction::None
                } else {
                    AppAction::ForkSessionAtSeq(seq)
                }
            }
            OverlayAction::OpenSettingModelPicker { key } => {
                if !self.providers.iter().any(|p| !p.models.is_empty()) {
                    self.push_notice(NoticeLevel::Warn, "当前没有可用的模型路由列表");
                    return AppAction::None;
                }
                if let Some(OverlayKind::SettingsModal {
                    items,
                    query,
                    selected,
                    save_path_hint,
                }) = self.overlay.active.take()
                {
                    let current_hint =
                        items
                            .iter()
                            .find(|it| it.key == key)
                            .and_then(|it| match &it.kind {
                                SettingItemKind::Model {
                                    provider: Some(p),
                                    model: Some(m),
                                } if !p.is_empty() && !m.is_empty() => Some(format!("{p}/{m}")),
                                SettingItemKind::Model { model: Some(m), .. } if !m.is_empty() => {
                                    Some(m.clone())
                                }
                                _ => None,
                            });
                    self.pending_settings_model_pick = Some(PendingSettingsModelPick {
                        setting_key: key,
                        items,
                        query,
                        selected,
                        save_path_hint,
                    });
                    self.overlay
                        .open_setting_model_picker(&self.providers, current_hint.as_deref());
                }
                AppAction::None
            }
            OverlayAction::UpdateSetting { key, value } => {
                if key.starts_with("plugin.") {
                    if self.is_running_activity() {
                        self.overlay.set_setting_toggle(&key, value != "true");
                        let msg = "当前有正在执行的任务，请等待完成后再切换模块";
                        self.push_notice(NoticeLevel::Warn, msg);
                        self.set_flash(msg);
                        return AppAction::None;
                    }
                    return AppAction::ApplySetting { key, value };
                }
                match key.as_str() {
                    "ui.theme" => {
                        if let Some(preset) = ThemePreset::parse(&value) {
                            self.set_theme_preset(preset);
                        }
                    }
                    "ui.showThinking" => {
                        self.show_thinking = value == "true";
                        self.item_cache.clear();
                    }
                    "ui.verboseExpanded" => {
                        self.verbose_expanded = value == "true";
                        self.item_cache.clear();
                    }
                    "ui.sidePanelOpen" => {
                        self.side_panel.open = self.features.side_panel && value == "true";
                        self.selection.clear();
                        self.item_cache.clear();
                    }
                    "ui.mouseCapture" => {
                        self.mouse_capture_enabled = value == "true";
                    }
                    "ui.approvalMode" => {
                        self.approval_mode = if value == "ask" {
                            ApprovalMode::Ask
                        } else {
                            ApprovalMode::Auto
                        };
                    }
                    "reasoningEffort" => {
                        if value == "default" {
                            self.reasoning_effort = None;
                        } else if let Some(eff) = parse_effort(&value) {
                            self.reasoning_effort = Some(eff);
                        }
                    }
                    _ => {}
                }
                self.set_flash(format!("✓ 已更新 {key} = {value}"));
                AppAction::ApplySetting { key, value }
            }
            OverlayAction::ResolveApproval { decision } => {
                if let Some(OverlayKind::Approval(mut ap)) = self.overlay.active.take() {
                    if decision == ApprovalDecision::AllowAlwaysForTool {
                        self.always_allowed_tools.insert(ap.tool_name.clone());
                    }
                    if let Some(tx) = ap.responder.take() {
                        let _ = tx.send(decision);
                    }
                    if !self.promote_next_approval() {
                        self.record_program_status(
                            if self.is_busy() {
                                ProgramStatusState::Working
                            } else {
                                ProgramStatusState::Idle
                            },
                            None,
                            None,
                        );
                    }
                }
                AppAction::None
            }
        };
        if matches!(out, AppAction::None) && self.overlay.active.is_none() {
            let _ = self.promote_next_approval();
        }
        out
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> AppAction {
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return AppAction::None;
        }

        // 1. 全屏轨迹场景优先接管全部键盘输入
        if self.features.trajectory_view && self.trajectory_scene.open {
            return match self.trajectory_scene.handle_key(key) {
                TrajectoryKeyOutcome::ReloadReport => {
                    AppAction::OpenTrajectoryScene(self.trajectory_scene.mode)
                }
                TrajectoryKeyOutcome::Consumed | TrajectoryKeyOutcome::Closed => AppAction::None,
            };
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        // 2. Ctrl+C：若有活跃鼠标选区，优先复制选区并保持界面；队列选择器打开时只收起它；
        // 模态弹窗则先关闭/拒绝；
        // 非空输入优先由单次 Ctrl+C 清空（含补全浮层与运行中草稿）；否则走中断或双击退出
        if matches!(key.code, KeyCode::Char('c')) && ctrl {
            if self.features.selection
                && self.selection.is_active()
                && let Some(buf) = &self.last_buffer
                && let Some(copied) =
                    extract_selected_text(buf, &self.selection, &self.selection_geom)
            {
                let _ = copy_to_clipboard(&copied);
                let chars = copied.chars().count();
                self.set_flash(format!("✓ 已复制 {chars} 字符"));
                return AppAction::None;
            }
            // 选择器打开时 Ctrl+C 只收起选择器，不能误触成 CancelRun / 退出。
            if self.queue.selected_id().is_some() {
                self.queue.clear_selection();
                return AppAction::None;
            }
            if self.overlay.is_modal() {
                if let Some(OverlayKind::Approval(_)) = self.overlay.active {
                    return self.apply_overlay_action(OverlayAction::ResolveApproval {
                        decision: ApprovalDecision::Deny,
                    });
                }
                if let Some(OverlayKind::ThemePicker { original, .. }) = self.overlay.active {
                    return self.apply_overlay_action(OverlayAction::RevertTheme(original));
                }
                self.pending_settings_model_pick = None;
                return self.apply_overlay_action(OverlayAction::Close);
            }
            if !self.input.is_empty() {
                self.input.clear();
                self.sync_input_completion_overlay();
                self.last_ctrl_c_at = None;
                return AppAction::None;
            }
            if self.is_running_activity() {
                return AppAction::CancelRun;
            }
            let double_press = self
                .last_ctrl_c_at
                .is_some_and(|t| t.elapsed() < Duration::from_millis(1200));
            if double_press || self.items.is_empty() {
                self.should_quit = true;
                return AppAction::Quit;
            }
            self.last_ctrl_c_at = Some(Instant::now());
            self.set_flash("再按一次 Ctrl+C 退出");
            return AppAction::None;
        }

        // 3. 浮层栈（模态与补全）按键分派
        if let Some(action) =
            self.overlay
                .handle_key(key, &self.user_skill_names, self.input.history())
        {
            return self.apply_overlay_action(action);
        }

        // 3.5 队列选择器：浮层之后、全局快捷键之前，才不会被补全或历史回溯劫走
        if self.queue.selected_id().is_some()
            && let Some(action) = self.handle_queue_selector_key(key, ctrl, alt, shift)
        {
            return action;
        }

        // 4. 全局功能快捷键
        match key.code {
            KeyCode::Char('d') if ctrl => {
                if !self.is_running_activity() && self.input.is_empty() {
                    self.should_quit = true;
                    return AppAction::Quit;
                }
                return AppAction::None;
            }
            KeyCode::Char('t') if ctrl => {
                if !self.features.trajectory_view {
                    return AppAction::None;
                }
                // 把 Ctrl+T 设为打开全屏轨迹（对齐 dsh-TUI），Ctrl+O 开关思考/详情展开
                return AppAction::OpenTrajectoryScene(self.trajectory_scene.mode);
            }
            KeyCode::Char('o') if ctrl => {
                let next = !self.verbose_expanded;
                self.verbose_expanded = next;
                self.show_thinking = next;
                self.item_cache.clear();
                let state = if next { "展开" } else { "收起" };
                self.set_flash(format!("思考与工具详情已{state}"));
                return AppAction::None;
            }
            KeyCode::Char('b') if ctrl => {
                if !self.features.side_panel {
                    return AppAction::None;
                }
                self.side_panel.open = !self.side_panel.open;
                self.selection.clear();
                self.item_cache.clear();
                if self.side_panel.open {
                    return AppAction::RefreshSidePanelData;
                }
                return AppAction::None;
            }
            KeyCode::Char('r') if ctrl => {
                self.overlay.open_history_search(self.input.history());
                return AppAction::None;
            }
            KeyCode::Char('e') if ctrl => {
                if self.input.expand_pastes_in_place() {
                    self.set_flash("已展开折叠粘贴块");
                }
                return AppAction::None;
            }
            KeyCode::Char('z') if ctrl => {
                self.input.undo();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Char('y') if ctrl => {
                if self.input.yank() {
                    self.sync_input_completion_overlay();
                }
                return AppAction::None;
            }
            KeyCode::Char('g') if ctrl => {
                if !self.is_running_activity() {
                    return AppAction::OpenExternalEditor;
                }
                return AppAction::None;
            }
            KeyCode::F(2) => {
                self.mouse_capture_enabled = !self.mouse_capture_enabled;
                return AppAction::SetMouseCapture(self.mouse_capture_enabled);
            }
            KeyCode::F(3) => {
                let next = match self.screen_mode {
                    ScreenMode::Fullscreen => ScreenMode::Inline,
                    ScreenMode::Inline => ScreenMode::Fullscreen,
                };
                return AppAction::SwitchScreenMode(next);
            }
            KeyCode::Char('w') if ctrl => {
                self.input.delete_word_backward();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Char('k') if ctrl => {
                self.input.delete_to_line_end();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Char('l') if ctrl => {
                return self.open_model_selector();
            }
            KeyCode::Char('v' | 'V') if ctrl && !alt => {
                if !self.features.clipboard {
                    self.set_flash("剪贴板扩展已禁用");
                    return AppAction::None;
                }
                return AppAction::PasteClipboard;
            }
            KeyCode::Esc => {
                if self.selection.is_active() {
                    self.selection.clear();
                    return AppAction::None;
                }
                if self.is_running_activity() {
                    return AppAction::CancelRun;
                }
                if !self.input.is_empty() {
                    self.input.clear();
                    self.sync_input_completion_overlay();
                }
                return AppAction::None;
            }
            KeyCode::Char('?') if !ctrl && !alt && self.input.is_empty() => {
                self.overlay.open_help();
                return AppAction::None;
            }
            KeyCode::PageUp => {
                self.scroll_up(self.page_step());
                return AppAction::None;
            }
            KeyCode::PageDown => {
                self.scroll_down(self.page_step());
                return AppAction::None;
            }
            KeyCode::Char('u') if ctrl => {
                self.scroll_up(self.page_step());
                return AppAction::None;
            }
            KeyCode::Tab if !ctrl && !alt => {
                self.cycle_reasoning_effort();
                return AppAction::None;
            }
            KeyCode::Enter if ctrl && !alt && !shift => return self.prepare_submit(true),
            KeyCode::Enter if alt && !ctrl && !shift => {
                // 草稿空时 Alt+Enter 原来的语义（往空草稿里塞换行）没有意义，
                // 正好当作不依赖 Kitty 键盘协议的立即发送备用入口。
                if self.input.is_empty() && !self.queue.is_empty() {
                    return self.prepare_submit(true);
                }
                self.input.insert_char('\n');
                return AppAction::None;
            }
            KeyCode::Enter if shift || alt || ctrl => {
                self.input.insert_char('\n');
                return AppAction::None;
            }
            KeyCode::Char('j') if ctrl => {
                self.input.insert_char('\n');
                return AppAction::None;
            }
            KeyCode::Enter => return self.prepare_submit(false),
            KeyCode::Backspace => {
                self.input.backspace();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Delete => {
                self.input.delete();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Left if alt || ctrl => {
                self.input.move_word_left();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Right if alt || ctrl => {
                self.input.move_word_right();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Left => {
                self.input.move_left();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Right => {
                self.input.move_right();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Up if alt && !ctrl && !shift => {
                // Alt+↑：一步把队尾撤回输入框直接编辑，不用先进选择器再按 ⏎。
                // 草稿非空时与草稿交换（草稿进队尾），不会丢字。
                //
                // 终端兼容性：kitty / xterm modifyOtherKeys / Ghostty / iTerm2 发的是
                // CSI 1;3A，解析为 Up+ALT，走这条路；少数开启 altSendsEscape 的终端
                // 发的是 ESC ESC [ A，crossterm 会拆成 Esc + 字面 "[A"，
                // 那种终端请用 ↑ 进选择器（实测见 issue #15）。
                if let Some(last) = self.queue.items().last().map(|item| item.id)
                    && !self.is_bash_running()
                {
                    self.retract_queued(last);
                    return AppAction::None;
                }
                self.input.move_up();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Up => {
                // 草稿空且队列非空时，↑ 进入队列选择器（优先于历史回溯）。
                if self.input.is_empty()
                    && !self.overlay.is_open()
                    && !self.is_bash_running()
                    && self.queue.open_selector_at_tail()
                {
                    return AppAction::None;
                }
                self.input.move_up();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Down => {
                self.input.move_down();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Home if ctrl => {
                self.auto_scroll = false;
                self.scroll_from_top = 0;
                return AppAction::None;
            }
            KeyCode::End if ctrl => {
                self.scroll_to_bottom();
                return AppAction::None;
            }
            KeyCode::Home => {
                self.input.move_home();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::End => {
                self.input.move_end();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Char(ch) if !ctrl && !alt => {
                self.input.insert_char(ch);
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            _ => {}
        }
        AppAction::None
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) -> AppAction {
        if self.features.trajectory_view && self.trajectory_scene.open {
            return match self.trajectory_scene.handle_mouse(mouse) {
                TrajectoryKeyOutcome::ReloadReport => {
                    AppAction::OpenTrajectoryScene(self.trajectory_scene.mode)
                }
                _ => AppAction::None,
            };
        }

        if let Some(action) = self.overlay.handle_mouse(mouse, &self.user_skill_names) {
            return self.apply_overlay_action(action);
        }

        let col = mouse.column;
        let row = mouse.row;
        let total_w = self
            .last_buffer
            .as_ref()
            .map(|b| b.area.width)
            .unwrap_or(120);

        match mouse.kind {
            MouseEventKind::Moved => {
                if self.features.side_panel {
                    self.side_panel.hovered_divider = self.side_panel.divider_col == Some(col);
                }
                if self.features.timeline_rail {
                    self.hovered_rail_row = self
                        .rail_geo
                        .as_ref()
                        .and_then(|g| hit_test_rail(g, col, row).map(|_| row));
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                if self.features.side_panel {
                    // 1. 侧栏分割线拖拽
                    if self.side_panel.divider_col == Some(col) {
                        self.side_panel.dragging_divider = true;
                        return AppAction::None;
                    }
                    // 2. 侧栏页签点击
                    for (tab, rect) in &self.side_panel.tab_hit_boxes {
                        if col >= rect.x
                            && col < rect.x + rect.width
                            && row >= rect.y
                            && row < rect.y + rect.height
                        {
                            self.side_panel.tab = *tab;
                            self.side_panel.scroll = 0;
                            return AppAction::RefreshSidePanelData;
                        }
                    }
                }
                // 3. 时间轴导轨点击跳转
                if self.features.timeline_rail
                    && let Some(geo) = &self.rail_geo
                    && let Some(hit) = hit_test_rail(geo, col, row)
                {
                    let target_idx = match hit {
                        RailHit::Up => self.timeline_snap.up_idx,
                        RailHit::Down => self.timeline_snap.down_idx,
                        RailHit::Tick(idx) => Some(idx),
                    };
                    if let Some(idx) = target_idx
                        && let Some(turn) = self.timeline_snap.turns.get(idx)
                    {
                        self.auto_scroll = false;
                        self.scroll_from_top = turn.top_line.min(u16::MAX as usize) as u16;
                    }
                    return AppAction::None;
                }
                // 4. 启动应用级文本选区
                if self.features.selection {
                    self.selection.start_press(
                        col,
                        row,
                        &self.selection_geom,
                        self.last_buffer.as_ref(),
                    );
                }
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if self.features.side_panel && self.side_panel.dragging_divider {
                    self.side_panel.set_width_from_divider_col(total_w, col);
                    return AppAction::None;
                }
                if self.features.selection {
                    self.selection
                        .update_drag(col, row, self.last_buffer.as_ref());
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if self.features.side_panel && self.side_panel.dragging_divider {
                    self.side_panel.dragging_divider = false;
                    return AppAction::None;
                }
                if self.features.selection {
                    let had_drag_selection = self.selection.finish_release();
                    if had_drag_selection {
                        if let Some(buf) = &self.last_buffer
                            && let Some(text) =
                                extract_selected_text(buf, &self.selection, &self.selection_geom)
                        {
                            let _ = copy_to_clipboard(&text);
                            self.set_flash(format!("✓ 已复制 {} 字符", text.chars().count()));
                        }
                        return AppAction::None;
                    }
                }
                // 单击未拖动：检查是否点中 Launchpad 快捷入口
                for (qa, rect) in &self.launchpad_hit_rects {
                    if col >= rect.x
                        && col < rect.x + rect.width
                        && row >= rect.y
                        && row < rect.y + rect.height
                    {
                        return match qa.clone() {
                            LaunchpadQuickAction::ModelPicker => self.open_model_selector(),
                            LaunchpadQuickAction::ThemePicker => {
                                self.overlay.open_theme_picker(self.theme_preset);
                                AppAction::None
                            }
                            LaunchpadQuickAction::SkillPicker => AppAction::OpenSkillPicker,
                            LaunchpadQuickAction::TrajectoryScene => {
                                if !self.features.trajectory_view {
                                    return AppAction::None;
                                }
                                AppAction::OpenTrajectoryScene(self.trajectory_scene.mode)
                            }
                            LaunchpadQuickAction::ToggleSidePanel => {
                                if !self.features.side_panel {
                                    return AppAction::None;
                                }
                                self.side_panel.open = !self.side_panel.open;
                                self.item_cache.clear();
                                if self.side_panel.open {
                                    AppAction::RefreshSidePanelData
                                } else {
                                    AppAction::None
                                }
                            }
                        };
                    }
                }
            }
            MouseEventKind::ScrollUp => {
                if self.features.side_panel
                    && let Some(sp_rect) = self.selection_geom.side_panel_rect
                    && col >= sp_rect.x
                {
                    self.side_panel.scroll = self.side_panel.scroll.saturating_sub(2);
                } else {
                    self.scroll_up(3);
                }
            }
            MouseEventKind::ScrollDown => {
                if self.features.side_panel
                    && let Some(sp_rect) = self.selection_geom.side_panel_rect
                    && col >= sp_rect.x
                {
                    self.side_panel.scroll = self.side_panel.scroll.saturating_add(2);
                } else {
                    self.scroll_down(3);
                }
            }
            _ => {}
        }
        AppAction::None
    }

    pub fn handle_paste(&mut self, data: &str) {
        self.input.insert_paste(data);
        self.sync_input_completion_overlay();
    }

    /// 队列选择器的按键处理：↑↓ 移动，⏎ 撤回，Ctrl/Alt+⏎ 撤回并立即发送，Del 丢弃，Esc 关闭。
    fn handle_queue_selector_key(
        &mut self,
        key: KeyEvent,
        ctrl: bool,
        alt: bool,
        shift: bool,
    ) -> Option<AppAction> {
        let selected = self.queue.selected_id()?;
        match key.code {
            KeyCode::Up if !ctrl && !alt && !shift => {
                if !self.queue.move_selection_up() {
                    // 越过首项就交出选择器，把这次按键还给历史回溯。
                    self.input.move_up();
                    self.sync_input_completion_overlay();
                }
                Some(AppAction::None)
            }
            KeyCode::Down if !ctrl && !alt && !shift => {
                if !self.queue.move_selection_down() {
                    self.input.move_down();
                    self.sync_input_completion_overlay();
                }
                Some(AppAction::None)
            }
            KeyCode::Enter if ctrl || alt => {
                let item = self.dequeue_prompt(selected)?;
                if !item.images.is_empty() && !self.active_model_supports_images() {
                    self.queue.clear_selection();
                    self.input
                        .replace_draft_from_queue_with_images(&item.text, &item.images);
                    self.reject_unsupported_images_notice();
                    return Some(AppAction::None);
                }
                self.set_flash("⚡ 已撤回队列项并立即发送…");
                if item.images.is_empty() {
                    Some(AppAction::SteerPrompt(item.text))
                } else {
                    Some(AppAction::SteerPromptWithImages {
                        text: item.text,
                        images: item.images.into_iter().map(|(_, img)| img).collect(),
                    })
                }
            }
            KeyCode::Enter if !shift => {
                self.retract_queued(selected);
                Some(AppAction::None)
            }
            KeyCode::Delete if !ctrl && !alt => {
                self.discard_queued(selected);
                Some(AppAction::None)
            }
            KeyCode::Char('x') if ctrl => {
                self.discard_queued(selected);
                Some(AppAction::None)
            }
            KeyCode::Esc => {
                self.queue.clear_selection();
                Some(AppAction::None)
            }
            // 其他键（含普通字符）交还给正常编辑：先收起选择器，再让按键继续下落。
            _ => {
                self.queue.clear_selection();
                None
            }
        }
    }

    fn prepare_submit(&mut self, immediate: bool) -> AppAction {
        let preview = self.input.text().trim();
        if preview.is_empty() {
            self.input.take();
            // 草稿为空时队列才是「主人现在想发的东西」：立即发送插队发队首，
            // 普通回车在空闲时直接发队首，运行中则保留排队等本轮收尾。
            if self.queue.selected_id().is_some() {
                return AppAction::None;
            }
            if !self.queue.is_empty() {
                if immediate {
                    return self.steer_queue_head();
                }
                if !self.is_running_activity() {
                    return self.submit_queue_head();
                }
            }
            return AppAction::None;
        }

        if self.features.local_bash
            && let Some(mode) = BashInputMode::from_input(preview)
        {
            let expanded = self.input.expand_text();
            let cmd = mode.extract_command(&expanded);
            if cmd.is_empty() {
                self.set_flash("请输入要在 Bash 中执行的命令（或按 Esc 退出）");
                return AppAction::None;
            }
            if self.is_busy() {
                self.set_flash("模型正在生成中，请等待回合结束或按 Esc 取消后再运行 Bash 命令");
                return AppAction::None;
            }
            if self.is_bash_running() {
                self.set_flash("已有 Bash 命令正在运行，请先按 Esc 取消");
                return AppAction::None;
            }
            if self.is_compacting() {
                self.set_flash("上下文正在压缩中，请等待完成或按 Esc 取消后再运行 Bash 命令");
                return AppAction::None;
            }

            let raw = self.input.take();
            let trimmed = raw.trim();
            let command = mode.extract_command(trimmed).to_string();
            self.overlay.active = None;
            self.dismissed_mention = None;
            self.input.push_history(trimmed);
            self.scroll_to_bottom();
            return AppAction::RunLocalBash {
                command,
                exclude_from_context: mode.exclude_from_context(),
            };
        }

        if self.is_bash_running()
            || matches!(self.activity, super::ActivityState::Compacting { .. })
        {
            let is_safe_slash = preview
                .strip_prefix('/')
                .and_then(|s| s.split_whitespace().next())
                .is_some_and(|c| {
                    matches!(
                        c.to_ascii_lowercase().as_str(),
                        "exit" | "quit" | "q" | "help" | "h" | "view" | "mouse" | "session"
                    )
                });
            if !is_safe_slash {
                let hint = if self.is_bash_running() {
                    "Bash 命令正在运行，请等待完成或按 Esc 取消后再发送"
                } else {
                    "上下文正在压缩中，请等待完成或按 Esc 取消后再发送"
                };
                self.set_flash(hint);
                return AppAction::None;
            }
        }

        let is_builtin_cmd = preview
            .strip_prefix('/')
            .map(str::trim)
            .and_then(|s| s.split_whitespace().next())
            .is_some_and(|first| is_builtin_slash_cmd(&first.to_ascii_lowercase()));

        if !is_builtin_cmd && self.input.has_images() {
            if self.input.image_count() > MAX_IMAGES_PER_MESSAGE {
                let msg = format!(
                    "单条消息最多附加 {MAX_IMAGES_PER_MESSAGE} 张图片，请先删减多余图片后再发送"
                );
                self.set_flash(msg.clone());
                self.push_notice(NoticeLevel::Warn, msg);
                return AppAction::None;
            }
            if !self.active_model_supports_images() {
                self.reject_unsupported_images_notice();
                return AppAction::None;
            }
        }

        let (raw, images) = self.input.take_with_images();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return AppAction::None;
        }
        self.overlay.active = None;
        self.dismissed_mention = None;
        self.input.push_history(trimmed);
        self.scroll_to_bottom();

        if let Some(cmd_line) = trimmed.strip_prefix('/') {
            let cmd_line = cmd_line.trim();
            let first = cmd_line.split_whitespace().next().unwrap_or("");
            // 对齐 dsh-TUI / pi：仅拦截已注册内置命令；
            // 技能手势（/<skill>）、绝对路径（/Users/...）以及未知 / 前缀一律作为普通消息发给模型。
            if is_builtin_slash_cmd(&first.to_ascii_lowercase()) {
                return self.handle_slash_command(cmd_line);
            }
        }

        let raw_images: Vec<ImageAttachment> = images.iter().map(|(_, img)| img.clone()).collect();
        if self.is_busy() {
            let prompt = trimmed.to_string();
            if immediate {
                self.set_flash("⚡ 正在中断当前回合并立即发送…");
                if raw_images.is_empty() {
                    return AppAction::SteerPrompt(prompt);
                }
                return AppAction::SteerPromptWithImages {
                    text: prompt,
                    images: raw_images,
                };
            }
            self.enqueue_prompt_with_images(prompt, images);
            return AppAction::None;
        }

        self.push_user_message(trimmed);
        if raw_images.is_empty() {
            AppAction::SubmitPrompt(trimmed.to_string())
        } else {
            AppAction::SubmitPromptWithImages {
                text: trimmed.to_string(),
                images: raw_images,
            }
        }
    }

    fn handle_slash_command(&mut self, cmd_line: &str) -> AppAction {
        let mut parts = cmd_line.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
        let arg = parts.next().map(str::trim).filter(|s| !s.is_empty());

        match cmd.as_str() {
            "exit" | "quit" | "q" => {
                self.should_quit = true;
                AppAction::Quit
            }
            "help" | "h" => {
                self.overlay.open_help();
                AppAction::None
            }
            "settings" | "config" => AppAction::OpenSettings(arg.map(ToString::to_string)),
            "tools" => AppAction::ShowTools,
            "mcp" => AppAction::ShowMcp,
            "skills" => match arg {
                Some("list") => AppAction::ShowSkills,
                _ => AppAction::OpenSkillPicker,
            },
            "instructions" => AppAction::ShowInstructions,
            "tree" => AppAction::OpenSessionTree,
            "view" => {
                let next = match arg {
                    Some("inline") => ScreenMode::Inline,
                    Some("fullscreen" | "full") => ScreenMode::Fullscreen,
                    None => match self.screen_mode {
                        ScreenMode::Fullscreen => ScreenMode::Inline,
                        ScreenMode::Inline => ScreenMode::Fullscreen,
                    },
                    Some(other) => {
                        self.push_notice(
                            NoticeLevel::Warn,
                            format!("未知视图模式 `{other}`，可选：fullscreen / inline"),
                        );
                        return AppAction::None;
                    }
                };
                AppAction::SwitchScreenMode(next)
            }
            "mouse" => {
                let next = match arg {
                    Some("on" | "true") => true,
                    Some("off" | "false") => false,
                    _ => !self.mouse_capture_enabled,
                };
                self.mouse_capture_enabled = next;
                AppAction::SetMouseCapture(next)
            }
            "trace" => {
                if !self.features.trajectory_view {
                    return AppAction::None;
                }
                AppAction::OpenTrajectoryScene(self.trajectory_scene.mode)
            }
            "panel" => {
                if !self.features.side_panel {
                    return AppAction::None;
                }
                self.side_panel.open = !self.side_panel.open;
                self.selection.clear();
                self.item_cache.clear();
                if self.side_panel.open {
                    AppAction::RefreshSidePanelData
                } else {
                    AppAction::None
                }
            }
            "theme" => match arg {
                None => {
                    self.overlay.open_theme_picker(self.theme_preset);
                    AppAction::None
                }
                Some(name) => {
                    if let Some(preset) = ThemePreset::parse(name) {
                        self.set_theme_preset(preset);
                        self.set_flash(format!("✓ 主题已切换为 {}", preset.name()));
                    } else {
                        self.push_notice(
                            NoticeLevel::Warn,
                            format!(
                                "未知主题 `{name}`，可选：auto / mist-dark / mist-light / sakura-dark / system / ansi16"
                            ),
                        );
                    }
                    AppAction::None
                }
            },
            "approval" => match arg {
                None => {
                    self.push_notice(
                        NoticeLevel::Info,
                        format!(
                            "当前工具审批模式：{}（可用 `/approval auto` 或 `/approval ask` 切换）",
                            self.approval_mode.as_str()
                        ),
                    );
                    AppAction::None
                }
                Some("auto" | "off") => {
                    self.approval_mode = ApprovalMode::Auto;
                    self.push_notice(NoticeLevel::Info, "工具审批模式已设为：auto（自动放行）");
                    AppAction::None
                }
                Some("ask" | "on") => {
                    self.approval_mode = ApprovalMode::Ask;
                    self.push_notice(
                        NoticeLevel::Info,
                        "工具审批模式已设为：ask（写文件与 Shell 将弹窗确认）",
                    );
                    AppAction::None
                }
                Some(other) => {
                    self.push_notice(
                        NoticeLevel::Warn,
                        format!("未知审批模式 `{other}`，可选：auto / ask"),
                    );
                    AppAction::None
                }
            },
            "clear" | "new" => {
                if self.is_running_activity() {
                    self.push_notice(NoticeLevel::Warn, "请先按 Esc 中断当前任务再开启新会话");
                    AppAction::None
                } else {
                    AppAction::NewSession
                }
            }
            "thinking" => {
                self.show_thinking = !self.show_thinking;
                self.item_cache.clear();
                let state = if self.show_thinking {
                    "开启"
                } else {
                    "隐藏"
                };
                self.push_notice(NoticeLevel::Info, format!("思考过程显示已{state}"));
                AppAction::None
            }
            "model" => match arg {
                None => self.open_model_selector(),
                other => AppAction::ShowOrSwitchModel(other.map(ToString::to_string)),
            },
            "effort" => match arg {
                None => {
                    self.overlay.open_effort_picker(self.reasoning_effort);
                    AppAction::None
                }
                Some(raw_effort) => {
                    if let Some(parsed) = parse_effort(raw_effort) {
                        self.set_effort(parsed);
                    } else {
                        self.push_notice(
                            NoticeLevel::Warn,
                            format!(
                                "未知推理强度 `{raw_effort}`，可选：off / minimal / low / medium / high / xhigh / max"
                            ),
                        );
                    }
                    AppAction::None
                }
            },
            "stats" => AppAction::ShowStats,
            "session" => AppAction::ShowSessionInfo,
            "compact" => {
                if self.is_running_activity() {
                    self.push_notice(
                        NoticeLevel::Warn,
                        "当前已有任务或压缩正在运行，请先等待完成或按 Esc 取消后再执行 /compact",
                    );
                    AppAction::None
                } else {
                    AppAction::CompactSession(arg.map(ToString::to_string))
                }
            }
            "resume" => {
                if self.is_running_activity() {
                    self.push_notice(
                        NoticeLevel::Warn,
                        "当前已有任务正在运行，请等待完成或按 Esc 取消后再切换会话",
                    );
                    AppAction::None
                } else {
                    match arg.map(str::trim).filter(|s| !s.is_empty()) {
                        Some(session_id) => AppAction::ResumeSession(session_id.to_string()),
                        None => AppAction::OpenSessionPicker,
                    }
                }
            }
            "rename" => AppAction::RenameSession(
                arg.map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string),
            ),
            "export" => AppAction::ExportTrajectory(arg.map(PathBuf::from)),
            _ => AppAction::None,
        }
    }
}

fn is_builtin_slash_cmd(cmd: &str) -> bool {
    matches!(
        cmd,
        "exit"
            | "quit"
            | "q"
            | "help"
            | "h"
            | "resume"
            | "rename"
            | "settings"
            | "config"
            | "tools"
            | "mcp"
            | "skills"
            | "instructions"
            | "clear"
            | "new"
            | "thinking"
            | "model"
            | "effort"
            | "stats"
            | "session"
            | "compact"
            | "export"
            | "trace"
            | "tree"
            | "view"
            | "mouse"
            | "panel"
            | "theme"
            | "approval"
    )
}

fn parse_effort(raw: &str) -> Option<ReasoningEffort> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "off" => Some(ReasoningEffort::Off),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}
