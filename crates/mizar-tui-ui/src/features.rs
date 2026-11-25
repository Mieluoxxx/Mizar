//! TUI 插件化特性开关：由宿主的 `HostExtensionFlags` 派生，控制各前端交互与渲染通道的挂载。

use mizar_client_core::HostExtensionFlags;

/// 控制 `mizar-tui-ui` 可选前端扩展的运行时开关集合。
///
/// 所有字段默认全 `true`（全功能交互模式），关闭任一开关时对应的快捷键、斜杠命令、
/// 鼠标命中与渲染分区都会安全跳过，不会影响主聊天输入与事件循环。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TuiFeatures {
    pub emphasis: bool,
    pub side_panel: bool,
    pub timeline_rail: bool,
    pub trajectory_view: bool,
    pub selection: bool,
    pub clipboard: bool,
    pub local_bash: bool,
}

impl Default for TuiFeatures {
    fn default() -> Self {
        Self {
            emphasis: true,
            side_panel: true,
            timeline_rail: true,
            trajectory_view: true,
            selection: true,
            clipboard: true,
            local_bash: true,
        }
    }
}

impl From<&HostExtensionFlags> for TuiFeatures {
    fn from(flags: &HostExtensionFlags) -> Self {
        Self {
            emphasis: flags.tui && flags.tui_emphasis,
            side_panel: flags.tui && flags.tui_side_panel,
            timeline_rail: flags.tui && flags.tui_timeline_rail,
            trajectory_view: flags.tui && flags.tui_trajectory_view,
            selection: flags.tui && flags.tui_selection,
            clipboard: flags.tui && flags.tui_clipboard,
            local_bash: flags.local_bash,
        }
    }
}

impl From<HostExtensionFlags> for TuiFeatures {
    fn from(flags: HostExtensionFlags) -> Self {
        Self::from(&flags)
    }
}

#[cfg(test)]
mod tests {
    use mizar_client_core::{
        PLUGIN_LOCAL_BASH, PLUGIN_TUI, PLUGIN_TUI_CLIPBOARD, PLUGIN_TUI_SIDE_PANEL,
    };

    use super::*;

    #[test]
    fn maps_host_extension_flags_and_tui_cascade_accurately() {
        assert_eq!(
            TuiFeatures::from(HostExtensionFlags::default()),
            TuiFeatures::default()
        );

        let tui_disabled = HostExtensionFlags::from_disabled_plugins(&[PLUGIN_TUI.to_string()]);
        let features = TuiFeatures::from(&tui_disabled);
        assert!(!features.emphasis);
        assert!(!features.side_panel);
        assert!(!features.timeline_rail);
        assert!(!features.trajectory_view);
        assert!(!features.selection);
        assert!(!features.clipboard);
        assert!(features.local_bash);

        let selective = HostExtensionFlags::from_disabled_plugins(&[
            PLUGIN_TUI_SIDE_PANEL.to_string(),
            PLUGIN_TUI_CLIPBOARD.to_string(),
            PLUGIN_LOCAL_BASH.to_string(),
        ]);
        let selective_features = TuiFeatures::from(selective);
        assert!(selective_features.emphasis);
        assert!(!selective_features.side_panel);
        assert!(selective_features.timeline_rail);
        assert!(selective_features.trajectory_view);
        assert!(selective_features.selection);
        assert!(!selective_features.clipboard);
        assert!(!selective_features.local_bash);
    }
}
