//! 跨平台系统剪贴板与 SSH 下 OSC 5522 终端剪贴板读取器。
//!
//! 优先级严格遵循 `files → image → text`：
//! - macOS：`osascript -e 'clipboard info'` 探测 `«class furl»` → `«class PNGf»` / `«class TIFF»`（安全临时目录 `0700` + `0600` 文件，TIFF 经 `sips` 转 PNG）→ `pbpaste`
//! - Linux：`wl-paste --list-types` → `xclip -selection clipboard -t TARGETS -o` → `xsel --clipboard --output`
//! - SSH 远端（`SSH_CONNECTION` / `SSH_CLIENT` / `SSH_TTY`）：走 OSC 5522 协议（`.` 探针 + 分片 `DATA`/`DONE` 重组）

use std::future::Future;
use std::io::{self, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(all(unix, any(target_os = "macos", test)))]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use thiserror::Error;
use tokio::process::Command;

const NATIVE_CMD_TIMEOUT: Duration = Duration::from_secs(4);
const OSC5522_TIMEOUT: Duration = Duration::from_millis(1200);
const OSC5522_MAX_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipboardContent {
    Files(Vec<PathBuf>),
    Image {
        bytes: Vec<u8>,
        source_label: String,
    },
    Text(String),
}

#[derive(Debug, Error)]
pub enum ClipboardError {
    #[error("剪贴板为空，或当前内容既不是受支持的图片也不是文本")]
    Empty,
    #[error("无法访问系统剪贴板：{0}")]
    ToolUnavailable(String),
    #[error("读取剪贴板失败：{0}")]
    CommandFailed(String),
    #[error("终端拒绝了 OSC 5522 剪贴板读取权限")]
    Osc5522Denied,
    #[error("当前终端不支持或未响应 OSC 5522 剪贴板协议：{0}")]
    Osc5522Unsupported(String),
    #[error("剪贴板读取发生 I/O 错误：{0}")]
    Io(#[from] io::Error),
}

/// 判断是否处于原生剪贴板不可直接触达的 SSH 远端会话中。
pub fn native_unreachable() -> bool {
    native_unreachable_with(|key| std::env::var_os(key).is_some_and(|v| !v.is_empty()))
}

pub fn native_unreachable_with<F>(has_env: F) -> bool
where
    F: Fn(&str) -> bool,
{
    has_env("SSH_CONNECTION") || has_env("SSH_CLIENT") || has_env("SSH_TTY")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MimeChoice {
    Files(String),
    Image(String),
    Text(String),
}

/// 按 `files → image → text` 严格优先级从 MIME 列表选取目标类型。
pub fn select_preferred_mime(types: &[String]) -> Option<MimeChoice> {
    let trimmed: Vec<&str> = types
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if trimmed.is_empty() {
        return None;
    }

    // 1. 文件 URI 列表优先（复制图片/代码文件优先识别为文件）
    for candidate in [
        "text/uri-list",
        "x-special/gnome-copied-files",
        "application/vnd.portal.files",
    ] {
        if let Some(found) = trimmed.iter().find(|t| t.eq_ignore_ascii_case(candidate)) {
            return Some(MimeChoice::Files((*found).to_string()));
        }
    }

    // 2. 图片（优先直接无损 / 标准 Web 编码格式，避免优先拿到 BMP/TIFF 等包装）
    for preferred in [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/gif",
    ] {
        if let Some(found) = trimmed.iter().find(|t| t.eq_ignore_ascii_case(preferred)) {
            return Some(MimeChoice::Image((*found).to_string()));
        }
    }
    if let Some(found) = trimmed.iter().find(|t| {
        let lower = t.to_ascii_lowercase();
        lower.starts_with("image/") && !lower.contains("svg")
    }) {
        return Some(MimeChoice::Image((*found).to_string()));
    }

    // 3. 文本回退
    for preferred in [
        "text/plain;charset=utf-8",
        "text/plain; charset=utf-8",
        "UTF8_STRING",
        "text/plain",
        "STRING",
        "TEXT",
    ] {
        if let Some(found) = trimmed.iter().find(|t| t.eq_ignore_ascii_case(preferred)) {
            return Some(MimeChoice::Text((*found).to_string()));
        }
    }
    if let Some(found) = trimmed
        .iter()
        .find(|t| t.to_ascii_lowercase().starts_with("text/plain"))
    {
        return Some(MimeChoice::Text((*found).to_string()));
    }

    None
}

/// 解析 `text/uri-list` 或 `x-special/gnome-copied-files` 负载中的本地文件路径。
pub fn decode_file_uri_list(raw: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed == "copy" || trimmed == "cut" {
            continue;
        }
        if let Some(path) = parse_file_uri_or_path(trimmed) {
            paths.push(path);
        }
    }
    paths
}

fn parse_file_uri_or_path(s: &str) -> Option<PathBuf> {
    if let Some(rest) = s.strip_prefix("file://") {
        let path_part = if rest.starts_with('/') {
            rest
        } else if rest.starts_with("localhost/") {
            &rest["localhost".len()..]
        } else {
            return None;
        };
        let decoded = percent_decode_utf8(path_part);
        if decoded.is_empty() {
            None
        } else {
            Some(PathBuf::from(decoded))
        }
    } else if s.starts_with('/') {
        Some(PathBuf::from(s))
    } else {
        None
    }
}

fn percent_decode_utf8(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push(((h << 4) | l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MacClipboardInfo {
    pub has_files: bool,
    pub has_png: bool,
    pub has_tiff: bool,
    pub has_jpeg: bool,
    pub has_gif: bool,
    pub has_webp: bool,
}

/// 解析 `osascript -e 'clipboard info'` 的输出。
pub fn parse_macos_clipboard_info(info: &str) -> MacClipboardInfo {
    MacClipboardInfo {
        has_files: info.contains("«class furl»") || info.contains("public.file-url"),
        has_png: info.contains("«class PNGf»") || info.contains("public.png"),
        has_tiff: info.contains("«class TIFF»") || info.contains("public.tiff"),
        has_jpeg: info.contains("JPEG picture")
            || info.contains("«class JPEG»")
            || info.contains("public.jpeg"),
        has_gif: info.contains("GIF picture")
            || info.contains("«class GIFf»")
            || info.contains("com.compuserve.gif"),
        has_webp: info.contains("org.webmproject.webp"),
    }
}

#[cfg(all(unix, any(target_os = "macos", test)))]
struct TempClipboardDir {
    path: PathBuf,
}

#[cfg(all(unix, any(target_os = "macos", test)))]
impl TempClipboardDir {
    fn create(prefix: &str) -> io::Result<Self> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "mizar_clip_{prefix}_{}_{nanos}",
            std::process::id()
        ));
        std::fs::DirBuilder::new()
            .recursive(false)
            .mode(0o700)
            .create(&dir)?;
        Ok(Self { path: dir })
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn create_secret_file(&self, filename: &str) -> io::Result<PathBuf> {
        let path = self.path.join(filename);
        let _ = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        Ok(path)
    }
}

#[cfg(all(unix, any(target_os = "macos", test)))]
impl Drop for TempClipboardDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

async fn run_cmd_with_timeout(
    mut cmd: Command,
    timeout: Duration,
) -> Result<std::process::Output, String> {
    let child = cmd.kill_on_drop(true);
    match tokio::time::timeout(timeout, child.output()).await {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err(format!("命令超时（{}ms）", timeout.as_millis())),
    }
}

#[cfg(target_os = "macos")]
async fn read_native_macos() -> Result<ClipboardContent, ClipboardError> {
    let mut info_cmd = Command::new("osascript");
    info_cmd.args(["-e", "clipboard info"]);
    let info_out = run_cmd_with_timeout(info_cmd, NATIVE_CMD_TIMEOUT)
        .await
        .map_err(|e| ClipboardError::ToolUnavailable(format!("osascript 不可用：{e}")))?;

    let info_text = if info_out.status.success() {
        String::from_utf8_lossy(&info_out.stdout).into_owned()
    } else {
        String::new()
    };
    let probe = parse_macos_clipboard_info(&info_text);

    // 1. 文件优先
    if probe.has_files {
        let jxa = r#"ObjC.import("AppKit");
var pb = $.NSPasteboard.generalPasteboard;
var items = pb.readObjectsForClassesOptions($.NSArray.arrayWithObject($.NSURL), $());
var out = [];
if (items) {
  for (var i = 0; i < items.count; i++) {
    var u = items.objectAtIndex(i);
    if (u.isFileURL && u.path) out.push(ObjC.unwrap(u.path));
  }
}
out.join("\n");"#;
        let mut cmd = Command::new("osascript");
        cmd.args(["-l", "JavaScript", "-e", jxa]);
        if let Ok(out) = run_cmd_with_timeout(cmd, NATIVE_CMD_TIMEOUT).await
            && out.status.success()
        {
            let paths: Vec<PathBuf> = String::from_utf8_lossy(&out.stdout)
                .lines()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect();
            if !paths.is_empty() {
                return Ok(ClipboardContent::Files(paths));
            }
        }
        let mut fallback = Command::new("osascript");
        fallback.args(["-e", "POSIX path of (the clipboard as «class furl»)"]);
        if let Ok(out) = run_cmd_with_timeout(fallback, NATIVE_CMD_TIMEOUT).await
            && out.status.success()
        {
            let raw = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !raw.is_empty() {
                return Ok(ClipboardContent::Files(vec![PathBuf::from(raw)]));
            }
        }
    }

    // 2. 图片（通过 0700 临时目录 + 0600 临时文件落盘导出）
    if probe.has_png || probe.has_tiff || probe.has_jpeg || probe.has_gif || probe.has_webp {
        let tmp_dir = TempClipboardDir::create("macos")?;
        if probe.has_png {
            let png_path = tmp_dir.create_secret_file("clip.png")?;
            let path_str = png_path.to_string_lossy().replace('"', "\\\"");
            let script = format!(
                "set f to open for access POSIX file \"{path_str}\" with write permission\n\
                 set eof of f to 0\n\
                 write (the clipboard as «class PNGf») to f\n\
                 close access f"
            );
            let mut cmd = Command::new("osascript");
            cmd.args(["-e", &script]);
            let out = run_cmd_with_timeout(cmd, NATIVE_CMD_TIMEOUT)
                .await
                .map_err(|e| ClipboardError::CommandFailed(format!("导出 PNG 失败：{e}")))?;
            if out.status.success() {
                let bytes = std::fs::read(&png_path)?;
                if !bytes.is_empty() {
                    return Ok(ClipboardContent::Image {
                        bytes,
                        source_label: "macOS clipboard PNG".into(),
                    });
                }
            }
        }
        if probe.has_tiff {
            let tiff_path = tmp_dir.create_secret_file("clip.tiff")?;
            let png_path = tmp_dir.path().join("clip_converted.png");
            let tiff_str = tiff_path.to_string_lossy().replace('"', "\\\"");
            let script = format!(
                "set f to open for access POSIX file \"{tiff_str}\" with write permission\n\
                 set eof of f to 0\n\
                 write (the clipboard as «class TIFF») to f\n\
                 close access f"
            );
            let mut cmd = Command::new("osascript");
            cmd.args(["-e", &script]);
            let out = run_cmd_with_timeout(cmd, NATIVE_CMD_TIMEOUT)
                .await
                .map_err(|e| ClipboardError::CommandFailed(format!("导出 TIFF 失败：{e}")))?;
            if out.status.success() {
                let mut sips = Command::new("sips");
                sips.args(["-s", "format", "png"])
                    .arg(&tiff_path)
                    .args(["--out"])
                    .arg(&png_path);
                let sips_out = run_cmd_with_timeout(sips, NATIVE_CMD_TIMEOUT)
                    .await
                    .map_err(|e| {
                        ClipboardError::CommandFailed(format!("sips 转换 TIFF 失败：{e}"))
                    })?;
                if sips_out.status.success() {
                    let bytes = std::fs::read(&png_path)?;
                    if !bytes.is_empty() {
                        return Ok(ClipboardContent::Image {
                            bytes,
                            source_label: "macOS clipboard TIFF->PNG".into(),
                        });
                    }
                } else {
                    let stderr = String::from_utf8_lossy(&sips_out.stderr);
                    return Err(ClipboardError::CommandFailed(format!(
                        "sips 转换 TIFF 失败：{}",
                        stderr.trim()
                    )));
                }
            }
        }
        if probe.has_jpeg {
            let jpg_path = tmp_dir.create_secret_file("clip.jpg")?;
            let jpg_str = jpg_path.to_string_lossy().replace('"', "\\\"");
            let script = format!(
                "set f to open for access POSIX file \"{jpg_str}\" with write permission\n\
                 set eof of f to 0\n\
                 write (the clipboard as JPEG picture) to f\n\
                 close access f"
            );
            let mut cmd = Command::new("osascript");
            cmd.args(["-e", &script]);
            if let Ok(out) = run_cmd_with_timeout(cmd, NATIVE_CMD_TIMEOUT).await
                && out.status.success()
            {
                let bytes = std::fs::read(&jpg_path)?;
                if !bytes.is_empty() {
                    return Ok(ClipboardContent::Image {
                        bytes,
                        source_label: "macOS clipboard JPEG".into(),
                    });
                }
            }
        }
        if probe.has_gif {
            let gif_path = tmp_dir.create_secret_file("clip.gif")?;
            let gif_str = gif_path.to_string_lossy().replace('"', "\\\"");
            let script = format!(
                "set f to open for access POSIX file \"{gif_str}\" with write permission\n\
                 set eof of f to 0\n\
                 write (the clipboard as GIF picture) to f\n\
                 close access f"
            );
            let mut cmd = Command::new("osascript");
            cmd.args(["-e", &script]);
            if let Ok(out) = run_cmd_with_timeout(cmd, NATIVE_CMD_TIMEOUT).await
                && out.status.success()
            {
                let bytes = std::fs::read(&gif_path)?;
                if !bytes.is_empty() {
                    return Ok(ClipboardContent::Image {
                        bytes,
                        source_label: "macOS clipboard GIF".into(),
                    });
                }
            }
        }
    }

    // 3. 文本回退
    let mut pb = Command::new("pbpaste");
    pb.env("LANG", "en_US.UTF-8");
    let pb_out = run_cmd_with_timeout(pb, NATIVE_CMD_TIMEOUT)
        .await
        .map_err(|e| ClipboardError::ToolUnavailable(format!("pbpaste 不可用：{e}")))?;
    if !pb_out.status.success() {
        let stderr = String::from_utf8_lossy(&pb_out.stderr);
        return Err(ClipboardError::CommandFailed(format!(
            "pbpaste 执行失败：{}",
            stderr.trim()
        )));
    }
    let text = String::from_utf8_lossy(&pb_out.stdout).into_owned();
    if text.is_empty() {
        Err(ClipboardError::Empty)
    } else {
        Ok(ClipboardContent::Text(text))
    }
}

#[cfg(not(target_os = "macos"))]
async fn read_native_linux() -> Result<ClipboardContent, ClipboardError> {
    let mut probe_notes = Vec::new();

    // 1. Wayland: wl-paste --list-types
    let mut wl_list = Command::new("wl-paste");
    wl_list.arg("--list-types");
    match run_cmd_with_timeout(wl_list, NATIVE_CMD_TIMEOUT).await {
        Ok(out) if out.status.success() => {
            let types: Vec<String> = String::from_utf8_lossy(&out.stdout)
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            match select_preferred_mime(&types) {
                Some(MimeChoice::Files(mime)) => {
                    let mut get = Command::new("wl-paste");
                    get.args(["--type", &mime]);
                    let res = run_cmd_with_timeout(get, NATIVE_CMD_TIMEOUT)
                        .await
                        .map_err(|e| {
                            ClipboardError::CommandFailed(format!("wl-paste 读取文件列表失败：{e}"))
                        })?;
                    if res.status.success() {
                        let paths = decode_file_uri_list(&String::from_utf8_lossy(&res.stdout));
                        if !paths.is_empty() {
                            return Ok(ClipboardContent::Files(paths));
                        }
                    }
                }
                Some(MimeChoice::Image(mime)) => {
                    let mut get = Command::new("wl-paste");
                    get.args(["--type", &mime]);
                    let res = run_cmd_with_timeout(get, NATIVE_CMD_TIMEOUT)
                        .await
                        .map_err(|e| {
                            ClipboardError::CommandFailed(format!("wl-paste 读取图片失败：{e}"))
                        })?;
                    if res.status.success() && !res.stdout.is_empty() {
                        return Ok(ClipboardContent::Image {
                            bytes: res.stdout,
                            source_label: format!("wl-paste ({mime})"),
                        });
                    }
                    let stderr = String::from_utf8_lossy(&res.stderr);
                    return Err(ClipboardError::CommandFailed(format!(
                        "wl-paste 提取 `{mime}` 失败：{}",
                        stderr.trim()
                    )));
                }
                Some(MimeChoice::Text(_)) => {
                    let mut get = Command::new("wl-paste");
                    get.arg("--no-newline");
                    let res = run_cmd_with_timeout(get, NATIVE_CMD_TIMEOUT)
                        .await
                        .map_err(|e| {
                            ClipboardError::CommandFailed(format!("wl-paste 读取文本失败：{e}"))
                        })?;
                    if res.status.success() {
                        let text = String::from_utf8_lossy(&res.stdout).into_owned();
                        if !text.is_empty() {
                            return Ok(ClipboardContent::Text(text));
                        }
                    }
                }
                None => return Err(ClipboardError::Empty),
            }
        }
        Ok(out) => {
            let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
            if err.to_ascii_lowercase().contains("nothing is copied") {
                return Err(ClipboardError::Empty);
            }
            probe_notes.push(format!("wl-paste: {err}"));
        }
        Err(e) => probe_notes.push(format!("wl-paste: {e}")),
    }

    // 2. X11: xclip -selection clipboard -t TARGETS -o
    let mut xclip_targets = Command::new("xclip");
    xclip_targets.args(["-selection", "clipboard", "-t", "TARGETS", "-o"]);
    match run_cmd_with_timeout(xclip_targets, NATIVE_CMD_TIMEOUT).await {
        Ok(out) if out.status.success() => {
            let types: Vec<String> = String::from_utf8_lossy(&out.stdout)
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            match select_preferred_mime(&types) {
                Some(MimeChoice::Files(mime)) => {
                    let mut get = Command::new("xclip");
                    get.args(["-selection", "clipboard", "-t", &mime, "-o"]);
                    let res = run_cmd_with_timeout(get, NATIVE_CMD_TIMEOUT)
                        .await
                        .map_err(|e| {
                            ClipboardError::CommandFailed(format!("xclip 读取文件列表失败：{e}"))
                        })?;
                    if res.status.success() {
                        let paths = decode_file_uri_list(&String::from_utf8_lossy(&res.stdout));
                        if !paths.is_empty() {
                            return Ok(ClipboardContent::Files(paths));
                        }
                    }
                }
                Some(MimeChoice::Image(mime)) => {
                    let mut get = Command::new("xclip");
                    get.args(["-selection", "clipboard", "-t", &mime, "-o"]);
                    let res = run_cmd_with_timeout(get, NATIVE_CMD_TIMEOUT)
                        .await
                        .map_err(|e| {
                            ClipboardError::CommandFailed(format!("xclip 读取图片失败：{e}"))
                        })?;
                    if res.status.success() && !res.stdout.is_empty() {
                        return Ok(ClipboardContent::Image {
                            bytes: res.stdout,
                            source_label: format!("xclip ({mime})"),
                        });
                    }
                    let stderr = String::from_utf8_lossy(&res.stderr);
                    return Err(ClipboardError::CommandFailed(format!(
                        "xclip 提取 `{mime}` 失败：{}",
                        stderr.trim()
                    )));
                }
                Some(MimeChoice::Text(mime)) => {
                    let mut get = Command::new("xclip");
                    get.args(["-selection", "clipboard", "-t", &mime, "-o"]);
                    let res = run_cmd_with_timeout(get, NATIVE_CMD_TIMEOUT)
                        .await
                        .map_err(|e| {
                            ClipboardError::CommandFailed(format!("xclip 读取文本失败：{e}"))
                        })?;
                    if res.status.success() {
                        let text = String::from_utf8_lossy(&res.stdout).into_owned();
                        if !text.is_empty() {
                            return Ok(ClipboardContent::Text(text));
                        }
                    }
                }
                None => return Err(ClipboardError::Empty),
            }
        }
        Ok(out) => probe_notes.push(format!(
            "xclip: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )),
        Err(e) => probe_notes.push(format!("xclip: {e}")),
    }

    // 3. X11 文本兜底: xsel --clipboard --output
    let mut xsel = Command::new("xsel");
    xsel.args(["--clipboard", "--output"]);
    match run_cmd_with_timeout(xsel, NATIVE_CMD_TIMEOUT).await {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout).into_owned();
            if text.is_empty() {
                return Err(ClipboardError::Empty);
            }
            return Ok(ClipboardContent::Text(text));
        }
        Ok(out) => probe_notes.push(format!(
            "xsel: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )),
        Err(e) => probe_notes.push(format!("xsel: {e}")),
    }

    Err(ClipboardError::ToolUnavailable(format!(
        "请确认已安装 wl-paste / xclip / xsel 且图形会话可用（{}）",
        probe_notes.join("; ")
    )))
}

async fn read_native_clipboard() -> Result<ClipboardContent, ClipboardError> {
    #[cfg(target_os = "macos")]
    {
        read_native_macos().await
    }
    #[cfg(not(target_os = "macos"))]
    {
        read_native_linux().await
    }
}

/// 构造 Kitty OSC 5522 MIME 列表探测指令（不弹权限框，`.` base64 为 `Lg==`）。
pub fn build_osc5522_probe_request() -> String {
    "\x1b]5522;type=read;Lg==\x1b\\".to_string()
}

/// 构造 Kitty OSC 5522 指定 MIME 的读取请求。
pub fn build_osc5522_read_request(password: &str, app_name: &str, mime: &str) -> String {
    let pw_b64 = STANDARD.encode(password.as_bytes());
    let name_b64 = STANDARD.encode(app_name.as_bytes());
    let mime_b64 = STANDARD.encode(mime.as_bytes());
    format!("\x1b]5522;type=read:pw={pw_b64}:name={name_b64};{mime_b64}\x1b\\")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Osc5522Packet {
    pub kind: Option<String>,
    pub status: Option<String>,
    pub mime: Option<String>,
    pub payload_b64: String,
}

/// 从终端原始字节缓冲中提取已完整接收的 OSC 5522 报文（支持 ST `\x1b\\` 与 BEL `\x07`）。
pub fn extract_osc5522_packets(buf: &mut Vec<u8>) -> Vec<Osc5522Packet> {
    const PREFIX: &[u8] = b"\x1b]5522;";
    let mut packets = Vec::new();

    loop {
        let Some(start) = buf.windows(PREFIX.len()).position(|w| w == PREFIX) else {
            // 仅保留末尾可能尚未收完的 prefix 字节，避免无害按键污染后续匹配
            let keep = buf.len().min(PREFIX.len().saturating_sub(1));
            let drop_len = buf.len() - keep;
            if drop_len > 0 {
                buf.drain(..drop_len);
            }
            break;
        };
        if start > 0 {
            buf.drain(..start);
        }

        let after_prefix = &buf[PREFIX.len()..];
        let mut end_and_term_len = None;
        for i in 0..after_prefix.len() {
            if after_prefix[i] == 0x07 {
                end_and_term_len = Some((PREFIX.len() + i, 1));
                break;
            }
            if after_prefix[i] == 0x1B && i + 1 < after_prefix.len() && after_prefix[i + 1] == b'\\'
            {
                end_and_term_len = Some((PREFIX.len() + i, 2));
                break;
            }
        }
        let Some((body_end, term_len)) = end_and_term_len else {
            break;
        };

        let body = String::from_utf8_lossy(&buf[PREFIX.len()..body_end]).into_owned();
        buf.drain(..body_end + term_len);
        if let Some(pkt) = parse_osc5522_packet_body(&body) {
            packets.push(pkt);
        }
    }

    packets
}

pub fn parse_osc5522_packet_body(body: &str) -> Option<Osc5522Packet> {
    let (meta_str, payload_b64) = match body.split_once(';') {
        Some((m, p)) => (m, p.trim().to_string()),
        None => (body, String::new()),
    };
    let mut kind = None;
    let mut status = None;
    let mut mime = None;
    for pair in meta_str.split(':') {
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        match k {
            "type" => kind = Some(v.to_string()),
            "status" => status = Some(v.to_string()),
            "mime" => {
                if let Ok(decoded) = STANDARD.decode(v.as_bytes())
                    && let Ok(s) = String::from_utf8(decoded)
                {
                    mime = Some(s);
                }
            }
            _ => {}
        }
    }
    Some(Osc5522Packet {
        kind,
        status,
        mime,
        payload_b64,
    })
}

/// 在 raw mode 暂停 `EventStream` 后，直接面向 `/dev/tty`（回退 `stdin`）执行 OSC 5522 会话。
#[cfg(unix)]
pub fn read_osc5522_from_tty() -> Result<ClipboardContent, ClipboardError> {
    let tty_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty");
    let (mut writer, read_fd): (Box<dyn Write>, i32) = match &tty_file {
        Ok(f) => {
            let fd = f.as_raw_fd();
            (Box::new(f.try_clone()?), fd)
        }
        Err(_) => (Box::new(io::stdout()), io::stdin().as_raw_fd()),
    };

    // 第 1 步：以 `.` 探测当前可用 MIME 列表
    writer.write_all(build_osc5522_probe_request().as_bytes())?;
    writer.flush()?;

    let probe_bytes = collect_osc5522_response(read_fd, OSC5522_TIMEOUT)?;
    if probe_bytes.is_empty() {
        return Err(ClipboardError::Empty);
    }
    let mime_list: Vec<String> = String::from_utf8_lossy(&probe_bytes)
        .split_ascii_whitespace()
        .map(ToString::to_string)
        .collect();
    let choice = select_preferred_mime(&mime_list).ok_or(ClipboardError::Empty)?;
    let target_mime = match &choice {
        MimeChoice::Files(m) | MimeChoice::Image(m) | MimeChoice::Text(m) => m.clone(),
    };

    // 第 2 步：携带会话级一次性口令读取选定 MIME 内容
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pw = format!("mizar-{}-{nanos}", std::process::id());
    let req = build_osc5522_read_request(&pw, "mizar", &target_mime);
    writer.write_all(req.as_bytes())?;
    writer.flush()?;

    let payload = collect_osc5522_response(read_fd, Duration::from_secs(3))?;
    if payload.is_empty() {
        return Err(ClipboardError::Empty);
    }

    match choice {
        MimeChoice::Files(_) => {
            let paths = decode_file_uri_list(&String::from_utf8_lossy(&payload));
            if paths.is_empty() {
                Err(ClipboardError::Empty)
            } else {
                Ok(ClipboardContent::Files(paths))
            }
        }
        MimeChoice::Image(m) => Ok(ClipboardContent::Image {
            bytes: payload,
            source_label: format!("OSC 5522 ({m})"),
        }),
        MimeChoice::Text(_) => Ok(ClipboardContent::Text(
            String::from_utf8_lossy(&payload).into_owned(),
        )),
    }
}

#[cfg(not(unix))]
pub fn read_osc5522_from_tty() -> Result<ClipboardContent, ClipboardError> {
    Err(ClipboardError::Osc5522Unsupported(
        "非 Unix 平台暂不支持 OSC 5522".into(),
    ))
}

#[cfg(unix)]
fn collect_osc5522_response(fd: i32, timeout: Duration) -> Result<Vec<u8>, ClipboardError> {
    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    const POLLIN_EVENT: i16 = 0x0001;
    unsafe extern "C" {
        fn poll(
            fds: *mut PollFd,
            nfds: std::os::raw::c_uint,
            timeout: std::os::raw::c_int,
        ) -> std::os::raw::c_int;
        fn read(fd: std::os::raw::c_int, buf: *mut std::os::raw::c_void, count: usize) -> isize;
    }

    let deadline = Instant::now() + timeout;
    let mut raw_buf = Vec::with_capacity(4096);
    let mut assembled = Vec::new();
    let mut saw_ok = false;
    let mut chunk = [0u8; 4096];

    while Instant::now() < deadline {
        let rem_ms = deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .min(i32::MAX as u128) as i32;
        let mut pfd = PollFd {
            fd,
            events: POLLIN_EVENT,
            revents: 0,
        };
        let ready = unsafe { poll(&mut pfd, 1, rem_ms) };
        if ready <= 0 {
            break;
        }
        let n = unsafe { read(fd, chunk.as_mut_ptr().cast(), chunk.len()) };
        if n <= 0 {
            break;
        }
        raw_buf.extend_from_slice(&chunk[..n as usize]);
        for pkt in extract_osc5522_packets(&mut raw_buf) {
            match pkt.status.as_deref() {
                Some("OK") => {
                    saw_ok = true;
                }
                Some("DATA") => {
                    saw_ok = true;
                    if !pkt.payload_b64.is_empty()
                        && let Ok(decoded) = STANDARD.decode(pkt.payload_b64.as_bytes())
                    {
                        if assembled.len().saturating_add(decoded.len()) > OSC5522_MAX_BYTES {
                            return Err(ClipboardError::CommandFailed(
                                "OSC 5522 返回数据超过 32MB 上限".into(),
                            ));
                        }
                        assembled.extend_from_slice(&decoded);
                    }
                }
                Some("DONE") => return Ok(assembled),
                Some("EPERM") => return Err(ClipboardError::Osc5522Denied),
                Some("ENOSYS") => {
                    return Err(ClipboardError::Osc5522Unsupported("终端返回 ENOSYS".into()));
                }
                Some(code) => {
                    return Err(ClipboardError::Osc5522Unsupported(format!(
                        "终端返回状态 {code}"
                    )));
                }
                None => {}
            }
        }
    }

    if saw_ok && !assembled.is_empty() {
        Ok(assembled)
    } else if saw_ok {
        Err(ClipboardError::Empty)
    } else {
        Err(ClipboardError::Osc5522Unsupported(
            "等待终端 OSC 5522 应答超时（可通过环境变量禁止或检查终端是否支持）".into(),
        ))
    }
}

/// 统一读取剪贴板入口：本地会话走系统原生剪贴板；SSH 会话先走 `osc_hook`（OSC 5522）。
pub async fn read_clipboard_with_osc5522_hook<F, Fut>(
    osc_hook: F,
) -> Result<ClipboardContent, ClipboardError>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<ClipboardContent, ClipboardError>>,
{
    if native_unreachable() {
        match osc_hook().await {
            Ok(content) => return Ok(content),
            Err(osc_err @ ClipboardError::Osc5522Denied) => return Err(osc_err),
            Err(osc_err @ ClipboardError::Empty) => return Err(osc_err),
            Err(osc_err) => {
                // 若 SSH 同时启用了 X11 / Wayland 转发，回落尝试一次原生通道；否则明确汇报 OSC 5522 错误
                let has_forwarded_display = std::env::var_os("DISPLAY")
                    .is_some_and(|v| !v.is_empty())
                    || std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty());
                if has_forwarded_display && let Ok(native) = read_native_clipboard().await {
                    return Ok(native);
                }
                return Err(osc_err);
            }
        }
    }

    read_native_clipboard().await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn mime_selection_strictly_prefers_files_then_image_then_text() {
        // 同屏同时出现 text + image + file URI 时必须优先 files
        let all = vec![
            "text/plain;charset=utf-8".to_string(),
            "image/png".to_string(),
            "text/uri-list".to_string(),
        ];
        assert_eq!(
            select_preferred_mime(&all),
            Some(MimeChoice::Files("text/uri-list".into()))
        );

        // 截图常同时带有 text/html（或图像元信息）与 image/bmp + image/png，必须优先 image/png
        let img_and_text = vec![
            "text/plain".to_string(),
            "image/bmp".to_string(),
            "image/png".to_string(),
        ];
        assert_eq!(
            select_preferred_mime(&img_and_text),
            Some(MimeChoice::Image("image/png".into()))
        );

        let text_only = vec!["STRING".to_string(), "UTF8_STRING".to_string()];
        assert_eq!(
            select_preferred_mime(&text_only),
            Some(MimeChoice::Text("UTF8_STRING".into()))
        );
    }

    #[test]
    fn decodes_uri_list_and_gnome_copied_files_with_percent_encoding() {
        let uri_list =
            "# comment line\r\nfile:///tmp/my%20shot%231.png\r\nfile://localhost/var/log/a.txt\n";
        assert_eq!(
            decode_file_uri_list(uri_list),
            vec![
                PathBuf::from("/tmp/my shot#1.png"),
                PathBuf::from("/var/log/a.txt"),
            ]
        );

        let gnome = "copy\nfile:///home/user/%E6%88%AA%E5%9B%BE.png\n";
        assert_eq!(
            decode_file_uri_list(gnome),
            vec![PathBuf::from("/home/user/截图.png")]
        );
    }

    #[test]
    fn parses_macos_clipboard_info_tokens_accurately() {
        let info = "«class PNGf», 48210, «class TIFF», 120000, Unicode text, 18";
        let p = parse_macos_clipboard_info(info);
        assert!(!p.has_files);
        assert!(p.has_png);
        assert!(p.has_tiff);

        let furl = "«class furl», 144, «class utf8», 22";
        let pf = parse_macos_clipboard_info(furl);
        assert!(pf.has_files);
        assert!(!pf.has_png);
    }

    #[test]
    fn detects_ssh_native_unreachable_env_vars() {
        assert!(!native_unreachable_with(|_| false));
        assert!(native_unreachable_with(|k| k == "SSH_CONNECTION"));
        assert!(native_unreachable_with(|k| k == "SSH_CLIENT"));
        assert!(native_unreachable_with(|k| k == "SSH_TTY"));
    }

    #[test]
    fn encodes_and_reassembles_chunked_osc5522_packets() {
        assert_eq!(
            build_osc5522_probe_request(),
            "\x1b]5522;type=read;Lg==\x1b\\"
        );
        let req = build_osc5522_read_request("secret", "mizar", "image/png");
        assert!(req.starts_with("\x1b]5522;type=read:pw="));
        assert!(req.contains(":name=bWl6YXI="));
        assert!(req.ends_with(";aW1hZ2UvcG5n\x1b\\"));

        // 模拟分片到达 + 混杂键盘前缀 + BEL/ST 双终结符 + 分段 base64 数据
        let mime_b64 = STANDARD.encode(b"image/png");
        let chunk1 = STANDARD.encode(b"hello ");
        let chunk2 = STANDARD.encode(b"world");
        let stream = format!(
            "noise\x1b]5522;type=read:status=OK\x1b\\\
             \x1b]5522;type=read:status=DATA:mime={mime_b64};{chunk1}\x07\
             \x1b]5522;type=read:status=DATA:mime={mime_b64};{chunk2}\x1b\\\
             \x1b]5522;type=read:status=DONE\x1b\\"
        );
        let mut buf = stream.as_bytes()[..stream.len() - 12].to_vec();
        let first_pass = extract_osc5522_packets(&mut buf);
        assert_eq!(first_pass.len(), 3);
        assert_eq!(first_pass[0].status.as_deref(), Some("OK"));
        assert_eq!(first_pass[1].mime.as_deref(), Some("image/png"));
        assert_eq!(first_pass[1].payload_b64, chunk1);
        assert_eq!(first_pass[2].payload_b64, chunk2);

        // 补全最后 12 字节 DONE 包
        buf.extend_from_slice(&stream.as_bytes()[stream.len() - 12..]);
        let second_pass = extract_osc5522_packets(&mut buf);
        assert_eq!(second_pass.len(), 1);
        assert_eq!(second_pass[0].status.as_deref(), Some("DONE"));
        assert!(buf.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn temp_clipboard_dir_enforces_0700_and_0600_permissions() {
        let dir = TempClipboardDir::create("permtest").unwrap();
        let dir_mode = std::fs::metadata(dir.path()).unwrap().permissions().mode() & 0o777;
        assert_eq!(dir_mode, 0o700);
        let file = dir.create_secret_file("secret.png").unwrap();
        let file_mode = std::fs::metadata(&file).unwrap().permissions().mode() & 0o777;
        assert_eq!(file_mode, 0o600);
    }
}
