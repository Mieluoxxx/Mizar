//! 自更新与启动异步版本检查（参考 `pi` 的 version-check + 自更新校验闭环，适配单二进制 GitHub Releases 分发）。

use std::cmp::Ordering;
use std::env;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mizar_instructions::sha256_hex;
use reqwest::redirect::Policy;

pub const DEFAULT_RELEASE_REPO_URL: &str = "https://github.com/Mieluoxxx/Mizar";
pub const VERSION_CHECK_TIMEOUT: Duration = Duration::from_secs(6);
pub const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(120);

/// 返回当前二进制声明的版本号（优先读取构建期注入的 `MIZAR_VERSION`，否则回退 `CARGO_PKG_VERSION`）。
pub fn current_version() -> &'static str {
    option_env!("MIZAR_VERSION")
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or(env!("CARGO_PKG_VERSION"))
}

fn is_truthy_env(key: &str) -> bool {
    env::var(key)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub fn is_offline_enabled() -> bool {
    is_truthy_env("MIZAR_OFFLINE")
}

pub fn is_version_check_disabled() -> bool {
    is_offline_enabled() || is_truthy_env("MIZAR_SKIP_VERSION_CHECK")
}

pub fn release_base_url() -> String {
    env::var("MIZAR_RELEASE_BASE_URL")
        .ok()
        .map(|v| v.trim().trim_end_matches('/').to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_RELEASE_REPO_URL.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PreId {
    Num(u64),
    Str(String),
}

impl Ord for PreId {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Num(a), Self::Num(b)) => a.cmp(b),
            (Self::Num(_), Self::Str(_)) => Ordering::Less,
            (Self::Str(_), Self::Num(_)) => Ordering::Greater,
            (Self::Str(a), Self::Str(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for PreId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 轻量 SemVer 2.0.0 版本号，用于精确比较正式版与预发布标签（`v0.1.0` > `v0.1.0-alpha.1`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemVer {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pre: Vec<PreId>,
}

impl SemVer {
    pub fn parse(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        let without_v = trimmed
            .strip_prefix('v')
            .or_else(|| trimmed.strip_prefix('V'))
            .unwrap_or(trimmed);
        let without_build = without_v
            .split_once('+')
            .map(|(core, _)| core)
            .unwrap_or(without_v);
        let (core, pre_str) = match without_build.split_once('-') {
            Some((c, p)) if !p.is_empty() => (c, Some(p)),
            Some(_) => return None,
            None => (without_build, None),
        };

        let mut parts = core.split('.');
        let major = parse_numeric_part(parts.next()?)?;
        let minor = parse_numeric_part(parts.next()?)?;
        let patch = parse_numeric_part(parts.next()?)?;
        if parts.next().is_some() {
            return None;
        }

        let mut pre = Vec::new();
        if let Some(p) = pre_str {
            for seg in p.split('.') {
                if seg.is_empty() {
                    return None;
                }
                if seg.bytes().all(|b| b.is_ascii_digit()) {
                    pre.push(PreId::Num(seg.parse().ok()?));
                } else {
                    pre.push(PreId::Str(seg.to_string()));
                }
            }
        }

        Some(Self {
            major,
            minor,
            patch,
            pre,
        })
    }
}

fn parse_numeric_part(part: &str) -> Option<u64> {
    if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    part.parse().ok()
}

impl Ord for SemVer {
    fn cmp(&self, other: &Self) -> Ordering {
        self.major
            .cmp(&other.major)
            .then_with(|| self.minor.cmp(&other.minor))
            .then_with(|| self.patch.cmp(&other.patch))
            .then_with(|| match (self.pre.is_empty(), other.pre.is_empty()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => self.pre.cmp(&other.pre),
            })
    }
}

impl PartialOrd for SemVer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn normalize_tag(raw: &str) -> String {
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix('v')
        .or_else(|| trimmed.strip_prefix('V'))
        .unwrap_or(trimmed);
    format!("v{stripped}")
}

pub fn is_newer_version(candidate: &str, current: &str) -> bool {
    match (SemVer::parse(candidate), SemVer::parse(current)) {
        (Some(cand), Some(curr)) => cand > curr,
        _ => normalize_tag(candidate) != normalize_tag(current),
    }
}

fn extract_tag_from_location_or_url(raw: &str) -> Option<String> {
    let path_only = raw.split(['?', '#']).next().unwrap_or(raw);
    let (_, after_tag) = path_only.rsplit_once("/releases/tag/")?;
    let tag = after_tag.trim_end_matches('/');
    if tag.is_empty() || tag.contains('/') {
        return None;
    }
    Some(tag.to_string())
}

fn build_release_client(timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(timeout)
        .user_agent(format!("mizar/{}", current_version()))
        // 一旦重定向指向 `/releases/tag/...` 即停止，避免下载整页 Release HTML，同时免受 GitHub API 限流
        .redirect(Policy::custom(|attempt| {
            if attempt.previous().len() >= 5 || attempt.url().path().contains("/releases/tag/") {
                attempt.stop()
            } else {
                attempt.follow()
            }
        }))
        .build()
        .map_err(|e| format!("创建 HTTP 客户端失败：{e}"))
}

/// 通过 `<base_url>/releases/latest` 的重定向头解析最新 Release 标签。
pub async fn fetch_latest_tag(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<Option<String>, String> {
    let url = format!("{}/releases/latest", base_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("请求最新版本失败：{e}"))?;

    if resp.status().is_redirection() {
        if let Some(loc) = resp
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            && let Some(tag) = extract_tag_from_location_or_url(loc)
        {
            return Ok(Some(tag));
        }
        return Ok(None);
    }

    if resp.status().is_success() {
        return Ok(extract_tag_from_location_or_url(resp.url().path()));
    }

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }

    Err(format!("远端返回异常状态码：HTTP {}", resp.status()))
}

/// TUI 启动后在后台静默检查更新；断网、超时或无新版均返回 `None`。
pub async fn check_for_new_version() -> Option<String> {
    if is_version_check_disabled() {
        return None;
    }
    let client = build_release_client(VERSION_CHECK_TIMEOUT).ok()?;
    let tag = fetch_latest_tag(&client, &release_base_url())
        .await
        .ok()??;
    if is_newer_version(&tag, current_version()) {
        Some(tag)
    } else {
        None
    }
}

pub fn format_update_notice(latest_tag: &str) -> String {
    let curr = current_version().trim_start_matches('v');
    format!("发现新版本 {latest_tag}（当前 v{curr}），退出后运行 `mizar update` 即可更新")
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UpdateOptions {
    pub version: Option<String>,
    pub force: bool,
    pub check_only: bool,
    pub help: bool,
}

pub fn format_update_help() -> &'static str {
    "\
Update mizar to the latest GitHub Release

Usage: mizar update [OPTIONS]

Options:
  -v, --version <ver>   Install a specific release version (e.g. v0.1.1 or 0.1.1)
  -f, --force           Reinstall even if already on target version or in dev/managed path
      --check           Only check whether a newer version is available
  -h, --help            Print this help message"
}

pub fn parse_update_args_from<I>(args: I) -> Result<UpdateOptions, String>
where
    I: IntoIterator<Item = String>,
{
    let mut opts = UpdateOptions::default();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-v" | "--version" => {
                let val = iter
                    .next()
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| format!("缺少 {arg} 的参数值"))?;
                opts.version = Some(val);
            }
            "-f" | "--force" => opts.force = true,
            "--check" => opts.check_only = true,
            "-h" | "--help" => opts.help = true,
            other => return Err(format!("未知参数：{other}")),
        }
    }
    Ok(opts)
}

pub fn detect_release_target() -> Result<&'static str, String> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        Ok("aarch64-apple-darwin")
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        Ok("x86_64-apple-darwin")
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        Ok("x86_64-unknown-linux-gnu")
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        Ok("aarch64-unknown-linux-gnu")
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
    )))]
    {
        Err(
            "当前操作系统或架构暂不支持自动更新，请从 GitHub Releases 手动下载或源码编译"
                .to_string(),
        )
    }
}

pub fn parse_checksum_for_asset(checksums: &str, asset_name: &str) -> Option<String> {
    for line in checksums.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(hash) = parts.next() else {
            continue;
        };
        let Some(file_part) = parts.next() else {
            continue;
        };
        let cleaned = file_part.trim_start_matches('*').trim_start_matches("./");
        if cleaned == asset_name && hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Some(hash.to_ascii_lowercase());
        }
    }
    None
}

pub fn verify_archive_sha256(
    archive_bytes: &[u8],
    checksums: &str,
    archive_name: &str,
) -> Result<(), String> {
    let expected = parse_checksum_for_asset(checksums, archive_name)
        .ok_or_else(|| format!("checksums.txt 中未找到 {archive_name} 的校验记录"))?;
    let actual = sha256_hex(archive_bytes);
    if actual != expected {
        return Err(format!(
            "{archive_name} SHA256 校验失败（期望 {expected}，实际 {actual}）"
        ));
    }
    Ok(())
}

fn validate_install_target(exe_path: &Path, force: bool) -> Result<(), String> {
    let normalized = exe_path.to_string_lossy().replace('\\', "/");
    if !force && normalized.contains("/Cellar/mizar/") {
        return Err(format!(
            "当前 mizar 位于 Homebrew 目录（{}），请直接运行 `brew upgrade mizar`（如需强制覆盖请加 --force）",
            exe_path.display()
        ));
    }
    if !force && (normalized.contains("/target/debug/") || normalized.contains("/target/release/"))
    {
        return Err(format!(
            "当前运行的是源码构建产物（{}），已跳过原地覆盖；如需强制替换请加 --force",
            exe_path.display()
        ));
    }
    Ok(())
}

struct StagingDirGuard {
    path: PathBuf,
}

impl StagingDirGuard {
    fn create_in(parent: &Path) -> Result<Self, String> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for attempt in 0..8u32 {
            let dir = parent.join(format!(
                ".mizar-update-{}-{nanos}-{attempt}",
                std::process::id()
            ));
            match fs::create_dir(&dir) {
                Ok(()) => return Ok(Self { path: dir }),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => {
                    return Err(format!(
                        "没有目录写入权限（{}）：{err}；请检查权限或使用 sudo / install.sh 更新",
                        parent.display()
                    ));
                }
                Err(err) => {
                    return Err(format!(
                        "创建更新暂存目录失败（{}）：{err}",
                        parent.display()
                    ));
                }
            }
        }
        Err(format!("无法在 {} 创建唯一更新暂存目录", parent.display()))
    }
}

impl Drop for StagingDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// 校验压缩包 SHA256、解压到同目录暂存区、试运行 `--version` 后再原子替换目标可执行文件。
pub fn stage_and_replace_binary(
    exe_path: &Path,
    target_tag: &str,
    target_triple: &str,
    archive_bytes: &[u8],
    checksums_text: &str,
) -> Result<String, String> {
    let archive_name = format!("mizar-{target_tag}-{target_triple}.tar.gz");
    verify_archive_sha256(archive_bytes, checksums_text, &archive_name)?;

    let exe_dir = exe_path
        .parent()
        .ok_or_else(|| format!("无法解析可执行文件所在目录：{}", exe_path.display()))?;
    let staging = StagingDirGuard::create_in(exe_dir)?;

    let archive_path = staging.path.join(&archive_name);
    fs::write(&archive_path, archive_bytes).map_err(|e| format!("写入临时压缩包失败：{e}"))?;

    let tar_status = Command::new("tar")
        .arg("-xzf")
        .arg(&archive_path)
        .arg("-C")
        .arg(&staging.path)
        .status()
        .map_err(|e| format!("调用 tar 解压失败：{e}"))?;
    if !tar_status.success() {
        return Err(format!(
            "tar 解压 {archive_name} 失败（退出码：{tar_status}）"
        ));
    }

    let expected_bin = staging
        .path
        .join(format!("mizar-{target_tag}-{target_triple}"))
        .join("mizar");
    let fallback_bin = staging.path.join("mizar");
    let staged_bin = if expected_bin.is_file() {
        expected_bin
    } else if fallback_bin.is_file() {
        fallback_bin
    } else {
        return Err(format!("压缩包 {archive_name} 中未找到 mizar 可执行文件"));
    };

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o755);
        fs::set_permissions(&staged_bin, perms).map_err(|e| format!("设置可执行权限失败：{e}"))?;
    }

    let check = Command::new(&staged_bin)
        .arg("--version")
        .output()
        .map_err(|e| format!("校验新版本二进制失败：{e}"))?;
    if !check.status.success() {
        let stderr = String::from_utf8_lossy(&check.stderr);
        return Err(format!(
            "新二进制自检 `--version` 失败（{}）：{}",
            check.status,
            stderr.trim()
        ));
    }
    let reported_version = String::from_utf8_lossy(&check.stdout).trim().to_string();
    if !reported_version.starts_with("mizar ") {
        return Err(format!(
            "新二进制 `--version` 输出异常：{reported_version:?}"
        ));
    }

    // 同一挂载点内 rename 在 Unix 下为原子替换，正在运行的其他 mizar 进程不受影响
    fs::rename(&staged_bin, exe_path)
        .map_err(|e| format!("原子替换可执行文件失败（{}）：{e}", exe_path.display()))?;

    Ok(reported_version)
}

fn read_version_on_path() -> Option<String> {
    let out = Command::new("mizar").arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

async fn download_bytes(
    client: &reqwest::Client,
    url: &str,
    label: &str,
) -> Result<Vec<u8>, String> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("下载 {label} 失败：{e}"))?;
    if !resp.status().is_success() {
        return Err(format!("下载 {label} 失败：HTTP {} ({url})", resp.status()));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("读取 {label} 响应体失败：{e}"))
}

pub async fn run_update_with_base_and_exe(
    opts: &UpdateOptions,
    base_url: &str,
    exe_path: &Path,
) -> Result<(), String> {
    if is_offline_enabled() {
        return Err("当前处于离线模式（MIZAR_OFFLINE），无法检查或下载更新".to_string());
    }

    let client = build_release_client(DOWNLOAD_TIMEOUT)?;
    let target_tag = match &opts.version {
        Some(v) => normalize_tag(v),
        None => {
            let first = fetch_latest_tag(&client, base_url).await;
            let fetched = match first {
                Ok(v) => v,
                Err(_) => fetch_latest_tag(&client, base_url).await?,
            };
            fetched.ok_or_else(|| {
                "未找到可用的正式 Release（若需安装预发布版本，请指定 --version vX.Y.Z-alpha.N）"
                    .to_string()
            })?
        }
    };

    let current = current_version();
    let has_newer = is_newer_version(&target_tag, current);

    if opts.check_only {
        if has_newer {
            println!(
                "发现新版本 {target_tag}（当前 v{}），运行 `mizar update` 更新",
                current.trim_start_matches('v')
            );
        } else {
            println!(
                "mizar 已是最新版本（当前 v{}，远端 {target_tag}）",
                current.trim_start_matches('v')
            );
        }
        return Ok(());
    }

    if !opts.force && !has_newer {
        println!(
            "mizar 已是最新版本（当前 v{}，远端 {target_tag}）",
            current.trim_start_matches('v')
        );
        return Ok(());
    }

    let canonical_exe = fs::canonicalize(exe_path).unwrap_or_else(|_| exe_path.to_path_buf());
    validate_install_target(&canonical_exe, opts.force)?;

    let exe_dir = canonical_exe
        .parent()
        .ok_or_else(|| format!("无法解析可执行文件目录：{}", canonical_exe.display()))?;
    // 下载大文件前先试探目录写权限，避免白下几十 MB 后才报权限错误
    drop(StagingDirGuard::create_in(exe_dir)?);

    let target_triple = detect_release_target()?;
    let archive_name = format!("mizar-{target_tag}-{target_triple}.tar.gz");
    let base = base_url.trim_end_matches('/');
    let checksums_url = format!("{base}/releases/download/{target_tag}/checksums.txt");
    let archive_url = format!("{base}/releases/download/{target_tag}/{archive_name}");

    if std::io::stdout().is_terminal() {
        println!(
            "正在更新 mizar：v{} -> {target_tag} ({target_triple})...",
            current.trim_start_matches('v')
        );
    }

    let checksums_bytes = download_bytes(&client, &checksums_url, "checksums.txt").await?;
    let checksums_text =
        String::from_utf8(checksums_bytes).map_err(|e| format!("checksums.txt 编码无效：{e}"))?;
    let archive_bytes = download_bytes(&client, &archive_url, &archive_name).await?;

    let reported = stage_and_replace_binary(
        &canonical_exe,
        &target_tag,
        target_triple,
        &archive_bytes,
        &checksums_text,
    )?;

    println!(
        "已将 mizar 从 v{} 更新至 {target_tag}（{}，校验输出：{reported}）",
        current.trim_start_matches('v'),
        canonical_exe.display()
    );

    if let Some(path_ver) = read_version_on_path()
        && path_ver != reported
    {
        eprintln!(
            "提示：当前 PATH 中的 `mizar --version` 返回 `{path_ver}`，与刚更新的 `{reported}` 不一致，请检查是否存在多个安装副本。"
        );
    }

    Ok(())
}

pub async fn run_update_cli(args: &[String]) -> Result<(), String> {
    let opts = parse_update_args_from(args.iter().cloned())?;
    if opts.help {
        println!("{}", format_update_help());
        return Ok(());
    }
    let exe_path = env::current_exe().map_err(|e| format!("无法获取当前可执行文件路径：{e}"))?;
    let base_url = release_base_url();
    run_update_with_base_and_exe(&opts, &base_url, &exe_path).await
}
