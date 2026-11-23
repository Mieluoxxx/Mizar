//! 自更新与启动异步版本检查（参考 `pi` 的 version-check + 自更新校验闭环，适配单二进制 GitHub Releases 分发）。

use std::cmp::Ordering;
use std::env;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mizar_core::instructions::sha256_hex;
use reqwest::redirect::Policy;

pub const DEFAULT_RELEASE_REPO_URL: &str = "https://github.com/Mieluoxxx/Mizar";
pub const VERSION_CHECK_TIMEOUT: Duration = Duration::from_secs(6);
pub const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(120);

fn is_truthy_str(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn is_truthy_env(key: &str) -> bool {
    env::var(key)
        .ok()
        .map(|v| is_truthy_str(&v))
        .unwrap_or(false)
}

/// 判断当前二进制是否属于官方发布构建（需构建期注入 `MIZAR_RELEASE_BUILD=1` 或显式 `MIZAR_VERSION`）。
pub fn is_release_build() -> bool {
    option_env!("MIZAR_RELEASE_BUILD").is_some_and(is_truthy_str)
        || option_env!("MIZAR_VERSION")
            .map(str::trim)
            .is_some_and(|v| !v.is_empty())
}

/// 返回当前二进制声明的版本号：优先 `MIZAR_VERSION` 覆盖，其次官方发布构建读
/// `CARGO_PKG_VERSION`，本地源码/开发构建附带 `+dev` 构建元数据以区分。
pub fn current_version() -> &'static str {
    if let Some(v) = option_env!("MIZAR_VERSION")
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        return v;
    }
    if is_release_build() {
        env!("CARGO_PKG_VERSION")
    } else {
        concat!(env!("CARGO_PKG_VERSION"), "+dev")
    }
}

pub fn is_offline_enabled() -> bool {
    is_truthy_env("MIZAR_OFFLINE")
}

pub fn is_version_check_disabled() -> bool {
    is_offline_enabled() || is_truthy_env("MIZAR_SKIP_VERSION_CHECK")
}

/// 判断是否应在 TUI 启动时执行后台版本检查。
///
/// 仅官方发布构建且位于可原地自更新路径时开启；本地源码构建、`/target/{debug,release}/`
/// 与 Homebrew 安装目录默认跳过，与 `validate_install_target` 保持一致。
pub fn should_check_for_updates_at_startup() -> bool {
    let exe_path = env::current_exe().ok();
    should_check_for_updates_for(
        is_release_build(),
        is_version_check_disabled(),
        exe_path.as_deref(),
    )
}

fn should_check_for_updates_for(
    release_build: bool,
    disabled_by_env: bool,
    exe_path: Option<&Path>,
) -> bool {
    if disabled_by_env || !release_build {
        return false;
    }
    match exe_path {
        Some(path) => {
            let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
            validate_install_target(&canonical, false).is_ok()
        }
        None => true,
    }
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
    mizar_llm::http::configure_custom_ca_roots(
        reqwest::Client::builder()
            .timeout(timeout)
            .user_agent(format!("mizar/{}", current_version()))
            // 一旦重定向指向 `/releases/tag/...` 即停止，避免下载整页 Release HTML，同时免受 GitHub API 限流
            .redirect(Policy::custom(|attempt| {
                if attempt.previous().len() >= 5 || attempt.url().path().contains("/releases/tag/")
                {
                    attempt.stop()
                } else {
                    attempt.follow()
                }
            })),
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn semver_sha256_tag_and_install_target_guards() {
        assert!(is_newer_version("v0.1.1", "0.1.0"));
        assert!(is_newer_version("v0.1.0", "0.1.0-alpha.1"));
        assert!(is_newer_version("v0.1.0-alpha.10", "0.1.0-alpha.2"));
        assert!(!is_newer_version("v0.1.0-alpha.1", "0.1.0"));

        let opts = parse_update_args_from([
            "--version".to_string(),
            "0.2.0".to_string(),
            "--force".to_string(),
            "--check".to_string(),
        ])
        .unwrap();
        assert_eq!(opts.version.as_deref(), Some("0.2.0"));
        assert!(opts.force && opts.check_only);
        assert!(parse_update_args_from(["--bogus".to_string()]).is_err());

        assert_eq!(
            extract_tag_from_location_or_url("/Mieluoxxx/mizar/releases/tag/v0.2.0-rc.1?foo=1"),
            Some("v0.2.0-rc.1".to_string())
        );

        let payload = b"mizar-binary-payload";
        let checksums = format!(
            "{}  mizar-v0.1.1-aarch64-apple-darwin.tar.gz\n",
            sha256_hex(payload)
        );
        assert!(
            verify_archive_sha256(
                payload,
                &checksums,
                "mizar-v0.1.1-aarch64-apple-darwin.tar.gz"
            )
            .is_ok()
        );
        assert!(
            verify_archive_sha256(
                b"tampered",
                &checksums,
                "mizar-v0.1.1-aarch64-apple-darwin.tar.gz"
            )
            .is_err()
        );

        assert!(
            validate_install_target(Path::new("/tmp/repo/target/release/mizar"), false).is_err()
        );
        assert!(
            validate_install_target(
                Path::new("/opt/homebrew/Cellar/mizar/0.1.0/bin/mizar"),
                false
            )
            .is_err()
        );
        assert!(validate_install_target(Path::new("/Users/me/.local/bin/mizar"), false).is_ok());
    }

    #[test]
    fn release_vs_dev_version_and_startup_update_gate() {
        assert!(!is_newer_version("v0.1.6", "0.1.6+dev"));
        assert!(is_newer_version("v0.1.7", "0.1.6+dev"));

        if let Some(override_ver) = option_env!("MIZAR_VERSION")
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            assert_eq!(current_version(), override_ver);
        } else if is_release_build() {
            assert_eq!(current_version(), env!("CARGO_PKG_VERSION"));
        } else {
            assert_eq!(
                current_version(),
                concat!(env!("CARGO_PKG_VERSION"), "+dev")
            );
        }

        assert!(!should_check_for_updates_for(
            false,
            false,
            Some(Path::new("/Users/me/.local/bin/mizar"))
        ));
        assert!(!should_check_for_updates_for(
            true,
            true,
            Some(Path::new("/Users/me/.local/bin/mizar"))
        ));
        assert!(!should_check_for_updates_for(
            true,
            false,
            Some(Path::new("/tmp/repo/target/debug/mizar"))
        ));
        assert!(!should_check_for_updates_for(
            true,
            false,
            Some(Path::new("/tmp/repo/target/release/mizar"))
        ));
        assert!(!should_check_for_updates_for(
            true,
            false,
            Some(Path::new("/opt/homebrew/Cellar/mizar/0.1.6/bin/mizar"))
        ));
        assert!(should_check_for_updates_for(
            true,
            false,
            Some(Path::new("/Users/me/.local/bin/mizar"))
        ));
        assert!(should_check_for_updates_for(true, false, None));
    }

    #[tokio::test]
    async fn end_to_end_mock_release_server_updates_and_verifies_checksum() {
        let temp_root = std::env::temp_dir().join(format!(
            "mizar_update_e2e_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_root).unwrap();

        let target_triple = detect_release_target().unwrap();
        let target_tag = "v99.0.0";
        let archive_name = format!("mizar-{target_tag}-{target_triple}.tar.gz");

        // 构造待安装的 mock 可执行脚本与 tar.gz 压缩包
        let build_stage = temp_root.join("build_stage");
        let pkg_dir = build_stage.join(format!("mizar-{target_tag}-{target_triple}"));
        fs::create_dir_all(&pkg_dir).unwrap();
        let new_bin = pkg_dir.join("mizar");
        fs::write(&new_bin, "#!/bin/sh\necho 'mizar 99.0.0'\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&new_bin, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let archive_path = temp_root.join(&archive_name);
        let tar_status = Command::new("tar")
            .arg("-C")
            .arg(&build_stage)
            .arg("-czf")
            .arg(&archive_path)
            .arg(format!("mizar-{target_tag}-{target_triple}"))
            .status()
            .unwrap();
        assert!(tar_status.success());

        let archive_bytes = fs::read(&archive_path).unwrap();
        let checksum_line = format!("{}  {}\n", sha256_hex(&archive_bytes), archive_name);

        // 在本地安装目录放置旧版可执行文件
        let install_dir = temp_root.join("bin");
        fs::create_dir_all(&install_dir).unwrap();
        let installed_exe = install_dir.join("mizar");
        fs::write(&installed_exe, "#!/bin/sh\necho 'mizar 0.1.0'\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&installed_exe, fs::Permissions::from_mode(0o755)).unwrap();
        }

        // 启动轻量本地 HTTP 服务模拟 GitHub Releases 重定向与文件下载
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let checksum_clone = checksum_line.clone();
        let archive_clone = archive_bytes.clone();
        let archive_route = format!("/releases/download/{target_tag}/{archive_name}");
        let checksum_route = format!("/releases/download/{target_tag}/checksums.txt");

        let server = thread::spawn(move || {
            for _ in 0..4 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut buf = [0u8; 2048];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let path = req
                    .lines()
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .unwrap_or("/");
                if path == "/releases/latest" {
                    let resp = format!(
                        "HTTP/1.1 302 Found\r\nLocation: http://{addr}/releases/tag/{target_tag}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else if path == checksum_route {
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        checksum_clone.len(),
                        checksum_clone
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else if path == archive_route {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        archive_clone.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&archive_clone);
                } else {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
            }
        });

        let base_url = format!("http://{addr}");
        let client = build_release_client(VERSION_CHECK_TIMEOUT).unwrap();
        let latest = fetch_latest_tag(&client, &base_url).await.unwrap();
        assert_eq!(latest.as_deref(), Some(target_tag));

        let opts = UpdateOptions::default();
        run_update_with_base_and_exe(&opts, &base_url, &installed_exe)
            .await
            .unwrap();

        let after_out = Command::new(&installed_exe)
            .arg("--version")
            .output()
            .unwrap();
        assert!(after_out.status.success());
        assert_eq!(
            String::from_utf8_lossy(&after_out.stdout).trim(),
            "mizar 99.0.0"
        );

        // 破坏校验和时应拦截且不替换原有可执行文件
        let bad_res = stage_and_replace_binary(
            &installed_exe,
            target_tag,
            target_triple,
            &archive_bytes,
            &format!("{}  {archive_name}\n", "0".repeat(64)),
        );
        assert!(bad_res.is_err());

        let _ = server.join();
        let _ = fs::remove_dir_all(&temp_root);
    }
}
