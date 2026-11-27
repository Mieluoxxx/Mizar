use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use futures::future::join_all;
use mizar_tools_coding::{mizar_bin_dir, resolve_managed_bin};
use reqwest::{Client, header::LOCATION, redirect::Policy};
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::OnceCell;
use tokio::time::{Instant, sleep_until, timeout};

use crate::update::{
    DOWNLOAD_TIMEOUT, StagingDirGuard, current_version, extract_tag_from_location_or_url,
    is_truthy_env,
};

const HEADER_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone)]
pub(crate) struct InstalledTool {
    pub(crate) path: PathBuf,
    pub(crate) source: String,
    version: Option<String>,
}

type InstallResult = Result<InstalledTool, String>;

#[derive(PartialEq)]
enum DownloadSource {
    Direct,
    Mirror,
}

struct Tool {
    name: &'static str,
    repo: &'static str,
    asset: &'static str,
    tag_prefix: &'static str,
    flight: Mutex<Weak<OnceCell<InstallResult>>>,
}

static TOOLS: [Tool; 2] = [
    Tool::new("rg", "BurntSushi/ripgrep", "ripgrep", ""),
    Tool::new("fd", "sharkdp/fd", "fd", "v"),
];

impl Tool {
    const fn new(
        name: &'static str,
        repo: &'static str,
        asset: &'static str,
        tag_prefix: &'static str,
    ) -> Self {
        Self {
            name,
            repo,
            asset,
            tag_prefix,
            flight: Mutex::new(Weak::new()),
        }
    }

    async fn ensure(&self, force: bool) -> InstallResult {
        // 只共享仍有调用方等待的结果，失败或取消后下一次调用可以重新安装。
        let flight = {
            let mut slot = self.flight.lock().unwrap();
            slot.upgrade().unwrap_or_else(|| {
                let flight = Arc::new(OnceCell::new());
                *slot = Arc::downgrade(&flight);
                flight
            })
        };
        if !force && let Some(path) = resolve_managed_bin(self.name) {
            #[cfg(unix)]
            let path = if self.name == "fd" && path.file_name().is_some_and(|name| name == "fdfind")
            {
                // Debian 的包只提供 fdfind，补上本地别名才能在 bash 中直接使用 fd。
                let dir = bin_dir()?;
                fs::create_dir_all(&dir)
                    .await
                    .map_err(|e| format!("无法创建 {}：{e}", dir.display()))?;
                let alias = dir.join("fd");
                if let Err(e) = fs::symlink(&path, &alias).await
                    && (e.kind() != io::ErrorKind::AlreadyExists
                        || resolve_managed_bin("fd").as_ref() != Some(&alias))
                {
                    return Err(format!("无法创建 {}：{e}", alias.display()));
                }
                alias
            } else {
                path
            };
            return Ok(InstalledTool {
                path,
                source: "本地".into(),
                version: None,
            });
        }
        if is_truthy_env("MIZAR_OFFLINE") || is_truthy_env("PI_OFFLINE") {
            return Err("离线模式已跳过下载（MIZAR_OFFLINE / PI_OFFLINE）".into());
        }

        flight.get_or_init(|| self.install()).await.clone()
    }

    async fn install(&self) -> InstallResult {
        let target = match (env::consts::OS, env::consts::ARCH) {
            ("macos", "aarch64") => "aarch64-apple-darwin",
            ("macos", "x86_64") => "x86_64-apple-darwin",
            ("linux", "aarch64") => "aarch64-unknown-linux-musl",
            ("linux", "x86_64") => "x86_64-unknown-linux-musl",
            (os, arch) => {
                return Err(format!(
                    "暂不支持 {os}/{arch} 的托管包，请用系统包管理器安装 {}",
                    self.name
                ));
            }
        };
        let dir = bin_dir()?;
        fs::create_dir_all(&dir)
            .await
            .map_err(|e| format!("无法创建 {}：{e}", dir.display()))?;
        let staging = StagingDirGuard::create_in(&dir, self.name)?;
        let base = env::var("MIZAR_GITHUB_BASE_URL").unwrap_or_default();
        let base = base.trim().trim_end_matches('/');
        let base = if base.is_empty() {
            "https://github.com"
        } else {
            base
        };
        let repo = format!("{base}/{}", self.repo);
        let mirror =
            env::var("MIZAR_GITHUB_MIRROR").unwrap_or_else(|_| "https://ghfast.top".into());
        let mirror = mirror.trim().trim_end_matches('/');
        let mirror = (!matches!(mirror.to_ascii_lowercase().as_str(), "" | "off" | "none"))
            .then_some(mirror);
        let client = || Client::builder().user_agent(format!("mizar/{}", current_version()));

        let version = if self.name == "fd" && target == "x86_64-apple-darwin" {
            "10.3.0".into()
        } else {
            let client = client()
                .redirect(Policy::none())
                .timeout(HEADER_TIMEOUT)
                .build()
                .map_err(|e| e.to_string())?;
            let (version, _) =
                from_sources(&format!("{repo}/releases/latest"), mirror, |url, _| {
                    latest_version(&client, url, self.tag_prefix)
                })
                .await?;
            version
        };

        let tag = format!("{}{version}", self.tag_prefix);
        let url = format!(
            "{repo}/releases/download/{tag}/{}-{tag}-{target}.tar.gz",
            self.asset
        );
        let client = client()
            .timeout(DOWNLOAD_TIMEOUT)
            .build()
            .map_err(|e| e.to_string())?;
        let archive = staging.path.join("archive.tar.gz");
        let (_, source) = from_sources(&url, mirror, |url, source| {
            download(
                &client,
                url,
                &archive,
                source == DownloadSource::Direct && mirror.is_some(),
            )
        })
        .await?;

        let staged = staging.path.join(self.name);
        timeout(
            DOWNLOAD_TIMEOUT,
            extract_binary(&archive, self.name, &staged),
        )
        .await
        .map_err(|_| "解压托管包超时")??;
        let reported = check_version(&staged).await?;
        let path = dir.join(self.name);
        fs::rename(&staged, &path)
            .await
            .map_err(|e| format!("无法替换 {}：{e}", path.display()))?;
        Ok(InstalledTool {
            path,
            source,
            version: Some(reported),
        })
    }
}

fn bin_dir() -> Result<PathBuf, String> {
    mizar_bin_dir().ok_or_else(|| "找不到工具目录，请设置 MIZAR_HOME".into())
}

async fn from_sources<T, F: Future<Output = Result<T, String>>>(
    url: &str,
    mirror: Option<&str>,
    fetch: impl Fn(String, DownloadSource) -> F,
) -> Result<(T, String), String> {
    match fetch(url.to_owned(), DownloadSource::Direct).await {
        Ok(value) => Ok((value, format!("直连 {url}"))),
        Err(err) => {
            let direct = format!("直连 {url} 失败：{err}");
            let Some(mirror) = mirror else {
                return Err(direct);
            };
            let url = format!("{mirror}/{url}");
            match fetch(url.clone(), DownloadSource::Mirror).await {
                Ok(value) => Ok((value, format!("镜像 {url}（{direct}）"))),
                Err(err) => Err(format!("{direct}，镜像 {url} 失败：{err}")),
            }
        }
    }
}

async fn latest_version(client: &Client, url: String, tag_prefix: &str) -> Result<String, String> {
    let response = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !response.status().is_redirection() {
        return Err(format!(
            "版本查询返回 HTTP {}，需要重定向",
            response.status()
        ));
    }
    let tag = response
        .headers()
        .get(LOCATION)
        .and_then(|v| v.to_str().ok())
        .and_then(extract_tag_from_location_or_url);
    let version = tag
        .as_deref()
        .and_then(|tag| tag.strip_prefix(tag_prefix))
        .filter(|v| {
            v.starts_with(|c: char| c.is_ascii_digit())
                && v.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b".-+".contains(&b))
        })
        .ok_or("版本重定向缺少有效的 Release 标签")?;
    Ok(version.to_owned())
}

fn env_number(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

async fn download(
    client: &Client,
    url: String,
    archive: &Path,
    watch_speed: bool,
) -> Result<(), String> {
    let mut response = timeout(HEADER_TIMEOUT, client.get(url).send())
        .await
        .map_err(|_| "等待响应头超时")?
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }
    if response
        .content_length()
        .is_some_and(|size| size > MAX_BYTES)
    {
        return Err("压缩包超过 64 MiB".into());
    }
    let mut file = File::create(archive).await.map_err(|e| e.to_string())?;
    let slow_bps = env_number("MIZAR_BIN_SLOW_BPS", 65_536);
    let warmup =
        Duration::from_millis(env_number("MIZAR_BIN_WARMUP_MS", 5_000)).min(DOWNLOAD_TIMEOUT);
    let stall =
        Duration::from_millis(env_number("MIZAR_BIN_STALL_MS", 8_000)).min(DOWNLOAD_TIMEOUT);
    let start = Instant::now();
    let mut last_byte = start;
    let mut received = 0u64;
    loop {
        // 即使下一块迟迟未到，也要在累计字节数撑不住最低均速时及时切源。
        let speed_window = Duration::from_secs_f64(received as f64 / slow_bps.max(1) as f64);
        let chunk = tokio::select! {
            chunk = response.chunk() => chunk.map_err(|e| e.to_string())?,
            _ = sleep_until(start + warmup.max(speed_window)), if watch_speed && slow_bps > 0 => return Err(format!("平均下载速率低于 {slow_bps} B/s")),
            _ = sleep_until(last_byte + stall) => return Err(format!("下载停滞 {} ms", stall.as_millis())),
        };
        let Some(chunk) = chunk else { break };
        if chunk.is_empty() {
            continue;
        }
        last_byte = Instant::now();
        received += chunk.len() as u64;
        if received > MAX_BYTES {
            return Err("压缩包超过 64 MiB".into());
        }
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
    }
    file.flush().await.map_err(|e| e.to_string())
}

async fn copy_output(
    cmd: &mut Command,
    out: &mut (impl AsyncWrite + Unpin),
    limit: u64,
) -> io::Result<()> {
    let mut child = cmd
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut stdout = child.stdout.take().unwrap().take(limit + 1);
    if tokio::io::copy(&mut stdout, out).await? > limit {
        return Err(io::Error::other(format!("输出超过 {limit} 字节")));
    }
    let status = child.wait().await?;
    if !status.success() {
        return Err(io::Error::other(format!("退出状态 {status}")));
    }
    Ok(())
}

async fn extract_binary(archive: &Path, name: &str, staged: &Path) -> Result<(), String> {
    // 只让目标成员的内容落盘，归档里的路径、链接和权限不能影响暂存区以外的文件。
    let mut listing = Vec::new();
    copy_output(
        Command::new("tar").arg("-tzf").arg(archive),
        &mut listing,
        1024 * 1024,
    )
    .await
    .map_err(|e| format!("无法读取 tar 压缩包：{e}"))?;
    let listing = std::str::from_utf8(&listing).map_err(|e| e.to_string())?;
    let member = listing
        .lines()
        .find(|line| line.rsplit('/').next() == Some(name))
        .ok_or_else(|| format!("压缩包中找不到 {name}"))?;
    let mut file = File::create(staged).await.map_err(|e| e.to_string())?;
    copy_output(
        Command::new("tar")
            .arg("-xzf")
            .arg(archive)
            .args(["-O", "--", member]),
        &mut file,
        MAX_BYTES,
    )
    .await
    .map_err(|e| format!("无法从 tar 解压 {name}：{e}"))?;
    file.flush().await.map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o755))
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

async fn check_version(path: &Path) -> Result<String, String> {
    let mut stdout = Vec::new();
    timeout(
        HEADER_TIMEOUT,
        copy_output(Command::new(path).arg("--version"), &mut stdout, 64 * 1024),
    )
    .await
    .map_err(|_| "运行 --version 超时")?
    .map_err(|e| format!("{} --version 验证失败：{e}", path.display()))?;
    let version = String::from_utf8_lossy(&stdout).trim().to_owned();
    if version.is_empty() {
        return Err(format!("{} --version 输出为空", path.display()));
    }
    Ok(version)
}

fn lookup(name: &str) -> Result<&'static Tool, String> {
    TOOLS
        .iter()
        .find(|tool| tool.name == name)
        .ok_or_else(|| format!("未知工具 {name}，可选 rg 或 fd"))
}

pub(crate) async fn ensure_tool(name: &str) -> InstallResult {
    lookup(name)?.ensure(false).await
}

pub async fn run_install_tools_cli(args: &[String]) -> Result<(), String> {
    let mut requested: Vec<&Tool> = Vec::new();
    let mut force = false;
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: mizar install-tools [--force] [rg|fd ...]\n\nInstall both tools by default. Use --force to replace local tools in the Mizar bin directory."
                );
                return Ok(());
            }
            "--force" => force = true,
            name => {
                let tool = lookup(name)?;
                if !requested.iter().any(|item| item.name == name) {
                    requested.push(tool);
                }
            }
        }
    }
    if requested.is_empty() {
        requested.extend(&TOOLS);
    }
    let installs = requested.iter().map(|tool| async move {
        let installed = tool.ensure(force).await?;
        let version = match installed.version {
            Some(version) => version,
            None => check_version(&installed.path).await?,
        };
        println!(
            "{} [{}] {}\n{version}",
            tool.name,
            installed.source,
            installed.path.display()
        );
        Ok::<_, String>(())
    });
    let mut errors = Vec::new();
    for (tool, result) in requested.iter().zip(join_all(installs).await) {
        if let Err(err) = result {
            errors.push(format!("{}：{err}", tool.name));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}
