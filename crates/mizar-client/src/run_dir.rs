//! 每进程运行目录（`~/.mizar/run/`）、进程元信息注册表、死进程清理与单写者会话锁。
//!
//! 对齐 `docs/session-postmortem-ef394bfa.md` 第 6.2 节：
//! - 运行目录权限 `0700`，进程套接字权限 `0600`；
//! - 每个活跃 Mizar 进程原子写入 `p-<pid>.json`（附带 `ppid` / `tty` / `herdr_pane_id` 供 Herdr 精准关联）；
//! - 扫描时通过 `kill(pid, 0)` 自动清理残留死进程的 `.json` / `.sock`；
//! - 同一 `session_id` 任一时刻仅允许一个活进程持有写锁。

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(unix)]
use std::ffi::CStr;
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
#[cfg(unix)]
static TTYNAME_LOCK: Mutex<()> = Mutex::new(());

/// 单个活跃 Mizar 进程在运行目录登记的元信息（`p-<pid>.json`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessMeta {
    pub pid: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ppid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tty: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub herdr_pane_id: Option<String>,
    pub session_id: String,
    pub cwd: String,
    pub socket: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub started_at_ms: u64,
}

impl ProcessMeta {
    /// 采集当前进程的 `pid`、`ppid`、`tty` 与 `herdr_pane_id`，构建标准注册元数据。
    pub fn for_current_process(
        run_dir: &Path,
        session_id: impl Into<String>,
        cwd: impl Into<String>,
        provider: Option<String>,
        model: Option<String>,
    ) -> Self {
        let pid = std::process::id();
        let socket = socket_path_for_pid(run_dir, pid)
            .to_string_lossy()
            .into_owned();
        Self {
            pid,
            ppid: resolve_ppid(),
            tty: resolve_tty(),
            herdr_pane_id: resolve_herdr_pane_id(),
            session_id: session_id.into(),
            cwd: cwd.into(),
            socket,
            provider,
            model,
            started_at_ms: now_unix_ms(),
        }
    }
}

/// 单写者会话锁冲突错误：目标 `session_id` 已被另一活进程持有写权。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("会话 `{session_id}` 已被活进程 PID {owner_pid} 持有写锁（socket: {socket_path}）")]
pub struct SessionLockError {
    pub session_id: String,
    pub owner_pid: u32,
    pub socket_path: String,
}

/// 返回进程元数据文件路径 `<run_dir>/p-<pid>.json`。
pub fn meta_path_for_pid(run_dir: &Path, pid: u32) -> PathBuf {
    run_dir.join(format!("p-{pid}.json"))
}

/// 返回进程 Unix Socket 文件路径 `<run_dir>/p-<pid>.sock`。
pub fn socket_path_for_pid(run_dir: &Path, pid: u32) -> PathBuf {
    run_dir.join(format!("p-{pid}.sock"))
}

/// 解析运行目录路径（优先 `MIZAR_RUN_DIR`，否则 `~/.mizar/run/` 或 `$MIZAR_HOME/run/`），不创建目录。
pub fn resolve_run_dir_path() -> io::Result<PathBuf> {
    if let Some(raw) = std::env::var_os("MIZAR_RUN_DIR") {
        let trimmed = raw.to_string_lossy().trim().to_string();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }
    mizar_core::spine::resolve_global_config_dir()
        .map(|dir| dir.join("run"))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "无法解析 ~/.mizar/run 运行目录路径",
            )
        })
}

/// 确保运行目录存在且具备 Unix `0700` 权限。
pub fn ensure_run_dir(dir: &Path) -> io::Result<()> {
    if dir.exists() {
        if !dir.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("运行目录路径已存在且非目录: {}", dir.display()),
            ));
        }
    } else {
        std::fs::create_dir_all(dir)?;
    }
    #[cfg(unix)]
    {
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// 解析并确保运行目录存在（含 `0700` 权限）。
pub fn resolve_run_dir() -> io::Result<PathBuf> {
    let dir = resolve_run_dir_path()?;
    ensure_run_dir(&dir)?;
    Ok(dir)
}

/// 将 Unix Socket 文件权限收敛为仅当前用户可读写（`0600`）。
pub fn set_socket_permissions_0600(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
    }
    #[cfg(not(unix))]
    {
        std::fs::metadata(path).map(|_| ())
    }
}

/// 通过 `kill(pid, 0)` 检测目标进程是否存活（`0` 或 `EPERM` 为存活，`ESRCH`/无效 PID 为已退出）。
pub fn is_pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        let Ok(pid_t) = libc::pid_t::try_from(pid) else {
            return false;
        };
        // SAFETY: `pid_t > 0`（已排除 `0` 与负值），`signal = 0` 仅检查进程存在性与权限，不投递信号。
        let rc = unsafe { libc::kill(pid_t, 0) };
        if rc == 0 {
            return true;
        }
        matches!(io::Error::last_os_error().raw_os_error(), Some(libc::EPERM))
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// 扫描运行目录中的所有活进程元信息，同时自动清理已退出进程的残留 `.json` 与 `.sock`。
pub fn scan_live_processes(run_dir: &Path) -> io::Result<Vec<ProcessMeta>> {
    if !run_dir.exists() {
        return Ok(Vec::new());
    }
    let mut live = Vec::new();
    for entry in std::fs::read_dir(run_dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };

        if let Some(pid) = parse_pid_filename(name, ".json") {
            if !is_pid_alive(pid) {
                remove_stale_pid_files(run_dir, pid);
                continue;
            }
            let Ok(bytes) = std::fs::read(entry.path()) else {
                continue;
            };
            let Ok(meta) = serde_json::from_slice::<ProcessMeta>(&bytes) else {
                continue;
            };
            if !is_pid_alive(meta.pid) {
                remove_stale_pid_files(run_dir, pid);
                continue;
            }
            live.push(meta);
        } else if let Some(pid) = parse_pid_filename(name, ".sock")
            && !is_pid_alive(pid)
        {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    live.sort_by_key(|m| (m.started_at_ms, m.pid));
    Ok(live)
}

/// 校验指定 `session_id` 是否可由 `self_pid` 写入；若被其他活进程持有则拒绝双写。
pub fn check_session_writable(
    run_dir: &Path,
    session_id: &str,
    self_pid: u32,
) -> Result<(), SessionLockError> {
    let live = scan_live_processes(run_dir).unwrap_or_default();
    for meta in live {
        if meta.session_id == session_id && meta.pid != self_pid {
            return Err(SessionLockError {
                session_id: session_id.to_string(),
                owner_pid: meta.pid,
                socket_path: meta.socket,
            });
        }
    }
    Ok(())
}

/// 运行期进程注册句柄：通过 `tmp + rename` 原子发布 `p-<pid>.json`，并在 `Drop` 时自动清理文件与 Socket。
#[derive(Debug)]
pub struct ProcessRegistration {
    run_dir: PathBuf,
    meta_path: PathBuf,
    socket_path: PathBuf,
    meta: ProcessMeta,
}

impl ProcessRegistration {
    /// 创建运行目录并原子写入 `p-<pid>.json`，返回持有清理职责的注册句柄。
    pub fn create(run_dir: impl Into<PathBuf>, mut meta: ProcessMeta) -> io::Result<Self> {
        let run_dir = run_dir.into();
        ensure_run_dir(&run_dir)?;
        let meta_path = meta_path_for_pid(&run_dir, meta.pid);
        let socket_path = if meta.socket.trim().is_empty() {
            let path = socket_path_for_pid(&run_dir, meta.pid);
            meta.socket = path.to_string_lossy().into_owned();
            path
        } else {
            PathBuf::from(&meta.socket)
        };
        write_meta_atomic(&meta_path, &meta)?;
        Ok(Self {
            run_dir,
            meta_path,
            socket_path,
            meta,
        })
    }

    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    pub fn meta_path(&self) -> &Path {
        &self.meta_path
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub fn meta(&self) -> &ProcessMeta {
        &self.meta
    }

    /// 原子更新活跃 `session_id`（供 `/resume` / `/new` 切换会话时保持 Socket 不断连）。
    pub fn update_session(&mut self, session_id: impl Into<String>) -> io::Result<()> {
        self.meta.session_id = session_id.into();
        write_meta_atomic(&self.meta_path, &self.meta)
    }

    /// 原子更新当前使用的 `provider` 与 `model`。
    pub fn update_model(
        &mut self,
        provider: Option<String>,
        model: Option<String>,
    ) -> io::Result<()> {
        self.meta.provider = provider;
        self.meta.model = model;
        write_meta_atomic(&self.meta_path, &self.meta)
    }
}

impl Drop for ProcessRegistration {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.meta_path);
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

fn write_meta_atomic(meta_path: &Path, meta: &ProcessMeta) -> io::Result<()> {
    let parent = meta_path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("元数据路径缺少父目录: {}", meta_path.display()),
        )
    })?;
    let mut bytes = serde_json::to_vec_pretty(meta)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    bytes.push(b'\n');

    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_name = format!(".p-{}.json.tmp.{}.{}", meta.pid, std::process::id(), seq);
    let tmp_path = parent.join(tmp_name);

    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    opts.mode(0o600);

    let write_res = (|| -> io::Result<()> {
        let mut file = opts.open(&tmp_path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(err) = write_res {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }

    if let Err(err) = std::fs::rename(&tmp_path, meta_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }
    Ok(())
}

fn parse_pid_filename(name: &str, suffix: &str) -> Option<u32> {
    let inner = name.strip_prefix("p-")?.strip_suffix(suffix)?;
    if inner.is_empty() || !inner.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    inner.parse::<u32>().ok().filter(|&pid| pid > 0)
}

fn remove_stale_pid_files(run_dir: &Path, pid: u32) {
    let _ = std::fs::remove_file(meta_path_for_pid(run_dir, pid));
    let _ = std::fs::remove_file(socket_path_for_pid(run_dir, pid));
}

fn resolve_ppid() -> Option<u32> {
    #[cfg(unix)]
    {
        // SAFETY: `getppid` 无输入参数与内存别名，始终返回当前父进程 PID。
        let ppid = unsafe { libc::getppid() };
        (ppid > 0).then_some(ppid as u32)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn resolve_tty() -> Option<String> {
    #[cfg(unix)]
    {
        let _guard = TTYNAME_LOCK.lock().unwrap();
        // SAFETY: `TTYNAME_LOCK` 串行化对 libc 内部静态缓冲区的访问，且返回指针在锁内立即拷贝为 `String`。
        let ptr = unsafe { libc::ttyname(libc::STDIN_FILENO) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ttyname` 非空时返回以 `\0` 结尾的 C 字符串，生命周期持续至下次 `ttyname` 调用（受 `TTYNAME_LOCK` 保护）。
        let name = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .trim()
            .to_string();
        (!name.is_empty()).then_some(name)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn resolve_herdr_pane_id() -> Option<String> {
    for key in ["HERDR_PANE_ID", "HERDR_PANE"] {
        if let Some(raw) = std::env::var_os(key) {
            let trimmed = raw.to_string_lossy().trim().to_string();
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    None
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, Command, Stdio};

    static ENV_LOCK: Mutex<()> = Mutex::new(());
    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_run_dir(tag: &str) -> PathBuf {
        let seq = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "mizar_run_dir_{tag}_{}_{}",
            std::process::id(),
            seq
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    struct TempDirGuard(PathBuf);
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct ChildGuard(Option<Child>);
    impl ChildGuard {
        fn spawn_sleeper() -> Self {
            let child = Command::new("sleep")
                .arg("30")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn sleep helper");
            Self(Some(child))
        }

        fn pid(&self) -> u32 {
            self.0.as_ref().expect("child still alive").id()
        }

        fn kill_and_wait(&mut self) -> u32 {
            let mut child = self.0.take().expect("child already reaped");
            let pid = child.id();
            let _ = child.kill();
            let _ = child.wait();
            pid
        }
    }

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    #[test]
    fn resolve_run_dir_respects_override_and_sets_0700_permissions() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = temp_run_dir("override");
        let _dir_guard = TempDirGuard(dir.clone());

        let prev_run = std::env::var_os("MIZAR_RUN_DIR");
        let prev_home = std::env::var_os("MIZAR_HOME");
        let prev_pane_id = std::env::var_os("HERDR_PANE_ID");
        let prev_pane = std::env::var_os("HERDR_PANE");
        // SAFETY: 持有 `ENV_LOCK` 互斥锁，测试期间无并发环境读写。
        unsafe {
            std::env::set_var("MIZAR_RUN_DIR", &dir);
            std::env::set_var("HERDR_PANE_ID", "pane_42");
        }

        let resolved = resolve_run_dir().unwrap();
        assert_eq!(resolved, dir);
        assert!(resolved.is_dir());

        #[cfg(unix)]
        {
            let mode = std::fs::metadata(&resolved).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700);
        }

        let meta = ProcessMeta::for_current_process(
            &resolved,
            "sess-1",
            "/ws",
            Some("magpie".into()),
            Some("gpt-6".into()),
        );
        assert_eq!(meta.pid, std::process::id());
        assert_eq!(meta.herdr_pane_id.as_deref(), Some("pane_42"));
        #[cfg(unix)]
        assert!(meta.ppid.is_some());

        // 未设置 `MIZAR_RUN_DIR` 时回退到 `$MIZAR_HOME/run`，未设置 `HERDR_PANE_ID` 时回退 `HERDR_PANE`。
        let mizar_home = temp_run_dir("mizar_home");
        let _home_guard = TempDirGuard(mizar_home.clone());
        // SAFETY: 持有 `ENV_LOCK` 互斥锁，测试期间无并发环境读写。
        unsafe {
            std::env::remove_var("MIZAR_RUN_DIR");
            std::env::set_var("MIZAR_HOME", &mizar_home);
            std::env::remove_var("HERDR_PANE_ID");
            std::env::set_var("HERDR_PANE", "pane_legacy");
        }
        let fallback_run = resolve_run_dir().unwrap();
        assert_eq!(fallback_run, mizar_home.join("run"));
        assert_eq!(resolve_herdr_pane_id().as_deref(), Some("pane_legacy"));

        // SAFETY: 持有 `ENV_LOCK` 互斥锁恢复环境变量。
        unsafe {
            match prev_run {
                Some(v) => std::env::set_var("MIZAR_RUN_DIR", v),
                None => std::env::remove_var("MIZAR_RUN_DIR"),
            }
            match prev_home {
                Some(v) => std::env::set_var("MIZAR_HOME", v),
                None => std::env::remove_var("MIZAR_HOME"),
            }
            match prev_pane_id {
                Some(v) => std::env::set_var("HERDR_PANE_ID", v),
                None => std::env::remove_var("HERDR_PANE_ID"),
            }
            match prev_pane {
                Some(v) => std::env::set_var("HERDR_PANE", v),
                None => std::env::remove_var("HERDR_PANE"),
            }
        }
    }

    #[test]
    fn socket_permissions_0600_and_pid_liveness() {
        let dir = temp_run_dir("perms");
        let _dir_guard = TempDirGuard(dir.clone());
        ensure_run_dir(&dir).unwrap();

        let sock_path = socket_path_for_pid(&dir, std::process::id());
        std::fs::write(&sock_path, b"").unwrap();
        set_socket_permissions_0600(&sock_path).unwrap();

        #[cfg(unix)]
        {
            let mode = std::fs::metadata(&sock_path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }

        assert!(is_pid_alive(std::process::id()));
        assert!(!is_pid_alive(0));
        assert!(!is_pid_alive(u32::MAX));

        let mut child = Command::new("true").spawn().expect("spawn true");
        let dead_pid = child.id();
        let _ = child.wait();
        assert!(!is_pid_alive(dead_pid));
    }

    #[test]
    fn process_registration_atomic_updates_and_drop_cleanup() {
        let dir = temp_run_dir("reg");
        let _dir_guard = TempDirGuard(dir.clone());

        let pid = std::process::id();
        let meta = ProcessMeta {
            pid,
            ppid: Some(1),
            tty: Some("/dev/ttys001".into()),
            herdr_pane_id: Some("pane_1".into()),
            session_id: "sess-a".into(),
            cwd: "/repo".into(),
            socket: socket_path_for_pid(&dir, pid)
                .to_string_lossy()
                .into_owned(),
            provider: Some("magpie".into()),
            model: Some("gpt-a".into()),
            started_at_ms: 1000,
        };

        let meta_path;
        let sock_path;
        {
            let mut reg = ProcessRegistration::create(&dir, meta).unwrap();
            meta_path = reg.meta_path().to_path_buf();
            sock_path = reg.socket_path().to_path_buf();

            std::fs::write(&sock_path, b"").unwrap();
            set_socket_permissions_0600(&sock_path).unwrap();
            assert!(meta_path.exists());
            assert!(sock_path.exists());

            reg.update_session("sess-b").unwrap();
            reg.update_model(Some("isok".into()), Some("gpt-b".into()))
                .unwrap();

            let parsed: ProcessMeta =
                serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
            assert_eq!(parsed.session_id, "sess-b");
            assert_eq!(parsed.provider.as_deref(), Some("isok"));
            assert_eq!(parsed.model.as_deref(), Some("gpt-b"));

            // 原子替换不得残留临时文件。
            let leftovers: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
                .collect();
            assert!(leftovers.is_empty(), "leftover tmp files: {leftovers:?}");
        }

        assert!(!meta_path.exists());
        assert!(!sock_path.exists());
    }

    #[test]
    fn scan_cleans_dead_pids_and_enforces_single_writer_lock() {
        let dir = temp_run_dir("lock");
        let _dir_guard = TempDirGuard(dir.clone());
        ensure_run_dir(&dir).unwrap();

        // 1. 构造已退出子进程的残留文件，验证扫描自动清理。
        let mut stale_child = Command::new("true").spawn().expect("spawn true");
        let dead_pid = stale_child.id();
        let _ = stale_child.wait();

        let stale_meta = ProcessMeta {
            pid: dead_pid,
            ppid: None,
            tty: None,
            herdr_pane_id: None,
            session_id: "sess-target".into(),
            cwd: "/repo".into(),
            socket: socket_path_for_pid(&dir, dead_pid)
                .to_string_lossy()
                .into_owned(),
            provider: None,
            model: None,
            started_at_ms: 100,
        };
        let stale_json = meta_path_for_pid(&dir, dead_pid);
        let stale_sock = socket_path_for_pid(&dir, dead_pid);
        std::fs::write(&stale_json, serde_json::to_vec(&stale_meta).unwrap()).unwrap();
        std::fs::write(&stale_sock, b"").unwrap();

        let live = scan_live_processes(&dir).unwrap();
        assert!(live.is_empty());
        assert!(!stale_json.exists());
        assert!(!stale_sock.exists());

        // 2. 启动存活子进程作为单写者锁持有者，验证锁冲突与死亡后自动释放。
        let mut owner = ChildGuard::spawn_sleeper();
        let owner_pid = owner.pid();
        let owner_meta = ProcessMeta {
            pid: owner_pid,
            ppid: Some(std::process::id()),
            tty: None,
            herdr_pane_id: None,
            session_id: "sess-target".into(),
            cwd: "/repo".into(),
            socket: socket_path_for_pid(&dir, owner_pid)
                .to_string_lossy()
                .into_owned(),
            provider: Some("magpie".into()),
            model: Some("gpt-6".into()),
            started_at_ms: 200,
        };
        let _owner_reg = ProcessRegistration::create(&dir, owner_meta.clone()).unwrap();

        let live_after = scan_live_processes(&dir).unwrap();
        assert_eq!(live_after, vec![owner_meta]);

        // 本进程自己检查（同 PID）应放行；其他 PID 检查同一会话必须报错。
        assert!(check_session_writable(&dir, "sess-target", owner_pid).is_ok());
        assert!(check_session_writable(&dir, "sess-other", std::process::id()).is_ok());

        let err = check_session_writable(&dir, "sess-target", std::process::id()).unwrap_err();
        assert_eq!(err.session_id, "sess-target");
        assert_eq!(err.owner_pid, owner_pid);
        assert_eq!(
            err.socket_path,
            socket_path_for_pid(&dir, owner_pid).to_string_lossy()
        );

        // 终止锁持有者后，再次检查应自动清理旧文件并放行。
        let reaped_pid = owner.kill_and_wait();
        assert_eq!(reaped_pid, owner_pid);
        assert!(!is_pid_alive(owner_pid));
        assert!(check_session_writable(&dir, "sess-target", std::process::id()).is_ok());
        assert!(!meta_path_for_pid(&dir, owner_pid).exists());
    }
}
