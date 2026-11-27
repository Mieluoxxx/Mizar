use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cordis::BoxFuture;

pub type EnsureBinFn =
    Arc<dyn Fn(&'static str) -> BoxFuture<'static, Option<PathBuf>> + Send + Sync>;

pub fn mizar_bin_dir() -> Option<PathBuf> {
    let home = env::var("MIZAR_HOME").ok();
    let root = home
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|dir| dir.join(".mizar")))?;
    std::path::absolute(root.join("bin")).ok()
}

fn executable_in(dir: &Path, name: &str) -> Option<PathBuf> {
    let path = dir.join(format!("{name}{}", env::consts::EXE_SUFFIX));
    let metadata = path.metadata().ok()?;
    if !metadata.is_file() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return None;
        }
    }
    // 工具子进程可能切换工作目录，不能把相对路径交给它解析。
    std::path::absolute(path).ok()
}

pub fn resolve_managed_bin(tool: &str) -> Option<PathBuf> {
    let names: &[&str] = match tool {
        "rg" => &["rg"],
        "fd" => &["fd", "fdfind"],
        _ => return None,
    };
    if let Some(path) = mizar_bin_dir().and_then(|dir| executable_in(&dir, tool)) {
        return Some(path);
    }
    let path = env::var_os("PATH")?;
    names
        .iter()
        .find_map(|name| env::split_paths(&path).find_map(|dir| executable_in(&dir, name)))
}
