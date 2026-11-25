//! 路径规范化与解析辅助函数（兼容 `~`、`@` 前缀及 macOS 常见 Unicode 空格变体）。

use std::path::{Component, Path, PathBuf};

fn normalize_unicode_spaces(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            _ => c,
        })
        .collect()
}

pub fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(out.components().next_back(), Some(Component::Normal(_))) {
                    out.pop();
                } else if !matches!(
                    out.components().next_back(),
                    Some(Component::RootDir | Component::Prefix(_))
                ) {
                    out.push(Component::ParentDir);
                }
            }
            Component::Normal(s) => out.push(s),
        }
    }
    out
}

pub fn expand_user_path(raw: &str) -> PathBuf {
    let trimmed = normalize_unicode_spaces(raw.trim());
    let without_at = trimmed.strip_prefix('@').unwrap_or(&trimmed);
    if without_at == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = without_at.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(without_at)
}

/// 向上追溯至最近已存在祖先并解析真实规范路径（支持尚不存在的尾路径及悬空/循环符号链接）。
fn resolve_canonical_target(target: &Path) -> Result<Option<PathBuf>, ()> {
    let mut cur = target.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut hops = 0usize;

    loop {
        if let Ok(canon) = std::fs::canonicalize(&cur) {
            let mut rebuilt = canon;
            for seg in tail.iter().rev() {
                rebuilt.push(seg);
            }
            return Ok(Some(normalize_lexical(&rebuilt)));
        }

        if let Ok(meta) = std::fs::symlink_metadata(&cur)
            && meta.file_type().is_symlink()
        {
            if hops >= 40 {
                return Err(());
            }
            let link_target = std::fs::read_link(&cur).map_err(|_| ())?;
            let base = cur.parent().unwrap_or_else(|| Path::new("/"));
            let mut next = if link_target.is_absolute() {
                link_target
            } else {
                base.join(link_target)
            };
            for seg in tail.iter().rev() {
                next.push(seg);
            }
            cur = normalize_lexical(&next);
            tail.clear();
            hops += 1;
            continue;
        }

        let Some(name) = cur.file_name() else {
            return Ok(None);
        };
        tail.push(name.to_os_string());
        let Some(parent) = cur.parent() else {
            return Ok(None);
        };
        cur = parent.to_path_buf();
    }
}

pub fn resolve_to_cwd(
    cwd: &Path,
    raw: &str,
    restrict_to_workspace: bool,
) -> Result<PathBuf, String> {
    let abs_cwd = if cwd.is_absolute() {
        cwd.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(cwd)
    };
    let norm_root = normalize_lexical(&abs_cwd);
    let expanded = expand_user_path(raw);
    let candidate = if expanded.is_absolute() {
        expanded
    } else {
        norm_root.join(expanded)
    };
    let norm_target = normalize_lexical(&candidate);

    if restrict_to_workspace {
        if !norm_target.starts_with(&norm_root) {
            return Err(format!("不允许访问工作区以外的路径：{raw}"));
        }
        if let Ok(canon_root) = std::fs::canonicalize(&norm_root) {
            let resolved = resolve_canonical_target(&norm_target)
                .map_err(|()| format!("不允许访问工作区以外的路径：{raw}"))?;
            if let Some(canon_target) = resolved
                && !canon_target.starts_with(&canon_root)
            {
                return Err(format!("不允许访问工作区以外的路径：{raw}"));
            }
        }
    }

    Ok(norm_target)
}

pub async fn resolve_read_path(
    cwd: &Path,
    raw: &str,
    restrict_to_workspace: bool,
) -> Result<PathBuf, String> {
    let primary = resolve_to_cwd(cwd, raw, restrict_to_workspace)?;
    if tokio::fs::try_exists(&primary).await.unwrap_or(false) {
        return Ok(primary);
    }

    let s = primary.to_string_lossy();
    let am_pm = s
        .replace(" AM.", "\u{202F}AM.")
        .replace(" PM.", "\u{202F}PM.");
    if am_pm != s {
        let alt = PathBuf::from(&am_pm);
        if tokio::fs::try_exists(&alt).await.unwrap_or(false)
            && (!restrict_to_workspace || is_within_root(cwd, &alt))
        {
            return Ok(alt);
        }
    }

    Ok(primary)
}

fn is_within_root(root: &Path, target: &Path) -> bool {
    let abs_root = if root.is_absolute() {
        normalize_lexical(root)
    } else {
        std::env::current_dir()
            .map(|c| normalize_lexical(&c.join(root)))
            .unwrap_or_else(|_| normalize_lexical(root))
    };
    let norm_target = normalize_lexical(target);
    if !norm_target.starts_with(&abs_root) {
        return false;
    }
    if let Ok(canon_root) = std::fs::canonicalize(&abs_root) {
        match resolve_canonical_target(&norm_target) {
            Ok(Some(canon_target)) => canon_target.starts_with(&canon_root),
            Ok(None) => true,
            Err(()) => false,
        }
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn restrict_to_workspace_rejects_existing_and_non_existing_symlink_escapes() {
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base =
            std::env::temp_dir().join(format!("mizar_path_utils_{}_{uniq}", std::process::id()));
        let ws = base.join("ws");
        let outside = base.join("outside");
        std::fs::create_dir_all(ws.join("sub")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "top-secret").unwrap();

        std::os::unix::fs::symlink(&outside, ws.join("link")).unwrap();
        std::os::unix::fs::symlink(outside.join("not_created"), ws.join("dangling")).unwrap();
        std::os::unix::fs::symlink(ws.join("sub"), ws.join("in_link")).unwrap();
        std::os::unix::fs::symlink(ws.join("loop_b"), ws.join("loop_a")).unwrap();
        std::os::unix::fs::symlink(ws.join("loop_a"), ws.join("loop_b")).unwrap();
        std::os::unix::fs::symlink(ws.join("self_loop"), ws.join("self_loop")).unwrap();

        // 已存在目标逃逸
        assert!(resolve_to_cwd(&ws, "link/secret.txt", true).is_err());
        // 尚不存在的尾路径（单层与多层）通过符号链接逃逸
        assert!(resolve_to_cwd(&ws, "link/new.txt", true).is_err());
        assert!(resolve_to_cwd(&ws, "link/a/b/new.txt", true).is_err());
        // 悬空符号链接逃逸
        assert!(resolve_to_cwd(&ws, "dangling", true).is_err());
        assert!(resolve_to_cwd(&ws, "dangling/sub.txt", true).is_err());
        // 循环符号链接（互循环与自循环，含子路径）必须被拒绝且不死循环
        assert!(resolve_to_cwd(&ws, "loop_a", true).is_err());
        assert!(resolve_to_cwd(&ws, "loop_a/child.txt", true).is_err());
        assert!(resolve_to_cwd(&ws, "self_loop/child.txt", true).is_err());

        // 指向工作区内部的符号链接（含未创建文件）应放行
        assert!(resolve_to_cwd(&ws, "in_link/new.txt", true).is_ok());

        // 非限制模式应放行外部符号链接
        assert!(resolve_to_cwd(&ws, "link/new.txt", false).is_ok());

        let _ = std::fs::remove_dir_all(&base);
    }
}
