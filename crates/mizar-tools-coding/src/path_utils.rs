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
