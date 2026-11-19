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
        if let Ok(canon_root) = std::fs::canonicalize(&norm_root)
            && let Ok(canon_target) = std::fs::canonicalize(&norm_target)
            && !canon_target.starts_with(&canon_root)
        {
            return Err(format!("不允许访问工作区以外的路径：{raw}"));
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
    normalize_lexical(target).starts_with(abs_root)
}
