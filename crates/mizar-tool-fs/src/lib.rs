//! 只读文件工具：`read_file`、`list_dir`，只允许访问 `root` 目录以内的文件
//!（对应 `NamiDSH/packages/tool-fs`）。

use std::path::{Component, Path, PathBuf};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_tools::{ToolDefinition, ToolsContextExt};
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct ToolFsConfig {
    /// 允许访问的根目录。
    pub root: PathBuf,
    /// 单个文件最多读取多少字节（默认 20_000）。
    pub max_bytes: usize,
}

impl ToolFsConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_bytes: 20_000,
        }
    }
}

/// 纯词法路径规范化（无需目标路径已存在）。
fn normalize_lexical(path: &Path) -> PathBuf {
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

/// 防止 `../` 或绝对路径跳出根目录，同时校验已存在及尚不存在的符号链接后代不逃逸。
pub fn safe_path(root: &Path, req_path: &str) -> Result<PathBuf, String> {
    let abs_root = if root.is_absolute() {
        root.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(root)
    };
    let norm_root = normalize_lexical(&abs_root);
    let candidate = if Path::new(req_path).is_absolute() {
        PathBuf::from(req_path)
    } else {
        norm_root.join(req_path)
    };
    let norm_target = normalize_lexical(&candidate);

    if !norm_target.starts_with(&norm_root) {
        return Err(format!("不允许访问根目录以外的路径：{req_path}"));
    }

    if let Ok(canon_root) = std::fs::canonicalize(&norm_root) {
        let resolved = resolve_canonical_target(&norm_target)
            .map_err(|()| format!("不允许访问根目录以外的路径：{req_path}"))?;
        if let Some(canon_target) = resolved
            && !canon_target.starts_with(&canon_root)
        {
            return Err(format!("不允许访问根目录以外的路径：{req_path}"));
        }
    }

    Ok(norm_target)
}

#[derive(Default, Clone, Copy)]
pub struct ToolFsPlugin;

impl Plugin for ToolFsPlugin {
    type Config = ToolFsConfig;

    fn name(&self) -> Option<&str> {
        Some("tool-fs")
    }

    fn inject(&self) -> Inject {
        Inject::from(["tools"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let tools = ctx.tools()?;
            let root_list = config.root.clone();
            ctx.effect("tool-fs.list_dir", || {
                Ok::<_, mizar_tools::ToolError>(Some(tools.register(
                    ToolDefinition::async_tool(
                        "list_dir",
                        "列出目录里的文件和子目录（路径相对于工作目录）",
                        json!({
                            "type": "object",
                            "properties": {
                                "path": { "type": "string", "description": "目录路径，默认 ." }
                            }
                        }),
                        move |args: Value, _exec| {
                            let root = root_list.clone();
                            async move {
                                let req = args.get("path").and_then(Value::as_str).unwrap_or(".");
                                let full = safe_path(&root, req)?;
                                let mut entries = tokio::fs::read_dir(&full).await?;
                                let mut names = Vec::new();
                                while let Some(entry) = entries.next_entry().await? {
                                    let name = entry.file_name().to_string_lossy().into_owned();
                                    if name.starts_with('.') || name == "node_modules" {
                                        continue;
                                    }
                                    let is_dir = entry
                                        .file_type()
                                        .await
                                        .map(|t| t.is_dir())
                                        .unwrap_or(false);
                                    if is_dir {
                                        names.push(format!("{name}/"));
                                    } else {
                                        names.push(name);
                                    }
                                }
                                names.sort();
                                if names.is_empty() {
                                    Ok::<_, BoxError>("（空目录）".to_string())
                                } else {
                                    Ok(names.join("\n"))
                                }
                            }
                        },
                    ),
                )?))
            })?;

            let root_read = config.root;
            let max_bytes = config.max_bytes;
            ctx.effect("tool-fs.read_file", || {
                Ok::<_, mizar_tools::ToolError>(Some(tools.register(
                    ToolDefinition::async_tool(
                        "read_file",
                        "读取一个文本文件的内容（路径相对于工作目录）",
                        json!({
                            "type": "object",
                            "properties": { "path": { "type": "string" } },
                            "required": ["path"]
                        }),
                        move |args: Value, _exec| {
                            let root = root_read.clone();
                            async move {
                                let req = args
                                    .get("path")
                                    .and_then(Value::as_str)
                                    .ok_or("缺少 path 参数")?;
                                let full = safe_path(&root, req)?;
                                let meta = tokio::fs::metadata(&full).await?;
                                let content = tokio::fs::read_to_string(&full).await?;
                                if meta.len() > max_bytes as u64 {
                                    // 按字符边界安全截断，避免切在多字节 UTF-8 字符中间
                                    let mut end = max_bytes.min(content.len());
                                    while end > 0 && !content.is_char_boundary(end) {
                                        end -= 1;
                                    }
                                    Ok::<_, BoxError>(format!(
                                        "{}\n…（文件太大，只读了前 {} 字节）",
                                        &content[..end],
                                        max_bytes
                                    ))
                                } else {
                                    Ok(content)
                                }
                            }
                        },
                    ),
                )?))
            })?;

            Ok(None)
        })
    }
}
