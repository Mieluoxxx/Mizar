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
                if !out.pop() {
                    out.push(Component::ParentDir);
                }
            }
            Component::Normal(s) => out.push(s),
        }
    }
    out
}

/// 防止 `../` 或绝对路径跳出根目录，同时校验已存在路径的符号链接不逃逸。
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

    if let Ok(canon_root) = std::fs::canonicalize(&norm_root)
        && let Ok(canon_target) = std::fs::canonicalize(&norm_target)
        && !canon_target.starts_with(&canon_root)
    {
        return Err(format!("不允许访问根目录以外的路径：{req_path}"));
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
                Ok::<_, mizar_tools::ToolError>(Some(tools.register(ToolDefinition::async_tool(
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
                ))?))
            })?;

            let root_read = config.root;
            let max_bytes = config.max_bytes;
            ctx.effect("tool-fs.read_file", || {
                Ok::<_, mizar_tools::ToolError>(Some(tools.register(ToolDefinition::async_tool(
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
                ))?))
            })?;

            Ok(None)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cordis::{Context, Service};
    use mizar_tools::{ToolExecution, ToolRuntime, ToolsContextExt};

    #[test]
    fn blocks_path_traversal() {
        let root = Path::new("/tmp/mizar-test-root");
        assert!(safe_path(root, ".").is_ok());
        assert!(safe_path(root, "src/main.rs").is_ok());
        assert!(safe_path(root, "../secret.txt").is_err());
        assert!(safe_path(root, "/etc/passwd").is_err());
    }

    #[tokio::test]
    async fn lists_dirs_and_truncates_utf8_files() {
        let dir = std::env::temp_dir().join(format!("mizar_tool_fs_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::create_dir_all(dir.join(".hidden")).unwrap();
        std::fs::create_dir_all(dir.join("node_modules")).unwrap();
        std::fs::write(dir.join("hello.txt"), "你好世界12345").unwrap();

        let app = Context::new();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app.plugin(
            ToolFsPlugin,
            ToolFsConfig {
                root: dir.clone(),
                max_bytes: 7,
            },
        )
        .await
        .unwrap();

        let tools = app.tools().unwrap();
        let list_res = tools
            .execute(ToolExecution {
                name: "list_dir".into(),
                args: json!({ "path": "." }),
                session_id: None,
            })
            .await;
        assert!(!list_res.is_error);
        assert_eq!(list_res.content, "hello.txt\nsub/");

        let read_res = tools
            .execute(ToolExecution {
                name: "read_file".into(),
                args: json!({ "path": "hello.txt" }),
                session_id: None,
            })
            .await;
        assert!(!read_res.is_error);
        assert!(read_res.content.starts_with("你好"));
        assert!(read_res.content.contains("只读了前 7 字节"));

        let escape_res = tools
            .execute(ToolExecution {
                name: "read_file".into(),
                args: json!({ "path": "../outside.txt" }),
                session_id: None,
            })
            .await;
        assert!(escape_res.is_error);
        assert!(escape_res.content.contains("不允许访问根目录以外的路径"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
