//! `edit` 工具：对文件进行精准/模糊多 hunk 局部替换，保留 BOM 与 CRLF/LF，并发受写锁队列保护。

use std::path::PathBuf;

use mizar_tools::mutation_queue::with_file_mutation_queue;
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};

use crate::edit_diff::{
    EditOp, apply_edits_to_normalized_content, detect_line_ending, generate_diff, normalize_to_lf,
    restore_line_endings, strip_bom,
};
use crate::path_utils::resolve_to_cwd;

fn extract_edit_op(obj: &serde_json::Map<String, Value>) -> Option<EditOp> {
    let old_text = obj
        .get("oldText")
        .or_else(|| obj.get("old_text"))
        .and_then(Value::as_str)?;
    let new_text = obj
        .get("newText")
        .or_else(|| obj.get("new_text"))
        .and_then(Value::as_str)?;
    Some(EditOp {
        old_text: old_text.to_string(),
        new_text: new_text.to_string(),
    })
}

/// 兼容模型常见的入参变体：`edits` 数组、单个 `edits` 对象、JSON 字符串 `edits`、或顶层 `oldText`/`newText`。
pub fn prepare_edit_arguments(args: &Value) -> Result<(String, Vec<EditOp>), String> {
    let obj = args
        .as_object()
        .ok_or_else(|| "edit 参数必须是 JSON 对象".to_string())?;
    let path = obj
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "缺少 path 参数".to_string())?
        .to_string();

    if let Some(edits_val) = obj.get("edits") {
        let ops: Vec<EditOp> = match edits_val {
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_object().and_then(extract_edit_op))
                .collect(),
            Value::Object(single) => extract_edit_op(single).into_iter().collect(),
            Value::String(raw_json) => {
                let parsed: Value = serde_json::from_str(raw_json)
                    .map_err(|e| format!("edits JSON 字符串解析失败：{e}"))?;
                match parsed {
                    Value::Array(arr) => arr
                        .iter()
                        .filter_map(|v| v.as_object().and_then(extract_edit_op))
                        .collect(),
                    Value::Object(single) => extract_edit_op(&single).into_iter().collect(),
                    _ => Vec::new(),
                }
            }
            _ => Vec::new(),
        };
        if !ops.is_empty() {
            return Ok((path, ops));
        }
    }

    if let Some(legacy) = extract_edit_op(obj) {
        return Ok((path, vec![legacy]));
    }

    Err("缺少有效的 edits（每个元素需包含 oldText 与 newText）".into())
}

pub fn create_edit_tool(cwd: PathBuf, restrict_to_workspace: bool) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "edit",
        "Edit a file by replacing exact text. Supports multi-hunk edits in one call. All edits[].oldText are matched against the original file content and must match uniquely.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to edit (relative or absolute)" },
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "description": "One or more non-overlapping text replacements to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string", "description": "Exact text for this edit to replace" },
                            "newText": { "type": "string", "description": "New text to replace oldText with" }
                        },
                        "required": ["oldText", "newText"]
                    }
                }
            },
            "required": ["path", "edits"]
        }),
        move |args, call_ctx| {
            let cwd = cwd.clone();
            async move {
                if call_ctx.cancel.is_cancelled() {
                    return Err("Operation aborted".into());
                }
                let (raw_path, edits) = prepare_edit_arguments(&args)
                    .map_err(|e| -> cordis::BoxError { e.into() })?;
                let target = resolve_to_cwd(&cwd, &raw_path, restrict_to_workspace)
                    .map_err(|e| -> cordis::BoxError { e.into() })?;
                let edit_count = edits.len();

                let result = with_file_mutation_queue(&target, || async {
                    let raw = tokio::fs::read_to_string(&target).await.map_err(|e| {
                        if e.kind() == std::io::ErrorKind::NotFound {
                            format!("File not found: {}", target.display())
                        } else {
                            e.to_string()
                        }
                    })?;

                    let (bom, content_no_bom) = strip_bom(&raw);
                    let line_ending = detect_line_ending(content_no_bom);
                    let normalized = normalize_to_lf(content_no_bom);
                    let applied = apply_edits_to_normalized_content(&normalized, &edits, &raw_path)?;

                    let final_text = format!(
                        "{bom}{}",
                        restore_line_endings(&applied.new_content, line_ending)
                    );
                    tokio::fs::write(&target, final_text.as_bytes())
                        .await
                        .map_err(|e| e.to_string())?;

                    let (diff, first_changed_line) =
                        generate_diff(&raw_path, &applied.base_content, &applied.new_content, 4);
                    Ok::<_, String>((diff, first_changed_line, applied.used_fuzzy_match))
                })
                .await
                .map_err(|e| -> cordis::BoxError { e.into() })?;

                let (diff, first_changed_line, used_fuzzy_match) = result;
                let structured = json!({
                    "path": raw_path,
                    "edits_applied": edit_count,
                    "first_changed_line": first_changed_line,
                    "used_fuzzy_match": used_fuzzy_match,
                });
                let details = json!({
                    "path": raw_path,
                    "edits_applied": edit_count,
                    "first_changed_line": first_changed_line,
                    "used_fuzzy_match": used_fuzzy_match,
                    "diff": diff,
                });

                Ok(ToolOutput::text(format!(
                    "Successfully replaced {edit_count} edit(s) in {raw_path}."
                ))
                .with_structured(structured)
                .with_details(details))
            }
        },
    )
    .with_execution_mode(ToolExecutionMode::Sequential)
    .with_annotations(ToolAnnotations {
        read_only: false,
        destructive: true,
        idempotent: false,
        open_world: false,
    })
    .with_output_schema(json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "edits_applied": { "type": "integer" },
            "first_changed_line": { "type": ["integer", "null"] },
            "used_fuzzy_match": { "type": "boolean" }
        }
    }))
    .with_prompt_snippet("Make precise edits to existing files")
    .with_prompt_guidelines(vec![
        "Use edit for surgical changes to existing files instead of rewriting the whole file with write."
            .into(),
    ])
}
