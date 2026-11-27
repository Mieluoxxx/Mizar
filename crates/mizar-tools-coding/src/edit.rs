//! `edit` 工具：对文件进行 Pi 式多段原子精确/模糊局部替换（支持 `edits[]` 与 `replaceAll`），保留 BOM 与 CRLF/LF，并发受写锁队列保护。

use std::path::PathBuf;

use mizar_tools::mutation_queue::with_file_mutation_queue;
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};

use crate::edit_diff::{
    EDIT_DIFF_MAX_BYTES, EDIT_DIFF_MAX_LINES, EditOp, apply_edits_to_normalized_content,
    detect_line_ending, format_compact_diff, generate_diff, normalize_to_lf, restore_line_endings,
    strip_bom,
};
use crate::path_utils::resolve_to_cwd;

fn extract_edit_op(
    obj: &serde_json::Map<String, Value>,
    default_replace_all: bool,
) -> Result<EditOp, String> {
    let old_text = obj
        .get("oldText")
        .or_else(|| obj.get("old_text"))
        .and_then(Value::as_str)
        .ok_or_else(|| "Each edit must include a string oldText.".to_string())?;
    let new_text = obj
        .get("newText")
        .or_else(|| obj.get("new_text"))
        .and_then(Value::as_str)
        .ok_or_else(|| "Each edit must include a string newText.".to_string())?;
    let replace_all = obj
        .get("replaceAll")
        .or_else(|| obj.get("replace_all"))
        .and_then(Value::as_bool)
        .unwrap_or(default_replace_all);
    Ok(EditOp {
        old_text: old_text.to_string(),
        new_text: new_text.to_string(),
        replace_all,
    })
}

fn parse_edits_value(val: &Value, default_replace_all: bool) -> Result<Vec<EditOp>, String> {
    match val {
        Value::Array(arr) => {
            let mut ops = Vec::with_capacity(arr.len());
            for (idx, item) in arr.iter().enumerate() {
                let obj = item
                    .as_object()
                    .ok_or_else(|| format!("edits[{idx}] must be a JSON object."))?;
                ops.push(
                    extract_edit_op(obj, default_replace_all)
                        .map_err(|e| format!("Invalid edits[{idx}]: {e}"))?,
                );
            }
            Ok(ops)
        }
        Value::Object(single) => Ok(vec![extract_edit_op(single, default_replace_all)?]),
        _ => Err("edits must be a non-empty array of { oldText, newText } objects.".into()),
    }
}

/// 兼容模型常见的入参变体：`edits` 数组、单个 `edits` 对象、JSON 字符串 `edits`、或顶层 `oldText`/`newText`。
pub fn prepare_edit_arguments(args: &Value) -> Result<(String, Vec<EditOp>), String> {
    let obj = args
        .as_object()
        .ok_or_else(|| "edit arguments must be a JSON object.".to_string())?;
    let path = obj
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "Missing required parameter: path".to_string())?
        .to_string();
    let default_replace_all = obj
        .get("replaceAll")
        .or_else(|| obj.get("replace_all"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if let Some(edits_val) = obj.get("edits") {
        let mut ops = match edits_val {
            Value::String(raw_json) => {
                let parsed: Value = serde_json::from_str(raw_json)
                    .map_err(|e| format!("Failed to parse edits JSON string: {e}"))?;
                parse_edits_value(&parsed, default_replace_all)?
            }
            other => parse_edits_value(other, default_replace_all)?,
        };
        if let Ok(legacy) = extract_edit_op(obj, default_replace_all) {
            ops.push(legacy);
        }
        if ops.is_empty() {
            return Err("edits must contain at least one replacement.".into());
        }
        return Ok((path, ops));
    }

    if let Ok(legacy) = extract_edit_op(obj, default_replace_all) {
        return Ok((path, vec![legacy]));
    }

    Err("Missing required edits (provide edits: [{ oldText, newText }]).".into())
}

pub fn create_edit_tool(cwd: PathBuf, restrict_to_workspace: bool) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "edit",
        "Edit a single file using exact text replacement. Supports multiple disjoint edits in one atomic call via edits[]. Every edits[].oldText is matched against the original file and must be unique unless replaceAll=true.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to edit (relative or absolute)" },
                "replaceAll": { "type": "boolean", "description": "Replace all occurrences when oldText appears multiple times (default: false)" },
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "description": "One or more non-overlapping targeted replacements matched against the original file",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string", "description": "Exact text in the original file to replace" },
                            "newText": { "type": "string", "description": "Replacement text" },
                            "replaceAll": { "type": "boolean", "description": "Replace all occurrences of this oldText (overrides top-level replaceAll)" }
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
                    let applied =
                        apply_edits_to_normalized_content(&normalized, &edits, &raw_path)?;

                    // 先计算 diff 再落盘，防止生成阶段出错留下半成品状态。
                    let (diff, first_changed_line) =
                        generate_diff(&raw_path, &applied.base_content, &applied.new_content, 4);

                    let final_text = format!(
                        "{bom}{}",
                        restore_line_endings(&applied.new_content, line_ending)
                    );
                    tokio::fs::write(&target, final_text.as_bytes())
                        .await
                        .map_err(|e| e.to_string())?;

                    Ok::<_, String>((
                        diff,
                        first_changed_line,
                        applied.used_fuzzy_match,
                        applied.replacements_applied,
                    ))
                })
                .await
                .map_err(|e| -> cordis::BoxError { e.into() })?;

                let (diff, first_changed_line, used_fuzzy_match, replacements_applied) = result;
                let (compact_diff, diff_truncated) =
                    format_compact_diff(&diff, EDIT_DIFF_MAX_LINES, EDIT_DIFF_MAX_BYTES);

                let structured = json!({
                    "path": raw_path,
                    "edits_applied": edit_count,
                    "replacements_applied": replacements_applied,
                    "first_changed_line": first_changed_line,
                    "used_fuzzy_match": used_fuzzy_match,
                    "diff_truncated": diff_truncated,
                });
                let details = json!({
                    "path": raw_path,
                    "edits_applied": edit_count,
                    "replacements_applied": replacements_applied,
                    "first_changed_line": first_changed_line,
                    "used_fuzzy_match": used_fuzzy_match,
                    "diff": diff,
                });

                let mut summary = if replacements_applied == edit_count {
                    format!("Successfully replaced {edit_count} block(s) in {raw_path}.")
                } else {
                    format!(
                        "Successfully applied {edit_count} edit(s) ({replacements_applied} replacement(s)) in {raw_path}."
                    )
                };
                if !compact_diff.is_empty() {
                    summary.push_str("\n\n");
                    summary.push_str(&compact_diff);
                }

                Ok(ToolOutput::text(summary)
                    .with_structured(structured)
                    .with_details(details))
            }
        },
    )
    .with_execution_mode(ToolExecutionMode::Parallel)
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
            "replacements_applied": { "type": "integer" },
            "first_changed_line": { "type": ["integer", "null"] },
            "used_fuzzy_match": { "type": "boolean" },
            "diff_truncated": { "type": "boolean" }
        }
    }))
    .with_prompt_snippet("Make precise file edits with exact text replacement, including multiple disjoint edits in one call")
    .with_prompt_guidelines(vec![
        "Use edit for targeted changes. Batch changes to one file in a single call.".into(),
        "Each edits[].oldText must exactly and uniquely match the original file. Keep matches short and non-overlapping; merge nearby changes without including large unchanged regions.".into(),
    ])
}
