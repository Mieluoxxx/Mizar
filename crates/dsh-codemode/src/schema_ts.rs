//! JSON Schema 到 TypeScript 类型声明渲染（直接对齐 `pi-codemode` `declarations.ts` 与 `describeOutput`）。

use std::collections::HashSet;

use dsh_tools::ToolRuntimeInfo;
use serde_json::Value;

use crate::declarations::to_codemode_identifier;

const INDENT: &str = "  ";
/// 单个输入参数类型渲染的默认最大字符数；超出后会先剥离属性注释重试，仍超出则回退为 `unknown`。
pub const DEFAULT_INPUT_SCHEMA_MAX_CHARS: usize = 16_000;
/// 单个 schema 内允许展开本地 `$ref` 的最大次数，防止共享定义放大输出。
const MAX_REF_EXPANSIONS: usize = 32;
/// 单个属性标签最大长度；过长的默认示例或正则直接略过。
const MAX_TAG_CHARS: usize = 80;

const TAG_KEYWORDS: &[&str] = &[
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
];

fn is_lower_bound(keyword: &str) -> bool {
    matches!(
        keyword,
        "minimum" | "exclusiveMinimum" | "minLength" | "minItems"
    )
}

fn is_upper_bound(keyword: &str) -> bool {
    matches!(
        keyword,
        "maximum" | "exclusiveMaximum" | "maxLength" | "maxItems"
    )
}

fn is_valid_js_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_' || first == '$') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

fn property_key(name: &str) -> String {
    if is_valid_js_identifier(name) {
        name.to_string()
    } else {
        serde_json::to_string(name).unwrap_or_else(|_| format!("\"{name}\""))
    }
}

fn union_types(types: Vec<String>) -> String {
    let mut unique = Vec::new();
    for ty in types {
        if !unique.contains(&ty) {
            unique.push(ty);
        }
    }
    if unique.iter().any(|t| t == "unknown") {
        return "unknown".to_string();
    }
    if unique.is_empty() {
        "never".to_string()
    } else {
        unique.join(" | ")
    }
}

fn percent_decode(segment: &str) -> Option<String> {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return None;
            }
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
            let val = u8::from_str_radix(hex, 16).ok()?;
            out.push(val);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// 解析本地 JSON Pointer `$ref`（`#` 或 `#/$defs/...`），忽略远程引用和格式错误的转义。
pub fn resolve_schema_ref<'a>(ref_str: &str, root: &'a Value) -> Option<&'a Value> {
    if ref_str == "#" {
        return matches!(root, Value::Bool(_) | Value::Object(_)).then_some(root);
    }
    let tail = ref_str.strip_prefix("#/")?;
    let mut current = root;
    for segment in tail.split('/').filter(|s| !s.is_empty()) {
        let decoded = percent_decode(segment)?;
        let key = decoded.replace("~1", "/").replace("~0", "~");
        current = current.as_object()?.get(&key)?;
    }
    matches!(current, Value::Bool(_) | Value::Object(_)).then_some(current)
}

struct SchemaContext<'a> {
    root: &'a Value,
    resolving: HashSet<String>,
    expansions: usize,
    tags: bool,
}

/// 将 JSON Schema 转为紧凑 TypeScript 类型表达式，超长时先剥离注释标签再退化为 `unknown`。
pub fn schema_to_type(schema: &Value, max_chars: Option<usize>) -> String {
    let render = |tags: bool| -> String {
        let mut ctx = SchemaContext {
            root: schema,
            resolving: HashSet::new(),
            expansions: 0,
            tags,
        };
        to_type(schema, &mut ctx)
    };
    let rendered = render(true);
    let Some(limit) = max_chars else {
        return rendered;
    };
    if rendered.chars().count() <= limit {
        return rendered;
    }
    let untagged = render(false);
    if untagged.chars().count() <= limit {
        untagged
    } else {
        "unknown".to_string()
    }
}

fn to_type(schema: &Value, ctx: &mut SchemaContext<'_>) -> String {
    if let Some(b) = schema.as_bool() {
        return if b { "unknown" } else { "never" }.to_string();
    }
    let Some(obj) = schema.as_object() else {
        return "unknown".to_string();
    };

    if let Some(ref_str) = obj.get("$ref").and_then(Value::as_str) {
        if ctx.resolving.contains(ref_str) || ctx.expansions >= MAX_REF_EXPANSIONS {
            return "unknown".to_string();
        }
        let Some(target) = resolve_schema_ref(ref_str, ctx.root) else {
            return "unknown".to_string();
        };
        ctx.expansions += 1;
        ctx.resolving.insert(ref_str.to_string());
        let resolved = to_type(target, ctx);
        ctx.resolving.remove(ref_str);
        return resolved;
    }

    if let Some(const_val) = obj.get("const") {
        return serde_json::to_string(const_val).unwrap_or_else(|_| "unknown".to_string());
    }
    if let Some(enum_vals) = obj.get("enum").and_then(Value::as_array) {
        let variants = enum_vals
            .iter()
            .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "unknown".to_string()))
            .collect();
        return union_types(variants);
    }

    if let Some(variants) = obj
        .get("anyOf")
        .or_else(|| obj.get("oneOf"))
        .and_then(Value::as_array)
    {
        let mapped = variants.iter().map(|v| to_type(v, ctx)).collect();
        return union_types(mapped);
    }

    if let Some(all_of) = obj.get("allOf").and_then(Value::as_array) {
        let parts: Vec<String> = all_of
            .iter()
            .map(|part| to_type(part, ctx))
            .filter(|part| part != "unknown")
            .collect();
        if parts.is_empty() {
            return "unknown".to_string();
        }
        return parts
            .into_iter()
            .map(|part| {
                if part.contains(" | ") {
                    format!("({part})")
                } else {
                    part
                }
            })
            .collect::<Vec<_>>()
            .join(" & ");
    }

    if let Some(types) = obj.get("type").and_then(Value::as_array) {
        let variants = types
            .iter()
            .map(|entry| {
                let mut clone = obj.clone();
                clone.insert("type".to_string(), entry.clone());
                to_type(&Value::Object(clone), ctx)
            })
            .collect();
        return union_types(variants);
    }

    match obj.get("type").and_then(Value::as_str) {
        Some("string") => "string".to_string(),
        Some("number") | Some("integer") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some("null") => "null".to_string(),
        Some("array") => array_type(obj, ctx),
        Some("object") => object_type(obj, ctx),
        None => {
            if obj.contains_key("properties")
                || obj.contains_key("additionalProperties")
                || obj.contains_key("required")
            {
                object_type(obj, ctx)
            } else if obj.contains_key("items") || obj.contains_key("prefixItems") {
                array_type(obj, ctx)
            } else {
                "unknown".to_string()
            }
        }
        Some(_) => "unknown".to_string(),
    }
}

fn array_type(obj: &serde_json::Map<String, Value>, ctx: &mut SchemaContext<'_>) -> String {
    if let Some(items) = obj.get("items")
        && !items.is_array()
    {
        return format!("Array<{}>", to_type(items, ctx));
    }
    let tuple = obj
        .get("prefixItems")
        .and_then(Value::as_array)
        .or_else(|| obj.get("items").and_then(Value::as_array));
    if let Some(items) = tuple
        && !items.is_empty()
    {
        let elems: Vec<String> = items.iter().map(|item| to_type(item, ctx)).collect();
        return format!("[{}]", elems.join(", "));
    }
    "unknown[]".to_string()
}

fn split_comment_lines(raw: &str) -> Vec<String> {
    raw.trim()
        .split(['\r', '\n', '\u{2028}', '\u{2029}'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn sanitize_line_terminators(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\r' => out.push_str("\\u000d"),
            '\n' => out.push_str("\\u000a"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            _ => out.push(ch),
        }
    }
    out
}

fn normalize_draft04_bounds(obj: &serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
    let mut out = obj.clone();
    if obj.get("exclusiveMinimum").and_then(Value::as_bool) == Some(true) {
        if let Some(min_val) = obj.get("minimum").cloned() {
            out.insert("exclusiveMinimum".to_string(), min_val);
        }
        out.remove("minimum");
    }
    if obj.get("exclusiveMaximum").and_then(Value::as_bool) == Some(true) {
        if let Some(max_val) = obj.get("maximum").cloned() {
            out.insert("exclusiveMaximum".to_string(), max_val);
        }
        out.remove("maximum");
    }
    out
}

fn has_integer_type(val: Option<&Value>) -> bool {
    match val {
        Some(Value::String(s)) => s == "integer",
        Some(Value::Array(arr)) => arr.iter().any(|v| v.as_str() == Some("integer")),
        _ => false,
    }
}

fn constraint_tags(schema: &Value) -> Vec<String> {
    let Some(prop) = schema.as_object() else {
        return Vec::new();
    };
    let variants = prop
        .get("anyOf")
        .or_else(|| prop.get("oneOf"))
        .and_then(Value::as_array);
    let non_null: Vec<&Value> = variants
        .map(|arr| {
            arr.iter()
                .filter(|v| v.get("type").and_then(Value::as_str) != Some("null"))
                .collect()
        })
        .unwrap_or_default();
    let mut raw_branches = vec![prop];
    if non_null.len() == 1
        && let Some(branch_obj) = non_null[0].as_object()
    {
        raw_branches.push(branch_obj);
    }
    let branches: Vec<serde_json::Map<String, Value>> = raw_branches
        .into_iter()
        .map(normalize_draft04_bounds)
        .collect();

    let mut tags = Vec::new();
    if branches.iter().any(|b| has_integer_type(b.get("type"))) {
        tags.push("@integer".to_string());
    }
    for &kw in TAG_KEYWORDS {
        if is_lower_bound(kw) || is_upper_bound(kw) {
            let nums: Vec<f64> = branches
                .iter()
                .filter_map(|b| b.get(kw).and_then(Value::as_f64))
                .collect();
            if nums.is_empty() {
                continue;
            }
            let picked = if is_lower_bound(kw) {
                nums.into_iter().fold(f64::NEG_INFINITY, f64::max)
            } else {
                nums.into_iter().fold(f64::INFINITY, f64::min)
            };
            let formatted = if picked.fract() == 0.0 && picked.abs() < (i64::MAX as f64) {
                format!("{}", picked as i64)
            } else {
                format!("{picked}")
            };
            tags.push(format!("@{kw} {formatted}"));
            continue;
        }
        let mut seen = Vec::new();
        for b in &branches {
            let Some(v) = b.get(kw) else {
                continue;
            };
            let rendered = match v {
                Value::Number(n) => Some(n.to_string()),
                Value::String(s) => Some(s.clone()),
                _ => None,
            };
            if let Some(val_str) = rendered
                && !seen.contains(&val_str)
            {
                seen.push(val_str.clone());
                tags.push(format!("@{kw} {val_str}"));
            }
        }
    }
    if branches
        .iter()
        .any(|b| b.get("uniqueItems").and_then(Value::as_bool) == Some(true))
    {
        tags.push("@uniqueItems".to_string());
    }
    if let Some(default_val) = prop.get("default")
        && !default_val.is_null()
        && let Ok(s) = serde_json::to_string(default_val)
    {
        tags.push(format!("@default {s}"));
    }
    if prop.get("deprecated").and_then(Value::as_bool) == Some(true) {
        tags.push("@deprecated".to_string());
    }

    let joined = tags
        .into_iter()
        .filter(|tag| tag.chars().count() <= MAX_TAG_CHARS)
        .map(|tag| sanitize_line_terminators(&tag))
        .collect::<Vec<_>>()
        .join(" ");
    if joined.is_empty() {
        Vec::new()
    } else {
        vec![joined]
    }
}

fn object_type(obj: &serde_json::Map<String, Value>, ctx: &mut SchemaContext<'_>) -> String {
    let empty_props = serde_json::Map::new();
    let properties = obj
        .get("properties")
        .and_then(Value::as_object)
        .unwrap_or(&empty_props);
    let required: HashSet<&str> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let mut names: Vec<&str> = properties.keys().map(String::as_str).collect();
    names.sort_unstable();

    let mut members: Vec<String> = names
        .iter()
        .map(|&name| {
            let opt = if required.contains(name) { "" } else { "?" };
            let ty = to_type(&properties[name], ctx);
            format!("{}{opt}: {ty};", property_key(name))
        })
        .collect();

    let additional = obj.get("additionalProperties");
    match additional {
        Some(Value::Bool(true)) => members.push("[key: string]: unknown;".to_string()),
        Some(Value::Bool(false)) => {}
        Some(other) => {
            let ty = to_type(other, ctx);
            members.push(format!("[key: string]: {ty};"));
        }
        None if names.is_empty() => members.push("[key: string]: unknown;".to_string()),
        None => {}
    }

    if members.is_empty() {
        return "{}".to_string();
    }

    let comments: Vec<Vec<String>> = names
        .iter()
        .map(|&name| {
            let prop_val = &properties[name];
            let mut lines = prop_val
                .get("description")
                .and_then(Value::as_str)
                .map(split_comment_lines)
                .unwrap_or_default();
            if ctx.tags {
                lines.extend(constraint_tags(prop_val));
            }
            lines
        })
        .collect();

    if !comments.iter().any(|c| !c.is_empty()) {
        return format!("{{ {} }}", members.join(" "));
    }

    let mut lines = vec!["{".to_string()];
    for (idx, comment) in comments.into_iter().enumerate() {
        for line in comment {
            lines.push(format!("{INDENT}// {line}"));
        }
        let indented_member = members[idx].replace('\n', &format!("\n{INDENT}"));
        lines.push(format!("{INDENT}{indented_member}"));
    }
    for member in members.iter().skip(names.len()) {
        lines.push(format!("{INDENT}{member}"));
    }
    lines.push("}".to_string());
    lines.join("\n")
}

/// 渲染工具在 `tools` 命名空间下的单个 TypeScript 成员签名：`id(args: T): Promise<R>;`。
pub fn render_tool_signature(info: &ToolRuntimeInfo, input_max_chars: usize) -> String {
    let id = to_codemode_identifier(&info.name);
    let input = schema_to_type(&info.parameters, Some(input_max_chars));
    let output = render_tool_output_type(info.output_schema.as_ref());
    format!("{id}(args: {input}): Promise<{output}>;")
}

/// 渲染工具返回的 TypeScript 类型（未声明 `output_schema` 时为 `string`）。
pub fn render_tool_output_type(output_schema: Option<&Value>) -> String {
    match output_schema {
        Some(schema) => schema_to_type(schema, None),
        None => "string".to_string(),
    }
}

/// 渲染工具的说明正文与 TypeScript 声明片段（对齐 `pi-codemode` `renderToolSample`）。
pub fn render_tool_sample(info: &ToolRuntimeInfo, input_max_chars: usize) -> String {
    let mut desc = info.description.trim().to_string();
    let bullets: Vec<String> = info
        .prompt_guidelines
        .iter()
        .map(|g| g.trim())
        .filter(|g| !g.is_empty())
        .map(|g| format!("- {g}"))
        .collect();
    if !bullets.is_empty() {
        if !desc.is_empty() {
            desc.push_str("\n\n");
        }
        desc.push_str(&bullets.join("\n"));
    }
    let decl = format!(
        "declare const tools: {{ {} }};",
        render_tool_signature(info, input_max_chars)
    );
    format!("{desc}\n\ncodemode tool declaration:\n```ts\n{decl}\n```")
}

fn follow_ref<'a>(schema: &'a Value, root: &'a Value) -> &'a Value {
    let mut seen: Vec<*const Value> = Vec::new();
    let mut current = schema;
    while let Some(obj) = current.as_object() {
        let ptr = current as *const Value;
        if seen.contains(&ptr) {
            break;
        }
        let Some(ref_str) = obj.get("$ref").and_then(Value::as_str) else {
            break;
        };
        let Some(target) = resolve_schema_ref(ref_str, root) else {
            break;
        };
        seen.push(ptr);
        current = target;
    }
    current
}

fn describe_fields(schema: &Value, root: &Value, depth: usize) -> Option<String> {
    let obj = follow_ref(schema, root).as_object()?;
    if obj.get("type").and_then(Value::as_str) != Some("object") {
        return None;
    }
    let props = obj.get("properties")?.as_object()?;
    let required: HashSet<&str> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let fields: Vec<String> = props
        .iter()
        .map(|(name, field)| {
            let const_val = follow_ref(field, root).as_object().and_then(|o| o.get("const"));
            let nested = if let Some(cv) = const_val {
                serde_json::to_string(cv).ok()
            } else if depth > 0 {
                describe_fields(field, root, depth - 1)
            } else {
                None
            };
            let opt = if required.contains(name.as_str()) {
                ""
            } else {
                "?"
            };
            match nested {
                Some(n) => format!("{name}{opt}: {n}"),
                None => format!("{name}{opt}"),
            }
        })
        .collect();
    Some(format!("{{ {} }}", fields.join(", ")))
}

fn describe_union_fields(schema: &Value, root: &Value) -> Option<String> {
    let obj = follow_ref(schema, root).as_object()?;
    let variants = obj
        .get("anyOf")
        .or_else(|| obj.get("oneOf"))
        .and_then(Value::as_array)?;
    if variants.is_empty() {
        return None;
    }
    let described: Option<Vec<String>> = variants
        .iter()
        .map(|v| describe_fields(v, root, 0))
        .collect();
    described.map(|parts| parts.join(" | "))
}

/// 为已直接声明的工具生成单行返回值摘要（过长时引导调 `describeTool()`）。
pub fn describe_output(name: &str, output_schema: Option<&Value>) -> String {
    let ty = render_tool_output_type(output_schema);
    if ty == "string" {
        return "a string".to_string();
    }
    if let Some(schema) = output_schema
        && let Some(fields) =
            describe_fields(schema, schema, 1).or_else(|| describe_union_fields(schema, schema))
    {
        return format!("`{fields}`");
    }
    if !ty.contains('\n') {
        return format!("`{ty}`");
    }
    let quoted = serde_json::to_string(name).unwrap_or_else(|_| format!("\"{name}\""));
    format!("a value; `await describeTool({quoted})` shows its type")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsh_tools::{ToolAnnotations, ToolExecutionMode, ToolExposure};
    use serde_json::json;

    fn sample_info(name: &str, parameters: Value, output_schema: Option<Value>) -> ToolRuntimeInfo {
        ToolRuntimeInfo {
            name: name.to_string(),
            description: "Sample tool".to_string(),
            parameters,
            output_schema,
            exposure: ToolExposure::Codemode,
            execution_mode: ToolExecutionMode::Parallel,
            annotations: ToolAnnotations::default(),
            namespace: None,
            default_active: None,
            prompt_snippet: None,
            prompt_guidelines: Vec::new(),
        }
    }

    #[test]
    fn converts_json_schema_to_compact_typescript_and_resolves_local_refs() {
        let schema = json!({
            "type": "object",
            "$defs": {
                "Item": { "type": "object", "properties": { "id": { "type": "integer" } }, "required": ["id"] }
            },
            "properties": {
                "limit": { "type": "integer", "minimum": 1, "default": 20 },
                "item": { "$ref": "#/$defs/Item" },
                "mode": { "enum": ["fast", "full"] }
            },
            "required": ["item"]
        });
        let ty = schema_to_type(&schema, Some(DEFAULT_INPUT_SCHEMA_MAX_CHARS));
        assert!(ty.contains("// @integer\n    id: number;"), "{ty}");
        assert!(ty.contains("// @integer @minimum 1 @default 20"), "{ty}");
        assert!(ty.contains("limit?: number;"), "{ty}");
        assert!(ty.contains("mode?: \"fast\" | \"full\";"), "{ty}");
    }

    #[test]
    fn handles_malformed_percent_escapes_and_oversized_schemas_gracefully() {
        let schema = json!({
            "type": "object",
            "properties": {
                "bad": { "$ref": "#/$defs/%E0" },
                "count": { "type": "integer", "minimum": 1, "default": 20 }
            }
        });
        let ty = schema_to_type(&schema, Some(DEFAULT_INPUT_SCHEMA_MAX_CHARS));
        assert!(ty.contains("bad?: unknown;"), "{ty}");
        assert!(ty.contains("// @integer @minimum 1 @default 20"), "{ty}");

        let stripped = schema_to_type(&schema, Some(40));
        assert_eq!(stripped, "{ bad?: unknown; count?: number; }");
        assert_eq!(schema_to_type(&schema, Some(10)), "unknown");
    }

    #[test]
    fn renders_tool_sample_and_single_line_output_summary() {
        let info = sample_info(
            "mcp__fff__grep",
            json!({
                "type": "object",
                "properties": { "pattern": { "type": "string" } },
                "required": ["pattern"]
            }),
            Some(json!({
                "type": "object",
                "properties": {
                    "matches": { "type": "array", "items": { "type": "string" } },
                    "truncated": { "type": "boolean" }
                },
                "required": ["matches"]
            })),
        );
        let sample = render_tool_sample(&info, DEFAULT_INPUT_SCHEMA_MAX_CHARS);
        assert!(sample.contains("codemode tool declaration:\n```ts"), "{sample}");
        assert!(
            sample.contains("mcp__fff__grep(args: { pattern: string; }): Promise<{"),
            "{sample}"
        );
        assert!(!sample.contains("\"properties\""), "{sample}");

        let out_summary = describe_output("mcp__fff__grep", info.output_schema.as_ref());
        assert_eq!(out_summary, "`{ matches, truncated? }`");
        assert_eq!(describe_output("read", None), "a string");
    }
}
