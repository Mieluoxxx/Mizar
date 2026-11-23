use std::collections::BTreeMap;

use serde_json::Value;

use crate::skills::types::{SkillInvocationPolicy, is_skill_name};

#[derive(Debug, Clone)]
pub struct ParsedSkillFile {
    pub name: String,
    pub description: String,
    pub when_to_use: Option<String>,
    pub invocation: SkillInvocationPolicy,
    pub metadata: Option<BTreeMap<String, Value>>,
    pub body: String,
}

fn unquote_yaml_scalar(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        let inner = &trimmed[1..trimmed.len() - 1];
        return inner
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
    }
    if trimmed.len() >= 2 && trimmed.starts_with('\'') && trimmed.ends_with('\'') {
        let inner = &trimmed[1..trimmed.len() - 1];
        return inner.replace("''", "'");
    }
    trimmed.to_string()
}

fn parse_yaml_bool(raw: &str, key: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Ok(true),
        "false" | "no" | "off" | "0" => Ok(false),
        other => Err(format!(
            "frontmatter field \"{key}\" must be a boolean, got `{other}`"
        )),
    }
}

fn extract_frontmatter(raw: &str) -> Option<(&str, &str)> {
    let raw = raw.strip_prefix('\u{FEFF}').unwrap_or(raw);
    let mut lines = raw.split_inclusive('\n');
    let first = lines.next()?;
    let first_trimmed = first.trim_end_matches(['\r', '\n']);
    if first_trimmed != "---" {
        return None;
    }

    let mut offset = first.len();
    let yaml_start = offset;
    for line in lines {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            let yaml = &raw[yaml_start..offset];
            let body = &raw[offset + line.len()..];
            return Some((yaml, body));
        }
        offset += line.len();
    }
    None
}

fn parse_scalar_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }
    if let Ok(num) = trimmed.parse::<i64>() {
        return Value::Number(num.into());
    }
    Value::String(unquote_yaml_scalar(trimmed))
}

/// 解析技能 Markdown 文件的 YAML Frontmatter 与正文。
pub fn parse_skill_markdown(raw: &str) -> Result<ParsedSkillFile, String> {
    let (yaml, body) = extract_frontmatter(raw)
        .ok_or_else(|| "missing YAML frontmatter (`--- ... ---`)".to_string())?;

    let mut fields: BTreeMap<String, String> = BTreeMap::new();
    let mut metadata_map: BTreeMap<String, Value> = BTreeMap::new();

    let lines: Vec<&str> = yaml.lines().collect();
    let mut idx = 0usize;
    while idx < lines.len() {
        let line = lines[idx].trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            idx += 1;
            continue;
        }
        // 只处理顶层键（无前导缩进）
        if line.starts_with([' ', '\t']) {
            idx += 1;
            continue;
        }
        let Some((key_raw, val_raw)) = line.split_once(':') else {
            return Err(format!("invalid frontmatter line: `{line}`"));
        };
        let key = key_raw.trim();
        let val = val_raw.trim();

        match key {
            "disableModelInvocation" | "modelInvocable" => {
                return Err(format!(
                    "frontmatter field \"{key}\" is unsupported; use \"disable-model-invocation\""
                ));
            }
            "userInvocable" => {
                return Err(
                    "frontmatter field \"userInvocable\" is unsupported; use \"user-invocable\""
                        .to_string(),
                );
            }
            "metadata" if val.is_empty() => {
                idx += 1;
                while idx < lines.len() {
                    let sub = lines[idx].trim_end_matches('\r');
                    if sub.trim().is_empty() || sub.trim().starts_with('#') {
                        idx += 1;
                        continue;
                    }
                    if !sub.starts_with([' ', '\t']) {
                        break;
                    }
                    if let Some((mk, mv)) = sub.trim().split_once(':') {
                        metadata_map.insert(mk.trim().to_string(), parse_scalar_value(mv));
                    }
                    idx += 1;
                }
                continue;
            }
            _ => {}
        }

        // 支持 YAML 多行块标量（`|` / `>` / `|-` / `>-`）
        if matches!(val, "|" | "|-" | ">" | ">-") {
            let fold = val.starts_with('>');
            let mut block_lines = Vec::new();
            idx += 1;
            let mut base_indent: Option<usize> = None;
            while idx < lines.len() {
                let sub = lines[idx].trim_end_matches('\r');
                if sub.trim().is_empty() {
                    block_lines.push(String::new());
                    idx += 1;
                    continue;
                }
                let indent = sub.chars().take_while(|c| *c == ' ').count();
                if indent == 0 {
                    break;
                }
                let bi = *base_indent.get_or_insert(indent);
                let cut = indent.min(bi);
                block_lines.push(sub[cut..].to_string());
                idx += 1;
            }
            let joined = if fold {
                block_lines
                    .into_iter()
                    .map(|l| if l.is_empty() { "\n".to_string() } else { l })
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                block_lines.join("\n")
            };
            fields.insert(key.to_string(), joined.trim().to_string());
            continue;
        }

        fields.insert(key.to_string(), unquote_yaml_scalar(val));
        idx += 1;
    }

    let name = fields
        .get("name")
        .filter(|s| !s.is_empty())
        .cloned()
        .ok_or_else(|| "frontmatter requires `name`".to_string())?;
    if !is_skill_name(&name) {
        return Err(format!(
            "invalid skill name \"{name}\" (must be kebab-case)"
        ));
    }

    let description = fields
        .get("description")
        .filter(|s| !s.is_empty())
        .cloned()
        .ok_or_else(|| "frontmatter requires `description`".to_string())?;

    let when_to_use = fields
        .get("when-to-use")
        .or_else(|| fields.get("whenToUse"))
        .filter(|s| !s.is_empty())
        .cloned();

    let disable_model = match fields.get("disable-model-invocation") {
        Some(raw) => parse_yaml_bool(raw, "disable-model-invocation")?,
        None => false,
    };
    let user_invocable = match fields.get("user-invocable") {
        Some(raw) => parse_yaml_bool(raw, "user-invocable")?,
        None => true,
    };

    Ok(ParsedSkillFile {
        name,
        description,
        when_to_use,
        invocation: SkillInvocationPolicy {
            model_invocable: !disable_model,
            user_invocable,
        },
        metadata: (!metadata_map.is_empty()).then_some(metadata_map),
        body: body.trim().to_string(),
    })
}
