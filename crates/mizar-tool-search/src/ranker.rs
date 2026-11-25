//! BM25 工具元数据检索排序器（对齐 `pi` `tool-search/tool.ts`）。

use std::collections::{HashMap, HashSet};

use mizar_tools::ToolRuntimeInfo;
use serde_json::Value;

const NAME_WEIGHT: f64 = 4.0;
const NAMESPACE_WEIGHT: f64 = 2.0;

#[derive(Debug, Clone)]
pub struct ToolSearchDocument {
    pub name: String,
    pub text: String,
    pub namespace_text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolSearchMatch {
    pub name: String,
    pub score: f64,
}

fn is_stop_word(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "this"
            | "to"
            | "with"
    )
}

fn stem(term: &str) -> String {
    if term.len() > 4 && term.ends_with("ies") {
        return format!("{}y", &term[..term.len() - 3]);
    }
    if term.len() > 4
        && (term.ends_with("ches")
            || term.ends_with("shes")
            || term.ends_with("sses")
            || term.ends_with("xes")
            || term.ends_with("zes"))
    {
        return term[..term.len() - 2].to_string();
    }
    if term.len() > 3 && term.ends_with('s') && !term.ends_with("ss") {
        return term[..term.len() - 1].to_string();
    }
    term.to_string()
}

/// 按标识符边界（`camelCase` / `snake_case` / `__`）与空白拆分英文/Unicode词元，过滤停用词并做轻量单数词干化。
pub fn tokenize(text: &str) -> Vec<String> {
    let mut raw_tokens = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if ch.is_uppercase() && prev_lower && !cur.is_empty() {
                raw_tokens.push(std::mem::take(&mut cur));
            }
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
            prev_lower = ch.is_lowercase();
        } else {
            if !cur.is_empty() {
                raw_tokens.push(std::mem::take(&mut cur));
            }
            prev_lower = false;
        }
    }
    if !cur.is_empty() {
        raw_tokens.push(cur);
    }

    raw_tokens
        .into_iter()
        .filter(|t| !is_stop_word(t))
        .map(|t| stem(&t))
        .collect()
}

fn normalize_name(s: &str) -> String {
    s.trim().to_ascii_lowercase()
}

fn collect_schema_terms(schema: &Value, depth: usize, out: &mut Vec<String>) {
    if depth > 4 {
        return;
    }
    let Some(obj) = schema.as_object() else {
        return;
    };
    if let Some(desc) = obj.get("description").and_then(Value::as_str) {
        out.push(desc.to_string());
    }
    if let Some(props) = obj.get("properties").and_then(Value::as_object) {
        for (prop_name, prop_schema) in props {
            out.push(prop_name.clone());
            collect_schema_terms(prop_schema, depth + 1, out);
        }
    }
    if let Some(items) = obj.get("items") {
        collect_schema_terms(items, depth + 1, out);
    }
}

pub fn create_tool_search_document(info: &ToolRuntimeInfo) -> ToolSearchDocument {
    let mut text_parts = vec![info.description.clone()];
    collect_schema_terms(&info.parameters, 0, &mut text_parts);

    let ns_text = info
        .namespace
        .as_ref()
        .map(|ns| {
            let mut parts = vec![ns.name.clone()];
            parts.extend(ns.aliases.iter().cloned());
            if let Some(d) = &ns.description {
                parts.push(d.clone());
            }
            if let Some(inst) = &ns.instructions {
                parts.push(inst.clone());
            }
            parts.join(" ")
        })
        .unwrap_or_default();

    ToolSearchDocument {
        name: info.name.clone(),
        text: text_parts.join(" "),
        namespace_text: ns_text,
    }
}

struct IndexedDoc<'a> {
    name: &'a str,
    name_terms: HashSet<String>,
    local_name_terms: Vec<String>,
    namespace_terms: HashSet<String>,
    text_counts: HashMap<String, usize>,
    text_length: usize,
}

fn local_tool_name(name: &str) -> &str {
    if let Some(rest) = name.strip_prefix("mcp__")
        && let Some((_, local)) = rest.split_once("__")
    {
        return local;
    }
    name
}

fn is_exact_name_query(exact_query: &str, tool_name: &str) -> bool {
    let norm = normalize_name(tool_name);
    exact_query == norm || exact_query == normalize_name(local_tool_name(tool_name))
}

fn index_doc<'a>(doc: &'a ToolSearchDocument) -> IndexedDoc<'a> {
    let name_terms: HashSet<String> = tokenize(&doc.name).into_iter().collect();
    let local_name_terms = tokenize(local_tool_name(&doc.name));
    let namespace_terms: HashSet<String> = tokenize(&doc.namespace_text).into_iter().collect();
    let text_tokens = tokenize(&doc.text);
    let text_length = text_tokens.len();
    let mut text_counts = HashMap::new();
    for tok in text_tokens {
        *text_counts.entry(tok).or_insert(0) += 1;
    }
    IndexedDoc {
        name: &doc.name,
        name_terms,
        local_name_terms,
        namespace_terms,
        text_counts,
        text_length,
    }
}

pub struct Bm25Ranker {
    pub k1: f64,
    pub b: f64,
}

impl Default for Bm25Ranker {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl Bm25Ranker {
    pub fn rank(
        &self,
        query: &str,
        documents: &[ToolSearchDocument],
        limit: usize,
    ) -> Vec<ToolSearchMatch> {
        let query_terms: HashSet<String> = tokenize(query).into_iter().collect();
        if query_terms.is_empty() || documents.is_empty() || limit == 0 {
            return Vec::new();
        }
        let indexed: Vec<IndexedDoc<'_>> = documents.iter().map(index_doc).collect();
        let avg_len = (indexed.iter().map(|d| d.text_length).sum::<usize>() as f64
            / indexed.len() as f64)
            .max(1.0);
        let doc_count = indexed.len() as f64;

        let idf: HashMap<&str, f64> = query_terms
            .iter()
            .map(|term| {
                let freq = indexed
                    .iter()
                    .filter(|d| {
                        d.name_terms.contains(term)
                            || d.text_counts.contains_key(term)
                            || d.namespace_terms.contains(term)
                    })
                    .count() as f64;
                let score = (1.0 + (doc_count - freq + 0.5) / (freq + 0.5)).ln();
                (term.as_str(), score)
            })
            .collect();

        let full_name_bonus: f64 = idf.values().sum();
        let exact_query = normalize_name(query);

        struct Candidate<'a> {
            name: &'a str,
            score: f64,
            covered: usize,
            exact: bool,
            order: usize,
        }

        let mut matches = Vec::new();
        for (order, doc) in indexed.iter().enumerate() {
            let norm = self.k1 * (1.0 - self.b + (self.b * doc.text_length as f64) / avg_len);
            let mut score = 0.0;
            let mut covered = 0usize;
            for term in &query_terms {
                let count = *doc.text_counts.get(term).unwrap_or(&0) as f64;
                let mut weight = 0.0;
                if doc.name_terms.contains(term) {
                    weight += NAME_WEIGHT;
                }
                if doc.namespace_terms.contains(term) {
                    weight += NAMESPACE_WEIGHT;
                }
                if count > 0.0 {
                    weight += (count * (self.k1 + 1.0)) / (count + norm);
                }
                if weight > 0.0 {
                    covered += 1;
                }
                score += idf.get(term.as_str()).copied().unwrap_or(0.0) * weight;
            }
            if score == 0.0 {
                continue;
            }
            if !doc.local_name_terms.is_empty()
                && doc
                    .local_name_terms
                    .iter()
                    .all(|term| query_terms.contains(term))
            {
                score += full_name_bonus;
            }
            matches.push(Candidate {
                name: doc.name,
                score,
                covered,
                exact: is_exact_name_query(&exact_query, doc.name),
                order,
            });
        }

        matches.sort_by(|a, b| {
            b.exact
                .cmp(&a.exact)
                .then_with(|| b.covered.cmp(&a.covered))
                .then_with(|| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.order.cmp(&b.order))
        });

        matches
            .into_iter()
            .take(limit)
            .map(|c| ToolSearchMatch {
                name: c.name.to_string(),
                score: c.score,
            })
            .collect()
    }
}
