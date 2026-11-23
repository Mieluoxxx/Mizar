//! 只读会话投影、模型选择优先级解析与会话报告生成（零终端依赖）。

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Local};
use cordis::Context;
use mizar_core::spine::strip_summary_wrapper;
use mizar_llm::{ContentBlock, LlmContextExt, ModelCostRates, ModelInfo, ProviderInfo, TokenUsage};
use mizar_session::{
    Session, SessionEvent, SessionListQuery, SessionPersistenceContextExt, SessionStore,
    SessionTraceEvent, TraceSeverity,
};
use mizar_types::client::{
    CacheMissAudit, ContextBreakdownSnapshot, NoticeLevel, SessionPickerItem, SessionReplayItem,
    SessionSelection, SessionTreeEntry, SessionViewData, UsageAuditSample, abbreviate_home_path,
    compute_cache_miss_audit, format_tokens, short_id, truncate_display,
};

/// 选择解析的输入：各来源的值与「是否显式指定」标记。
///
/// 优先级链（高的先，逐字段独立判定）：
/// ```text
/// 1. 环境变量 MODEL            → 压过一切，包括会话记录
/// 2. 会话记录（仅 resume）     → 恢复语义：会话当时的模型优先于全局配置
/// 3. settings.json 的默认值
/// 4. 最近一个会话的记录（仅冷启动）  → 「沿用上次」
/// 5. 现有推导（models.json 字典序 + models[0]）
/// ```
#[derive(Debug, Clone, Default)]
pub struct SelectionSources {
    /// 环境变量是否指定了模型。
    pub model_from_env: bool,
    /// `settings.json` 是否写了 `defaultModel`。
    pub model_from_settings: bool,
    /// `settings.json` 是否写了 `reasoningEffort`。
    pub reasoning_effort_from_settings: bool,
    /// 启动时已解析出的默认选择（来自 `models.json` 推导或 settings）。
    pub startup: SessionSelection,
    /// resume 时该会话自己的记录。
    pub session: Option<SessionSelection>,
    /// 当前工作区最近一个会话的记录（冷启动用）。
    pub latest_workspace: Option<SessionSelection>,
}

/// 按文档化的优先级链算出应采用的模型与思考强度。
///
/// 返回 `None` 表示启动值已经是最优解，调用方无需覆盖。
/// 返回 `Some` 时三个字段各自可能为 `None` —— 表示该字段本次没有更高优先级的来源。
pub fn resolve_selection(sources: &SelectionSources, resumed: bool) -> Option<SessionSelection> {
    // 环境变量压过一切，直接不覆盖
    if sources.model_from_env {
        return None;
    }

    // resume：会话记录优先于 settings；冷启动：settings 优先于最近会话记录
    let session = resumed.then_some(sources.session.as_ref()).flatten();
    let latest = (!resumed)
        .then_some(sources.latest_workspace.as_ref())
        .flatten();

    let model = session
        .and_then(|s| s.model.clone())
        .or_else(|| {
            (!sources.model_from_settings)
                .then(|| latest.and_then(|s| s.model.clone()))
                .flatten()
        })
        .or_else(|| sources.startup.model.clone());
    // provider 与 model 同源优先级，避免 provider/model 交错
    let provider = session
        .and_then(|s| s.provider.clone())
        .or_else(|| {
            (!sources.model_from_settings)
                .then(|| latest.and_then(|s| s.provider.clone()))
                .flatten()
        })
        .or_else(|| sources.startup.provider.clone());
    let effort = session
        .and_then(|s| s.reasoning_effort)
        .or_else(|| {
            (!sources.reasoning_effort_from_settings)
                .then(|| latest.and_then(|s| s.reasoning_effort))
                .flatten()
        })
        .or(sources.startup.reasoning_effort);

    let resolved = SessionSelection {
        provider,
        model,
        reasoning_effort: effort,
    };
    (resolved != sources.startup).then_some(resolved)
}

/// 把会话层的选择投影为客户端契约形态（`mizar-client-core` 禁止依赖 `mizar-session`）。
pub fn to_ui_selection(sel: mizar_session::SessionSelection) -> SessionSelection {
    SessionSelection {
        provider: sel.provider,
        model: sel.model,
        reasoning_effort: sel.reasoning_effort,
    }
}

/// 取当前工作区最近一个会话记录的选择。
///
/// 只在 `settings.json` 未指定模型时使用，且语义为「跨会话沿用上次」而非「跨工作区」，
/// 因此按 `cwd` 过滤（走 `idx_sessions_cwd_updated` 索引）。
pub fn latest_workspace_selection(
    sessions: &SessionStore,
    workspace: &Path,
) -> Option<mizar_session::SessionSelection> {
    let latest = sessions
        .list_headers(&SessionListQuery {
            cwd: Some(workspace.display().to_string()),
            limit: Some(1),
            exclude_subagents: true,
        })
        .ok()?
        .into_iter()
        .next()?;
    sessions.peek_selection(&latest.id)
}

/// 在指定 Provider 下按 ID 精确匹配或按名称忽略大小写查找模型定义。
pub fn find_model_in_provider<'a>(provider: &'a ProviderInfo, spec: &str) -> Option<&'a ModelInfo> {
    provider.models.iter().find(|m| m.id == spec).or_else(|| {
        let lower = spec.to_ascii_lowercase();
        provider
            .models
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(&lower))
    })
}

/// 解析 `/model <spec>`：优先识别带 `/` 的完整模型 ID（如 `mayoru-completions/gemini-4-hydrogen`），
/// 仅当 `/` 前缀匹配已注册 provider ID 时才拆分 `<provider>/<model>`。
pub fn resolve_model_switch(
    providers: &[ProviderInfo],
    current_provider: Option<&str>,
    spec: &str,
) -> (Option<String>, String) {
    if let Some(cur) = current_provider
        && let Some(prov) = providers.iter().find(|p| p.id == cur)
        && let Some(model) = find_model_in_provider(prov, spec)
    {
        return (None, model.id.clone());
    }

    if let Some((prov_prefix, rest)) = spec.split_once('/') {
        let prov_prefix = prov_prefix.trim();
        let rest = rest.trim();
        if !rest.is_empty()
            && let Some(prov) = providers.iter().find(|p| p.id == prov_prefix)
        {
            let matched = find_model_in_provider(prov, rest)
                .map(|m| m.id.clone())
                .unwrap_or_else(|| rest.to_string());
            return (Some(prov.id.clone()), matched);
        }
    }

    for prov in providers {
        if let Some(model) = find_model_in_provider(prov, spec) {
            return (Some(prov.id.clone()), model.id.clone());
        }
    }

    (None, spec.to_string())
}

fn accumulate_usage(acc: &mut TokenUsage, usage: &TokenUsage) {
    acc.input_tokens = acc.input_tokens.saturating_add(usage.input_tokens);
    acc.output_tokens = acc.output_tokens.saturating_add(usage.output_tokens);
    if let Some(cr) = usage.cache_read_tokens {
        acc.cache_read_tokens = Some(acc.cache_read_tokens.unwrap_or(0).saturating_add(cr));
    }
    if let Some(cw) = usage.cache_write_tokens {
        acc.cache_write_tokens = Some(acc.cache_write_tokens.unwrap_or(0).saturating_add(cw));
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn format_local_datetime_ms(ts_ms: u128) -> String {
    if ts_ms == 0 {
        return "未知".to_string();
    }
    let Ok(ms) = i64::try_from(ts_ms) else {
        return "未知".to_string();
    };
    DateTime::from_timestamp_millis(ms)
        .map(|utc| {
            utc.with_timezone(&Local)
                .format("%Y-%m-%d %H:%M")
                .to_string()
        })
        .unwrap_or_else(|| "未知".to_string())
}

fn format_relative_activity(updated_at_ms: u128, now_ms: u128) -> String {
    if updated_at_ms == 0 {
        return "未知".to_string();
    }
    let elapsed_secs = now_ms.saturating_sub(updated_at_ms) / 1_000;
    if elapsed_secs < 60 {
        "刚刚".to_string()
    } else if elapsed_secs < 3_600 {
        format!("{} 分钟前", elapsed_secs / 60)
    } else if elapsed_secs < 86_400 {
        format!("{} 小时前", elapsed_secs / 3_600)
    } else {
        format!("{} 天前", elapsed_secs / 86_400)
    }
}

fn format_context_window(window: u64) -> String {
    if window >= 1_000_000 && window.is_multiple_of(1_000_000) {
        format!("{}M", window / 1_000_000)
    } else if (1_000..1_000_000).contains(&window) && window.is_multiple_of(1_000) {
        format!("{}k", window / 1_000)
    } else {
        format_tokens(window)
    }
}

fn join_mid_dot(parts: &[String]) -> String {
    let mut out = String::new();
    for part in parts {
        if !out.is_empty() {
            if out.ends_with('）') {
                out.push_str("· ");
            } else {
                out.push_str(" · ");
            }
        }
        out.push_str(part);
    }
    out
}

fn lookup_model_rates(
    providers: &[ProviderInfo],
    provider: Option<&str>,
    model: Option<&str>,
) -> Option<ModelCostRates> {
    let mod_id = model.filter(|s| !s.is_empty())?;
    if let Some(prov_id) = provider.filter(|s| !s.is_empty()) {
        return providers
            .iter()
            .find(|p| p.id == prov_id)
            .and_then(|p| find_model_in_provider(p, mod_id))
            .and_then(|m| m.cost.clone());
    }
    providers
        .iter()
        .find_map(|p| find_model_in_provider(p, mod_id))
        .and_then(|m| m.cost.clone())
}

fn collect_usage_audit_samples(
    entries: &[mizar_session::SessionEntry],
    providers: &[ProviderInfo],
) -> (Vec<UsageAuditSample>, Option<ModelCostRates>) {
    let mut samples = Vec::new();
    let mut active_provider: Option<String> = None;
    let mut active_model: Option<String> = None;
    let mut last_rates: Option<ModelCostRates> = None;

    for entry in entries {
        match &entry.event {
            SessionEvent::Selection {
                provider, model, ..
            } => {
                if provider.is_some() {
                    active_provider = provider.clone();
                }
                if model.is_some() {
                    active_model = model.clone();
                }
                if let Some(r) = lookup_model_rates(
                    providers,
                    active_provider.as_deref(),
                    active_model.as_deref(),
                ) {
                    last_rates = Some(r);
                }
            }
            SessionEvent::Compaction { .. } => {
                samples.push(UsageAuditSample::Compaction);
            }
            SessionEvent::Assistant {
                replay,
                usage: Some(u),
                ..
            } => {
                let rates = replay
                    .as_ref()
                    .and_then(|env| {
                        lookup_model_rates(providers, Some(&env.provider), Some(&env.model))
                    })
                    .or_else(|| {
                        lookup_model_rates(
                            providers,
                            active_provider.as_deref(),
                            active_model.as_deref(),
                        )
                    });
                if let Some(ref r) = rates {
                    last_rates = Some(r.clone());
                }
                samples.push(UsageAuditSample::Assistant { usage: *u, rates });
            }
            SessionEvent::Usage {
                kind,
                provider,
                model,
                usage,
                ..
            } if kind == "cache_warm" => {
                let rates = lookup_model_rates(providers, Some(provider), Some(model));
                if let Some(ref r) = rates {
                    last_rates = Some(r.clone());
                }
                samples.push(UsageAuditSample::CacheWarm {
                    usage: *usage,
                    rates,
                });
            }
            _ => {}
        }
    }

    (samples, last_rates)
}

/// 构造 `/session` 命令的只读会话身份与账目报告。
pub fn build_session_info_report(
    ctx: &Context,
    session: &Session,
    session_db_path: Option<&Path>,
    context_window: u64,
    last_prompt_tokens: Option<u64>,
    context_breakdown: &ContextBreakdownSnapshot,
) -> String {
    let providers = ctx
        .llm()
        .map(|llm| llm.provider_infos())
        .unwrap_or_default();
    build_session_info_report_at_with_cache(
        session,
        ctx.session_persistence().is_some(),
        session_db_path,
        context_window,
        last_prompt_tokens,
        context_breakdown,
        now_millis(),
        &providers,
    )
}

/// 以显式持久化标记、路径、上下文快照与当前时间戳构造 `/session` 报告（便于确定性测试）。
pub fn build_session_info_report_at(
    session: &Session,
    persisted: bool,
    session_db_path: Option<&Path>,
    context_window: u64,
    last_prompt_tokens: Option<u64>,
    context_breakdown: &ContextBreakdownSnapshot,
    now_ms: u128,
) -> String {
    build_session_info_report_at_with_cache(
        session,
        persisted,
        session_db_path,
        context_window,
        last_prompt_tokens,
        context_breakdown,
        now_ms,
        &[],
    )
}

/// 附带路由计价表的完整 `/session` 报告构造函数。
#[allow(clippy::too_many_arguments)]
pub fn build_session_info_report_at_with_cache(
    session: &Session,
    persisted: bool,
    session_db_path: Option<&Path>,
    context_window: u64,
    last_prompt_tokens: Option<u64>,
    context_breakdown: &ContextBreakdownSnapshot,
    now_ms: u128,
    providers: &[ProviderInfo],
) -> String {
    let header = session.header();

    let mut total_events = 0u64;
    let mut surface_events = 0u64;
    let mut user_count = 0u64;
    let mut assistant_count = 0u64;
    let mut tool_result_count = 0u64;
    let mut tool_error_count = 0u64;
    let mut bash_count = 0u64;
    let mut bash_local_count = 0u64;
    let mut injection_counts = HashMap::<String, u64>::new();
    let mut compaction_count = 0u64;
    let mut latest_compaction: Option<(u64, Option<u64>, Option<u64>)> = None;
    let mut total_usage = TokenUsage::default();
    let mut latest_context_tokens: Option<u64> = None;
    let mut audit = CacheMissAudit::default();

    session.with_all_entries(|entries| {
        total_events = entries.len() as u64;
        let (samples, fallback_rates) = collect_usage_audit_samples(entries, providers);
        audit = compute_cache_miss_audit(&samples, fallback_rates.as_ref());
        for entry in entries {
            if entry.is_surface() {
                surface_events += 1;
            }
            match &entry.event {
                SessionEvent::User { .. } => user_count += 1,
                SessionEvent::ContextNotice { source, .. } => {
                    *injection_counts.entry(source.clone()).or_insert(0) += 1;
                }
                SessionEvent::Assistant { usage, .. } => {
                    assistant_count += 1;
                    if let Some(u) = usage {
                        accumulate_usage(&mut total_usage, u);
                        let prompt = u.prompt_tokens();
                        if prompt > 0 {
                            latest_context_tokens = Some(prompt);
                        }
                    }
                }
                SessionEvent::ToolResult { is_error, .. } => {
                    tool_result_count += 1;
                    if *is_error {
                        tool_error_count += 1;
                    }
                }
                SessionEvent::Compaction {
                    first_kept_seq,
                    tokens_before,
                    tokens_after,
                    ..
                } => {
                    compaction_count += 1;
                    latest_compaction = Some((*first_kept_seq, *tokens_before, *tokens_after));
                    if let Some(after) = tokens_after.filter(|&v| v > 0) {
                        latest_context_tokens = Some(after);
                    }
                }
                SessionEvent::BashExecution {
                    exclude_from_context,
                    ..
                } => {
                    bash_count += 1;
                    if *exclude_from_context {
                        bash_local_count += 1;
                    }
                }
                SessionEvent::Trace { .. }
                | SessionEvent::Selection { .. }
                | SessionEvent::Usage { .. } => {}
            }
        }
    });

    let log_only_events = total_events.saturating_sub(surface_events);
    let title = header
        .title
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("（未命名）");
    let cwd = header
        .cwd
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("（未记录）");
    let created_str = format_local_datetime_ms(header.created_at_ms);
    let activity_str = format_relative_activity(header.updated_at_ms, now_ms);

    let mut lines = vec![
        "会话信息".to_string(),
        String::new(),
        format!("ID       {}", header.id),
        format!("标题     {title}"),
        format!("工作区   {cwd}"),
        format!("创建     {created_str}  ·  最后活动 {activity_str}"),
    ];

    if let Some(parent_id) = header.parent_id.as_deref().filter(|s| !s.is_empty()) {
        let lineage = match header.fork_seq {
            Some(seq) => format!("分叉自 {} @ #{seq}", short_id(parent_id)),
            None => format!("分叉自 {}", short_id(parent_id)),
        };
        lines.push(format!("谱系     {lineage}"));
    }

    lines.push(String::new());

    let storage = if persisted {
        match session_db_path {
            Some(path) => {
                format!(
                    "SQLite · {}",
                    abbreviate_home_path(&path.display().to_string())
                )
            }
            None => "SQLite（已持久化）".to_string(),
        }
    } else {
        "仅内存（未持久化）".to_string()
    };
    lines.push(format!("存储     {storage}"));
    lines.push(format!(
        "事件     {total_events} 条（surface {surface_events} / log-only {log_only_events}）"
    ));
    lines.push(format!(
        "对话     user {user_count}  ·  assistant {assistant_count}"
    ));

    let mut tool_parts = Vec::new();
    if tool_error_count > 0 {
        tool_parts.push(format!(
            "tool-result {tool_result_count}（err {tool_error_count}）"
        ));
    } else {
        tool_parts.push(format!("tool-result {tool_result_count}"));
    }
    if bash_count > 0 {
        if bash_local_count > 0 {
            tool_parts.push(format!("bash {bash_count}（本地 {bash_local_count}）"));
        } else {
            tool_parts.push(format!("bash {bash_count}"));
        }
    }
    lines.push(format!("工具     {}", join_mid_dot(&tool_parts)));

    const KNOWN_INJECTION_SOURCES: [(&str, &str); 5] = [
        ("agent-instructions", "instructions"),
        ("skill-catalog", "skill-catalog"),
        ("skill-invocation", "skill-invocation"),
        ("file-mentions", "file-mentions"),
        ("repeat-tool-reminder", "repeat-tool-reminder"),
    ];
    let mut inj_parts = Vec::new();
    for (raw_src, label) in KNOWN_INJECTION_SOURCES {
        if let Some(count) = injection_counts.remove(raw_src)
            && count > 0
        {
            inj_parts.push(format!("{label} {count}"));
        }
    }
    let mut extra_sources: Vec<(String, u64)> = injection_counts
        .into_iter()
        .filter(|(_, c)| *c > 0)
        .collect();
    extra_sources.sort_by(|a, b| a.0.cmp(&b.0));
    for (src, count) in extra_sources {
        inj_parts.push(format!("{src} {count}"));
    }
    let inj_summary = if inj_parts.is_empty() {
        "无".to_string()
    } else {
        inj_parts.join(" · ")
    };
    lines.push(format!("注入     {inj_summary}"));

    if let Some((first_kept_seq, tokens_before, tokens_after)) = latest_compaction {
        let mut comp_parts = vec![
            format!("{compaction_count} 次"),
            format!("最新保留自 #{first_kept_seq}"),
        ];
        match (tokens_before, tokens_after) {
            (Some(before), Some(after)) => comp_parts.push(format!(
                "最近 {} → {}",
                format_tokens(before),
                format_tokens(after)
            )),
            (Some(before), None) => comp_parts.push(format!("压前 {}", format_tokens(before))),
            _ => {}
        }
        lines.push(format!("压缩     {}", comp_parts.join(" · ")));
    }

    let prompt_total = total_usage.prompt_tokens();
    let cache_read = total_usage.cache_read_tokens.unwrap_or(0);
    let cache_write = total_usage.cache_write_tokens.unwrap_or(0);
    let cache_str = match (cache_read, cache_write) {
        (0, 0) => String::new(),
        (cr, 0) => format!("（cache {}）", format_tokens(cr)),
        (0, cw) => format!("（cache {} 写）", format_tokens(cw)),
        (cr, cw) => format!(
            "（cache {} 读 / {} 写）",
            format_tokens(cr),
            format_tokens(cw)
        ),
    };
    lines.push(format!(
        "Token    累计 ↑{}{}↓{}",
        format_tokens(prompt_total),
        if cache_str.is_empty() {
            " "
        } else {
            &cache_str
        },
        format_tokens(total_usage.output_tokens),
    ));

    let window = context_window.max(1);
    let effective_measured = latest_context_tokens.or(last_prompt_tokens);
    let ctx_summary = if let Some(measured) = effective_measured {
        let pct = (measured as f64 / window as f64) * 100.0;
        format!(
            "{} / {}（{pct:.1}%）",
            format_tokens(measured),
            format_context_window(window)
        )
    } else if surface_events == 0 {
        format!("0 / {}（0.0%）", format_context_window(window))
    } else {
        let est = context_breakdown
            .system_tokens
            .saturating_add(context_breakdown.user_tokens)
            .saturating_add(context_breakdown.assistant_tokens)
            .saturating_add(context_breakdown.reasoning_tokens)
            .saturating_add(context_breakdown.tool_tokens);
        let pct = (est as f64 / window as f64) * 100.0;
        format!(
            "~{} / {}（{pct:.1}%）",
            format_tokens(est),
            format_context_window(window)
        )
    };
    lines.push(format!("上下文   {ctx_summary}"));

    if audit.miss_events > 0 || audit.warm_events > 0 || cache_read > 0 || cache_write > 0 {
        let mut audit_parts = vec![format!(
            "Cache Re-billed: {} tokens",
            format_tokens(audit.cache_rebilled_tokens)
        )];
        if audit.miss_events > 0 {
            if audit.cache_rebilled_cost_usd > 0.0 {
                audit_parts.push(format!(
                    "{} 次意外失效 · 额外 ${:.4}",
                    audit.miss_events, audit.cache_rebilled_cost_usd
                ));
            } else {
                audit_parts.push(format!("{} 次意外失效", audit.miss_events));
            }
        }
        if audit.warm_events > 0 {
            if audit.total_warm_cost_usd > 0.0 {
                audit_parts.push(format!(
                    "续命 {} 次（${:.4}）",
                    audit.warm_events, audit.total_warm_cost_usd
                ));
            } else {
                audit_parts.push(format!("续命 {} 次", audit.warm_events));
            }
        }
        lines.push(format!("缓存审计 {}", audit_parts.join(" · ")));
    }

    lines.join("\n")
}

/// 将持久化会话日志投影为纯前端可重放的 `SessionViewData`（含 `!` 与 `!!` 本地命令记录）。
pub fn project_session_view(
    session: &Session,
    finish_notice: impl Into<String>,
) -> SessionViewData {
    let header = session.header();
    let entries = session.all_entries();
    let (audit_samples, fallback_rates) = collect_usage_audit_samples(&entries, &[]);
    let cache_miss_audit = compute_cache_miss_audit(&audit_samples, fallback_rates.as_ref());
    let mut total_usage = TokenUsage::default();
    let mut saw_usage = false;
    let mut last_prompt_tokens: Option<u64> = None;
    let mut items = Vec::new();
    for entry in entries {
        match entry.event {
            SessionEvent::User { text, images } => {
                let display = if !images.is_empty() && !text.contains("[Image #") {
                    if text.is_empty() {
                        format!("[图片 {} 张]", images.len())
                    } else {
                        format!("[图片 {} 张]\n{text}", images.len())
                    }
                } else {
                    text
                };
                items.push(SessionReplayItem::User { text: display });
            }
            SessionEvent::ContextNotice {
                source,
                text,
                images,
            } => {
                let display = if !images.is_empty() && text.is_empty() {
                    format!("<attached-image count=\"{}\" />", images.len())
                } else {
                    text
                };
                items.push(SessionReplayItem::ContextNotice {
                    source,
                    text: display,
                });
            }
            SessionEvent::Compaction {
                summary,
                first_kept_seq,
                tokens_after,
                ..
            } => {
                if let Some(after) = tokens_after.filter(|&v| v > 0) {
                    last_prompt_tokens = Some(after);
                }
                items.push(SessionReplayItem::Compaction {
                    summary: strip_summary_wrapper(&summary),
                    first_kept_seq,
                });
            }
            SessionEvent::Assistant { blocks, usage, .. } => {
                if let Some(u) = usage {
                    saw_usage = true;
                    accumulate_usage(&mut total_usage, &u);
                    let prompt = u.prompt_tokens();
                    if prompt > 0 {
                        last_prompt_tokens = Some(prompt);
                    }
                }
                items.push(SessionReplayItem::Assistant { blocks });
            }
            SessionEvent::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => items.push(SessionReplayItem::ToolResult {
                tool_call_id,
                content,
                is_error,
            }),
            SessionEvent::BashExecution {
                command,
                output,
                exit_code,
                cancelled,
                exclude_from_context,
            } => items.push(SessionReplayItem::BashExecution {
                command,
                output,
                exit_code,
                cancelled,
                exclude_from_context,
            }),
            SessionEvent::Trace { detail } => match detail {
                SessionTraceEvent::TurnNotice { severity, text, .. } => {
                    let level = match severity {
                        TraceSeverity::Info => NoticeLevel::Info,
                        TraceSeverity::Warn => NoticeLevel::Warn,
                        TraceSeverity::Error => NoticeLevel::Error,
                    };
                    items.push(SessionReplayItem::Notice { level, text });
                }
                SessionTraceEvent::TurnEnd { reason, .. } if reason == "truncated" => {
                    items.push(SessionReplayItem::Notice {
                        level: NoticeLevel::Warn,
                        text: "输出因达到 max_tokens 上限被截断，回复可能不完整".to_string(),
                    });
                }
                _ => {}
            },
            SessionEvent::Selection { .. } | SessionEvent::Usage { .. } => {}
        }
    }
    if saw_usage {
        total_usage.total_tokens = Some(total_usage.grand_total());
    }
    SessionViewData {
        session_id: session.id.clone(),
        session_title: header.title,
        items,
        finish_notice: finish_notice.into(),
        selection: session.latest_selection().map(to_ui_selection),
        total_usage,
        last_prompt_tokens,
        cache_miss_audit,
    }
}

/// 生成恢复会话的重放视图。
pub fn project_resume_view(session: &Session) -> SessionViewData {
    let header = session.header();
    let title_suffix = header
        .title
        .as_deref()
        .map(|t| format!(" · {t}"))
        .unwrap_or_default();
    let finish_notice = format!(
        "已恢复历史会话 `{}`{}（共 {} 条表面事件）",
        short_id(&session.id),
        title_suffix,
        header.surface_count,
    );
    project_session_view(session, finish_notice)
}

/// 生成分叉会话的重放视图。
pub fn project_fork_view(session: &Session, from_seq: u64) -> SessionViewData {
    let finish_notice = format!(
        "🌿 已从事件 #{from_seq} 分叉至新会话 `{}`",
        short_id(&session.id),
    );
    project_session_view(session, finish_notice)
}

/// 将持久化会话日志投影为 `/tree` 覆盖层条目列表（保留 `!` 与 `!!` 归属标识）。
pub fn project_session_tree(session: &Session) -> Vec<SessionTreeEntry> {
    session
        .all_entries()
        .into_iter()
        .filter_map(|entry| match entry.event {
            SessionEvent::User { text, images } => {
                let prefix = if images.is_empty() {
                    String::new()
                } else {
                    format!("[图×{}] ", images.len())
                };
                Some(SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: "USER".to_string(),
                    summary: format!("{prefix}{}", truncate_display(&text.replace('\n', " "), 64)),
                    depth: 0,
                    can_fork: true,
                })
            }
            SessionEvent::ContextNotice { source, text, .. } => Some(SessionTreeEntry {
                seq: entry.seq,
                kind_tag: "CTX ".to_string(),
                summary: format!(
                    "[{source}] {}",
                    truncate_display(&text.replace('\n', " "), 60)
                ),
                depth: 1,
                can_fork: false,
            }),
            SessionEvent::Assistant { blocks, .. } => {
                let mut parts = Vec::new();
                for block in &blocks {
                    match block {
                        ContentBlock::Text { text } if !text.trim().is_empty() => {
                            parts.push(text.trim().replace('\n', " "));
                        }
                        ContentBlock::ToolCall { name, .. } => {
                            parts.push(format!("[call:{name}]"));
                        }
                        ContentBlock::Reasoning { .. } | ContentBlock::Text { .. } => {}
                    }
                }
                let summary = if parts.is_empty() {
                    "(empty assistant turn)".to_string()
                } else {
                    truncate_display(&parts.join(" "), 72)
                };
                Some(SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: "ASST".to_string(),
                    summary,
                    depth: 1,
                    can_fork: true,
                })
            }
            SessionEvent::ToolResult {
                name,
                content,
                is_error,
                ..
            } => Some(SessionTreeEntry {
                seq: entry.seq,
                kind_tag: if is_error { "ERR " } else { "TOOL" }.to_string(),
                summary: format!(
                    "{name}: {}",
                    truncate_display(&content.replace('\n', " "), 60)
                ),
                depth: 1,
                can_fork: true,
            }),
            SessionEvent::Compaction {
                summary,
                first_kept_seq,
                ..
            } => Some(SessionTreeEntry {
                seq: entry.seq,
                kind_tag: "COMP".to_string(),
                summary: format!(
                    "[keep>={first_kept_seq}] {}",
                    truncate_display(&summary.replace('\n', " "), 56)
                ),
                depth: 0,
                can_fork: false,
            }),
            SessionEvent::BashExecution {
                command,
                exit_code,
                cancelled,
                exclude_from_context,
                ..
            } => {
                let kind_tag = if exclude_from_context {
                    "BASH!!"
                } else {
                    "BASH! "
                };
                let prefix = if exclude_from_context { "!!" } else { "!" };
                let status_hint = if cancelled {
                    "[取消] "
                } else if let Some(code) = exit_code
                    && code != 0
                {
                    "[失败] "
                } else {
                    ""
                };
                Some(SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: kind_tag.to_string(),
                    summary: format!(
                        "{prefix} {status_hint}{}",
                        truncate_display(&command.replace('\n', " "), 60)
                    ),
                    depth: 0,
                    can_fork: true,
                })
            }
            SessionEvent::Selection { .. }
            | SessionEvent::Usage { .. }
            | SessionEvent::Trace { .. } => None,
        })
        .collect()
}

fn format_cwd_tail(raw: Option<&str>) -> String {
    let Some(cwd) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return "-".to_string();
    };
    let segs: Vec<&str> = cwd.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
    match segs.len() {
        0 => cwd.to_string(),
        1 => segs[0].to_string(),
        n => {
            let two = format!("{}/{}", segs[n - 2], segs[n - 1]);
            if two.chars().count() <= 20 {
                two
            } else {
                segs[n - 1].to_string()
            }
        }
    }
}

fn format_session_updated_at(ms: u128) -> String {
    i64::try_from(ms)
        .ok()
        .filter(|&v| v > 0)
        .and_then(DateTime::from_timestamp_millis)
        .map(|utc| utc.with_timezone(&Local).format("%m-%d %H:%M").to_string())
        .unwrap_or_else(|| "-".to_string())
}

/// 仅通过 `list_headers` 轻量查询非空历史会话列表用于 `/resume` 选择器（不加载事件日志）。
pub fn collect_session_picker_items(sessions: &SessionStore) -> Vec<SessionPickerItem> {
    sessions
        .list_headers(&SessionListQuery {
            cwd: None,
            limit: Some(100),
            exclude_subagents: true,
        })
        .unwrap_or_default()
        .into_iter()
        .filter(|h| h.event_count > 0)
        .map(|h| {
            let title = h
                .title
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("(无标题 #{})", short_id(&h.id)));
            SessionPickerItem {
                title,
                cwd_tail: format_cwd_tail(h.cwd.as_deref()),
                event_count: h.event_count,
                updated_at: format_session_updated_at(h.updated_at_ms),
                session_id: h.id,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use mizar_llm::{ModelInfo, ProviderInfo, ReasoningEffort};

    use super::*;

    fn sel(provider: &str, model: &str, effort: ReasoningEffort) -> SessionSelection {
        SessionSelection {
            provider: Some(provider.to_string()),
            model: Some(model.to_string()),
            reasoning_effort: Some(effort),
        }
    }

    /// 优先级链：MODEL 环境变量 > resume 的本会话记录 > settings.json > 最近会话记录 > models.json 推导。
    #[test]
    fn resolve_selection_enforces_precedence_chain() {
        let env_wins = SelectionSources {
            model_from_env: true,
            model_from_settings: true,
            reasoning_effort_from_settings: true,
            startup: sel("magpie", "from-env", ReasoningEffort::High),
            session: Some(sel("magpie", "from-session", ReasoningEffort::Max)),
            latest_workspace: Some(sel("magpie", "from-latest", ReasoningEffort::Low)),
        };
        assert_eq!(resolve_selection(&env_wins, true), None);
        assert_eq!(resolve_selection(&env_wins, false), None);

        let resume_over_settings = SelectionSources {
            model_from_settings: true,
            reasoning_effort_from_settings: true,
            startup: sel("magpie", "settings-model", ReasoningEffort::Low),
            session: Some(sel("isok", "session-model", ReasoningEffort::Max)),
            latest_workspace: Some(sel("magpie", "latest-model", ReasoningEffort::Medium)),
            ..Default::default()
        };
        assert_eq!(
            resolve_selection(&resume_over_settings, true),
            Some(sel("isok", "session-model", ReasoningEffort::Max))
        );
        // 冷启动时 settings 压过最近会话记录。
        assert_eq!(resolve_selection(&resume_over_settings, false), None);

        // 冷启动且 settings 静默时沿用最近会话；强度被 settings 钉住时仅 resume 可覆盖。
        let cold_carry = SelectionSources {
            startup: sel("magpie", "derived", ReasoningEffort::Medium),
            latest_workspace: Some(sel("isok", "last-used", ReasoningEffort::Max)),
            ..Default::default()
        };
        assert_eq!(
            resolve_selection(&cold_carry, false),
            Some(sel("isok", "last-used", ReasoningEffort::Max))
        );

        let same_as_startup = SelectionSources {
            startup: cold_carry.startup.clone(),
            session: Some(cold_carry.startup.clone()),
            latest_workspace: Some(cold_carry.startup.clone()),
            ..Default::default()
        };
        assert_eq!(resolve_selection(&same_as_startup, true), None);

        let effort_pinned = SelectionSources {
            reasoning_effort_from_settings: true,
            startup: sel("magpie", "settings-model", ReasoningEffort::Low),
            latest_workspace: Some(SessionSelection {
                reasoning_effort: Some(ReasoningEffort::Max),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(resolve_selection(&effort_pinned, false), None);
        let resumed_effort = SelectionSources {
            session: effort_pinned.latest_workspace.clone(),
            ..effort_pinned
        };
        assert_eq!(
            resolve_selection(&resumed_effort, true).map(|s| s.reasoning_effort),
            Some(Some(ReasoningEffort::Max))
        );
    }

    #[test]
    fn resolve_model_switch_handles_slashes_in_model_ids() {
        let providers = vec![ProviderInfo {
            id: "magpie".to_string(),
            name: "magpie".to_string(),
            protocol: "openai-completions".to_string(),
            models: vec![
                ModelInfo {
                    id: "mayoru-completions/gemini-4-hydrogen".to_string(),
                    name: "Gemini 4 Hydrogen".to_string(),
                    context_window: None,
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                    input: Vec::new(),
                    ..Default::default()
                },
                ModelInfo {
                    id: "mayoru-messages/claude-sonnet-5-5".to_string(),
                    name: "Claude Sonnet 5.5".to_string(),
                    context_window: None,
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                    input: Vec::new(),
                    ..Default::default()
                },
            ],
        }];

        // 带 "/" 的完整模型 ID 不应被误拆成 provider = "mayoru-completions"
        assert_eq!(
            resolve_model_switch(
                &providers,
                Some("magpie"),
                "mayoru-messages/claude-sonnet-5-5",
            ),
            (None, "mayoru-messages/claude-sonnet-5-5".to_string())
        );

        // 前缀精确命中已注册 provider ID 时，正常切换 provider 与内部带 "/" 的模型 ID
        assert_eq!(
            resolve_model_switch(
                &providers,
                None,
                "magpie/mayoru-completions/gemini-4-hydrogen",
            ),
            (
                Some("magpie".to_string()),
                "mayoru-completions/gemini-4-hydrogen".to_string(),
            )
        );
    }

    #[tokio::test]
    async fn project_resume_view_maps_turn_notice_and_truncated_turn_end_to_notice_never_assistant()
    {
        use mizar_core::spine::{AgentSpineConfig, AgentSpinePlugin, AgentToolsMode};
        use mizar_session::SessionContextExt;

        let ctx = Context::new();
        let mut cfg = AgentSpineConfig::coding(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        if let AgentToolsMode::Coding {
            ref mut mcp,
            ref mut skills,
            ref mut instructions,
            ..
        } = cfg.tools_mode
        {
            mcp.load_config_files = false;
            skills.include_default_roots = false;
            instructions.include_default_global = false;
        }
        ctx.plugin(AgentSpinePlugin, cfg);
        crate::events::wait_for_services(&ctx).await.unwrap();

        let session = ctx.sessions().unwrap().create(None);
        session.append(SessionEvent::user("hi"));
        session.append_trace(SessionTraceEvent::TurnNotice {
            turn: 1,
            severity: TraceSeverity::Warn,
            text: "Reached max steps (max_steps=2)".into(),
        });
        session.append_trace(SessionTraceEvent::TurnEnd {
            turn: 1,
            reason: "max_steps".into(),
        });
        session.append_trace(SessionTraceEvent::TurnEnd {
            turn: 2,
            reason: "truncated".into(),
        });

        assert!(
            !session
                .log()
                .iter()
                .any(|e| matches!(&e.event, SessionEvent::Assistant { .. })),
            "TurnNotice 不得进入 session.log() 作为 Assistant 消息"
        );
        assert!(
            !session
                .derive_messages()
                .iter()
                .any(|m| format!("{m:?}").contains("Reached max steps")),
            "TurnNotice 不得污染 derive_messages() 模型上下文"
        );

        let view = project_resume_view(&session);
        assert!(
            !view
                .items
                .iter()
                .any(|i| matches!(i, SessionReplayItem::Assistant { .. })),
            "重放条目中不应出现伪造的 Assistant"
        );
        assert!(view.items.iter().any(|i| matches!(
            i,
            SessionReplayItem::Notice {
                level: NoticeLevel::Warn,
                text,
            } if text == "Reached max steps (max_steps=2)"
        )));
        assert!(view.items.iter().any(|i| matches!(
            i,
            SessionReplayItem::Notice {
                level: NoticeLevel::Warn,
                text,
            } if text.contains("max_tokens")
        )));

        ctx.fiber().dispose().await;
    }

    #[tokio::test]
    async fn build_session_info_report_covers_full_identity_ledger_and_empty_omissions() {
        use cordis::Service;
        use mizar_session::{
            CompactionDigest, SessionContextExt, SessionCreateOptions, SessionEvent, SessionStore,
            SessionTraceEvent,
        };

        let ctx = Context::new();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        let store = ctx.sessions().unwrap();

        let parent = store.create(None);
        parent.append(SessionEvent::user("parent"));
        let child_id = "6f3d2c1a-8b4e-4f2a-9c71-2e5d8a0b4f13";
        let child = store
            .fork(
                &parent.id,
                1,
                SessionCreateOptions {
                    id: Some(child_id.into()),
                    cwd: Some("/repo/session-ws".into()),
                    title: Some("feat(session): /session".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        child.append_trace(SessionTraceEvent::TurnStart { turn: 1 });
        child.append(SessionEvent::Assistant {
            blocks: vec![ContentBlock::Text { text: "ok".into() }],
            replay: None,
            usage: Some(TokenUsage {
                input_tokens: 10_000,
                output_tokens: 8_700,
                cache_read_tokens: Some(40_000),
                cache_write_tokens: Some(2_000),
                ..Default::default()
            }),
        });
        child.compact_at(
            "sum",
            2,
            Some(42_100),
            Some(6_800),
            CompactionDigest::default(),
        );

        let entries_before = child.all_entries().len();
        let fake_db = PathBuf::from("/tmp/mizar_sessions.sqlite");
        let breakdown = ContextBreakdownSnapshot::default();
        let report = build_session_info_report_at(
            &child,
            true,
            Some(&fake_db),
            128_000,
            None,
            &breakdown,
            child.header().updated_at_ms + 3 * 60 * 1_000,
        );
        // 只读查询不能改写会话日志
        assert_eq!(child.all_entries().len(), entries_before);
        assert!(report.contains(&format!("ID       {child_id}")), "{report}");
        assert!(report.contains("工作区   /repo/session-ws"), "{report}");
        assert!(
            !report.contains("/completely/different/workspace"),
            "{report}"
        );
        assert!(report.contains(&format!("分叉自 {}", &parent.id[..8])));
        assert!(report.contains("压缩     1 次 · 最新保留自 #2 · 最近 42.1k → 6.8k"));
        assert!(report.contains("Token    累计 ↑52.0k（cache 40.0k 读 / 2.0k 写）↓8.7k"));
        assert!(report.contains("上下文   6.8k / 128k（5.3%）"), "{report}");

        // project_session_view 汇总用法供 --resume 还原
        let view = project_session_view(&child, "resumed");
        assert_eq!(view.total_usage.grand_total(), 60_700);

        // 空白内存会话省略谱系与压缩行
        let fresh = store.create(None);
        let fresh_report = build_session_info_report_at(
            &fresh,
            false,
            None,
            128_000,
            None,
            &breakdown,
            fresh.header().updated_at_ms + 5_000,
        );
        assert!(!fresh_report.contains("谱系") && !fresh_report.contains("压缩"));
        assert!(fresh_report.contains("存储     仅内存（未持久化）"));

        ctx.fiber().dispose().await;
    }

    #[tokio::test]
    async fn build_session_info_report_renders_rebilled_audit() {
        use cordis::Service;
        use mizar_llm::ModelCostRates;
        use mizar_session::{SessionContextExt, SessionEvent, SessionStore};

        let ctx = Context::new();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        let store = ctx.sessions().unwrap();
        let session = store.create(None);
        session.record_selection(Some("anthropic".into()), Some("claude-sonnet".into()), None);
        session.append(SessionEvent::Assistant {
            blocks: vec![ContentBlock::Text { text: "s1".into() }],
            replay: None,
            usage: Some(TokenUsage {
                input_tokens: 500,
                cache_write_tokens: Some(80_000),
                output_tokens: 100,
                ..Default::default()
            }),
        });
        session.append(SessionEvent::usage(
            "cache_warm",
            "anthropic",
            "claude-sonnet",
            TokenUsage {
                cache_read_tokens: Some(80_000),
                output_tokens: 1,
                ..Default::default()
            },
            None,
        ));
        session.append(SessionEvent::Assistant {
            blocks: vec![ContentBlock::Text { text: "s2".into() }],
            replay: None,
            usage: Some(TokenUsage {
                input_tokens: 200,
                cache_read_tokens: Some(0),
                cache_write_tokens: Some(80_000),
                output_tokens: 120,
                ..Default::default()
            }),
        });

        let providers = vec![ProviderInfo {
            id: "anthropic".into(),
            name: "Anthropic".into(),
            protocol: "anthropic-messages".into(),
            models: vec![ModelInfo {
                id: "claude-sonnet".into(),
                name: "Claude Sonnet".into(),
                cost: Some(ModelCostRates {
                    input: 3.0,
                    output: 15.0,
                    cache_read: 0.3,
                    cache_write: 3.75,
                    tiers: Vec::new(),
                }),
                ..Default::default()
            }],
        }];
        let breakdown = ContextBreakdownSnapshot::default();
        let report = build_session_info_report_at_with_cache(
            &session,
            false,
            None,
            200_000,
            Some(80_200),
            &breakdown,
            session.header().updated_at_ms + 1_000,
            &providers,
        );

        assert!(!report.contains("缓存续命"), "{report}");
        assert!(
            report.contains("缓存审计 Cache Re-billed: 80.0k tokens · 1 次意外失效 · 额外 $0.2760 · 续命 1 次（$0.0240）"),
            "{report}"
        );

        let view = project_session_view(&session, "ok");
        assert_eq!(view.cache_miss_audit.cache_rebilled_tokens, 80_000);
        assert_eq!(view.cache_miss_audit.miss_events, 1);
        assert_eq!(view.cache_miss_audit.warm_events, 1);

        ctx.fiber().dispose().await;
    }

    #[tokio::test]
    async fn latest_workspace_selection_skips_newer_subagent_child_session() {
        use cordis::Service;
        use mizar_session::{SessionContextExt, SessionCreateOptions};

        let ctx = Context::new();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        let store = ctx.sessions().unwrap();
        let ws = PathBuf::from("/tmp/mizar-ws-sel");

        let parent = store.create_with(SessionCreateOptions {
            id: Some("parent-main".into()),
            cwd: Some(ws.display().to_string()),
            ..Default::default()
        });
        parent.record_selection(
            Some("anthropic".to_string()),
            Some("claude-sonnet".to_string()),
            None,
        );

        std::thread::sleep(std::time::Duration::from_millis(5));
        let child = store.create_with(SessionCreateOptions {
            id: Some("child-sub".into()),
            cwd: Some(ws.display().to_string()),
            parent_id: Some("parent-main".into()),
            ..Default::default()
        });
        child.append(SessionEvent::user("child task"));

        let sel = latest_workspace_selection(&store, &ws).expect("should find parent selection");
        assert_eq!(sel.provider.as_deref(), Some("anthropic"));
        assert_eq!(sel.model.as_deref(), Some("claude-sonnet"));

        ctx.fiber().dispose().await;
    }
}
