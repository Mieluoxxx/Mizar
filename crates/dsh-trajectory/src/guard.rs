//! 实时重复工具调用守卫（直接对照 `deepseek-harness/packages/guard/repeat-tool-reminder`）。
//!
//! 挂载在 `tools/post-execute` waterfall 上：
//! - 使用深度键排序规范化参数（`{"a":1,"b":2}` 与 `{"b":2,"a":1}` 视为同一次调用）
//! - 支持 `*` 通配符 `include` / `exclude`（被排除的工具对重复链透明，既不计数也不重置）
//! - 被 `tools/pre-execute` 拦截的调用同样会计数
//! - 新一轮用户输入（`agent/turn-start`）自动清零该会话的重复链

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use cordis::{
    BoxError, BoxFuture, Cleanup, Context, Next, Plugin, ValidationError, ValidationIssue,
};
use dsh_agent_loop::{AgentTurnStartEvent, EVENT_AGENT_TURN_START};
use dsh_tools::{EVENT_TOOLS_POST_EXECUTE, PostExecuteInput, ToolResult};

use dsh_trajectory_core::{canonicalize_json, preview_chars, wildcard_matches};

pub const EVENT_GUARD_REPEAT: &str = "guard/repeat-tool";

const GENTLE_REMINDER: &str = "You are repeating the exact same tool call with identical arguments. Carefully analyze the previous result before calling again: if the task is not complete, try a different approach or different arguments instead of repeating the call.";

fn detailed_reminder(tool_name: &str, count: usize, canonical_args: &str) -> String {
    format!(
        "Repeated tool call detected:\n\
         - tool: {tool_name}\n\
         - consecutive_calls: {count}\n\
         - arguments: {canonical_args}\n\
         The repeated calls are not making progress. Do not call this tool with these exact arguments again. Inspect the latest result and choose a different action, different arguments, or finish the task if enough evidence has been gathered."
    )
}

#[derive(Debug, Clone)]
pub struct GuardRepeatEvent {
    pub session_id: String,
    pub tool_name: String,
    pub count: usize,
    pub canonical_args: String,
    pub blocked: bool,
}

#[derive(Debug, Clone)]
pub struct RepeatGuardConfig {
    /// 触发提醒的连续重复次数阈值列表（默认 `[3, 5, 8]`）。
    pub thresholds: Vec<usize>,
    /// 仅跟踪匹配这些通配符的工具；为空表示跟踪全部工具。
    pub include: Vec<String>,
    /// 排除这些工具（对重复链透明：既不计数也不重置）。
    pub exclude: Vec<String>,
    /// 详细提醒中参数预览的最大字符数（默认 500）。
    pub arguments_preview_chars: usize,
    /// 达到阈值时是否向模型追加合成上下文提醒（默认 true）。
    pub inject_reminder: bool,
    /// 可选：连续重复达到该次数后强制拦截（默认 None，仅建议不阻断）。
    pub hard_block_after: Option<usize>,
}

impl Default for RepeatGuardConfig {
    fn default() -> Self {
        Self {
            thresholds: vec![3, 5, 8],
            include: Vec::new(),
            exclude: Vec::new(),
            arguments_preview_chars: 500,
            inject_reminder: true,
            hard_block_after: None,
        }
    }
}

#[derive(Clone)]
struct Chain {
    key: (String, String),
    count: usize,
}

#[derive(Default, Clone, Copy)]
pub struct RepeatToolGuardPlugin;

impl Plugin for RepeatToolGuardPlugin {
    type Config = RepeatGuardConfig;

    fn name(&self) -> Option<&str> {
        Some("repeat-tool-guard")
    }

    fn validate(&self, config: &Self::Config) -> Result<(), ValidationError> {
        let mut issues = Vec::new();
        if config.thresholds.is_empty() {
            issues.push(
                ValidationIssue::new("repeat-tool-guard: `thresholds` must not be empty")
                    .at(["thresholds"]),
            );
        }
        let mut seen = HashSet::new();
        for &v in &config.thresholds {
            if v < 2 {
                issues.push(
                    ValidationIssue::new(format!(
                        "repeat-tool-guard: invalid threshold {v} — every threshold must be >= 2"
                    ))
                    .at(["thresholds"]),
                );
            }
            if !seen.insert(v) {
                issues.push(
                    ValidationIssue::new("repeat-tool-guard: `thresholds` must not contain duplicates")
                        .at(["thresholds"]),
                );
                break;
            }
        }
        if config.arguments_preview_chars < 1 {
            issues.push(
                ValidationIssue::new("repeat-tool-guard: `arguments_preview_chars` must be >= 1")
                    .at(["argumentsPreviewChars"]),
            );
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(ValidationError::new(issues))
        }
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let mut thresholds = config.thresholds.clone();
            thresholds.sort_unstable();
            let first_threshold = thresholds[0];
            let threshold_set: HashSet<usize> = thresholds.into_iter().collect();
            let include = config.include;
            let exclude = config.exclude;
            let preview_cap = config.arguments_preview_chars;
            let inject_reminder = config.inject_reminder;
            let hard_block_after = config.hard_block_after;

            let chains: Arc<Mutex<HashMap<String, Chain>>> = Arc::new(Mutex::new(HashMap::new()));

            let chains_post = Arc::clone(&chains);
            let ctx_post = ctx.clone();
            ctx.on_waterfall(
                EVENT_TOOLS_POST_EXECUTE,
                move |input: PostExecuteInput,
                      next: Next<PostExecuteInput, BoxFuture<'static, ToolResult>>| {
                    let chains = Arc::clone(&chains_post);
                    let ctx = ctx_post.clone();
                    let include = include.clone();
                    let exclude = exclude.clone();
                    let threshold_set = threshold_set.clone();

                    Box::pin(async move {
                        let tool_name = input.exec.name.clone();
                        let tracked = (include.is_empty()
                            || include.iter().any(|p| wildcard_matches(p, &tool_name)))
                            && !exclude.iter().any(|p| wildcard_matches(p, &tool_name));

                        let observation = if tracked
                            && let Some(session_id) = input.exec.session_id.clone()
                        {
                            let canonical = canonicalize_json(&input.exec.args);
                            let key = (tool_name.clone(), canonical.clone());
                            let count = {
                                let mut map = chains.lock().unwrap();
                                let count = match map.get(&session_id) {
                                    Some(prev) if prev.key == key => prev.count + 1,
                                    _ => 1,
                                };
                                map.insert(session_id.clone(), Chain { key, count });
                                count
                            };
                            Some((session_id, canonical, count))
                        } else {
                            None
                        };

                        let mut result = next.call(input).await;

                        if let Some((session_id, canonical, count)) = observation {
                            let hard_blocked =
                                hard_block_after.is_some_and(|limit| count >= limit);
                            if hard_blocked && !result.blocked {
                                let existing_notes = std::mem::take(&mut result.additional_contexts);
                                result = ToolResult::denied(format!(
                                    "重复调用护栏已拦截：{tool_name} 连续使用相同参数调用了 {count} 次"
                                ));
                                result.additional_contexts = existing_notes;
                            }

                            if threshold_set.contains(&count) || hard_blocked {
                                ctx.emit(
                                    EVENT_GUARD_REPEAT,
                                    GuardRepeatEvent {
                                        session_id,
                                        tool_name: tool_name.clone(),
                                        count,
                                        canonical_args: canonical.clone(),
                                        blocked: hard_blocked,
                                    },
                                );
                                if inject_reminder {
                                    let text = if count == first_threshold {
                                        GENTLE_REMINDER.to_string()
                                    } else {
                                        detailed_reminder(
                                            &tool_name,
                                            count,
                                            &preview_chars(&canonical, preview_cap),
                                        )
                                    };
                                    result = result.prepend_context("repeat-tool-reminder", text);
                                }
                            }
                        }

                        result
                    })
                },
            )?;

            // 用户新一轮提问开始时重置重复链，跨轮次的合理复查不当作死循环
            let chains_reset = Arc::clone(&chains);
            ctx.on(EVENT_AGENT_TURN_START, move |ev: &AgentTurnStartEvent| {
                chains_reset.lock().unwrap().remove(&ev.session.id);
            })?;

            Ok(None)
        })
    }
}
