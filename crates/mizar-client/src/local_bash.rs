//! 本地 `!` / `!!` Shell 命令执行、会话审计落盘与 `LocalBashFinished` 事件构造。

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use mizar_client_core::{ClientEvent, LocalBashOutput};
use mizar_session::{Session, SessionEvent};
use mizar_tools_coding::bash::execute_bash_process;
use tokio_util::sync::CancellationToken;

static LOCAL_BASH_SEQ: AtomicU64 = AtomicU64::new(1);

/// 分配进程内唯一的本地 `!` / `!!` 命令调用 ID。
pub fn next_local_bash_call_id() -> String {
    let seq = LOCAL_BASH_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("local-bash-{seq}")
}

/// 在指定工作区与会话上执行本地 `!` / `!!` Shell 命令：
///
/// 1. 复用底层 `execute_bash_process` 完成流式预览、整棵子进程树取消与大输出截断落盘；
/// 2. 将截断结果与上下文排除标记写入 `SessionEvent::BashExecution`；
/// 3. 构造并返回 `ClientEvent::LocalBashFinished` 供上层前端更新状态与侧栏。
pub async fn execute_local_bash_for_session<F>(
    session: &Arc<Session>,
    call_id: String,
    command: String,
    cwd: &Path,
    exclude_from_context: bool,
    cancel: CancellationToken,
    on_preview: F,
) -> ClientEvent
where
    F: FnMut(&str) + Send + 'static,
{
    let res = execute_bash_process(&command, cwd, None, cancel, on_preview).await;

    let (session_out, exit_code, cancelled) = match &res {
        Ok(out) => (out.session_output(), out.exit_code, out.cancelled),
        Err(err) => (err.clone(), Some(1), false),
    };
    session.append(SessionEvent::BashExecution {
        command: command.clone(),
        output: session_out.clone(),
        exit_code,
        cancelled,
        exclude_from_context,
    });

    let payload = res.map(|out| LocalBashOutput {
        display: out.display,
        session_output: session_out,
        exit_code: out.exit_code,
        cancelled: out.cancelled,
        timed_out: out.timed_out,
    });

    ClientEvent::LocalBashFinished {
        call_id,
        command,
        exclude_from_context,
        output: payload,
    }
}
