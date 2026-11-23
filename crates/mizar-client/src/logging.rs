//! 轻量 `tracing` 日志桥接器：将 `mizar::*` 域内日志统一路由至客户端事件或带脱敏的 stderr 输出。

use std::fmt::{self, Write as _};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use mizar_types::client::{UiLogLevel, UiLogRecord};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

type HandlerFn = Arc<dyn Fn(&UiLogRecord) + Send + Sync>;

struct RegisteredHandler {
    id: u64,
    threshold: UiLogLevel,
    handler: HandlerFn,
}

static TRACING_INSTALLED: OnceLock<()> = OnceLock::new();
static NEXT_HANDLER_ID: AtomicU64 = AtomicU64::new(1);

fn handlers() -> &'static RwLock<Vec<RegisteredHandler>> {
    static HANDLERS: OnceLock<RwLock<Vec<RegisteredHandler>>> = OnceLock::new();
    HANDLERS.get_or_init(|| RwLock::new(Vec::new()))
}

/// 将字符串解析为 `UiLogLevel`（未知取值回退为 `UiLogLevel::Info`）。
pub fn parse_log_level(raw: &str) -> UiLogLevel {
    match raw.trim().to_ascii_lowercase().as_str() {
        "debug" => UiLogLevel::Debug,
        "info" => UiLogLevel::Info,
        "warn" => UiLogLevel::Warn,
        "error" => UiLogLevel::Error,
        "silent" => UiLogLevel::Silent,
        _ => UiLogLevel::Info,
    }
}

/// 读取环境变量 `MIZAR_LOG` 的显式日志阈值覆盖。
pub fn env_log_level_override() -> Option<UiLogLevel> {
    std::env::var("MIZAR_LOG")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(|s| parse_log_level(&s))
}

/// 读取当前进程环境变量 `MIZAR_LOG` 阈值（默认 `UiLogLevel::Info`）。
pub fn env_log_level() -> UiLogLevel {
    env_log_level_override().unwrap_or(UiLogLevel::Info)
}

/// 返回日志级别对应的单字符短标签。
pub fn ui_log_level_tag(level: UiLogLevel) -> char {
    match level {
        UiLogLevel::Debug => 'D',
        UiLogLevel::Info => 'I',
        UiLogLevel::Warn => 'W',
        UiLogLevel::Error => 'E',
        UiLogLevel::Silent => 'S',
    }
}

fn severity_rank(level: UiLogLevel) -> u8 {
    match level {
        UiLogLevel::Debug => 0,
        UiLogLevel::Info => 1,
        UiLogLevel::Warn => 2,
        UiLogLevel::Error => 3,
        UiLogLevel::Silent => 4,
    }
}

fn level_matches(event_level: UiLogLevel, threshold: UiLogLevel) -> bool {
    threshold != UiLogLevel::Silent && severity_rank(event_level) >= severity_rank(threshold)
}

fn map_tracing_level(level: &Level) -> UiLogLevel {
    match *level {
        Level::ERROR => UiLogLevel::Error,
        Level::WARN => UiLogLevel::Warn,
        Level::INFO => UiLogLevel::Info,
        Level::DEBUG | Level::TRACE => UiLogLevel::Debug,
    }
}

fn is_mizar_target(target: &str) -> bool {
    target == "mizar" || target.starts_with("mizar::")
}

/// 全局日志桥接订阅句柄：Drop 时自动注销对应回调。
#[must_use = "dropping TracingBridgeGuard immediately unregisters the tracing handler"]
pub struct TracingBridgeGuard {
    id: u64,
}

impl fmt::Debug for TracingBridgeGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracingBridgeGuard")
            .field("id", &self.id)
            .finish()
    }
}

impl Drop for TracingBridgeGuard {
    fn drop(&mut self) {
        handlers()
            .write()
            .unwrap()
            .retain(|entry| entry.id != self.id);
    }
}

/// 确保全局 `tracing` 分发订阅器已安装（每进程仅安装一次，幂等）。
pub fn ensure_tracing_installed() {
    TRACING_INSTALLED.get_or_init(|| {
        let _ = tracing::subscriber::set_global_default(MizarTracingSubscriber);
    });
}

/// 注册一条 `tracing` 回调路由（用于 TUI / `exec` / 并发测试隔离捕获）。
pub fn install_tracing_handler(
    threshold: UiLogLevel,
    handler: impl Fn(&UiLogRecord) + Send + Sync + 'static,
) -> TracingBridgeGuard {
    ensure_tracing_installed();
    let id = NEXT_HANDLER_ID.fetch_add(1, Ordering::Relaxed);
    handlers().write().unwrap().push(RegisteredHandler {
        id,
        threshold,
        handler: Arc::new(handler),
    });
    TracingBridgeGuard { id }
}

/// 注册静默处理器，屏蔽默认 stderr 回退输出（直到返回的句柄 Drop）。
pub fn install_silent_tracing() -> TracingBridgeGuard {
    install_tracing_handler(UiLogLevel::Silent, |_| {})
}

struct MessageVisitor {
    message: Option<String>,
    extras: Vec<String>,
}

impl MessageVisitor {
    fn new() -> Self {
        Self {
            message: None,
            extras: Vec::new(),
        }
    }

    fn push_field(&mut self, field: &Field, value: String) {
        if field.name() == "message" {
            self.message = Some(value);
        } else {
            self.extras.push(format!("{}={value}", field.name()));
        }
    }

    fn finish(self) -> String {
        match (self.message, self.extras.is_empty()) {
            (Some(msg), true) => msg,
            (Some(msg), false) => format!("{msg} {}", self.extras.join(" ")),
            (None, true) => String::new(),
            (None, false) => self.extras.join(" "),
        }
    }
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.push_field(field, value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let mut buf = String::new();
        let _ = write!(&mut buf, "{value:?}");
        self.push_field(field, buf);
    }
}

struct MizarTracingSubscriber;

impl Subscriber for MizarTracingSubscriber {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        is_mizar_target(metadata.target())
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let meta = event.metadata();
        if !is_mizar_target(meta.target()) {
            return;
        }
        let level = map_tracing_level(meta.level());
        let name = meta
            .target()
            .strip_prefix("mizar::")
            .unwrap_or(meta.target())
            .to_string();

        let mut visitor = MessageVisitor::new();
        event.record(&mut visitor);
        let message = visitor.finish();
        let record = UiLogRecord {
            level,
            name,
            message,
        };

        // 先拷贝回调列表再释放读锁，避免回调内部 Drop Guard 时重入死锁。
        let snapshot: Vec<(UiLogLevel, HandlerFn)> = {
            let guard = handlers().read().unwrap();
            guard
                .iter()
                .map(|entry| (entry.threshold, Arc::clone(&entry.handler)))
                .collect()
        };

        if snapshot.is_empty() {
            if level_matches(record.level, env_log_level()) {
                eprintln!(
                    "[{} {}] {}",
                    ui_log_level_tag(record.level),
                    record.name,
                    record.message
                );
            }
            return;
        }

        for (threshold, cb) in snapshot {
            if level_matches(record.level, threshold) {
                cb(&record);
            }
        }
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn parse_log_level_handles_known_and_unknown_values() {
        assert_eq!(parse_log_level("debug"), UiLogLevel::Debug);
        assert_eq!(parse_log_level(" INFO "), UiLogLevel::Info);
        assert_eq!(parse_log_level("Warn"), UiLogLevel::Warn);
        assert_eq!(parse_log_level("ERROR"), UiLogLevel::Error);
        assert_eq!(parse_log_level("silent"), UiLogLevel::Silent);
        assert_eq!(parse_log_level("verbose"), UiLogLevel::Info);
    }

    #[test]
    fn tracing_handler_captures_mizar_events_respects_threshold_and_unregisters_on_drop() {
        let captured: Arc<Mutex<Vec<UiLogRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&captured);
        let guard = install_tracing_handler(UiLogLevel::Warn, move |rec| {
            sink.lock().unwrap().push(rec.clone());
        });

        tracing::info!(target: "mizar::test-unit", "should-be-filtered");
        tracing::warn!(target: "other::crate", "should-be-ignored-by-target");
        tracing::warn!(target: "mizar::test-unit", "captured-warning");
        tracing::error!(target: "mizar::test-unit", code = 42, "captured-error");

        {
            let items = captured.lock().unwrap();
            assert_eq!(items.len(), 2, "unexpected records: {items:?}");
            assert_eq!(items[0].level, UiLogLevel::Warn);
            assert_eq!(items[0].name, "test-unit");
            assert_eq!(items[0].message, "captured-warning");
            assert_eq!(items[1].level, UiLogLevel::Error);
            assert_eq!(items[1].message, "captured-error code=42");
        }

        drop(guard);
        let _silent = install_silent_tracing();
        tracing::warn!(target: "mizar::test-unit", "after-drop");
        assert_eq!(captured.lock().unwrap().len(), 2);
    }
}
