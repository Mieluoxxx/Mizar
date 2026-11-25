//! 轻量日志门面（对应 `packages/cordis/src/logger.ts`）。
//!
//! 用环境变量 `CORDIS_LOG` 调整日志级别：`debug` / `info` / `warn` / `error` / `silent`。
//! 支持在运行时切换输出目标（如 TUI 模式下重定向或静默，避免污染 `stdout`）。

use std::fmt::{self, Display};
use std::sync::{Arc, OnceLock, RwLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Silent = 4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogRecord {
    pub level: LogLevel,
    pub tag: char,
    pub name: String,
    pub message: String,
}

pub type LogHandler = Arc<dyn Fn(&LogRecord) + Send + Sync>;

#[derive(Clone, Default)]
pub enum LogSink {
    #[default]
    Stdio,
    Silent,
    Handler(LogHandler),
}

impl fmt::Debug for LogSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stdio => f.write_str("Stdio"),
            Self::Silent => f.write_str("Silent"),
            Self::Handler(_) => f.write_str("Handler(..)"),
        }
    }
}

#[derive(Debug)]
struct LoggerState {
    level: LogLevel,
    sink: LogSink,
}

fn env_threshold() -> LogLevel {
    static THRESHOLD: OnceLock<LogLevel> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        match std::env::var("CORDIS_LOG")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "debug" => LogLevel::Debug,
            "info" => LogLevel::Info,
            "warn" => LogLevel::Warn,
            "error" => LogLevel::Error,
            "silent" => LogLevel::Silent,
            _ => LogLevel::Info,
        }
    })
}

/// 日志器：`ctx.logger().info(...)`、`ctx.logger().child("name").warn(...)`。
#[derive(Debug, Clone)]
pub struct Logger {
    name: String,
    state: Arc<RwLock<LoggerState>>,
}

impl Default for Logger {
    fn default() -> Self {
        Self::new("app")
    }
}

impl Logger {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: Arc::new(RwLock::new(LoggerState {
                level: env_threshold(),
                sink: LogSink::Stdio,
            })),
        }
    }

    /// 创建子日志器（对应 JS 版 `ctx.logger('name')`），与父日志器共享级别与输出目标。
    pub fn child(&self, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: Arc::clone(&self.state),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn level(&self) -> LogLevel {
        self.state.read().unwrap().level
    }

    pub fn set_level(&self, level: LogLevel) {
        self.state.write().unwrap().level = level;
    }

    pub fn set_sink(&self, sink: LogSink) {
        self.state.write().unwrap().sink = sink;
    }

    pub fn set_silent(&self) {
        self.set_sink(LogSink::Silent);
    }

    pub fn set_handler(&self, handler: impl Fn(&LogRecord) + Send + Sync + 'static) {
        self.set_sink(LogSink::Handler(Arc::new(handler)));
    }

    pub fn debug(&self, msg: impl Display) {
        self.log(LogLevel::Debug, 'D', msg);
    }

    pub fn info(&self, msg: impl Display) {
        self.log(LogLevel::Info, 'I', msg);
    }

    pub fn warn(&self, msg: impl Display) {
        self.log(LogLevel::Warn, 'W', msg);
    }

    pub fn error(&self, msg: impl Display) {
        self.log(LogLevel::Error, 'E', msg);
    }

    fn log(&self, level: LogLevel, tag: char, msg: impl Display) {
        let (threshold, sink) = {
            let guard = self.state.read().unwrap();
            (guard.level, guard.sink.clone())
        };
        if level < threshold {
            return;
        }
        match sink {
            LogSink::Stdio => {
                if level == LogLevel::Error {
                    eprintln!("[{tag}] {} {msg}", self.name);
                } else {
                    println!("[{tag}] {} {msg}", self.name);
                }
            }
            LogSink::Silent => {}
            LogSink::Handler(handler) => {
                let record = LogRecord {
                    level,
                    tag,
                    name: self.name.clone(),
                    message: msg.to_string(),
                };
                handler(&record);
            }
        }
    }
}
