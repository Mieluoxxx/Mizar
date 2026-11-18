//! Cordis 兼容性测试：覆盖 `NamiDSH/tests/compat.ts` 的全部 14 个核心场景，
//! 验证状态转换顺序、错误隔离、epoch 重启、五种事件分发与异步 effect。

use std::sync::{Arc, Mutex};
use std::time::Duration;

use cordis::{
    BoxError, BoxFuture, Cleanup, Context, CordisError, INTERNAL_STATUS, Inject,
    InternalStatusEvent, Next, PluginDef, Service, ValidationError,
};

#[derive(Clone, Default)]
struct LogBuf(Arc<Mutex<Vec<String>>>);

impl LogBuf {
    fn push(&self, s: impl Into<String>) {
        self.0.lock().unwrap().push(s.into());
    }

    fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

fn setup(log: &LogBuf) -> Context {
    let app = Context::new();
    let log = log.clone();
    app.on(INTERNAL_STATUS, move |ev: &InternalStatusEvent| {
        log.push(format!(
            "  [status] {}: {} -> {}",
            ev.fiber.name(),
            ev.old_state,
            ev.new_state
        ));
    })
    .unwrap();
    app
}

async fn tick(ms: u64) {
    tokio::time::sleep(Duration::from_millis(ms)).await;
}

#[derive(Debug, Clone)]
struct Db {
    name: String,
}

#[derive(Debug, Clone)]
struct LlmCfg {
    model: String,
}

#[derive(Debug, Clone)]
struct LlmSvc {
    model: String,
}

#[tokio::test]
async fn compat_01_async_plugin_loading() {
    let log = LogBuf::default();
    let app = setup(&log);

    let log_apply = log.clone();
    app.plugin(
        PluginDef::sync("hello", move |_ctx, ()| {
            log_apply.push("  apply 执行");
        }),
        (),
    );
    log.push("  ctx.plugin() 已返回");
    tick(15).await;

    assert_eq!(
        log.lines(),
        vec![
            "  [status] hello: PENDING -> LOADING",
            "  ctx.plugin() 已返回",
            "  apply 执行",
            "  [status] hello: LOADING -> ACTIVE",
        ]
    );
}

#[tokio::test]
async fn compat_02_dependency_wait_consumer_before_provider() {
    let log = LogBuf::default();
    let app = setup(&log);

    let log_c = log.clone();
    app.plugin(
        PluginDef::sync("consumer", move |ctx, ()| {
            let db = ctx.service::<Db>("db")?;
            log_c.push(format!("  consumer 拿到 {}", db.name));
            Ok::<_, CordisError>(())
        })
        .inject(["db"]),
        (),
    );
    tick(15).await;

    app.plugin(
        PluginDef::sync("provider", |ctx, ()| {
            ctx.provide_value(
                "db",
                Db {
                    name: "sqlite".to_string(),
                },
            )?;
            Ok::<_, CordisError>(())
        }),
        (),
    );
    tick(15).await;

    assert_eq!(
        log.lines(),
        vec![
            "  [status] provider: PENDING -> LOADING",
            "  [status] provider: LOADING -> ACTIVE",
            "  [status] consumer: PENDING -> LOADING",
            "  consumer 拿到 sqlite",
            "  [status] consumer: LOADING -> ACTIVE",
        ]
    );
}

#[tokio::test]
async fn compat_03_dispose_provider_consumer_cleans_up_while_service_still_readable() {
    let log = LogBuf::default();
    let app = setup(&log);

    let p = app.plugin(
        PluginDef::sync("provider", |ctx, ()| {
            ctx.provide_value(
                "db",
                Db {
                    name: "sqlite".to_string(),
                },
            )?;
            Ok::<_, CordisError>(())
        }),
        (),
    );

    let log_c = log.clone();
    app.plugin(
        PluginDef::sync("consumer", move |ctx, ()| {
            let ctx_cleanup = ctx.clone();
            let log_cleanup = log_c.clone();
            ctx.effect("consumer.cleanup", move || {
                Ok::<_, CordisError>(Some(Cleanup::sync(move || {
                    let db = ctx_cleanup.service::<Db>("db").unwrap();
                    log_cleanup.push(format!("  consumer 清理时还能用 {}", db.name));
                })))
            })?;
            Ok::<_, CordisError>(())
        })
        .inject(["db"]),
        (),
    );

    tick(15).await;
    p.dispose().await;
    tick(15).await;

    log.push(format!(
        "  根上还能取到 db 吗： {}",
        if app.get::<Db>("db").is_none() {
            "不能"
        } else {
            "能"
        }
    ));

    assert_eq!(
        log.lines(),
        vec![
            "  [status] provider: PENDING -> LOADING",
            "  [status] provider: LOADING -> ACTIVE",
            "  [status] consumer: PENDING -> LOADING",
            "  [status] consumer: LOADING -> ACTIVE",
            "  [status] provider: ACTIVE -> UNLOADING",
            "  [status] consumer: ACTIVE -> UNLOADING",
            "  consumer 清理时还能用 sqlite",
            "  [status] consumer: UNLOADING -> PENDING",
            "  [status] provider: UNLOADING -> DISPOSED",
            "  根上还能取到 db 吗： 不能",
        ]
    );
}

#[tokio::test]
async fn compat_04_duplicate_service_conflict() {
    let log = LogBuf::default();
    let app = setup(&log);

    app.plugin(
        PluginDef::sync("a", |ctx, ()| {
            ctx.provide_value("db", "A".to_string())?;
            Ok::<_, CordisError>(())
        }),
        (),
    );
    tick(15).await;

    let b = app.plugin(
        PluginDef::sync("b", |ctx, ()| {
            ctx.provide_value("db", "B".to_string())?;
            Ok::<_, CordisError>(())
        }),
        (),
    );
    if let Err(err) = b.await {
        log.push(format!("  b: {err}"));
    }
    let db = app.get::<String>("db").unwrap();
    log.push(format!("  db = {db}"));

    assert_eq!(
        log.lines(),
        vec![
            "  [status] a: PENDING -> LOADING",
            "  [status] a: LOADING -> ACTIVE",
            "  [status] b: PENDING -> LOADING",
            "  [status] b: LOADING -> UNLOADING",
            "  [status] b: UNLOADING -> FAILED",
            "  b: service \"db\" has been registered at <a>",
            "  db = A",
        ]
    );
}

#[tokio::test]
async fn compat_05_update_config_restarts_provider_and_consumer() {
    let log = LogBuf::default();
    let app = setup(&log);

    let p = app.plugin(
        PluginDef::sync("llm", |ctx, cfg: LlmCfg| {
            ctx.provide_value("llm", LlmSvc { model: cfg.model })?;
            Ok::<_, CordisError>(())
        }),
        LlmCfg {
            model: "flash".to_string(),
        },
    );

    let log_l = log.clone();
    app.plugin(
        PluginDef::sync("loop", move |ctx, ()| {
            let llm = ctx.service::<LlmSvc>("llm")?;
            log_l.push(format!("  loop 使用 {}", llm.model));
            Ok::<_, CordisError>(())
        })
        .inject(["llm"]),
        (),
    );

    tick(15).await;
    p.ctx()
        .fiber()
        .update(LlmCfg {
            model: "pro".to_string(),
        })
        .await
        .unwrap();
    tick(15).await;

    assert_eq!(
        log.lines(),
        vec![
            "  [status] llm: PENDING -> LOADING",
            "  [status] llm: LOADING -> ACTIVE",
            "  [status] loop: PENDING -> LOADING",
            "  loop 使用 flash",
            "  [status] loop: LOADING -> ACTIVE",
            "  [status] llm: ACTIVE -> UNLOADING",
            "  [status] loop: ACTIVE -> UNLOADING",
            "  [status] loop: UNLOADING -> PENDING",
            "  [status] llm: UNLOADING -> LOADING",
            "  [status] llm: LOADING -> ACTIVE",
            "  [status] loop: PENDING -> LOADING",
            "  loop 使用 pro",
            "  [status] loop: LOADING -> ACTIVE",
        ]
    );
}

#[tokio::test]
async fn compat_06_error_isolation() {
    let log = LogBuf::default();
    let app = setup(&log);

    let bad = app.plugin(
        PluginDef::new("bad", |_ctx, ()| async {
            tick(2).await;
            Err::<(), BoxError>("boom".into())
        }),
        (),
    );
    let log_g = log.clone();
    app.plugin(
        PluginDef::sync("good", move |_ctx, ()| {
            log_g.push("  good 正常运行");
        }),
        (),
    );

    if let Err(err) = bad.await {
        log.push(format!("  await 拿到错误: {err}"));
    }
    tick(15).await;

    assert_eq!(
        log.lines(),
        vec![
            "  [status] bad: PENDING -> LOADING",
            "  [status] good: PENDING -> LOADING",
            "  good 正常运行",
            "  [status] good: LOADING -> ACTIVE",
            "  [status] bad: LOADING -> UNLOADING",
            "  [status] bad: UNLOADING -> FAILED",
            "  await 拿到错误: boom",
        ]
    );
}

#[tokio::test]
async fn compat_07_config_validation_failure() {
    #[derive(Clone, Default)]
    struct Cfg {
        port: Option<u16>,
    }

    let log = LogBuf::default();
    let app = setup(&log);

    let p = app.plugin(
        PluginDef::sync("cfg", |_ctx, _cfg: Cfg| {}).validate(|v| {
            if v.port.is_some() {
                Ok(())
            } else {
                Err(ValidationError::single("port is required"))
            }
        }),
        Cfg::default(),
    );

    if let Err(err) = p.await {
        let line1 = err.to_string();
        let second_line = line1.lines().nth(1).unwrap_or("");
        log.push(format!("   {} | {second_line}", err.name()));
    }

    assert_eq!(
        log.lines(),
        vec![
            "  [status] cfg: PENDING -> LOADING",
            "  [status] cfg: LOADING -> UNLOADING",
            "  [status] cfg: UNLOADING -> FAILED",
            "   ValidationError |   - port is required",
        ]
    );
}

#[tokio::test]
async fn compat_08_service_class_inject_config_async_init() {
    struct Store;
    impl Service for Store {
        type Config = ();
        fn service_name() -> &'static str {
            "store"
        }
        fn plugin_name() -> &'static str {
            "Store"
        }
        fn create(_ctx: &Context, _config: ()) -> Result<Arc<Self>, CordisError> {
            Ok(Arc::new(Self))
        }
    }

    #[derive(Clone)]
    struct LoopConfig {
        max: Option<usize>,
        log: LogBuf,
    }
    struct Loop {
        max: usize,
        log: LogBuf,
    }
    impl Service for Loop {
        type Config = LoopConfig;
        fn service_name() -> &'static str {
            "loop"
        }
        fn plugin_name() -> &'static str {
            "Loop"
        }
        fn inject() -> Inject {
            Inject::from(["store"])
        }
        fn validate(config: &Self::Config) -> Result<(), ValidationError> {
            if config.max.is_some() {
                Ok(())
            } else {
                Err(ValidationError::single("max must be number"))
            }
        }
        fn create(_ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
            Ok(Arc::new(Self {
                max: config.max.unwrap(),
                log: config.log,
            }))
        }
        fn init(self: Arc<Self>, _ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
            Box::pin(async move {
                tick(1).await;
                self.log.push(format!("  Loop init 完成, max = {}", self.max));
                Ok(())
            })
        }
    }

    let log = LogBuf::default();
    let app = setup(&log);

    app.plugin(
        Loop::plugin(),
        LoopConfig {
            max: Some(3),
            log: log.clone(),
        },
    );
    app.plugin(Store::plugin(), ());
    tick(30).await;

    let loop_svc = app.service::<Loop>("loop").unwrap();
    log.push(format!("  root 取 loop.max = {}", loop_svc.max));

    assert_eq!(
        log.lines(),
        vec![
            "  [status] Store: PENDING -> LOADING",
            "  [status] Store: LOADING -> ACTIVE",
            "  [status] Loop: PENDING -> LOADING",
            "  Loop init 完成, max = 3",
            "  [status] Loop: LOADING -> ACTIVE",
            "  root 取 loop.max = 3",
        ]
    );
}

#[tokio::test]
async fn compat_09_events_all_five_modes() {
    let log = LogBuf::default();
    let app = setup(&log);

    let log_l = log.clone();
    app.plugin(
        PluginDef::sync("listeners", move |ctx, ()| {
            let l1 = log_l.clone();
            ctx.on("greet", move |who: &String| {
                l1.push(format!("  emit 收到 {who}"));
            })?;
            ctx.on_bail("pick", |_: &()| -> Option<&'static str> { None })?;
            ctx.on_bail("pick", |_: &()| Some("second"))?;
            ctx.on_bail("pick", |_: &()| Some("third"))?;

            ctx.on_serial("pick", |_: ()| async { None::<&'static str> })?;
            ctx.on_serial("pick", |_: ()| async { Some("second") })?;
            ctx.on_serial("pick", |_: ()| async { Some("third") })?;

            let l2 = log_l.clone();
            ctx.on_parallel("job", move |_: ()| {
                let l2 = l2.clone();
                async move {
                    tick(1).await;
                    l2.push("  parallel 完成");
                    Ok::<_, BoxError>(())
                }
            })?;

            ctx.on_waterfall("exec", |cmd: String, next: Next<String, String>| {
                if cmd == "rm" {
                    "blocked".to_string()
                } else {
                    next.call(cmd)
                }
            })?;
            ctx.on_waterfall("exec", |cmd: String, next: Next<String, String>| {
                next.call(cmd).to_uppercase()
            })?;
            Ok::<_, CordisError>(())
        }),
        (),
    );
    tick(15).await;

    app.emit("greet", "nami".to_string());
    log.push(format!("  bail = {}", app.bail::<(), &str>("pick", ()).unwrap()));
    log.push(format!(
        "  serial = {}",
        app.serial::<(), &str>("pick", ()).await.unwrap()
    ));
    app.parallel("job", ()).await.unwrap();
    log.push(format!(
        "  waterfall rm = {}",
        app.waterfall("exec", "rm".to_string(), |c| format!("ran {c}"))
    ));
    log.push(format!(
        "  waterfall ls = {}",
        app.waterfall("exec", "ls".to_string(), |c| format!("ran {c}"))
    ));

    assert_eq!(
        log.lines(),
        vec![
            "  [status] listeners: PENDING -> LOADING",
            "  [status] listeners: LOADING -> ACTIVE",
            "  emit 收到 nami",
            "  bail = second",
            "  serial = second",
            "  parallel 完成",
            "  waterfall rm = blocked",
            "  waterfall ls = RAN LS",
        ]
    );
}

#[tokio::test]
async fn compat_10_dispose_unregisters_event_listeners() {
    let log = LogBuf::default();
    let app = setup(&log);

    let log_l = log.clone();
    let p = app.plugin(
        PluginDef::sync("l", move |ctx, ()| {
            let log_l = log_l.clone();
            ctx.on_signal("ping", move || log_l.push("  pong"))?;
            Ok::<_, CordisError>(())
        }),
        (),
    );
    tick(15).await;
    app.emit("ping", ());
    p.dispose().await;
    app.emit("ping", ());
    log.push("  卸载后 emit：无输出");

    assert_eq!(
        log.lines(),
        vec![
            "  [status] l: PENDING -> LOADING",
            "  [status] l: LOADING -> ACTIVE",
            "  pong",
            "  [status] l: ACTIVE -> UNLOADING",
            "  [status] l: UNLOADING -> DISPOSED",
            "  卸载后 emit：无输出",
        ]
    );
}

#[tokio::test]
async fn compat_11_nested_plugin_cascading_dispose() {
    let log = LogBuf::default();
    let app = setup(&log);

    let log_p = log.clone();
    let p = app.plugin(
        PluginDef::sync("parent", move |ctx, ()| {
            let log_c = log_p.clone();
            ctx.plugin(
                PluginDef::sync("child", move |_c, ()| {
                    let log_c = log_c.clone();
                    Cleanup::sync(move || log_c.push("  child 清理"))
                }),
                (),
            );
            let log_p2 = log_p.clone();
            Cleanup::sync(move || log_p2.push("  parent 清理"))
        }),
        (),
    );
    tick(15).await;
    p.dispose().await;
    tick(15).await;

    assert_eq!(
        log.lines(),
        vec![
            "  [status] parent: PENDING -> LOADING",
            "  [status] child: PENDING -> LOADING",
            "  [status] parent: LOADING -> ACTIVE",
            "  [status] child: LOADING -> ACTIVE",
            "  [status] parent: ACTIVE -> UNLOADING",
            "  parent 清理",
            "  [status] child: ACTIVE -> UNLOADING",
            "  child 清理",
            "  [status] child: UNLOADING -> DISPOSED",
            "  [status] parent: UNLOADING -> DISPOSED",
        ]
    );
}

#[tokio::test]
async fn compat_12_uninjected_service_cannot_be_read_via_service() {
    let log = LogBuf::default();
    let app = setup(&log);

    app.plugin(
        PluginDef::sync("p", |ctx, ()| {
            ctx.provide_value("secret", 42_i32)?;
            Ok::<_, CordisError>(())
        }),
        (),
    );
    tick(15).await;

    let log_q = log.clone();
    app.plugin(
        PluginDef::sync("q", move |ctx, ()| {
            if let Err(err) = ctx.service::<i32>("secret") {
                log_q.push(format!("  直接读: {err}"));
            }
            log_q.push(format!("  ctx.get 可以: {}", ctx.get::<i32>("secret").unwrap()));
        }),
        (),
    );
    tick(15).await;

    assert_eq!(
        log.lines(),
        vec![
            "  [status] p: PENDING -> LOADING",
            "  [status] p: LOADING -> ACTIVE",
            "  [status] q: PENDING -> LOADING",
            "  直接读: cannot get property \"secret\" without inject",
            "  ctx.get 可以: 42",
            "  [status] q: LOADING -> ACTIVE",
        ]
    );
}

#[tokio::test]
async fn compat_13_ctx_inject_optional_dependency() {
    let log = LogBuf::default();
    let app = setup(&log);

    let log_m = log.clone();
    app.plugin(
        PluginDef::sync("main", move |ctx, ()| {
            let log_inner = log_m.clone();
            ctx.inject(["cache"], move |c| {
                let log_inner = log_inner.clone();
                async move {
                    let cache = c.service::<String>("cache")?;
                    log_inner.push(format!("  cache 到了: {cache}"));
                    Ok::<_, CordisError>(())
                }
            });
        }),
        (),
    );
    tick(15).await;
    log.push("  （cache 还没有）");

    app.plugin(
        PluginDef::sync("cache", |ctx, ()| {
            ctx.provide_value("cache", "redis".to_string())?;
            Ok::<_, CordisError>(())
        }),
        (),
    );
    tick(15).await;

    assert_eq!(
        log.lines(),
        vec![
            "  [status] main: PENDING -> LOADING",
            "  [status] main: LOADING -> ACTIVE",
            "  （cache 还没有）",
            "  [status] cache: PENDING -> LOADING",
            "  [status] cache: LOADING -> ACTIVE",
            "  [status] main: PENDING -> LOADING",
            "  cache 到了: redis",
            "  [status] main: LOADING -> ACTIVE",
        ]
    );
}

#[tokio::test]
async fn compat_14_async_effect() {
    let log = LogBuf::default();
    let app = setup(&log);

    let log_conn = log.clone();
    let p = app.plugin(
        PluginDef::sync("conn", move |ctx, ()| {
            let log_eff = log_conn.clone();
            ctx.effect_async("conn.open", move || async move {
                tick(2).await;
                log_eff.push("  连接建立");
                let log_close = log_eff.clone();
                Ok::<_, CordisError>(Some(Cleanup::sync(move || log_close.push("  连接关闭"))))
            })?;
            Ok::<_, CordisError>(())
        }),
        (),
    );
    tick(25).await;
    p.dispose().await;

    assert_eq!(
        log.lines(),
        vec![
            "  [status] conn: PENDING -> LOADING",
            "  [status] conn: LOADING -> ACTIVE",
            "  连接建立",
            "  [status] conn: ACTIVE -> UNLOADING",
            "  连接关闭",
            "  [status] conn: UNLOADING -> DISPOSED",
        ]
    );
}
