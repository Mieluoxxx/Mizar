//! `mizar rpc` 无头单会话宿主的端到端集成测试（验证 `--ready-file`、`0600`、`auth.hello`、`bash.run`、`session.new` 与 `--idle-exit-secs` 自动退出）。

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use mizar_client::{ProcessMeta, meta_path_for_pid, socket_path_for_pid};
use mizar_types::rpc::{
    AuthHelloResult, BashRunResult, JsonRpcId, JsonRpcRequest, JsonRpcResponse, METHOD_AUTH_HELLO,
    METHOD_BASH_RUN, METHOD_SESSION_NEW, METHOD_SESSION_SUBSCRIBE, SessionNewResult,
    SessionSubscribeSnapshot,
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

static SEQ: AtomicU64 = AtomicU64::new(1);

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let id = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mizar_{tag}_{}_{id}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &PathBuf {
        &self.0
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct ChildGuard(Option<Child>);

impl ChildGuard {
    fn pid(&self) -> u32 {
        self.0.as_ref().expect("child present").id()
    }

    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        match self.0.as_mut() {
            Some(c) => c.try_wait(),
            None => Ok(None),
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

async fn rpc_call<T: serde::de::DeserializeOwned>(
    reader: &mut BufReader<tokio::io::ReadHalf<TcpStream>>,
    writer: &mut tokio::io::WriteHalf<TcpStream>,
    id: i64,
    method: &str,
    params: Value,
) -> T {
    let req = JsonRpcRequest::new(JsonRpcId::Number(id), method, Some(params));
    let mut line = serde_json::to_string(&req).unwrap();
    line.push('\n');
    writer.write_all(line.as_bytes()).await.unwrap();
    writer.flush().await.unwrap();

    let expected = JsonRpcId::Number(id);
    loop {
        let mut raw = String::new();
        let n = tokio::time::timeout(Duration::from_secs(5), reader.read_line(&mut raw))
            .await
            .expect("timed out waiting RPC response")
            .expect("read RPC line");
        assert!(n > 0, "unexpected EOF waiting response for {method}");
        let val: Value = serde_json::from_str(raw.trim_end()).unwrap();
        if val.get("id").is_some() {
            let resp: JsonRpcResponse = serde_json::from_value(val).unwrap();
            if resp.id.as_ref() == Some(&expected) {
                assert!(resp.error.is_none(), "{method} failed: {:?}", resp.error);
                return serde_json::from_value(resp.result.unwrap()).unwrap();
            }
        }
    }
}

#[tokio::test]
async fn rpc_subcommand_writes_ready_file_handles_rpc_and_exits_on_idle_disconnect() {
    let tmp = TempDirGuard::new("rpc_cli");
    let mizar_home = tmp.path().join("home");
    let ws = tmp.path().join("workspace");
    std::fs::create_dir_all(&ws).unwrap();
    let ready_file = tmp.path().join("ready").join("host.json");
    let run_dir = mizar_home.join("run");

    let bin = env!("CARGO_BIN_EXE_mizar");
    let mut cmd = Command::new(bin);
    cmd.arg("rpc")
        .arg("--workspace")
        .arg(&ws)
        .arg("--ready-file")
        .arg(&ready_file)
        .arg("--idle-exit-secs")
        .arg("1")
        .env("MIZAR_HOME", &mizar_home)
        .env_remove("API_KEY")
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("DEEPSEEK_API_KEY")
        .env_remove("MODEL")
        .env_remove("BASE_URL")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let mut child = ChildGuard(Some(cmd.spawn().expect("spawn mizar rpc")));
    let child_pid = child.pid();

    let deadline = Instant::now() + Duration::from_secs(8);
    while !ready_file.exists() {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("mizar rpc exited prematurely with {status}");
        }
        assert!(Instant::now() < deadline, "timed out waiting ready file");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let mode = std::fs::metadata(&ready_file).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "ready file must use 0600 permissions");

    let meta: ProcessMeta = serde_json::from_slice(&std::fs::read(&ready_file).unwrap()).unwrap();
    assert_eq!(meta.pid, child_pid);
    let tcp_addr = meta.tcp_addr.expect("tcp_addr in ready file");
    let token = meta.rpc_token.expect("rpc_token in ready file");

    let pid_meta_path = meta_path_for_pid(&run_dir, child_pid);
    let pid_sock_path = socket_path_for_pid(&run_dir, child_pid);
    assert!(pid_meta_path.exists());
    assert!(pid_sock_path.exists());

    let stream = TcpStream::connect(&tcp_addr)
        .await
        .expect("connect to mizar rpc loopback tcp");
    let (read_half, mut writer) = tokio::io::split(stream);
    let mut reader = BufReader::new(read_half);

    let auth: AuthHelloResult = rpc_call(
        &mut reader,
        &mut writer,
        1,
        METHOD_AUTH_HELLO,
        json!({ "token": token }),
    )
    .await;
    assert!(auth.ok);

    let snap: SessionSubscribeSnapshot = rpc_call(
        &mut reader,
        &mut writer,
        2,
        METHOD_SESSION_SUBSCRIBE,
        json!({ "session_id": meta.session_id }),
    )
    .await;
    assert_eq!(snap.session.id, meta.session_id);

    let bash: BashRunResult = rpc_call(
        &mut reader,
        &mut writer,
        3,
        METHOD_BASH_RUN,
        json!({
            "session_id": meta.session_id,
            "command": "echo RPC_CLI_SMOKE_OK",
            "exclude_from_context": false
        }),
    )
    .await;
    assert_eq!(bash.exit_code, Some(0));
    assert!(bash.output.contains("RPC_CLI_SMOKE_OK"));

    let new_sess: SessionNewResult = rpc_call(
        &mut reader,
        &mut writer,
        4,
        METHOD_SESSION_NEW,
        json!({ "title": "Headless Next" }),
    )
    .await;
    assert_ne!(new_sess.session.id, meta.session_id);

    // 断开客户端后等待 --idle-exit-secs 1 自动退出，并验证运行态文件清理完毕
    drop(writer);
    drop(reader);

    let exit_deadline = Instant::now() + Duration::from_secs(8);
    let exit_status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < exit_deadline,
            "mizar rpc did not exit after idle disconnect"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert!(exit_status.success());
    assert!(!pid_meta_path.exists(), "p-<pid>.json should be cleaned up");
    assert!(!pid_sock_path.exists(), "p-<pid>.sock should be cleaned up");
}
