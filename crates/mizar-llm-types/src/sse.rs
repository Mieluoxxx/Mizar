use serde_json::Value;

/// 解码后的单个 SSE 事件帧。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
}

/// 增量式 WHATWG SSE 解析器：跨分片缓冲、支持 `event:` / 多行 `data:` / 注释心跳。
#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,
    event: Option<String>,
    data_lines: Vec<String>,
    id: Option<String>,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// 喂入一块原始响应字节，返回本次解析出的完整 SSE 事件列表。
    /// 若收到注释心跳行（`:...`）或有效事件，`on_activity` 会被调用一次。
    pub fn push(&mut self, bytes: &[u8], mut on_activity: impl FnMut()) -> Vec<SseEvent> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();

        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line_bytes: Vec<u8> = self.buf.drain(..=pos).collect();
            // 去掉末尾的 \n 和可选的 \r
            if line_bytes.last() == Some(&b'\n') {
                line_bytes.pop();
            }
            if line_bytes.last() == Some(&b'\r') {
                line_bytes.pop();
            }
            let line = String::from_utf8_lossy(&line_bytes);
            if let Some(ev) = self.process_line(&line, &mut on_activity) {
                out.push(ev);
            }
        }
        out
    }

    /// 流结束时刷出最后残留的非空行/事件（处理未以空行结尾的服务器响应）。
    pub fn finish(&mut self, mut on_activity: impl FnMut()) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let mut rest = std::mem::take(&mut self.buf);
            if rest.last() == Some(&b'\r') {
                rest.pop();
            }
            let line = String::from_utf8_lossy(&rest);
            if let Some(ev) = self.process_line(&line, &mut on_activity) {
                out.push(ev);
            }
        }
        if let Some(ev) = self.flush_event() {
            out.push(ev);
        }
        out
    }

    fn process_line(&mut self, line: &str, on_activity: &mut impl FnMut()) -> Option<SseEvent> {
        if line.is_empty() {
            return self.flush_event();
        }
        if line.starts_with(':') {
            on_activity();
            return None;
        }
        on_activity();
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => self.data_lines.push(value.to_string()),
            "id" if !value.contains('\0') => self.id = Some(value.to_string()),
            _ => {}
        }
        None
    }

    fn flush_event(&mut self) -> Option<SseEvent> {
        let event = self.event.take();
        let id = self.id.take();
        if self.data_lines.is_empty() && event.is_none() {
            return None;
        }
        let data = std::mem::take(&mut self.data_lines).join("\n");
        Some(SseEvent { event, data, id })
    }
}

/// 对有序前缀 JSON（如 system + tools + 前置 messages）计算确定性 128-bit FNV-1a 摘要，
/// 用于 Anthropic 签名 thinking 块的前缀绑定校验，零额外依赖。
pub fn prefix_digest(value: &Value) -> String {
    let serialized = serde_json::to_vec(value).unwrap_or_default();
    const FNV_OFFSET: u128 = 0x6c62272e07bb0142_62b821756295c58d;
    const FNV_PRIME: u128 = 0x0000000001000000_000000000000013b;
    let mut hash = FNV_OFFSET;
    for byte in serialized {
        hash ^= u128::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:032x}")
}
