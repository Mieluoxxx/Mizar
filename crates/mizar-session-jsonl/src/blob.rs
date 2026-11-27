//! 基于 SHA-256 内容寻址的图片附件外置存储（消除单行 JSONL 体积膨胀与多会话重复副本）。

use std::borrow::Cow;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use mizar_llm_types::ImageAttachment;
use mizar_session::{PersistenceResult, SessionEvent};

pub const BLOB_REF_PREFIX: &str = "blob:sha256:";

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// 按 RFC 6234 计算小写十六进制 SHA-256 摘要（零外部加密库依赖）。
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((data.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for block in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = String::with_capacity(64);
    for word in h {
        let _ = write!(&mut out, "{word:08x}");
    }
    out
}

pub fn resolve_blobs_dir(sessions_dir: &Path) -> PathBuf {
    match sessions_dir.parent() {
        // `Path::new("sessions").parent()` 返回 `Some("")`，只有非空父目录才挂同级 `blobs/`
        Some(parent) if !parent.as_os_str().is_empty() => parent.join("blobs"),
        _ => sessions_dir.join("blobs"),
    }
}

#[derive(Debug, Clone)]
pub struct BlobStore {
    blobs_dir: PathBuf,
    tmp_seq: Arc<AtomicU64>,
}

impl BlobStore {
    pub fn new(sessions_dir: &Path) -> Self {
        Self {
            blobs_dir: resolve_blobs_dir(sessions_dir),
            tmp_seq: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn blobs_dir(&self) -> &Path {
        &self.blobs_dir
    }

    fn store_base64(&self, data: &str) -> PersistenceResult<String> {
        let hex = sha256_hex(data.as_bytes());
        let target = self.blobs_dir.join(&hex);
        let blob_ref = format!("{BLOB_REF_PREFIX}{hex}");
        if target.try_exists()? {
            return Ok(blob_ref);
        }

        std::fs::create_dir_all(&self.blobs_dir)?;
        let seq = self.tmp_seq.fetch_add(1, Ordering::Relaxed);
        let tmp = self
            .blobs_dir
            .join(format!(".{hex}.{}.{seq}.tmp", std::process::id()));
        if let Err(err) = std::fs::write(&tmp, data.as_bytes()) {
            let _ = std::fs::remove_file(&tmp);
            return Err(err.into());
        }
        if let Err(err) = std::fs::rename(&tmp, &target) {
            let _ = std::fs::remove_file(&tmp);
            // 多进程并发写入同一内容寻址文件时，目标文件已存在即可直接复用
            if target.try_exists().unwrap_or(false) {
                return Ok(blob_ref);
            }
            return Err(err.into());
        }
        Ok(blob_ref)
    }

    fn load_base64_hex(&self, hex: &str) -> Result<String, String> {
        if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("invalid sha256 digest".to_string());
        }
        let path = self.blobs_dir.join(hex.to_ascii_lowercase());
        let data = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        if data.is_empty() {
            return Err("empty blob file".to_string());
        }
        Ok(data)
    }

    /// 在写入 JSONL 前将内联图片 base64 原子外置到 `blobs/<sha256>`，替换为引用串。
    pub fn externalize_event<'a>(
        &self,
        event: &'a SessionEvent,
    ) -> PersistenceResult<Cow<'a, SessionEvent>> {
        let needs_externalize = event_images(event).is_some_and(|imgs| {
            imgs.iter()
                .any(|img| !img.data.is_empty() && !img.data.starts_with(BLOB_REF_PREFIX))
        });
        if !needs_externalize {
            return Ok(Cow::Borrowed(event));
        }

        let mut cloned = event.clone();
        if let Some(images) = event_images_mut(&mut cloned) {
            for img in images {
                if !img.data.is_empty() && !img.data.starts_with(BLOB_REF_PREFIX) {
                    img.data = self.store_base64(&img.data)?;
                }
            }
        }
        Ok(Cow::Owned(cloned))
    }

    /// 还原事件中引用外置 Blob 的图片；若对应文件缺失或损坏，仅剔除该图片并告警，不中断会话恢复。
    pub fn restore_event(&self, event: &mut SessionEvent, logger: Option<&cordis::Logger>) {
        let Some(images) = event_images_mut(event) else {
            return;
        };
        let mut idx = 0usize;
        while idx < images.len() {
            if let Some(hex_ref) = images[idx].data.strip_prefix(BLOB_REF_PREFIX) {
                match self.load_base64_hex(hex_ref) {
                    Ok(data) => {
                        images[idx].data = data;
                        idx += 1;
                    }
                    Err(reason) => {
                        let msg = format!(
                            "图片附件 {}{} 读取失败，已从会话事件中剔除：{reason}",
                            BLOB_REF_PREFIX, hex_ref
                        );
                        if let Some(logger) = logger {
                            logger.warn(msg);
                        } else {
                            eprintln!("[W session-jsonl] {msg}");
                        }
                        images.remove(idx);
                    }
                }
            } else {
                idx += 1;
            }
        }
    }
}

fn event_images(event: &SessionEvent) -> Option<&[ImageAttachment]> {
    match event {
        SessionEvent::User { images, .. }
        | SessionEvent::ContextNotice { images, .. }
        | SessionEvent::ToolResult { images, .. } => Some(images),
        _ => None,
    }
}

fn event_images_mut(event: &mut SessionEvent) -> Option<&mut Vec<ImageAttachment>> {
    match event {
        SessionEvent::User { images, .. }
        | SessionEvent::ContextNotice { images, .. }
        | SessionEvent::ToolResult { images, .. } => Some(images),
        _ => None,
    }
}
