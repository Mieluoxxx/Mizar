//! 图片嗅探、头部宽高校验、EXIF 方向校正、动图首帧提取与自适应压缩管线。

use std::io::Cursor;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::codecs::gif::GifDecoder;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngDecoder;
use image::codecs::webp::WebPDecoder;
use image::imageops::FilterType;
use image::metadata::Orientation;
use image::{
    AnimationDecoder, DynamicImage, GenericImageView, ImageDecoder, ImageFormat, ImageReader,
    Limits, RgbImage,
};
use mizar_llm::ImageAttachment;
use thiserror::Error;

/// 输出图片长边像素上限。
pub const MAX_IMAGE_EDGE: u32 = 2048;
/// base64 编码后体积上限（5 MB，对齐主流视觉模型 API 硬约束）。
pub const MAX_IMAGE_BASE64_BYTES: usize = 5_000_000;
/// 允许解码的最大像素总数（1 亿像素），超限在读完文件头后立刻拒绝。
pub const MAX_IMAGE_PIXELS: u64 = 100_000_000;
/// PNG 超体积回退 JPEG 时采用的默认编码质量。
pub const JPEG_QUALITY: u8 = 85;
/// 逐步降长边重试序列。
const EDGE_STEP_DOWNS: &[u32] = &[1536, 1024, 768, 512, 384, 256, 128];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessedImage {
    pub attachment: ImageAttachment,
    pub original_width: u32,
    pub original_height: u32,
    pub original_media_type: &'static str,
    pub animated_first_frame: bool,
}

impl ProcessedImage {
    /// 生成 `[Image #N · ...]` 占位符后缀，无缩放/转码/动图降帧时返回 `None`。
    pub fn marker_suffix(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let (Some(w), Some(h)) = (self.attachment.width, self.attachment.height)
            && (w != self.original_width || h != self.original_height)
        {
            parts.push(format!(
                "{}×{} → {w}×{h}",
                self.original_width, self.original_height
            ));
        }
        if self.attachment.media_type != self.original_media_type {
            parts.push(format!(
                "{} → {}",
                self.original_media_type, self.attachment.media_type
            ));
        }
        if self.animated_first_frame {
            parts.push("first frame".to_string());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" · "))
        }
    }
}

#[derive(Debug, Error)]
pub enum ImageProcessError {
    #[error("不支持的图片格式（仅支持 PNG / JPEG / WebP / GIF）")]
    UnsupportedFormat,
    #[error("图片文件头损坏或尺寸无效：{0}")]
    CorruptHeader(String),
    #[error("图片分辨率过高（{width}×{height} = {pixels} 像素，超过上限 {limit}）")]
    TooManyPixels {
        width: u32,
        height: u32,
        pixels: u64,
        limit: u64,
    },
    #[error("图片解码失败：{0}")]
    DecodeFailed(String),
    #[error("图片编码失败：{0}")]
    EncodeFailed(String),
    #[error("图片即使压缩降维后仍超出大小上限（{base64_len} > {limit} base64 字节）")]
    BudgetExceeded { base64_len: usize, limit: usize },
}

/// 检查前 16 字节魔数，仅放行 PNG / JPEG / WebP / GIF。
pub fn sniff_image_format(bytes: &[u8]) -> Option<ImageFormat> {
    let head = &bytes[..bytes.len().min(16)];
    if head.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some(ImageFormat::Png);
    }
    if head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(ImageFormat::Jpeg);
    }
    if head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a") {
        return Some(ImageFormat::Gif);
    }
    if head.len() >= 12 && &head[0..4] == b"RIFF" && &head[8..12] == b"WEBP" {
        return Some(ImageFormat::WebP);
    }
    None
}

pub fn media_type_for_format(format: ImageFormat) -> Option<&'static str> {
    match format {
        ImageFormat::Png => Some("image/png"),
        ImageFormat::Jpeg => Some("image/jpeg"),
        ImageFormat::WebP => Some("image/webp"),
        ImageFormat::Gif => Some("image/gif"),
        _ => None,
    }
}

#[inline]
pub fn base64_encoded_len(raw_len: usize) -> usize {
    raw_len.div_ceil(3) * 4
}

fn fit_within_edge(width: u32, height: u32, max_edge: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= max_edge || longest == 0 {
        return (width, height);
    }
    let scale = f64::from(max_edge) / f64::from(longest);
    let nw = ((f64::from(width) * scale).round() as u32)
        .max(1)
        .min(max_edge);
    let nh = ((f64::from(height) * scale).round() as u32)
        .max(1)
        .min(max_edge);
    (nw, nh)
}

fn decode_with_metadata(
    bytes: &[u8],
    format: ImageFormat,
    orig_w: u32,
    orig_h: u32,
) -> Result<(DynamicImage, Orientation, bool), ImageProcessError> {
    let mut limits = Limits::default();
    limits.max_alloc = Some(512 * 1024 * 1024);
    limits.max_image_width = Some(orig_w);
    limits.max_image_height = Some(orig_h);

    match format {
        ImageFormat::Png => {
            let mut dec = PngDecoder::with_limits(Cursor::new(bytes), limits)
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
            let orientation = dec.orientation().unwrap_or(Orientation::NoTransforms);
            if dec
                .is_apng()
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?
            {
                let apng = dec
                    .apng()
                    .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
                let mut frames = apng.into_frames();
                let first = frames
                    .next()
                    .transpose()
                    .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?
                    .ok_or_else(|| {
                        ImageProcessError::DecodeFailed("APNG 不包含任何帧".to_string())
                    })?;
                let animated = frames.next().is_some();
                Ok((
                    DynamicImage::ImageRgba8(first.into_buffer()),
                    orientation,
                    animated,
                ))
            } else {
                let img = DynamicImage::from_decoder(dec)
                    .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
                Ok((img, orientation, false))
            }
        }
        ImageFormat::Jpeg => {
            let mut reader = ImageReader::with_format(Cursor::new(bytes), ImageFormat::Jpeg);
            reader.limits(limits);
            let mut dec = reader
                .into_decoder()
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
            let orientation = dec.orientation().unwrap_or(Orientation::NoTransforms);
            let img = DynamicImage::from_decoder(dec)
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
            Ok((img, orientation, false))
        }
        ImageFormat::WebP => {
            let mut dec = WebPDecoder::new(Cursor::new(bytes))
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
            limits
                .reserve(dec.total_bytes())
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
            dec.set_limits(limits)
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
            let orientation = dec.orientation().unwrap_or(Orientation::NoTransforms);
            if dec.has_animation() {
                let mut frames = dec.into_frames();
                let first = frames
                    .next()
                    .transpose()
                    .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?
                    .ok_or_else(|| {
                        ImageProcessError::DecodeFailed("动图 WebP 不包含任何帧".to_string())
                    })?;
                let animated = frames.next().is_some();
                Ok((
                    DynamicImage::ImageRgba8(first.into_buffer()),
                    orientation,
                    animated,
                ))
            } else {
                let img = DynamicImage::from_decoder(dec)
                    .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
                Ok((img, orientation, false))
            }
        }
        ImageFormat::Gif => {
            let mut dec = GifDecoder::new(Cursor::new(bytes))
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
            limits
                .reserve(dec.total_bytes())
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
            dec.set_limits(limits)
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?;
            let orientation = dec.orientation().unwrap_or(Orientation::NoTransforms);
            let mut frames = dec.into_frames();
            let first = frames
                .next()
                .transpose()
                .map_err(|e| ImageProcessError::DecodeFailed(e.to_string()))?
                .ok_or_else(|| ImageProcessError::DecodeFailed("GIF 不包含任何帧".to_string()))?;
            let animated = frames.next().transpose().is_ok_and(|f| f.is_some());
            Ok((
                DynamicImage::ImageRgba8(first.into_buffer()),
                orientation,
                animated,
            ))
        }
        _ => Err(ImageProcessError::UnsupportedFormat),
    }
}

fn encode_png_bytes(img: &DynamicImage) -> Result<Vec<u8>, ImageProcessError> {
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .map_err(|e| ImageProcessError::EncodeFailed(e.to_string()))?;
    Ok(buf)
}

fn encode_jpeg_bytes(img: &DynamicImage, quality: u8) -> Result<Vec<u8>, ImageProcessError> {
    let rgb = if img.color().has_alpha() {
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let mut out = RgbImage::new(w, h);
        for (x, y, px) in rgba.enumerate_pixels() {
            let a = u16::from(px.0[3]);
            let inv = 255 - a;
            let r = ((u16::from(px.0[0]) * a + 255 * inv) / 255) as u8;
            let g = ((u16::from(px.0[1]) * a + 255 * inv) / 255) as u8;
            let b = ((u16::from(px.0[2]) * a + 255 * inv) / 255) as u8;
            out.put_pixel(x, y, image::Rgb([r, g, b]));
        }
        out
    } else {
        img.to_rgb8()
    };
    let mut buf = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    rgb.write_with_encoder(encoder)
        .map_err(|e| ImageProcessError::EncodeFailed(e.to_string()))?;
    Ok(buf)
}

/// 对原始图片字节运行完整校验、方向纠正与自适应压缩流水线。
///
/// 顺序严格遵循：
/// 1. 读前 16 字节魔数（仅 PNG / JPEG / WebP / GIF）
/// 2. 仅读 header 取分辨率，超过 `MAX_IMAGE_PIXELS` 直接拒绝（不分配解码缓冲）
/// 3. 解码（受 `Limits` 约束），动图仅取首帧
/// 4. 若分辨率、方向、5MB 体积均合规且非动图，原样 base64 传出
/// 5. 先应用 EXIF 方向，再按 `MAX_IMAGE_EDGE` 等比缩放
/// 6. 优先尝试无损 PNG；超 5MB 转 JPEG q=85；仍超则逐档降长边重试
pub fn process_image_bytes(bytes: &[u8]) -> Result<ProcessedImage, ImageProcessError> {
    process_image_bytes_with_budget(bytes, MAX_IMAGE_BASE64_BYTES)
}

fn parse_png_ihdr_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    const PNG_SIG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if bytes.len() < 33 || !bytes.starts_with(&PNG_SIG) {
        return None;
    }
    let chunk_len = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if chunk_len != 13 || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let expected_crc = u32::from_be_bytes([bytes[29], bytes[30], bytes[31], bytes[32]]);
    if crc32_ieee(&bytes[12..29]) != expected_crc {
        return None;
    }
    let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    Some((w, h))
}

fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn process_image_bytes_with_budget(
    bytes: &[u8],
    max_base64_bytes: usize,
) -> Result<ProcessedImage, ImageProcessError> {
    let format = sniff_image_format(bytes).ok_or(ImageProcessError::UnsupportedFormat)?;
    let orig_mime = media_type_for_format(format).ok_or(ImageProcessError::UnsupportedFormat)?;

    // PNG 的 IHDR 固定位于前 33 字节；先校验 IHDR 并拒绝超大分辨率，避免无 IDAT 或畸形块触发误报或扫描整文件
    if format == ImageFormat::Png
        && let Some((ihdr_w, ihdr_h)) = parse_png_ihdr_dimensions(bytes)
    {
        if ihdr_w == 0 || ihdr_h == 0 {
            return Err(ImageProcessError::CorruptHeader(format!(
                "宽高不能为 0 ({ihdr_w}×{ihdr_h})"
            )));
        }
        let pixels = u64::from(ihdr_w).saturating_mul(u64::from(ihdr_h));
        if pixels > MAX_IMAGE_PIXELS {
            return Err(ImageProcessError::TooManyPixels {
                width: ihdr_w,
                height: ihdr_h,
                pixels,
                limit: MAX_IMAGE_PIXELS,
            });
        }
    }

    let (orig_w, orig_h) = ImageReader::with_format(Cursor::new(bytes), format)
        .into_dimensions()
        .map_err(|e| ImageProcessError::CorruptHeader(e.to_string()))?;
    if orig_w == 0 || orig_h == 0 {
        return Err(ImageProcessError::CorruptHeader(format!(
            "宽高不能为 0 ({orig_w}×{orig_h})"
        )));
    }
    let pixels = u64::from(orig_w).saturating_mul(u64::from(orig_h));
    if pixels > MAX_IMAGE_PIXELS {
        return Err(ImageProcessError::TooManyPixels {
            width: orig_w,
            height: orig_h,
            pixels,
            limit: MAX_IMAGE_PIXELS,
        });
    }

    let (mut img, orientation, animated) = decode_with_metadata(bytes, format, orig_w, orig_h)?;

    if !animated
        && orientation == Orientation::NoTransforms
        && orig_w <= MAX_IMAGE_EDGE
        && orig_h <= MAX_IMAGE_EDGE
        && base64_encoded_len(bytes.len()) <= max_base64_bytes
    {
        return Ok(ProcessedImage {
            attachment: ImageAttachment::new(orig_mime, STANDARD.encode(bytes))
                .with_dimensions(orig_w, orig_h),
            original_width: orig_w,
            original_height: orig_h,
            original_media_type: orig_mime,
            animated_first_frame: false,
        });
    }

    // 先正方向再算等比缩放，避免竖拍/横拍在带 EXIF 旋转时误算宽高
    img.apply_orientation(orientation);
    let (oriented_w, oriented_h) = img.dimensions();

    let (target_w, target_h) = fit_within_edge(oriented_w, oriented_h, MAX_IMAGE_EDGE);
    let resized = if (target_w, target_h) != (oriented_w, oriented_h) {
        img.resize_exact(target_w, target_h, FilterType::Lanczos3)
    } else {
        img.clone()
    };

    // 优先尝试无损 PNG（终端与代码截图文字锐利且通常小于 5MB base64）
    let png_bytes = encode_png_bytes(&resized)?;
    if base64_encoded_len(png_bytes.len()) <= max_base64_bytes {
        return Ok(ProcessedImage {
            attachment: ImageAttachment::new("image/png", STANDARD.encode(&png_bytes))
                .with_dimensions(target_w, target_h),
            original_width: oriented_w,
            original_height: oriented_h,
            original_media_type: orig_mime,
            animated_first_frame: animated,
        });
    }

    // 超出预算则转为 JPEG q=85
    let jpg_bytes = encode_jpeg_bytes(&resized, JPEG_QUALITY)?;
    if base64_encoded_len(jpg_bytes.len()) <= max_base64_bytes {
        return Ok(ProcessedImage {
            attachment: ImageAttachment::new("image/jpeg", STANDARD.encode(&jpg_bytes))
                .with_dimensions(target_w, target_h),
            original_width: oriented_w,
            original_height: oriented_h,
            original_media_type: orig_mime,
            animated_first_frame: animated,
        });
    }

    // 仍超则逐档缩小长边重编 JPEG
    let current_longest = target_w.max(target_h);
    let mut last_b64_len = base64_encoded_len(jpg_bytes.len());
    for &cap in EDGE_STEP_DOWNS {
        if cap >= current_longest {
            continue;
        }
        let (sw, sh) = fit_within_edge(oriented_w, oriented_h, cap);
        let stepped = img.resize_exact(sw, sh, FilterType::Lanczos3);
        let step_jpg = encode_jpeg_bytes(&stepped, JPEG_QUALITY)?;
        last_b64_len = base64_encoded_len(step_jpg.len());
        if last_b64_len <= max_base64_bytes {
            return Ok(ProcessedImage {
                attachment: ImageAttachment::new("image/jpeg", STANDARD.encode(&step_jpg))
                    .with_dimensions(sw, sh),
                original_width: oriented_w,
                original_height: oriented_h,
                original_media_type: orig_mime,
                animated_first_frame: animated,
            });
        }
    }

    Err(ImageProcessError::BudgetExceeded {
        base64_len: last_b64_len,
        limit: max_base64_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb, Rgba};

    #[test]
    fn sniffs_only_supported_magic_bytes_within_first_16_bytes() {
        assert_eq!(
            sniff_image_format(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0]),
            Some(ImageFormat::Png)
        );
        assert_eq!(
            sniff_image_format(&[0xFF, 0xD8, 0xFF, 0xE0]),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(sniff_image_format(b"GIF89a......"), Some(ImageFormat::Gif));
        assert_eq!(
            sniff_image_format(b"RIFF\x00\x00\x00\x00WEBP...."),
            Some(ImageFormat::WebP)
        );
        // BMP / SVG / Rust 源码均不识别为支持的图片
        assert_eq!(sniff_image_format(b"BM\x00\x00"), None);
        assert_eq!(sniff_image_format(b"<svg></svg>"), None);
        assert_eq!(sniff_image_format(b"fn main() {}"), None);
    }

    #[test]
    fn small_valid_png_passes_through_without_reencoding() {
        let img =
            DynamicImage::ImageRgba8(ImageBuffer::from_pixel(64, 48, Rgba([12, 34, 56, 255])));
        let raw = encode_png_bytes(&img).unwrap();
        let processed = process_image_bytes(&raw).unwrap();
        assert_eq!(processed.attachment.media_type, "image/png");
        assert_eq!(processed.attachment.width, Some(64));
        assert_eq!(processed.attachment.height, Some(48));
        assert_eq!(processed.attachment.data, STANDARD.encode(&raw));
        assert_eq!(processed.marker_suffix(), None);
    }

    #[test]
    fn oversized_dimensions_preserve_aspect_ratio_within_max_edge() {
        let landscape =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4032, 3024, Rgb([20, 40, 80])));
        let raw_landscape = encode_png_bytes(&landscape).unwrap();
        let out_landscape = process_image_bytes(&raw_landscape).unwrap();
        assert_eq!(out_landscape.attachment.width, Some(2048));
        assert_eq!(out_landscape.attachment.height, Some(1536));
        assert_eq!(
            out_landscape.marker_suffix().as_deref(),
            Some("4032×3024 → 2048×1536")
        );

        let portrait =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1080, 2400, Rgb([90, 50, 10])));
        let raw_portrait = encode_png_bytes(&portrait).unwrap();
        let out_portrait = process_image_bytes(&raw_portrait).unwrap();
        assert_eq!(out_portrait.attachment.width, Some(922));
        assert_eq!(out_portrait.attachment.height, Some(2048));
        assert_eq!(
            out_portrait.marker_suffix().as_deref(),
            Some("1080×2400 → 922×2048")
        );
    }

    #[test]
    fn rejects_over_max_pixels_from_header_without_decoding_idat() {
        // 构造一个仅含合法 IHDR（20000×20000 = 400M 像素 > 100M）但完全没有 IDAT 数据块的最小 PNG。
        // 若先解码再判分辨率则会报 DecodeFailed/Truncated；必须在 header 阶段直接返回 TooManyPixels。
        let mut hdr_only = Vec::new();
        hdr_only.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&20_000u32.to_be_bytes());
        ihdr.extend_from_slice(&20_000u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // bit depth 8, RGB, deflate/filter/interlace
        hdr_only.extend_from_slice(&(ihdr.len() as u32).to_be_bytes());
        hdr_only.extend_from_slice(b"IHDR");
        hdr_only.extend_from_slice(&ihdr);
        let crc = crc32_ieee([b"IHDR".as_slice(), ihdr.as_slice()].concat().as_slice());
        hdr_only.extend_from_slice(&crc.to_be_bytes());

        let err = process_image_bytes(&hdr_only).expect_err("must reject >100M pixel header");
        assert!(
            matches!(
                err,
                ImageProcessError::TooManyPixels {
                    width: 20_000,
                    height: 20_000,
                    pixels: 400_000_000,
                    limit: MAX_IMAGE_PIXELS,
                }
            ),
            "expected TooManyPixels before decoding pixel body, got: {err:?}"
        );
    }

    #[test]
    fn high_entropy_png_exceeding_budget_falls_back_to_jpeg_within_limit() {
        // 构造 2048×2048 高熵伪随机噪声图像：PNG 无法压缩（~12.6MB > 5MB base64），必须回退 JPEG q=85。
        let mut buf = RgbImage::new(2048, 2048);
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for px in buf.pixels_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (state >> 16) as u8;
            let g = (state >> 32) as u8;
            let b = (state >> 48) as u8;
            *px = Rgb([r, g, b]);
        }
        let png_raw = encode_png_bytes(&DynamicImage::ImageRgb8(buf)).unwrap();
        assert!(
            base64_encoded_len(png_raw.len()) > MAX_IMAGE_BASE64_BYTES,
            "fixture PNG must exceed 5MB base64 budget"
        );

        let processed = process_image_bytes(&png_raw).unwrap();
        assert_eq!(processed.attachment.media_type, "image/jpeg");
        assert!(processed.attachment.data.len() <= MAX_IMAGE_BASE64_BYTES);
        assert!(
            processed
                .marker_suffix()
                .as_deref()
                .is_some_and(|s| s.contains("image/png → image/jpeg")),
            "marker suffix: {:?}",
            processed.marker_suffix()
        );
    }

    #[test]
    fn applies_exif_orientation_before_scaling_to_preserve_visual_aspect() {
        // 构造 4000×2000 编码分辨率、但带有 EXIF Orientation=6 (Rotate90) 的 JPEG：
        // 实际视觉方向为 2000×4000（竖图），校正方向后缩放应变成 1024×2048，而非 2048×1024。
        let raw_img =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4000, 2000, Rgb([40, 80, 120])));
        let jpeg_base = encode_jpeg_bytes(&raw_img, 90).unwrap();
        let exif_jpeg = inject_exif_orientation_app1(&jpeg_base, 6);

        let processed = process_image_bytes(&exif_jpeg).unwrap();
        assert_eq!(processed.attachment.width, Some(1024));
        assert_eq!(processed.attachment.height, Some(2048));
    }

    #[test]
    fn animated_gif_keeps_only_first_frame_and_marks_suffix() {
        let mut gif_bytes = Vec::new();
        {
            let mut encoder = image::codecs::gif::GifEncoder::new(&mut gif_bytes);
            let f1 = image::Frame::new(ImageBuffer::from_pixel(32, 24, Rgba([255, 0, 0, 255])));
            let f2 = image::Frame::new(ImageBuffer::from_pixel(32, 24, Rgba([0, 255, 0, 255])));
            encoder.encode_frames([f1, f2]).unwrap();
        }
        let processed = process_image_bytes(&gif_bytes).unwrap();
        assert!(processed.animated_first_frame);
        assert_eq!(processed.attachment.media_type, "image/png");
        assert_eq!(processed.attachment.width, Some(32));
        assert_eq!(processed.attachment.height, Some(24));
        assert_eq!(
            processed.marker_suffix().as_deref(),
            Some("image/gif → image/png · first frame")
        );
        // 校验输出内容确实是首帧红色而非第二帧绿色
        let decoded_bytes = STANDARD.decode(&processed.attachment.data).unwrap();
        let decoded = image::load_from_memory(&decoded_bytes).unwrap().to_rgba8();
        assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0, 255]);
    }

    #[test]
    fn synthetic_code_screenshot_stays_crisp_lossless_png_under_budget() {
        // 验证 Issue #23 重点关切：2560×1440 代码/终端截图缩放至 2048×1152 后是否能稳定留在 PNG 无损路径。
        // 模拟深色终端背景 + 等宽字符条带与高亮行（低噪声、多纯色块/横竖笔画）。
        let mut img = RgbImage::from_pixel(2560, 1440, Rgb([24, 26, 31]));
        for row in 0..48u32 {
            let y0 = row * 30;
            // 状态栏与高亮块
            let color = if row == 0 || row == 47 {
                Rgb([45, 52, 68])
            } else {
                Rgb([180, 200, 220])
            };
            for col in 0..((row * 17) % 120 + 20) {
                let x0 = 40 + col * 14;
                if x0 + 8 < 2520 && y0 + 16 < 1440 {
                    for dy in 4..14 {
                        for dx in 0..8 {
                            if (dx + dy + row) % 3 == 0 {
                                img.put_pixel(x0 + dx, y0 + dy, color);
                            }
                        }
                    }
                }
            }
        }
        let raw = encode_png_bytes(&DynamicImage::ImageRgb8(img)).unwrap();
        let processed = process_image_bytes(&raw).unwrap();
        assert_eq!(processed.attachment.media_type, "image/png");
        assert_eq!(processed.attachment.width, Some(2048));
        assert_eq!(processed.attachment.height, Some(1152));
        assert!(processed.attachment.data.len() <= MAX_IMAGE_BASE64_BYTES);
    }

    #[test]
    fn progressive_edge_step_down_shrinks_image_when_jpeg_exceeds_budget() {
        let mut buf = RgbImage::new(2048, 2048);
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        for px in buf.pixels_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *px = Rgb([
                (state >> 16) as u8,
                (state >> 32) as u8,
                (state >> 48) as u8,
            ]);
        }
        let png_raw = encode_png_bytes(&DynamicImage::ImageRgb8(buf)).unwrap();
        // 使用更紧预算迫使首次 JPEG q=85（2048×2048）超限，进而触发逐档降长边重试
        let budget = 250_000usize;
        let processed = process_image_bytes_with_budget(&png_raw, budget).unwrap();
        assert_eq!(processed.attachment.media_type, "image/jpeg");
        assert!(processed.attachment.width.unwrap() < 2048);
        assert_eq!(processed.attachment.width, processed.attachment.height);
        assert!(processed.attachment.data.len() <= budget);
    }

    fn inject_exif_orientation_app1(jpeg: &[u8], orientation: u16) -> Vec<u8> {
        assert!(jpeg.starts_with(&[0xFF, 0xD8]));
        // 标准 Big-Endian TIFF + IFD0 含单个 Orientation (0x0112) 标签
        let mut app1_payload = Vec::new();
        app1_payload.extend_from_slice(b"Exif\0\0");
        app1_payload.extend_from_slice(b"MM"); // Big-endian
        app1_payload.extend_from_slice(&42u16.to_be_bytes());
        app1_payload.extend_from_slice(&8u32.to_be_bytes()); // IFD0 offset
        app1_payload.extend_from_slice(&1u16.to_be_bytes()); // 1 entry
        app1_payload.extend_from_slice(&0x0112u16.to_be_bytes()); // Tag: Orientation
        app1_payload.extend_from_slice(&3u16.to_be_bytes()); // Type: SHORT
        app1_payload.extend_from_slice(&1u32.to_be_bytes()); // Count: 1
        app1_payload.extend_from_slice(&orientation.to_be_bytes());
        app1_payload.extend_from_slice(&0u16.to_be_bytes()); // padding
        app1_payload.extend_from_slice(&0u32.to_be_bytes()); // next IFD offset

        let seg_len = (app1_payload.len() + 2) as u16;
        let mut out = Vec::with_capacity(jpeg.len() + app1_payload.len() + 4);
        out.extend_from_slice(&jpeg[..2]);
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&seg_len.to_be_bytes());
        out.extend_from_slice(&app1_payload);
        out.extend_from_slice(&jpeg[2..]);
        out
    }
}
