//! 客户端附件与工具回执共用同一套校验和压缩规则，以符合模型的图片输入限制。

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
use mizar_llm_types::ImageAttachment;
use thiserror::Error;

pub const MAX_IMAGE_EDGE: u32 = 2048;
/// 视觉模型按编码后的体积限制请求。
pub const MAX_IMAGE_BASE64_BYTES: usize = 5_000_000;
pub const MAX_IMAGE_PIXELS: u64 = 100_000_000;
pub const MAX_IMAGE_INPUT_BYTES: usize = 32 * 1024 * 1024;
pub const JPEG_QUALITY: u8 = 85;
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

/// 文件扩展名与服务端声明的格式均可能与实际内容不符。
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

/// 解码前校验像素数，避免小体积压缩文件触发过大的内存分配。
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
            original_width: orig_w,
            original_height: orig_h,
            original_media_type: orig_mime,
            animated_first_frame: animated,
        });
    }

    let jpg_bytes = encode_jpeg_bytes(&resized, JPEG_QUALITY)?;
    if base64_encoded_len(jpg_bytes.len()) <= max_base64_bytes {
        return Ok(ProcessedImage {
            attachment: ImageAttachment::new("image/jpeg", STANDARD.encode(&jpg_bytes))
                .with_dimensions(target_w, target_h),
            original_width: orig_w,
            original_height: orig_h,
            original_media_type: orig_mime,
            animated_first_frame: animated,
        });
    }

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
                original_width: orig_w,
                original_height: orig_h,
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
