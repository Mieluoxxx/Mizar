//! Oklab / OKHSL 感知色彩引擎与终端自适应主题系统（融合 `pi` 对比度衍生与 `dsh-TUI` 雾蓝/樱粉语义色板）。

use ratatui::style::Color;

/// 线性 sRGB <-> Oklab <-> OKHSL 色彩数学（基于 Björn Ottosson 参考实现）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb8 {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub const fn to_color(self) -> Color {
        Color::Rgb(self.r, self.g, self.b)
    }

    /// 感知亮度（ITU-R BT.601，与 `dsh-TUI` 一致以 140 为亮/暗分界）。
    pub fn luma(self) -> f64 {
        0.299 * f64::from(self.r) + 0.587 * f64::from(self.g) + 0.114 * f64::from(self.b)
    }

    pub fn is_light(self) -> bool {
        self.luma() > 140.0
    }
}

/// 终端探测得到的底层配色信息（`OSC 10` / `OSC 11` / `OSC 4`）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TerminalColors {
    pub foreground: Option<Rgb8>,
    pub background: Option<Rgb8>,
    pub palette: Option<[Rgb8; 16]>,
}

const K1: f64 = 0.206;
const K2: f64 = 0.03;
const K3: f64 = (1.0 + K1) / (1.0 + K2);

#[inline]
fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(v: f64) -> f64 {
    if v > 0.003_130_8 {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * v
    }
}

#[inline]
fn okhsl_to_oklab_lightness(x: f64) -> f64 {
    (x * x + K1 * x) / (K3 * (x + K2))
}

#[inline]
fn oklab_to_okhsl_lightness(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    0.5 * (K3 * x - K1 + ((K3 * x - K1).powi(2) + 4.0 * K2 * K3 * x).sqrt())
}

const LINEAR_SRGB_TO_LMS: [[f64; 3]; 3] = [
    [0.4122214694707629, 0.5363325372617349, 0.0514459932675022],
    [0.2119034958178251, 0.6806995506452344, 0.1073969535369405],
    [0.0883024591900564, 0.2817188391361215, 0.6299787016738222],
];

const LMS_TO_LAB: [[f64; 3]; 3] = [
    [0.210454268309314, 0.793617774702305, -0.0040720430116193],
    [1.9779985324311684, -2.42859224204858, 0.450593709617411],
    [0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
];

const LAB_TO_LMS: [[f64; 3]; 3] = [
    [1.0, 0.3963377773761749, 0.2158037573099136],
    [1.0, -0.1055613458156586, -0.0638541728258133],
    [1.0, -0.0894841775298119, -1.2914855480194092],
];

const LMS_TO_LINEAR_SRGB: [[f64; 3]; 3] = [
    [4.076_741_636_075_958, -3.307_711_539_258_063, 0.2309699031821043],
    [-1.2684379732850315, 2.609_757_349_287_688, -0.341319376002657],
    [-0.0041960761386756, -0.7034186179359362, 1.7076146940746117],
];

const SATURATION_FIT: [([f64; 2], [f64; 5]); 3] = [
    (
        [-1.8817031, -0.80936501],
        [1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245],
    ),
    (
        [1.8144408, -1.19445267],
        [0.73956515, -0.45954404, 0.08285427, 0.12541073, -0.14503204],
    ),
    (
        [0.13110758, 1.81333971],
        [1.35733652, -0.00915799, -1.1513021, -0.50559606, 0.00692167],
    ),
];

#[inline]
fn mul_mat(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn oklab_to_linear_srgb(lab: [f64; 3]) -> [f64; 3] {
    let lms_cbrt = mul_mat(&LAB_TO_LMS, lab);
    let lms = [lms_cbrt[0].powi(3), lms_cbrt[1].powi(3), lms_cbrt[2].powi(3)];
    mul_mat(&LMS_TO_LINEAR_SRGB, lms)
}

fn linear_srgb_to_oklab(rgb: [f64; 3]) -> [f64; 3] {
    let lms = mul_mat(&LINEAR_SRGB_TO_LMS, rgb);
    let lms_cbrt = [lms[0].cbrt(), lms[1].cbrt(), lms[2].cbrt()];
    mul_mat(&LMS_TO_LAB, lms_cbrt)
}

pub fn rgb_to_oklab(rgb: Rgb8) -> [f64; 3] {
    linear_srgb_to_oklab([
        srgb_to_linear(f64::from(rgb.r) / 255.0),
        srgb_to_linear(f64::from(rgb.g) / 255.0),
        srgb_to_linear(f64::from(rgb.b) / 255.0),
    ])
}

fn linear_srgb_to_rgb8(linear: [f64; 3]) -> Rgb8 {
    let ch = |v: f64| -> u8 { (linear_to_srgb(v).clamp(0.0, 1.0) * 255.0).round() as u8 };
    Rgb8::new(ch(linear[0]), ch(linear[1]), ch(linear[2]))
}

fn max_saturation(a: f64, b: f64) -> f64 {
    let channel = if SATURATION_FIT[0].0[0] * a + SATURATION_FIT[0].0[1] * b > 1.0 {
        0
    } else if SATURATION_FIT[1].0[0] * a + SATURATION_FIT[1].0[1] * b > 1.0 {
        1
    } else {
        2
    };
    let [k0, k1, k2, k3, k4] = SATURATION_FIT[channel].1;
    let weights = LMS_TO_LINEAR_SRGB[channel];
    let sat = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;
    let slopes = [
        LAB_TO_LMS[0][1] * a + LAB_TO_LMS[0][2] * b,
        LAB_TO_LMS[1][1] * a + LAB_TO_LMS[1][2] * b,
        LAB_TO_LMS[2][1] * a + LAB_TO_LMS[2][2] * b,
    ];
    let base = [
        1.0 + sat * slopes[0],
        1.0 + sat * slopes[1],
        1.0 + sat * slopes[2],
    ];
    let f = weights[0] * base[0].powi(3)
        + weights[1] * base[1].powi(3)
        + weights[2] * base[2].powi(3);
    let f1 = weights[0] * 3.0 * slopes[0] * base[0].powi(2)
        + weights[1] * 3.0 * slopes[1] * base[1].powi(2)
        + weights[2] * 3.0 * slopes[2] * base[2].powi(2);
    let f2 = weights[0] * 6.0 * slopes[0].powi(2) * base[0]
        + weights[1] * 6.0 * slopes[1].powi(2) * base[1]
        + weights[2] * 6.0 * slopes[2].powi(2) * base[2];
    sat - (f * f1) / (f1 * f1 - 0.5 * f * f2)
}

fn cusp(a: f64, b: f64) -> (f64, f64) {
    let sat = max_saturation(a, b);
    let lin = oklab_to_linear_srgb([1.0, sat * a, sat * b]);
    let m = lin[0].max(lin[1]).max(lin[2]).max(1e-9);
    let l = (1.0 / m).cbrt();
    (l, l * sat)
}

fn max_chroma(a: f64, b: f64, l: f64, (cusp_l, cusp_c): (f64, f64)) -> f64 {
    if l <= cusp_l {
        if cusp_l <= 1e-9 {
            return 0.0;
        }
        return (cusp_c * l) / cusp_l;
    }
    let t = (cusp_c * (l - 1.0)) / (cusp_l - 1.0).min(-1e-9);
    let slopes = [
        LAB_TO_LMS[0][1] * a + LAB_TO_LMS[0][2] * b,
        LAB_TO_LMS[1][1] * a + LAB_TO_LMS[1][2] * b,
        LAB_TO_LMS[2][1] * a + LAB_TO_LMS[2][2] * b,
    ];
    let lms = [l + t * slopes[0], l + t * slopes[1], l + t * slopes[2]];
    let cubes = [lms[0].powi(3), lms[1].powi(3), lms[2].powi(3)];
    let first = [
        3.0 * slopes[0] * lms[0].powi(2),
        3.0 * slopes[1] * lms[1].powi(2),
        3.0 * slopes[2] * lms[2].powi(2),
    ];
    let second = [
        6.0 * slopes[0].powi(2) * lms[0],
        6.0 * slopes[1].powi(2) * lms[1],
        6.0 * slopes[2].powi(2) * lms[2],
    ];
    let mut best = f64::MAX;
    for row in LMS_TO_LINEAR_SRGB {
        let f = row[0] * cubes[0] + row[1] * cubes[1] + row[2] * cubes[2] - 1.0;
        let f1 = row[0] * first[0] + row[1] * first[1] + row[2] * first[2];
        let f2 = row[0] * second[0] + row[1] * second[1] + row[2] * second[2];
        let denom = f1 * f1 - 0.5 * f * f2;
        if denom.abs() > 1e-12 {
            let u = f1 / denom;
            if u >= 0.0 {
                best = best.min(-f * u);
            }
        }
    }
    (t + best).max(0.0)
}

fn chroma_stops(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let peak = cusp(a, b);
    let c_max = max_chroma(a, b, l, peak);
    let denom = (l * (peak.1 / peak.0.max(1e-9)))
        .min((1.0 - l) * (peak.1 / (1.0 - peak.0).max(1e-9)))
        .max(1e-9);
    let k = c_max / denom;
    let mid_s = 0.11516993
        + 1.0
            / (7.4477897
                + 4.1590124 * b
                + a * (-2.19557347
                    + 1.75198401 * b
                    + a * (-2.13704948
                        - 10.02301043 * b
                        + a * (-4.24894561 + 5.38770819 * b + 4.69891013 * a))));
    let mid_t = 0.11239642
        + 1.0
            / (1.6132032 - 0.68124379 * b
                + a * (0.40370612
                    + 0.90148123 * b
                    + a * (-0.27087943
                        + 0.6122399 * b
                        + a * (0.00299215 - 0.45399568 * b - 0.14661872 * a))));
    let c_mid = 0.9
        * k
        * (1.0 / ((1.0 / (l * mid_s).powi(4)) + (1.0 / ((1.0 - l) * mid_t).powi(4))))
            .sqrt()
            .sqrt();
    let c0 = (1.0 / ((1.0 / (l * 0.4).powi(2)) + (1.0 / ((1.0 - l) * 0.8).powi(2)))).sqrt();
    (c0, c_mid, c_max)
}

/// 将 OKHSL (`hue` 0..360, `sat` 0..1, `light` 0..1) 转换为 sRGB8，保证色域安全且感知饱和度一致。
pub fn okhsl_to_rgb(hue_deg: f64, saturation: f64, lightness: f64) -> Rgb8 {
    let sat = saturation.clamp(0.0, 1.0);
    let light = lightness.clamp(0.0, 1.0);
    let l = okhsl_to_oklab_lightness(light);
    if l <= 0.0 || l >= 1.0 || sat <= 0.0 {
        return linear_srgb_to_rgb8(oklab_to_linear_srgb([l, 0.0, 0.0]));
    }
    let angle = (2.0 * std::f64::consts::PI * hue_deg.rem_euclid(360.0)) / 360.0;
    let a = angle.cos();
    let b = angle.sin();
    let (c0, c_mid, c_max) = chroma_stops(l, a, b);
    let chroma = if sat < 0.8 {
        let t = 1.25 * sat;
        let k1 = 0.8 * c0.max(1e-9);
        (t * k1) / (1.0 - (1.0 - k1 / c_mid.max(1e-9)) * t).max(1e-9)
    } else {
        let t = 5.0 * (sat - 0.8);
        let k1 = (0.2 * c_mid * c_mid * 1.25 * 1.25) / c0.max(1e-9);
        c_mid
            + (t * k1)
                / (1.0 - (1.0 - k1 / (c_max - c_mid).max(1e-9)) * t).max(1e-9)
    };
    linear_srgb_to_rgb8(oklab_to_linear_srgb([l, chroma * a, chroma * b]))
}

/// 从 sRGB8 提取 OKHSL `(h, s, l)`。
pub fn rgb_to_okhsl(rgb: Rgb8) -> (f64, f64, f64) {
    let [l, lab_a, lab_b] = rgb_to_oklab(rgb);
    let chroma = lab_a.hypot(lab_b);
    let lightness = oklab_to_okhsl_lightness(l);
    if chroma < 1e-9 || lightness <= 0.0 || lightness >= 1.0 {
        return (0.0, 0.0, lightness);
    }
    let hue = (lab_b.atan2(lab_a).to_degrees() + 360.0) % 360.0;
    let (c0, c_mid, c_max) = chroma_stops(l, lab_a / chroma, lab_b / chroma);
    let sat = if chroma < c_mid {
        let k1 = 0.8 * c0.max(1e-9);
        0.8 * (chroma / (k1 + (1.0 - k1 / c_mid.max(1e-9)) * chroma).max(1e-9))
    } else {
        let k1 = (0.2 * c_mid * c_mid * 1.25 * 1.25) / c0.max(1e-9);
        let offset = chroma - c_mid;
        0.8 + 0.2 * (offset / (k1 + (1.0 - k1 / (c_max - c_mid).max(1e-9)) * offset).max(1e-9))
    };
    (hue, sat.clamp(0.0, 1.0), lightness)
}

/// 在 Oklab 感知均匀空间中混合两种颜色（`amount = 0.0` 返回 `a`，`1.0` 返回 `b`）。
pub fn mix_oklab(a: Rgb8, b: Rgb8, amount: f64) -> Rgb8 {
    let t = amount.clamp(0.0, 1.0);
    let la = rgb_to_oklab(a);
    let lb = rgb_to_oklab(b);
    let lab = [
        la[0] + (lb[0] - la[0]) * t,
        la[1] + (lb[1] - la[1]) * t,
        la[2] + (lb[2] - la[2]) * t,
    ];
    linear_srgb_to_rgb8(oklab_to_linear_srgb(lab))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemePreset {
    Auto,
    MistDark,
    MistLight,
    SakuraDark,
    System,
    Ansi16,
}

impl ThemePreset {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "mist-dark" | "dark" => Some(Self::MistDark),
            "mist-light" | "light" => Some(Self::MistLight),
            "sakura-dark" | "sakura" | "pink" => Some(Self::SakuraDark),
            "system" => Some(Self::System),
            "ansi" | "ansi16" => Some(Self::Ansi16),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::MistDark => "mist-dark",
            Self::MistLight => "mist-light",
            Self::SakuraDark => "sakura-dark",
            Self::System => "system",
            Self::Ansi16 => "ansi16",
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::Auto,
            Self::MistDark,
            Self::MistLight,
            Self::SakuraDark,
            Self::System,
            Self::Ansi16,
        ]
    }
}

/// 语义化 TUI 主题色板。
#[derive(Debug, Clone)]
pub struct Theme {
    pub id: &'static str,
    pub is_light: bool,
    pub truecolor: bool,

    // 表面与层级
    pub panel_bg: Color,
    pub card_bg: Color,
    pub card_bg_alt: Color,
    pub user_bg: Color,
    pub selection_bg: Color,
    pub selection_fg: Color,

    // 文字与边框
    pub text: Color,
    pub text_muted: Color,
    pub text_dim: Color,
    pub accent: Color,
    pub accent_soft: Color,
    pub link: Color,
    pub border: Color,
    pub border_focus: Color,
    pub border_busy: Color,
    pub border_plan: Color,

    // 语义状态
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub info: Color,
    pub magenta: Color,

    // Diff 与语法
    pub diff_add_fg: Color,
    pub diff_add_bg: Color,
    pub diff_add_word_bg: Color,
    pub diff_del_fg: Color,
    pub diff_del_bg: Color,
    pub diff_del_word_bg: Color,
    pub code_fg: Color,
    pub code_bg: Color,
    pub syntax_keyword: Color,
    pub syntax_string: Color,
    pub syntax_comment: Color,
    pub syntax_number: Color,
    pub syntax_type: Color,

    // 上下文分段条（System / User / Assistant / Reasoning / Tools / Free）
    pub context_segments: [Color; 5],
    pub context_free: Color,

    // pi-emphasis 状态胶囊色调 (fg, bg)
    pub pill_success: (Color, Color),
    pub pill_error: (Color, Color),
    pub pill_warning: (Color, Color),
    pub pill_danger: (Color, Color),
    pub pill_mode: (Color, Color),
    pub pill_owner: (Color, Color),
    pub pill_neutral: (Color, Color),

    // 7 档思考深度光谱 (Off -> Max)
    pub effort_colors: [Color; 7],
}

/// 粗略探测当前终端是否支持 24-bit TrueColor。
pub fn detect_truecolor_support() -> bool {
    if let Ok(ct) = std::env::var("COLORTERM") {
        let lower = ct.to_ascii_lowercase();
        if lower.contains("truecolor") || lower.contains("24bit") {
            return true;
        }
    }
    if let Ok(term) = std::env::var("TERM") {
        let lower = term.to_ascii_lowercase();
        if lower.contains("256color") || lower.contains("kitty") || lower.contains("ghostty") {
            return true;
        }
        if lower == "dumb" || lower == "linux" {
            return false;
        }
    }
    true
}

/// 从 `COLORFGBG` 环境变量做零阻塞初始亮暗启发（例如 `0;15` 为亮底）。
pub fn env_prefers_light() -> Option<bool> {
    let raw = std::env::var("COLORFGBG").ok()?;
    let bg = raw.rsplit(';').next()?.trim().parse::<u8>().ok()?;
    Some(matches!(bg, 7 | 15))
}

impl Theme {
    pub fn from_preset(preset: ThemePreset, probed: Option<&TerminalColors>) -> Self {
        let truecolor = detect_truecolor_support();
        if !truecolor && matches!(preset, ThemePreset::Auto | ThemePreset::System) {
            return Self::ansi16();
        }
        match preset {
            ThemePreset::Ansi16 => Self::ansi16(),
            ThemePreset::MistDark => Self::mist_dark(),
            ThemePreset::MistLight => Self::mist_light(),
            ThemePreset::SakuraDark => Self::sakura_dark(),
            ThemePreset::Auto => {
                let is_light = probed
                    .and_then(|c| c.background)
                    .map(Rgb8::is_light)
                    .or_else(env_prefers_light)
                    .unwrap_or(false);
                if is_light {
                    Self::mist_light()
                } else {
                    Self::mist_dark()
                }
            }
            ThemePreset::System => Self::from_terminal_colors(probed),
        }
    }

    /// 雾蓝深色（`dsh-TUI` Gentle Mist Blue Dark）。
    pub fn mist_dark() -> Self {
        let bg = Rgb8::new(22, 26, 34);
        let accent = Rgb8::new(125, 161, 222); // #7da1de
        let text = Rgb8::new(232, 236, 242);
        Self {
            id: "mist-dark",
            is_light: false,
            truecolor: true,
            panel_bg: Rgb8::new(28, 33, 43).to_color(),
            card_bg: Rgb8::new(30, 36, 46).to_color(),
            card_bg_alt: Rgb8::new(36, 43, 56).to_color(),
            user_bg: Rgb8::new(38, 46, 61).to_color(),
            selection_bg: Rgb8::new(62, 95, 150).to_color(),
            selection_fg: Rgb8::new(250, 252, 255).to_color(),
            text: text.to_color(),
            text_muted: Rgb8::new(150, 160, 178).to_color(),
            text_dim: Rgb8::new(98, 108, 124).to_color(),
            accent: accent.to_color(),
            accent_soft: mix_oklab(accent, bg, 0.35).to_color(),
            link: Rgb8::new(116, 178, 247).to_color(),
            border: Rgb8::new(62, 72, 92).to_color(),
            border_focus: accent.to_color(),
            border_busy: Rgb8::new(230, 180, 90).to_color(),
            border_plan: Rgb8::new(136, 192, 160).to_color(),
            success: Rgb8::new(126, 211, 148).to_color(),
            warning: Rgb8::new(240, 192, 96).to_color(),
            error: Rgb8::new(242, 118, 130).to_color(),
            info: Rgb8::new(125, 161, 222).to_color(),
            magenta: Rgb8::new(196, 152, 238).to_color(),
            diff_add_fg: Rgb8::new(138, 222, 156).to_color(),
            diff_add_bg: Rgb8::new(26, 48, 35).to_color(),
            diff_add_word_bg: Rgb8::new(45, 92, 62).to_color(),
            diff_del_fg: Rgb8::new(242, 132, 142).to_color(),
            diff_del_bg: Rgb8::new(56, 29, 34).to_color(),
            diff_del_word_bg: Rgb8::new(108, 46, 54).to_color(),
            code_fg: Rgb8::new(165, 214, 255).to_color(),
            code_bg: Rgb8::new(34, 40, 52).to_color(),
            syntax_keyword: Rgb8::new(138, 173, 244).to_color(),
            syntax_string: Rgb8::new(166, 218, 149).to_color(),
            syntax_comment: Rgb8::new(110, 118, 136).to_color(),
            syntax_number: Rgb8::new(245, 169, 127).to_color(),
            syntax_type: Rgb8::new(238, 212, 159).to_color(),
            context_segments: [
                Rgb8::new(136, 152, 178).to_color(),
                Rgb8::new(125, 161, 222).to_color(),
                Rgb8::new(126, 211, 148).to_color(),
                Rgb8::new(196, 152, 238).to_color(),
                Rgb8::new(240, 192, 96).to_color(),
            ],
            context_free: Rgb8::new(42, 48, 60).to_color(),
            pill_success: (Rgb8::new(142, 228, 164).to_color(), Rgb8::new(31, 62, 42).to_color()),
            pill_error: (Rgb8::new(252, 142, 152).to_color(), Rgb8::new(74, 31, 38).to_color()),
            pill_warning: (Rgb8::new(248, 206, 114).to_color(), Rgb8::new(72, 56, 26).to_color()),
            pill_danger: (Rgb8::new(255, 245, 245).to_color(), Rgb8::new(178, 44, 58).to_color()),
            pill_mode: (Rgb8::new(238, 252, 255).to_color(), Rgb8::new(34, 110, 134).to_color()),
            pill_owner: (Rgb8::new(240, 247, 255).to_color(), Rgb8::new(48, 92, 168).to_color()),
            pill_neutral: (Rgb8::new(208, 218, 234).to_color(), Rgb8::new(48, 56, 72).to_color()),
            effort_colors: [
                Rgb8::new(110, 118, 134).to_color(),
                Rgb8::new(128, 156, 196).to_color(),
                Rgb8::new(125, 161, 222).to_color(),
                Rgb8::new(146, 152, 235).to_color(),
                Rgb8::new(186, 144, 238).to_color(),
                Rgb8::new(228, 132, 202).to_color(),
                Rgb8::new(244, 118, 128).to_color(),
            ],
        }
    }

    /// 雾蓝浅色（`dsh-TUI` Gentle Mist Blue Light）。
    pub fn mist_light() -> Self {
        let accent = Rgb8::new(58, 102, 178);
        Self {
            id: "mist-light",
            is_light: true,
            truecolor: true,
            panel_bg: Rgb8::new(245, 247, 251).to_color(),
            card_bg: Rgb8::new(240, 244, 250).to_color(),
            card_bg_alt: Rgb8::new(233, 238, 247).to_color(),
            user_bg: Rgb8::new(228, 236, 249).to_color(),
            selection_bg: Rgb8::new(184, 208, 244).to_color(),
            selection_fg: Rgb8::new(24, 30, 42).to_color(),
            text: Rgb8::new(52, 57, 69).to_color(), // #343945
            text_muted: Rgb8::new(104, 114, 132).to_color(),
            text_dim: Rgb8::new(148, 156, 172).to_color(),
            accent: accent.to_color(),
            accent_soft: Rgb8::new(106, 144, 206).to_color(),
            link: Rgb8::new(36, 98, 188).to_color(),
            border: Rgb8::new(204, 212, 226).to_color(),
            border_focus: accent.to_color(),
            border_busy: Rgb8::new(196, 128, 28).to_color(),
            border_plan: Rgb8::new(52, 138, 96).to_color(),
            success: Rgb8::new(34, 134, 72).to_color(),
            warning: Rgb8::new(184, 118, 14).to_color(),
            error: Rgb8::new(196, 48, 66).to_color(),
            info: accent.to_color(),
            magenta: Rgb8::new(136, 68, 188).to_color(),
            diff_add_fg: Rgb8::new(26, 118, 62).to_color(),
            diff_add_bg: Rgb8::new(230, 247, 236).to_color(),
            diff_add_word_bg: Rgb8::new(188, 232, 202).to_color(),
            diff_del_fg: Rgb8::new(184, 42, 58).to_color(),
            diff_del_bg: Rgb8::new(252, 234, 236).to_color(),
            diff_del_word_bg: Rgb8::new(244, 194, 200).to_color(),
            code_fg: Rgb8::new(40, 88, 160).to_color(),
            code_bg: Rgb8::new(236, 240, 248).to_color(),
            syntax_keyword: Rgb8::new(44, 88, 172).to_color(),
            syntax_string: Rgb8::new(38, 128, 72).to_color(),
            syntax_comment: Rgb8::new(136, 144, 160).to_color(),
            syntax_number: Rgb8::new(186, 92, 24).to_color(),
            syntax_type: Rgb8::new(136, 86, 168).to_color(),
            context_segments: [
                Rgb8::new(112, 126, 148).to_color(),
                Rgb8::new(66, 114, 194).to_color(),
                Rgb8::new(46, 152, 92).to_color(),
                Rgb8::new(146, 82, 198).to_color(),
                Rgb8::new(204, 134, 28).to_color(),
            ],
            context_free: Rgb8::new(224, 229, 238).to_color(),
            pill_success: (Rgb8::new(22, 108, 54).to_color(), Rgb8::new(214, 242, 224).to_color()),
            pill_error: (Rgb8::new(178, 34, 52).to_color(), Rgb8::new(250, 222, 226).to_color()),
            pill_warning: (Rgb8::new(148, 92, 10).to_color(), Rgb8::new(252, 238, 204).to_color()),
            pill_danger: (Rgb8::new(255, 255, 255).to_color(), Rgb8::new(192, 42, 58).to_color()),
            pill_mode: (Rgb8::new(255, 255, 255).to_color(), Rgb8::new(28, 124, 148).to_color()),
            pill_owner: (Rgb8::new(255, 255, 255).to_color(), Rgb8::new(48, 96, 178).to_color()),
            pill_neutral: (Rgb8::new(56, 66, 84).to_color(), Rgb8::new(226, 232, 242).to_color()),
            effort_colors: [
                Rgb8::new(142, 148, 162).to_color(),
                Rgb8::new(88, 124, 168).to_color(),
                Rgb8::new(58, 102, 178).to_color(),
                Rgb8::new(98, 96, 192).to_color(),
                Rgb8::new(142, 72, 194).to_color(),
                Rgb8::new(186, 56, 152).to_color(),
                Rgb8::new(196, 44, 62).to_color(),
            ],
        }
    }

    /// 樱粉深色（`dsh-TUI` Sakura Night）。
    pub fn sakura_dark() -> Self {
        let mut t = Self::mist_dark();
        t.id = "sakura-dark";
        let accent = Rgb8::new(236, 156, 184);
        t.accent = accent.to_color();
        t.accent_soft = Rgb8::new(182, 120, 146).to_color();
        t.border_focus = accent.to_color();
        t.link = Rgb8::new(244, 174, 198).to_color();
        t.user_bg = Rgb8::new(46, 36, 44).to_color();
        t.card_bg = Rgb8::new(34, 29, 36).to_color();
        t.card_bg_alt = Rgb8::new(42, 36, 44).to_color();
        t.selection_bg = Rgb8::new(138, 78, 102).to_color();
        t
    }

    /// 基于终端自身背景色和 16 色板在 OKHSL 空间动态合成对比度均衡的主题（借鉴 `pi`）。
    pub fn from_terminal_colors(probed: Option<&TerminalColors>) -> Self {
        let Some(colors) = probed else {
            return if env_prefers_light().unwrap_or(false) {
                Self::mist_light()
            } else {
                Self::mist_dark()
            };
        };

        let Some(bg) = colors.background else {
            return Self::ansi16();
        };

        let is_light = bg.is_light();
        let (_, _, l_bg) = rgb_to_okhsl(bg);

        // 从终端调色板抽取色相与饱和度（如果终端未返回色板则用中性基准色相）
        let hue_sat = |slot: usize, default_h: f64, default_s: f64| -> (f64, f64) {
            if let Some(pal) = colors.palette {
                let (h, s, _) = rgb_to_okhsl(pal[slot]);
                if s > 0.08 {
                    return (h, s.clamp(0.25, 0.85));
                }
            }
            (default_h, default_s)
        };

        let target_fg_l = if is_light {
            (l_bg - 0.62).clamp(0.14, 0.34)
        } else {
            (l_bg + 0.68).clamp(0.76, 0.94)
        };
        let target_accent_l = if is_light {
            (l_bg - 0.42).clamp(0.28, 0.50)
        } else {
            (l_bg + 0.52).clamp(0.64, 0.82)
        };

        let (h_blue, s_blue) = hue_sat(4, 232.0, 0.68);
        let (h_green, s_green) = hue_sat(2, 158.0, 0.72);
        let (h_red, s_red) = hue_sat(1, 20.0, 0.84);
        let (h_yellow, s_yellow) = hue_sat(3, 82.0, 0.85);
        let (h_magenta, s_magenta) = hue_sat(5, 295.0, 0.62);
        let (h_cyan, s_cyan) = hue_sat(6, 202.0, 0.72);

        let accent = okhsl_to_rgb(h_blue, s_blue, target_accent_l);
        let success = okhsl_to_rgb(h_green, s_green, target_accent_l);
        let error = okhsl_to_rgb(h_red, s_red, target_accent_l);
        let warning = okhsl_to_rgb(h_yellow, s_yellow, target_accent_l);
        let magenta = okhsl_to_rgb(h_magenta, s_magenta, target_accent_l);
        let cyan = okhsl_to_rgb(h_cyan, s_cyan, target_accent_l);

        let text = colors
            .foreground
            .unwrap_or_else(|| okhsl_to_rgb(h_blue, 0.06, target_fg_l));
        let text_muted = mix_oklab(text, bg, 0.38);
        let text_dim = mix_oklab(text, bg, 0.58);
        let panel_bg = mix_oklab(bg, text, if is_light { 0.04 } else { 0.05 });
        let card_bg = mix_oklab(bg, text, if is_light { 0.07 } else { 0.08 });
        let card_bg_alt = mix_oklab(bg, text, if is_light { 0.11 } else { 0.12 });
        let user_bg = mix_oklab(bg, accent, if is_light { 0.14 } else { 0.18 });
        let border = mix_oklab(bg, text, if is_light { 0.22 } else { 0.24 });

        let mut base = if is_light {
            Self::mist_light()
        } else {
            Self::mist_dark()
        };
        base.id = "system";
        base.is_light = is_light;
        base.panel_bg = panel_bg.to_color();
        base.card_bg = card_bg.to_color();
        base.card_bg_alt = card_bg_alt.to_color();
        base.user_bg = user_bg.to_color();
        base.selection_bg = mix_oklab(bg, accent, 0.38).to_color();
        base.selection_fg = text.to_color();
        base.text = text.to_color();
        base.text_muted = text_muted.to_color();
        base.text_dim = text_dim.to_color();
        base.accent = accent.to_color();
        base.accent_soft = mix_oklab(accent, bg, 0.35).to_color();
        base.link = cyan.to_color();
        base.border = border.to_color();
        base.border_focus = accent.to_color();
        base.border_busy = warning.to_color();
        base.border_plan = success.to_color();
        base.success = success.to_color();
        base.warning = warning.to_color();
        base.error = error.to_color();
        base.info = accent.to_color();
        base.magenta = magenta.to_color();
        base.diff_add_fg = success.to_color();
        base.diff_add_bg = mix_oklab(bg, success, 0.14).to_color();
        base.diff_add_word_bg = mix_oklab(bg, success, 0.32).to_color();
        base.diff_del_fg = error.to_color();
        base.diff_del_bg = mix_oklab(bg, error, 0.14).to_color();
        base.diff_del_word_bg = mix_oklab(bg, error, 0.32).to_color();
        base.pill_success = (success.to_color(), mix_oklab(bg, success, 0.20).to_color());
        base.pill_error = (error.to_color(), mix_oklab(bg, error, 0.20).to_color());
        base.pill_warning = (warning.to_color(), mix_oklab(bg, warning, 0.20).to_color());
        base.pill_neutral = (text.to_color(), card_bg_alt.to_color());
        base
    }

    /// 16 色降级调色板（兼容无 TrueColor 的终端/测试环境）。
    pub fn ansi16() -> Self {
        Self {
            id: "ansi16",
            is_light: false,
            truecolor: false,
            panel_bg: Color::Reset,
            card_bg: Color::Reset,
            card_bg_alt: Color::Reset,
            user_bg: Color::Reset,
            selection_bg: Color::Blue,
            selection_fg: Color::White,
            text: Color::Reset,
            text_muted: Color::Gray,
            text_dim: Color::DarkGray,
            accent: Color::Cyan,
            accent_soft: Color::Blue,
            link: Color::LightBlue,
            border: Color::DarkGray,
            border_focus: Color::Cyan,
            border_busy: Color::Yellow,
            border_plan: Color::Green,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            info: Color::Cyan,
            magenta: Color::Magenta,
            diff_add_fg: Color::Green,
            diff_add_bg: Color::Reset,
            diff_add_word_bg: Color::Green,
            diff_del_fg: Color::Red,
            diff_del_bg: Color::Reset,
            diff_del_word_bg: Color::Red,
            code_fg: Color::Cyan,
            code_bg: Color::Reset,
            syntax_keyword: Color::Blue,
            syntax_string: Color::Green,
            syntax_comment: Color::DarkGray,
            syntax_number: Color::Yellow,
            syntax_type: Color::Magenta,
            context_segments: [
                Color::DarkGray,
                Color::Blue,
                Color::Green,
                Color::Magenta,
                Color::Yellow,
            ],
            context_free: Color::DarkGray,
            pill_success: (Color::Black, Color::Green),
            pill_error: (Color::White, Color::Red),
            pill_warning: (Color::Black, Color::Yellow),
            pill_danger: (Color::White, Color::LightRed),
            pill_mode: (Color::Black, Color::Cyan),
            pill_owner: (Color::White, Color::Blue),
            pill_neutral: (Color::White, Color::DarkGray),
            effort_colors: [
                Color::DarkGray,
                Color::Gray,
                Color::Cyan,
                Color::Blue,
                Color::Magenta,
                Color::LightMagenta,
                Color::Red,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn okhsl_roundtrip_preserves_perceptual_hue_and_lightness() {
        let rgb = okhsl_to_rgb(220.0, 0.70, 0.60);
        let (h, s, l) = rgb_to_okhsl(rgb);
        assert!((h - 220.0).abs() < 2.0, "h={h}");
        assert!((s - 0.70).abs() < 0.03, "s={s}");
        assert!((l - 0.60).abs() < 0.02, "l={l}");
    }

    #[test]
    fn system_theme_adapts_to_light_and_dark_terminal_backgrounds() {
        let dark_probe = TerminalColors {
            background: Some(Rgb8::new(18, 20, 26)),
            foreground: Some(Rgb8::new(230, 234, 240)),
            palette: None,
        };
        let dark = Theme::from_preset(ThemePreset::System, Some(&dark_probe));
        assert!(!dark.is_light);
        assert_eq!(dark.id, "system");

        let light_probe = TerminalColors {
            background: Some(Rgb8::new(250, 249, 245)),
            foreground: Some(Rgb8::new(36, 40, 48)),
            palette: None,
        };
        let light = Theme::from_preset(ThemePreset::System, Some(&light_probe));
        assert!(light.is_light);
        assert_eq!(light.id, "system");
    }
}
