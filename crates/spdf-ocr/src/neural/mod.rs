//! Built-in neural OCR engine: PaddleOCR PP-OCRv5 mobile text detection
//! (DBNet) + recognition (SVTR-LCNet, CTC), compiled to native Burn code at
//! build time by `burn-onnx` and run on the GPU through wgpu (Vulkan / DX12 /
//! Metal), falling back to Burn's portable CPU backend when no adapter is
//! available.
//!
//! Pipeline per page image:
//! 1. detect text-line regions (probability map → rotated rectangles),
//! 2. estimate the page's reading direction (0/90/180/270° plus skew) from
//!    the line geometry and a recognition-confidence vote,
//! 3. crop every line along its own rotated rectangle and recognise it,
//! 4. split lines into words and report their boxes in the *upright* page
//!    frame, so rotated and skewed scans feed the layout projection like
//!    straight text.
//!
//! Environment:
//! - `SPDF_OCR_DEVICE=cpu|wgpu|cuda` forces a backend (`cuda` needs the
//!   `neural-cuda` feature).

pub mod ctc;
pub mod geom;

use std::sync::{Mutex, OnceLock};

use spdf_types::{SpdfError, SpdfResult};

use crate::engine::{OcrEngine, OcrOptions, OcrResult};
use ctc::Charset;
use geom::{DbParams, DetBox, Rgb, RotRect};

#[cfg(spdf_neural_models)]
#[allow(clippy::all, unused)]
mod det_model {
    include!(concat!(env!("OUT_DIR"), "/model/ch_PP-OCRv5_det_mobile.rs"));
}
#[cfg(spdf_neural_models)]
#[allow(clippy::all, unused)]
mod rec_model {
    include!(concat!(env!("OUT_DIR"), "/model/ch_PP-OCRv5_rec_mobile.rs"));
}

/// Recogniser input height for PP-OCRv5.
const REC_H: usize = 48;
/// Recogniser width downsampling (input pixels per CTC timestep).
const REC_STRIDE: usize = 8;
/// Widest line crop fed to the recogniser.
const REC_MAX_W: usize = 3200;
/// Crops per recogniser batch.
const REC_BATCH: usize = 8;
/// Pad crop widths up to a multiple of this, so GPU kernels see few shapes.
const REC_W_BUCKET: usize = 64;
/// Longest side fed to the detector.
const DET_MAX_SIDE: usize = 2560;

/// Built-in neural OCR engine. Cheap to construct: the models and the GPU
/// device are initialised lazily on the first page that needs OCR.
pub struct NeuralOcrEngine {
    runtime: OnceLock<Result<Runtime, String>>,
}

impl Default for NeuralOcrEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralOcrEngine {
    pub fn new() -> Self {
        Self {
            runtime: OnceLock::new(),
        }
    }

    /// True when this build embeds the OCR models.
    pub fn models_available() -> bool {
        cfg!(spdf_neural_models)
    }

    fn runtime(&self) -> SpdfResult<&Runtime> {
        self.runtime
            .get_or_init(Runtime::init)
            .as_ref()
            .map_err(|e| SpdfError::Ocr(e.clone()))
    }

    /// Name of the compute backend in use (`wgpu`, `cuda`, `cpu`), after
    /// initialising it if needed.
    pub fn backend(&self) -> SpdfResult<&'static str> {
        Ok(self.runtime()?.backend)
    }
}

impl OcrEngine for NeuralOcrEngine {
    fn name(&self) -> &'static str {
        "neural"
    }

    fn recognize(&self, image: &[u8], _options: &OcrOptions) -> SpdfResult<Vec<OcrResult>> {
        let rt = self.runtime()?;
        let img = image::load_from_memory(image)
            .map_err(|e| SpdfError::Ocr(format!("decode page image: {e}")))?
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = Rgb {
            data: img.as_raw(),
            w,
            h,
        };
        let t0 = std::time::Instant::now();
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.ocr_page(&rgb)))
            .map_err(|_| SpdfError::Ocr("neural OCR backend panicked".into()))??;
        tracing::debug!(
            backend = rt.backend,
            ms = t0.elapsed().as_millis() as u64,
            words = out.len(),
            "spdf: neural OCR page"
        );
        Ok(out)
    }
}

/// A recognised line crop.
struct LineRec {
    chars: Vec<ctc::DecodedChar>,
    content_steps: usize,
}

struct Runtime {
    backend: &'static str,
    charset: Charset,
    #[cfg(spdf_neural_models)]
    models: Mutex<Models>,
}

#[cfg(spdf_neural_models)]
struct Models {
    device: burn::prelude::Device,
    det: det_model::Model,
    rec: rec_model::Model,
}

impl Runtime {
    #[cfg(not(spdf_neural_models))]
    fn init() -> Result<Self, String> {
        let _ = Mutex::new(());
        Err("neural OCR models were not available when spdf was built".into())
    }

    #[cfg(spdf_neural_models)]
    fn init() -> Result<Self, String> {
        use burn::tensor::Bytes;
        const DET: &[u8] = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/model/ch_PP-OCRv5_det_mobile.bpk"
        ));
        const REC: &[u8] = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/model/ch_PP-OCRv5_rec_mobile.bpk"
        ));
        const DICT: &str = include_str!(concat!(env!("OUT_DIR"), "/model/ppocrv5_dict.txt"));

        let (device, backend) = pick_device();
        let load = || {
            let det = det_model::Model::from_bytes(Bytes::from_bytes_vec(DET.to_vec()), &device);
            let rec = rec_model::Model::from_bytes(Bytes::from_bytes_vec(REC.to_vec()), &device);
            (det, rec)
        };
        let (det, rec) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(load))
            .map_err(|_| "failed to load neural OCR weights".to_string())?;
        tracing::info!(backend, "spdf: neural OCR ready");
        Ok(Self {
            backend,
            charset: Charset::from_dict(DICT),
            models: Mutex::new(Models { device, det, rec }),
        })
    }

    #[cfg(not(spdf_neural_models))]
    fn detect(&self, _img: &Rgb<'_>) -> SpdfResult<Vec<DetBox>> {
        Err(SpdfError::Ocr("neural OCR unavailable".into()))
    }

    #[cfg(not(spdf_neural_models))]
    fn recognize_crops(&self, _crops: &[(Vec<f32>, usize)]) -> SpdfResult<Vec<LineRec>> {
        Err(SpdfError::Ocr("neural OCR unavailable".into()))
    }

    /// Run the detector; boxes come back in image pixel coordinates.
    #[cfg(spdf_neural_models)]
    fn detect(&self, img: &Rgb<'_>) -> SpdfResult<Vec<DetBox>> {
        use burn::prelude::*;
        let (dw, dh) = det_size(img.w, img.h);
        let input = det_input(img, dw, dh);
        let prob: Vec<f32> = {
            let m = self
                .models
                .lock()
                .map_err(|_| SpdfError::Ocr("ocr lock poisoned".into()))?;
            let x = Tensor::<4>::from_data(TensorData::new(input, [1, 3, dh, dw]), &m.device);
            let y = m.det.forward(x);
            y.into_data()
                .try_to_vec::<f32>()
                .map_err(|e| SpdfError::Ocr(format!("det output: {e:?}")))?
        };
        if prob.len() != dw * dh {
            return Err(SpdfError::Ocr(format!(
                "det output has {} values, expected {}",
                prob.len(),
                dw * dh
            )));
        }
        let (sx, sy) = (img.w as f64 / dw as f64, img.h as f64 / dh as f64);
        let mut boxes = geom::db_boxes(&prob, dw, dh, &DbParams::default());
        for b in &mut boxes {
            for r in [&mut b.rect, &mut b.core] {
                *r = scale_rect(r, sx, sy);
            }
        }
        Ok(boxes)
    }

    /// Recognise normalised-height crops (`REC_H` rows, CHW RGB 0..255).
    #[cfg(spdf_neural_models)]
    fn recognize_crops(&self, crops: &[(Vec<f32>, usize)]) -> SpdfResult<Vec<LineRec>> {
        use burn::prelude::*;
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by_key(|&i| crops[i].1);
        let mut out: Vec<Option<LineRec>> = (0..crops.len()).map(|_| None).collect();
        let m = self
            .models
            .lock()
            .map_err(|_| SpdfError::Ocr("ocr lock poisoned".into()))?;
        for chunk in order.chunks(REC_BATCH) {
            let max_w = chunk
                .iter()
                .map(|&i| crops[i].1)
                .max()
                .unwrap_or(REC_W_BUCKET);
            let bw = max_w.div_ceil(REC_W_BUCKET) * REC_W_BUCKET;
            let b = chunk.len();
            let plane = REC_H * bw;
            let mut data = vec![0f32; b * 3 * plane];
            for (bi, &i) in chunk.iter().enumerate() {
                let (crop, cw) = (&crops[i].0, crops[i].1);
                let cplane = REC_H * cw;
                for c in 0..3 {
                    // Paddle models were trained on BGR input.
                    let src = &crop[(2 - c) * cplane..(3 - c) * cplane];
                    let dst = &mut data[(bi * 3 + c) * plane..(bi * 3 + c + 1) * plane];
                    for y in 0..REC_H {
                        for x in 0..cw {
                            dst[y * bw + x] = src[y * cw + x] / 127.5 - 1.0;
                        }
                    }
                }
            }
            let x = Tensor::<4>::from_data(TensorData::new(data, [b, 3, REC_H, bw]), &m.device);
            let y = m.rec.forward(x);
            let [_, steps, classes] = y.dims();
            if classes != self.charset.num_classes() {
                return Err(SpdfError::Ocr(format!(
                    "rec model has {classes} classes, dictionary expects {}",
                    self.charset.num_classes()
                )));
            }
            let (maxp, idx) = y.max_dim_with_indices(2);
            let probs: Vec<f32> = maxp
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| SpdfError::Ocr(format!("rec output: {e:?}")))?;
            let idx: Vec<usize> = idx
                .into_data()
                .iter::<i64>()
                .map(|v| v.max(0) as usize)
                .collect();
            for (bi, &i) in chunk.iter().enumerate() {
                let r = bi * steps..(bi + 1) * steps;
                let content_steps = (crops[i].1 * steps).div_ceil(bw).clamp(1, steps);
                let chars = ctc::greedy_decode(
                    &idx[r.clone()][..content_steps],
                    &probs[r][..content_steps],
                    &self.charset,
                );
                out[i] = Some(LineRec {
                    chars,
                    content_steps,
                });
            }
        }
        Ok(out
            .into_iter()
            .map(|o| o.expect("every crop recognised"))
            .collect())
    }

    fn ocr_page(&self, img: &Rgb<'_>) -> SpdfResult<Vec<OcrResult>> {
        let boxes = self.detect(img)?;
        if boxes.is_empty() {
            return Ok(Vec::new());
        }
        let phi = self.reading_angle(img, &boxes)?;
        let aligned: Vec<DetBox> = boxes.iter().map(|b| align_box(b, phi)).collect();
        let crops: Vec<(Vec<f32>, usize)> = aligned
            .iter()
            .map(|b| crop_line(img, &b.rect, false))
            .collect();
        let recs = self.recognize_crops(&crops)?;
        let frame = UprightFrame::new(img.w as f64, img.h as f64, phi);
        let mut out = Vec::new();
        for (b, rec) in aligned.iter().zip(recs) {
            let words = ctc::split_words(&rec.chars, rec.content_steps);
            // Inter-word gaps in pixels; a gap wider than ~1.6 text heights
            // is a cell / column break rather than a word space.
            let max_gap = 1.6 * b.core.h / b.rect.w.max(1.0);
            for run in ctc::group_runs(&words, max_gap) {
                let s0 = (run.start - 0.5) * b.rect.w;
                let s1 = (run.end - 0.5) * b.rect.w;
                let (x, y) = b.rect.at((s0 + s1) / 2.0, 0.0);
                let (ux, uy) = frame.map(x, y);
                let half_w = (s1 - s0) / 2.0;
                let half_h = b.core.h / 2.0;
                out.push(OcrResult {
                    text: run.text,
                    bbox: [ux - half_w, uy - half_h, ux + half_w, uy + half_h],
                    confidence: run.confidence,
                });
            }
        }
        Ok(out)
    }

    /// Dominant reading direction of the page in radians (0 = upright,
    /// ±π/2 = rotated a quarter turn, π = upside down), including skew.
    fn reading_angle(&self, img: &Rgb<'_>, boxes: &[DetBox]) -> SpdfResult<f64> {
        use std::f64::consts::{FRAC_PI_2, PI};
        // Only clearly elongated regions say anything about orientation.
        let lines: Vec<&DetBox> = boxes
            .iter()
            .filter(|b| b.core.w >= 2.0 * b.core.h)
            .collect();
        if lines.is_empty() {
            return Ok(0.0);
        }
        let vert: f64 = lines
            .iter()
            .filter(|b| b.rect.is_vertical())
            .map(|b| b.rect.w)
            .sum();
        let total: f64 = lines.iter().map(|b| b.rect.w).sum();
        let vertical = vert > total / 2.0;
        let mut major: Vec<&DetBox> = lines
            .into_iter()
            .filter(|b| b.rect.is_vertical() == vertical)
            .collect();
        // Median axis angle, unwrapped around the dominant direction.
        let base = if vertical { FRAC_PI_2 } else { 0.0 };
        let mut angles: Vec<f64> = major
            .iter()
            .map(|b| {
                let mut a = b.rect.angle;
                while a - base > FRAC_PI_2 {
                    a -= PI;
                }
                while a - base <= -FRAC_PI_2 {
                    a += PI;
                }
                a
            })
            .collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let axis = angles[angles.len() / 2];

        // Vote between the two reading directions along that axis using the
        // recogniser's confidence on the longest lines.
        major.sort_by(|a, b| b.rect.w.partial_cmp(&a.rect.w).unwrap());
        major.truncate(6);
        let mut crops = Vec::with_capacity(major.len() * 2);
        for b in &major {
            let r = align_box(b, axis).rect;
            crops.push(crop_line(img, &r, false));
            crops.push(crop_line(img, &r, true));
        }
        let recs = self.recognize_crops(&crops)?;
        let (mut fwd, mut back) = (0.0, 0.0);
        for pair in recs.chunks(2) {
            fwd += ctc::line_confidence(&pair[0].chars) * pair[0].chars.len().max(1) as f64;
            back += ctc::line_confidence(&pair[1].chars) * pair[1].chars.len().max(1) as f64;
        }
        Ok(if back > fwd * 1.15 { axis + PI } else { axis })
    }
}

/// Re-express a box's rectangle so its reading axis is the one of the four
/// quarter-turn variants closest to `phi`.
fn align_box(b: &DetBox, phi: f64) -> DetBox {
    let pick = |r: &RotRect| -> RotRect {
        let mut best = *r;
        let mut best_d = f64::MAX;
        for k in 0..4 {
            let a = r.angle + k as f64 * std::f64::consts::FRAC_PI_2;
            let d = angle_dist(a, phi);
            if d < best_d {
                best_d = d;
                let (w, h) = if k % 2 == 0 { (r.w, r.h) } else { (r.h, r.w) };
                best = RotRect {
                    w,
                    h,
                    angle: a,
                    ..*r
                };
            }
        }
        best
    };
    DetBox {
        rect: pick(&b.rect),
        core: pick(&b.core),
        score: b.score,
    }
}

fn angle_dist(a: f64, b: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let d = (a - b).rem_euclid(tau);
    d.min(tau - d)
}

/// Maps image coordinates into the frame where text reads left-to-right
/// horizontally: rotate by `-phi` about the image centre, then shift so the
/// rotated image's bounding box starts at the origin.
struct UprightFrame {
    cos: f64,
    sin: f64,
    cx: f64,
    cy: f64,
    ox: f64,
    oy: f64,
}

impl UprightFrame {
    fn new(w: f64, h: f64, phi: f64) -> Self {
        let (sin, cos) = (-phi).sin_cos();
        let nw = (w * cos).abs() + (h * sin).abs();
        let nh = (w * sin).abs() + (h * cos).abs();
        Self {
            cos,
            sin,
            cx: w / 2.0,
            cy: h / 2.0,
            ox: nw / 2.0,
            oy: nh / 2.0,
        }
    }

    fn map(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.cx, y - self.cy);
        (
            self.ox + dx * self.cos - dy * self.sin,
            self.oy + dx * self.sin + dy * self.cos,
        )
    }
}

fn scale_rect(r: &RotRect, sx: f64, sy: f64) -> RotRect {
    // Detector input keeps the aspect ratio to within rounding, so a uniform
    // scale on the extents is accurate enough; the centre scales exactly.
    let s = (sx + sy) / 2.0;
    RotRect {
        cx: r.cx * sx,
        cy: r.cy * sy,
        w: r.w * s,
        h: r.h * s,
        ..*r
    }
}

/// Detector input size: the image scaled so the longest side is at most
/// `DET_MAX_SIDE`, both sides rounded to a multiple of 32.
fn det_size(w: usize, h: usize) -> (usize, usize) {
    let longest = w.max(h) as f64;
    let scale = if longest > DET_MAX_SIDE as f64 {
        DET_MAX_SIDE as f64 / longest
    } else {
        1.0
    };
    let r = |v: usize| (((v as f64 * scale) / 32.0).round() as usize).max(1) * 32;
    (r(w), r(h))
}

/// Resize to `dw × dh` and normalise with the ImageNet statistics the
/// detector was trained with (BGR channel order).
#[cfg_attr(not(spdf_neural_models), allow(dead_code))]
fn det_input(img: &Rgb<'_>, dw: usize, dh: usize) -> Vec<f32> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let full = RotRect {
        cx: img.w as f64 / 2.0,
        cy: img.h as f64 / 2.0,
        w: img.w as f64,
        h: img.h as f64,
        angle: 0.0,
    };
    let mut rgb = geom::crop_rotrect(img, &full, dw, dh, false);
    let plane = dw * dh;
    // RGB planes → BGR planes.
    let (r, rest) = rgb.split_at_mut(plane);
    rest[plane..].swap_with_slice(r);
    for c in 0..3 {
        for v in &mut rgb[c * plane..(c + 1) * plane] {
            *v = (*v / 255.0 - MEAN[c]) / STD[c];
        }
    }
    rgb
}

/// Crop a text line to the recogniser's input height, preserving aspect.
fn crop_line(img: &Rgb<'_>, r: &RotRect, flip: bool) -> (Vec<f32>, usize) {
    let w = ((REC_H as f64 * r.w / r.h.max(1.0)).round() as usize).clamp(REC_STRIDE, REC_MAX_W);
    (geom::crop_rotrect(img, r, w, REC_H, flip), w)
}

/// Pick the compute device: `SPDF_OCR_DEVICE` if set, else a wgpu GPU
/// adapter if one works, else the portable CPU backend.
#[cfg(spdf_neural_models)]
fn pick_device() -> (burn::prelude::Device, &'static str) {
    use burn::prelude::*;
    let works = |d: &Device| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Tensor::<1>::ones([4], d).sum().into_scalar::<f32>() == 4.0
        }))
        .unwrap_or(false)
    };
    let want = std::env::var("SPDF_OCR_DEVICE")
        .unwrap_or_default()
        .to_lowercase();
    #[cfg(feature = "neural-cuda")]
    if want == "cuda" {
        let d = Device::cuda(0);
        if works(&d) {
            return (d, "cuda");
        }
    }
    if want != "cpu" {
        // A missing adapter panics inside wgpu; keep that from reaching users.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let d = std::panic::catch_unwind(|| Device::wgpu(Default::default())).ok();
        let ok = d.as_ref().is_some_and(works);
        std::panic::set_hook(prev);
        if ok {
            return (d.expect("checked"), "wgpu");
        }
    }
    (Device::flex(), "cpu")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn det_size_rounds_to_32_and_caps() {
        assert_eq!(det_size(1275, 1651), (1280, 1664));
        let (w, h) = det_size(5000, 2500);
        assert_eq!(w, 2560);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn upright_frame_identity_and_quarter_turn() {
        let f = UprightFrame::new(100.0, 50.0, 0.0);
        let (x, y) = f.map(10.0, 20.0);
        assert!((x - 10.0).abs() < 1e-9 && (y - 20.0).abs() < 1e-9);
        // Text reading downwards (phi = +90°): the image's top-left corner
        // ends up at the upright frame's top-right.
        let f = UprightFrame::new(100.0, 50.0, FRAC_PI_2);
        let (x, y) = f.map(0.0, 0.0);
        assert!(
            (x - 0.0).abs() < 1e-9 && (y - 100.0).abs() < 1e-9,
            "{x},{y}"
        );
        let (x, y) = f.map(100.0, 0.0);
        assert!((x - 0.0).abs() < 1e-9 && (y - 0.0).abs() < 1e-9, "{x},{y}");
        let (x, y) = f.map(0.0, 50.0);
        assert!(
            (x - 50.0).abs() < 1e-9 && (y - 100.0).abs() < 1e-9,
            "{x},{y}"
        );
    }

    #[test]
    fn upright_frame_keeps_line_order_on_upside_down_page() {
        let f = UprightFrame::new(100.0, 100.0, PI);
        let (_, y_top) = f.map(50.0, 10.0);
        let (_, y_bot) = f.map(50.0, 90.0);
        assert!(y_top > y_bot);
    }

    #[test]
    fn align_box_picks_quarter_turn_closest_to_phi() {
        let r = RotRect {
            cx: 0.0,
            cy: 0.0,
            w: 30.0,
            h: 10.0,
            angle: 0.0,
        };
        let b = DetBox {
            rect: r,
            core: r,
            score: 1.0,
        };
        let a = align_box(&b, FRAC_PI_2);
        assert!((a.rect.angle - FRAC_PI_2).abs() < 1e-9);
        assert_eq!((a.rect.w, a.rect.h), (10.0, 30.0));
        let a = align_box(&b, PI + 0.02);
        assert!(angle_dist(a.rect.angle, PI) < 1e-9);
        assert_eq!((a.rect.w, a.rect.h), (30.0, 10.0));
    }

    #[test]
    fn engine_reports_models_flag() {
        // Construction must be cheap and never touch the GPU.
        let e = NeuralOcrEngine::new();
        assert_eq!(e.name(), "neural");
        let _ = NeuralOcrEngine::models_available();
    }
}
