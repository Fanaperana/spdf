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
//! Backend choice: a GPU pays a fixed start-up cost per process (adapter and
//! device creation, kernel compilation: 0.4-1 s) and a few hundred MB of
//! driver memory, which only pays off on long jobs. Documents with fewer
//! than [`GPU_MIN_PAGES`] pages to OCR therefore run on the CPU; longer ones
//! use a wgpu adapter when one works.
//!
//! Environment:
//! - `SPDF_OCR_DEVICE=cpu|wgpu|cuda` forces a backend (`cuda` needs the
//!   `neural-cuda` feature).
//! - `SPDF_OCR_THREADS=N` sets the CPU backend's thread count.

pub mod ctc;
#[cfg(test)]
mod fold;
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
#[cfg(spdf_neural_models)]
#[allow(clippy::all, unused)]
mod rec_server_model {
    include!(concat!(env!("OUT_DIR"), "/model/ch_PP-OCRv5_rec_server.rs"));
}
#[cfg(spdf_neural_models)]
#[allow(clippy::all, unused)]
mod rec_latin_model {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/latin_PP-OCRv5_rec_mobile.rs"
    ));
}

/// Upper bound on CPU inference threads by default; beyond this the
/// small OCR tensors stop scaling.
const MAX_CPU_THREADS: usize = 16;
/// Recogniser input height for PP-OCRv5.
const REC_H: usize = 48;
/// Recogniser width downsampling (input pixels per CTC timestep).
const REC_STRIDE: usize = 8;
/// Widest line crop fed to the recogniser.
const REC_MAX_W: usize = 3200;
/// Most crops per recogniser batch.
const REC_BATCH: usize = 8;
/// Most padded input columns per recogniser batch. Activation memory grows
/// with batch × width, so long lines go in small batches.
const REC_BATCH_COLS: usize = 2048;
/// Column budget of a CPU recogniser batch: small, so separate lines run
/// on separate cores instead of padding into one wide forward pass.
const REC_CPU_BATCH_COLS: usize = 512;
/// Most padded recogniser columns in flight at once on the CPU; bounds the
/// activation memory of concurrent batches.
const REC_CPU_INFLIGHT_COLS: usize = 4096;
/// Pad crop widths up to a multiple of this, so GPU kernels see few shapes.
const REC_W_BUCKET: usize = 64;
/// Longest side fed to the detector.
const DET_MAX_SIDE: usize = 2560;
/// Longest side of the first, fast detection pass (a Letter page at 150 dpi
/// scaled by ~0.78). Smaller sizes find the same lines but with box edges
/// coarse enough to cost the recogniser the odd glyph (`ll`, `Ø`).
const DET_FAST_SIDE: usize = 1280;
/// When the typical line box (DB-unclipped, so ~1.5x the glyph height)
/// found by the fast pass is shorter than this in detector pixels, the page
/// has small print: detect again at full size.
const DET_MIN_LINE_H: f64 = 14.0;
/// Width cap (in multiples of the line height) of the crops used for the
/// reading-direction vote; a dozen glyphs settle it as well as a full line.
const ORIENT_MAX_ASPECT: f64 = 10.0;
/// Longest lines read both ways for the reading-direction vote.
const ORIENT_LINES: usize = 4;

/// Built-in neural OCR engine. Cheap to construct: the models and the GPU
/// device are initialised lazily on the first page that needs OCR.
pub struct NeuralOcrEngine {
    runtime: OnceLock<Result<Runtime, String>>,
    /// Pages the current job will OCR (see [`OcrEngine::prepare`]).
    pages_hint: std::sync::atomic::AtomicUsize,
    /// One page at a time: a page already spreads its work over every
    /// core, and concurrent pages would multiply the activation memory.
    /// Taken on the caller's thread, never inside the inference pool, where
    /// work stealing could hand a waiting page to the thread holding it.
    page_lock: Mutex<()>,
}

/// Fewest pages to OCR in one document for the GPU to be worth its
/// start-up cost; smaller jobs run on the CPU.
pub const GPU_MIN_PAGES: usize = 8;

impl Default for NeuralOcrEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralOcrEngine {
    pub fn new() -> Self {
        Self {
            runtime: OnceLock::new(),
            pages_hint: std::sync::atomic::AtomicUsize::new(1),
            page_lock: Mutex::new(()),
        }
    }

    /// True when this build embeds the OCR models.
    pub fn models_available() -> bool {
        cfg!(spdf_neural_models)
    }

    fn runtime(&self) -> SpdfResult<&Runtime> {
        self.runtime
            .get_or_init(|| {
                Runtime::init(self.pages_hint.load(std::sync::atomic::Ordering::Relaxed))
            })
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

    fn prepare(&self, pages: usize) {
        // Only the first job decides the backend; later jobs reuse it.
        self.pages_hint
            .store(pages, std::sync::atomic::Ordering::Relaxed);
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
        // A poisoned lock only means another page panicked; the models are
        // read-only, so carry on.
        let _page = self.page_lock.lock().unwrap_or_else(|e| e.into_inner());
        let t0 = std::time::Instant::now();
        let run = || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.ocr_page(&rgb)))
                .map_err(|_| SpdfError::Ocr("neural OCR backend panicked".into()))
        };
        // CPU kernels and recogniser batches parallelise over the engine's
        // own pool rather than the caller's page workers.
        let out = match &rt.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }??;
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
    /// Thread pool for the CPU backend's kernels (`None` on a GPU).
    pool: Option<rayon::ThreadPool>,
    charset: Charset,
    latin_charset: Charset,
    #[cfg(spdf_neural_models)]
    models: Models,
}

#[cfg(spdf_neural_models)]
struct Models {
    device: burn::prelude::Device,
    det: det_model::Model,
    rec: rec_model::Model,
    /// Server recogniser, loaded on the first CJK line.
    rec_server: OnceLock<rec_server_model::Model>,
    /// Latin recogniser, loaded on the first Latin-script line.
    rec_latin: OnceLock<rec_latin_model::Model>,
}

// The models are only read during inference, so pages and recogniser
// batches can run them concurrently without a lock.
#[cfg(spdf_neural_models)]
const _: fn() = || {
    fn sync<T: Send + Sync>() {}
    sync::<Models>();
};

/// Embedded weights, aligned so tensors can alias them without a copy.
#[cfg(spdf_neural_models)]
#[repr(C, align(64))]
struct Aligned<B: ?Sized>(B);

#[cfg(spdf_neural_models)]
static DET_BPK: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/model/ch_PP-OCRv5_det_mobile.bpk"
)));
#[cfg(spdf_neural_models)]
static REC_BPK: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/model/ch_PP-OCRv5_rec_mobile.bpk"
)));
#[cfg(spdf_neural_models)]
static REC_SERVER_BPK: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/model/ch_PP-OCRv5_rec_server.bpk"
)));
#[cfg(spdf_neural_models)]
static REC_LATIN_BPK: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/model/latin_PP-OCRv5_rec_mobile.bpk"
)));

/// Load a model's weights straight from the binary's read-only data: on the
/// CPU backend the tensors alias those pages instead of copying them (the
/// server recogniser alone is ~85 MB).
#[cfg(spdf_neural_models)]
fn load_static<M: burn_store::ModuleSnapshot>(mut model: M, data: &'static [u8]) -> M {
    let mut store = burn_store::BurnpackStore::from_static(data);
    model
        .load_from(&mut store)
        .unwrap_or_else(|e| panic!("failed to load embedded OCR weights: {e}"));
    model
}

impl Runtime {
    /// True when recogniser batches should run concurrently (CPU backend).
    fn parallel(&self) -> bool {
        self.pool.is_some()
    }

    #[cfg(not(spdf_neural_models))]
    fn init(_pages: usize) -> Result<Self, String> {
        let _ = (
            cpu_pool,
            want_gpu,
            rec_tasks,
            rec_waves,
            batch_cols,
            step_maxima,
        );
        let _ = (REC_OUTPUT_LOGITS, Mutex::new(()));
        Err("neural OCR models were not available when spdf was built".into())
    }

    #[cfg(spdf_neural_models)]
    fn init(pages: usize) -> Result<Self, String> {
        const DICT: &str = include_str!(concat!(env!("OUT_DIR"), "/model/ppocrv5_dict.txt"));
        const LATIN_DICT: &str =
            include_str!(concat!(env!("OUT_DIR"), "/model/ppocrv5_latin_dict.txt"));

        let t0 = std::time::Instant::now();
        let (device, backend) = pick_device(pages);
        let t_dev = t0.elapsed();
        let load = || {
            let det = load_static(det_model::Model::new(&device), &DET_BPK.0);
            let rec = load_static(rec_model::Model::new(&device), &REC_BPK.0);
            (det, rec)
        };
        let (det, rec) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(load))
            .map_err(|_| "failed to load neural OCR weights".to_string())?;
        tracing::info!(
            backend,
            device_ms = t_dev.as_millis() as u64,
            load_ms = (t0.elapsed() - t_dev).as_millis() as u64,
            "spdf: neural OCR ready"
        );
        let pool = (backend == "cpu").then(cpu_pool).flatten();
        Ok(Self {
            backend,
            pool,
            charset: Charset::from_dict(DICT),
            latin_charset: Charset::from_dict(LATIN_DICT),
            models: Models {
                device,
                det,
                rec,
                rec_server: OnceLock::new(),
                rec_latin: OnceLock::new(),
            },
        })
    }

    #[cfg(not(spdf_neural_models))]
    fn detect(&self, _img: &Rgb<'_>, _max_side: usize) -> SpdfResult<(Vec<DetBox>, f64)> {
        Err(SpdfError::Ocr("neural OCR unavailable".into()))
    }

    #[cfg(not(spdf_neural_models))]
    fn recognize_crops(&self, _crops: &[(Vec<f32>, usize)]) -> SpdfResult<Vec<LineRec>> {
        Err(SpdfError::Ocr("neural OCR unavailable".into()))
    }

    #[cfg(not(spdf_neural_models))]
    fn recognize_crops_latin(&self, _crops: &[(Vec<f32>, usize)]) -> SpdfResult<Vec<LineRec>> {
        Err(SpdfError::Ocr("neural OCR unavailable".into()))
    }

    #[cfg(not(spdf_neural_models))]
    fn recognize_crops_server(&self, _crops: &[(Vec<f32>, usize)]) -> SpdfResult<Vec<LineRec>> {
        Err(SpdfError::Ocr("neural OCR unavailable".into()))
    }

    /// Run the detector on the image scaled so its longest side is at most
    /// `max_side`. Boxes come back in image pixel coordinates, together with
    /// the detector-to-image scale factor.
    #[cfg(spdf_neural_models)]
    fn detect(&self, img: &Rgb<'_>, max_side: usize) -> SpdfResult<(Vec<DetBox>, f64)> {
        use burn::prelude::*;
        let (dw, dh) = det_size(img.w, img.h, max_side);
        let input = det_input(img, dw, dh);
        let prob: Vec<f32> = {
            let m = &self.models;
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
        Ok((boxes, (sx + sy) / 2.0))
    }

    /// Detect text lines: a fast pass on a downscaled page, repeated at
    /// full size only when that pass finds nothing or mostly small print.
    fn detect_adaptive(&self, img: &Rgb<'_>) -> SpdfResult<Vec<DetBox>> {
        if det_size(img.w, img.h, DET_FAST_SIDE) == det_size(img.w, img.h, DET_MAX_SIDE) {
            return Ok(self.detect(img, DET_MAX_SIDE)?.0);
        }
        let (boxes, scale) = self.detect(img, DET_FAST_SIDE)?;
        if !needs_full_res(&boxes, scale) {
            return Ok(boxes);
        }
        tracing::debug!(n = boxes.len(), "ocr: small print, detecting at full size");
        Ok(self.detect(img, DET_MAX_SIDE)?.0)
    }

    /// Recognise normalised-height crops (`REC_H` rows, CHW RGB 0..255)
    /// with the fast mobile recogniser.
    #[cfg(spdf_neural_models)]
    fn recognize_crops(&self, crops: &[(Vec<f32>, usize)]) -> SpdfResult<Vec<LineRec>> {
        let m = &self.models;
        run_rec(self.parallel(), &m.device, &self.charset, crops, |x| {
            m.rec.forward(x)
        })
    }

    /// Recognise crops with the larger server recogniser (loaded lazily).
    #[cfg(spdf_neural_models)]
    fn recognize_crops_server(&self, crops: &[(Vec<f32>, usize)]) -> SpdfResult<Vec<LineRec>> {
        let m = &self.models;
        let rec = m.rec_server.get_or_init(|| {
            load_static(rec_server_model::Model::new(&m.device), &REC_SERVER_BPK.0)
        });
        run_rec(self.parallel(), &m.device, &self.charset, crops, |x| {
            rec.forward(x)
        })
    }

    /// Recognise crops with the Latin recogniser (loaded lazily).
    #[cfg(spdf_neural_models)]
    fn recognize_crops_latin(&self, crops: &[(Vec<f32>, usize)]) -> SpdfResult<Vec<LineRec>> {
        let m = &self.models;
        let rec = m
            .rec_latin
            .get_or_init(|| load_static(rec_latin_model::Model::new(&m.device), &REC_LATIN_BPK.0));
        run_rec(
            self.parallel(),
            &m.device,
            &self.latin_charset,
            crops,
            |x| rec.forward(x),
        )
    }

    /// Re-read every non-CJK line with the Latin recogniser and fuse the two
    /// decodes character by character (see [`ctc::fuse`]): the primary
    /// dictionary has no `ï`, `Ø`, `ź`, ..., while the Latin one has no
    /// Vietnamese tone marks, so each fills the other's gaps.
    fn refine_latin(&self, crops: &[(Vec<f32>, usize)], recs: &mut [LineRec]) -> SpdfResult<()> {
        let idx: Vec<usize> = recs
            .iter()
            .enumerate()
            .filter(|(_, r)| !mostly_cjk(&r.chars))
            .map(|(i, _)| i)
            .collect();
        if idx.is_empty() {
            return Ok(());
        }
        let sub: Vec<(Vec<f32>, usize)> = idx.iter().map(|&i| crops[i].clone()).collect();
        let latin = self.recognize_crops_latin(&sub)?;
        for (i, l) in idx.into_iter().zip(latin) {
            if l.content_steps != recs[i].content_steps {
                continue;
            }
            recs[i].chars = ctc::fuse(&recs[i].chars, &l.chars, &self.charset, &self.latin_charset);
        }
        Ok(())
    }

    /// Re-read lines the mobile model decoded as mostly CJK with the server
    /// recogniser, which is markedly better on Han / kana glyphs. Latin
    /// lines stay with the mobile model, whose word spacing is better.
    fn refine_cjk(&self, crops: &[(Vec<f32>, usize)], recs: &mut [LineRec]) -> SpdfResult<()> {
        let idx: Vec<usize> = recs
            .iter()
            .enumerate()
            .filter(|(_, r)| mostly_cjk(&r.chars))
            .map(|(i, _)| i)
            .collect();
        if idx.is_empty() {
            return Ok(());
        }
        let sub: Vec<(Vec<f32>, usize)> = idx.iter().map(|&i| crops[i].clone()).collect();
        let better = self.recognize_crops_server(&sub)?;
        for (i, r) in idx.into_iter().zip(better) {
            if !r.chars.is_empty() {
                recs[i] = r;
            }
        }
        Ok(())
    }
}

/// Run a CTC recogniser over normalised-height crops, batching crops of
/// similar width. With `parallel` (CPU backend) the batches are small and
/// run concurrently on the current rayon pool: a single forward pass only
/// keeps a few cores busy, so independent lines are the real parallelism.
#[cfg(spdf_neural_models)]
fn run_rec(
    parallel: bool,
    device: &burn::prelude::Device,
    charset: &Charset,
    crops: &[(Vec<f32>, usize)],
    forward: impl Fn(burn::prelude::Tensor<4>) -> burn::prelude::Tensor<3> + Sync,
) -> SpdfResult<Vec<LineRec>> {
    use rayon::prelude::*;
    let mut order: Vec<usize> = (0..crops.len()).collect();
    order.sort_by_key(|&i| crops[i].1);
    let widths: Vec<usize> = order.iter().map(|&i| crops[i].1).collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    for len in rec_tasks(&widths, parallel) {
        chunks.push(&order[start..start + len]);
        start += len;
    }
    let run = |chunk: &&[usize]| rec_batch(device, charset, crops, chunk, &forward);
    let mut out: Vec<Option<LineRec>> = (0..crops.len()).map(|_| None).collect();
    if parallel {
        // Concurrent batches each hold their activations, so run them in
        // waves that bound the padded columns in flight (peak memory).
        let cols: Vec<usize> = chunks.iter().map(|c| batch_cols(crops, c)).collect();
        let budget = REC_CPU_INFLIGHT_COLS;
        let mut start = 0;
        for len in rec_waves(&cols, budget) {
            let wave = &chunks[start..start + len];
            start += len;
            let done: Vec<Vec<(usize, LineRec)>> =
                wave.par_iter().map(run).collect::<SpdfResult<_>>()?;
            for (i, r) in done.into_iter().flatten() {
                out[i] = Some(r);
            }
        }
    } else {
        for chunk in &chunks {
            for (i, r) in run(chunk)? {
                out[i] = Some(r);
            }
        }
    }
    Ok(out
        .into_iter()
        .map(|o| o.expect("every crop recognised"))
        .collect())
}

/// Whether the recognisers output logits (the build stripped their final
/// softmax, see `fold.rs`) rather than probabilities.
const REC_OUTPUT_LOGITS: bool = cfg!(spdf_rec_logits);

/// Logits more than this below a step's maximum add under `e^-20` each to
/// the softmax denominator; skipping their `exp` keeps the decode cheap.
const SOFTMAX_CUTOFF: f32 = 20.0;

/// Per CTC step (row of `classes` scores): the arg-max class and its
/// probability. With `logits`, the probability is the softmax of the
/// maximum, `1 / Σ exp(l - max)`, computed without materialising the full
/// distribution; otherwise the scores already are probabilities.
fn step_maxima(scores: &[f32], classes: usize, logits: bool) -> (Vec<usize>, Vec<f32>) {
    let steps = scores.len() / classes.max(1);
    let mut idx = Vec::with_capacity(steps);
    let mut probs = Vec::with_capacity(steps);
    for row in scores.chunks_exact(classes.max(1)) {
        let (mut best, mut max) = (0, f32::NEG_INFINITY);
        for (c, &v) in row.iter().enumerate() {
            if v > max {
                (best, max) = (c, v);
            }
        }
        let p = if logits {
            let floor = max - SOFTMAX_CUTOFF;
            let sum: f32 = row
                .iter()
                .filter(|&&v| v >= floor)
                .map(|&v| (v - max).exp())
                .sum();
            1.0 / sum
        } else {
            max
        };
        idx.push(best);
        probs.push(p);
    }
    (idx, probs)
}

/// Padded input columns of one recogniser batch (crops × padded width).
fn batch_cols(crops: &[(Vec<f32>, usize)], chunk: &[usize]) -> usize {
    let max_w = chunk.iter().map(|&i| crops[i].1).max().unwrap_or(0);
    chunk.len() * max_w.div_ceil(REC_W_BUCKET) * REC_W_BUCKET
}

/// Group consecutive batches (by their padded column counts) into waves
/// of at most `budget` columns; returns the wave lengths. A batch larger
/// than the budget forms a wave alone.
fn rec_waves(cols: &[usize], budget: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let (mut n, mut sum) = (0, 0);
    for &c in cols {
        if n > 0 && sum + c > budget {
            out.push(n);
            (n, sum) = (0, 0);
        }
        n += 1;
        sum += c;
    }
    if n > 0 {
        out.push(n);
    }
    out
}

/// Recognise one batch of crops (indices into `crops`).
#[cfg(spdf_neural_models)]
fn rec_batch(
    device: &burn::prelude::Device,
    charset: &Charset,
    crops: &[(Vec<f32>, usize)],
    chunk: &[usize],
    forward: &impl Fn(burn::prelude::Tensor<4>) -> burn::prelude::Tensor<3>,
) -> SpdfResult<Vec<(usize, LineRec)>> {
    use burn::prelude::*;
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
    let x = Tensor::<4>::from_data(TensorData::new(data, [b, 3, REC_H, bw]), device);
    let y = forward(x);
    let [_, steps, classes] = y.dims();
    if classes != charset.num_classes() {
        return Err(SpdfError::Ocr(format!(
            "rec model has {classes} classes, dictionary expects {}",
            charset.num_classes()
        )));
    }
    let scores: Vec<f32> = y
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| SpdfError::Ocr(format!("rec output: {e:?}")))?;
    let (idx, probs) = step_maxima(&scores, classes, REC_OUTPUT_LOGITS);
    drop(scores);
    let mut out = Vec::with_capacity(b);
    for (bi, &i) in chunk.iter().enumerate() {
        let r = bi * steps..(bi + 1) * steps;
        let content_steps = (crops[i].1 * steps).div_ceil(bw).clamp(1, steps);
        let chars = ctc::greedy_decode(
            &idx[r.clone()][..content_steps],
            &probs[r][..content_steps],
            charset,
        );
        out.push((
            i,
            LineRec {
                chars,
                content_steps,
            },
        ));
    }
    Ok(out)
}

/// Batch lengths for recognising crops sorted by width: GPU batches are
/// large ([`REC_BATCH`] crops, [`REC_BATCH_COLS`] columns) to keep the
/// device busy; CPU batches only group short crops (up to
/// [`REC_CPU_BATCH_COLS`] columns) so lines can run on separate cores.
fn rec_tasks(sorted_widths: &[usize], parallel: bool) -> Vec<usize> {
    if parallel {
        rec_batches(sorted_widths, REC_BATCH, REC_CPU_BATCH_COLS)
    } else {
        rec_batches(sorted_widths, REC_BATCH, REC_BATCH_COLS)
    }
}

/// Split crops (sorted by width, ascending) into consecutive batches of at
/// most `max_n` crops and `max_cols` padded columns; returns the batch
/// lengths. A crop wider than the column budget gets a batch alone.
fn rec_batches(sorted_widths: &[usize], max_n: usize, max_cols: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut n = 0;
    for &w in sorted_widths {
        // Widths ascend, so the newest crop sets the batch's padded width.
        let bw = w.div_ceil(REC_W_BUCKET) * REC_W_BUCKET;
        if n > 0 && (n == max_n || (n + 1) * bw > max_cols) {
            out.push(n);
            n = 0;
        }
        n += 1;
    }
    if n > 0 {
        out.push(n);
    }
    out
}

/// True for Han ideographs, kana, Hangul and CJK punctuation.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x1100..=0x11FF
        | 0x3000..=0x30FF
        | 0x3130..=0x318F
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xAC00..=0xD7AF
        | 0xF900..=0xFAFF
        | 0xFF00..=0xFFEF
        | 0x20000..=0x2FA1F)
}

/// A decoded line is CJK when at least a third of its visible characters
/// are CJK (so one stray glyph on a Latin line does not count).
fn mostly_cjk(chars: &[ctc::DecodedChar]) -> bool {
    let (mut cjk, mut total) = (0usize, 0usize);
    for c in chars.iter().flat_map(|d| d.text.chars()) {
        if c.is_whitespace() {
            continue;
        }
        total += 1;
        cjk += usize::from(is_cjk(c));
    }
    cjk > 0 && cjk * 3 >= total
}

impl Runtime {
    fn ocr_page(&self, img: &Rgb<'_>) -> SpdfResult<Vec<OcrResult>> {
        let t = std::time::Instant::now();
        let boxes = self.detect_adaptive(img)?;
        tracing::debug!(
            ms = t.elapsed().as_millis() as u64,
            n = boxes.len(),
            "ocr: detect"
        );
        if boxes.is_empty() {
            return Ok(Vec::new());
        }
        let t = std::time::Instant::now();
        let phi = self.reading_angle(img, &boxes)?;
        tracing::debug!(ms = t.elapsed().as_millis() as u64, "ocr: orientation");
        let aligned: Vec<DetBox> = boxes.iter().map(|b| align_box(b, phi)).collect();
        let crops: Vec<(Vec<f32>, usize)> = aligned
            .iter()
            .map(|b| crop_line(img, &b.rect, false))
            .collect();
        let t = std::time::Instant::now();
        let mut recs = self.recognize_crops(&crops)?;
        tracing::debug!(ms = t.elapsed().as_millis() as u64, "ocr: rec");
        let t = std::time::Instant::now();
        self.refine_cjk(&crops, &mut recs)?;
        tracing::debug!(ms = t.elapsed().as_millis() as u64, "ocr: rec cjk");
        let t = std::time::Instant::now();
        self.refine_latin(&crops, &mut recs)?;
        tracing::debug!(ms = t.elapsed().as_millis() as u64, "ocr: rec latin");
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
        major.truncate(ORIENT_LINES);
        let mut crops = Vec::with_capacity(major.len() * 2);
        for b in &major {
            let r = orient_rect(&align_box(b, axis).rect);
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

/// True when the fast detection pass is not trustworthy: it found no text,
/// or its median line is too short (in detector pixels) for reliable
/// detection, so small print may have been missed or merged.
fn needs_full_res(boxes: &[DetBox], scale: f64) -> bool {
    let mut hs: Vec<f64> = boxes
        .iter()
        .map(|b| b.rect.w.min(b.rect.h) / scale.max(1e-9))
        .collect();
    if hs.is_empty() {
        return true;
    }
    hs.sort_by(|a, b| a.total_cmp(b));
    hs[hs.len() / 2] < DET_MIN_LINE_H
}

/// The central part of a line rectangle, at most `ORIENT_MAX_ASPECT` line
/// heights long, for the reading-direction vote.
fn orient_rect(r: &RotRect) -> RotRect {
    RotRect {
        w: r.w.min(ORIENT_MAX_ASPECT * r.h),
        ..*r
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
/// `max_side`, both sides rounded to a multiple of 32.
fn det_size(w: usize, h: usize, max_side: usize) -> (usize, usize) {
    let longest = w.max(h) as f64;
    let scale = if longest > max_side as f64 {
        max_side as f64 / longest
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

/// Threads for CPU inference: `SPDF_OCR_THREADS` if set, else every
/// available core up to [`MAX_CPU_THREADS`].
fn cpu_threads() -> usize {
    std::env::var("SPDF_OCR_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(MAX_CPU_THREADS)
        })
}

fn cpu_pool() -> Option<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(cpu_threads())
        .thread_name(|i| format!("spdf-nn-{i}"))
        .build()
        .ok()
}

/// Whether to try a GPU for a job of `pages` pages: `SPDF_OCR_DEVICE`
/// decides when set (`cpu` → no, anything else → yes), otherwise only jobs
/// of at least [`GPU_MIN_PAGES`] pages.
fn want_gpu(device_env: &str, pages: usize) -> bool {
    match device_env {
        "" => pages >= GPU_MIN_PAGES,
        "cpu" => false,
        _ => true,
    }
}

/// Pick the compute device: `SPDF_OCR_DEVICE` if set, else (for long jobs)
/// a wgpu GPU adapter if one works, else the portable CPU backend.
#[cfg(spdf_neural_models)]
fn pick_device(pages: usize) -> (burn::prelude::Device, &'static str) {
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
    if want_gpu(&want, pages) {
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
        assert_eq!(det_size(1275, 1651, DET_MAX_SIDE), (1280, 1664));
        let (w, h) = det_size(5000, 2500, DET_MAX_SIDE);
        assert_eq!(w, 2560);
        assert_eq!(h % 32, 0);
        // The fast pass keeps the aspect ratio within rounding.
        let (w, h) = det_size(1275, 1651, DET_FAST_SIDE);
        assert_eq!((w, h), (992, 1280));
    }

    fn line_box(w: f64, h: f64) -> DetBox {
        let r = RotRect {
            cx: 100.0,
            cy: 100.0,
            w,
            h,
            angle: 0.0,
        };
        DetBox {
            rect: r,
            core: r,
            score: 0.9,
        }
    }

    #[test]
    fn full_res_detection_only_for_empty_or_small_print() {
        // Image-space boxes; detector ran at half size (scale 2).
        let body: Vec<DetBox> = (0..5).map(|_| line_box(400.0, 30.0)).collect();
        assert!(!needs_full_res(&body, 2.0));
        let tiny: Vec<DetBox> = (0..5).map(|_| line_box(200.0, 16.0)).collect();
        assert!(needs_full_res(&tiny, 2.0));
        assert!(!needs_full_res(&tiny, 1.0));
        assert!(needs_full_res(&[], 2.0));
        // Vertical lines are measured across, not along.
        let vertical: Vec<DetBox> = (0..3).map(|_| line_box(30.0, 400.0)).collect();
        assert!(!needs_full_res(&vertical, 2.0));
    }

    #[test]
    fn orientation_crop_keeps_centre_and_caps_length() {
        let r = RotRect {
            cx: 50.0,
            cy: 60.0,
            w: 1000.0,
            h: 20.0,
            angle: 0.3,
        };
        let o = orient_rect(&r);
        assert_eq!((o.cx, o.cy, o.h, o.angle), (50.0, 60.0, 20.0, 0.3));
        assert!((o.w - ORIENT_MAX_ASPECT * 20.0).abs() < 1e-9);
        let short = RotRect { w: 100.0, ..r };
        assert_eq!(orient_rect(&short).w, 100.0);
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

    fn decoded(s: &str) -> Vec<ctc::DecodedChar> {
        s.chars()
            .enumerate()
            .map(|(i, c)| ctc::DecodedChar {
                text: c.to_string(),
                step: i * 3,
                prob: 0.9,
            })
            .collect()
    }

    #[test]
    fn cjk_line_detection() {
        assert!(mostly_cjk(&decoded("第三季度总收为千百万元。")));
        assert!(mostly_cjk(&decoded("2024年第三季度")));
        assert!(mostly_cjk(&decoded("ひらがな カタカナ")));
        assert!(mostly_cjk(&decoded("한국어 텍스트")));
        // Latin lines, including one with a stray ideograph, stay Latin.
        assert!(!mostly_cjk(&decoded("Façade naïve coöperate")));
        assert!(!mostly_cjk(&decoded("Total revenue 中 was up")));
        assert!(!mostly_cjk(&decoded("")));
        assert!(!mostly_cjk(&decoded("   ")));
    }

    #[test]
    fn gpu_only_for_long_jobs_unless_forced() {
        assert!(!want_gpu("", 1));
        assert!(!want_gpu("", GPU_MIN_PAGES - 1));
        assert!(want_gpu("", GPU_MIN_PAGES));
        assert!(!want_gpu("cpu", 500));
        assert!(want_gpu("wgpu", 1));
        assert!(want_gpu("cuda", 1));
    }

    #[test]
    fn rec_batches_respect_count_and_column_budget() {
        let gpu = |w: &[usize]| rec_tasks(w, false);
        assert_eq!(gpu(&[]), Vec::<usize>::new());
        // Short crops: limited by the crop count.
        assert_eq!(gpu(&[100; 20]), vec![8, 8, 4]);
        // 1000 px pads to 1024: two per 2048-column batch.
        assert_eq!(gpu(&[1000; 5]), vec![2, 2, 1]);
        // A crop wider than the whole budget still gets recognised.
        assert_eq!(gpu(&[200, 300, 3200]), vec![2, 1]);
        let total: usize = gpu(&[64, 128, 640, 700, 900, 1500, 3000]).iter().sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn step_maxima_matches_softmax() {
        let logits = [1.0f32, 3.0, 0.5, -40.0, 2.0, 2.5, 2.4, 0.0];
        let (idx, p) = step_maxima(&logits, 4, true);
        assert_eq!(idx, vec![1, 1]);
        for (row, &got) in logits.chunks(4).zip(&p) {
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let want = 1.0 / row.iter().map(|v| (v - max).exp()).sum::<f32>();
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        // Probabilities pass through unchanged.
        let probs = [0.1f32, 0.7, 0.2, 0.6, 0.3, 0.1];
        assert_eq!(step_maxima(&probs, 3, false), (vec![1, 0], vec![0.7, 0.6]));
    }

    #[test]
    fn rec_waves_bound_columns_in_flight() {
        assert_eq!(rec_waves(&[], 100), Vec::<usize>::new());
        assert_eq!(rec_waves(&[40, 40, 40, 40], 100), vec![2, 2]);
        // An oversized batch still runs, alone.
        assert_eq!(rec_waves(&[30, 500, 30], 100), vec![1, 1, 1]);
        assert_eq!(rec_waves(&[10; 5], 1000), vec![5]);
    }

    #[test]
    fn cpu_rec_tasks_keep_long_lines_apart() {
        let cpu = |w: &[usize]| rec_tasks(w, true);
        // Short crops still share a forward pass ...
        assert_eq!(cpu(&[100; 10]), vec![4, 4, 2]);
        // ... long lines each get their own, so they run on separate cores.
        assert_eq!(cpu(&[900, 1200, 1500]), vec![1, 1, 1]);
        assert_eq!(cpu(&[60, 60, 1500]), vec![2, 1]);
    }

    #[test]
    fn engine_reports_models_flag() {
        // Construction must be cheap and never touch the GPU.
        let e = NeuralOcrEngine::new();
        assert_eq!(e.name(), "neural");
        let _ = NeuralOcrEngine::models_available();
    }
}
