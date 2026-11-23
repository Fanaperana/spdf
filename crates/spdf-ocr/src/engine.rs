use serde::{Deserialize, Serialize};
use spdf_types::SpdfResult;

/// OCR invocation options.
#[derive(Debug, Clone)]
pub struct OcrOptions {
    /// Language code(s). HTTP servers use ISO 639-1 (`"en"`); Tesseract uses
    /// ISO 639-3 (`"eng"`). The caller chooses the right form.
    pub languages: Vec<String>,
    pub correct_rotation: bool,
    /// Render DPI the input image was produced at. Used by Tesseract to size
    /// characters correctly. `None` leaves it to the engine's heuristics.
    pub dpi: Option<u32>,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            languages: vec!["en".into()],
            correct_rotation: false,
            dpi: None,
        }
    }
}

/// One detected text region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    pub text: String,
    /// `[x1, y1, x2, y2]` in image pixels.
    pub bbox: [f64; 4],
    pub confidence: f64,
}

/// OCR engine contract. Mirrors `OcrEngine` in
/// `liteparse/src/engines/ocr/interface.ts`.
pub trait OcrEngine: Send + Sync {
    fn name(&self) -> &'static str;

    /// Called once per document, before any page is recognised, with the
    /// number of pages that will be. Engines with an expensive setup choice
    /// (e.g. GPU vs CPU) can size it to the job. Default: no-op.
    fn prepare(&self, _pages: usize) {}

    /// Run OCR on an image (PNG/JPEG bytes).
    fn recognize(&self, image: &[u8], options: &OcrOptions) -> SpdfResult<Vec<OcrResult>>;

    /// Whether [`OcrEngine::recognize_rgb`] is implemented, so callers can
    /// hand over decoded pixels instead of an encoded image.
    fn accepts_rgb(&self) -> bool {
        false
    }

    /// Run OCR on packed 8-bit RGB pixels (`width × height × 3` bytes),
    /// skipping an encode/decode round trip. Only called when
    /// [`OcrEngine::accepts_rgb`] is true; the default reports it as
    /// unsupported.
    fn recognize_rgb(
        &self,
        _rgb: &[u8],
        _width: usize,
        _height: usize,
        _options: &OcrOptions,
    ) -> SpdfResult<Vec<OcrResult>> {
        Err(spdf_types::SpdfError::Ocr(format!(
            "{} OCR does not take raw pixels",
            self.name()
        )))
    }

    /// Default batch impl delegates to `recognize` sequentially. Engines with
    /// real batch APIs (e.g. a remote server that accepts a JSON array) should
    /// override for a meaningful throughput win.
    fn recognize_batch(
        &self,
        images: &[&[u8]],
        options: &OcrOptions,
    ) -> SpdfResult<Vec<Vec<OcrResult>>> {
        images
            .iter()
            .map(|img| self.recognize(img, options))
            .collect()
    }
}
