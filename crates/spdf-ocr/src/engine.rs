use serde::{Deserialize, Serialize};
use spdf_types::SpdfResult;

/// OCR invocation options.
#[derive(Debug, Clone)]
pub struct OcrOptions {
    /// Language code(s). HTTP servers use ISO 639-1 (`"en"`); Tesseract uses
    /// ISO 639-3 (`"eng"`). The caller chooses the right form.
    pub languages: Vec<String>,
    pub correct_rotation: bool,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            languages: vec!["en".into()],
            correct_rotation: false,
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

    /// Run OCR on an image (PNG/JPEG bytes).
    fn recognize(&self, image: &[u8], options: &OcrOptions) -> SpdfResult<Vec<OcrResult>>;

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
