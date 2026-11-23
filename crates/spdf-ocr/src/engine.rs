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

    /// [`OcrEngine::recognize_rgb`] on a buffer the engine may keep or free
    /// early: callers that have no further use for the pixels hand them
    /// over, so an engine can release the page before its memory-heavy
    /// stages. Default: borrow and delegate.
    fn recognize_rgb_owned(
        &self,
        rgb: Vec<u8>,
        width: usize,
        height: usize,
        options: &OcrOptions,
    ) -> SpdfResult<Vec<OcrResult>> {
        self.recognize_rgb(&rgb, width, height, options)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Reports the size and first byte of the pixels it was given.
    struct Echo;

    impl OcrEngine for Echo {
        fn name(&self) -> &'static str {
            "echo"
        }

        fn recognize(&self, _image: &[u8], _options: &OcrOptions) -> SpdfResult<Vec<OcrResult>> {
            Ok(Vec::new())
        }

        fn accepts_rgb(&self) -> bool {
            true
        }

        fn recognize_rgb(
            &self,
            rgb: &[u8],
            width: usize,
            height: usize,
            _options: &OcrOptions,
        ) -> SpdfResult<Vec<OcrResult>> {
            Ok(vec![OcrResult {
                text: format!("{width}x{height}:{}:{}", rgb.len(), rgb[0]),
                bbox: [0.0; 4],
                confidence: 1.0,
            }])
        }
    }

    #[test]
    fn owned_pixels_default_to_the_borrowing_path() {
        let opts = OcrOptions::default();
        let px = vec![7u8; 2 * 3 * 3];
        let borrowed = Echo.recognize_rgb(&px, 2, 3, &opts).unwrap();
        let owned = Echo.recognize_rgb_owned(px, 2, 3, &opts).unwrap();
        assert_eq!(owned[0].text, "2x3:18:7");
        assert_eq!(owned[0].text, borrowed[0].text);
    }
}
