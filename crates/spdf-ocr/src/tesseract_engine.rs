//! Tesseract-backed OCR engine. Only built with `--features tesseract`.
//!
//! Requires libtesseract + leptonica on the system. The default "en" language
//! needs the `eng.traineddata` file discoverable via `TESSDATA_PREFIX` or an
//! explicit path in `ParseConfig::tessdata_path`.

use spdf_types::{SpdfError, SpdfResult};

use crate::engine::{OcrEngine, OcrOptions, OcrResult};

#[derive(Debug)]
pub struct TesseractEngine {
    datapath: Option<String>,
}

impl TesseractEngine {
    pub fn new(datapath: Option<String>) -> Self {
        Self { datapath }
    }

    fn iso1_to_iso3(code: &str) -> &str {
        match code {
            "en" => "eng",
            "fr" => "fra",
            "de" => "deu",
            "es" => "spa",
            "it" => "ita",
            "pt" => "por",
            "zh" => "chi_sim",
            "ja" => "jpn",
            "ko" => "kor",
            "ar" => "ara",
            other => other,
        }
    }
}

impl OcrEngine for TesseractEngine {
    fn name(&self) -> &'static str {
        "tesseract"
    }

    fn recognize(&self, image: &[u8], options: &OcrOptions) -> SpdfResult<Vec<OcrResult>> {
        let lang = options
            .languages
            .iter()
            .map(|c| Self::iso1_to_iso3(c))
            .collect::<Vec<_>>()
            .join("+");

        let datapath = self.datapath.as_deref();
        let mut tess = tesseract::Tesseract::new(datapath, Some(&lang))
            .map_err(|e| SpdfError::Ocr(format!("tesseract init: {e}")))?;

        tess = tess
            .set_image_from_mem(image)
            .map_err(|e| SpdfError::Ocr(format!("tesseract set_image: {e}")))?;

        // A single page-level OCRResult with the full recognized text. The
        // full implementation will use `get_hocr_text` or `Component::Word`
        // iteration to produce per-line bounding boxes.
        let text = tess
            .get_text()
            .map_err(|e| SpdfError::Ocr(format!("tesseract get_text: {e}")))?;

        Ok(vec![OcrResult {
            text,
            bbox: [0.0, 0.0, 0.0, 0.0],
            confidence: 1.0,
        }])
    }
}
