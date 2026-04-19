//! Tesseract-backed OCR engine. Only built with `--features tesseract`.
//!
//! Requires `libtesseract` + `libleptonica` at runtime, and `*.traineddata`
//! files for the selected language(s). Discovery order:
//!
//! 1. `tessdata_path` passed into [`TesseractEngine::new`] (from
//!    `ParseConfig::tessdata_path` / `--tessdata-path`).
//! 2. `$TESSDATA_PREFIX` (libtesseract reads this automatically when `None`).
//! 3. System default (e.g. `/usr/share/tessdata`).
//!
//! ## Per-word output
//!
//! Mirrors `liteparse`'s `tesseract.js` engine: we ask libtesseract for the
//! TSV report (level=5 rows == words) and yield one [`OcrResult`] per word,
//! with a `[x1, y1, x2, y2]` pixel bbox and a `0.0..=1.0` confidence. Callers
//! filter low-confidence noise downstream.
//!
//! ## Instance caching
//!
//! Initializing a `Tesseract` loads the traineddata (~100–300 MB RAM for
//! `eng`). Doing that per page would crater throughput. We keep one
//! `Tesseract` alive per worker thread via a `thread_local!`, keyed by
//! `(datapath, lang)`. Threads are typically rayon workers, so a 4-core
//! machine warms up 4 engines and reuses them for every page.
//!
//! `tesseract::Tesseract` is `Send + !Sync`, which matches the thread-local
//! strategy exactly.

use std::cell::RefCell;

use spdf_types::{SpdfError, SpdfResult};

use crate::engine::{OcrEngine, OcrOptions, OcrResult};

thread_local! {
    /// One reusable Tesseract instance per worker thread, keyed by
    /// `(datapath, lang)`. A new key replaces the entry and drops the old
    /// instance so we don't hold multiple language models at once.
    static TESS_CACHE: RefCell<Option<CacheEntry>> = const { RefCell::new(None) };
}

struct CacheEntry {
    key: (Option<String>, String),
    /// `Option` so we can `take()` the instance to call consuming methods
    /// (`set_image_from_mem`, `recognize`) and put it back afterwards.
    tess: Option<tesseract::Tesseract>,
}

#[derive(Debug)]
pub struct TesseractEngine {
    datapath: Option<String>,
}

impl TesseractEngine {
    pub fn new(datapath: Option<String>) -> Self {
        Self { datapath }
    }

    /// Normalize language codes to Tesseract ISO 639-3 form.
    /// Mirrors `TesseractEngine.normalizeLanguage` in liteparse.
    fn normalize_language(code: &str) -> String {
        match code.to_ascii_lowercase().as_str() {
            "en" | "en-us" | "en-gb" => "eng".into(),
            "fr" => "fra".into(),
            "de" => "deu".into(),
            "es" => "spa".into(),
            "it" => "ita".into(),
            "pt" => "por".into(),
            "ru" => "rus".into(),
            "nl" => "nld".into(),
            "pl" => "pol".into(),
            "tr" => "tur".into(),
            "vi" => "vie".into(),
            "zh" | "zh-cn" => "chi_sim".into(),
            "zh-tw" => "chi_tra".into(),
            "ja" => "jpn".into(),
            "ko" => "kor".into(),
            "ar" => "ara".into(),
            // Pass-through: already ISO 639-3 (`eng`, `fra`, `chi_sim`, ...)
            // or anything exotic the user wants to try.
            _ => code.to_string(),
        }
    }

    fn joined_language(langs: &[String]) -> String {
        langs
            .iter()
            .map(|c| Self::normalize_language(c))
            .collect::<Vec<_>>()
            .join("+")
    }
}

impl OcrEngine for TesseractEngine {
    fn name(&self) -> &'static str {
        "tesseract"
    }

    fn recognize(&self, image: &[u8], options: &OcrOptions) -> SpdfResult<Vec<OcrResult>> {
        if options.languages.is_empty() {
            return Err(SpdfError::Ocr(
                "tesseract: no language codes provided".into(),
            ));
        }
        let lang = Self::joined_language(&options.languages);

        let tsv = TESS_CACHE.with(|cell| -> SpdfResult<String> {
            let mut slot = cell.borrow_mut();
            let key = (self.datapath.clone(), lang.clone());

            let needs_new = match slot.as_ref() {
                Some(e) => e.key != key,
                None => true,
            };
            if needs_new {
                // Drop the old instance first (frees its language model RAM).
                *slot = None;
                let tess = tesseract::Tesseract::new(self.datapath.as_deref(), Some(&lang))
                    .map_err(|e| SpdfError::Ocr(format!("tesseract init ({lang}): {e}")))?;
                *slot = Some(CacheEntry {
                    key,
                    tess: Some(tess),
                });
            }
            let entry = slot.as_mut().expect("initialised above");

            // Move the instance through the consuming-builder chain and put
            // it back for the next page on this thread.
            let tess = entry.tess.take().expect("tesseract instance present");
            let tess = tess
                .set_image_from_mem(image)
                .map_err(|e| SpdfError::Ocr(format!("tesseract set_image: {e}")))?;
            let mut tess = tess
                .recognize()
                .map_err(|e| SpdfError::Ocr(format!("tesseract recognize: {e}")))?;
            let tsv = tess
                .get_tsv_text(0)
                .map_err(|e| SpdfError::Ocr(format!("tesseract get_tsv_text: {e}")))?;
            entry.tess = Some(tess);
            Ok(tsv)
        })?;

        Ok(parse_tsv(&tsv))
    }
}

/// Parse Tesseract's TSV output into per-word [`OcrResult`]s.
///
/// Columns (tab-separated):
/// ```text
/// level page_num block_num par_num line_num word_num left top width height conf text
/// ```
/// Only `level == 5` (word) rows are kept. `conf` is 0..100 from Tesseract;
/// we normalize to `0.0..=1.0`. Rows with `conf < 0` (Tesseract's "no
/// confidence" sentinel) or empty text are dropped.
fn parse_tsv(tsv: &str) -> Vec<OcrResult> {
    let mut out = Vec::new();
    for line in tsv.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 12 {
            continue;
        }
        let level: i32 = match cols[0].parse() {
            Ok(v) => v,
            Err(_) => continue, // header row has level="level"
        };
        if level != 5 {
            continue;
        }
        let left: f64 = cols[6].parse().unwrap_or(0.0);
        let top: f64 = cols[7].parse().unwrap_or(0.0);
        let width: f64 = cols[8].parse().unwrap_or(0.0);
        let height: f64 = cols[9].parse().unwrap_or(0.0);
        let conf: f64 = cols[10].parse().unwrap_or(-1.0);
        if conf < 0.0 || width <= 0.0 || height <= 0.0 {
            continue;
        }
        let text = cols[11].trim();
        if text.is_empty() {
            continue;
        }
        out.push(OcrResult {
            text: text.to_string(),
            bbox: [left, top, left + width, top + height],
            confidence: conf / 100.0,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tsv_keeps_only_word_level_rows() {
        let tsv = "\
level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t
2\t1\t1\t0\t0\t0\t0\t0\t100\t100\t-1\t
5\t1\t1\t1\t1\t1\t10\t20\t30\t15\t92\tHello
5\t1\t1\t1\t1\t2\t50\t20\t40\t15\t88\tworld
5\t1\t1\t1\t1\t3\t0\t0\t0\t0\t-1\tnoise
5\t1\t1\t1\t1\t4\t100\t20\t20\t15\t10\t
";
        let out = parse_tsv(tsv);
        assert_eq!(out.len(), 2, "only real words survive, got {:?}", out);
        assert_eq!(out[0].text, "Hello");
        assert_eq!(out[0].bbox, [10.0, 20.0, 40.0, 35.0]);
        assert!((out[0].confidence - 0.92).abs() < 1e-9);
        assert_eq!(out[1].text, "world");
    }

    #[test]
    fn normalize_language_maps_common_codes() {
        assert_eq!(TesseractEngine::normalize_language("en"), "eng");
        assert_eq!(TesseractEngine::normalize_language("EN-US"), "eng");
        assert_eq!(TesseractEngine::normalize_language("zh"), "chi_sim");
        assert_eq!(TesseractEngine::normalize_language("eng"), "eng");
        assert_eq!(TesseractEngine::normalize_language("klingon"), "klingon");
    }
}
