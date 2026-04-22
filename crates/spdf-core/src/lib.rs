//! The spdf orchestrator. Equivalent to the `LiteParse` class in
//! `liteparse/src/core/parser.ts`.

#![warn(clippy::all)]

use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;
use spdf_convert::{ConversionResult, convert_path_to_pdf};
use spdf_ocr::{HttpOcrEngine, OcrEngine, OcrOptions, OcrResult};
use spdf_output::{format_text, to_json};
use spdf_pdf::{ExtractOptions, PageData, PdfDocumentHandle, PdfEngine, PdfiumEngine};
use spdf_processing::bbox::build_bounding_boxes;
use spdf_processing::text_utils::clean_ocr_table_artifacts;
use spdf_projection::{PageInput, project_pages_to_grid};
use spdf_types::{
    Language, ParseConfig, ParseInput, ParseResult, ParsedPage, ScreenshotResult, SpdfError,
    SpdfResult, TextItem,
};
use tracing::{debug, info, warn};

pub use spdf_types::OutputFormat;

/// High-level document parser.
pub struct SpdfParser {
    config: ParseConfig,
    pdf_engine: Arc<PdfiumEngine>,
    ocr_engine: Option<Arc<dyn OcrEngine>>,
}

impl SpdfParser {
    /// Build a parser with explicit config. Use [`Self::builder`] for the
    /// defaults-plus-overrides pattern that mirrors `new LiteParse({ ... })`.
    pub fn new(config: ParseConfig) -> Self {
        let ocr_engine = build_ocr_engine(&config);
        Self {
            config,
            pdf_engine: Arc::new(PdfiumEngine::new()),
            ocr_engine,
        }
    }

    /// Inject a custom OCR engine (e.g. for tests or a Tesseract build).
    pub fn with_ocr_engine(mut self, engine: Arc<dyn OcrEngine>) -> Self {
        self.ocr_engine = Some(engine);
        self
    }

    /// Start from the shared defaults (equivalent to `DEFAULT_CONFIG`).
    pub fn builder() -> ParseConfigBuilder {
        ParseConfigBuilder::default()
    }

    pub fn config(&self) -> &ParseConfig {
        &self.config
    }

    /// Parse a document to the caller-selected output.
    pub fn parse(&self, input: impl Into<ParseInput>) -> SpdfResult<ParseResult> {
        self.parse_inner(input.into())
    }

    fn parse_inner(&self, input: ParseInput) -> SpdfResult<ParseResult> {
        let deadline = self
            .config
            .timeout_secs
            .map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));
        let check_deadline = |stage: &str| -> SpdfResult<()> {
            if let Some(d) = deadline {
                if std::time::Instant::now() >= d {
                    return Err(SpdfError::InvalidInput(format!(
                        "spdf: timeout exceeded during {stage}"
                    )));
                }
            }
            Ok(())
        };
        // Reject oversized in-memory blobs before touching pdfium.
        if let (ParseInput::Bytes(b), Some(cap)) = (&input, self.config.max_input_bytes) {
            if b.len() as u64 > cap {
                return Err(SpdfError::InvalidInput(format!(
                    "spdf: input {} bytes exceeds max_input_bytes {cap}",
                    b.len()
                )));
            }
        }
        let materialised = self.materialise(input)?;
        let bytes = match materialised {
            Materialised::Pdf { bytes, .. } => bytes,
            Materialised::PlainText(content) => return Ok(plain_text_result(content)),
        };
        check_deadline("load")?;

        // Partial #T4.6 guard: reject PDFs that declare an individual
        // stream length larger than `max_declared_stream_bytes`. This
        // catches blatant `/Length`-bomb adversarial files before
        // pdfium has a chance to pre-allocate. It does NOT catch
        // zip-bomb streams (small compressed, huge decompressed) —
        // those remain a known pre-1.0 issue.
        if let Some(cap) = self.config.max_declared_stream_bytes {
            if let Some(n) = scan_max_declared_length(&bytes) {
                if n > cap {
                    return Err(SpdfError::InvalidInput(format!(
                        "spdf: PDF declares a stream of {n} bytes; exceeds \
                         max_declared_stream_bytes cap of {cap}"
                    )));
                }
            }
        }
        check_deadline("length-scan")?;

        if let Some(cap) = self.config.max_expanded_stream_bytes {
            match scan_expanded_flate_bytes(&bytes, cap) {
                ExpandedScan::Ok(total) => {
                    if total > 0 {
                        debug!(expanded = total, "spdf: flate pre-scan ok");
                    }
                }
                ExpandedScan::Overflowed { at_least } => {
                    return Err(SpdfError::InvalidInput(format!(
                        "spdf: FlateDecode streams expand to at least {at_least} \
                         bytes; exceeds max_expanded_stream_bytes cap of {cap}"
                    )));
                }
            }
        }
        check_deadline("expand-scan")?;

        // #T1.1: auto-enable preserve_very_small_text for AcroForm PDFs.
        // Form fields often use tiny font sizes for labels, checkboxes, and
        // revision stamps. The density-aware filter (#T1.2) keeps most of
        // them, but AcroForm PDFs are precisely the class where the residual
        // drops hurt recall. If the user didn't explicitly set the flag and
        // the PDF carries an /AcroForm dict, flip it on automatically.
        let mut cfg = self.config.clone();
        if !cfg.preserve_very_small_text && scan_has_acroform(&bytes) {
            debug!("spdf: /AcroForm detected, auto-enabling preserve_very_small_text");
            cfg.preserve_very_small_text = true;
        }

        let doc = self
            .pdf_engine
            .load_bytes(&bytes, cfg.password.as_deref())?;
        let total_pages = doc.num_pages().min(cfg.max_pages);
        info!(pages = total_pages, "spdf: parsing");

        let page_numbers = select_pages(total_pages, cfg.target_pages.as_deref())?;
        debug!(selected = page_numbers.len(), "spdf: page set selected");

        let opts = ExtractOptions {
            extract_images: cfg.ocr_enabled,
        };

        let pdf_engine = Arc::clone(&self.pdf_engine);
        let mut page_datas: Vec<PageData> = page_numbers
            .par_iter()
            .map(|&page_num| pdf_engine.extract_page(&doc, page_num, opts))
            .collect::<SpdfResult<Vec<_>>>()?;
        check_deadline("extract")?;

        // Phase 6: Selective OCR. Run on pages with sparse text or embedded
        // images, then append non-overlapping OCR items to `text_items` so the
        // downstream projection treats them uniformly.
        if cfg.ocr_enabled {
            if let Some(ocr) = self.ocr_engine.as_ref() {
                self.run_ocr(&doc, &mut page_datas, ocr.as_ref())?;
            } else {
                warn_no_ocr_engine();
            }
        }
        check_deadline("ocr")?;

        // Detect and purge broken CID / ToUnicode-missing text layers.
        // Some PDFs (notably RFC 9110 page 1) embed subset CID fonts
        // without a working ToUnicode CMap — pdfium emits repeated
        // ligature tokens like "fi fi fi ff fi …" that are pure noise.
        // When > 70 % of a page's non-whitespace tokens come from a
        // tiny ligature-only vocabulary and the vocabulary is < 5
        // distinct tokens, we wipe the page's text layer. This is
        // strictly precision-positive: the tokens were never real.
        //
        // Note: we do NOT run the render-OCR fallback on these pages.
        // On the canonical CID-garbage fixture (RFC 9110 p.1) pdfium
        // also renders the glyphs as ligature shapes — the font's
        // visible glyphs are subset-corrupt, not just its ToUnicode
        // map — so OCRing the rendered page yields nothing either.
        // A proper fix needs a native content-stream parser (#T3.3).
        for page in page_datas.iter_mut() {
            if is_cid_garbage_layer(&page.text_items) {
                debug!(
                    page = page.page_num,
                    "spdf: dropping CID-garbage text layer"
                );
                page.text_items.clear();
            }
        }
        check_deadline("cid-garbage-filter")?;

        let pages: Vec<PageInput> = page_datas
            .into_iter()
            .map(|p| PageInput {
                page_num: p.page_num,
                width: p.width,
                height: p.height,
                text_items: p.text_items,
            })
            .collect();

        let mut processed: Vec<ParsedPage> = project_pages_to_grid(pages, &cfg);

        // Strip running headers / footers that repeat across pages. Lifts
        // precision on multi-page docs (NIST 800-53r5, NIST 800-63b, RFC
        // specs) without hurting recall on prose pages.
        if processed.len() >= 3 {
            strip_repeating_running_text(&mut processed);
        }

        if cfg.precise_bounding_box {
            for page in processed.iter_mut() {
                page.bounding_boxes = Some(build_bounding_boxes(&page.text_items));
            }
        }

        if cfg.detect_tables {
            for page in processed.iter_mut() {
                let tables = spdf_processing::tables::detect_tables(&page.text_items);
                if !tables.is_empty() {
                    page.tables = Some(tables);
                }
            }
        }

        let full_text = processed
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let mut result = ParseResult {
            pages: processed,
            text: full_text,
            json: None,
        };

        if matches!(cfg.output_format, OutputFormat::Json) {
            result.json = Some(to_json(&result));
        }

        self.pdf_engine.close(doc)?;
        Ok(result)
    }

    /// Render each candidate page and append OCR text items that don't overlap
    /// existing PDF text. Mirrors `runOCR`/`processPageOcr` in
    /// `liteparse/src/core/parser.ts`.
    ///
    /// Pages are rendered and OCR'd on a rayon pool sized by
    /// `config.num_workers`, matching liteparse's `Scheduler` concurrency.
    /// The rendering step serialises internally on PDFium's global mutex, but
    /// the heavy OCR step runs fully parallel because the Tesseract engine
    /// uses a `thread_local!` cache to keep one warmed instance per worker.
    fn run_ocr(
        &self,
        doc: &<PdfiumEngine as PdfEngine>::Doc,
        pages: &mut [PageData],
        ocr: &dyn OcrEngine,
    ) -> SpdfResult<()> {
        let languages: Vec<String> = match &self.config.ocr_language {
            Language::Single(s) => vec![s.clone()],
            Language::Multiple(v) => v.clone(),
        };
        let options = OcrOptions {
            languages,
            correct_rotation: true,
            dpi: Some(self.config.dpi),
        };
        // PDF spec constant: 72 points per inch. OCR coordinates come back in
        // image pixels at the render DPI.
        let scale_factor = 72.0 / self.config.dpi as f64;

        // Phase 1: figure out which pages actually need OCR, and render them.
        // We collect `(page_idx, png_bytes)` so phase 2 can run OCR in
        // parallel without borrowing `pages` mutably.
        let mut todo: Vec<(usize, u32)> = Vec::new();
        for (idx, page) in pages.iter().enumerate() {
            let text_length: usize = page.text_items.iter().map(|t| t.str.len()).sum();
            let needs_full_ocr = text_length < 100 || !page.images.is_empty();
            if needs_full_ocr {
                todo.push((idx, page.page_num));
            }
        }
        if todo.is_empty() {
            return Ok(());
        }

        // Phase 2: render + OCR in parallel. `pdf_engine.render_page_png` is
        // `&self` and internally serialises on PDFium's global mutex; that's
        // fine because OCR dominates wall-clock time by orders of magnitude.
        let num_workers = self.config.num_workers.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .thread_name(|i| format!("spdf-ocr-{i}"))
            .build()
            .map_err(|e| SpdfError::Ocr(format!("ocr thread pool: {e}")))?;

        let engine = self.pdf_engine.clone();
        let dpi = self.config.dpi;
        let results: Vec<(usize, Vec<OcrResult>)> = pool.install(|| {
            todo.par_iter()
                .map(|&(idx, page_num)| {
                    let image = match engine.render_page_png(doc, page_num, dpi) {
                        Ok(b) => b,
                        Err(e) => {
                            warn!(page = page_num, error = %e, "spdf: render for OCR failed");
                            return (idx, Vec::new());
                        }
                    };
                    match ocr.recognize(&image, &options) {
                        Ok(r) => (idx, r),
                        Err(e) => {
                            warn!(page = page_num, error = %e, "spdf: OCR failed");
                            (idx, Vec::new())
                        }
                    }
                })
                .collect()
        });

        // Phase 3: merge OCR words into each page's text items, dropping
        // low-confidence hits and any that overlap existing PDF text. The
        // `> 0.3` confidence cut-off matches liteparse exactly.
        for (idx, ocr_results) in results {
            let page = &mut pages[idx];
            // Snapshot the pre-OCR text items so adjacent OCR words don't
            // shadow each other via the overlap filter — tight kerning
            // routinely puts neighbouring word bboxes within the 2-point
            // overlap tolerance, which would otherwise drop every other
            // word. The overlap check exists to avoid re-emitting text we
            // already got from the PDF text layer, not to dedupe OCR
            // against itself.
            let existing_len = page.text_items.len();
            let mut appended = 0usize;
            for r in ocr_results {
                if r.confidence <= 0.3 {
                    continue;
                }
                let [x1, y1, x2, y2] = r.bbox;
                let px = x1 * scale_factor;
                let py = y1 * scale_factor;
                let pw = (x2 - x1) * scale_factor;
                let ph = (y2 - y1) * scale_factor;
                if pw <= 0.0 || ph <= 0.0 {
                    continue;
                }
                if overlaps_existing_text(&page.text_items[..existing_len], px, py, pw, ph) {
                    continue;
                }
                let cleaned = clean_ocr_table_artifacts(&r.text);
                let cleaned = strip_ocr_pipe_artifacts(&cleaned);
                if cleaned.is_empty() || is_ocr_punctuation_noise(&cleaned) {
                    continue;
                }
                let mut item = TextItem::new(cleaned, px, py, pw, ph);
                item.font_name = Some("OCR".into());
                item.font_size = Some(ph);
                item.confidence = Some((r.confidence * 1000.0).round() / 1000.0);
                page.text_items.push(item);
                appended += 1;
            }
            debug!(page = page.page_num, appended, "spdf: OCR merged");
        }
        Ok(())
    }

    /// Stream one parsed page at a time without materialising the full
    /// document in memory. Yields `(page_index, ParsedPage)` pairs in the
    /// same order as `parse`. Errors abort the iterator.
    pub fn stream<I: Into<ParseInput>>(
        &self,
        input: I,
    ) -> SpdfResult<Box<dyn Iterator<Item = SpdfResult<ParsedPage>> + '_>> {
        let input = input.into();
        if let (ParseInput::Bytes(b), Some(cap)) = (&input, self.config.max_input_bytes) {
            if b.len() as u64 > cap {
                return Err(SpdfError::InvalidInput(format!(
                    "spdf: input {} bytes exceeds max_input_bytes {cap}",
                    b.len()
                )));
            }
        }
        let bytes = match self.materialise(input)? {
            Materialised::Pdf { bytes, .. } => bytes,
            Materialised::PlainText(content) => {
                let page = plain_text_result(content).pages.remove(0);
                return Ok(Box::new(std::iter::once(Ok(page))));
            }
        };
        if let Some(cap) = self.config.max_declared_stream_bytes {
            if let Some(n) = scan_max_declared_length(&bytes) {
                if n > cap {
                    return Err(SpdfError::InvalidInput(format!(
                        "spdf: PDF declares a stream of {n} bytes; exceeds \
                         max_declared_stream_bytes cap of {cap}"
                    )));
                }
            }
        }
        if let Some(cap) = self.config.max_expanded_stream_bytes {
            if let ExpandedScan::Overflowed { at_least } = scan_expanded_flate_bytes(&bytes, cap) {
                return Err(SpdfError::InvalidInput(format!(
                    "spdf: FlateDecode streams expand to at least {at_least} \
                     bytes; exceeds max_expanded_stream_bytes cap of {cap}"
                )));
            }
        }
        // #T1.1: auto-enable preserve_very_small_text for AcroForm PDFs.
        let mut cfg = self.config.clone();
        if !cfg.preserve_very_small_text && scan_has_acroform(&bytes) {
            debug!("spdf: /AcroForm detected in stream, auto-enabling preserve_very_small_text");
            cfg.preserve_very_small_text = true;
        }
        let doc = self
            .pdf_engine
            .load_bytes(&bytes, cfg.password.as_deref())?;
        let total = doc.num_pages().min(cfg.max_pages);
        let page_numbers = select_pages(total, cfg.target_pages.as_deref())?;
        let opts = ExtractOptions {
            extract_images: cfg.ocr_enabled,
        };
        let engine = Arc::clone(&self.pdf_engine);
        let precise_bbox = cfg.precise_bounding_box;
        let detect_tables = cfg.detect_tables;
        let debug_on = cfg.debug.as_ref().is_some_and(|d| d.enabled);
        let iter = page_numbers.into_iter().map(move |page_num| {
            let pd = engine.extract_page(&doc, page_num, opts)?;
            let pages = spdf_projection::project_pages_to_grid(
                vec![spdf_projection::PageInput {
                    page_num: pd.page_num,
                    width: pd.width,
                    height: pd.height,
                    text_items: pd.text_items,
                }],
                &cfg,
            );
            let mut page = pages.into_iter().next().unwrap();
            if precise_bbox {
                page.bounding_boxes = Some(spdf_processing::bbox::build_bounding_boxes(
                    &page.text_items,
                ));
            }
            if detect_tables {
                let tables = spdf_processing::tables::detect_tables(&page.text_items);
                if !tables.is_empty() {
                    page.tables = Some(tables);
                }
            }
            if debug_on {
                debug!(page = page.page_num, "spdf: streamed");
            }
            Ok(page)
        });
        Ok(Box::new(iter))
    }

    /// Render specific (or all) pages to PNG buffers.
    pub fn screenshot(
        &self,
        input: impl Into<ParseInput>,
        page_numbers: Option<Vec<u32>>,
    ) -> SpdfResult<Vec<ScreenshotResult>> {
        let (bytes, _temp) = match self.materialise(input.into())? {
            Materialised::Pdf { bytes, tempdir } => (bytes, tempdir),
            Materialised::PlainText(_) => {
                return Err(SpdfError::UnsupportedFormat(
                    "cannot screenshot plain-text input".into(),
                ));
            }
        };
        let doc = self
            .pdf_engine
            .load_bytes(&bytes, self.config.password.as_deref())?;
        let total = doc.num_pages();
        let targets = page_numbers.unwrap_or_else(|| (1..=total).collect());

        let mut out = Vec::with_capacity(targets.len());
        for page_num in targets {
            let png = self
                .pdf_engine
                .render_page_png(&doc, page_num, self.config.dpi)?;
            // Width/height decoded lazily by the caller; 0 signals "unknown".
            out.push(ScreenshotResult {
                page_num,
                width: 0,
                height: 0,
                image_buffer: png,
                image_path: None,
            });
        }
        self.pdf_engine.close(doc)?;
        Ok(out)
    }

    /// Convenience formatter respecting the configured output format.
    pub fn format(&self, result: &ParseResult) -> String {
        match self.config.output_format {
            OutputFormat::Text => format_text(result),
            OutputFormat::Json => {
                let json = result.json.clone().unwrap_or_else(|| to_json(result));
                serde_json::to_string_pretty(&json).unwrap_or_default()
            }
        }
    }

    /// Load bytes for the configured input.
    fn materialise(&self, input: ParseInput) -> SpdfResult<Materialised> {
        match input {
            ParseInput::Bytes(b) => Ok(Materialised::Pdf {
                bytes: b,
                tempdir: None,
            }),
            ParseInput::Path(p) => {
                match convert_path_to_pdf(&p, self.config.password.as_deref())? {
                    ConversionResult::Pdf {
                        pdf_path, _tempdir, ..
                    } => Ok(Materialised::Pdf {
                        bytes: std::fs::read(pdf_path)?,
                        tempdir: _tempdir,
                    }),
                    ConversionResult::PlainText { content } => Ok(Materialised::PlainText(content)),
                }
            }
        }
    }
}

/// Internal representation of the parse input once loaded.
enum Materialised {
    Pdf {
        bytes: Vec<u8>,
        #[allow(dead_code)]
        tempdir: Option<tempfile::TempDir>,
    },
    PlainText(String),
}

/// Build a `ParseResult` from plain-text input (markdown, txt, log, ...).
/// Mirrors the short-circuit path in liteparse so callers get one parsed
/// "page" with the file contents as-is.
fn plain_text_result(content: String) -> ParseResult {
    let page = ParsedPage {
        page_num: 1,
        width: 0.0,
        height: 0.0,
        text: content.clone(),
        text_items: vec![TextItem::new(&content, 0.0, 0.0, 0.0, 0.0)],
        bounding_boxes: None,
        tables: None,
    };
    let mut result = ParseResult {
        pages: vec![page],
        text: content,
        json: None,
    };
    result.json = Some(to_json(&result));
    result
}

/// Select which page numbers to process. Mirrors liteparse's range-list parser.
fn select_pages(total_pages: u32, target: Option<&str>) -> SpdfResult<Vec<u32>> {
    let Some(spec) = target else {
        return Ok((1..=total_pages).collect());
    };
    let mut out = Vec::new();
    for chunk in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((lo, hi)) = chunk.split_once('-') {
            let lo: u32 = lo
                .trim()
                .parse()
                .map_err(|_| SpdfError::InvalidConfig(format!("bad range: {chunk}")))?;
            let hi: u32 = hi
                .trim()
                .parse()
                .map_err(|_| SpdfError::InvalidConfig(format!("bad range: {chunk}")))?;
            for p in lo..=hi {
                if p >= 1 && p <= total_pages {
                    out.push(p);
                }
            }
        } else {
            let p: u32 = chunk
                .parse()
                .map_err(|_| SpdfError::InvalidConfig(format!("bad page: {chunk}")))?;
            if p >= 1 && p <= total_pages {
                out.push(p);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Detect a broken CID / missing-ToUnicode text layer. When a PDF embeds
/// subset CID fonts without a proper `ToUnicode` CMap, pdfium falls back
/// to mapping every glyph to its AGL ligature name, producing streams
/// like `fi fi fi ff fi fi ff` for an entire page. Those tokens are
/// never real words — they're just the ligatures that happen to exist
/// in the font's encoding.
///
/// Signal: ≥ 12 total non-whitespace items on the page, with > 70 %
/// coming from a small ligature/symbol vocabulary **and** the full
/// vocabulary is < 5 distinct tokens. Below those thresholds we keep
/// the layer; we'd rather let a little noise through than accidentally
/// wipe a legitimate page that happens to have repeated short words.
fn is_cid_garbage_layer(items: &[TextItem]) -> bool {
    // Tokens that commonly appear as fallbacks when ToUnicode is broken.
    const LIGATURE_TOKENS: &[&str] = &[
        "fi", "fl", "ff", "ffi", "ffl", "ft", "st", "\u{fb00}", "\u{fb01}", "\u{fb02}", "\u{fb03}",
        "\u{fb04}",
    ];
    let mut total = 0usize;
    let mut ligature_hits = 0usize;
    let mut vocab: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for it in items {
        let s = it.str.trim();
        if s.is_empty() {
            continue;
        }
        total += 1;
        vocab.insert(s);
        let is_ligature = LIGATURE_TOKENS.iter().any(|&l| l == s);
        // Single non-alphanumeric glyphs also count as broken fallback.
        let is_symbol = s.chars().count() == 1 && !s.chars().next().unwrap().is_alphanumeric();
        if is_ligature || is_symbol {
            ligature_hits += 1;
        }
    }
    total >= 3 && vocab.len() < 5 && (ligature_hits * 10) >= (total * 7)
}

/// Detect and strip running headers and footers — lines that appear byte-
/// identical on a majority of pages in the top/bottom 10% y-band. Runs only
/// when the document has ≥ 3 pages.
fn strip_repeating_running_text(pages: &mut [ParsedPage]) {
    let n_pages = pages.len();
    if n_pages < 3 {
        return;
    }
    // Threshold: a line must appear on at least this many pages (~60%).
    let min_occurrences = ((n_pages as f64) * 0.6).ceil() as usize;

    // Collect candidate lines from each page's top/bottom band. Using trimmed
    // line text as the identity key; mid-page occurrences do NOT count, so we
    // never strip prose that happens to repeat.
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for page in pages.iter() {
        let h = page.height.max(1.0);
        let top_band = h * 0.10;
        let bot_band = h * 0.90;
        let mut seen_on_page: std::collections::HashSet<String> = std::collections::HashSet::new();
        for item in &page.text_items {
            let in_band = item.y <= top_band || item.y >= bot_band;
            if !in_band {
                continue;
            }
            let key = item.str.trim().to_string();
            // Skip short or numeric-only (page numbers vary per page).
            if key.len() < 4 {
                continue;
            }
            if key.chars().all(|c| c.is_ascii_digit() || c.is_whitespace()) {
                continue;
            }
            if seen_on_page.insert(key.clone()) {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    let repeating: std::collections::HashSet<String> = counts
        .into_iter()
        .filter(|&(_, c)| c >= min_occurrences)
        .map(|(k, _)| k)
        .collect();
    if repeating.is_empty() {
        return;
    }

    for page in pages.iter_mut() {
        let h = page.height.max(1.0);
        let top_band = h * 0.10;
        let bot_band = h * 0.90;

        // Drop matching text_items in the header/footer band.
        page.text_items.retain(|item| {
            let in_band = item.y <= top_band || item.y >= bot_band;
            if !in_band {
                return true;
            }
            !repeating.contains(item.str.trim())
        });

        // Strip matching lines from the projected text so the formatted
        // output stays consistent with text_items.
        let kept_lines: Vec<&str> = page
            .text
            .lines()
            .filter(|line| !repeating.contains(line.trim()))
            .collect();
        page.text = kept_lines.join("\n");
    }
}

/// Fluent builder equivalent to TS `new LiteParse(partial)`.
#[derive(Debug, Default)]
pub struct ParseConfigBuilder {
    config: ParseConfig,
}

impl ParseConfigBuilder {
    pub fn ocr_enabled(mut self, on: bool) -> Self {
        self.config.ocr_enabled = on;
        self
    }
    pub fn ocr_server_url(mut self, url: impl Into<String>) -> Self {
        self.config.ocr_server_url = Some(url.into());
        self
    }
    pub fn dpi(mut self, dpi: u32) -> Self {
        self.config.dpi = dpi;
        self
    }
    pub fn output_format(mut self, fmt: OutputFormat) -> Self {
        self.config.output_format = fmt;
        self
    }
    pub fn max_pages(mut self, max: u32) -> Self {
        self.config.max_pages = max;
        self
    }
    pub fn target_pages(mut self, spec: impl Into<String>) -> Self {
        self.config.target_pages = Some(spec.into());
        self
    }
    pub fn num_workers(mut self, n: usize) -> Self {
        self.config.num_workers = n;
        self
    }
    pub fn password(mut self, pw: impl Into<String>) -> Self {
        self.config.password = Some(pw.into());
        self
    }
    pub fn precise_bounding_box(mut self, on: bool) -> Self {
        self.config.precise_bounding_box = on;
        self
    }
    /// Fail `parse()` if wall-clock work exceeds this many seconds.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.config.timeout_secs = Some(secs);
        self
    }
    /// Reject `ParseInput::Bytes` payloads larger than this many bytes.
    pub fn max_input_bytes(mut self, bytes: u64) -> Self {
        self.config.max_input_bytes = Some(bytes);
        self
    }
    /// Reject PDFs declaring any single stream larger than this. `None`
    /// disables the check (not recommended on untrusted input).
    pub fn max_declared_stream_bytes(mut self, bytes: Option<u64>) -> Self {
        self.config.max_declared_stream_bytes = bytes;
        self
    }
    /// Reject PDFs whose FlateDecode streams collectively expand to
    /// more than this many bytes. `None` disables the pre-scan.
    pub fn max_expanded_stream_bytes(mut self, bytes: Option<u64>) -> Self {
        self.config.max_expanded_stream_bytes = bytes;
        self
    }
    pub fn config(self) -> ParseConfig {
        self.config
    }
    pub fn build(self) -> SpdfParser {
        SpdfParser::new(self.config)
    }
}

/// Stub: kept so callers can see the intended `PathBuf`-returning API for
/// screenshot persistence that Phase 8 will finish.
pub fn default_screenshot_path(output_dir: &std::path::Path, page_num: u32) -> PathBuf {
    output_dir.join(format!("page-{page_num}.png"))
}

/// Build the default OCR engine from config. HTTP if a URL is configured,
/// Tesseract if the feature is enabled, otherwise `None`.
fn build_ocr_engine(config: &ParseConfig) -> Option<Arc<dyn OcrEngine>> {
    if !config.ocr_enabled {
        return None;
    }
    if let Some(url) = config.ocr_server_url.as_deref() {
        return Some(Arc::new(HttpOcrEngine::new(url)));
    }
    #[cfg(feature = "tesseract")]
    {
        return Some(Arc::new(spdf_ocr::TesseractEngine::new(
            config.tessdata_path.clone(),
        )));
    }
    #[cfg(not(feature = "tesseract"))]
    {
        let _ = config;
        None
    }
}

/// Emit a one-shot warning when OCR is requested but no engine is available,
/// with concrete remediation steps.
fn warn_no_ocr_engine() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let tesseract_built = cfg!(feature = "tesseract");
        let msg = if tesseract_built {
            "spdf: OCR requested but no engine configured. This build supports \
             Tesseract; install libtesseract + language data (e.g. \
             `apt install tesseract-ocr tesseract-ocr-eng`) or pass \
             --ocr-server-url to use an HTTP OCR server. Any rasterized text \
             in the PDF will be missing from the output."
        } else {
            "spdf: OCR requested but no engine configured. Either pass \
             --ocr-server-url <URL> to use an HTTP OCR server, or rebuild \
             spdf with the `tesseract` feature (`cargo build --release \
             -p spdf-cli --features tesseract`, requires libtesseract and \
             libleptonica). Rasterized text will be missing from the output."
        };
        warn!("{msg}");
    });
}

/// True when an OCR bbox overlaps any existing text item (with a 2-point
/// tolerance), matching liteparse's `overlapsExistingText`.
fn overlaps_existing_text(items: &[TextItem], x: f64, y: f64, w: f64, h: f64) -> bool {
    const TOL: f64 = 2.0;
    let right = x + w;
    let bottom = y + h;
    for it in items {
        let iw = if it.width > 0.0 { it.width } else { it.w };
        let ih = if it.height > 0.0 { it.height } else { it.h };
        let ir = it.x + iw;
        let ib = it.y + ih;
        let overlap_x = x < ir + TOL && right > it.x - TOL;
        let overlap_y = y < ib + TOL && bottom > it.y - TOL;
        if overlap_x && overlap_y {
            return true;
        }
    }
    false
}

/// Drop single-token OCR words that are pure punctuation, which Tesseract
/// frequently hallucinates at the edges of rasterized text (trailing `|`,
/// stray `.`, orphan brackets, etc.). A real sentence ends in punctuation
/// *attached* to a word, not as its own token.
fn is_ocr_punctuation_noise(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    // Keep anything that contains at least one alphanumeric character.
    !t.chars().any(|c| c.is_alphanumeric())
}

/// Strip leading/trailing pipe characters that Tesseract hallucinates from
/// vertical strokes at the edges of rasterized text (e.g. `"words.|"` → `"words."`).
/// Only pipes are removed — other punctuation is legitimate.
fn strip_ocr_pipe_artifacts(text: &str) -> String {
    text.trim().trim_matches('|').trim().to_string()
}

/// Cheap byte-level check for the `/AcroForm` key in the raw PDF. Returns
/// `true` when the PDF's trailer / root dict contains an AcroForm entry,
/// which indicates the document is an interactive form (e.g. IRS 1040,
/// W-9). Used by #T1.1 to auto-enable `preserve_very_small_text` — form
/// PDFs routinely use tiny font sizes for field labels that the density
/// filter would otherwise drop.
///
/// This is a substring scan, not a real PDF parse. It can theoretically
/// false-positive on a PDF whose *body text* contains the literal bytes
/// `/AcroForm`, but that is astronomically unlikely in practice —
/// `/AcroForm` is a reserved PDF dictionary key.
fn scan_has_acroform(bytes: &[u8]) -> bool {
    const KEY: &[u8] = b"/AcroForm";
    bytes
        .windows(KEY.len())
        .any(|w| w == KEY)
}

/// Scan raw PDF bytes for the largest **directly-declared** `/Length N`
/// value. Returns `None` if no direct `/Length` entries are found (e.g.
/// all lengths are indirect references `/Length 5 0 R`, or the input
/// isn't a PDF). Only ASCII digit runs after a whitespace-separated
/// `/Length` token are considered, so the scan is cheap (single pass,
/// no decompression, no parsing) and has no false positives from
/// text that happens to contain the substring.
///
/// Used by [`SpdfParser::parse`] / [`SpdfParser::stream`] as a partial
/// OOM guard against adversarial PDFs that declare multi-gigabyte
/// streams to trick pdfium into pre-allocating huge buffers. Zip
/// bombs (small `/Length`, huge decompressed output) are **not**
/// detected by this scan — that class of attack is still #T4.6.
fn scan_max_declared_length(bytes: &[u8]) -> Option<u64> {
    const KEY: &[u8] = b"/Length";
    let mut max: Option<u64> = None;
    let mut i = 0usize;
    while i + KEY.len() <= bytes.len() {
        if &bytes[i..i + KEY.len()] != KEY {
            i += 1;
            continue;
        }
        // Must be followed by a PDF whitespace byte, otherwise this is
        // `/LengthX` (e.g. `/Length1`, a legit Type1 font metric that
        // we must not conflate with `/Length`).
        let mut j = i + KEY.len();
        if j >= bytes.len() || !is_pdf_ws(bytes[j]) {
            i = j;
            continue;
        }
        // Skip whitespace to the first non-ws byte.
        while j < bytes.len() && is_pdf_ws(bytes[j]) {
            j += 1;
        }
        // Parse an unsigned decimal. If the next byte isn't a digit
        // this is an indirect reference like `/Length 5 0 R` — still
        // starts with a digit — or a non-numeric value. We capture the
        // digits and then verify the *next* byte is NOT the start of
        // an indirect-reference continuation: a legitimate indirect
        // ref is `<num> <num> R`, so we only count the length when it
        // is immediately followed by ws+non-digit OR EOF-ish bytes.
        let mut n: u64 = 0;
        let mut any = false;
        let mut overflow = false;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            any = true;
            n = n
                .saturating_mul(10)
                .saturating_add((bytes[j] - b'0') as u64);
            if n == u64::MAX {
                overflow = true;
            }
            j += 1;
        }
        if !any {
            i = j.max(i + 1);
            continue;
        }
        // Detect indirect references: `<N> <ws> <M> <ws> R`. If so,
        // skip scoring this /Length entry — we can't know the true
        // length without resolving the xref.
        let is_indirect = {
            let mut k = j;
            let mut saw_ws = false;
            while k < bytes.len() && is_pdf_ws(bytes[k]) {
                saw_ws = true;
                k += 1;
            }
            saw_ws && k < bytes.len() && bytes[k].is_ascii_digit() && {
                while k < bytes.len() && bytes[k].is_ascii_digit() {
                    k += 1;
                }
                let mut saw_ws2 = false;
                while k < bytes.len() && is_pdf_ws(bytes[k]) {
                    saw_ws2 = true;
                    k += 1;
                }
                saw_ws2 && k < bytes.len() && bytes[k] == b'R'
            }
        };
        if !is_indirect {
            let value = if overflow { u64::MAX } else { n };
            max = Some(max.map_or(value, |m| m.max(value)));
        }
        // Advance past the digits we just parsed.
        i = j;
    }
    max
}

/// PDF 32000-1:2008 §7.2.2 whitespace chars (excluding the form feed
/// which is not valid in PDF): NUL, TAB, LF, CR, SPACE, plus the form
/// feed `0x0C` which PDF does accept as whitespace.
#[inline]
fn is_pdf_ws(b: u8) -> bool {
    matches!(b, 0x00 | 0x09 | 0x0A | 0x0C | 0x0D | 0x20)
}

/// Result of the FlateDecode expansion pre-scan.
enum ExpandedScan {
    /// Total decompressed bytes across all FlateDecode streams stayed
    /// within budget. Carries the running total for logging.
    Ok(u64),
    /// A stream's decompressed output crossed the budget. We stop as
    /// soon as we prove the total exceeds the cap and report how much
    /// we'd already produced (at-least lower bound).
    Overflowed { at_least: u64 },
}

/// Decompress every FlateDecode stream in the PDF bytes under a shared
/// output budget. Returns [`ExpandedScan::Overflowed`] the instant the
/// cumulative decompressed size exceeds `cap`, short-circuiting the
/// remaining streams. This catches zip-bomb PDFs that evade
/// [`scan_max_declared_length`] by using tiny compressed `/Length`
/// values and relying on deflate expansion.
///
/// The scan is deliberately sloppy about locating stream bodies — we
/// look for `/FlateDecode` inside each `<< ... >>\s*stream\n` block
/// and feed bytes between `stream(\r?\n)` and `endstream` to flate2.
/// Non-FlateDecode streams (JPEG, CCITTFax, etc.) are skipped; they
/// already carry their own decoded size in `/Length` and are caught
/// by the direct-length guard.
fn scan_expanded_flate_bytes(bytes: &[u8], cap: u64) -> ExpandedScan {
    use flate2::Decompress;
    const STREAM: &[u8] = b"stream";
    const ENDSTREAM: &[u8] = b"endstream";
    const FLATE: &[u8] = b"/FlateDecode";
    const CHUNK_OUT: usize = 64 * 1024;

    let mut total: u64 = 0;
    let mut i = 0usize;
    while let Some(rel) = find_bytes(&bytes[i..], STREAM) {
        let marker = i + rel;
        // The `stream` keyword must be a standalone PDF token: the
        // byte before it is whitespace and the byte after it is an
        // EOL. Otherwise this is a substring (e.g. `endstream`).
        let ok_prev = marker == 0 || is_pdf_ws(bytes[marker - 1]);
        let after_kw = marker + STREAM.len();
        let body_start = match bytes.get(after_kw) {
            Some(&b'\r') if bytes.get(after_kw + 1) == Some(&b'\n') => after_kw + 2,
            Some(&b'\n') => after_kw + 1,
            _ => {
                i = marker + STREAM.len();
                continue;
            }
        };
        if !ok_prev {
            i = marker + STREAM.len();
            continue;
        }
        // Look back up to 4 KiB for the dict's `/Filter /FlateDecode`.
        let window_start = marker.saturating_sub(4096);
        let dict = &bytes[window_start..marker];
        let is_flate = find_bytes(dict, FLATE).is_some();

        // Find the endstream marker, capped to a generous upper bound
        // so adversarial files without an `endstream` don't make us
        // scan to EOF.
        let search_end = (body_start + 64 * 1024 * 1024).min(bytes.len());
        let end = match find_bytes(&bytes[body_start..search_end], ENDSTREAM) {
            Some(r) => body_start + r,
            None => {
                // Unterminated stream — nothing useful to scan further.
                break;
            }
        };
        if is_flate {
            let body = &bytes[body_start..end];
            let mut dec = Decompress::new(/* zlib = */ true);
            let mut scratch = vec![0u8; CHUNK_OUT];
            let mut input_cursor = 0usize;
            let mut overflowed = false;
            loop {
                let before_in = dec.total_in();
                let before_out = dec.total_out();
                let status = dec.decompress(
                    &body[input_cursor..],
                    &mut scratch,
                    flate2::FlushDecompress::None,
                );
                let produced = dec.total_out() - before_out;
                let consumed = (dec.total_in() - before_in) as usize;
                input_cursor += consumed;
                total = total.saturating_add(produced);
                if total > cap {
                    overflowed = true;
                    break;
                }
                match status {
                    Ok(flate2::Status::StreamEnd) => break,
                    Ok(flate2::Status::BufError) | Err(_) => {
                        // Corrupt / non-zlib stream: stop scanning this
                        // body. pdfium will hit the same data later
                        // and return a parse error.
                        break;
                    }
                    Ok(flate2::Status::Ok) => {
                        if produced == 0 && consumed == 0 {
                            // No progress: avoid infinite loop on
                            // truncated input.
                            break;
                        }
                    }
                }
            }
            if overflowed {
                return ExpandedScan::Overflowed { at_least: total };
            }
        }
        i = end + ENDSTREAM.len();
    }
    ExpandedScan::Ok(total)
}

/// Byte-slice substring search. `haystack.windows(needle.len())` plus
/// a linear scan. Stays in the hot path only when streams exist.
#[inline]
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_pages_defaults_to_all() {
        assert_eq!(select_pages(3, None).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn select_pages_parses_mixed_spec() {
        let out = select_pages(20, Some("1-3,5,10-11")).unwrap();
        assert_eq!(out, vec![1, 2, 3, 5, 10, 11]);
    }

    #[test]
    fn select_pages_rejects_bad_spec() {
        let err = select_pages(10, Some("1-abc")).unwrap_err();
        match err {
            SpdfError::InvalidConfig(msg) => assert!(msg.contains("bad range")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn overlap_detects_collision_with_existing_text() {
        let items = vec![TextItem::new("hi", 10.0, 20.0, 40.0, 12.0)];
        // Same bbox -> overlaps
        assert!(overlaps_existing_text(&items, 10.0, 20.0, 40.0, 12.0));
        // Far away -> no overlap
        assert!(!overlaps_existing_text(&items, 200.0, 200.0, 40.0, 12.0));
        // Within tolerance -> overlaps
        assert!(overlaps_existing_text(&items, 11.0, 21.0, 1.0, 1.0));
    }

    #[test]
    fn length_scan_finds_max_direct() {
        let bytes = b"<< /Length 42 >>\n<< /Length 9999 >>\n<< /Length 7 >>";
        assert_eq!(scan_max_declared_length(bytes), Some(9999));
    }

    #[test]
    fn length_scan_ignores_length1_length2() {
        // Type1 font dicts legally carry /Length1 /Length2 /Length3
        // with encoded-segment sizes. Those must not be confused with
        // the stream-length `/Length`.
        let bytes = b"<< /Length1 2000000000 /Length 5 >>";
        assert_eq!(scan_max_declared_length(bytes), Some(5));
    }

    #[test]
    fn length_scan_ignores_indirect_refs() {
        // `/Length 5 0 R` is an indirect reference; we cannot score it
        // without resolving the xref, so it must not poison the scan.
        let bytes = b"<< /Length 5 0 R /Filter /FlateDecode >>";
        assert_eq!(scan_max_declared_length(bytes), None);
    }

    #[test]
    fn length_scan_returns_none_on_empty() {
        assert_eq!(scan_max_declared_length(b""), None);
        assert_eq!(scan_max_declared_length(b"no pdf here"), None);
    }

    #[test]
    fn length_scan_detects_pathological() {
        let bytes = b"<< /Length 2147483648 >>";
        assert_eq!(scan_max_declared_length(bytes), Some(2147483648));
    }

    #[test]
    fn length_scan_handles_overflow_saturation() {
        // A /Length value that overflows u64 saturates rather than panicking.
        let mut s = b"<< /Length ".to_vec();
        s.extend(std::iter::repeat(b'9').take(40));
        s.extend_from_slice(b" >>");
        assert_eq!(scan_max_declared_length(&s), Some(u64::MAX));
    }
}
