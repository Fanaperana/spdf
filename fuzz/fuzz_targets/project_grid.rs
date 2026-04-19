#![no_main]
//! Fuzz `project_pages_to_grid` + `detect_tables` directly, bypassing
//! pdfium. These are pure-Rust pipelines, so any panic / OOB / infinite
//! loop is strictly our bug and easy to fix. Coverage-guided mutation
//! over synthetic `TextItem` vectors exercises every branch of the
//! clustering / grid / header-footer / table passes.

use arbitrary::Arbitrary;
use libfuzzer_sys::{Corpus, fuzz_target};
use spdf_processing::tables::detect_tables;
use spdf_projection::{PageInput, project_pages_to_grid};
use spdf_types::{ParseConfig, TextItem};

/// Fuzz-friendly projection of `TextItem` — every field constrained to
/// a sane finite range so we hit interesting geometry rather than
/// NaN/∞ edge cases (those already have unit tests).
#[derive(Arbitrary, Debug)]
struct FuzzItem {
    str: String,
    x: u16,
    y: u16,
    w: u16,
    h: u16,
    font_size: Option<u8>,
}

impl From<FuzzItem> for TextItem {
    fn from(f: FuzzItem) -> Self {
        let x = f64::from(f.x) * 0.1;
        let y = f64::from(f.y) * 0.1;
        let w = (f64::from(f.w) * 0.05).max(0.5);
        let h = (f64::from(f.h) * 0.05).max(0.5);
        let mut t = TextItem::new(f.str, x, y, w, h);
        if let Some(s) = f.font_size {
            t.font_size = Some(f64::from(s) * 0.25 + 1.0);
        }
        t
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzPage {
    items: Vec<FuzzItem>,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    pages: Vec<FuzzPage>,
    preserve_small: bool,
    detect_tables: bool,
}

fuzz_target!(|input: FuzzInput| -> Corpus {
    // Hard cap total work so we don't waste the budget on inputs that
    // just blow through libfuzzer's per-iter time slice. Anything over
    // a few thousand items is exercising the same code paths as the
    // smaller inputs; real PDFs rarely exceed ~500 text items per page.
    let total: usize = input.pages.iter().map(|p| p.items.len()).sum();
    if total > 4096 || input.pages.len() > 16 {
        return Corpus::Reject;
    }

    let pages: Vec<PageInput> = input
        .pages
        .into_iter()
        .enumerate()
        .map(|(i, p)| PageInput {
            page_num: (i as u32) + 1,
            width: 612.0,
            height: 792.0,
            text_items: p.items.into_iter().map(Into::into).collect(),
        })
        .collect();

    let mut config = ParseConfig::default();
    config.preserve_very_small_text = input.preserve_small;
    config.detect_tables = input.detect_tables;
    config.ocr_enabled = false;

    let projected = project_pages_to_grid(pages, &config);

    if input.detect_tables {
        for page in &projected {
            let _ = detect_tables(&page.text_items);
        }
    }

    Corpus::Keep
});
