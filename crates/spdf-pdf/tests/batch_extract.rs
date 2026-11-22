//! The batched extraction path must return exactly what per-page
//! extraction returns.
#![cfg(feature = "pdfium")]

use spdf_pdf::{ExtractOptions, PdfEngine, PdfiumEngine};

fn fixture(name: &str) -> Option<Vec<u8>> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../example/corpus")
        .join(name);
    std::fs::read(p).ok()
}

#[test]
fn batch_matches_per_page_extraction() {
    let Some(bytes) = fixture("rfc8446-p1-2.pdf") else {
        eprintln!("skipping: fixture not found");
        return;
    };
    let engine = PdfiumEngine::new();
    let doc = match engine.load_bytes(&bytes, None) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: pdfium unavailable: {e}");
            return;
        }
    };
    let opts = ExtractOptions {
        extract_images: true,
        extract_rulings: true,
    };
    let order = [2u32, 1];
    let batch = engine.extract_pages(&doc, &order, opts).unwrap();
    assert_eq!(batch.len(), order.len());
    for (got, &n) in batch.iter().zip(&order) {
        let single = engine.extract_page(&doc, n, opts).unwrap();
        assert_eq!(got.page_num, n);
        assert_eq!(got.width, single.width);
        assert_eq!(got.height, single.height);
        assert_eq!(got.rulings, single.rulings);
        assert_eq!(got.images.len(), single.images.len());
        let a: Vec<_> = got.text_items.iter().map(|t| (&t.str, t.x, t.y)).collect();
        let b: Vec<_> = single
            .text_items
            .iter()
            .map(|t| (&t.str, t.x, t.y))
            .collect();
        assert!(!a.is_empty());
        assert_eq!(a, b);
    }
}

#[test]
fn batch_rejects_page_zero() {
    let Some(bytes) = fixture("rfc8446-p1-2.pdf") else {
        return;
    };
    let engine = PdfiumEngine::new();
    let Ok(doc) = engine.load_bytes(&bytes, None) else {
        return;
    };
    assert!(
        engine
            .extract_pages(&doc, &[1, 0], ExtractOptions::default())
            .is_err()
    );
}
