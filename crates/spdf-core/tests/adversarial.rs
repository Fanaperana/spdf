//! Adversarial-fixture smoke tests.
//!
//! These run `SpdfParser::parse` against the PDFs in
//! `example/corpus/` that exercise edge cases (encrypted, malformed,
//! CJK) and assert the pipeline behaves as documented: typed error,
//! no panic, no runaway.

use spdf_core::SpdfParser;
use spdf_types::{ParseInput, SpdfError};

fn read_fixture(relative: &str) -> Option<Vec<u8>> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest.parent()?.parent()?.join("example").join(relative);
    std::fs::read(path).ok()
}

/// An encrypted PDF without the password must produce a typed
/// `PasswordRequired` (or `InvalidPassword`) error — never a panic.
#[test]
fn encrypted_pdf_requires_password() {
    let Some(pdf) = read_fixture("corpus/encrypted.pdf") else {
        return;
    };
    let parser = SpdfParser::builder().ocr_enabled(false).build();
    match parser.parse(ParseInput::Bytes(pdf)) {
        Err(SpdfError::PasswordRequired) | Err(SpdfError::InvalidPassword) => {}
        Err(SpdfError::Pdf(msg)) => {
            // pdfium may surface this as a generic Pdf error on some
            // platforms; accept it as long as the message mentions the
            // password.
            assert!(
                msg.to_lowercase().contains("password"),
                "Pdf error without password hint: {msg}"
            );
        }
        other => panic!("expected password error, got {other:?}"),
    }
}

/// A 200-byte truncated PDF must fail cleanly with a typed error. The
/// point of the test is "never panic, never hang, always return Err".
#[test]
fn malformed_pdf_fails_cleanly() {
    let Some(pdf) = read_fixture("corpus/malformed.pdf") else {
        return;
    };
    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_pages(8)
        .build();
    let err = parser
        .parse(ParseInput::Bytes(pdf))
        .expect_err("truncated PDF must not parse");
    // Typed error — don't care which kind, as long as it's one of ours.
    let _: &SpdfError = &err;
}

/// A CJK PDF must round-trip without losing pages or tripping a UTF-8
/// boundary assertion. We don't benchmark accuracy against English-only
/// ground truth — we just assert the parser produces *some* text items
/// and no panic.
#[test]
fn cjk_pdf_produces_text_items() {
    let Some(pdf) = read_fixture("corpus/cjk-unicode-p1-2.pdf") else {
        return;
    };
    let parser = SpdfParser::builder().ocr_enabled(false).build();
    let result = parser
        .parse(ParseInput::Bytes(pdf))
        .expect("cjk pdf must parse");
    assert!(!result.pages.is_empty());
    let total_items: usize = result.pages.iter().map(|p| p.text_items.len()).sum();
    assert!(total_items > 0, "no text extracted from CJK fixture");
}
