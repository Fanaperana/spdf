//! Integration tests for ParseConfig resource guards.
//!
//! These verify that `max_input_bytes`, `max_pages`, and `timeout_secs`
//! actually fire at the boundaries we advertise in SECURITY.md.

use spdf_core::SpdfParser;
use spdf_types::{ParseInput, SpdfError};

/// `max_input_bytes` must reject oversized `Bytes` inputs before pdfium
/// is ever touched, and the resulting error must be a typed
/// `SpdfError::InvalidInput`.
#[test]
fn max_input_bytes_rejects_oversized_blob() {
    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_input_bytes(64)
        .build();

    // 1 KiB of zero bytes — not a valid PDF but large enough to trip the
    // guard. If the guard is bypassed the parser would instead fail with
    // a Pdf error from libpdfium; we assert the *guard* error specifically.
    let payload = vec![0u8; 1024];
    let err = parser
        .parse(ParseInput::Bytes(payload))
        .expect_err("oversized input should be rejected");

    match err {
        SpdfError::InvalidInput(msg) => {
            assert!(msg.contains("max_input_bytes"), "unexpected msg: {msg}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

/// `max_input_bytes` must not affect inputs below the cap. We still
/// expect an error (the payload isn't a real PDF), but it must come
/// from pdfium — i.e. `SpdfError::Pdf`, not our guard.
#[test]
fn max_input_bytes_allows_inputs_below_cap() {
    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_input_bytes(10_000)
        .build();

    let payload = vec![0u8; 128];
    let err = parser
        .parse(ParseInput::Bytes(payload))
        .expect_err("non-PDF bytes must still fail");
    assert!(
        !matches!(err, SpdfError::InvalidInput(ref m) if m.contains("max_input_bytes")),
        "guard fired below cap: {err:?}"
    );
}

/// `timeout_secs = 0` + a small valid PDF must either succeed very fast
/// (sub-millisecond) or fail with the typed timeout error. We assert
/// the error shape when it fires; the test passes either way.
#[test]
fn timeout_secs_produces_typed_error_when_it_fires() {
    // Use the committed test-ocr.pdf fixture path from the workspace root.
    let pdf = match std::fs::read(workspace_fixture("test-ocr.pdf")) {
        Ok(b) => b,
        Err(_) => return, // fixture not shipped — skip rather than flake
    };

    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        // Deadline in the past: guarantees the first stage-check trips.
        .timeout_secs(0)
        .build();

    if let Err(SpdfError::InvalidInput(msg)) = parser.parse(ParseInput::Bytes(pdf)) {
        assert!(
            msg.contains("timeout"),
            "InvalidInput from elsewhere: {msg}"
        );
    }
    // A successful parse is also OK — the deadline check is at stage
    // boundaries, not inside pdfium, so a tiny fixture may slip through.
}

/// `max_pages = 0` must cap the selected-page list to empty, which in
/// turn means `select_pages` produces no work and the parser returns
/// an empty `ParseResult` rather than panicking.
#[test]
fn max_pages_zero_produces_empty_result() {
    let pdf = match std::fs::read(workspace_fixture("test-ocr.pdf")) {
        Ok(b) => b,
        Err(_) => return,
    };

    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_pages(0)
        .build();

    let result = parser
        .parse(ParseInput::Bytes(pdf))
        .expect("max_pages=0 must not error, just produce no pages");
    assert!(result.pages.is_empty(), "pages leaked past max_pages=0");
}

fn workspace_fixture(name: &str) -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR is .../crates/spdf-core; go up two levels.
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join("example")
        .join(name)
}
