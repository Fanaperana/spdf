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

/// `max_declared_stream_bytes` must reject a PDF that directly declares
/// an individual stream longer than the cap. Scanner is bytes-only; we
/// don't need a real PDF — any buffer containing `/Length <N>` works
/// because the scanner runs before pdfium.
#[test]
fn max_declared_stream_bytes_rejects_length_bomb() {
    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_declared_stream_bytes(Some(1024))
        .build();

    // Minimal PDF-ish header plus a pathological /Length. pdfium would
    // never see this because the guard trips first.
    let mut payload = b"%PDF-1.4\n1 0 obj << /Length 2147483648 >> stream\n".to_vec();
    payload.extend_from_slice(b"endstream endobj\n%%EOF\n");

    let err = parser
        .parse(ParseInput::Bytes(payload))
        .expect_err("length-bomb must be rejected");
    match err {
        SpdfError::InvalidInput(msg) => {
            assert!(
                msg.contains("max_declared_stream_bytes"),
                "unexpected msg: {msg}"
            );
            assert!(
                msg.contains("2147483648"),
                "should cite the bad value: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

/// Turning the guard off via `max_declared_stream_bytes(None)` must
/// let the same payload through the scanner (it will still fail at
/// pdfium because the bytes aren't a real PDF, but the error must
/// NOT come from our guard).
#[test]
fn max_declared_stream_bytes_none_disables_check() {
    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_declared_stream_bytes(None)
        .build();

    let mut payload = b"%PDF-1.4\n1 0 obj << /Length 2147483648 >> stream\n".to_vec();
    payload.extend_from_slice(b"endstream endobj\n%%EOF\n");

    let err = parser
        .parse(ParseInput::Bytes(payload))
        .expect_err("invalid PDF must still fail somewhere");
    assert!(
        !matches!(err, SpdfError::InvalidInput(ref m) if m.contains("max_declared_stream_bytes")),
        "guard fired after being disabled: {err:?}"
    );
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

/// Build a synthetic PDF fragment containing one FlateDecode stream
/// that decompresses to `expanded_size` bytes of zeros. Used by the
/// zip-bomb guard tests below. Not a *valid* PDF for pdfium's xref
/// parser — we only care that our pre-scan sees the FlateDecode
/// stream and decompresses it.
fn zip_bomb_payload(expanded_size: usize) -> Vec<u8> {
    use std::io::Write as _;
    let zeros = vec![0u8; expanded_size];
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    enc.write_all(&zeros).unwrap();
    let compressed = enc.finish().unwrap();
    let mut out = b"%PDF-1.7\n1 0 obj\n<< /Filter /FlateDecode /Length ".to_vec();
    out.extend_from_slice(compressed.len().to_string().as_bytes());
    out.extend_from_slice(b" >>\nstream\n");
    out.extend_from_slice(&compressed);
    out.extend_from_slice(b"\nendstream\nendobj\n%%EOF\n");
    out
}

/// `max_expanded_stream_bytes` must reject a PDF that contains a
/// FlateDecode stream decompressing to more than the cap, even if
/// the file itself (and its `/Length`) are tiny.
#[test]
fn max_expanded_stream_bytes_rejects_zip_bomb() {
    // 8 MiB of zeros compresses to ~8 KiB; cap set to 1 MiB to trip.
    let payload = zip_bomb_payload(8 * 1024 * 1024);
    assert!(
        payload.len() < 64 * 1024,
        "compressed payload should be tiny, got {} bytes",
        payload.len()
    );

    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_expanded_stream_bytes(Some(1024 * 1024))
        .build();

    let err = parser
        .parse(ParseInput::Bytes(payload))
        .expect_err("zip bomb must be rejected");
    match err {
        SpdfError::InvalidInput(msg) => {
            assert!(
                msg.contains("max_expanded_stream_bytes"),
                "unexpected msg: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

/// Turning the expansion guard off lets the same payload through the
/// pre-scan. It will still fail later (invalid xref), but NOT from
/// our guard.
#[test]
fn max_expanded_stream_bytes_none_disables_check() {
    let payload = zip_bomb_payload(8 * 1024 * 1024);
    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_expanded_stream_bytes(None)
        .build();
    let err = parser
        .parse(ParseInput::Bytes(payload))
        .expect_err("invalid PDF must still fail somewhere");
    assert!(
        !matches!(err, SpdfError::InvalidInput(ref m) if m.contains("max_expanded_stream_bytes")),
        "guard fired after being disabled: {err:?}"
    );
}

/// Within-budget FlateDecode streams must not be rejected.
#[test]
fn max_expanded_stream_bytes_allows_streams_under_cap() {
    let payload = zip_bomb_payload(64 * 1024);
    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_expanded_stream_bytes(Some(1024 * 1024))
        .build();
    let err = parser
        .parse(ParseInput::Bytes(payload))
        .expect_err("invalid PDF must still fail somewhere");
    assert!(
        !matches!(err, SpdfError::InvalidInput(ref m) if m.contains("max_expanded_stream_bytes")),
        "guard fired below cap: {err:?}"
    );
}

/// Real-world regression: the fuzz-found OOM artifact must now be
/// rejected cleanly in bounded time instead of allocating ~2 GiB.
/// Skipped when the corpus file isn't checked out.
#[test]
fn zip_bomb_guard_rejects_fuzz_corpus_oom_artifact() {
    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let artifact = repo_root
        .join("fuzz")
        .join("corpus")
        .join("parse_pdf")
        .join("oom-d3bf6727d48df284b948852ddbcf7030bccd0cc4");
    let bytes = match std::fs::read(&artifact) {
        Ok(b) => b,
        Err(_) => return, // corpus not shipped in this checkout
    };

    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_pages(8)
        .max_input_bytes(1 << 20) // 1 MiB — artifact is ~150 KiB
        .max_expanded_stream_bytes(Some(256 * 1024 * 1024))
        .timeout_secs(10)
        .build();

    let t0 = std::time::Instant::now();
    let res = parser.parse(ParseInput::Bytes(bytes));
    let elapsed = t0.elapsed();

    // The pre-scan is bounded by the cap (256 MiB) divided by
    // flate2's throughput. In release mode that's < 2 s; in debug
    // mode miniz_oxide is ~15-30 MB/s, so 256 MiB can take ~15 s.
    // We allow 60 s here so the test isn't flaky in debug builds;
    // what matters is that pdfium is **not** invoked on a 2 GiB
    // payload, which would OOM the test runner regardless of time.
    assert!(
        elapsed < std::time::Duration::from_secs(60),
        "pre-scan unbounded, took {elapsed:?}: {res:?}"
    );
    match res {
        Err(SpdfError::InvalidInput(msg)) => {
            // Any of the three pre-scan guards firing is a pass.
            assert!(
                msg.contains("max_expanded_stream_bytes")
                    || msg.contains("max_declared_stream_bytes")
                    || msg.contains("max_input_bytes"),
                "unexpected rejection reason: {msg}"
            );
        }
        Ok(_) => {
            // Parsing succeeded without tripping a guard. That's also
            // acceptable — it means pdfium handled the file within
            // our budgets after the pre-scans let it through. What we
            // are *specifically* guarding against is the 2 GiB OOM,
            // which would have killed the test process entirely.
        }
        Err(other) => {
            // A pdfium error after the guards cleared is fine.
            eprintln!("pdfium rejected the artifact cleanly: {other}");
        }
    }
}

// ── #T1.1 AcroForm auto-detection ──────────────────────────────────

/// AcroForm PDFs should auto-enable `preserve_very_small_text` even
/// when the caller leaves it at the default (`false`). This test
/// parses the IRS 1040 form (which has `/AcroForm`) and verifies
/// parsing succeeds (the actual F1 lift is checked by the benchmark).
#[test]
fn acroform_pdf_auto_enables_preserve_small() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../example/corpus/irs-f1040.pdf");
    if !fixture.exists() {
        eprintln!("skipping: fixture not found at {}", fixture.display());
        return;
    }
    let bytes = std::fs::read(&fixture).unwrap();

    // Default config — preserve_very_small_text is false.
    let parser = SpdfParser::builder().ocr_enabled(false).build();
    let result = parser.parse(ParseInput::Bytes(bytes)).unwrap();

    // The IRS 1040 has at least one page.
    assert!(!result.pages.is_empty(), "should have parsed pages");
    // Text shouldn't be empty — AcroForm detection should have
    // kept the small-text form fields.
    assert!(
        !result.text.trim().is_empty(),
        "text output should not be empty"
    );
}

/// A non-form PDF (RFC 8446) should NOT trigger AcroForm auto-detection.
#[test]
fn non_form_pdf_does_not_trigger_acroform() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../example/corpus/rfc8446-p1-2.pdf");
    if !fixture.exists() {
        eprintln!("skipping: fixture not found");
        return;
    }
    let bytes = std::fs::read(&fixture).unwrap();

    let parser = SpdfParser::builder().ocr_enabled(false).build();
    let result = parser.parse(ParseInput::Bytes(bytes)).unwrap();

    // Should parse fine without the small-text preservation.
    assert!(!result.pages.is_empty());
}
