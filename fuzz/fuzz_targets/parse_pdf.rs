#![no_main]
//! Fuzz the public `SpdfParser::parse(bytes)` entry point with arbitrary
//! byte blobs. The parser should return an `Err` for malformed input
//! rather than panicking, crashing, or running unboundedly.

use libfuzzer_sys::fuzz_target;
use spdf_core::SpdfParser;
use spdf_types::ParseInput;

fuzz_target!(|data: &[u8]| {
    // Cap input size so libfuzzer spends its budget on structural
    // interesting-ness rather than megabytes of noise.
    if data.len() > 512 * 1024 {
        return;
    }

    let parser = SpdfParser::builder()
        .ocr_enabled(false)
        .max_pages(8)
        .max_input_bytes(1 << 20) // 1 MiB
        .timeout_secs(10)
        .build();

    let _ = parser.parse(ParseInput::Bytes(data.to_vec()));
});
