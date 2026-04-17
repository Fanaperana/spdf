//! Port target: `liteparse/src/processing/textUtils.ts`.
//! TODO(phase-3): port subscript/superscript normalization and OCR artifact cleanup.

/// Stub of `strToSubscriptString`. Returns the input unchanged for now.
pub fn str_to_subscript_string(s: &str) -> String {
    s.to_owned()
}

/// Stub of `strToPostScript`. Returns the input unchanged for now.
pub fn str_to_post_script(s: &str) -> String {
    s.to_owned()
}

/// Stub of `cleanOcrTableArtifacts`. Passes text through unchanged.
pub fn clean_ocr_table_artifacts(text: &str) -> String {
    text.to_owned()
}
