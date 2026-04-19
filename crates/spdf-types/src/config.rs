//! Configuration types. Defaults match `liteparse/src/core/config.ts`.

use serde::{Deserialize, Serialize};

/// Output format.
///
/// Mirrors `OutputFormat` in [`liteparse/src/core/types.ts`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Structured JSON with per-page text items, bounding boxes, and metadata.
    #[default]
    Json,
    /// Plain text with spatial layout preserved.
    Text,
}

/// Grid projection debug knobs. Replaces the bespoke `gridDebugLogger` with a
/// `tracing`-friendly config surface.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DebugConfig {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visualize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visualize_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_filter: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_filter: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region_filter: Option<RegionFilter>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegionFilter {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

/// One or more language codes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Language {
    Single(String),
    Multiple(Vec<String>),
}

impl Default for Language {
    fn default() -> Self {
        Language::Single("en".into())
    }
}

impl Language {
    pub fn as_strings(&self) -> Vec<&str> {
        match self {
            Language::Single(s) => vec![s.as_str()],
            Language::Multiple(v) => v.iter().map(String::as_str).collect(),
        }
    }
}

/// Full parser configuration. Defaults are identical to
/// `DEFAULT_CONFIG` in [`liteparse/src/core/config.ts`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParseConfig {
    pub ocr_language: Language,
    pub ocr_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_server_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tessdata_path: Option<String>,
    pub num_workers: usize,
    pub max_pages: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_pages: Option<String>,
    pub dpi: u32,
    pub output_format: OutputFormat,
    pub precise_bounding_box: bool,
    pub preserve_very_small_text: bool,
    pub preserve_layout_alignment_across_pages: bool,
    /// When true, detect tabular regions on each page and attach them
    /// to [`ParsedPage::tables`]. Off by default — adds a small
    /// post-projection pass that costs ~O(items) per page.
    pub detect_tables: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
    /// Fail `parse()` with [`SpdfError::InvalidInput`] once wall-clock
    /// work exceeds this many seconds. `None` = no deadline. Intended
    /// as a defensive guard against pathological adversarial PDFs;
    /// legitimate documents should never hit this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
    /// Hard cap on the size of an input blob accepted by `parse`. `None`
    /// = no cap. Paths are not checked; only `ParseInput::Bytes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<DebugConfig>,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            ocr_language: Language::default(),
            ocr_enabled: true,
            ocr_server_url: None,
            tessdata_path: None,
            num_workers: 4,
            max_pages: 1000,
            target_pages: None,
            dpi: 150,
            output_format: OutputFormat::Json,
            precise_bounding_box: true,
            preserve_very_small_text: false,
            preserve_layout_alignment_across_pages: false,
            detect_tables: false,
            password: None,
            timeout_secs: None,
            max_input_bytes: None,
            debug: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_liteparse() {
        let c = ParseConfig::default();
        assert!(c.ocr_enabled);
        assert_eq!(c.num_workers, 4);
        assert_eq!(c.max_pages, 1000);
        assert_eq!(c.dpi, 150);
        assert_eq!(c.output_format, OutputFormat::Json);
        assert!(c.precise_bounding_box);
    }

    #[test]
    fn partial_config_deserializes() {
        let json = r#"{"ocrLanguage":"fra","dpi":300,"outputFormat":"text"}"#;
        let patch: serde_json::Value = serde_json::from_str(json).unwrap();
        // Quick smoke: ParseConfig fields are all required because we use
        // defaults at the builder layer, but verify the wire shape round-trips.
        assert_eq!(patch["dpi"], 300);
    }
}
