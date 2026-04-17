//! Document / text data model.
//!
//! Serde attributes keep the JSON wire format byte-compatible with
//! `liteparse`'s `ParseResultJson`.

use serde::{Deserialize, Serialize};

/// Markup annotation data associated with a text item.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarkupData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highlight: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub underline: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub squiggly: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strikeout: Option<bool>,
}

/// An individual text element extracted from a page.
///
/// Matches `TextItem` in [`liteparse/src/core/types.ts`]. Coordinates use PDF
/// points, top-left origin, y increasing downward.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextItem {
    pub str: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub w: f64,
    pub h: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ry: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markup: Option<MarkupData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vgap: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_placeholder: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

impl TextItem {
    /// Create a `TextItem` with sensible defaults. `w`/`h` are kept in lockstep
    /// with `width`/`height`, matching liteparse's aliasing convention.
    pub fn new(str: impl Into<String>, x: f64, y: f64, width: f64, height: f64) -> Self {
        let s = str.into();
        Self {
            str: s,
            x,
            y,
            width,
            height,
            w: width,
            h: height,
            font_name: None,
            font_size: None,
            r: None,
            rx: None,
            ry: None,
            markup: None,
            vgap: None,
            is_placeholder: None,
            confidence: None,
        }
    }
}

/// Snap alignment for a projection box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Snap {
    Left,
    Right,
    Center,
}

/// A working copy of a text item used during grid projection. Separate from
/// `TextItem` because it carries algorithm-internal metadata.
#[derive(Debug, Clone)]
pub struct ProjectionTextBox {
    pub str: String,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub rx: Option<f64>,
    pub ry: Option<f64>,
    pub r: Option<i32>,
    pub str_length: usize,
    pub markup: Option<MarkupData>,
    pub page_bbox: Option<Coordinates>,
    pub vgap: bool,
    pub is_placeholder: bool,
    pub from_ocr: bool,

    pub snap: Option<Snap>,
    pub left_anchor: Option<String>,
    pub right_anchor: Option<String>,
    pub center_anchor: Option<String>,
    pub is_dup: bool,
    pub rendered: bool,
    pub is_margin_line_number: bool,
    pub should_space: Option<f64>,
    pub force_unsnapped: bool,
    pub rotated: bool,
    pub d: Option<f64>,
    pub font_name: Option<String>,
    pub font_size: Option<f64>,
    pub confidence: Option<f64>,
}

/// A rectangle defined by position and dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Coordinates {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Axis-aligned bounding box defined by top-left and bottom-right corners.
///
/// Deprecated in the TypeScript source; kept for byte-level JSON parity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundingBox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

/// Raw OCR detection result before conversion to [`TextItem`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrData {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
    pub text: String,
}

/// Parsed data for a single page.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParsedPage {
    pub page_num: u32,
    pub width: f64,
    pub height: f64,
    pub text: String,
    pub text_items: Vec<TextItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounding_boxes: Option<Vec<BoundingBox>>,
}

/// A text element from JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonTextItem {
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// One page of the JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonPage {
    pub page: u32,
    pub width: f64,
    pub height: f64,
    pub text: String,
    pub text_items: Vec<JsonTextItem>,
    pub bounding_boxes: Vec<BoundingBox>,
}

/// Structured JSON output. Returned when `output_format == Json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResultJson {
    pub pages: Vec<JsonPage>,
}

/// The result of parsing a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    pub pages: Vec<ParsedPage>,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<ParseResultJson>,
}

/// Embedded image bounds, used for selective OCR.
#[derive(Debug, Clone)]
pub struct Image {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    /// Decoded image bytes in memory. When present, OCR runs directly on these
    /// bytes; otherwise the engine re-renders the region from the PDF.
    pub data: Option<Vec<u8>>,
    pub scale_factor: Option<f64>,
    pub original_orientation_angle: Option<i32>,
    pub image_type: Option<String>,
}

/// Result of rendering a page to an image.
#[derive(Debug, Clone)]
pub struct ScreenshotResult {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub image_buffer: Vec<u8>,
    pub image_path: Option<std::path::PathBuf>,
}
