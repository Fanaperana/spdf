//! Port target: `liteparse/src/processing/ocrUtils.ts`.
//! TODO(phase-3/phase-6): helpers that turn raw OCR results into [`TextItem`]s
//! with correct PDF-space coordinates.

use spdf_types::{OcrData, TextItem};

/// Convert OCR image-space detections into PDF-space [`TextItem`]s.
///
/// Placeholder: applies only a linear scale + offset. The full liteparse
/// implementation also handles rotation correction and garbled-region culling.
pub fn ocr_to_text_items(
    ocr: &[OcrData],
    scale_x: f64,
    scale_y: f64,
    offset_x: f64,
    offset_y: f64,
) -> Vec<TextItem> {
    ocr.iter()
        .map(|d| {
            let mut item = TextItem::new(
                d.text.clone(),
                d.x * scale_x + offset_x,
                d.y * scale_y + offset_y,
                d.w * scale_x,
                d.h * scale_y,
            );
            item.confidence = Some(d.confidence);
            item.font_name = Some("OCR".into());
            item
        })
        .collect()
}
