//! Heuristics for deciding when OCR needs to run.
//!
//! Turns raw OCR detections into [`TextItem`]s with PDF-space coordinates and
//! filters out detections that duplicate native text or fall below a
//! confidence floor.

use spdf_types::{OcrData, TextItem};

use crate::bbox::{Rect, ocr_overlaps_text};

/// Detections below this confidence are dropped. Matches liteparse's
/// `OCR_CONFIDENCE_THRESHOLD`.
pub const OCR_CONFIDENCE_THRESHOLD: f64 = 0.1;

/// Linear transform from image-space to PDF-space.
#[derive(Debug, Clone, Copy)]
pub struct OcrTransform {
    pub scale_x: f64,
    pub scale_y: f64,
    pub offset_x: f64,
    pub offset_y: f64,
}

impl Default for OcrTransform {
    fn default() -> Self {
        Self {
            scale_x: 1.0,
            scale_y: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }
}

/// Convert OCR detections into [`TextItem`]s after confidence + overlap
/// filtering.
///
/// * `native_text_rects` is the list of rectangles already present in the
///   native PDF text layer — used to suppress duplicate detections.
pub fn ocr_to_text_items(
    ocr: &[OcrData],
    xform: OcrTransform,
    native_text_rects: &[Rect],
) -> Vec<TextItem> {
    ocr.iter()
        .filter(|d| d.confidence >= OCR_CONFIDENCE_THRESHOLD)
        .filter(|d| !d.text.trim().is_empty())
        .filter_map(|d| {
            let x = d.x * xform.scale_x + xform.offset_x;
            let y = d.y * xform.scale_y + xform.offset_y;
            let w = d.w * xform.scale_x;
            let h = d.h * xform.scale_y;
            if !native_text_rects.is_empty()
                && ocr_overlaps_text(Rect { x, y, w, h }, native_text_rects)
            {
                return None;
            }
            let cleaned = crate::text_utils::clean_ocr_table_artifacts(&d.text);
            if cleaned.is_empty() {
                return None;
            }
            let mut item = TextItem::new(cleaned, x, y, w, h);
            item.confidence = Some((d.confidence * 1000.0).round() / 1000.0);
            item.font_name = Some("OCR".into());
            Some(item)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(text: &str, x: f64, y: f64, w: f64, h: f64, c: f64) -> OcrData {
        OcrData {
            text: text.into(),
            x,
            y,
            w,
            h,
            confidence: c,
        }
    }

    #[test]
    fn drops_low_confidence() {
        let ocr = vec![mk("hi", 0.0, 0.0, 10.0, 10.0, 0.05)];
        assert!(ocr_to_text_items(&ocr, OcrTransform::default(), &[]).is_empty());
    }

    #[test]
    fn applies_scale_and_offset() {
        let ocr = vec![mk("hi", 10.0, 20.0, 5.0, 5.0, 0.9)];
        let xform = OcrTransform {
            scale_x: 2.0,
            scale_y: 3.0,
            offset_x: 1.0,
            offset_y: 2.0,
        };
        let out = ocr_to_text_items(&ocr, xform, &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].x, 21.0);
        assert_eq!(out[0].y, 62.0);
        assert_eq!(out[0].w, 10.0);
        assert_eq!(out[0].h, 15.0);
        assert_eq!(out[0].font_name.as_deref(), Some("OCR"));
    }

    #[test]
    fn drops_overlap_with_native_text() {
        let ocr = vec![mk("hi", 0.0, 0.0, 10.0, 10.0, 0.9)];
        let native = vec![Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        }];
        assert!(ocr_to_text_items(&ocr, OcrTransform::default(), &native).is_empty());
    }

    #[test]
    fn keeps_non_overlapping_ocr() {
        let ocr = vec![mk("hi", 100.0, 100.0, 10.0, 10.0, 0.9)];
        let native = vec![Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        }];
        assert_eq!(
            ocr_to_text_items(&ocr, OcrTransform::default(), &native).len(),
            1
        );
    }
}
