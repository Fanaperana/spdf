//! Port of `liteparse/src/processing/bbox.ts` (`buildBoundingBoxes`).

use spdf_types::{BoundingBox, TextItem};

/// Build per-item bounding boxes, skipping whitespace-only items.
///
/// Mirrors liteparse's `buildBoundingBoxes`. The `w`/`h` fields take priority
/// over `width`/`height` when non-zero to match the aliasing convention.
pub fn build_bounding_boxes(items: &[TextItem]) -> Vec<BoundingBox> {
    items
        .iter()
        .filter(|it| !it.str.trim().is_empty())
        .map(|it| {
            let w = if it.w > 0.0 { it.w } else { it.width };
            let h = if it.h > 0.0 { it.h } else { it.height };
            BoundingBox {
                x1: it.x,
                y1: it.y,
                x2: it.x + w,
                y2: it.y + h,
            }
        })
        .collect()
}

/// Rectangle used by overlap helpers.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Return the overlapping area between two rectangles, or `0.0` if disjoint.
/// Mirrors `getOverlapArea` from bbox.ts.
pub fn overlap_area(a: Rect, b: Rect) -> f64 {
    let left = a.x.max(b.x);
    let right = (a.x + a.w).min(b.x + b.w);
    let top = a.y.max(b.y);
    let bottom = (a.y + a.h).min(b.y + b.h);
    if left >= right || top >= bottom {
        0.0
    } else {
        (right - left) * (bottom - top)
    }
}

/// Overlap threshold used to reject OCR blocks that duplicate native text.
pub const OCR_OVERLAP_THRESHOLD: f64 = 0.5;

/// Returns true when an OCR rectangle should be rejected because it duplicates
/// one or more native text rectangles — either the total overlap covers ≥50%
/// of the OCR area or the OCR covers ≥50% of a single text item.
pub fn ocr_overlaps_text(ocr: Rect, text_rects: &[Rect]) -> bool {
    let ocr_area = ocr.w * ocr.h;
    if ocr_area <= 0.0 {
        return true;
    }
    let mut total = 0.0;
    for t in text_rects {
        let o = overlap_area(ocr, *t);
        if o <= 0.0 {
            continue;
        }
        let t_area = t.w * t.h;
        if t_area > 0.0 && o / t_area >= OCR_OVERLAP_THRESHOLD {
            return true;
        }
        total += o;
    }
    total / ocr_area >= OCR_OVERLAP_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_items_yield_empty_boxes() {
        assert!(build_bounding_boxes(&[]).is_empty());
    }

    #[test]
    fn whitespace_items_are_skipped() {
        let items = vec![
            TextItem::new("  ", 0.0, 0.0, 10.0, 10.0),
            TextItem::new("hi", 10.0, 0.0, 10.0, 10.0),
        ];
        assert_eq!(build_bounding_boxes(&items).len(), 1);
    }

    #[test]
    fn single_item_yields_single_box() {
        let item = TextItem::new("hi", 10.0, 20.0, 30.0, 12.0);
        let boxes = build_bounding_boxes(&[item]);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].x1, 10.0);
        assert_eq!(boxes[0].x2, 40.0);
    }

    #[test]
    fn overlap_area_disjoint_is_zero() {
        let a = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        let b = Rect {
            x: 20.0,
            y: 20.0,
            w: 5.0,
            h: 5.0,
        };
        assert_eq!(overlap_area(a, b), 0.0);
    }

    #[test]
    fn overlap_area_partial() {
        let a = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        let b = Rect {
            x: 5.0,
            y: 5.0,
            w: 10.0,
            h: 10.0,
        };
        assert_eq!(overlap_area(a, b), 25.0);
    }

    #[test]
    fn ocr_overlaps_text_rejects_duplicate() {
        let native = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        let ocr = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        assert!(ocr_overlaps_text(ocr, &[native]));
    }

    #[test]
    fn ocr_overlaps_text_allows_independent_region() {
        let native = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        };
        let ocr = Rect {
            x: 100.0,
            y: 100.0,
            w: 10.0,
            h: 10.0,
        };
        assert!(!ocr_overlaps_text(ocr, &[native]));
    }
}
