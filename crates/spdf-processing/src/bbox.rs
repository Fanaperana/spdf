//! Port target: `liteparse/src/processing/bbox.ts`.
//! TODO(phase-3): port `buildBoundingBoxes` and friends.

use spdf_types::{BoundingBox, TextItem};

/// Build one [`BoundingBox`] per line from a set of text items.
///
/// Placeholder implementation: wraps each item in its own bbox. The real
/// algorithm merges items onto the same line and collapses touching spans;
/// see the TypeScript source for the full logic.
pub fn build_bounding_boxes(items: &[TextItem]) -> Vec<BoundingBox> {
    items
        .iter()
        .map(|t| BoundingBox {
            x1: t.x,
            y1: t.y,
            x2: t.x + t.width,
            y2: t.y + t.height,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_items_yield_empty_boxes() {
        assert!(build_bounding_boxes(&[]).is_empty());
    }

    #[test]
    fn single_item_yields_single_box() {
        let item = TextItem::new("hi", 10.0, 20.0, 30.0, 12.0);
        let boxes = build_bounding_boxes(&[item]);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].x1, 10.0);
        assert_eq!(boxes[0].x2, 40.0);
    }
}
