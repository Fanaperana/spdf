//! The visible page frame.
//!
//! PDFium reports glyph, path, image and annotation positions in the page's
//! user space: bottom-left origin at the MediaBox origin, before `/Rotate`.
//! What a viewer shows is the CropBox (clipped to the MediaBox), whose
//! origin need not be `(0, 0)`. Book and journal PDFs often crop a larger
//! sheet, or place a two-page spread on a page that shows only one half.
//! [`PageFrame`] maps user-space points into the frame's top-left system
//! and tells which text lies entirely outside it (and is never displayed).

use spdf_types::{Image, Ruling, TextItem};

/// Text this far (in points) past the frame edge still counts as visible:
/// glyph boxes routinely overhang the box by a hair.
const EDGE_TOLERANCE: f64 = 1.0;

/// The visible box of a page in PDF user space (bottom-left origin).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PageFrame {
    pub left: f64,
    pub bottom: f64,
    pub right: f64,
    pub top: f64,
}

impl PageFrame {
    /// Build a frame from user-space box edges, normalising swapped edges.
    /// Returns `None` for degenerate or non-finite boxes.
    pub fn new(left: f64, bottom: f64, right: f64, top: f64) -> Option<Self> {
        if ![left, bottom, right, top].iter().all(|v| v.is_finite()) {
            return None;
        }
        let f = PageFrame {
            left: left.min(right),
            bottom: bottom.min(top),
            right: left.max(right),
            top: bottom.max(top),
        };
        (f.right - f.left >= 1.0 && f.top - f.bottom >= 1.0).then_some(f)
    }

    /// Frame of a page whose box starts at the origin.
    pub fn origin(width: f64, height: f64) -> Self {
        PageFrame {
            left: 0.0,
            bottom: 0.0,
            right: width.max(0.0),
            top: height.max(0.0),
        }
    }

    /// Frame-relative, top-down y of a user-space y.
    pub fn y(&self, y: f64) -> f64 {
        self.top - y
    }

    fn width(&self) -> f64 {
        self.right - self.left
    }

    fn height(&self) -> f64 {
        self.top - self.bottom
    }

    /// Shift x coordinates that were flipped with [`PageFrame::y`] but are
    /// still user-space horizontally.
    pub fn shift_items(&self, items: &mut [TextItem]) {
        if self.left == 0.0 {
            return;
        }
        for it in items {
            it.x -= self.left;
        }
    }

    pub fn shift_rulings(&self, rulings: &mut [Ruling]) {
        if self.left == 0.0 {
            return;
        }
        for r in rulings {
            r.x1 -= self.left;
            r.x2 -= self.left;
        }
    }

    pub fn shift_images(&self, images: &mut [Image]) {
        if self.left == 0.0 {
            return;
        }
        for im in images {
            im.x -= self.left;
        }
    }

    /// Whether a frame-relative item box overlaps the visible frame.
    pub fn shows(&self, it: &TextItem) -> bool {
        let (w, h) = (self.width(), self.height());
        let t = EDGE_TOLERANCE;
        it.x < w + t
            && it.x + it.width.max(0.0) > -t
            && it.y < h + t
            && it.y + it.height.max(0.0) > -t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(x: f64, y: f64, w: f64, h: f64) -> TextItem {
        TextItem::new("t", x, y, w, h)
    }

    #[test]
    fn crop_origin_is_subtracted() {
        // A 612x792 sheet cropped to [83, 96, 515, 744].
        let f = PageFrame::new(83.16, 96.264, 515.088, 744.294).unwrap();
        // A glyph at the crop box's top-left corner maps to (0, 0).
        assert!((f.y(744.294)).abs() < 1e-9);
        let mut items = vec![item(144.0, 10.0, 20.0, 10.0)];
        f.shift_items(&mut items);
        assert!((items[0].x - (144.0 - 83.16)).abs() < 1e-9);
    }

    #[test]
    fn swapped_and_degenerate_boxes() {
        let f = PageFrame::new(100.0, 50.0, 0.0, 0.0).unwrap();
        assert_eq!(f, PageFrame::new(0.0, 0.0, 100.0, 50.0).unwrap());
        assert!(PageFrame::new(0.0, 0.0, 0.0, 10.0).is_none());
        assert!(PageFrame::new(f64::NAN, 0.0, 10.0, 10.0).is_none());
    }

    #[test]
    fn text_beyond_the_frame_is_hidden() {
        // Right half of a two-page spread drawn past a 510pt wide page.
        let f = PageFrame::origin(510.0, 737.0);
        assert!(f.shows(&item(76.0, 357.0, 60.0, 11.0)));
        assert!(!f.shows(&item(586.0, 55.0, 80.0, 11.0)));
        assert!(!f.shows(&item(-435.0, 55.0, 300.0, 11.0)));
        assert!(!f.shows(&item(40.0, -30.0, 30.0, 10.0)));
        assert!(!f.shows(&item(40.0, 800.0, 30.0, 10.0)));
        // Partly visible text and hairline overhangs stay.
        assert!(f.shows(&item(480.0, 100.0, 60.0, 11.0)));
        assert!(f.shows(&item(510.5, 100.0, 5.0, 11.0)));
        assert!(f.shows(&item(10.0, -10.5, 30.0, 10.0)));
    }
}
