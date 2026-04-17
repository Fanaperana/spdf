//! Spatial text reconstruction.
//!
//! This is a pragmatic, ~200-line port of the ~2200-line
//! `liteparse/src/processing/gridProjection.ts`. It captures the core
//! behaviour that matters for correct reading order:
//!
//!   1. Group text items into rows by baseline overlap.
//!   2. Within each row, sort left-to-right.
//!   3. Insert spaces when the gap between neighbours exceeds roughly one
//!      glyph width; otherwise concatenate directly.
//!   4. Emit one row per line.
//!
//! Full table/column detection and anchor-based projection are tracked as
//! future work but not required for correct paragraph reading order.

#![warn(clippy::all)]

use spdf_processing::clean_text::clean_raw_text;
use spdf_types::{ParseConfig, ParsedPage, TextItem};
use tracing::trace;

/// Per-page layout reconstruction input. Mirrors `PageData` in the PDF engine.
#[derive(Debug, Clone)]
pub struct PageInput {
    pub page_num: u32,
    pub width: f64,
    pub height: f64,
    pub text_items: Vec<TextItem>,
}

/// Project a batch of pages to their final text + layout.
pub fn project_pages_to_grid(pages: Vec<PageInput>, config: &ParseConfig) -> Vec<ParsedPage> {
    let debug = config.debug.as_ref().is_some_and(|d| d.enabled);
    pages
        .into_iter()
        .map(|p| project_page(p, debug))
        .collect()
}

/// Project a single page using row grouping + in-row ordering + spacing.
pub fn project_page(page: PageInput, debug: bool) -> ParsedPage {
    let PageInput {
        page_num,
        width,
        height,
        mut text_items,
    } = page;

    // Skip empty items upfront.
    text_items.retain(|t| !t.str.trim().is_empty());
    if text_items.is_empty() {
        return ParsedPage {
            page_num,
            width,
            height,
            text: String::new(),
            text_items,
            bounding_boxes: None,
        };
    }

    // Sort stably by y then x to make grouping deterministic.
    text_items.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));

    // Group into rows by vertical overlap.
    let rows = group_into_rows(&text_items);
    if debug {
        trace!(page = page_num, rows = rows.len(), "spdf: projection rows");
    }

    // Build output text row by row.
    let mut out = String::new();
    for (row_idx, row) in rows.iter().enumerate() {
        if row_idx > 0 {
            out.push('\n');
        }
        render_row(&text_items, row, &mut out);
    }
    let text = clean_raw_text(&out);

    ParsedPage {
        page_num,
        width,
        height,
        text,
        text_items,
        bounding_boxes: None,
    }
}

/// A row is a list of text-item indices that share a baseline band.
type Row = Vec<usize>;

fn group_into_rows(items: &[TextItem]) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    // Track each row's accumulated vertical extent so we can decide when a new
    // item extends vs starts a new row.
    let mut extents: Vec<(f64, f64)> = Vec::new(); // (top, bottom)

    for (i, it) in items.iter().enumerate() {
        let top = it.y;
        let bottom = it.y + it.height.max(1.0);
        let mid = (top + bottom) * 0.5;
        let h = (bottom - top).max(1.0);

        // Find an existing row whose midline overlaps this item by at least
        // ~50% of the item's height. That tolerance absorbs superscripts,
        // subscripts, and tight leading without bleeding into the next line.
        let matched = rows.iter().zip(extents.iter()).position(|(_, (rt, rb))| {
            let row_mid = (rt + rb) * 0.5;
            (row_mid - mid).abs() < h * 0.5
        });
        match matched {
            Some(idx) => {
                rows[idx].push(i);
                let (rt, rb) = &mut extents[idx];
                *rt = rt.min(top);
                *rb = rb.max(bottom);
            }
            None => {
                rows.push(vec![i]);
                extents.push((top, bottom));
            }
        }
    }

    // Order rows top-to-bottom by their midline.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| {
        let ma = (extents[a].0 + extents[a].1) * 0.5;
        let mb = (extents[b].0 + extents[b].1) * 0.5;
        ma.total_cmp(&mb)
    });

    // Within each row, sort by x.
    let mut out: Vec<Row> = order.into_iter().map(|i| rows[i].clone()).collect();
    for row in out.iter_mut() {
        row.sort_by(|&a, &b| items[a].x.total_cmp(&items[b].x));
    }
    out
}

fn render_row(items: &[TextItem], row: &Row, out: &mut String) {
    let mut prev_right: Option<f64> = None;
    let mut prev_avg_char_w: f64 = 0.0;

    for &idx in row {
        let it = &items[idx];
        let s = it.str.as_str();
        if s.is_empty() {
            continue;
        }

        if let Some(pr) = prev_right {
            let gap = it.x - pr;
            // Heuristic: a gap wider than ~0.3 × prev-char-width is a space.
            // Use the larger of this item's and the previous item's
            // estimated char width, so tiny glyphs don't over-space.
            let cur_char_w = char_width(it);
            let threshold = prev_avg_char_w.max(cur_char_w) * 0.3;
            // Only insert a space if the previous chunk didn't already end
            // with whitespace and the current doesn't start with it.
            let ends_ws = out.chars().last().is_some_and(|c| c.is_whitespace());
            let starts_ws = s.chars().next().is_some_and(|c| c.is_whitespace());
            if gap > threshold && !ends_ws && !starts_ws {
                out.push(' ');
            }
        }
        out.push_str(s);
        prev_right = Some(it.x + it.width.max(0.0));
        prev_avg_char_w = char_width(it);
    }
}

fn char_width(it: &TextItem) -> f64 {
    // Approximate per-glyph width. Fallback to the item height (a decent proxy
    // for monospace / narrow glyphs) when we can't compute a sane ratio.
    let n = it.str.chars().count().max(1) as f64;
    let w = it.width.max(0.0);
    if w > 0.0 {
        w / n
    } else {
        it.height.max(1.0) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ti(s: &str, x: f64, y: f64, w: f64, h: f64) -> TextItem {
        TextItem::new(s, x, y, w, h)
    }

    #[test]
    fn single_line_joins_with_spaces_by_gap() {
        let items = vec![
            ti("Hello", 10.0, 100.0, 30.0, 12.0),
            ti("world", 45.0, 100.0, 30.0, 12.0),
        ];
        let page = PageInput {
            page_num: 1,
            width: 500.0,
            height: 700.0,
            text_items: items,
        };
        let out = project_page(page, false);
        assert_eq!(out.text.trim(), "Hello world");
    }

    #[test]
    fn two_rows_separated_by_newline() {
        let items = vec![
            ti("Top", 10.0, 100.0, 20.0, 12.0),
            ti("Bot", 10.0, 140.0, 20.0, 12.0),
        ];
        let page = PageInput {
            page_num: 1,
            width: 500.0,
            height: 700.0,
            text_items: items,
        };
        let out = project_page(page, false);
        let lines: Vec<_> = out.text.lines().map(str::trim).collect();
        assert_eq!(lines, vec!["Top", "Bot"]);
    }

    #[test]
    fn reading_order_is_x_sorted_per_row() {
        // Simulates PDFium returning segments in draw order, not reading order.
        // Glyph x positions are contiguous (tight kerning) so no word-internal
        // gap triggers an unwanted space.
        let items = vec![
            ti("fi", 80.0, 100.0, 8.0, 12.0),
            ti("Dumm", 20.0, 100.0, 26.0, 12.0),
            ti(" PDF ", 52.0, 100.0, 28.0, 12.0),
            ti("le", 88.0, 100.0, 10.0, 12.0),
            ti("y", 46.0, 100.0, 6.0, 12.0),
        ];
        let page = PageInput {
            page_num: 1,
            width: 500.0,
            height: 700.0,
            text_items: items,
        };
        let out = project_page(page, false);
        assert_eq!(out.text.trim(), "Dummy PDF file");
    }
}
