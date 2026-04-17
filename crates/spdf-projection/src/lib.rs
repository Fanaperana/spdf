//! Spatial text reconstruction.
//!
//! Groups text items into rows by baseline overlap, then lays each row onto a
//! character grid so that the output preserves the document's horizontal
//! layout (columns, tables, leading indentation). Also deduplicates the
//! overlapping "shadow" glyphs that PDFium emits for faux-bold text.
//!
//! This is a pragmatic subset of `liteparse/src/processing/gridProjection.ts`.
//! It captures correct reading order, column preservation, and duplicate-run
//! suppression — enough for most paragraphs and simple tables. Anchor-based
//! alignment and full table inference are future work.

#![warn(clippy::all)]

use spdf_processing::markup::apply_markup_tags;
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

/// Project a single page using row grouping + grid layout + duplicate
/// suppression.
pub fn project_page(page: PageInput, debug: bool) -> ParsedPage {
    let PageInput {
        page_num,
        width,
        height,
        mut text_items,
    } = page;

    // Drop empty/placeholder items upfront.
    text_items.retain(|t| !t.str.trim().is_empty() && !t.is_placeholder.unwrap_or(false));
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

    // Sort by y then x for deterministic grouping.
    text_items.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));

    // Drop overlapping duplicate glyphs (the faux-bold "shadow" runs PDFium
    // emits). Performed before row grouping so the grid counts unique glyphs.
    deduplicate_overlapping(&mut text_items);

    // Compute a global character unit (one output column = `char_unit` pts).
    // Use the median per-glyph width; fall back to font-size/2 or 6pt.
    let char_unit = estimate_char_unit(&text_items);

    // Anchor indentation relative to the leftmost x across the whole page so
    // rows keep their relative column positions.
    let page_left = text_items
        .iter()
        .map(|it| it.x)
        .fold(f64::INFINITY, f64::min);

    let rows = group_into_rows(&text_items);
    if debug {
        trace!(
            page = page_num,
            rows = rows.len(),
            char_unit = char_unit,
            "spdf: projection rows"
        );
    }

    // Render each row onto a fixed-width character grid.
    let mut out = String::new();
    for (row_idx, row) in rows.iter().enumerate() {
        if row_idx > 0 {
            out.push('\n');
        }
        render_row_grid(&text_items, row, char_unit, page_left, &mut out);
    }

    // Light cleanup: strip trailing whitespace per line and collapse more than
    // one consecutive blank line. We intentionally preserve intra-line runs of
    // spaces so column alignment survives.
    let text = tidy_layout(&out);

    ParsedPage {
        page_num,
        width,
        height,
        text,
        text_items,
        bounding_boxes: None,
    }
}

/// Remove items whose bounding box significantly overlaps a previously kept
/// item with the same text. This fixes the "TaTax Infofo" duplication where
/// PDFium emits a glyph twice (main + offset shadow) to simulate bold.
///
/// We consider two items "duplicates" when:
///   1. Their strings are identical, AND
///   2. Their rectangles overlap by more than 50% of the smaller area, AND
///   3. They are on the same baseline.
fn deduplicate_overlapping(items: &mut Vec<TextItem>) {
    let mut keep: Vec<bool> = vec![true; items.len()];
    for i in 0..items.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..items.len() {
            if !keep[j] {
                continue;
            }
            if items[i].str != items[j].str {
                continue;
            }
            let h = items[i].height.max(items[j].height).max(1.0);
            if (items[i].y - items[j].y).abs() > h * 0.5 {
                continue;
            }
            let overlap = rect_overlap_area(
                items[i].x,
                items[i].y,
                items[i].width,
                items[i].height,
                items[j].x,
                items[j].y,
                items[j].width,
                items[j].height,
            );
            let a = (items[i].width * items[i].height).max(1.0);
            let b = (items[j].width * items[j].height).max(1.0);
            if overlap / a.min(b) > 0.5 {
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    items.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

fn rect_overlap_area(
    ax: f64,
    ay: f64,
    aw: f64,
    ah: f64,
    bx: f64,
    by: f64,
    bw: f64,
    bh: f64,
) -> f64 {
    let left = ax.max(bx);
    let right = (ax + aw).min(bx + bw);
    let top = ay.max(by);
    let bot = (ay + ah).min(by + bh);
    if left >= right || top >= bot {
        0.0
    } else {
        (right - left) * (bot - top)
    }
}

/// Median per-glyph width across all items, clamped to a sensible range.
fn estimate_char_unit(items: &[TextItem]) -> f64 {
    let mut widths: Vec<f64> = items
        .iter()
        .filter_map(|it| {
            let n = it.str.chars().count();
            if n == 0 || it.width <= 0.0 {
                None
            } else {
                Some(it.width / n as f64)
            }
        })
        .collect();
    if widths.is_empty() {
        return 6.0;
    }
    widths.sort_by(|a, b| a.total_cmp(b));
    let median = widths[widths.len() / 2];
    // Clamp to [3, 12] pt — wide enough to avoid glyph collisions, narrow
    // enough that column padding stays legible.
    median.clamp(3.0, 12.0)
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

fn render_row_grid(
    items: &[TextItem],
    row: &Row,
    char_unit: f64,
    page_left: f64,
    out: &mut String,
) {
    let row_start = out.chars().count();
    let mut prev_right: Option<f64> = None;

    for (pos, &idx) in row.iter().enumerate() {
        let it = &items[idx];
        let base = it.str.as_str();
        if base.is_empty() {
            continue;
        }
        let rendered = match &it.markup {
            Some(m) => apply_markup_tags(m, base),
            None => base.to_owned(),
        };

        if pos == 0 {
            // Indent to the item's absolute column on the page.
            let col = ((it.x - page_left) / char_unit).round().max(0.0) as usize;
            for _ in 0..col {
                out.push(' ');
            }
        } else {
            let pr = prev_right.unwrap_or(it.x);
            let gap_pts = it.x - pr;
            // A gap wider than ~1.5 char units is treated as a column break:
            // pad to the exact column the item lives in. Smaller gaps emit
            // either a single space (word break) or nothing (tight kerning).
            let cur_cols = out.chars().count() - row_start;
            if gap_pts > char_unit * 1.5 {
                let col = ((it.x - page_left) / char_unit).round().max(0.0) as usize;
                if cur_cols < col {
                    for _ in 0..(col - cur_cols) {
                        out.push(' ');
                    }
                } else if !out.ends_with(' ') {
                    out.push(' ');
                }
            } else if gap_pts > char_unit * 0.3 {
                let ends_ws = out.chars().last().is_some_and(|c| c.is_whitespace());
                let starts_ws = rendered.chars().next().is_some_and(|c| c.is_whitespace());
                if !ends_ws && !starts_ws {
                    out.push(' ');
                }
            }
            // else: tight kerning → concatenate directly.
        }
        out.push_str(&rendered);
        prev_right = Some(it.x + it.width.max(0.0));
    }
}

/// Strip trailing whitespace per line and collapse ≥2 consecutive blank lines
/// into one. Preserves intra-line column padding.
fn tidy_layout(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut blank_streak = 0usize;
    for line in input.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            blank_streak += 1;
            if blank_streak <= 1 {
                out.push('\n');
            }
            continue;
        }
        blank_streak = 0;
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(trimmed);
    }
    out
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
