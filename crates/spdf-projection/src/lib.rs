//! Spatial text reconstruction.
//!
//! Reconstructs readable text from PDFium's raw glyph-level extraction,
//! approximating the column-aware layout that `liteparse`'s
//! `src/processing/gridProjection.ts` produces from PDF.js word-level runs.
//!
//! ## Pipeline
//!
//! 1. **Drop placeholders / empty strings.**
//! 2. **Dedup faux-bold shadow glyphs** — PDFs often draw text twice at a
//!    slight offset to simulate bold. PDFium surfaces both copies. We detect
//!    and drop the shadow by looking for overlapping boxes with identical
//!    text (liteparse/gridProjection.ts: "same text rendered twice").
//! 3. **Merge continuous glyphs.** Adjacent items on the same baseline with
//!    touching or sub-pixel-overlapping boxes are concatenated (port of
//!    liteparse `canMerge`: y/h equal and xDelta ∈ [-1.0, 0.1]). This folds
//!    per-character PDFium runs back into words.
//! 4. **Group into rows** using a baseline-band clustering anchored to the
//!    page's median glyph height. Tall decorative items (vertical bars,
//!    rules) cannot stretch a row.
//! 5. **Merge rows vertically** if their x-ranges are disjoint and their
//!    y-extents overlap (liteparse final-pass line merger).
//! 6. **Render each row** onto a character grid anchored at the page's left
//!    edge. Gaps are classified: very-wide → column padding; narrow → space;
//!    tight kerning → concatenate.
//! 7. **Tidy layout**: strip trailing whitespace, collapse runs of blank
//!    lines, keep internal column padding.

#![warn(clippy::all)]

use spdf_processing::markup::apply_markup_tags;
use spdf_types::{ParseConfig, ParsedPage, TextItem};
use tracing::trace;

/// Per-page layout reconstruction input.
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

/// Project a single page using clustering + grid layout + duplicate
/// suppression.
pub fn project_page(page: PageInput, debug: bool) -> ParsedPage {
    let PageInput {
        page_num,
        width,
        height,
        mut text_items,
    } = page;

    // Drop empty/placeholder items upfront.
    text_items.retain(|t| !t.str.is_empty() && !t.is_placeholder.unwrap_or(false));
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

    // Sort by (y, x) for deterministic ordering.
    text_items.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));

    // Drop faux-bold shadow duplicates (same text, overlapping boxes).
    deduplicate_shadows(&mut text_items);

    // Stitch per-character runs back into words (xDelta tight + same baseline).
    merge_continuous_runs(&mut text_items);

    // Re-sort after mutation to keep deterministic order.
    text_items.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));

    // Page-level metrics used for clustering + rendering.
    let median_h = median_height(&text_items).max(1.0);
    let char_unit = estimate_char_unit(&text_items);
    let page_left = text_items
        .iter()
        .map(|it| it.x)
        .fold(f64::INFINITY, f64::min);

    // Cluster into rows by baseline band.
    let rows = cluster_rows(&text_items, median_h);

    // Vertically merge rows whose x-ranges are disjoint (multi-column
    // sections often produce "visual one line" spanning columns).
    let rows = merge_overlapping_rows(rows, &text_items);

    if debug {
        trace!(
            page = page_num,
            rows = rows.len(),
            char_unit = char_unit,
            median_h = median_h,
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

// ---------------------------------------------------------------------------
// Shadow dedup
// ---------------------------------------------------------------------------

/// Remove faux-bold shadow copies: items with identical text whose bboxes
/// share any x-overlap and nearly-same midline. The survivor's bounds are
/// extended to the union of both copies so the downstream gap calculation
/// doesn't produce artificial whitespace between the survivor and the next
/// glyph (port of liteparse's `mergePageBbox`).
fn deduplicate_shadows(items: &mut Vec<TextItem>) {
    let n = items.len();
    let mut keep = vec![true; n];
    // `union_into[j] = Some(i)` means glyph j should be merged into i.
    let mut merges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !keep[j] {
                continue;
            }
            if items[i].str != items[j].str {
                continue;
            }
            let hi = items[i].height.max(1.0);
            let hj = items[j].height.max(1.0);
            let mid_i = items[i].y + hi * 0.5;
            let mid_j = items[j].y + hj * 0.5;
            let h_max = hi.max(hj);
            if (mid_i - mid_j).abs() > h_max * 0.75 {
                continue;
            }
            let a_left = items[i].x;
            let a_right = items[i].x + items[i].width;
            let b_left = items[j].x;
            let b_right = items[j].x + items[j].width;
            let x_overlap = a_right.min(b_right) - a_left.max(b_left);
            if x_overlap < 0.0 {
                continue;
            }
            // Keep the taller (main) copy; union its bounds with the shadow.
            if hj < hi - 1e-6 {
                merges.push((i, j));
                keep[j] = false;
            } else if hi < hj - 1e-6 {
                merges.push((j, i));
                keep[i] = false;
                break;
            } else {
                merges.push((i, j));
                keep[j] = false;
            }
        }
    }
    // Apply bbox unions. Each merge (keeper, dropped).
    for (k, d) in merges {
        let d_left = items[d].x;
        let d_right = items[d].x + items[d].width;
        let k_left = items[k].x;
        let k_right = items[k].x + items[k].width;
        let new_left = k_left.min(d_left);
        let new_right = k_right.max(d_right);
        items[k].x = new_left;
        items[k].width = new_right - new_left;
    }
    let mut idx = 0;
    items.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

// ---------------------------------------------------------------------------
// Continuous-run merge (liteparse `canMerge`)
// ---------------------------------------------------------------------------

/// Merge adjacent items that share a baseline and touch horizontally. This
/// fuses per-glyph extraction into word-level runs.
///
/// Liteparse's rule: same y, same h, `xDelta ∈ [-1.0, 0.1]`. We relax slightly
/// because PDFium reports y/h with sub-pixel jitter.
fn merge_continuous_runs(items: &mut Vec<TextItem>) {
    if items.len() < 2 {
        return;
    }
    // Sort row-primary, column-secondary using a loose y-band so adjacent
    // glyphs merge regardless of micro-jitter.
    items.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));

    let mut i = 1;
    while i < items.len() {
        let (prev, curr) = {
            let (l, r) = items.split_at_mut(i);
            (&mut l[i - 1], &r[0])
        };
        let same_baseline = (prev.y - curr.y).abs() < 0.5
            && (prev.height - curr.height).abs() < 0.5;
        if same_baseline {
            let x_delta = curr.x - prev.x - prev.width;
            // Tight-touch window: overlap up to ~1pt, gap up to ~0.1pt.
            if (-1.0..=0.1).contains(&x_delta) {
                let new_right = (curr.x + curr.width).max(prev.x + prev.width);
                prev.width = (new_right - prev.x).max(prev.width);
                prev.str.push_str(&curr.str);
                items.remove(i);
                continue;
            }
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Row clustering
// ---------------------------------------------------------------------------

type Row = Vec<usize>;

/// Cluster items into rows using baseline midline bands. Band width is
/// anchored to the page's median glyph height so a single tall item (like a
/// vertical bar) cannot swallow neighbouring rows.
fn cluster_rows(items: &[TextItem], median_h: f64) -> Vec<Row> {
    // Each item's midline = y + h/2. Two items are in the same row if
    // their midlines are within `band`.
    let band = (median_h * 0.6).max(3.0);

    // Collect (idx, mid) and sort by mid.
    let mut pairs: Vec<(usize, f64)> = items
        .iter()
        .enumerate()
        .map(|(i, it)| (i, it.y + it.height.max(1.0) * 0.5))
        .collect();
    pairs.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut rows: Vec<Row> = Vec::new();
    let mut row_mid_sum: Vec<f64> = Vec::new();
    let mut row_count: Vec<f64> = Vec::new();

    for (idx, mid) in pairs {
        let row = rows
            .iter()
            .enumerate()
            .find(|(r, _)| (row_mid_sum[*r] / row_count[*r] - mid).abs() < band)
            .map(|(r, _)| r);
        match row {
            Some(r) => {
                rows[r].push(idx);
                row_mid_sum[r] += mid;
                row_count[r] += 1.0;
            }
            None => {
                rows.push(vec![idx]);
                row_mid_sum.push(mid);
                row_count.push(1.0);
            }
        }
    }

    // Sort rows top-to-bottom by average midline.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| {
        let ma = row_mid_sum[a] / row_count[a];
        let mb = row_mid_sum[b] / row_count[b];
        ma.total_cmp(&mb)
    });
    let mut out: Vec<Row> = order.into_iter().map(|i| rows[i].clone()).collect();

    // Within each row, sort by x.
    for row in out.iter_mut() {
        row.sort_by(|&a, &b| items[a].x.total_cmp(&items[b].x));
    }
    out
}

/// Merge consecutive rows whose y-ranges overlap AND whose x-footprints are
/// disjoint. This is liteparse's final-pass line merge for cases where a
/// split-baseline row was prematurely broken apart.
fn merge_overlapping_rows(mut rows: Vec<Row>, items: &[TextItem]) -> Vec<Row> {
    let mut i = 1;
    while i < rows.len() {
        let prev_y = y_range(&rows[i - 1], items);
        let cur_y = y_range(&rows[i], items);
        let y_overlap = prev_y.1 > cur_y.0 && prev_y.0 < cur_y.1;
        if y_overlap {
            let x_collide = rows_x_collide(&rows[i - 1], &rows[i], items);
            if !x_collide {
                let cur = std::mem::take(&mut rows[i]);
                rows[i - 1].extend(cur);
                rows[i - 1].sort_by(|&a, &b| items[a].x.total_cmp(&items[b].x));
                rows.remove(i);
                continue;
            }
        }
        i += 1;
    }
    rows
}

fn y_range(row: &Row, items: &[TextItem]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &idx in row {
        let it = &items[idx];
        lo = lo.min(it.y);
        hi = hi.max(it.y + it.height.max(1.0));
    }
    (lo, hi)
}

fn rows_x_collide(a: &Row, b: &Row, items: &[TextItem]) -> bool {
    for &i in a {
        let ia = &items[i];
        let ax0 = ia.x;
        let ax1 = ia.x + ia.width;
        for &j in b {
            let ib = &items[j];
            let bx0 = ib.x;
            let bx1 = ib.x + ib.width;
            let ov = ax1.min(bx1) - ax0.max(bx0);
            if ov > 1.0 {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

fn median_height(items: &[TextItem]) -> f64 {
    let mut h: Vec<f64> = items
        .iter()
        .filter(|it| it.height > 0.0)
        .map(|it| it.height)
        .collect();
    if h.is_empty() {
        return 10.0;
    }
    h.sort_by(|a, b| a.total_cmp(b));
    h[h.len() / 2]
}

/// Median per-glyph width, clamped to [3, 12] pt.
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
    widths[widths.len() / 2].clamp(3.0, 12.0)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn render_row_grid(
    items: &[TextItem],
    row: &Row,
    char_unit: f64,
    page_left: f64,
    out: &mut String,
) {
    let row_start = out.chars().count();
    let mut prev_right: Option<f64> = None;
    // A page-median char width gives a stable, font-agnostic threshold.
    // Narrow glyphs like "I" or "/" have tiny per-item widths that would
    // otherwise make every neighbour gap look like a word break.
    let space_gap = char_unit * 0.55;
    let column_gap = char_unit * 2.5;

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
            let col = ((it.x - page_left) / char_unit).round().max(0.0) as usize;
            for _ in 0..col {
                out.push(' ');
            }
        } else {
            let pr = prev_right.unwrap_or(it.x);
            let gap_pts = it.x - pr;
            let cur_cols = out.chars().count() - row_start;
            if gap_pts > column_gap {
                // Wide gap: pad to the item's absolute column.
                let col = ((it.x - page_left) / char_unit).round().max(0.0) as usize;
                if cur_cols < col {
                    for _ in 0..(col - cur_cols) {
                        out.push(' ');
                    }
                } else if !out.ends_with(' ') {
                    out.push(' ');
                }
            } else if gap_pts > space_gap {
                // Word boundary: at most one space.
                let ends_ws = out.chars().last().is_some_and(|c| c.is_whitespace());
                let starts_ws = rendered.chars().next().is_some_and(|c| c.is_whitespace());
                if !ends_ws && !starts_ws {
                    out.push(' ');
                }
            }
            // else: tight kerning within a word → concatenate.
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
    fn tight_kerning_concatenates_without_spaces() {
        // Simulates PDFium emitting per-glyph runs that should fuse.
        let items = vec![
            ti("1", 10.0, 100.0, 3.0, 10.0),
            ti("8", 13.0, 100.0, 3.0, 10.0),
            ti("3", 16.0, 100.0, 3.0, 10.0),
            ti("6", 19.0, 100.0, 3.0, 10.0),
        ];
        let page = PageInput {
            page_num: 1,
            width: 500.0,
            height: 700.0,
            text_items: items,
        };
        let out = project_page(page, false);
        assert_eq!(out.text.trim(), "1836");
    }

    #[test]
    fn faux_bold_shadow_is_deduped() {
        // Two copies of "Ta" offset by <1pt — simulated faux-bold. Keep taller.
        let items = vec![
            ti("Ta", 100.0, 50.0, 8.6, 11.0), // main
            ti("Ta", 100.5, 51.5, 6.4, 8.0),  // shadow (shorter)
            ti("x", 109.0, 50.0, 5.0, 11.0),
        ];
        let page = PageInput {
            page_num: 1,
            width: 500.0,
            height: 700.0,
            text_items: items,
        };
        let out = project_page(page, false);
        assert_eq!(out.text.trim(), "Tax");
    }

    #[test]
    fn ascender_and_descender_glyphs_stay_on_same_row() {
        // Uppercase at y=50 h=11, lowercase at y=53 h=8 — classic case.
        let items = vec![
            ti("L", 100.0, 49.7, 6.4, 11.0),
            ti("o", 107.0, 52.8, 7.6, 8.0),
            ti("w", 115.2, 52.9, 11.8, 7.8),
        ];
        let page = PageInput {
            page_num: 1,
            width: 500.0,
            height: 700.0,
            text_items: items,
        };
        let out = project_page(page, false);
        assert_eq!(out.text.trim(), "Low");
    }

    #[test]
    fn reading_order_is_x_sorted_per_row() {
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
