//! Lightweight table detection over already-projected `TextItem`s.
//!
//! The goal is NOT a faithful DOM-grade table extractor (that's Tier 3
//! territory). It's a best-effort detector that spots blocks of
//! horizontally-aligned rows — the kind of tables found in invoices,
//! receipts, tax forms, and spec appendices — and emits them as
//! structured cells alongside the free-flowing text. Callers that don't
//! need tables pay nothing (gated by `ParseConfig::detect_tables`).
//!
//! Algorithm:
//! 1. Cluster items into rows by y-baseline (same threshold spdf-projection
//!    uses).
//! 2. A row is a *candidate table row* if it has at least 3 items whose
//!    horizontal gaps exceed `GAP_RATIO * median_char_width`.
//! 3. Consecutive candidate rows with matching column signatures
//!    (≥ `MIN_SHARED_COLS` of their item x-centres line up within
//!    `COL_TOLERANCE_PT`) form a table.
//! 4. A table must have at least `MIN_TABLE_ROWS` rows total.
//! 5. Cells with no item in a given column become empty strings.

use spdf_types::{Table, TextItem};

const ROW_BAND_PT: f64 = 2.0;
const GAP_RATIO: f64 = 2.5;
const COL_TOLERANCE_PT: f64 = 6.0;
const MIN_SHARED_COLS: usize = 2;
const MIN_ROW_ITEMS: usize = 3;
const MIN_TABLE_ROWS: usize = 2;

/// Detect structured tables on a page and return them in reading order.
pub fn detect_tables(items: &[TextItem]) -> Vec<Table> {
    if items.len() < MIN_TABLE_ROWS * MIN_ROW_ITEMS {
        return Vec::new();
    }

    // Sort by (y, x) and cluster into rows by y-band.
    let mut idx: Vec<usize> = (0..items.len()).collect();
    idx.sort_by(|&a, &b| {
        items[a]
            .y
            .total_cmp(&items[b].y)
            .then(items[a].x.total_cmp(&items[b].x))
    });

    let mut rows: Vec<Vec<usize>> = Vec::new();
    for i in idx {
        let y = items[i].y;
        match rows.last_mut() {
            Some(r) => {
                let last_y = items[*r.last().unwrap()].y;
                if (y - last_y).abs() <= ROW_BAND_PT {
                    r.push(i);
                } else {
                    rows.push(vec![i]);
                }
            }
            None => rows.push(vec![i]),
        }
    }
    // Sort each row by x (clustering walked in (y,x) order, but merge of
    // two runs at same y-band can land out-of-order).
    for r in rows.iter_mut() {
        r.sort_by(|&a, &b| items[a].x.total_cmp(&items[b].x));
    }

    let median_char_w = median_char_width(items).max(1.0);
    let gap_threshold = GAP_RATIO * median_char_w;

    // Mark table-candidate rows and extract their column x-centres.
    let row_sigs: Vec<Option<Vec<f64>>> = rows
        .iter()
        .map(|r| row_column_centres(r, items, gap_threshold))
        .collect();

    // Scan for maximal runs of candidate rows with matching signatures.
    let mut tables: Vec<Table> = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        if row_sigs[i].is_none() {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        // Growing column set: any row in the run must share at least
        // MIN_SHARED_COLS columns with the canonical set.
        let mut columns: Vec<f64> = row_sigs[i].as_ref().unwrap().clone();
        while j < rows.len() {
            match &row_sigs[j] {
                Some(sig) if shares_columns(&columns, sig, MIN_SHARED_COLS) => {
                    merge_columns(&mut columns, sig);
                    j += 1;
                }
                _ => break,
            }
        }
        let run_len = j - i;
        if run_len >= MIN_TABLE_ROWS {
            columns.sort_by(f64::total_cmp);
            collapse_near_columns(&mut columns);
            tables.push(build_table(&rows[i..j], items, &columns, tables.len() as u32 + 1));
        }
        i = j.max(i + 1);
    }

    tables
}

/// After merging columns from multiple rows, collapse clusters of
/// near-duplicate x-centres (within `2 * COL_TOLERANCE_PT`) into a
/// single mean centre. Prevents rows that disagree slightly on a
/// column position from inflating the column count.
fn collapse_near_columns(columns: &mut Vec<f64>) {
    if columns.len() < 2 {
        return;
    }
    columns.sort_by(f64::total_cmp);
    let merge_gap = COL_TOLERANCE_PT * 2.0;
    let mut out: Vec<f64> = Vec::with_capacity(columns.len());
    let mut cluster_sum = columns[0];
    let mut cluster_n = 1usize;
    let mut cluster_last = columns[0];
    for &c in &columns[1..] {
        if c - cluster_last <= merge_gap {
            cluster_sum += c;
            cluster_n += 1;
            cluster_last = c;
        } else {
            out.push(cluster_sum / cluster_n as f64);
            cluster_sum = c;
            cluster_n = 1;
            cluster_last = c;
        }
    }
    out.push(cluster_sum / cluster_n as f64);
    *columns = out;
}

fn median_char_width(items: &[TextItem]) -> f64 {
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
        return 4.0;
    }
    widths.sort_by(f64::total_cmp);
    widths[widths.len() / 2]
}

/// If the row looks tabular (≥ MIN_ROW_ITEMS with gaps ≥ threshold
/// between at least MIN_ROW_ITEMS - 1 neighbours), return the x-centres
/// of each cell. Otherwise return None.
fn row_column_centres(
    row: &[usize],
    items: &[TextItem],
    gap_threshold: f64,
) -> Option<Vec<f64>> {
    if row.len() < MIN_ROW_ITEMS {
        return None;
    }
    let mut big_gaps = 0usize;
    for w in row.windows(2) {
        let left = &items[w[0]];
        let right = &items[w[1]];
        let gap = right.x - (left.x + left.width);
        if gap >= gap_threshold {
            big_gaps += 1;
        }
    }
    // Want at least row.len() - 1 gaps to be "big" so the row is
    // effectively MIN_ROW_ITEMS distinct columns.
    if big_gaps < MIN_ROW_ITEMS - 1 {
        return None;
    }
    Some(
        row.iter()
            .map(|&i| items[i].x + items[i].width * 0.5)
            .collect(),
    )
}

fn shares_columns(a: &[f64], b: &[f64], min_shared: usize) -> bool {
    let mut shared = 0usize;
    for &x in b {
        if a.iter().any(|&ax| (ax - x).abs() <= COL_TOLERANCE_PT) {
            shared += 1;
            if shared >= min_shared {
                return true;
            }
        }
    }
    false
}

fn merge_columns(dst: &mut Vec<f64>, src: &[f64]) {
    for &x in src {
        if !dst.iter().any(|&dx| (dx - x).abs() <= COL_TOLERANCE_PT) {
            dst.push(x);
        }
    }
}

fn build_table(
    rows: &[Vec<usize>],
    items: &[TextItem],
    columns: &[f64],
    id: u32,
) -> Table {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    let mut cell_rows: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut cells: Vec<String> = vec![String::new(); columns.len()];
        for &i in row {
            let it = &items[i];
            let cx = it.x + it.width * 0.5;
            let col = closest_column(columns, cx);
            if !cells[col].is_empty() {
                cells[col].push(' ');
            }
            cells[col].push_str(&it.str);
            if it.x < xmin {
                xmin = it.x;
            }
            let right = it.x + it.width;
            if right > xmax {
                xmax = right;
            }
            if it.y < ymin {
                ymin = it.y;
            }
            let bottom = it.y + it.height;
            if bottom > ymax {
                ymax = bottom;
            }
        }
        cell_rows.push(cells);
    }
    Table {
        id,
        x: xmin,
        y: ymin,
        width: (xmax - xmin).max(0.0),
        height: (ymax - ymin).max(0.0),
        column_centres: columns.to_vec(),
        rows: cell_rows,
    }
}

fn closest_column(columns: &[f64], cx: f64) -> usize {
    let mut best_i = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, &c) in columns.iter().enumerate() {
        let d = (c - cx).abs();
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(s: &str, x: f64, y: f64, w: f64) -> TextItem {
        TextItem::new(s, x, y, w, 10.0)
    }

    #[test]
    fn detects_simple_3col_table() {
        let items = vec![
            item("Name", 50.0, 100.0, 30.0),
            item("Qty", 200.0, 100.0, 20.0),
            item("Price", 350.0, 100.0, 30.0),
            item("Apple", 50.0, 120.0, 30.0),
            item("3", 200.0, 120.0, 6.0),
            item("$1.20", 350.0, 120.0, 30.0),
            item("Banana", 50.0, 140.0, 40.0),
            item("7", 200.0, 140.0, 6.0),
            item("$0.45", 350.0, 140.0, 30.0),
        ];
        let tables = detect_tables(&items);
        assert_eq!(tables.len(), 1, "expected a single table");
        let t = &tables[0];
        assert_eq!(t.rows.len(), 3);
        assert_eq!(t.column_centres.len(), 3);
        assert_eq!(t.rows[0][0], "Name");
        assert_eq!(t.rows[2][2], "$0.45");
    }

    #[test]
    fn ignores_prose() {
        let items = vec![
            item("The", 50.0, 100.0, 20.0),
            item("quick", 75.0, 100.0, 25.0),
            item("brown", 105.0, 100.0, 30.0),
            item("fox", 140.0, 100.0, 18.0),
            item("jumps", 165.0, 100.0, 25.0),
        ];
        assert!(detect_tables(&items).is_empty());
    }
}
