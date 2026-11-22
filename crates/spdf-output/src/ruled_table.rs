//! Table recovery from ruling lines.
//!
//! Bordered tables are drawn with vector strokes (or thin filled
//! rectangles). When the page's horizontal and vertical rules form a
//! connected grid, the grid itself defines rows, columns and merged cells
//! far more reliably than text alignment can: wrapped multi-line cells,
//! ragged columns and single-space gutters all stay inside their borders.
//!
//! This module turns a page's [`Ruling`]s into grids, assigns the page's
//! text items to grid cells, and locates the projected-text lines the
//! table replaces so the markdown formatter can swap them for a pipe table.

use std::collections::HashMap;

use spdf_types::{Ruling, TextItem};

/// Rules whose positions differ by less than this are the same line
/// (double strokes, the two long edges of a thin filled rectangle).
const SNAP: f64 = 2.0;
/// Gap bridged when joining collinear rules or testing intersections.
const JOIN: f64 = 3.0;
/// Grid boundaries closer than this collapse into one.
const MIN_CELL: f64 = 4.0;
/// Pages with more rule pairs than this are vector art, not tables.
const MAX_PAIRS: usize = 4_000_000;
/// Share of a table's words that must be found in consecutive projected
/// lines before those lines are replaced by the table.
const MIN_LOCATED_SHARE: f64 = 0.8;
/// Minimum share of non-empty cells in a table grid.
const MIN_FILL: f64 = 0.35;

#[derive(Debug, Clone, Copy)]
struct HLine {
    y: f64,
    x1: f64,
    x2: f64,
}

#[derive(Debug, Clone, Copy)]
struct VLine {
    x: f64,
    y1: f64,
    y2: f64,
}

/// A bordered table found on a page.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RuledTable {
    pub top: f64,
    pub bottom: f64,
    /// Rows of cell text; the first row is rendered as the header.
    pub rows: Vec<Vec<String>>,
    /// Whitespace-separated words of every text item inside the table.
    pub words: Vec<String>,
}

/// Merge collinear, overlapping (or nearly touching) segments.
fn merge_lines(mut v: Vec<(f64, f64, f64)>) -> Vec<(f64, f64, f64)> {
    // (position, start, end)
    v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut groups: Vec<Vec<(f64, f64, f64)>> = Vec::new();
    for l in v {
        match groups.last_mut() {
            Some(g) if (l.0 - g[0].0).abs() <= SNAP => g.push(l),
            _ => groups.push(vec![l]),
        }
    }
    let mut out = Vec::new();
    for mut g in groups {
        let pos = g.iter().map(|l| l.0).sum::<f64>() / g.len() as f64;
        g.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut cur: Option<(f64, f64)> = None;
        for l in g {
            cur = match cur {
                Some((s, e)) if l.1 <= e + JOIN => Some((s, e.max(l.2))),
                Some((s, e)) => {
                    out.push((pos, s, e));
                    Some((l.1, l.2))
                }
                None => Some((l.1, l.2)),
            };
        }
        if let Some((s, e)) = cur {
            out.push((pos, s, e));
        }
    }
    out
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Cluster sorted values: values within `MIN_CELL` of the previous
/// cluster's last member join it. Returns cluster representatives (mean).
fn cluster(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    let mut out: Vec<(f64, f64, usize)> = Vec::new(); // (last, sum, n)
    for x in v {
        match out.last_mut() {
            Some(c) if x - c.0 < MIN_CELL => {
                c.0 = x;
                c.1 += x;
                c.2 += 1;
            }
            _ => out.push((x, x, 1)),
        }
    }
    out.into_iter().map(|c| c.1 / c.2 as f64).collect()
}

/// Index of the interval of `bounds` (sorted) containing `v`.
fn slot(bounds: &[f64], v: f64) -> Option<usize> {
    if v < bounds[0] || v > bounds[bounds.len() - 1] {
        return None;
    }
    Some(
        bounds
            .windows(2)
            .position(|w| v <= w[1])
            .unwrap_or(bounds.len() - 2),
    )
}

fn nearest(bounds: &[f64], v: f64) -> usize {
    let mut best = 0;
    for (i, b) in bounds.iter().enumerate() {
        if (b - v).abs() < (bounds[best] - v).abs() {
            best = i;
        }
    }
    best
}

/// Detect bordered tables from `rulings` and fill them with `items`.
/// Tables are returned top to bottom.
pub(crate) fn detect(rulings: &[Ruling], items: &[TextItem]) -> Vec<RuledTable> {
    let mut hs = Vec::new();
    let mut vs = Vec::new();
    for r in rulings {
        if r.is_horizontal() {
            hs.push(((r.y1 + r.y2) / 2.0, r.x1.min(r.x2), r.x1.max(r.x2)));
        } else {
            vs.push(((r.x1 + r.x2) / 2.0, r.y1.min(r.y2), r.y1.max(r.y2)));
        }
    }
    let hs: Vec<HLine> = merge_lines(hs)
        .into_iter()
        .map(|(y, x1, x2)| HLine { y, x1, x2 })
        .collect();
    let vs: Vec<VLine> = merge_lines(vs)
        .into_iter()
        .map(|(x, y1, y2)| VLine { x, y1, y2 })
        .collect();
    if hs.len() < 2 || hs.len() * vs.len().max(1) > MAX_PAIRS {
        return Vec::new();
    }
    let mut out = if vs.len() >= 2 {
        grid_tables(&hs, &vs, items)
    } else {
        Vec::new()
    };
    // Tables ruled only horizontally (top / header / bottom rules), outside
    // any bordered grid.
    for t in rule_bounded_tables(&hs, &vs, items) {
        if !out.iter().any(|k| t.top < k.bottom && k.top < t.bottom) {
            out.push(t);
        }
    }
    out.sort_by(|a, b| a.top.total_cmp(&b.top));
    out
}

/// Bordered tables: connected grids of horizontal and vertical rules.
fn grid_tables(hs: &[HLine], vs: &[VLine], items: &[TextItem]) -> Vec<RuledTable> {
    // Connected components of intersecting rules.
    let n = hs.len() + vs.len();
    let mut parent: Vec<usize> = (0..n).collect();
    for (i, h) in hs.iter().enumerate() {
        for (j, v) in vs.iter().enumerate() {
            if v.x >= h.x1 - JOIN && v.x <= h.x2 + JOIN && h.y >= v.y1 - JOIN && h.y <= v.y2 + JOIN
            {
                let (a, b) = (find(&mut parent, i), find(&mut parent, hs.len() + j));
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }
    let mut comps: HashMap<usize, (Vec<HLine>, Vec<VLine>)> = HashMap::new();
    for (i, h) in hs.iter().enumerate() {
        comps.entry(find(&mut parent, i)).or_default().0.push(*h);
    }
    for (j, v) in vs.iter().enumerate() {
        comps
            .entry(find(&mut parent, hs.len() + j))
            .or_default()
            .1
            .push(*v);
    }

    let mut out: Vec<RuledTable> = comps
        .into_values()
        .filter_map(|(h, v)| build_table(&h, &v, items))
        .collect();
    out.sort_by(|a, b| a.top.total_cmp(&b.top));
    // Nested grids (a table drawn inside a frame) can both qualify; keep
    // the outer one only when they overlap.
    let mut kept: Vec<RuledTable> = Vec::new();
    for t in out {
        if kept.last().is_some_and(|k| t.top < k.bottom) {
            continue;
        }
        kept.push(t);
    }
    kept
}

/// Horizontal rules of the same extent stacked down the page bound a
/// table typeset without vertical rules (the booktabs / academic style:
/// a top rule, a rule under the header, a bottom rule). Each stack is a
/// candidate table whose columns come from the whitespace gutters of its
/// text. A stack that does not read as a table is retried as two stacks
/// split at its tallest band, since two such tables of the same width on
/// one page (with a caption or prose between them) share an extent.
fn rule_bounded_tables(hs: &[HLine], vs: &[VLine], items: &[TextItem]) -> Vec<RuledTable> {
    const SAME_EXTENT: f64 = 3.0;
    // Rules touched by a vertical rule belong to a grid (bordered table,
    // chart frame), which `grid_tables` judges.
    let free = |h: &HLine| {
        !vs.iter().any(|v| {
            v.x >= h.x1 - JOIN && v.x <= h.x2 + JOIN && h.y >= v.y1 - JOIN && h.y <= v.y2 + JOIN
        })
    };
    let mut sorted: Vec<HLine> = hs.iter().copied().filter(free).collect();
    sorted.sort_by(|a, b| a.y.total_cmp(&b.y));
    let mut stacks: Vec<Vec<HLine>> = Vec::new();
    for h in sorted {
        if h.x2 - h.x1 < 40.0 {
            continue;
        }
        let same = |s: &&mut Vec<HLine>| {
            let l = s[s.len() - 1];
            (l.x1 - h.x1).abs() <= SAME_EXTENT && (l.x2 - h.x2).abs() <= SAME_EXTENT
        };
        match stacks.iter_mut().find(same) {
            Some(s) => s.push(h),
            None => stacks.push(vec![h]),
        }
    }
    let mut out = Vec::new();
    let mut todo: Vec<Vec<HLine>> = stacks.into_iter().filter(|s| s.len() >= 2).collect();
    while let Some(stack) = todo.pop() {
        if let Some(t) = band_table(&stack, hs, items) {
            out.push(t);
            continue;
        }
        if stack.len() < 3 {
            continue;
        }
        let tallest = (0..stack.len() - 1)
            .max_by(|&a, &b| {
                let ga = stack[a + 1].y - stack[a].y;
                let gb = stack[b + 1].y - stack[b].y;
                ga.total_cmp(&gb)
            })
            .unwrap_or(0);
        let (upper, lower) = stack.split_at(tallest + 1);
        for part in [upper, lower] {
            if part.len() >= 2 {
                todo.push(part.to_vec());
            }
        }
    }
    out
}

/// Build a table from the text between the first and last rule of
/// `rules` (sorted top to bottom, same horizontal extent). `all_rules`
/// are every horizontal rule on the page.
fn band_table(rules: &[HLine], all_rules: &[HLine], items: &[TextItem]) -> Option<RuledTable> {
    /// Mean words per filled cell above which the band holds prose.
    const MAX_WORDS_PER_CELL: f64 = 6.0;
    let top = rules[0].y;
    let bottom = rules[rules.len() - 1].y;
    let left = rules.iter().map(|r| r.x1).fold(f64::INFINITY, f64::min);
    let right = rules.iter().map(|r| r.x2).fold(f64::NEG_INFINITY, f64::max);
    if bottom - top < 8.0 {
        return None;
    }
    let mut words: Vec<Word> = Vec::new();
    for it in items {
        if it.str.trim().is_empty() || it.r.unwrap_or(0) != 0 {
            continue;
        }
        let cx = it.x + it.width / 2.0;
        let cy = it.y + it.height / 2.0;
        if cx < left - JOIN || cx > right + JOIN || cy <= top || cy >= bottom {
            continue;
        }
        words.extend(item_words(it));
    }
    if words.len() < 3 {
        return None;
    }
    let mut heights: Vec<f64> = words.iter().map(|w| w.h).collect();
    heights.sort_by(f64::total_cmp);
    let text_h = heights[heights.len() / 2];
    let lines = text_lines(words);
    if lines.len() < 2 || lines.iter().any(|l| is_prose_line(l, text_h)) {
        return None;
    }
    // Table rules run between text lines; chart gridlines run through
    // their axis labels (which may sit just outside the gridlines' ends).
    let inner: Vec<&HLine> = all_rules
        .iter()
        .filter(|r| r.y > top + 1.0 && r.y < bottom - 1.0)
        .filter(|r| r.x2.min(right) - r.x1.max(left) >= 0.5 * (right - left))
        .collect();
    let reach = 4.0 * text_h;
    let struck = inner
        .iter()
        .filter(|r| {
            items.iter().any(|it| {
                !it.str.trim().is_empty()
                    && (it.y + it.height / 2.0 - r.y).abs() < 0.5 * it.height.max(text_h)
                    && it.x + it.width > r.x1 - reach
                    && it.x < r.x2 + reach
            })
        })
        .count();
    if struck > 0 && struck * 2 >= inner.len() {
        return None;
    }
    // Lines above the rule under the header form one header row.
    let header_end = if rules.len() >= 3 {
        let mid = rules[1].y;
        lines.iter().take_while(|l| l[0].y < mid).count()
    } else {
        0
    };
    let body = &lines[header_end..];
    let gutter_src = if body.len() >= 2 { body } else { &lines[..] };
    let bounds = column_bounds(gutter_src, left, right);
    let ncols = bounds.len() + 1;
    let col_of = |w: &Word| bounds.iter().filter(|&&b| w.x > b).count();

    let mut words_all = Vec::new();
    let mut rows: Vec<Vec<Vec<Word>>> = Vec::new();
    let mut header_rows = 0;
    for (i, line) in lines.into_iter().enumerate() {
        let mut cells: Vec<Vec<Word>> = (0..ncols).map(|_| Vec::new()).collect();
        for w in line {
            words_all.push(w.text.clone());
            let c = col_of(&w);
            cells[c].push(w);
        }
        let filled = cells.iter().filter(|c| !c.is_empty()).count();
        let in_header = i < header_end;
        // Header lines merge into one row; a body line with an empty key
        // column and few cells continues the row above (a wrapped cell).
        let fold = if in_header {
            i > 0
        } else {
            ncols > 1 && rows.len() > header_rows && cells[0].is_empty() && filled * 2 <= ncols
        };
        if fold {
            let prev = rows.last_mut().expect("row above");
            for (p, c) in prev.iter_mut().zip(cells) {
                p.extend(c);
            }
        } else {
            rows.push(cells);
            if in_header {
                header_rows = 1;
            }
        }
    }
    let rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|r| r.into_iter().map(cell_text).collect())
        .collect();
    if rows.len() < 2 {
        return None;
    }
    let filled: Vec<&String> = rows.iter().flatten().filter(|c| !c.is_empty()).collect();
    let nwords: usize = filled.iter().map(|c| c.split_whitespace().count()).sum();
    if nwords as f64 > MAX_WORDS_PER_CELL * filled.len() as f64 {
        return None;
    }
    if ncols == 1 {
        // A ruled single-column list needs its header rule and short,
        // list-like entries.
        if header_end == 0 || rows.len() < 3 {
            return None;
        }
    } else {
        if (filled.len() as f64) < MIN_FILL * (rows.len() * ncols) as f64 {
            return None;
        }
        if !rows
            .iter()
            .any(|r| r.iter().filter(|c| !c.is_empty()).count() >= 2)
        {
            return None;
        }
    }
    Some(RuledTable {
        top,
        bottom,
        rows,
        words: words_all,
    })
}

/// Group words into visual lines, top to bottom.
fn text_lines(mut words: Vec<Word>) -> Vec<Vec<Word>> {
    words.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));
    let mut lines: Vec<Vec<Word>> = Vec::new();
    for w in words {
        match lines.last_mut() {
            Some(l) if (w.y - l[0].y).abs() < 0.5 * w.h.min(l[0].h) => l.push(w),
            _ => lines.push(vec![w]),
        }
    }
    lines
}

/// Column boundaries (x positions, ascending) from the vertical gutters
/// of `lines`: strips at least one text height wide that are free of ink
/// on all lines but a few (a spanning title, one cell estimated too
/// wide), with inked columns on both sides.
fn column_bounds(lines: &[Vec<Word>], left: f64, right: f64) -> Vec<f64> {
    const BIN: f64 = 0.5;
    let x0 = left.min(
        lines
            .iter()
            .flatten()
            .map(|w| w.x0)
            .fold(f64::INFINITY, f64::min),
    );
    let x1 = right.max(
        lines
            .iter()
            .flatten()
            .map(|w| w.x1)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let nbins = (((x1 - x0) / BIN).ceil() as usize).clamp(1, 20_000);
    let mut ink = vec![0usize; nbins];
    for line in lines {
        let mut hit = vec![false; nbins];
        for w in line {
            let a = (((w.x0 - x0) / BIN).floor().max(0.0) as usize).min(nbins - 1);
            let b = (((w.x1 - x0) / BIN).ceil().max(0.0) as usize).min(nbins);
            for h in &mut hit[a..b] {
                *h = true;
            }
        }
        for (k, h) in ink.iter_mut().zip(hit) {
            *k += h as usize;
        }
    }
    let allow = lines.len() / 4;
    let mut hs: Vec<f64> = lines.iter().flatten().map(|w| w.h).collect();
    hs.sort_by(f64::total_cmp);
    let min_gap = hs[hs.len() / 2];
    // Blank strips between inked columns, plus strips crossed only by a
    // few lines between well-inked columns.
    let strict = blank_runs(&ink, 0, min_gap / BIN);
    let mut runs = strict.clone();
    if allow > 0 {
        for r in blank_runs(&ink, allow, min_gap / BIN) {
            if !strict.iter().any(|s| s.0 < r.1 && r.0 < s.1) {
                runs.push(r);
            }
        }
    }
    let mut out: Vec<f64> = runs
        .into_iter()
        .map(|(s, e)| x0 + (s + e) as f64 / 2.0 * BIN)
        .collect();
    out.sort_by(f64::total_cmp);
    out
}

/// Runs `[s, e)` of bins with `ink <= allow`, at least `min_len` bins
/// long, that have a bin with more ink on both sides.
fn blank_runs(ink: &[usize], allow: usize, min_len: f64) -> Vec<(usize, usize)> {
    let low: Vec<bool> = ink.iter().map(|&k| k <= allow).collect();
    let (Some(first), Some(last)) = (low.iter().position(|l| !l), low.iter().rposition(|l| !l))
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut i = first;
    while i <= last {
        if !low[i] {
            i += 1;
            continue;
        }
        let s = i;
        while low[i] {
            i += 1;
        }
        if (i - s) as f64 >= min_len {
            out.push((s, i));
        }
    }
    out
}

/// Does a line of `words` hold running prose (a caption or paragraph
/// caught between two tables' rules)? Prose is a long run of closely
/// spaced words, mostly starting in lowercase.
fn is_prose_line(line: &[Word], gap: f64) -> bool {
    const MIN_WORDS: usize = 8;
    let mut run: Vec<&Word> = Vec::new();
    let check = |run: &[&Word]| {
        run.len() >= MIN_WORDS
            && run
                .iter()
                .filter(|w| w.text.chars().next().is_some_and(char::is_lowercase))
                .count()
                * 2
                >= run.len()
    };
    for w in line {
        if run.last().is_some_and(|p| w.x0 - p.x1 >= gap) {
            if check(&run) {
                return true;
            }
            run.clear();
        }
        run.push(w);
    }
    check(&run)
}

/// One word of a text item with its estimated position.
struct Word {
    text: String,
    /// Horizontal extent and centre.
    x0: f64,
    x1: f64,
    x: f64,
    y: f64,
    h: f64,
}

/// Split an item into words with x positions estimated from an even
/// spread of its characters over its width.
fn item_words(it: &TextItem) -> Vec<Word> {
    let chars: Vec<char> = it.str.chars().collect();
    let cw = if chars.is_empty() {
        0.0
    } else {
        it.width / chars.len() as f64
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let s = i;
        while i < chars.len() && !chars[i].is_whitespace() {
            i += 1;
        }
        out.push(Word {
            text: chars[s..i].iter().collect(),
            x0: it.x + cw * s as f64,
            x1: it.x + cw * i as f64,
            x: it.x + cw * (s + i) as f64 / 2.0,
            y: it.y + it.height / 2.0,
            h: it.height.max(1.0),
        });
    }
    out
}

fn build_table(hs: &[HLine], vs: &[VLine], items: &[TextItem]) -> Option<RuledTable> {
    if hs.len() < 2 || vs.len() < 2 {
        return None;
    }
    let left = hs.iter().map(|h| h.x1).chain(vs.iter().map(|v| v.x));
    let left = left.fold(f64::INFINITY, f64::min);
    let right = hs.iter().map(|h| h.x2).chain(vs.iter().map(|v| v.x));
    let right = right.fold(f64::NEG_INFINITY, f64::max);
    let top = vs.iter().map(|v| v.y1).chain(hs.iter().map(|h| h.y));
    let top = top.fold(f64::INFINITY, f64::min);
    let bottom = vs.iter().map(|v| v.y2).chain(hs.iter().map(|h| h.y));
    let bottom = bottom.fold(f64::NEG_INFINITY, f64::max);
    if right - left < 20.0 || bottom - top < 10.0 {
        return None;
    }

    let xs = cluster(vs.iter().map(|v| v.x).chain([left, right]).collect());
    let ys = cluster(hs.iter().map(|h| h.y).chain([top, bottom]).collect());
    let (ncols, nrows) = (xs.len() - 1, ys.len() - 1);
    if ncols < 1 || nrows < 1 {
        return None;
    }

    // Which internal boundaries are actually drawn. A missing segment
    // between two atomic cells means they are one merged cell.
    let vdrawn = |c: usize, r: usize| {
        // Boundary at xs[c] across row r. A row crossed by no vertical rule
        // at all (rules only between rows) keeps the column structure.
        let mid = (ys[r] + ys[r + 1]) / 2.0;
        let covers = |v: &&VLine| v.y1 - JOIN <= mid && v.y2 + JOIN >= mid;
        let mut crossing = vs.iter().filter(covers).peekable();
        crossing.peek().is_none() || crossing.any(|v| nearest(&xs, v.x) == c)
    };
    let hdrawn = |r: usize, c: usize| {
        let mid = (xs[c] + xs[c + 1]) / 2.0;
        hs.iter()
            .any(|h| nearest(&ys, h.y) == r && h.x1 - JOIN <= mid && h.x2 + JOIN >= mid)
    };
    // owner[r][c]: the top-left atomic cell of the merged cell.
    let mut owner = vec![vec![(0usize, 0usize); ncols]; nrows];
    for r in 0..nrows {
        for c in 0..ncols {
            owner[r][c] = if r > 0 && !hdrawn(r, c) {
                owner[r - 1][c]
            } else if c > 0 && !vdrawn(c, r) {
                owner[r][c - 1]
            } else {
                (r, c)
            };
        }
    }

    // Assign words to cells.
    let mut cells: Vec<Vec<Vec<Word>>> = (0..nrows)
        .map(|_| (0..ncols).map(|_| Vec::new()).collect())
        .collect();
    let mut words_all = Vec::new();
    for it in items {
        if it.str.trim().is_empty() || it.r.unwrap_or(0) != 0 {
            continue;
        }
        let cx = it.x + it.width / 2.0;
        let cy = it.y + it.height / 2.0;
        if cx < left || cx > right || cy < top || cy > bottom {
            continue;
        }
        let Some(r) = slot(&ys, cy) else { continue };
        let ca = slot(&xs, it.x + 1.0).unwrap_or(0);
        let cb = slot(&xs, it.x + it.width - 1.0).unwrap_or(ncols - 1);
        for w in item_words(it) {
            let c = if ca == cb {
                ca
            } else {
                slot(&xs, w.x).unwrap_or(ca)
            };
            let (orow, ocol) = owner[r][c];
            words_all.push(w.text.clone());
            cells[orow][ocol].push(w);
        }
    }
    if words_all.is_empty() {
        return None;
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    for (r, row) in cells.into_iter().enumerate() {
        // Rows that share a merged cell with a neighbour stay whole.
        let self_contained = (0..ncols).all(|c| owner[r][c].0 == r)
            && (r + 1 == nrows || (0..ncols).all(|c| owner[r + 1][c].0 == r + 1));
        let split = if self_contained {
            split_unruled_rows(row)
        } else {
            vec![row]
        };
        for row in split {
            rows.push(row.into_iter().map(cell_text).collect());
        }
    }
    // Drop fully empty rows and columns (decorative double rules, spacer
    // strips).
    rows.retain(|r| r.iter().any(|c| !c.is_empty()));
    let keep: Vec<bool> = (0..ncols)
        .map(|c| rows.iter().any(|r| !r[c].is_empty()))
        .collect();
    for r in rows.iter_mut() {
        let mut i = 0;
        r.retain(|_| {
            i += 1;
            keep[i - 1]
        });
    }
    let filled = rows.iter().flatten().filter(|c| !c.is_empty()).count();
    let width = rows.first().map_or(0, Vec::len);
    // A single framed box is a callout, not a table; a single column
    // needs several ruled rows to read as a list-style table.
    let min_rows = if width == 1 { 3 } else { 2 };
    if rows.len() < min_rows || filled < 3 {
        return None;
    }
    // Mostly-empty grids are chart gridlines or form layouts.
    if (filled as f64) < MIN_FILL * (rows.len() * width) as f64 {
        return None;
    }
    Some(RuledTable {
        top,
        bottom,
        rows,
        words: words_all,
    })
}

/// Tables often rule only the header and the columns, leaving several
/// logical rows inside one grid row. Split a grid row at its text lines
/// that run across most occupied columns, including the leftmost one (the
/// row key). Other lines (wrapped cell text, vertically centred values)
/// join the nearest such line. A key-column line that is not a full line
/// means the key itself wraps, so the row is left whole.
fn split_unruled_rows(row: Vec<Vec<Word>>) -> Vec<Vec<Vec<Word>>> {
    const MIN_BAND_SHARE: f64 = 0.75;
    let occupied = row.iter().filter(|c| !c.is_empty()).count();
    let Some(key) = row.iter().position(|c| !c.is_empty()) else {
        return vec![row];
    };
    if occupied < 2 {
        return vec![row];
    }
    // Line bands over the whole row: (centre y, height).
    let mut ys: Vec<(f64, f64)> = row.iter().flatten().map(|w| (w.y, w.h)).collect();
    ys.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut bands: Vec<(f64, f64)> = Vec::new();
    for (y, h) in ys {
        match bands.last() {
            Some(b) if (y - b.0).abs() < 0.5 * h.min(b.1) => {}
            _ => bands.push((y, h)),
        }
    }
    if bands.len() < 2 {
        return vec![row];
    }
    let band_of = |w: &Word| {
        bands
            .iter()
            .rposition(|b| w.y >= b.0 - 0.5 * w.h.min(b.1))
            .unwrap_or(0)
    };
    let mut full = Vec::new();
    for bi in 0..bands.len() {
        let in_band = |c: &Vec<Word>| c.iter().any(|w| band_of(w) == bi);
        let cols = row.iter().filter(|c| in_band(c)).count();
        let has_key = in_band(&row[key]);
        if (cols as f64) >= MIN_BAND_SHARE * occupied as f64 && has_key {
            full.push(bi);
        } else if has_key {
            return vec![row];
        }
    }
    if full.len() < 2 {
        return vec![row];
    }
    // Map every band to the nearest full band.
    let target: Vec<usize> = (0..bands.len())
        .map(|bi| {
            (0..full.len())
                .min_by(|&a, &b| {
                    let da = (bands[full[a]].0 - bands[bi].0).abs();
                    let db = (bands[full[b]].0 - bands[bi].0).abs();
                    da.total_cmp(&db)
                })
                .unwrap_or(0)
        })
        .collect();
    let ncols = row.len();
    let mut out: Vec<Vec<Vec<Word>>> = (0..full.len())
        .map(|_| (0..ncols).map(|_| Vec::new()).collect())
        .collect();
    for (c, cell) in row.into_iter().enumerate() {
        for w in cell {
            let t = target[band_of(&w)];
            out[t][c].push(w);
        }
    }
    out
}

/// Join a cell's words in reading order: visual lines top to bottom,
/// words left to right.
fn cell_text(mut words: Vec<Word>) -> String {
    words.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));
    let mut lines: Vec<Vec<Word>> = Vec::new();
    for w in words {
        match lines.last_mut() {
            Some(l) if (w.y - l[0].y).abs() < 0.5 * w.h.min(l[0].h) => l.push(w),
            _ => lines.push(vec![w]),
        }
    }
    let mut out = String::new();
    for mut l in lines {
        l.sort_by(|a, b| a.x.total_cmp(&b.x));
        let mut prev_end: Option<f64> = None;
        for w in l {
            // Separately drawn runs that touch ("(" "SS" ")") form one word.
            let glued = prev_end.is_some_and(|e| w.x0 - e < 0.15 * w.h);
            if !out.is_empty() && !glued {
                out.push(' ');
            }
            out.push_str(&w.text);
            prev_end = Some(w.x1);
        }
    }
    out
}

/// Find the run of projected `lines` (starting at or after `from`) that
/// holds the table's words. Returns the inclusive line range.
pub(crate) fn locate(lines: &[&str], words: &[String], from: usize) -> Option<(usize, usize)> {
    let total = words.len();
    if total == 0 {
        return None;
    }
    let mut bag: HashMap<&str, usize> = HashMap::new();
    for w in words {
        *bag.entry(w.as_str()).or_default() += 1;
    }
    // Consume the line's words from `bag`; Some(consumed) when the line
    // belongs. A projected word may be several adjacent runs glued
    // together ("(" "SS" ")" -> "(SS)"), so words are matched as
    // concatenations of bag entries.
    let take = |bag: &mut HashMap<&str, usize>, line: &str| -> Option<usize> {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            return None;
        }
        let mut trial = bag.clone();
        let mut hits = 0;
        let mut used = 0;
        for t in &toks {
            let mut taken = Vec::new();
            if consume(&mut trial, t, &mut taken, 0) {
                hits += 1;
                used += taken.len();
            }
        }
        if hits * 4 < toks.len() * 3 {
            return None;
        }
        *bag = trial;
        Some(used)
    };
    for start in from..lines.len() {
        let mut b = bag.clone();
        let Some(mut got) = take(&mut b, lines[start]) else {
            continue;
        };
        let mut end = start;
        let mut j = start + 1;
        while j < lines.len() && got < total {
            if lines[j].trim().is_empty() {
                j += 1;
                continue;
            }
            match take(&mut b, lines[j]) {
                Some(n) => {
                    got += n;
                    end = j;
                    j += 1;
                }
                None => break,
            }
        }
        if got as f64 >= total as f64 * MIN_LOCATED_SHARE {
            return Some((start, end));
        }
    }
    None
}

/// Try to spell `tok` as a concatenation of words left in `bag`, taking
/// them out on success (recorded in `taken`).
fn consume<'a>(
    bag: &mut HashMap<&'a str, usize>,
    tok: &str,
    taken: &mut Vec<&'a str>,
    depth: usize,
) -> bool {
    if tok.is_empty() {
        return true;
    }
    if depth > 12 {
        return false;
    }
    let ends: Vec<usize> = tok
        .char_indices()
        .map(|(i, c)| i + c.len_utf8())
        .rev()
        .collect();
    for end in ends {
        let Some((&key, &n)) = bag.get_key_value(&tok[..end]) else {
            continue;
        };
        if n == 0 {
            continue;
        }
        *bag.get_mut(key).expect("present") -= 1;
        taken.push(key);
        if consume(bag, &tok[end..], taken, depth + 1) {
            return true;
        }
        taken.pop();
        *bag.get_mut(key).expect("present") += 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(y: f64, x1: f64, x2: f64) -> Ruling {
        Ruling {
            x1,
            y1: y,
            x2,
            y2: y,
        }
    }
    fn v(x: f64, y1: f64, y2: f64) -> Ruling {
        Ruling {
            x1: x,
            y1,
            x2: x,
            y2,
        }
    }
    fn item(s: &str, x: f64, y: f64) -> TextItem {
        TextItem::new(s, x, y, s.chars().count() as f64 * 5.0, 10.0)
    }

    /// 3 columns (x 0/100/200/300) x 3 rows (y 0/20/40/60), fully ruled.
    fn grid() -> Vec<Ruling> {
        let mut r = Vec::new();
        for y in [0.0, 20.0, 40.0, 60.0] {
            r.push(h(y, 0.0, 300.0));
        }
        for x in [0.0, 100.0, 200.0, 300.0] {
            r.push(v(x, 0.0, 60.0));
        }
        r
    }

    #[test]
    fn full_grid_with_wrapped_cell() {
        let items = vec![
            item("Name", 5.0, 5.0),
            item("Qty", 105.0, 5.0),
            item("Note", 205.0, 5.0),
            item("Apple", 5.0, 22.0),
            item("3", 105.0, 25.0),
            item("very long", 205.0, 21.0),
            item("comment", 205.0, 30.0),
            item("Pear", 5.0, 45.0),
            item("5", 105.0, 45.0),
            item("ok", 205.0, 45.0),
            item("outside", 5.0, 100.0),
        ];
        let t = detect(&grid(), &items);
        assert_eq!(t.len(), 1);
        assert_eq!(
            t[0].rows,
            vec![
                vec!["Name", "Qty", "Note"],
                vec!["Apple", "3", "very long comment"],
                vec!["Pear", "5", "ok"],
            ]
        );
        assert!(!t[0].words.contains(&"outside".to_string()));
    }

    #[test]
    fn missing_internal_rule_merges_cells() {
        // Header row spans the first two columns: no rule at x=100, y 0..20.
        let mut r: Vec<Ruling> = grid()
            .into_iter()
            .filter(|r| !(r.x1 == 100.0 && r.x2 == 100.0))
            .collect();
        r.push(v(100.0, 20.0, 60.0));
        let items = vec![
            item("Fruit facts", 60.0, 5.0),
            item("Note", 205.0, 5.0),
            item("Apple", 5.0, 25.0),
            item("3", 105.0, 25.0),
            item("x", 205.0, 25.0),
            item("Pear", 5.0, 45.0),
            item("5", 105.0, 45.0),
            item("y", 205.0, 45.0),
        ];
        let t = detect(&r, &items);
        assert_eq!(t[0].rows[0], vec!["Fruit facts", "", "Note"]);
    }

    #[test]
    fn single_box_is_not_a_table() {
        let r = vec![
            h(0.0, 0.0, 300.0),
            h(60.0, 0.0, 300.0),
            v(0.0, 0.0, 60.0),
            v(300.0, 0.0, 60.0),
        ];
        let items = vec![item("Just a framed note", 5.0, 5.0)];
        assert!(detect(&r, &items).is_empty());
    }

    #[test]
    fn item_spanning_columns_is_split_by_word() {
        let items = vec![
            // One run covering two cells: "A" at x 5..10 and "B" at x 105.
            TextItem::new("A                    B", 5.0, 5.0, 110.0, 10.0),
            item("C", 205.0, 5.0),
            item("1", 5.0, 25.0),
            item("2", 105.0, 25.0),
            item("3", 205.0, 25.0),
        ];
        let t = detect(&grid(), &items);
        assert_eq!(t[0].rows[0], vec!["A", "B", "C"]);
    }

    #[test]
    fn locate_finds_projected_lines() {
        let words: Vec<String> = ["Name", "Qty", "Apple", "3", "Pear", "5"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let lines = [
            "Intro text here",
            "Name     Qty",
            "Apple    3",
            "",
            "Pear     5",
            "After",
        ];
        assert_eq!(locate(&lines, &words, 0), Some((1, 4)));
        assert_eq!(locate(&lines[..2], &words, 0), None);
    }

    #[test]
    fn locate_matches_glued_runs() {
        let words: Vec<String> = ["Code", "(", "SS", ")", "A", "B"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let lines = ["Title", "Code (SS)", "A   B"];
        assert_eq!(locate(&lines, &words, 0), Some((1, 2)));
    }

    #[test]
    fn sparse_grid_is_rejected() {
        // A 6x6 chart grid with three labels is not a table.
        let mut r = Vec::new();
        for i in 0..=6 {
            let p = i as f64 * 20.0;
            r.push(h(p, 0.0, 120.0));
            r.push(v(p, 0.0, 120.0));
        }
        let items = vec![
            item("10", 2.0, 2.0),
            item("20", 22.0, 22.0),
            item("30", 42.0, 42.0),
        ];
        assert!(detect(&r, &items).is_empty());
    }

    #[test]
    fn boxed_single_column_list_is_a_table() {
        let mut r = vec![v(0.0, 0.0, 60.0), v(200.0, 0.0, 60.0)];
        for y in [0.0, 20.0, 40.0, 60.0] {
            r.push(h(y, 0.0, 200.0));
        }
        let items = vec![
            item("Header", 5.0, 5.0),
            item("one", 5.0, 25.0),
            item("two", 5.0, 45.0),
        ];
        let t = detect(&r, &items);
        assert_eq!(t[0].rows, vec![vec!["Header"], vec!["one"], vec!["two"]]);
    }

    #[test]
    fn unruled_body_rows_are_split_but_wrapped_cells_are_not() {
        // Header ruled at y=0/20, body 20..60 has no row rules.
        let mut r = vec![h(0.0, 0.0, 200.0), h(20.0, 0.0, 200.0), h(60.0, 0.0, 200.0)];
        for x in [0.0, 100.0, 200.0] {
            r.push(v(x, 0.0, 60.0));
        }
        let items = vec![
            item("K", 5.0, 5.0),
            item("V", 105.0, 5.0),
            item("a", 5.0, 25.0),
            item("1", 105.0, 25.0),
            item("b", 5.0, 40.0),
            item("2", 105.0, 40.0),
        ];
        let t = detect(&r, &items);
        assert_eq!(
            t[0].rows,
            vec![vec!["K", "V"], vec!["a", "1"], vec!["b", "2"]]
        );

        // Same grid, but the second line belongs to one wrapped cell.
        let items = vec![
            item("K", 5.0, 5.0),
            item("V", 105.0, 5.0),
            item("long", 5.0, 25.0),
            item("1", 105.0, 25.0),
            item("text", 5.0, 40.0),
        ];
        let t = detect(&r, &items);
        assert_eq!(t[0].rows, vec![vec!["K", "V"], vec!["long text", "1"]]);

        // Three columns; the middle cell of the first body row wraps.
        let mut r = vec![h(0.0, 0.0, 300.0), h(20.0, 0.0, 300.0), h(80.0, 0.0, 300.0)];
        for x in [0.0, 100.0, 200.0, 300.0] {
            r.push(v(x, 0.0, 80.0));
        }
        let items = vec![
            item("No", 5.0, 5.0),
            item("Name", 105.0, 5.0),
            item("Qty", 205.0, 5.0),
            item("1", 5.0, 25.0),
            item("Big", 105.0, 25.0),
            item("7", 205.0, 25.0),
            item("party", 105.0, 37.0),
            item("2", 5.0, 52.0),
            item("Small", 105.0, 52.0),
            item("9", 205.0, 52.0),
        ];
        let t = detect(&r, &items);
        assert_eq!(
            t[0].rows,
            vec![
                vec!["No", "Name", "Qty"],
                vec!["1", "Big party", "7"],
                vec!["2", "Small", "9"],
            ]
        );
    }

    /// Top, header and bottom rules only (booktabs style), 300pt wide.
    fn booktabs(rows: &[&[(&str, f64)]]) -> (Vec<Ruling>, Vec<TextItem>) {
        let mut items = Vec::new();
        for (r, row) in rows.iter().enumerate() {
            for &(s, x) in row.iter() {
                items.push(item(s, x, 104.0 + 14.0 * r as f64));
            }
        }
        let bottom = 104.0 + 14.0 * rows.len() as f64;
        let rules = vec![
            h(100.0, 0.0, 300.0),
            h(116.0, 0.0, 300.0),
            h(bottom, 0.0, 300.0),
        ];
        (rules, items)
    }

    #[test]
    fn horizontally_ruled_table_uses_gutters_for_columns() {
        let (rules, mut items) = booktabs(&[
            &[("Model", 5.0), ("Score", 120.0), ("Size", 220.0)],
            &[("Cand 1", 5.0), ("73.7", 120.0), ("7B", 220.0)],
            &[("Cand 2", 5.0), ("71.2", 120.0), ("13B", 220.0)],
            &[("Base", 5.0), ("70.0", 120.0), ("7B", 220.0)],
        ]);
        items.push(item("Prose below the table.", 5.0, 200.0));
        let t = detect(&rules, &items);
        assert_eq!(t.len(), 1);
        assert_eq!(
            t[0].rows,
            vec![
                vec!["Model", "Score", "Size"],
                vec!["Cand 1", "73.7", "7B"],
                vec!["Cand 2", "71.2", "13B"],
                vec!["Base", "70.0", "7B"],
            ]
        );
    }

    #[test]
    fn spanning_title_line_does_not_hide_a_gutter() {
        let rules = vec![h(100.0, 0.0, 300.0), h(175.0, 0.0, 300.0)];
        let items = vec![
            item("Fish on the Red List", 5.0, 104.0),
            item("Potosi", 5.0, 118.0),
            item("C. alvarezi", 150.0, 118.0),
            item("La Palma", 5.0, 132.0),
            item("C. longidorsalis", 150.0, 132.0),
            item("Golden", 5.0, 146.0),
            item("S. francesae", 150.0, 146.0),
            item("Butterfly", 5.0, 160.0),
            item("A. splendens", 150.0, 160.0),
        ];
        let t = detect(&rules, &items);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].rows.len(), 5);
        assert_eq!(t[0].rows[2], vec!["La Palma", "C. longidorsalis"]);
    }

    #[test]
    fn prose_between_rules_is_not_a_table() {
        let (rules, items) = booktabs(&[
            &[("A note", 5.0)],
            &[("this is a long line of ordinary running text set", 5.0)],
            &[("in a box with rules above and below it", 5.0)],
        ]);
        assert!(detect(&rules, &items).is_empty());
    }

    #[test]
    fn chart_gridlines_are_not_a_table() {
        // Gridlines every 20pt with their tick labels centred on them.
        let mut rules = Vec::new();
        let mut items = Vec::new();
        for (i, label) in ["80", "60", "40", "20"].iter().enumerate() {
            let y = 120.0 + 20.0 * i as f64;
            rules.push(h(y, 0.0, 300.0));
            items.push(item(label, 5.0, y - 5.0));
            items.push(item("12", 100.0, y + 5.0));
            items.push(item("34", 200.0, y + 5.0));
        }
        rules.push(h(100.0, 0.0, 300.0));
        rules.push(h(200.0, 0.0, 300.0));
        assert!(detect(&rules, &items).is_empty());
    }
}
