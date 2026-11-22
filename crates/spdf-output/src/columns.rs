//! Multi-column reading order for projected page text.
//!
//! The projection renders each page onto a character grid, so a
//! two-column article comes out with both columns side by side on every
//! line. Readers (and downstream chunkers) want the left column first,
//! then the right. This pass finds runs of lines that share a vertical
//! whitespace gutter, checks that both sides look like flowing prose (so
//! tables and aligned key/value layouts are left untouched), and emits the
//! columns one after another. Three or more columns are handled by
//! recursing into the right-hand side.

/// Narrowest whitespace gutter (in grid cells) that can separate columns.
const MIN_GUTTER: usize = 2;
/// Fewest lines that must carry text on both sides of the gutter.
const MIN_PAIRED_LINES: usize = 4;
/// Average non-space characters per line a prose column must reach.
const MIN_AVG_CHARS: f64 = 15.0;
/// Largest share of a column's lines that may contain a wide internal gap.
const MAX_GAPPY_SHARE: f64 = 0.25;
/// An internal run of this many spaces marks a line as "gappy" (tabular).
const WIDE_GAP: usize = 3;
/// A line "fills" its column when it reaches this share of the column width.
const FULL_LINE_RATIO: f64 = 0.65;
/// Smallest share of a prose column's lines that must fill the column.
const MIN_FULL_SHARE: f64 = 0.6;
/// Largest number of interior holes, relative to a side's text lines.
const MAX_HOLE_SHARE: f64 = 0.25;
/// Recursion guard for very many columns.
const MAX_DEPTH: usize = 6;

type Line = Vec<char>;

fn is_blank(l: &[char]) -> bool {
    l.iter().all(|c| c.is_whitespace())
}

fn nonspace(l: &[char]) -> usize {
    l.iter().filter(|c| !c.is_whitespace()).count()
}

/// Columns occupied by any non-space character in `lines`.
fn occupancy(lines: &[Line]) -> Vec<bool> {
    let width = lines.iter().map(Vec::len).max().unwrap_or(0);
    let mut occ = vec![false; width];
    for l in lines {
        for (c, ch) in l.iter().enumerate() {
            if !ch.is_whitespace() {
                occ[c] = true;
            }
        }
    }
    occ
}

/// Interior runs of empty columns `[start, end)` at least `MIN_GUTTER` wide,
/// with occupied columns on both sides.
fn gutters(occ: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let Some(first) = occ.iter().position(|&o| o) else {
        return out;
    };
    let mut c = first;
    while c < occ.len() {
        if occ[c] {
            c += 1;
            continue;
        }
        let start = c;
        while c < occ.len() && !occ[c] {
            c += 1;
        }
        if c < occ.len() && c - start >= MIN_GUTTER {
            out.push((start, c));
        }
    }
    out
}

fn slice(l: &[char], from: usize, to: usize) -> Line {
    let to = to.min(l.len());
    if from >= to {
        Vec::new()
    } else {
        l[from..to].to_vec()
    }
}

fn has_wide_gap(l: &[char]) -> bool {
    let s: String = l.iter().collect();
    let t = s.trim();
    let mut run = 0;
    for ch in t.chars() {
        if ch == ' ' {
            run += 1;
            if run >= WIDE_GAP {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Grid cells up to and including the last non-space character.
fn extent(l: &[char]) -> usize {
    l.iter()
        .rposition(|c| !c.is_whitespace())
        .map_or(0, |p| p + 1)
}

/// Does this side read like a single column of flowing text? Prose lines
/// are long, have only single word gaps, and mostly run to the column's
/// right edge; table cells and key/value labels are short and ragged.
fn is_prose(side: &[Line]) -> bool {
    let lines: Vec<&Line> = side.iter().filter(|l| !is_blank(l)).collect();
    if lines.is_empty() {
        return false;
    }
    let n = lines.len() as f64;
    let chars: usize = lines.iter().map(|l| nonspace(l)).sum();
    let gappy = lines.iter().filter(|l| has_wide_gap(l)).count();
    let width = lines.iter().map(|l| extent(l)).max().unwrap_or(0) as f64;
    let full = lines
        .iter()
        .filter(|l| extent(l) as f64 >= width * FULL_LINE_RATIO)
        .count();
    chars as f64 / n >= MIN_AVG_CHARS
        && gappy as f64 <= n * MAX_GAPPY_SHARE
        && full as f64 >= n * MIN_FULL_SHARE
}

/// Holes inside `side` (between its first and last text line): maximal runs
/// of blank rows that face text on `other`. Prose columns only have them at
/// paragraph breaks or where a figure interrupts the column (one long run);
/// the label column of a table has one for every wrapped or grouped row.
fn too_many_holes(side: &[Line], other: &[Line]) -> bool {
    let Some(first) = side.iter().position(|l| !is_blank(l)) else {
        return false;
    };
    let last = side.iter().rposition(|l| !is_blank(l)).unwrap_or(first);
    let nonblank = side.iter().filter(|l| !is_blank(l)).count();
    let mut holes = 0usize;
    let mut in_hole = false;
    for (i, line) in side.iter().enumerate().take(last + 1).skip(first) {
        let hole = is_blank(line) && other.get(i).is_some_and(|o| !is_blank(o));
        if hole && !in_hole {
            holes += 1;
        }
        in_hole = hole;
    }
    holes as f64 > nonblank as f64 * MAX_HOLE_SHARE
}

/// Split `lines` into a left prose column and the remaining right side.
fn split_block(lines: &[Line], depth: usize) -> Option<(Vec<Line>, Vec<Line>)> {
    if depth >= MAX_DEPTH {
        return None;
    }
    let occ = occupancy(lines);
    for (g0, g1) in gutters(&occ) {
        let left: Vec<Line> = lines.iter().map(|l| slice(l, 0, g0)).collect();
        let right: Vec<Line> = lines.iter().map(|l| slice(l, g1, usize::MAX)).collect();
        let paired = left
            .iter()
            .zip(&right)
            .filter(|(a, b)| !is_blank(a) && !is_blank(b))
            .count();
        let nonblank = lines.iter().filter(|l| !is_blank(l)).count();
        if paired < MIN_PAIRED_LINES || paired * 5 < nonblank * 2 {
            continue;
        }
        if !is_prose(&left) || too_many_holes(&left, &right) || too_many_holes(&right, &left) {
            continue;
        }
        if is_prose(&right) || split_block(&right, depth + 1).is_some() {
            return Some((left, right));
        }
    }
    None
}

fn trim_blank_edges(mut v: Vec<Line>) -> Vec<Line> {
    while v.last().is_some_and(|l| is_blank(l)) {
        v.pop();
    }
    let lead = v.iter().take_while(|l| is_blank(l)).count();
    v.drain(..lead);
    v
}

/// Emit `lines` column by column, appending rendered lines to `out`.
fn emit_columns(lines: &[Line], depth: usize, out: &mut Vec<String>) {
    match split_block(lines, depth) {
        Some((left, right)) => {
            emit_columns(&trim_blank_edges(left), depth + 1, out);
            let right = trim_blank_edges(right);
            let indent = right
                .iter()
                .filter(|l| !is_blank(l))
                .map(|l| l.iter().take_while(|c| c.is_whitespace()).count())
                .min()
                .unwrap_or(0);
            let right: Vec<Line> = right
                .iter()
                .map(|l| slice(l, indent.min(l.len()), usize::MAX))
                .collect();
            emit_columns(&right, depth + 1, out);
        }
        None => out.extend(
            lines
                .iter()
                .map(|l| l.iter().collect::<String>().trim_end().to_string()),
        ),
    }
}

/// Longest run of lines from `start` that keeps a common interior gutter
/// and splits into valid columns; returns its exclusive end.
fn find_region(lines: &[Line], start: usize) -> Option<usize> {
    let mut occ: Vec<bool> = Vec::new();
    let mut end = start;
    for (j, l) in lines.iter().enumerate().skip(start) {
        let mut next = occ.clone();
        if next.len() < l.len() {
            next.resize(l.len(), false);
        }
        for (c, ch) in l.iter().enumerate() {
            if !ch.is_whitespace() {
                next[c] = true;
            }
        }
        if gutters(&next).is_empty() && j > start {
            break;
        }
        occ = next;
        end = j + 1;
    }
    // Trailing blank lines belong to whatever follows.
    while end > start && is_blank(&lines[end - 1]) {
        end -= 1;
    }
    (end > start && split_block(&lines[start..end], 0).is_some()).then_some(end)
}

/// Rewrite side-by-side columns in `text` into column-major reading order.
/// Text without a detectable multi-column prose region is returned as is.
pub fn reflow_columns(text: &str) -> String {
    let lines: Vec<Line> = text.lines().map(|l| l.chars().collect()).collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;
    let mut changed = false;
    while i < lines.len() {
        if !is_blank(&lines[i]) {
            if let Some(end) = find_region(&lines, i) {
                emit_columns(&lines[i..end], 0, &mut out);
                changed = true;
                i = end;
                continue;
            }
        }
        out.push(lines[i].iter().collect());
        i += 1;
    }
    if changed {
        out.join("\n")
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_COL: &str = "\
Introduction to the topic here        Results were measured over
we describe a careful method          many runs and they show a
that is applied to real data          clear and steady gain over
and then evaluated at length.         the baseline in all cases.
Short closing line.                   Final words of column two.";

    #[test]
    fn two_columns_are_read_left_then_right() {
        let out = reflow_columns(TWO_COL);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "Introduction to the topic here");
        assert_eq!(lines[4], "Short closing line.");
        assert_eq!(lines[5], "Results were measured over");
        assert_eq!(lines[9], "Final words of column two.");
    }

    #[test]
    fn full_width_heading_stays_on_top() {
        let text = format!("            A Centered Title Spanning Both Columns\n\n{TWO_COL}");
        let out = reflow_columns(&text);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0].trim(), "A Centered Title Spanning Both Columns");
        assert_eq!(lines[2], "Introduction to the topic here");
        assert_eq!(lines[7], "Results were measured over");
    }

    #[test]
    fn three_columns_recurse() {
        let text = "\
alpha beta gamma delta eps   one two three four five six   red green blue cyan black
zeta eta theta iota kappa    seven eight nine ten eleven   white pink brown gray tan
lambda mu nu xi omicron pi   twelve thirteen fourteen ab   olive navy teal lime gold
rho sigma tau upsilon phi    fifteen sixteen seventeen x   plum ruby jade onyx opal";
        let out = reflow_columns(text);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 12);
        assert!(lines[0].starts_with("alpha"));
        assert!(lines[4].starts_with("one"));
        assert!(lines[8].starts_with("red"));
    }

    #[test]
    fn tables_are_left_alone() {
        let text = "\
Name          Qty     Price
Apples        10      1.20
Pears         4       0.80
Oranges       7       2.10
Bananas       12      0.30";
        assert_eq!(reflow_columns(text), text);
    }

    #[test]
    fn ragged_two_column_table_is_left_alone() {
        let text = "Area                                      Competence
1. Embodying sustainability values        1.1 Valuing sustainability
                                          1.2 Supporting fairness
                                          1.3 Promoting nature
2. Embracing complexity in                2.1 Systems thinking
sustainability
                                          2.2 Critical thinking
                                          2.3 Problem framing
3. Envisioning sustainable futures        3.1 Futures literacy
                                          3.2 Adaptability";
        assert_eq!(reflow_columns(text), text);
    }

    #[test]
    fn figure_gap_in_one_column_still_splits() {
        // The left column is interrupted by a figure (one long blank run)
        // while the right column keeps flowing; that is still prose.
        let mut text = String::new();
        let left = [
            "Introduction to the topic here",
            "we describe a careful method",
            "that is applied to real data",
            "and then evaluated at length.",
            "",
            "",
            "",
            "",
            "",
            "",
            "Figure caption under the plot",
            "and the column then resumes.",
        ];
        for (i, l) in left.iter().enumerate() {
            text.push_str(&format!(
                "{l:<38}right column line number {i:02} here
"
            ));
        }
        let out = reflow_columns(text.trim_end());
        let lines: Vec<&str> = out.lines().map(str::trim_end).collect();
        assert_eq!(lines[0], "Introduction to the topic here");
        assert_eq!(lines[11], "and the column then resumes.");
        assert_eq!(lines[12], "right column line number 00 here");
    }

    #[test]
    fn single_column_prose_is_unchanged() {
        let text = "This is a normal paragraph of text that\nwraps over a few lines without any gutter\nin the middle of it at all, as usual.";
        assert_eq!(reflow_columns(text), text);
    }
}
