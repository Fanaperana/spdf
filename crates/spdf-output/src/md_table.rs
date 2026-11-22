//! Table recovery from projected page text.
//!
//! The spatial projection already lays text out on a character grid, so a
//! table's columns show up as runs of text separated by wide (2+ space)
//! gutters that line up from one line to the next. This module finds such
//! blocks of lines, derives column spans from them, folds wrapped cell
//! lines (whose first column is empty) back into the row above, and renders
//! the result as a GitHub-flavoured pipe table.

/// Minimum run of spaces that separates two cells on a line.
const MIN_GUTTER: usize = 2;
/// Rows (after merging wrapped lines) a table needs.
const MIN_ROWS: usize = 2;
/// Lines with 2+ cells a block needs before it is called a table.
const MIN_MULTI_LINES: usize = 2;
/// Mean words per non-empty cell above which a block reads as prose laid
/// out in columns (e.g. a two-column article), not a table.
const MAX_MEAN_WORDS_PER_CELL: f64 = 6.0;
/// Share of grid cells that must hold text.
const MIN_FILL_RATIO: f64 = 0.5;
/// Max share of body cells starting in lowercase (sentence continuations).
const MAX_LOWERCASE_START: f64 = 0.3;

/// A run of text on one line: `[start, end)` in char columns.
#[derive(Debug, Clone)]
struct Seg {
    start: usize,
    end: usize,
    text: String,
}

fn segments(line: &str) -> Vec<Seg> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        let mut j = i;
        while j < chars.len() {
            if chars[j].is_whitespace() {
                let mut k = j;
                while k < chars.len() && chars[k].is_whitespace() {
                    k += 1;
                }
                if k - j >= MIN_GUTTER || k == chars.len() {
                    break;
                }
                j = k;
            } else {
                j += 1;
                end = j;
            }
        }
        out.push(Seg {
            start,
            end,
            text: chars[start..end].iter().collect(),
        });
        i = end;
    }
    out
}

fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// A detected table: line range `[first, last]` (inclusive) of the page
/// text it replaces, and its rows of cells (first row = header).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MdTable {
    pub first: usize,
    pub last: usize,
    pub rows: Vec<Vec<String>>,
}

/// Column spans of a block, kept sorted and disjoint.
struct Columns(Vec<(usize, usize)>);

impl Columns {
    /// Indices of the columns a span overlaps.
    fn hits(&self, s: (usize, usize)) -> Vec<usize> {
        (0..self.0.len())
            .filter(|&c| overlaps(self.0[c], s))
            .collect()
    }

    /// Can a multi-cell line join this block? Each of its cells must touch
    /// at most one column, no two cells may share a column, and it must
    /// line up with at least one existing column.
    fn accepts_multi(&self, segs: &[Seg]) -> bool {
        let mut used = vec![false; self.0.len()];
        let mut shared = 0;
        for s in segs {
            let h = self.hits((s.start, s.end));
            if h.len() > 1 {
                return false;
            }
            if let Some(&c) = h.first() {
                if used[c] {
                    return false;
                }
                used[c] = true;
                shared += 1;
            }
        }
        shared >= 1
    }

    fn add(&mut self, segs: &[Seg]) {
        for s in segs {
            let span = (s.start, s.end);
            match self.hits(span).first() {
                Some(&c) => {
                    let col = &mut self.0[c];
                    col.0 = col.0.min(span.0);
                    col.1 = col.1.max(span.1);
                }
                None => self.0.push(span),
            }
        }
        self.0.sort();
        // Widening may have made neighbours touch; merge them.
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.0.len());
        for &c in &self.0 {
            match merged.last_mut() {
                Some(m) if c.0 < m.1 => m.1 = m.1.max(c.1),
                _ => merged.push(c),
            }
        }
        self.0 = merged;
    }
}

/// Find tables in projected page `lines`, in order, non-overlapping.
pub(crate) fn find_tables(lines: &[&str]) -> Vec<MdTable> {
    let segs: Vec<Vec<Seg>> = lines.iter().map(|l| segments(l)).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if segs[i].len() < 2 {
            i += 1;
            continue;
        }
        // A rejected block is skipped whole: its sub-blocks share the
        // geometry that disqualified it.
        let (table, next) = grow_block(&segs, i);
        out.extend(table);
        i = next.max(i + 1);
    }
    out
}

/// Grow a table block starting at multi-cell line `start`. Returns the table
/// and the index of the first line after it.
fn grow_block(segs: &[Vec<Seg>], start: usize) -> (Option<MdTable>, usize) {
    let mut cols = Columns(Vec::new());
    cols.add(&segs[start]);
    let mut members = vec![start];
    let mut j = start + 1;
    while j < segs.len() {
        let s = &segs[j];
        if s.is_empty() {
            // A single blank line may separate rows of a loosely set table.
            let next = j + 1;
            if next < segs.len() && segs[next].len() >= 2 && cols.accepts_multi(&segs[next]) {
                j += 1;
                continue;
            }
            break;
        }
        if s.len() >= 2 {
            if !cols.accepts_multi(s) {
                break;
            }
            cols.add(s);
        } else {
            // One cell: must sit inside exactly one column.
            let h = cols.hits((s[0].start, s[0].end));
            if h.len() != 1 {
                break;
            }
        }
        members.push(j);
        j += 1;
    }
    // Trailing single-cell lines in the first column read as text after the
    // table, not as wrapped cells.
    while members.len() > 1 {
        let last = *members.last().unwrap();
        let s = &segs[last];
        if s.len() == 1 && cols.hits((s[0].start, s[0].end)) == [0] {
            members.pop();
        } else {
            break;
        }
    }
    let next = members.last().map_or(start + 1, |&m| m + 1);
    let multi = members.iter().filter(|&&m| segs[m].len() >= 2).count();
    if multi < MIN_MULTI_LINES || cols.0.len() < 2 {
        return (None, next);
    }

    let ncols = cols.0.len();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for &m in &members {
        let mut cells = vec![String::new(); ncols];
        for s in &segs[m] {
            let c = nearest_column(&cols, s);
            if !cells[c].is_empty() {
                cells[c].push(' ');
            }
            cells[c].push_str(&s.text);
        }
        let continuation = cells[0].is_empty() && !rows.is_empty();
        if continuation {
            let prev = rows.last_mut().unwrap();
            for (p, c) in prev.iter_mut().zip(cells) {
                if c.is_empty() {
                    continue;
                }
                if !p.is_empty() {
                    p.push(' ');
                }
                p.push_str(&c);
            }
        } else {
            rows.push(cells);
        }
    }
    if rows.len() < MIN_ROWS {
        return (None, next);
    }
    let (words, filled) = rows
        .iter()
        .flatten()
        .filter(|c| !c.is_empty())
        .fold((0usize, 0usize), |(w, n), c| {
            (w + c.split_whitespace().count(), n + 1)
        });
    if filled == 0 || words as f64 / filled as f64 > MAX_MEAN_WORDS_PER_CELL {
        return (None, next);
    }
    if !looks_tabular(&rows) {
        return (None, next);
    }
    let table = MdTable {
        first: start,
        last: next - 1,
        rows,
    };
    (Some(table), next)
}

/// Reject grids that are aligned but are not data tables: chart axes and
/// scattered labels (mostly empty cells), prose set in narrow columns
/// (cells continue sentences, so many start in lowercase), and tables of
/// contents (a title column followed by a page-number column).
fn looks_tabular(rows: &[Vec<String>]) -> bool {
    let ncols = rows[0].len();
    let cells = rows.len() * ncols;
    let filled: Vec<&String> = rows.iter().flatten().filter(|c| !c.is_empty()).collect();
    if (filled.len() as f64) < MIN_FILL_RATIO * cells as f64 {
        return false;
    }
    let body: Vec<&String> = rows[1..]
        .iter()
        .flatten()
        .filter(|c| !c.is_empty())
        .collect();
    let lower = body
        .iter()
        .filter(|c| c.chars().next().is_some_and(char::is_lowercase))
        .count();
    if !body.is_empty() && lower as f64 > MAX_LOWERCASE_START * body.len() as f64 {
        return false;
    }
    if ncols == 2 {
        let page_ref = rows
            .iter()
            .filter(|r| r[1].is_empty() || is_page_number(&r[1]))
            .count();
        if page_ref * 10 >= rows.len() * 8 {
            return false;
        }
    }
    // A first column made only of bullets / list markers is a list.
    if rows
        .iter()
        .all(|r| r[0].is_empty() || is_list_marker(&r[0]))
    {
        return false;
    }
    true
}

fn is_list_marker(s: &str) -> bool {
    let s = s.trim();
    if !s.chars().any(char::is_alphanumeric) {
        return s.chars().count() <= 2;
    }
    let core = s.trim_start_matches('(');
    let Some(stripped) = core.strip_suffix(['.', ')']) else {
        return false;
    };
    !stripped.is_empty()
        && stripped.chars().count() <= 3
        && stripped.chars().all(char::is_alphanumeric)
}

fn is_page_number(s: &str) -> bool {
    let s = s.trim();
    (!s.is_empty() && s.len() <= 4 && s.chars().all(|c| c.is_ascii_digit()))
        || (!s.is_empty() && s.len() <= 6 && s.chars().all(|c| "ivxlcdmIVXLCDM".contains(c)))
}

fn nearest_column(cols: &Columns, s: &Seg) -> usize {
    let span = (s.start, s.end);
    let dist = |c: (usize, usize)| -> usize {
        if overlaps(c, span) {
            0
        } else if span.1 <= c.0 {
            c.0 - span.1
        } else {
            span.0 - c.1
        }
    };
    (0..cols.0.len())
        .min_by_key(|&c| dist(cols.0[c]))
        .unwrap_or(0)
}

fn escape_cell(c: &str) -> String {
    c.replace('|', "\\|")
}

/// Render rows (first = header) as a pipe table.
pub(crate) fn render(rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    for (i, row) in rows.iter().enumerate() {
        out.push('|');
        for c in row {
            out.push(' ');
            out.push_str(&escape_cell(c));
            out.push_str(" |");
        }
        out.push('\n');
        if i == 0 {
            out.push('|');
            for _ in row {
                out.push_str("---|");
            }
            out.push('\n');
        }
    }
    out.pop();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_wide_gaps_only() {
        let s = segments("No.   Name of org     12,3");
        let t: Vec<_> = s.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(t, ["No.", "Name of org", "12,3"]);
    }

    #[test]
    fn wrapped_cells_fold_into_row_above() {
        let lines = [
            "Intro text that precedes the table.",
            "No.   Name of organization            Number of accredited",
            "                                      observers",
            "1     Union of Youth Federations           17,266",
            "      (UYFC)",
            "2     Women for Peace                       9,835",
            "Total                                      27,926",
            "15 https://example.org/some/footnote/link/here/long",
        ];
        let t = find_tables(&lines);
        assert_eq!(t.len(), 1, "{t:?}");
        let t = &t[0];
        assert_eq!((t.first, t.last), (1, 6));
        assert_eq!(
            t.rows[0],
            [
                "No.",
                "Name of organization",
                "Number of accredited observers"
            ]
        );
        assert_eq!(
            t.rows[1],
            ["1", "Union of Youth Federations (UYFC)", "17,266"]
        );
        assert_eq!(t.rows[3], ["Total", "", "27,926"]);
    }

    #[test]
    fn two_column_prose_is_not_a_table() {
        let lines = [
            "The quick brown fox jumps over the lazy    Lorem ipsum dolor sit amet consectetur",
            "dog and keeps running far into the night    adipiscing elit sed do eiusmod tempor",
            "while the moon rises over the quiet hill    incididunt ut labore et dolore magna",
        ];
        assert!(find_tables(&lines).is_empty());
    }

    #[test]
    fn table_of_contents_is_not_a_table() {
        let lines = [
            "Introduction                         7",
            "1. Changing Practices                9",
            "Part I: New Children                21",
            "Appendix                           xii",
        ];
        assert!(find_tables(&lines).is_empty());
    }

    #[test]
    fn sparse_chart_axis_is_not_a_table() {
        let lines = [
            "3,500    3,230   3,288",
            "3,000",
            "2,500                     2,907",
            "2,000",
            "1,500    1,100",
            "1,000",
        ];
        assert!(find_tables(&lines).is_empty());
    }

    #[test]
    fn narrow_prose_columns_are_not_a_table() {
        let lines = [
            "In this            article or book are correct. For",
            "context, we are    example, they may contact a person",
            "talking about      who is quoted in a proposed news",
            "fact-checking      article and ask the person whether",
        ];
        assert!(find_tables(&lines).is_empty());
    }

    #[test]
    fn bullet_list_is_not_a_table() {
        let lines = [
            "•    Liquid biomass: palm oil",
            "•    Unutilised wood: domestic thinned wood",
            "•    Biogas: methane from sludge",
        ];
        assert!(find_tables(&lines).is_empty());
        let lines = ["1.    First item", "2.    Second item", "3.    Third item"];
        assert!(find_tables(&lines).is_empty());
    }

    #[test]
    fn single_aligned_line_is_not_a_table() {
        let lines = ["Name     Value", "plain prose follows here"];
        assert!(find_tables(&lines).is_empty());
    }

    #[test]
    fn renders_pipe_table() {
        let rows = vec![
            vec!["a".to_string(), "b|c".to_string()],
            vec!["1".to_string(), "".to_string()],
        ];
        assert_eq!(render(&rows), "| a | b\\|c |\n|---|---|\n| 1 |  |");
    }
}
