//! Markdown formatter.
//!
//! The body of each page is the projected spatial text (so reading order is
//! identical to `--format text`). On top of that, lines typeset noticeably
//! larger than the document's body font, or entirely in bold when the body
//! is not bold, are promoted to ATX headings. Heading levels follow the
//! distinct heading font sizes, largest first.

use std::collections::HashMap;

use spdf_types::{ParseResult, ParsedPage, TextItem};

/// A line is a heading candidate when its font is at least this much larger
/// than the body font.
const SIZE_RATIO: f64 = 1.15;
/// Longest text (in chars) we will accept as a heading.
const MAX_HEADING_CHARS: usize = 150;
/// Most words we will accept as a heading.
const MAX_HEADING_WORDS: usize = 25;
/// Deepest level assigned to size-based headings.
const MAX_SIZE_LEVEL: usize = 4;

/// Document-wide typography statistics used to classify lines.
#[derive(Debug, Clone)]
pub struct DocStyle {
    /// Dominant (character-weighted) font size: the body text size.
    pub body_size: f64,
    /// Whether the body text itself is mostly bold (then bold is no signal).
    pub body_bold: bool,
    /// Distinct heading sizes (rounded to 0.5pt), largest first.
    pub heading_sizes: Vec<f64>,
}

fn round_half(v: f64) -> f64 {
    (v * 2.0).round() / 2.0
}

fn is_bold(it: &TextItem) -> bool {
    it.font_name.as_deref().is_some_and(|n| {
        let n = n.to_ascii_lowercase();
        ["bold", "black", "heavy", "semibold", "demi"]
            .iter()
            .any(|k| n.contains(k))
    })
}

fn usable(it: &TextItem) -> bool {
    !it.str.trim().is_empty()
        && it.r.unwrap_or(0) == 0
        && it.font_size.is_some_and(|s| s.is_finite() && s > 0.0)
}

fn visible_len(s: &str) -> usize {
    s.chars().filter(|c| !c.is_whitespace()).count()
}

impl DocStyle {
    /// Gather statistics over `pages`.
    pub fn from_pages<'a>(pages: impl IntoIterator<Item = &'a ParsedPage>) -> Self {
        let mut hist: HashMap<i64, usize> = HashMap::new();
        let mut bold_chars = 0usize;
        let mut total_chars = 0usize;
        let mut sizes: Vec<f64> = Vec::new();
        for page in pages {
            for it in page.text_items.iter().filter(|it| usable(it)) {
                let n = visible_len(&it.str);
                let size = round_half(it.font_size.unwrap_or_default());
                *hist.entry((size * 2.0) as i64).or_default() += n;
                total_chars += n;
                if is_bold(it) {
                    bold_chars += n;
                }
                sizes.push(size);
            }
        }
        // Ties resolve to the smaller size: body text is rarely the largest.
        let body_size = hist
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
            .map(|(k, _)| *k as f64 / 2.0)
            .unwrap_or(12.0);
        let mut heading_sizes: Vec<f64> = sizes
            .into_iter()
            .filter(|s| *s >= body_size * SIZE_RATIO)
            .collect();
        heading_sizes.sort_by(|a, b| b.total_cmp(a));
        heading_sizes.dedup();
        DocStyle {
            body_size,
            body_bold: total_chars > 0 && bold_chars * 2 > total_chars,
            heading_sizes,
        }
    }

    /// Heading level for an item, or `None` when it is body-styled.
    fn level_of(&self, it: &TextItem) -> Option<usize> {
        let size = round_half(it.font_size?);
        if size >= self.body_size * SIZE_RATIO {
            let idx = self
                .heading_sizes
                .iter()
                .position(|s| *s <= size)
                .unwrap_or(self.heading_sizes.len());
            return Some((idx + 1).min(MAX_SIZE_LEVEL));
        }
        if !self.body_bold && is_bold(it) && size >= self.body_size * 0.95 {
            return Some((self.heading_sizes.len().min(MAX_SIZE_LEVEL) + 1).min(6));
        }
        None
    }
}

/// Render a whole document as Markdown.
pub fn format_markdown(result: &ParseResult) -> String {
    let style = DocStyle::from_pages(&result.pages);
    result
        .pages
        .iter()
        .map(|p| page_to_markdown(p, &style))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn strip_ws(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Map from whitespace-stripped line text to heading level for every visual
/// line on the page made only of heading-styled items.
fn heading_lines(page: &ParsedPage, style: &DocStyle) -> HashMap<String, usize> {
    let mut items: Vec<(&TextItem, usize)> = page
        .text_items
        .iter()
        .filter(|it| usable(it))
        .filter_map(|it| style.level_of(it).map(|l| (it, l)))
        .collect();
    items.sort_by(|a, b| a.0.y.total_cmp(&b.0.y).then(a.0.x.total_cmp(&b.0.x)));

    // Greedy grouping into visual lines: same baseline band, same level,
    // horizontally close (gaps up to 4 em cover tab-separated numbering).
    struct Line<'a> {
        items: Vec<&'a TextItem>,
        level: usize,
        y_mid: f64,
        left: f64,
        right: f64,
        size: f64,
    }
    let mut lines: Vec<Line> = Vec::new();
    for (it, level) in items {
        let size = it.font_size.unwrap_or(it.height).max(1.0);
        let y_mid = it.y + it.height / 2.0;
        let found = lines.iter_mut().rev().take(8).find(|l| {
            l.level == level
                && (l.y_mid - y_mid).abs() < 0.5 * size.min(l.size)
                && it.x - l.right < 4.0 * size
                && l.left - (it.x + it.width) < 4.0 * size
        });
        match found {
            Some(l) => {
                l.left = l.left.min(it.x);
                l.right = l.right.max(it.x + it.width);
                l.items.push(it);
            }
            None => lines.push(Line {
                items: vec![it],
                level,
                y_mid,
                left: it.x,
                right: it.x + it.width,
                size,
            }),
        }
    }

    let mut out = HashMap::new();
    for mut l in lines {
        l.items.sort_by(|a, b| a.x.total_cmp(&b.x));
        let key: String = l.items.iter().map(|it| strip_ws(&it.str)).collect();
        out.entry(key)
            .and_modify(|lv: &mut usize| *lv = (*lv).min(l.level))
            .or_insert(l.level);
    }
    out
}

fn plausible_heading(text: &str) -> bool {
    let chars = text.chars().count();
    (2..=MAX_HEADING_CHARS).contains(&chars)
        && text.split_whitespace().count() <= MAX_HEADING_WORDS
        && text.chars().any(char::is_alphabetic)
        && !text.ends_with(',')
}

/// Tables on a page, in line order and non-overlapping: bordered tables
/// recovered from the page's rules first, then text-aligned tables found
/// in the remaining lines.
fn page_tables(page: &ParsedPage, lines: &[&str]) -> Vec<crate::md_table::MdTable> {
    let mut ruled = Vec::new();
    if !page.rulings.is_empty() {
        let mut from = 0;
        for t in crate::ruled_table::detect(&page.rulings, &page.text_items) {
            if let Some((first, last)) = crate::ruled_table::locate(lines, &t.words, from) {
                ruled.push(crate::md_table::MdTable {
                    first,
                    last,
                    rows: t.rows,
                });
                from = last + 1;
            }
        }
    }
    if ruled.is_empty() {
        return crate::md_table::find_tables(lines);
    }
    // Hide the ruled tables' lines from the alignment-based detector.
    let masked: Vec<&str> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| {
            if ruled.iter().any(|t| (t.first..=t.last).contains(&i)) {
                ""
            } else {
                *l
            }
        })
        .collect();
    let mut all = ruled;
    all.extend(crate::md_table::find_tables(&masked));
    all.sort_by_key(|t| t.first);
    all
}

/// Render a single page as Markdown using document-level `style`.
pub fn page_to_markdown(page: &ParsedPage, style: &DocStyle) -> String {
    let headings = heading_lines(page, style);
    let mut out: Vec<String> = Vec::new();
    // Level of the heading that is currently open (for multi-line headings).
    let mut open: Option<usize> = None;
    let text = crate::page_text(page);
    let lines: Vec<&str> = text.lines().collect();
    let tables = page_tables(page, &lines);
    let mut next_table = tables.iter().peekable();
    let mut idx = 0;
    while idx < lines.len() {
        if let Some(t) = next_table.next_if(|t| t.first == idx) {
            open = None;
            if out.last().is_some_and(|l| !l.is_empty()) {
                out.push(String::new());
            }
            out.push(crate::md_table::render(&t.rows));
            out.push(String::new());
            idx = t.last + 1;
            continue;
        }
        let raw = lines[idx];
        idx += 1;
        let line = raw.trim();
        if line.is_empty() {
            open = None;
            out.push(String::new());
            continue;
        }
        let text = collapse_ws(line);
        let level = headings
            .get(&strip_ws(line))
            .copied()
            .filter(|_| plausible_heading(&text));
        match level {
            Some(lv) if open == Some(lv) => {
                // Continuation of a heading wrapped over several lines.
                let last = out.last_mut().expect("open heading has a line");
                last.push(' ');
                last.push_str(&text);
            }
            Some(lv) => {
                if out.last().is_some_and(|l| !l.is_empty()) {
                    out.push(String::new());
                }
                out.push(format!("{} {}", "#".repeat(lv), text));
                open = Some(lv);
            }
            None => {
                if open.take().is_some() {
                    out.push(String::new());
                }
                if line.starts_with('#') {
                    out.push(format!("\\{line}"));
                } else {
                    out.push(line.to_string());
                }
            }
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(s: &str, x: f64, y: f64, size: f64, font: &str) -> TextItem {
        let mut it = TextItem::new(s, x, y, s.len() as f64 * size * 0.5, size);
        it.font_size = Some(size);
        it.font_name = Some(font.into());
        it
    }

    fn page(text: &str, items: Vec<TextItem>) -> ParsedPage {
        ParsedPage {
            page_num: 1,
            width: 612.0,
            height: 792.0,
            text: text.into(),
            text_items: items,
            bounding_boxes: None,
            tables: None,
            rulings: Vec::new(),
        }
    }

    fn body_items(n: usize, y0: f64) -> Vec<TextItem> {
        (0..n)
            .map(|i| {
                item(
                    "body text line with words",
                    50.0,
                    y0 + i as f64 * 12.0,
                    10.0,
                    "Times",
                )
            })
            .collect()
    }

    #[test]
    fn larger_font_lines_become_headings() {
        let mut items = vec![item("Introduction", 50.0, 40.0, 16.0, "Times")];
        items.extend(body_items(5, 70.0));
        let text = "Introduction\nbody text line with words\nbody text line with words";
        let p = page(text, items);
        let style = DocStyle::from_pages([&p]);
        assert_eq!(style.body_size, 10.0);
        let md = page_to_markdown(&p, &style);
        assert!(md.starts_with("# Introduction\n\nbody text"), "{md}");
    }

    #[test]
    fn levels_follow_size_order() {
        let mut items = vec![
            item("Title", 50.0, 20.0, 20.0, "Times"),
            item("Section", 50.0, 50.0, 14.0, "Times"),
            item("Sub", 50.0, 70.0, 10.0, "Times-Bold"),
        ];
        items.extend(body_items(6, 90.0));
        let p = page("Title\nSection\nSub\nbody text line with words", items);
        let style = DocStyle::from_pages([&p]);
        let md = page_to_markdown(&p, &style);
        assert!(md.contains("# Title\n"), "{md}");
        assert!(md.contains("## Section\n"), "{md}");
        assert!(md.contains("### Sub\n"), "{md}");
    }

    #[test]
    fn wrapped_heading_is_merged() {
        let mut items = vec![
            item("A Rather Long", 50.0, 20.0, 18.0, "Times"),
            item("Document Title", 50.0, 42.0, 18.0, "Times"),
        ];
        items.extend(body_items(6, 80.0));
        let p = page(
            "A Rather Long\nDocument Title\nbody text line with words",
            items,
        );
        let md = page_to_markdown(&p, &DocStyle::from_pages([&p]));
        assert!(md.starts_with("# A Rather Long Document Title\n"), "{md}");
    }

    #[test]
    fn mixed_lines_and_body_hash_are_not_headings() {
        let mut items = vec![
            item("Bold", 50.0, 20.0, 10.0, "Arial-BoldMT"),
            item("then normal", 80.0, 20.0, 10.0, "ArialMT"),
        ];
        items.extend(body_items(6, 50.0));
        let p = page(
            "Bold then normal\n# of rows\nbody text line with words",
            items,
        );
        let md = page_to_markdown(&p, &DocStyle::from_pages([&p]));
        assert!(md.starts_with("Bold then normal\n\\# of rows"), "{md}");
    }

    #[test]
    fn bold_body_gives_no_bold_headings() {
        let items: Vec<TextItem> = (0..5)
            .map(|i| {
                item(
                    "all bold text",
                    50.0,
                    20.0 + 12.0 * i as f64,
                    10.0,
                    "X-Bold",
                )
            })
            .collect();
        let p = page("all bold text\nall bold text", items);
        let md = page_to_markdown(&p, &DocStyle::from_pages([&p]));
        assert!(!md.contains('#'), "{md}");
    }
}
