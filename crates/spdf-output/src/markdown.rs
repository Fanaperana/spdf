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
    it.font_name.as_deref().is_some_and(is_bold_name)
}

/// Whether a PostScript font name denotes a bold face. Besides the usual
/// weight words this knows the TeX naming scheme, where weight is encoded
/// in the family abbreviation (`CMBX12`, `CMB10`, `CMSSBX10`, `ECBX1200`).
fn is_bold_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    // Drop a subset tag such as `ABCDEF+`.
    let base = match n.split_once('+') {
        Some((tag, rest)) if tag.len() == 6 => rest,
        _ => n.as_str(),
    };
    if ["bold", "black", "heavy", "semibold", "demi"]
        .iter()
        .any(|k| base.contains(k))
    {
        return true;
    }
    const TEX_BOLD: [&str; 6] = ["cmbx", "cmb", "cmssbx", "ecbx", "ecrb", "ecsx"];
    TEX_BOLD.iter().any(|p| {
        base.strip_prefix(p)
            .is_some_and(|rest| rest.chars().next().is_some_and(|c| c.is_ascii_digit()))
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
    let mut items: Vec<(&TextItem, usize)> = Vec::new();
    let mut others: Vec<&TextItem> = Vec::new();
    for it in page.text_items.iter().filter(|it| usable(it)) {
        match style.level_of(it) {
            Some(l) => items.push((it, l)),
            None => others.push(it),
        }
    }
    let heading_count = items.len();
    items.extend(small_caps(&items, &others));
    let is_heading = |it: &TextItem| {
        items[..heading_count]
            .iter()
            .any(|(h, _)| std::ptr::eq(*h, it))
    };
    let lines = group_lines(items.clone());
    let mut out = HashMap::new();
    for (line, level) in lines {
        // Lines made only of small-cap letters carry no heading signal.
        if !line.iter().any(|it| is_heading(it)) {
            continue;
        }
        let key: String = line.iter().map(|it| strip_ws(&it.str)).collect();
        out.entry(key)
            .and_modify(|lv: &mut usize| *lv = (*lv).min(level))
            .or_insert(level);
    }
    out
}

/// Small-capital companions of heading items. Small-caps headings typeset
/// each word's initial at the heading size and the remaining capitals
/// smaller, in the same font. Such reduced items are adopted at the level
/// of the heading font they belong to, so the whole line is recognised.
fn small_caps<'a>(
    headings: &[(&'a TextItem, usize)],
    others: &[&'a TextItem],
) -> Vec<(&'a TextItem, usize)> {
    if headings.is_empty() {
        return Vec::new();
    }
    others
        .iter()
        .filter(|it| {
            let letters = it.str.chars().filter(|c| c.is_alphabetic());
            let mut any = false;
            for c in letters {
                if !c.is_uppercase() {
                    return false;
                }
                any = true;
            }
            any
        })
        .filter_map(|it| {
            let size = it.font_size?;
            headings
                .iter()
                .filter(|(h, _)| {
                    let hs = h.font_size.unwrap_or_default();
                    h.font_name == it.font_name && size < hs && size >= 0.6 * hs
                })
                .map(|(_, lv)| *lv)
                .min()
                .map(|lv| (*it, lv))
        })
        .collect()
}

/// Group heading-styled items into visual lines, each sorted left to right.
///
/// Items of one level are first clustered into baseline bands (vertical
/// centres within half an em of the band's first item), so sub-point
/// baseline jitter between words of the same line cannot interleave them
/// with other fragments. Each band is then split wherever the horizontal
/// gap exceeds 4 em (wider than tab-separated numbering, narrower than a
/// column gutter), keeping side-by-side headings apart.
fn group_lines(mut items: Vec<(&TextItem, usize)>) -> Vec<(Vec<&TextItem>, usize)> {
    let size_of = |it: &TextItem| it.font_size.unwrap_or(it.height).max(1.0);
    let mid = |it: &TextItem| it.y + it.height / 2.0;
    items.sort_by(|a, b| a.1.cmp(&b.1).then(mid(a.0).total_cmp(&mid(b.0))));

    let mut bands: Vec<(Vec<&TextItem>, usize)> = Vec::new();
    for (it, level) in items {
        let joins = bands.last().is_some_and(|(band, lv)| {
            let first = band[0];
            *lv == level && (mid(first) - mid(it)).abs() < 0.5 * size_of(it).min(size_of(first))
        });
        if joins {
            bands.last_mut().expect("band exists").0.push(it);
        } else {
            bands.push((vec![it], level));
        }
    }

    let mut lines = Vec::new();
    for (mut band, level) in bands {
        band.sort_by(|a, b| a.x.total_cmp(&b.x));
        let mut cur: Vec<&TextItem> = Vec::new();
        let mut right = f64::NEG_INFINITY;
        for it in band {
            if !cur.is_empty() && it.x - right >= 4.0 * size_of(it) {
                lines.push((std::mem::take(&mut cur), level));
                right = f64::NEG_INFINITY;
            }
            right = right.max(it.x + it.width);
            cur.push(it);
        }
        if !cur.is_empty() {
            lines.push((cur, level));
        }
    }
    // Restore top-to-bottom order so the first occurrence of a key wins
    // consistently.
    lines.sort_by(|a, b| {
        a.0[0]
            .y
            .total_cmp(&b.0[0].y)
            .then(a.0[0].x.total_cmp(&b.0[0].x))
    });
    lines
}

/// Heading level of a projected text line, if any. A trailing footnote
/// marker (`Title 2`, `Title*`) is typeset smaller than the heading, so the
/// line is also matched with that marker removed.
fn heading_level(headings: &HashMap<String, usize>, line: &str) -> Option<usize> {
    if let Some(lv) = headings.get(&strip_ws(line)) {
        return Some(*lv);
    }
    let trimmed = line.trim_end();
    let is_marker = |c: char| c.is_ascii_digit() || "*∗†‡§".contains(c);
    let marker_len: usize = trimmed
        .chars()
        .rev()
        .take_while(|c| is_marker(*c))
        .map(char::len_utf8)
        .sum();
    let marker_chars = trimmed[trimmed.len() - marker_len..].chars().count();
    if marker_len == 0 || marker_chars > 2 {
        return None;
    }
    let rest = &trimmed[..trimmed.len() - marker_len];
    if !rest.ends_with(|c: char| c.is_alphabetic() || c.is_whitespace()) {
        return None;
    }
    let key = strip_ws(rest);
    // A bare number is not a heading with a footnote.
    if !key.chars().any(char::is_alphabetic) {
        return None;
    }
    headings.get(&key).copied()
}

/// Words that introduce a figure/table caption when followed by an
/// identifier (`Figure 2.1:`, `Table 13.2.`, `Fig. 4`). Captions are often
/// set bold or larger, but they label a float rather than open a section.
const CAPTION_WORDS: [&str; 14] = [
    "figure",
    "fig",
    "table",
    "tab",
    "chart",
    "exhibit",
    "graph",
    "plate",
    "map",
    "diagram",
    "scheme",
    "listing",
    "illustration",
    "photo",
];

/// Whether `tok` looks like a caption number: `3`, `2.1:`, `IV`, `A.1`, `S2`.
fn is_caption_id(tok: &str) -> bool {
    let t = tok.trim_end_matches([':', '.', ')', '-', '–', '—']);
    if t.is_empty() {
        return false;
    }
    let first = t.chars().next().unwrap_or(' ');
    if first.is_ascii_digit() {
        return t
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || ".-–".contains(c));
    }
    if t.chars().all(|c| "IVXLivxl".contains(c)) && t.len() <= 5 {
        return true;
    }
    // A letter-prefixed number such as `A.1` or `S2`.
    let mut cs = t.chars();
    cs.next();
    let rest: String = cs.collect();
    first.is_ascii_uppercase()
        && rest
            .trim_start_matches('.')
            .starts_with(|c: char| c.is_ascii_digit())
}

/// A figure or table caption such as `Figure 2.1: Surveyed firms by size`.
fn is_caption(text: &str) -> bool {
    let mut words = text.split_whitespace();
    let (Some(w0), Some(w1)) = (words.next(), words.next()) else {
        return false;
    };
    let kw = w0.trim_end_matches('.').to_lowercase();
    CAPTION_WORDS.contains(&kw.as_str()) && is_caption_id(w1)
}

/// Whether a word is a list/section enumerator: `a.`, `3.`, `2.1.`, `(iv)`, `IV.`.
fn is_enumerator(w: &str) -> bool {
    let core = w.trim_start_matches('(').trim_end_matches(['.', ')']);
    !core.is_empty()
        && w.len() > core.len()
        && (core.chars().all(|c| c.is_ascii_digit() || c == '.')
            || (core.chars().count() == 1 && core.chars().all(|c| c.is_alphabetic()))
            || core.chars().all(|c| "IVXivx".contains(c)))
}

/// A run-in heading: a short bold lead sentence followed on the same line
/// by body text (`d. Awareness of X. About 68% of respondents know ...`).
/// The line is a paragraph opening, not a section heading.
fn is_run_in(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    let start = usize::from(words.first().is_some_and(|w| is_enumerator(w)));
    for i in start + 1..words.len().saturating_sub(1) {
        let w = words[i];
        if !(w.ends_with('.') || w.ends_with(':')) {
            continue;
        }
        // Skip abbreviations and initials (`U.S.`, `e.g.`, `Dr.`).
        let core = w.trim_end_matches(['.', ':']);
        if core.contains('.') || core.chars().count() <= 2 {
            continue;
        }
        let before = i + 1 - start;
        let after = &words[i + 1..];
        let next_starts_sentence = after[0]
            .chars()
            .next()
            .is_some_and(|c| c.is_uppercase() || c.is_ascii_digit());
        if before >= 2 && after.len() >= 4 && next_starts_sentence {
            return true;
        }
    }
    false
}

/// Whether a line opens in the middle of a sentence: its first word starts
/// with a lowercase letter (`and the proliferation of ...`). Enumerators
/// (`a.`, `iv)`) and camel-cased names (`iPhone`, `eBay`) are exempt.
fn starts_mid_sentence(text: &str) -> bool {
    let Some(first) = text.split_whitespace().next() else {
        return false;
    };
    if is_enumerator(first) {
        return false;
    }
    let mut cs = first.chars();
    let starts_lower = cs.next().is_some_and(char::is_lowercase);
    let camel = cs.next().is_some_and(char::is_uppercase);
    starts_lower && !camel
}

/// Whether a line of body text stops mid-sentence: it ends in a lowercase
/// word or a comma, so the next line continues the same sentence.
fn ends_mid_sentence(text: &str) -> bool {
    text.trim_end()
        .chars()
        .last()
        .is_some_and(|c| c.is_lowercase() || c == ',')
}

/// Whether a line closes a sentence (`... and the United Kingdom.`),
/// ignoring trailing closing quotes, brackets and footnote digits.
fn ends_sentence(text: &str) -> bool {
    text.trim_end()
        .trim_end_matches(|c: char| c.is_ascii_digit() || "\"'”’)]".contains(c))
        .ends_with(['.', '!', '?'])
}

/// A heading-styled line that is really the tail of the running paragraph:
/// it starts in lowercase, or it finishes a sentence the previous body line
/// left open. Emphasised words (a bold list of names, an italic title)
/// wrapping onto their own line look exactly like a short heading.
fn continues_paragraph(prev_body: Option<&str>, text: &str) -> bool {
    starts_mid_sentence(text) || (prev_body.is_some_and(ends_mid_sentence) && ends_sentence(text))
}

fn plausible_heading(text: &str) -> bool {
    !is_caption(text) && !is_run_in(text) && !is_date_line(text) && sized_like_heading(text)
}

/// Whether a line carries a web address. Footnotes, source credits and
/// contact lines cite URLs; section headings do not.
fn has_url(text: &str) -> bool {
    let t = text.to_lowercase();
    t.contains("://")
        || t.split_whitespace().any(|w| {
            w.trim_start_matches(['(', '<', '"', '“'])
                .starts_with("www.")
        })
}

/// Whether a word names a calendar month (`June`, `Sept.`, `oct`).
fn is_month(w: &str) -> bool {
    const MONTHS: [&str; 12] = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let w = w.trim_end_matches(['.', ',']).to_lowercase();
    w.len() >= 3
        && MONTHS
            .iter()
            .any(|m| *m == w || (w.len() <= 4 && m.starts_with(w.as_str())))
}

/// A line that is only a date (`June 2023`, `March 15, 2021`,
/// `1 October 2020`). Cover pages and letterheads set the issue date large
/// or bold, but it labels the document rather than opening a section.
fn is_date_line(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if !(2..=4).contains(&words.len()) {
        return false;
    }
    let mut months = 0;
    let mut year = false;
    for w in &words {
        let core = w.trim_end_matches([',', '.']);
        if is_month(w) {
            months += 1;
        } else if core.len() == 4 && core.chars().all(|c| c.is_ascii_digit()) {
            year = true;
        } else if !(core.len() <= 2 && !core.is_empty() && core.chars().all(|c| c.is_ascii_digit()))
        {
            return false;
        }
    }
    months == 1 && year
}

/// Whether a line is the title of a table of contents (`Contents`,
/// `Table of Contents`), in a few common languages.
fn is_toc_title(text: &str) -> bool {
    const TITLES: [&str; 9] = [
        "contents",
        "table of contents",
        "inhalt",
        "inhaltsverzeichnis",
        "table des matières",
        "sommaire",
        "índice",
        "indice",
        "目录",
    ];
    let t = collapse_ws(text.trim().trim_end_matches(':')).to_lowercase();
    TITLES.contains(&t.as_str())
}

/// Whether the two non-blank lines after `idx` are contents entries.
fn opens_toc(lines: &[&str], idx: usize) -> bool {
    let mut next = lines[idx + 1..].iter().filter(|l| !l.trim().is_empty());
    matches!((next.next(), next.next()), (Some(a), Some(b)) if is_toc_entry(a) && is_toc_entry(b))
}

fn sized_like_heading(text: &str) -> bool {
    let chars = text.chars().count();
    (2..=MAX_HEADING_CHARS).contains(&chars)
        && text.split_whitespace().count() <= MAX_HEADING_WORDS
        && text.chars().any(char::is_alphabetic)
        && !text.ends_with(',')
}

/// Whether a word is a page locator in a table of contents: an arabic page
/// number (`17`) or a lowercase roman numeral (`vii`) for front matter.
fn is_page_locator(w: &str) -> bool {
    let n = w.chars().count();
    (1..=4).contains(&n) && w.chars().all(|c| c.is_ascii_digit())
        || (1..=6).contains(&n) && w.chars().all(|c| "ivxlc".contains(c))
}

/// Whether a dot leader (`.....`, `…`) fills the space before a locator.
fn has_dot_leader(text: &str) -> bool {
    text.contains("...") || text.contains('…')
}

/// A table-of-contents entry: a title followed, after a dot leader or a
/// wide gap, by a page locator (`Introduction ........ 1`,
/// `Acknowledgments          vii`).
fn is_toc_entry(raw: &str) -> bool {
    let line = raw.trim();
    let Some((head, last)) = line.rsplit_once(|c: char| c.is_whitespace() || c == '.' || c == '…')
    else {
        return false;
    };
    if !is_page_locator(last) || !head.chars().any(char::is_alphabetic) {
        return false;
    }
    has_dot_leader(head) || head.ends_with("  ")
}

/// Whether line `idx` sits inside a table of contents: the nearest
/// non-blank lines above and below are both contents entries. Bold or
/// enlarged part titles inside a contents listing repeat the document's
/// headings rather than open a section, so they stay plain text.
fn inside_toc(lines: &[&str], idx: usize) -> bool {
    let nonblank = |l: &&&str| !l.trim().is_empty();
    let prev = lines[..idx].iter().rev().find(nonblank);
    let next = lines[idx + 1..].iter().find(nonblank);
    matches!((prev, next), (Some(p), Some(n)) if is_toc_entry(p) && is_toc_entry(n))
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

fn push_heading(out: &mut Vec<String>, level: usize, text: &str) {
    if out.last().is_some_and(|l| !l.is_empty()) {
        out.push(String::new());
    }
    out.push(format!("{} {}", "#".repeat(level), text));
}

/// Render a single page as Markdown using document-level `style`.
pub fn page_to_markdown(page: &ParsedPage, style: &DocStyle) -> String {
    let headings = heading_lines(page, style);
    let mut out: Vec<String> = Vec::new();
    // Level of the heading that is currently open (for multi-line headings).
    let mut open: Option<usize> = None;
    // Heading style of the previous line when it was rejected as a heading.
    let mut suppressed: Option<usize> = None;
    // The body line directly above (no blank line in between), if any.
    let mut prev_body: Option<&str> = None;
    let text = crate::page_text(page);
    let lines: Vec<&str> = text.lines().collect();
    let tables = page_tables(page, &lines);
    let mut next_table = tables.iter().peekable();
    let mut idx = 0;
    while idx < lines.len() {
        if let Some(t) = next_table.next_if(|t| t.first == idx) {
            open = None;
            suppressed = None;
            prev_body = None;
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
            suppressed = None;
            prev_body = None;
            out.push(String::new());
            continue;
        }
        let text = collapse_ws(line);
        let styled = heading_level(&headings, line);
        // A long heading may wrap after a comma; a line ending in one is
        // only a heading when the next line continues in the same style.
        let wraps_on = styled.is_some()
            && text.ends_with(',')
            && lines
                .get(idx)
                .is_some_and(|next| heading_level(&headings, next) == styled);
        let candidate = if wraps_on {
            text.trim_end_matches(',')
        } else {
            text.as_str()
        };
        let mut level = styled.filter(|lv| {
            plausible_heading(candidate)
                // A URL rules out a new heading; on a continuation line it
                // is handled below by demoting the open heading.
                && (open == Some(*lv) || !has_url(candidate))
                && !(is_toc_entry(raw) && has_dot_leader(raw))
                && !inside_toc(&lines, idx - 1)
                // A wrapped heading may continue in lowercase.
                && (open == Some(*lv) || !continues_paragraph(prev_body, &text))
        });
        // The wrapped remainder of a rejected caption or run-in heading
        // shares its style; keep it with the body too.
        if level.is_some() && level == suppressed {
            level = None;
        }
        // The title of a table of contents opens a top-level section even
        // when it is set in the body font.
        if level.is_none() && open.is_none() && is_toc_title(&text) && opens_toc(&lines, idx - 1) {
            level = Some(1);
        }
        suppressed = match (styled, level) {
            (Some(lv), None) if suppressed.is_none_or(|s| s == lv) => Some(lv),
            _ => None,
        };
        match level {
            Some(lv) if open == Some(lv) && has_url(&text) => {
                // A heading-styled block that runs on into a URL is a
                // footnote or source credit set large, not a heading: give
                // its opening line back to the body.
                if let Some(last) = out.last_mut() {
                    *last = last.trim_start_matches('#').trim_start().to_string();
                }
                open = None;
                suppressed = Some(lv);
                prev_body = Some(line);
                out.push(line.to_string());
            }
            Some(lv) if open == Some(lv) => {
                // Continuation of a heading wrapped over several lines.
                let last = out.last_mut().expect("open heading has a line");
                last.push(' ');
                last.push_str(&text);
                prev_body = None;
            }
            Some(lv) => {
                push_heading(&mut out, lv, &text);
                open = Some(lv);
                prev_body = None;
            }
            None => {
                prev_body = Some(line);
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
    fn part_titles_inside_contents_are_not_headings() {
        let mut items = vec![
            item("Contents", 50.0, 20.0, 18.0, "Times"),
            item("Part I. Exploring Data", 50.0, 70.0, 10.0, "Times-Bold"),
            item("Part II. Testing", 50.0, 106.0, 10.0, "Times-Bold"),
        ];
        items.extend(body_items(8, 130.0));
        let text = "Contents\n\nIntroduction          1\nPart I. Exploring Data\n\
                    Section 1.1: Data ......... 3\nPreface ........ vii\nPart II. Testing\n\
                    Section 2.1: p Values ...... 12\nPart II. Testing\n\nbody text line with words";
        let p = page(text, items);
        let md = page_to_markdown(&p, &DocStyle::from_pages([&p]));
        assert!(md.starts_with("# Contents\n"), "{md}");
        assert!(md.contains("\nPart I. Exploring Data\n"), "{md}");
        assert!(md.contains("\nPart II. Testing\nSection 2.1"), "{md}");
        // Past the listing the same styled line is a heading again.
        assert!(md.contains("\n## Part II. Testing\n"), "{md}");
    }

    #[test]
    fn date_lines_are_not_headings() {
        assert!(is_date_line("June 2023"));
        assert!(is_date_line("March 15, 2021"));
        assert!(is_date_line("1 Oct. 2020"));
        assert!(!is_date_line("2023"));
        assert!(!is_date_line("May Day 2023"));
        assert!(!is_date_line("Results for June 2023"));
        assert!(!is_date_line("Marching 2023"));
        assert!(!plausible_heading("June 2023"));
        assert!(plausible_heading("June Elections"));
    }

    #[test]
    fn body_styled_contents_title_becomes_heading() {
        let text = "Table of Contents
Introduction .......... 3
Methods .......... 7
Table of Contents";
        let items = vec![
            item("Table of Contents", 50.0, 50.0, 12.0, "Arial"),
            item("Introduction .......... 3", 50.0, 70.0, 12.0, "Arial"),
            item("Methods .......... 7", 50.0, 90.0, 12.0, "Arial"),
        ];
        let p = page(text, items);
        let style = DocStyle::from_pages([&p]);
        let md = page_to_markdown(&p, &style);
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[0], "# Table of Contents");
        assert!(lines.contains(&"Introduction .......... 3"));
        // Without contents entries after it, the phrase stays body text.
        assert_eq!(*lines.last().unwrap(), "Table of Contents");
        assert!(is_toc_title("CONTENTS"));
        assert!(!is_toc_title("Contents of the box"));
    }

    #[test]
    fn toc_entry_predicates() {
        assert!(is_toc_entry("Introduction          1"));
        assert!(is_toc_entry("Acknowledgments     vii"));
        assert!(is_toc_entry("Lab worksheet..........3"));
        assert!(is_toc_entry("Cellular Biology……32"));
        assert!(!is_toc_entry("Revenue grew by 12"));
        assert!(!is_toc_entry("Table 3"));
        assert!(!is_toc_entry("2024"));
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
    fn heading_wrapped_after_comma_is_merged() {
        let mut items = vec![
            item("We offer three packs for data,", 50.0, 20.0, 18.0, "Times"),
            item("making a real impact", 50.0, 42.0, 18.0, "Times"),
        ];
        items.extend(body_items(6, 80.0));
        let p = page(
            "We offer three packs for data,\nmaking a real impact\nbody text line with words",
            items,
        );
        let md = page_to_markdown(&p, &DocStyle::from_pages([&p]));
        assert!(
            md.starts_with("# We offer three packs for data, making a real impact\n"),
            "{md}"
        );
    }

    #[test]
    fn comma_line_without_styled_continuation_is_body() {
        let mut items = vec![item("Alpha, Beta, Gamma,", 50.0, 20.0, 18.0, "Times")];
        items.extend(body_items(6, 50.0));
        let p = page(
            "Alpha, Beta, Gamma,\nbody text line with words\nbody text line with words",
            items,
        );
        let md = page_to_markdown(&p, &DocStyle::from_pages([&p]));
        assert!(md.starts_with("Alpha, Beta, Gamma,\n"), "{md}");
    }

    #[test]
    fn asterisk_operator_footnote_marker_keeps_heading() {
        let mut h = HashMap::new();
        h.insert(strip_ws("3.7.3 Formulae of higher accuracy"), 1);
        assert_eq!(
            heading_level(&h, "3.7.3 Formulae of higher accuracy \u{2217}"),
            Some(1)
        );
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

    #[test]
    fn jittered_baseline_words_form_one_heading() {
        // Words of one bold line whose baselines differ by a fraction of a
        // point must not be split into two interleaved fragments.
        let mut items = vec![
            item("PRICE", 57.0, 502.81, 11.0, "Lato-Bold"),
            item("CEBO", 177.0, 502.81, 11.0, "Lato-Bold"),
            item("EFFECT", 215.0, 502.81, 11.0, "Lato-Bold"),
            item("AND", 96.0, 502.90, 11.0, "Lato-Bold"),
            item("THE", 127.0, 502.90, 11.0, "Lato-Bold"),
            item("PLA", 155.0, 502.90, 11.0, "Lato-Bold"),
        ];
        items.extend((0..8).map(|i| {
            item(
                "body text line with words",
                57.0,
                520.0 + i as f64 * 13.0,
                11.0,
                "Crimson-Roman",
            )
        }));
        let p = page(
            "PRICE AND THE PLACEBO EFFECT
body text line with words",
            items,
        );
        let style = DocStyle::from_pages([&p]);
        let md = page_to_markdown(&p, &style);
        assert!(
            md.starts_with(
                "# PRICE AND THE PLACEBO EFFECT
"
            ),
            "{md}"
        );
    }

    #[test]
    fn distant_fragments_on_one_baseline_stay_apart() {
        let items = [
            item("Left", 50.0, 20.0, 10.0, "Times-Bold"),
            item("Right", 400.0, 20.0, 10.0, "Times-Bold"),
        ];
        let lines = group_lines(items.iter().map(|it| (it, 1)).collect());
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn tex_bold_font_names_are_bold() {
        assert!(is_bold_name("CMBX12"));
        assert!(is_bold_name("ABCDEF+CMBX10"));
        assert!(is_bold_name("CMB10"));
        assert!(is_bold_name("Helvetica-Bold"));
        assert!(!is_bold_name("CMR10"));
        assert!(!is_bold_name("CMBRIGHT"));
        assert!(!is_bold_name("Cambria"));
    }

    #[test]
    fn trailing_footnote_marker_keeps_heading() {
        let mut items = vec![
            item("Fact-Checking", 50.0, 40.0, 16.0, "Times"),
            item("2", 150.0, 38.0, 7.0, "Times"),
        ];
        items.extend(body_items(5, 70.0));
        let p = page(
            "Fact-Checking 2
body text line with words",
            items,
        );
        let style = DocStyle::from_pages([&p]);
        let md = page_to_markdown(&p, &style);
        assert!(
            md.starts_with(
                "# Fact-Checking 2
"
            ),
            "{md}"
        );
        // A body line that is just a number never becomes a heading.
        let mut h = HashMap::new();
        h.insert(String::new(), 1);
        assert_eq!(heading_level(&h, "12"), None);
    }

    #[test]
    fn small_caps_heading_is_recognised() {
        let mut items = vec![
            item("3.", 120.0, 89.5, 14.0, "Calibri-Light"),
            item("R", 158.0, 89.6, 14.0, "Calibri-Light"),
            item("ECOLLECTION OF", 165.0, 92.0, 11.0, "Calibri-Light"),
            item("N", 244.0, 89.6, 14.0, "Calibri-Light"),
            item("ATIONAL", 252.0, 92.0, 11.0, "Calibri-Light"),
        ];
        items.extend(body_items(8, 120.0));
        // Same font and size as the reduced caps but alone on its line.
        items.push(item("NOTE", 50.0, 300.0, 11.0, "Calibri-Light"));
        let p = page(
            "3. RECOLLECTION OF NATIONAL
body text line with words
NOTE",
            items,
        );
        let style = DocStyle::from_pages([&p]);
        let md = page_to_markdown(&p, &style);
        assert!(
            md.starts_with(
                "# 3. RECOLLECTION OF NATIONAL
"
            ),
            "{md}"
        );
        assert!(!md.contains("# NOTE"), "{md}");
    }

    #[test]
    fn captions_are_not_headings() {
        assert!(is_caption(
            "Figure 2.1: Surveyed MSMEs by size across sectors (%)"
        ));
        assert!(is_caption("Table 13.2. Effect of cations on flocculation"));
        assert!(is_caption("Fig. 4 Overview"));
        assert!(is_caption("TABLE IV Results"));
        assert!(is_caption("Table A.1: Extra"));
        assert!(!is_caption("Table of Contents"));
        assert!(!is_caption("Figures and Tables"));
        assert!(!is_caption("3. Impact on Business Operations"));
    }

    #[test]
    fn run_in_leads_are_not_headings() {
        assert!(is_run_in(
            "d. Awareness of Plastics Ordinance. About 68% of respondents know that there is a city"
        ));
        assert!(is_run_in(
            "Data collection: We surveyed three hundred firms in total"
        ));
        assert!(!is_run_in("3.1. Status of Business Operations"));
        assert!(!is_run_in(
            "Restrictions on Land Ownership by Foreigners in Selected Jurisdictions"
        ));
        assert!(!is_run_in("U.S. Policy Toward the Middle East and Beyond"));
        assert!(!is_run_in("Chapter 2. Methods"));
    }

    #[test]
    fn bold_caption_and_its_wrap_stay_body() {
        let mut items = vec![
            item(
                "Figure 2.1: Surveyed firms by size",
                50.0,
                40.0,
                10.0,
                "Arial-Bold",
            ),
            item("across all sectors (%)", 50.0, 52.0, 10.0, "Arial-Bold"),
            item("Results", 50.0, 90.0, 10.0, "Arial-Bold"),
        ];
        items.extend(body_items(8, 110.0));
        let p = page(
            "Figure 2.1: Surveyed firms by size
across all sectors (%)

Results
body text line with words",
            items,
        );
        let style = DocStyle::from_pages([&p]);
        let md = page_to_markdown(&p, &style);
        assert!(
            md.starts_with(
                "Figure 2.1: Surveyed firms by size
across all sectors (%)
"
            ),
            "{md}"
        );
        assert!(md.contains("# Results"), "{md}");
    }

    #[test]
    fn bold_sentence_tail_is_not_a_heading() {
        // A bold list of names wraps so its last entry fills a line alone.
        let mut items = body_items(8, 100.0);
        items.push(item("Sweden, and the", 50.0, 40.0, 10.0, "Times"));
        items.push(item("United Kingdom.", 50.0, 52.0, 10.0, "Times-Bold"));
        items.push(item("Results", 50.0, 70.0, 10.0, "Times-Bold"));
        let p = page(
            "Norway, Portugal, Sweden, and the\nUnited Kingdom.\n\nResults\nbody text line with words",
            items,
        );
        let style = DocStyle::from_pages([&p]);
        let md = page_to_markdown(&p, &style);
        assert!(md.contains("and the\nUnited Kingdom.\n"), "{md}");
        // A heading after a finished sentence is still recognised.
        assert!(md.contains("\n# Results\n"), "{md}");
    }

    #[test]
    fn lowercase_line_is_not_a_heading() {
        let mut items = body_items(8, 100.0);
        items.push(item(
            "and the proliferation of propaganda",
            50.0,
            40.0,
            14.0,
            "Times",
        ));
        let p = page(
            "body text line with words\nand the proliferation of propaganda\nbody text line with words",
            items,
        );
        let style = DocStyle::from_pages([&p]);
        let md = page_to_markdown(&p, &style);
        assert!(!md.contains('#'), "{md}");
    }

    #[test]
    fn wrapped_heading_may_continue_in_lowercase() {
        let mut items = vec![
            item("Determining CEC by replacing", 50.0, 20.0, 16.0, "Times"),
            item("adsorbed cations", 50.0, 40.0, 16.0, "Times"),
        ];
        items.extend(body_items(8, 80.0));
        let p = page(
            "Determining CEC by replacing\nadsorbed cations\nbody text line with words",
            items,
        );
        let style = DocStyle::from_pages([&p]);
        let md = page_to_markdown(&p, &style);
        assert!(
            md.starts_with("# Determining CEC by replacing adsorbed cations\n"),
            "{md}"
        );
    }

    #[test]
    fn mid_sentence_predicates() {
        assert!(starts_mid_sentence("and Wood Chips"));
        assert!(!starts_mid_sentence("iPhone Sales"));
        assert!(!starts_mid_sentence("b. Plastic"));
        assert!(!starts_mid_sentence("Results"));
        assert!(ends_mid_sentence("Sweden, and the"));
        assert!(ends_mid_sentence("Belgium, France,"));
        assert!(!ends_mid_sentence("of interest."));
        assert!(!ends_mid_sentence("CHAPTER 1"));
        assert!(ends_sentence("violence towards women.”"));
        assert!(ends_sentence("a Challenger).2"));
        assert!(!ends_sentence("Business Models"));
    }

    #[test]
    fn heading_styled_footnote_with_url_is_body() {
        let mut items = body_items(8, 40.0);
        items.push(item(
            "2. Content in this section is adapted from the",
            50.0,
            200.0,
            14.0,
            "Times",
        ));
        items.push(item(
            "entry (https://en.wikipedia.org/wiki/",
            50.0,
            216.0,
            14.0,
            "Times",
        ));
        let p = page(
            "body text line with words

2. Content in this section is adapted from the
entry (https://en.wikipedia.org/wiki/",
            items,
        );
        let md = page_to_markdown(&p, &DocStyle::from_pages([&p]));
        assert!(!md.contains('#'), "{md}");
        assert!(md.contains("2. Content in this section"), "{md}");
    }

    #[test]
    fn url_predicate() {
        assert!(has_url("see https://example.org/x"));
        assert!(has_url("Visit www.example.com today"));
        assert!(!has_url("Results and Discussion"));
    }
}
