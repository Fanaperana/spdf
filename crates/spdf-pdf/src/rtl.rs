//! Glyph-level line building for pages containing right-to-left scripts.
//!
//! PDFium's text "segments" are rectangle-based: a segment's text is whatever
//! lies inside its bounding box. For shaped Arabic (and kerned Hebrew), glyph
//! boxes overlap their neighbours, so each segment re-reads letters that
//! belong to the next one and the output fills with duplicated fragments.
//! RTL text is also very often stored in *visual* order (left-to-right glyph
//! placement, as emitted by shaping engines), so even correctly-split runs
//! come out reversed.
//!
//! This module rebuilds lines straight from individual glyphs instead:
//! glyphs are grouped by baseline, ordered by x, word gaps become spaces,
//! and lines whose strong characters are mostly RTL are converted to logical
//! order (with embedded LTR runs such as numbers or Latin words kept intact).
//! It is pure geometry so it can be unit-tested without PDFium.

/// One glyph (or ligature cluster) in top-left page coordinates.
#[derive(Debug, Clone)]
pub(crate) struct Glyph {
    pub text: String,
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
    /// Baseline y (top-left origin).
    pub baseline: f64,
    pub font_size: f64,
    pub font_name: Option<String>,
}

/// A rebuilt text run ready to become a `TextItem`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Run {
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub font_size: f64,
    pub font_name: Option<String>,
}

/// Strong right-to-left character (Hebrew, Arabic, Syriac, Thaana, NKo,
/// Samaritan, Mandaic and their presentation forms).
pub(crate) fn is_strong_rtl(c: char) -> bool {
    matches!(c as u32,
        0x0590..=0x08FF | 0xFB1D..=0xFDFF | 0xFE70..=0xFEFF | 0x10800..=0x10FFF | 0x1E800..=0x1EFFF)
        && !is_combining_or_digit_rtl(c)
}

/// Marks and Arabic-Indic digits inside the RTL blocks are not strong RTL.
fn is_combining_or_digit_rtl(c: char) -> bool {
    matches!(c as u32,
        0x0591..=0x05BD | 0x05BF | 0x05C1..=0x05C2 | 0x05C4..=0x05C5 | 0x05C7
        | 0x0610..=0x061A | 0x064B..=0x065F | 0x0660..=0x0669 | 0x0670
        | 0x06D6..=0x06DC | 0x06DF..=0x06E4 | 0x06E7..=0x06E8 | 0x06EA..=0x06ED
        | 0x06F0..=0x06F9 | 0xFE00..=0xFE0F)
}

fn is_strong_ltr(c: char) -> bool {
    c.is_alphabetic() && !is_strong_rtl(c) && !is_combining_or_digit_rtl(c)
}

/// Characters that belong to an embedded left-to-right run (Latin words,
/// numbers) inside an RTL line.
fn is_ltr_run_char(c: char) -> bool {
    is_strong_ltr(c) || c.is_ascii_digit() || matches!(c as u32, 0x0660..=0x0669 | 0x06F0..=0x06F9)
}

/// Does `text` contain any strong RTL character?
pub(crate) fn has_rtl(text: &str) -> bool {
    text.chars().any(is_strong_rtl)
}

fn mirror(c: char) -> char {
    match c {
        '(' => ')',
        ')' => '(',
        '[' => ']',
        ']' => '[',
        '{' => '}',
        '}' => '{',
        '<' => '>',
        '>' => '<',
        '«' => '»',
        '»' => '«',
        _ => c,
    }
}

/// Merge consecutive glyphs that share (nearly) the same box: PDFium reports
/// each code point of a multi-character ToUnicode mapping (ligatures such as
/// Arabic lam-alef) as a separate char with identical bounds. The cluster
/// text is already in logical order and must not be reversed internally.
pub(crate) fn merge_clusters(glyphs: Vec<Glyph>) -> Vec<Glyph> {
    let mut out: Vec<Glyph> = Vec::with_capacity(glyphs.len());
    for g in glyphs {
        if let Some(prev) = out.last_mut() {
            let same = (prev.left - g.left).abs() < 0.01
                && (prev.right - g.right).abs() < 0.01
                && (prev.baseline - g.baseline).abs() < 0.01;
            if same {
                prev.text.push_str(&g.text);
                prev.top = prev.top.min(g.top);
                prev.bottom = prev.bottom.max(g.bottom);
                continue;
            }
        }
        out.push(g);
    }
    for g in &mut out {
        fix_lam_alef(&mut g.text);
    }
    out
}

/// A single glyph covering both an alef and a lam can only be the lam-alef
/// ligature (alef never joins to a following letter, so "alef, lam" is
/// always drawn as two glyphs). Some producers emit the ligature's ToUnicode
/// in reversed order; restore the logical "lam, alef".
fn fix_lam_alef(text: &mut String) {
    let mut it = text.chars();
    if let (Some(a), Some('\u{0644}'), None) = (it.next(), it.next(), it.next())
        && matches!(a, '\u{0627}' | '\u{0622}' | '\u{0623}' | '\u{0625}')
    {
        *text = format!("\u{0644}{a}");
    }
}

/// Group glyphs into lines and runs. Input glyphs must exclude whitespace;
/// spaces are re-derived from horizontal gaps.
pub(crate) fn build_runs(glyphs: Vec<Glyph>) -> Vec<Run> {
    let glyphs = merge_clusters(glyphs);
    // Group by baseline, in first-seen order.
    let mut lines: Vec<Vec<Glyph>> = Vec::new();
    for g in glyphs {
        let tol = (g.font_size * 0.3).max(1.0);
        let slot = lines.iter_mut().rev().take(8).find(|l| {
            let base = l[0].baseline;
            let size = l[0].font_size;
            (base - g.baseline).abs() <= tol
                && (size - g.font_size).abs() <= size.max(g.font_size) * 0.5
        });
        match slot {
            Some(l) => l.push(g),
            None => lines.push(vec![g]),
        }
    }

    let mut runs = Vec::new();
    for mut line in lines {
        line.sort_by(|a, b| {
            let ca = (a.left + a.right) / 2.0;
            let cb = (b.left + b.right) / 2.0;
            ca.total_cmp(&cb)
        });
        // Split into chunks at column-sized gaps.
        let mut chunk: Vec<Glyph> = Vec::new();
        for g in line {
            if let Some(prev) = chunk.last() {
                let size = prev.font_size.max(g.font_size);
                if g.left - prev.right > size * 2.5 {
                    runs.push(chunk_to_run(std::mem::take(&mut chunk)));
                }
            }
            chunk.push(g);
        }
        if !chunk.is_empty() {
            runs.push(chunk_to_run(chunk));
        }
    }
    runs
}

enum Tok {
    Glyph(String),
    Space,
}

fn chunk_to_run(chunk: Vec<Glyph>) -> Run {
    let mut toks: Vec<Tok> = Vec::with_capacity(chunk.len() * 2);
    let mut prev_right: Option<f64> = None;
    let mut prev_size: f64 = 0.0;
    for g in &chunk {
        if let Some(pr) = prev_right {
            let size = prev_size.max(g.font_size);
            if g.left - pr > size * 0.15 {
                toks.push(Tok::Space);
            }
        }
        toks.push(Tok::Glyph(g.text.clone()));
        prev_right = Some(prev_right.map_or(g.right, |p: f64| p.max(g.right)));
        prev_size = g.font_size;
    }

    let (mut rtl, mut ltr) = (0usize, 0usize);
    for g in &chunk {
        for c in g.text.chars() {
            if is_strong_rtl(c) {
                rtl += 1;
            } else if is_strong_ltr(c) {
                ltr += 1;
            }
        }
    }
    let text = if rtl > ltr {
        visual_to_logical_rtl(toks)
    } else {
        toks.iter()
            .map(|t| match t {
                Tok::Glyph(s) => s.as_str(),
                Tok::Space => " ",
            })
            .collect()
    };

    let x = chunk.iter().map(|g| g.left).fold(f64::INFINITY, f64::min);
    let r = chunk
        .iter()
        .map(|g| g.right)
        .fold(f64::NEG_INFINITY, f64::max);
    let y = chunk.iter().map(|g| g.top).fold(f64::INFINITY, f64::min);
    let b = chunk
        .iter()
        .map(|g| g.bottom)
        .fold(f64::NEG_INFINITY, f64::max);
    let font_size = chunk[0].font_size;
    let font_name = chunk.iter().find_map(|g| g.font_name.clone());
    Run {
        text,
        x,
        y,
        width: (r - x).max(0.0),
        height: (b - y).max(0.0),
        font_size,
        font_name,
    }
}

/// Convert visually ordered tokens of an RTL line into logical order:
/// reverse token order, keep embedded LTR runs (digits, Latin) in their
/// original left-to-right order, and mirror paired punctuation.
fn visual_to_logical_rtl(toks: Vec<Tok>) -> String {
    let is_ltr_tok =
        |t: &Tok| matches!(t, Tok::Glyph(s) if s.chars().next().is_some_and(is_ltr_run_char));
    // Spaces only stay inside a run between Latin letters (bidi rule N1:
    // a space between a number and a word resolves to the RTL direction).
    let is_letter_tok =
        |t: &Tok| matches!(t, Tok::Glyph(s) if s.chars().next().is_some_and(is_strong_ltr));
    // Neutral glyphs that may sit inside an LTR run ("3.5", "1,000", "a-b").
    let is_joiner_tok =
        |t: &Tok| matches!(t, Tok::Glyph(s) if s.chars().all(|c| ".,:/-_'%".contains(c)));

    let mut out_toks: Vec<String> = Vec::with_capacity(toks.len());
    let n = toks.len();
    let mut i = n;
    while i > 0 {
        let t = &toks[i - 1];
        if is_ltr_tok(t) {
            // Extend the LTR run leftwards (in visual order) over LTR glyphs,
            // interior joiners and spaces that are followed by more LTR.
            let end = i; // exclusive
            let mut start = i - 1;
            while start > 0 {
                let p = &toks[start - 1];
                let joins = start >= 2
                    && ((is_joiner_tok(p) && is_ltr_tok(&toks[start - 2]))
                        || (matches!(p, Tok::Space)
                            && is_letter_tok(&toks[start - 2])
                            && is_letter_tok(&toks[start])));
                if is_ltr_tok(p) || joins {
                    start -= 1;
                } else {
                    break;
                }
            }
            for t in &toks[start..end] {
                out_toks.push(tok_str(t, false));
            }
            i = start;
        } else {
            out_toks.push(tok_str(t, true));
            i -= 1;
        }
    }
    out_toks.concat()
}

fn tok_str(t: &Tok, mirrored: bool) -> String {
    match t {
        Tok::Space => " ".to_string(),
        Tok::Glyph(s) if mirrored => s.chars().map(mirror).collect(),
        Tok::Glyph(s) => s.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lay out `glyphs` left-to-right; `' '` in the input becomes a word gap.
    fn line(visual: &str, baseline: f64) -> Vec<Glyph> {
        let mut x = 100.0;
        let mut out = Vec::new();
        for c in visual.chars() {
            if c == ' ' {
                x += 4.0;
                continue;
            }
            out.push(Glyph {
                text: c.to_string(),
                left: x,
                right: x + 7.0,
                top: baseline - 10.0,
                bottom: baseline + 2.0,
                baseline,
                font_size: 14.0,
                font_name: Some("Arial".into()),
            });
            x += 7.0;
        }
        out
    }

    #[test]
    fn hebrew_visual_order_becomes_logical() {
        // "שלום עולם" stored visually (left to right on the page).
        let visual: String = "שלום עולם".chars().rev().collect();
        let runs = build_runs(line(&visual, 100.0));
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].text, "שלום עולם");
    }

    #[test]
    fn glyphs_out_of_content_order_are_sorted_by_x() {
        let visual: String = "זהו מבחן".chars().rev().collect();
        let mut g = line(&visual, 50.0);
        g.reverse();
        let runs = build_runs(g);
        assert_eq!(runs[0].text, "זהו מבחן");
    }

    #[test]
    fn embedded_numbers_and_latin_keep_ltr_order() {
        // Logical: "מחיר 12.50 PDF" ; visual: "PDF 12.50 ריחמ"
        let runs = build_runs(line("PDF 12.50 ריחמ", 10.0));
        assert_eq!(runs[0].text, "מחיר 12.50 PDF");
        // Multi-word Latin phrases stay together.
        let visual = "New York םלוע םולש";
        let runs = build_runs(line(visual, 10.0));
        assert_eq!(runs[0].text, "שלום עולם New York");
    }

    #[test]
    fn brackets_are_mirrored_in_rtl() {
        // Logical "(שלום)" displays visually as "(םולש)".
        let runs = build_runs(line("(םולש)", 10.0));
        assert_eq!(runs[0].text, "(שלום)");
    }

    #[test]
    fn ltr_lines_are_untouched() {
        let runs = build_runs(line("Hello world", 10.0));
        assert_eq!(runs[0].text, "Hello world");
    }

    #[test]
    fn ligature_clusters_stay_in_logical_order() {
        // A lam-alef ligature reported as two chars with identical boxes.
        let mut g = line("ب", 10.0);
        let mut l = g[0].clone();
        l.left = 93.0;
        l.right = 100.0;
        l.text = "ل".into();
        let mut a = l.clone();
        a.text = "ا".into();
        g.insert(0, a);
        g.insert(0, l);
        // Content order: ل, ا (same box at x=93), then ب at x=100.
        let runs = build_runs(g);
        assert_eq!(runs[0].text, "بلا");
    }

    #[test]
    fn reversed_lam_alef_ligature_is_restored() {
        let mut g = line("ب", 10.0);
        let mut a = g[0].clone();
        a.left = 93.0;
        a.right = 100.0;
        a.text = "ا".into();
        let mut l = a.clone();
        l.text = "ل".into();
        g.insert(0, l);
        g.insert(0, a);
        assert_eq!(build_runs(g)[0].text, "بلا");
    }

    #[test]
    fn separate_lines_and_columns_split() {
        let mut g = line("אב", 10.0);
        g.extend(line("גד", 40.0));
        let mut far = line("הו", 10.0);
        for x in &mut far {
            x.left += 200.0;
            x.right += 200.0;
        }
        g.extend(far);
        let runs = build_runs(g);
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].text, "בא");
        assert_eq!(runs[1].text, "וה");
        assert_eq!(runs[2].text, "דג");
    }
}
