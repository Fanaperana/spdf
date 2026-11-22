//! Splitting PDFium text segments at column gutters.
//!
//! PDFium groups the characters of one baseline into a single "segment"
//! even when they are separated by wide horizontal gaps. In tables whose
//! cells are drawn as one text run (common in LaTeX and word-processor
//! output) a whole row then comes back as one item, and downstream layout
//! code can no longer see where one cell ends and the next begins. This
//! module cuts such segments at word boundaries whose gap is much wider
//! than an ordinary (even justified) word space.

/// Gap between words, in units of the font size, at or above which a
/// segment is cut. Ordinary word spaces are ~0.25-0.35 em and justified
/// lines rarely stretch them past ~0.8 em.
pub(crate) const GUTTER_EM: f64 = 1.2;

/// Average advance per character (in em) above which a segment probably
/// contains a gutter. Plain prose averages ~0.5 em per character; this
/// cheap test keeps the per-glyph lookup off ordinary text.
pub(crate) const SUSPECT_ADVANCE_EM: f64 = 0.65;

/// Is a segment with `text` spanning `width` points at
/// `font_size` worth checking for gutters?
pub(crate) fn suspicious(text: &str, width: f64, font_size: f64) -> bool {
    if font_size <= 0.0 || !text.trim().contains(char::is_whitespace) {
        return false;
    }
    let n = text.trim().chars().count();
    n > 1 && width / n as f64 > SUSPECT_ADVANCE_EM * font_size
}

/// One piece of a split segment: its words and horizontal extent.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Piece {
    pub text: String,
    pub left: f64,
    pub right: f64,
}

/// Split `text` at gutters, given the horizontal extents of its visible
/// (non-whitespace) glyphs in text order. Returns `None` when the segment
/// has no gutter or the glyphs cannot be matched one-to-one with the
/// text's visible characters (ligatures, generated characters, ...).
pub(crate) fn split(text: &str, glyphs: &[(f64, f64)], font_size: f64) -> Option<Vec<Piece>> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 2 || font_size <= 0.0 {
        return None;
    }
    let visible: usize = words.iter().map(|w| w.chars().count()).sum();
    if visible != glyphs.len() {
        return None;
    }
    // Extent of each word.
    let mut spans = Vec::with_capacity(words.len());
    let mut at = 0;
    for w in &words {
        let n = w.chars().count();
        let g = &glyphs[at..at + n];
        at += n;
        let left = g.iter().map(|b| b.0).fold(f64::INFINITY, f64::min);
        let right = g.iter().map(|b| b.1).fold(f64::NEG_INFINITY, f64::max);
        spans.push((left, right));
    }
    // Words must run left to right; anything else (RTL, overprinting) is
    // left to PDFium's grouping.
    if spans.windows(2).any(|p| p[1].0 < p[0].0) {
        return None;
    }
    let mut pieces: Vec<Piece> = Vec::new();
    for (w, &(left, right)) in words.iter().zip(&spans) {
        match pieces.last_mut() {
            Some(p) if left - p.right < GUTTER_EM * font_size => {
                p.text.push(' ');
                p.text.push_str(w);
                p.right = p.right.max(right);
            }
            _ => pieces.push(Piece {
                text: (*w).to_string(),
                left,
                right,
            }),
        }
    }
    (pieces.len() > 1).then_some(pieces)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Glyph boxes for `text` laid out from `x` with 5pt advance and the
    /// given extra gap (in points) after each word.
    fn layout(text: &str, x: f64, gaps: &[f64]) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        let mut pos = x;
        for (i, w) in text.split_whitespace().enumerate() {
            for _ in w.chars() {
                out.push((pos, pos + 5.0));
                pos += 5.0;
            }
            pos += gaps.get(i).copied().unwrap_or(0.0);
        }
        out
    }

    #[test]
    fn row_is_cut_at_wide_gaps_only() {
        let text = "Qwen 72B Pretrained 73.60";
        // "Qwen"-"72B" ordinary space, then two gutters.
        let g = layout(text, 100.0, &[3.0, 30.0, 25.0]);
        let p = split(text, &g, 10.0).expect("gutters");
        let texts: Vec<&str> = p.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(texts, ["Qwen 72B", "Pretrained", "73.60"]);
        assert_eq!(p[0].left, 100.0);
        assert_eq!(p[0].right, 100.0 + 20.0 + 3.0 + 15.0);
    }

    #[test]
    fn justified_prose_is_untouched() {
        let text = "a fairly loosely justified line";
        let g = layout(text, 0.0, &[7.0, 7.0, 7.0, 7.0]);
        assert_eq!(split(text, &g, 10.0), None);
    }

    #[test]
    fn mismatched_glyphs_are_left_alone() {
        // A ligature glyph makes the counts disagree.
        let g = vec![(0.0, 5.0), (40.0, 45.0)];
        assert_eq!(split("fi x", &g, 10.0), None);
    }

    #[test]
    fn right_to_left_order_is_left_alone() {
        let g = vec![(100.0, 105.0), (0.0, 5.0)];
        assert_eq!(split("a b", &g, 10.0), None);
    }

    #[test]
    fn suspicion_needs_wide_advance_and_a_space() {
        assert!(suspicious("74.20 71.08", 60.0, 6.0));
        assert!(!suspicious("normal words", 30.0, 6.0));
        assert!(!suspicious("SingleWord", 200.0, 6.0));
    }
}
