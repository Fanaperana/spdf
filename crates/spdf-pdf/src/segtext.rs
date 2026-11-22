//! Exact text for PDFium text segments.
//!
//! PDFium builds a page's text "segments" (`FPDFText_CountRects`) by walking
//! the characters in content order and emitting one rectangle per run of
//! characters from the same text object, each rectangle being the union of
//! the run's tight glyph boxes. pdfium-render then recovers a segment's text
//! with `FPDFText_GetBoundedText(rect)`, i.e. *every* glyph that overlaps the
//! rectangle. On tightly kerned text, text with baseline jitter, or text
//! objects that cut words in half, neighbouring segments' glyphs overlap the
//! rectangle and the text comes back with duplicated or dropped letters
//! ("available fofor analysis", "Titrate thextract").
//!
//! This module re-derives each segment's text from the characters
//! themselves: characters are visited in content order and assigned to the
//! first segment (at or after the current one) whose rectangle contains
//! their box, mirroring how PDFium produced the rectangles in the first
//! place. It is pure geometry so it can be unit-tested without PDFium.

/// Axis-aligned box in PDF user space (bottom-left origin).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rect {
    pub left: f64,
    pub bottom: f64,
    pub right: f64,
    pub top: f64,
}

impl Rect {
    fn contains(&self, o: &Rect) -> bool {
        const EPS: f64 = 0.05;
        o.left >= self.left - EPS
            && o.right <= self.right + EPS
            && o.bottom >= self.bottom - EPS
            && o.top <= self.top + EPS
    }

    fn is_empty(&self) -> bool {
        self.right - self.left <= 1e-6 || self.top - self.bottom <= 1e-6
    }
}

/// One page character in content order.
#[derive(Debug, Clone)]
pub(crate) struct PageChar {
    pub text: String,
    /// Inserted by PDFium (word spaces / line breaks it inferred).
    pub generated: bool,
    /// Loose glyph box, if PDFium reports one.
    pub rect: Option<Rect>,
    /// Horizontal extent used for layout (PDFium's loose box, which
    /// includes side bearings); falls back to `rect`.
    pub extent: Option<(f64, f64)>,
}

/// Text recovered for one segment, plus the horizontal extents of its
/// visible characters (one entry per non-whitespace `char`, in text order).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct SegText {
    pub text: String,
    pub glyphs: Vec<(f64, f64)>,
}

/// A character that draws something. PDFium's end-of-line soft-hyphen
/// marker (U+0002) is not; other control characters are kept because fonts
/// without a ToUnicode map surface raw glyph codes as such.
fn is_visible(c: char) -> bool {
    !c.is_whitespace() && c != '\u{2}'
}

/// Assign `chars` to `segments` (both in PDFium order). Segments may come
/// back empty (PDFium also emits rectangles for soft-hyphen markers and
/// lone spaces). Returns `None` when too many visible characters fit no
/// rectangle, in which case the caller should keep PDFium's own text.
pub(crate) fn assign(chars: &[PageChar], segments: &[Rect]) -> Option<Vec<SegText>> {
    if segments.is_empty() {
        return None;
    }
    let mut out = vec![SegText::default(); segments.len()];
    let mut cur = 0usize;
    let (mut unplaced, mut visible_total) = (0usize, 0usize);
    for ch in chars {
        let visible = ch.text.chars().any(is_visible);
        let marker = !visible && ch.text.contains('\u{2}');
        let placed = match ch.rect {
            Some(r) if !ch.generated && !r.is_empty() => {
                (cur..segments.len()).find(|&i| segments[i].contains(&r))
            }
            _ => None,
        };
        if !visible && !marker {
            if ch.text.chars().all(|c| c == ' ' || c == '\u{a0}') {
                // Spaces stay with the rectangle that holds them (PDFium's
                // own text keeps a segment's trailing space); generated
                // spaces attach to the segment they follow.
                let push_space = |t: &mut String, lead_ok: bool| {
                    if (lead_ok || !t.is_empty()) && !t.ends_with(' ') {
                        t.push(' ');
                    }
                };
                match placed {
                    // A space glyph opening the next rectangle separates it
                    // from the current one: both keep it, as PDFium's
                    // rectangle query does.
                    Some(i) => {
                        if i != cur {
                            push_space(&mut out[cur].text, false);
                            cur = i;
                        }
                        push_space(&mut out[cur].text, true);
                    }
                    None => push_space(&mut out[cur].text, false),
                }
            }
            continue;
        }
        let (idx, r) = match (placed, ch.rect) {
            (Some(i), Some(r)) => (i, r),
            // A glyph outside every remaining rectangle: PDFium skipped it
            // (zero-size box) or the order is unusual. Keep it with the
            // current segment rather than lose it.
            (_, r) => {
                if visible {
                    unplaced += 1;
                }
                (cur, r.unwrap_or(segments[cur]))
            }
        };
        if visible {
            visible_total += 1;
        }
        cur = idx;
        let seg = &mut out[cur];
        seg.text.push_str(&ch.text);
        for c in ch.text.chars() {
            if is_visible(c) {
                seg.glyphs.push(ch.extent.unwrap_or((r.left, r.right)));
            }
        }
    }
    // Too many glyphs outside every remaining rectangle means the content
    // order does not match PDFium's rectangles (unusual producers); keep
    // PDFium's own text then.
    if unplaced * 50 > visible_total.max(1) {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(l: f64, b: f64, r: f64, t: f64) -> Rect {
        Rect {
            left: l,
            bottom: b,
            right: r,
            top: t,
        }
    }

    /// Lay `s` out from `x` on a baseline with per-glyph width `w` and a
    /// vertical offset `dy`.
    fn run(s: &str, x: f64, w: f64, dy: f64) -> Vec<PageChar> {
        s.chars()
            .enumerate()
            .map(|(i, c)| PageChar {
                text: c.to_string(),
                generated: false,
                rect: Some(rect(
                    x + i as f64 * w,
                    100.0 + dy,
                    x + (i + 1) as f64 * w,
                    110.0 + dy,
                )),
                extent: None,
            })
            .collect()
    }

    fn union(chars: &[PageChar]) -> Rect {
        let mut r = chars[0].rect.unwrap();
        for c in chars {
            let o = c.rect.unwrap();
            r.left = r.left.min(o.left);
            r.right = r.right.max(o.right);
            r.bottom = r.bottom.min(o.bottom);
            r.top = r.top.max(o.top);
        }
        r
    }

    #[test]
    fn overlapping_kerned_runs_keep_their_own_letters() {
        // "available fo" + "r analysis": the second object starts with a
        // negative kern so its first glyph overlaps the first rectangle,
        // and the jitter puts the runs on slightly different baselines.
        let a = run("available fo", 0.0, 4.0, 0.0);
        let b = run("r analysis", 46.0, 4.0, 0.3);
        let segs = vec![union(&a), union(&b)];
        let mut chars = a.clone();
        chars.extend(b.clone());
        let got = assign(&chars, &segs).unwrap();
        assert_eq!(got[0].text, "available fo");
        assert_eq!(got[1].text, "r analysis");
        assert_eq!(got[0].glyphs.len(), 11);
        assert_eq!(got[1].glyphs.len(), 9);
    }

    #[test]
    fn generated_space_inside_a_segment_is_kept() {
        let mut chars = run("ab", 0.0, 5.0, 0.0);
        chars.push(PageChar {
            text: " ".into(),
            generated: true,
            rect: None,
            extent: None,
        });
        chars.extend(run("cd", 20.0, 5.0, 0.0));
        let segs = vec![union(
            &chars
                .iter()
                .filter(|c| c.rect.is_some())
                .cloned()
                .collect::<Vec<_>>(),
        )];
        let got = assign(&chars, &segs).unwrap();
        assert_eq!(got[0].text, "ab cd");
    }

    #[test]
    fn space_between_segments_is_dropped() {
        let a = run("Hello", 0.0, 5.0, 0.0);
        let b = run("World", 40.0, 5.0, 0.0);
        let mut chars = a.clone();
        chars.push(PageChar {
            text: "\r\n".into(),
            generated: true,
            rect: None,
            extent: None,
        });
        chars.extend(b.clone());
        let got = assign(&chars, &[union(&a), union(&b)]).unwrap();
        assert_eq!(got[0].text, "Hello");
        assert_eq!(got[1].text, "World");
    }

    #[test]
    fn trailing_space_and_soft_hyphen_marker_stay_with_their_segment() {
        // "pre\u{2} " then "trained": PDFium's end-of-line marker and a real
        // space glyph sit inside the first rectangle.
        let mut a = run("pre", 0.0, 5.0, 0.0);
        a.push(PageChar {
            text: "\u{2}".into(),
            generated: false,
            rect: Some(rect(15.0, 104.0, 17.0, 104.5)),
            extent: None,
        });
        a.push(PageChar {
            text: " ".into(),
            generated: false,
            rect: Some(rect(17.0, 100.0, 19.0, 100.01)),
            extent: None,
        });
        let b = run("trained", 0.0, 5.0, -14.0);
        let mut chars = a.clone();
        chars.extend(b.clone());
        let got = assign(&chars, &[union(&a), union(&b)]).unwrap();
        assert_eq!(got[0].text, "pre\u{2} ");
        assert_eq!(got[0].glyphs.len(), 3);
        assert_eq!(got[1].text, "trained");
    }

    #[test]
    fn glyphs_outside_all_rectangles_fall_back() {
        let a = run("abc", 0.0, 5.0, 0.0);
        let segs = vec![rect(200.0, 0.0, 210.0, 10.0)];
        assert!(assign(&a, &segs).is_none());
    }

    #[test]
    fn segment_without_visible_glyphs_is_empty() {
        let a = run("abc", 0.0, 5.0, 0.0);
        let segs = vec![union(&a), rect(200.0, 0.0, 210.0, 10.0)];
        let got = assign(&a, &segs).unwrap();
        assert_eq!(got[0].text, "abc");
        assert_eq!(got[1].text, "");
    }
}
