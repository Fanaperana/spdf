//! Greedy CTC decoding for the PP-OCR recognizer, plus splitting a decoded
//! line into words with their horizontal extent along the line.

/// Character table: class 0 is the CTC blank, classes `1..=n` are the
/// dictionary lines, class `n + 1` is the space.
#[derive(Debug, Clone)]
pub struct Charset {
    chars: Vec<String>,
}

impl Charset {
    pub fn from_dict(dict: &str) -> Self {
        let mut chars: Vec<String> = dict
            .lines()
            .map(|l| l.trim_end_matches('\r').to_string())
            .collect();
        chars.push(" ".to_string());
        Self { chars }
    }

    /// Number of model output classes this charset expects (incl. blank).
    pub fn num_classes(&self) -> usize {
        self.chars.len() + 1
    }

    fn get(&self, class: usize) -> Option<&str> {
        class
            .checked_sub(1)
            .and_then(|i| self.chars.get(i))
            .map(|s| s.as_str())
    }
}

/// One emitted character and the timestep it was emitted at.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedChar {
    pub text: String,
    pub step: usize,
    pub prob: f32,
}

/// Greedy CTC: collapse repeats, drop blanks.
pub fn greedy_decode(classes: &[usize], probs: &[f32], cs: &Charset) -> Vec<DecodedChar> {
    let mut out = Vec::new();
    let mut prev = usize::MAX;
    for (t, (&c, &p)) in classes.iter().zip(probs).enumerate() {
        if c != prev && c != 0 {
            if let Some(s) = cs.get(c) {
                out.push(DecodedChar {
                    text: s.to_string(),
                    step: t,
                    prob: p,
                });
            }
        }
        prev = c;
    }
    out
}

/// A word on a recognised line: text, mean confidence, and its extent along
/// the line as fractions `[start, end]` of the crop width.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub confidence: f64,
    pub start: f64,
    pub end: f64,
}

/// Split decoded characters into words at spaces. `steps` is the number of
/// timesteps the model produced for the crop; `content_steps` is the number
/// of steps covering real (unpadded) image content.
pub fn split_words(chars: &[DecodedChar], content_steps: usize) -> Vec<Word> {
    let content = content_steps.max(1) as f64;
    // Typical half-width of a character in steps: half the median gap
    // between consecutive non-space characters within a word.
    let mut gaps: Vec<f64> = chars
        .windows(2)
        .filter(|w| w[0].text != " " && w[1].text != " ")
        .map(|w| (w[1].step - w[0].step) as f64)
        .collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let half = gaps
        .get(gaps.len() / 2)
        .copied()
        .unwrap_or(2.0)
        .clamp(1.0, 6.0)
        / 2.0;

    let mut words = Vec::new();
    let mut cur: Vec<&DecodedChar> = Vec::new();
    let flush = |cur: &mut Vec<&DecodedChar>, words: &mut Vec<Word>| {
        if cur.is_empty() {
            return;
        }
        let text: String = cur.iter().map(|c| c.text.as_str()).collect();
        let conf = cur.iter().map(|c| c.prob as f64).sum::<f64>() / cur.len() as f64;
        let first = cur[0].step as f64 + 0.5;
        let last = cur[cur.len() - 1].step as f64 + 0.5;
        let start = ((first - half) / content).clamp(0.0, 1.0);
        let end = ((last + half) / content).clamp(0.0, 1.0);
        words.push(Word {
            text,
            confidence: conf,
            start,
            end: end.max(start),
        });
        cur.clear();
    };
    for c in chars {
        if c.text.trim().is_empty() {
            flush(&mut cur, &mut words);
        } else {
            cur.push(c);
        }
    }
    flush(&mut cur, &mut words);
    words
}

/// Merge consecutive words into space-joined runs, breaking wherever the
/// gap between two words (as a fraction of the line width) exceeds
/// `max_gap`. Runs keep the recogniser's own word spacing, so downstream
/// layout code never has to re-infer spaces from geometry, while wide gaps
/// (table cells, columns sharing one detected line) stay separate items.
pub fn group_runs(words: &[Word], max_gap: f64) -> Vec<Word> {
    let mut runs: Vec<Word> = Vec::new();
    let mut n_in_run = 0usize;
    for w in words {
        match runs.last_mut() {
            Some(r) if w.start - r.end <= max_gap => {
                r.text.push(' ');
                r.text.push_str(&w.text);
                let n = n_in_run as f64;
                r.confidence = (r.confidence * n + w.confidence) / (n + 1.0);
                r.end = r.end.max(w.end);
                n_in_run += 1;
            }
            _ => {
                runs.push(w.clone());
                n_in_run = 1;
            }
        }
    }
    runs
}

/// Mean probability of the emitted characters (0 when nothing decoded).
pub fn line_confidence(chars: &[DecodedChar]) -> f64 {
    if chars.is_empty() {
        return 0.0;
    }
    chars.iter().map(|c| c.prob as f64).sum::<f64>() / chars.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cs() -> Charset {
        Charset::from_dict("a\nb\nc\r\n")
    }

    #[test]
    fn charset_layout() {
        let c = cs();
        assert_eq!(c.num_classes(), 5); // blank, a, b, c, space
        assert_eq!(c.get(0), None);
        assert_eq!(c.get(1), Some("a"));
        assert_eq!(c.get(3), Some("c"));
        assert_eq!(c.get(4), Some(" "));
        assert_eq!(c.get(5), None);
    }

    #[test]
    fn greedy_collapses_repeats_and_blanks() {
        // a a _ a b b _ space c
        let classes = [1, 1, 0, 1, 2, 2, 0, 4, 3];
        let probs = [0.9f32; 9];
        let d = greedy_decode(&classes, &probs, &cs());
        let s: String = d.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(s, "aab c");
        assert_eq!(d[0].step, 0);
        assert_eq!(d[1].step, 3);
        assert_eq!(d[4].step, 8);
    }

    #[test]
    fn words_split_at_spaces_with_ordered_extents() {
        let classes = [1, 0, 2, 0, 4, 0, 3, 0, 1, 0];
        let probs = [0.8f32; 10];
        let d = greedy_decode(&classes, &probs, &cs());
        let w = split_words(&d, 10);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].text, "ab");
        assert_eq!(w[1].text, "ca");
        assert!(w[0].start < w[0].end && w[0].end <= w[1].start + 1e-9);
        assert!(w[1].end <= 1.0);
        assert!((w[0].confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn runs_join_close_words_and_split_wide_gaps() {
        let w = |t: &str, s: f64, e: f64| Word {
            text: t.into(),
            confidence: 0.9,
            start: s,
            end: e,
        };
        let words = [
            w("North", 0.0, 0.1),
            w("5,405", 0.4, 0.5),
            w("a", 0.52, 0.55),
            w("b", 0.57, 0.6),
        ];
        let runs = group_runs(&words, 0.05);
        let texts: Vec<&str> = runs.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts, ["North", "5,405 a b"]);
        assert_eq!((runs[1].start, runs[1].end), (0.4, 0.6));
        assert!(group_runs(&[], 0.1).is_empty());
    }

    #[test]
    fn empty_decode_has_zero_confidence() {
        assert_eq!(line_confidence(&[]), 0.0);
        assert!(split_words(&[], 10).is_empty());
    }
}
