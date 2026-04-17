//! Port target: `liteparse/src/processing/cleanText.ts`.

/// Collapse runs of whitespace while preserving newlines. Matches the minimal
/// contract liteparse's `cleanRawText` exposes for downstream consumers:
/// horizontal whitespace collapses to a single space, vertical whitespace is
/// preserved so the row structure produced by grid projection survives.
pub fn clean_raw_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last: char = '\n'; // avoid leading whitespace
    for ch in text.chars() {
        if ch == '\n' {
            // Collapse a trailing space before a newline and skip redundant
            // blank lines.
            while out.ends_with(' ') {
                out.pop();
            }
            if !out.is_empty() && last != '\n' {
                out.push('\n');
            }
            last = '\n';
        } else if ch.is_whitespace() {
            if last != ' ' && last != '\n' {
                out.push(' ');
                last = ' ';
            }
        } else {
            out.push(ch);
            last = ch;
        }
    }
    while out.ends_with(' ') || out.ends_with('\n') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_horizontal_whitespace() {
        assert_eq!(clean_raw_text("  hi  \t there "), "hi there");
    }

    #[test]
    fn preserves_newlines() {
        assert_eq!(clean_raw_text("hi\nthere"), "hi\nthere");
    }

    #[test]
    fn collapses_blank_lines() {
        assert_eq!(clean_raw_text("a\n\n\nb"), "a\nb");
    }

    #[test]
    fn empty_stays_empty() {
        assert_eq!(clean_raw_text(""), "");
    }
}
