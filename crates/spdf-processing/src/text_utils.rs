//! Port of `liteparse/src/processing/textUtils.ts`.

/// Clean common OCR artifacts from table documents. OCR often misreads
/// vertical table borders as bracket/pipe characters at the start or end of
/// cell content; when the remaining core looks numeric we strip them.
pub fn clean_ocr_table_artifacts(text: &str) -> String {
    let cleaned = text.trim().to_string();
    if cleaned.is_empty() {
        return cleaned;
    }
    let border = |c: char| matches!(c, '|' | '[' | ']' | '(' | ')' | '{' | '}');
    let without = cleaned
        .trim_start_matches(border)
        .trim_end_matches(border)
        .to_string();
    if !without.is_empty() && looks_numeric(without.trim()) {
        return without.trim().to_string();
    }
    cleaned
}

fn looks_numeric(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if matches!(s, "Z/A" | "N/A" | "Z" | "-") {
        return true;
    }
    let chars: Vec<char> = s.chars().collect();
    // Pattern A: ^[*+-]?[\d,.\s]+[%]?$
    {
        let mut i = 0;
        if matches!(chars.get(i), Some('*' | '+' | '-')) {
            i += 1;
        }
        let body_start = i;
        while i < chars.len()
            && (chars[i].is_ascii_digit() || matches!(chars[i], ',' | '.' | ' ' | '\t'))
        {
            i += 1;
        }
        let ok_body = i > body_start;
        let mut j = i;
        if matches!(chars.get(j), Some('%')) {
            j += 1;
        }
        if ok_body && j == chars.len() {
            return true;
        }
    }
    // Pattern B: ^[*]?-?[\d,.\s]+$
    let mut i = 0;
    if matches!(chars.get(i), Some('*')) {
        i += 1;
    }
    if matches!(chars.get(i), Some('-')) {
        i += 1;
    }
    let body_start = i;
    while i < chars.len()
        && (chars[i].is_ascii_digit() || matches!(chars[i], ',' | '.' | ' ' | '\t'))
    {
        i += 1;
    }
    i > body_start && i == chars.len()
}

/// Convert an ASCII string to Unicode subscript characters, or return the
/// input unchanged if any character has no subscript mapping.
pub fn str_to_subscript_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for ch in s.chars() {
        match subscript_for(ch) {
            Some(sub) => out.push(sub),
            None => return s.to_owned(),
        }
    }
    out
}

/// Convert an ASCII string to Unicode superscript characters, or return the
/// input unchanged if any character has no superscript mapping.
pub fn str_to_post_script(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for ch in s.chars() {
        match superscript_for(ch) {
            Some(sup) => out.push(sup),
            None => return s.to_owned(),
        }
    }
    out
}

fn subscript_for(c: char) -> Option<char> {
    Some(match c {
        '0' => '₀',
        '1' => '₁',
        '2' => '₂',
        '3' => '₃',
        '4' => '₄',
        '5' => '₅',
        '6' => '₆',
        '7' => '₇',
        '8' => '₈',
        '9' => '₉',
        '+' => '₊',
        '-' => '₋',
        'a' => 'ₐ',
        'e' => 'ₑ',
        'o' => 'ₒ',
        'x' => 'ₓ',
        'ə' => 'ₔ',
        'h' => 'ₕ',
        'k' => 'ₖ',
        'l' => 'ₗ',
        'm' => 'ₘ',
        'n' => 'ₙ',
        'p' => 'ₚ',
        'r' => 'ᵣ',
        's' => 'ₛ',
        't' => 'ₜ',
        _ => return None,
    })
}

fn superscript_for(c: char) -> Option<char> {
    Some(match c {
        '0' => '⁰',
        '1' => '¹',
        '2' => '²',
        '3' => '³',
        '4' => '⁴',
        '5' => '⁵',
        '6' => '⁶',
        '7' => '⁷',
        '8' => '⁸',
        '9' => '⁹',
        '+' => '⁺',
        '-' => '⁻',
        'a' => 'ᵃ',
        'b' => 'ᵇ',
        'c' => 'ᶜ',
        'd' => 'ᵈ',
        'e' => 'ᵉ',
        'f' => 'ᶠ',
        'g' => 'ᵍ',
        'h' => 'ʰ',
        'i' => 'ⁱ',
        'j' => 'ʲ',
        'k' => 'ᵏ',
        'l' => 'ˡ',
        'm' => 'ᵐ',
        'n' => 'ⁿ',
        'o' => 'ᵒ',
        'p' => 'ᵖ',
        'r' => 'ʳ',
        's' => 'ˢ',
        't' => 'ᵗ',
        'u' => 'ᵘ',
        'v' => 'ᵛ',
        'w' => 'ʷ',
        'x' => 'ˣ',
        'y' => 'ʸ',
        'z' => 'ᶻ',
        'A' => 'ᴬ',
        'B' => 'ᴮ',
        'D' => 'ᴰ',
        'E' => 'ᴱ',
        'G' => 'ᴳ',
        'H' => 'ᴴ',
        'I' => 'ᴵ',
        'J' => 'ᴶ',
        'K' => 'ᴷ',
        'L' => 'ᴸ',
        'M' => 'ᴹ',
        'N' => 'ᴺ',
        'O' => 'ᴼ',
        'P' => 'ᴾ',
        'R' => 'ᴿ',
        'T' => 'ᵀ',
        'U' => 'ᵁ',
        'V' => 'ⱽ',
        'W' => 'ᵂ',
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_trailing_bracket_on_number() {
        assert_eq!(clean_ocr_table_artifacts("44520]"), "44520");
    }

    #[test]
    fn strips_leading_pipe_on_number() {
        assert_eq!(clean_ocr_table_artifacts("|123"), "123");
    }

    #[test]
    fn strips_trailing_pipe_on_decimal() {
        assert_eq!(clean_ocr_table_artifacts("0.3|"), "0.3");
    }

    #[test]
    fn keeps_parenthesised_word() {
        assert_eq!(clean_ocr_table_artifacts("(note)"), "(note)");
    }

    #[test]
    fn keeps_na_placeholder() {
        assert_eq!(clean_ocr_table_artifacts("[N/A]"), "N/A");
    }

    #[test]
    fn subscript_maps_digits() {
        assert_eq!(str_to_subscript_string("123"), "₁₂₃");
    }

    #[test]
    fn subscript_passthrough_on_unmapped() {
        assert_eq!(str_to_subscript_string("H2O"), "H2O");
    }

    #[test]
    fn superscript_maps_letters() {
        assert_eq!(str_to_post_script("th"), "ᵗʰ");
    }

    #[test]
    fn superscript_passthrough_on_unmapped() {
        assert_eq!(str_to_post_script("Qx"), "Qx");
    }
}
