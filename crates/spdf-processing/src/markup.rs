//! Port of `liteparse/src/processing/markupUtils.ts`.

use spdf_types::MarkupData;

/// Apply markup tags to text based on markup data. Mirrors liteparse's
/// `applyMarkupTags`: strikeout wraps with `~~`, underline/squiggly with
/// `__`, highlight with `==`. Order matters — innermost tag listed first.
pub fn apply_markup_tags(markup: &MarkupData, text: &str) -> String {
    let mut result = text.to_string();
    if markup.strikeout.unwrap_or(false) {
        result = format!("~~{result}~~");
    }
    if markup.underline.unwrap_or(false) {
        result = format!("__{result}__");
    }
    if markup.squiggly.unwrap_or(false) {
        result = format!("__{result}__");
    }
    if markup.highlight.is_some() {
        result = format!("=={result}==");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_markup_returns_original() {
        let m = MarkupData::default();
        assert_eq!(apply_markup_tags(&m, "hi"), "hi");
    }

    #[test]
    fn strikeout_wraps_with_tildes() {
        let m = MarkupData {
            strikeout: Some(true),
            ..Default::default()
        };
        assert_eq!(apply_markup_tags(&m, "x"), "~~x~~");
    }

    #[test]
    fn combined_markup_order_matches_liteparse() {
        let m = MarkupData {
            strikeout: Some(true),
            underline: Some(true),
            highlight: Some("yellow".into()),
            ..Default::default()
        };
        assert_eq!(apply_markup_tags(&m, "x"), "==__~~x~~__==");
    }
}
