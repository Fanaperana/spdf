//! Filled AcroForm field values as text items.
//!
//! PDFium's text page only covers the page content stream; the values a
//! user typed into interactive form fields live in widget annotations and
//! never show up there. A filled form parsed without them loses exactly the
//! information it exists to carry, so we synthesise one text item per value
//! line, positioned inside the widget's rectangle.

use spdf_types::TextItem;

/// Font size used when the widget is tall enough (typical form default).
const DEFAULT_FIELD_FONT: f64 = 11.0;
/// Rough average glyph advance as a fraction of the font size.
const AVG_ADVANCE: f64 = 0.5;
/// Horizontal padding PDF viewers apply inside a text widget.
const PADDING: f64 = 2.0;

/// A form field widget in top-left page coordinates.
#[derive(Debug, Clone)]
pub(crate) struct FieldWidget {
    pub left: f64,
    pub top: f64,
    pub width: f64,
    pub height: f64,
    pub value: String,
}

/// Turn a field's value into text items laid out inside its rectangle.
/// Returns nothing for empty values or when page text already covers the
/// widget (a flattened form, or a value also drawn into the content stream),
/// so values are never duplicated.
pub(crate) fn field_items(field: &FieldWidget, existing: &[TextItem]) -> Vec<TextItem> {
    let lines: Vec<&str> = field
        .value
        .split(['\r', '\n'])
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() || field.width <= 0.0 || field.height <= 0.0 {
        return Vec::new();
    }
    if covered_by_text(field, existing) {
        return Vec::new();
    }
    let n = lines.len() as f64;
    let font = (field.height / n * 0.75).clamp(1.0, DEFAULT_FIELD_FONT);
    let line_h = font * 1.15;
    // Single-line fields are vertically centred; multi-line ones start at
    // the top, as viewers render them.
    let first_y = if lines.len() == 1 {
        field.top + (field.height - font) / 2.0
    } else {
        field.top + PADDING
    };
    let max_w = (field.width - 2.0 * PADDING).max(1.0);
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let w = (line.chars().count() as f64 * font * AVG_ADVANCE).min(max_w);
            let mut ti = TextItem::new(
                *line,
                field.left + PADDING,
                first_y + i as f64 * line_h,
                w,
                font,
            );
            ti.font_size = Some(font);
            ti
        })
        .collect()
}

/// True when any existing text item's centre falls inside the widget.
fn covered_by_text(field: &FieldWidget, existing: &[TextItem]) -> bool {
    existing.iter().any(|it| {
        let cx = it.x + it.width / 2.0;
        let cy = it.y + it.height / 2.0;
        cx >= field.left
            && cx <= field.left + field.width
            && cy >= field.top
            && cy <= field.top + field.height
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(value: &str) -> FieldWidget {
        FieldWidget {
            left: 200.0,
            top: 80.0,
            width: 200.0,
            height: 20.0,
            value: value.into(),
        }
    }

    #[test]
    fn single_line_value_is_centred_in_widget() {
        let items = field_items(&field("Jane Q Public"), &[]);
        assert_eq!(items.len(), 1);
        let it = &items[0];
        assert_eq!(it.str, "Jane Q Public");
        assert_eq!(it.x, 202.0);
        assert_eq!(it.font_size, Some(DEFAULT_FIELD_FONT));
        let mid = it.y + it.height / 2.0;
        assert!((mid - 90.0).abs() < 1e-9);
        assert!(it.width > 0.0 && it.width <= 196.0);
    }

    #[test]
    fn empty_value_yields_nothing() {
        assert!(field_items(&field("  "), &[]).is_empty());
        assert!(field_items(&field(""), &[]).is_empty());
    }

    #[test]
    fn multiline_value_splits_into_stacked_lines() {
        let mut f = field("12 Main St\r\nFort Wayne");
        f.height = 40.0;
        let items = field_items(&f, &[]);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].str, "12 Main St");
        assert_eq!(items[1].str, "Fort Wayne");
        assert!(items[1].y > items[0].y + items[0].height * 0.9);
    }

    #[test]
    fn value_already_on_page_is_not_duplicated() {
        let existing = vec![TextItem::new("Jane Q Public", 203.0, 84.0, 70.0, 11.0)];
        assert!(field_items(&field("Jane Q Public"), &existing).is_empty());
        // Text elsewhere on the page does not suppress the value.
        let far = vec![TextItem::new("Applicant name:", 72.0, 84.0, 80.0, 11.0)];
        assert_eq!(field_items(&field("Jane Q Public"), &far).len(), 1);
    }
}
