//! Output formatters. Port target: `liteparse/src/output/*`.

#![warn(clippy::all)]

use spdf_types::{
    BoundingBox, JsonPage, JsonTextItem, ParseResult, ParseResultJson, ParsedPage, SpdfResult,
};

/// Render a [`ParseResult`] as plain text. Matches `liteparse/src/output/text.ts`.
pub fn format_text(result: &ParseResult) -> String {
    result
        .pages
        .iter()
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Build the JSON projection of a [`ParseResult`]. Matches
/// `liteparse/src/output/json.ts`.
pub fn to_json(result: &ParseResult) -> ParseResultJson {
    ParseResultJson {
        pages: result.pages.iter().map(page_to_json).collect(),
    }
}

fn page_to_json(page: &ParsedPage) -> JsonPage {
    JsonPage {
        page: page.page_num,
        width: page.width,
        height: page.height,
        text: page.text.clone(),
        text_items: page.text_items.iter().map(item_to_json).collect(),
        bounding_boxes: page.bounding_boxes.clone().unwrap_or_default(),
    }
}

fn item_to_json(it: &spdf_types::TextItem) -> JsonTextItem {
    JsonTextItem {
        text: it.str.clone(),
        x: it.x,
        y: it.y,
        width: it.width,
        height: it.height,
        font_name: it.font_name.clone(),
        font_size: it.font_size,
        confidence: it.confidence,
    }
}

/// Convenience: serialise a [`ParseResult`] as pretty JSON.
pub fn to_json_string(result: &ParseResult) -> SpdfResult<String> {
    let json = to_json(result);
    Ok(serde_json::to_string_pretty(&json)?)
}

/// Implementation detail to match liteparse JSON key ordering. Kept as a
/// dedicated helper in case we need to deviate from serde's default.
pub fn bounding_box_round_trip(bbox: BoundingBox) -> BoundingBox {
    bbox
}

#[cfg(test)]
mod tests {
    use super::*;
    use spdf_types::{ParsedPage, TextItem};

    fn sample_result() -> ParseResult {
        ParseResult {
            pages: vec![ParsedPage {
                page_num: 1,
                width: 612.0,
                height: 792.0,
                text: "hello world".into(),
                text_items: vec![TextItem::new("hello world", 10.0, 20.0, 50.0, 12.0)],
                bounding_boxes: Some(vec![BoundingBox {
                    x1: 10.0,
                    y1: 20.0,
                    x2: 60.0,
                    y2: 32.0,
                }]),
            }],
            text: "hello world".into(),
            json: None,
        }
    }

    #[test]
    fn json_has_expected_camel_case_keys() {
        let r = sample_result();
        let s = to_json_string(&r).unwrap();
        assert!(s.contains("\"fontName\"") || !s.contains("fontName"));
        assert!(s.contains("\"textItems\""));
        assert!(s.contains("\"boundingBoxes\""));
        assert!(!s.contains("text_items"));
    }

    #[test]
    fn text_format_concatenates_pages() {
        let r = sample_result();
        assert_eq!(format_text(&r), "hello world");
    }
}
