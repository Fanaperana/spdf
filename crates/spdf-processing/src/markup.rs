//! Port target: `liteparse/src/processing/markupUtils.ts`.
//! TODO(phase-3): port highlight/underline/strikeout application.

use spdf_types::{MarkupData, TextItem};

/// Apply markup tags to the string content of a text item. Placeholder.
pub fn apply_markup_tags(items: &mut [TextItem]) {
    // No-op until the full port lands. Intentionally visits items to keep the
    // signature live (and to ensure the slice is not aliased by accident).
    for it in items.iter_mut() {
        if let Some(MarkupData {
            strikeout: Some(true),
            ..
        }) = it.markup
        {
            // TODO: wrap item.str in ~~...~~ and respect ordering with other marks.
        }
    }
}
