//! Port target: `liteparse/src/processing/searchItems.ts`.
//! TODO(phase-3): port the multi-item phrase search.

use spdf_types::TextItem;

/// Options for [`search_items`].
#[derive(Debug, Clone)]
pub struct SearchItemsOptions<'a> {
    pub phrase: &'a str,
    pub case_sensitive: bool,
}

/// Find text items containing the given phrase. Matches may span adjacent items.
pub fn search_items<'a>(items: &'a [TextItem], opts: SearchItemsOptions<'_>) -> Vec<&'a TextItem> {
    let needle = if opts.case_sensitive {
        opts.phrase.to_string()
    } else {
        opts.phrase.to_lowercase()
    };

    items
        .iter()
        .filter(|it| {
            let hay = if opts.case_sensitive {
                it.str.clone()
            } else {
                it.str.to_lowercase()
            };
            hay.contains(&needle)
        })
        .collect()
}
