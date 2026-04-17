//! Port of `liteparse/src/processing/searchItems.ts`.

use spdf_types::JsonTextItem;

/// Options for [`search_items`].
#[derive(Debug, Clone)]
pub struct SearchItemsOptions<'a> {
    pub phrase: &'a str,
    pub case_sensitive: bool,
}

/// Search text items for matches, returning synthetic merged items for each
/// match. For phrase searches, consecutive text items are concatenated and
/// searched — when a phrase spans multiple items, the result is a single
/// merged item with combined bounding box and the matched text. Font metadata
/// is taken from the first matched item.
pub fn search_items(items: &[JsonTextItem], opts: SearchItemsOptions<'_>) -> Vec<JsonTextItem> {
    let normalize = |s: &str| {
        if opts.case_sensitive {
            s.to_string()
        } else {
            s.to_lowercase()
        }
    };
    let q = normalize(opts.phrase);
    if q.is_empty() || items.is_empty() {
        return Vec::new();
    }

    // Precompute the separator between item i-1 and item i.
    let mut seps: Vec<&'static str> = vec![""; items.len()];
    for i in 1..items.len() {
        let prev = &items[i - 1];
        let cur = &items[i];
        let font_size = prev.font_size.or(cur.font_size).unwrap_or(12.0);
        let same_line = (cur.y - prev.y).abs() < font_size * 0.5;
        let gap = cur.x - (prev.x + prev.width);
        seps[i] = if same_line && gap <= font_size * 0.3 {
            ""
        } else {
            " "
        };
    }

    let mut results: Vec<JsonTextItem> = Vec::new();
    let mut start = 0usize;
    while start < items.len() {
        let mut combined = String::new();
        let mut matched_end: Option<usize> = None;
        for end in start..items.len() {
            if end > start {
                combined.push_str(seps[end]);
            }
            combined.push_str(&items[end].text);
            if normalize(&combined).contains(&q) {
                matched_end = Some(end);
                break;
            }
            if combined.chars().count() > q.chars().count() * 2 {
                break;
            }
        }
        let Some(end) = matched_end else {
            start += 1;
            continue;
        };

        // Narrow from the left: drop leading items that aren't part of the match.
        let mut s = start;
        let mut narrowed = combined.clone();
        while s < end {
            let drop_len = items[s].text.len() + seps[s + 1].len();
            if narrowed.len() < drop_len {
                break;
            }
            let candidate = narrowed[drop_len..].to_string();
            if normalize(&candidate).contains(&q) {
                narrowed = candidate;
                s += 1;
            } else {
                break;
            }
        }

        let matched = &items[s..=end];
        let x = matched
            .iter()
            .map(|m| m.x)
            .fold(f64::INFINITY, f64::min);
        let y = matched
            .iter()
            .map(|m| m.y)
            .fold(f64::INFINITY, f64::min);
        let x2 = matched
            .iter()
            .map(|m| m.x + m.width)
            .fold(f64::NEG_INFINITY, f64::max);
        let y2 = matched
            .iter()
            .map(|m| m.y + m.height)
            .fold(f64::NEG_INFINITY, f64::max);

        results.push(JsonTextItem {
            text: opts.phrase.to_string(),
            x,
            y,
            width: x2 - x,
            height: y2 - y,
            font_name: matched[0].font_name.clone(),
            font_size: matched[0].font_size,
            confidence: matched[0].confidence,
        });

        start = end + 1;
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(text: &str, x: f64, y: f64, w: f64) -> JsonTextItem {
        JsonTextItem {
            text: text.into(),
            x,
            y,
            width: w,
            height: 12.0,
            font_name: Some("F".into()),
            font_size: Some(12.0),
            confidence: None,
        }
    }

    #[test]
    fn finds_phrase_within_single_item() {
        let items = vec![mk("hello world", 0.0, 0.0, 80.0)];
        let hits = search_items(
            &items,
            SearchItemsOptions {
                phrase: "world",
                case_sensitive: false,
            },
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "world");
    }

    #[test]
    fn finds_phrase_across_items_same_line() {
        let items = vec![mk("hello", 0.0, 0.0, 30.0), mk("world", 40.0, 0.0, 30.0)];
        let hits = search_items(
            &items,
            SearchItemsOptions {
                phrase: "hello world",
                case_sensitive: false,
            },
        );
        assert_eq!(hits.len(), 1);
        assert!((hits[0].width - 70.0).abs() < 1e-6);
    }

    #[test]
    fn case_insensitive_by_default() {
        let items = vec![mk("Hello", 0.0, 0.0, 30.0)];
        let hits = search_items(
            &items,
            SearchItemsOptions {
                phrase: "hello",
                case_sensitive: false,
            },
        );
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn no_match_returns_empty() {
        let items = vec![mk("hello", 0.0, 0.0, 30.0)];
        let hits = search_items(
            &items,
            SearchItemsOptions {
                phrase: "zzz",
                case_sensitive: false,
            },
        );
        assert!(hits.is_empty());
    }
}
