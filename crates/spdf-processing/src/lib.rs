//! Text processing helpers ported from `liteparse/src/processing/*`
//! (excluding grid projection, which lives in `spdf-projection`).

#![warn(clippy::all)]

pub mod bbox;
pub mod clean_text;
pub mod markup;
pub mod ocr_utils;
pub mod search;
pub mod tables;
pub mod text_utils;
