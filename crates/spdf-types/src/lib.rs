//! Public schema and error types for the spdf workspace.
//!
//! Mirrors [`liteparse/src/core/types.ts`] so JSON output stays schema-compatible.

#![warn(clippy::all, missing_debug_implementations)]

mod config;
mod error;
mod input;
mod text;

pub use config::{DebugConfig, Language, OutputFormat, ParseConfig, RegionFilter};
pub use error::{SpdfError, SpdfResult};
pub use input::ParseInput;
pub use text::{
    BoundingBox, Coordinates, Image, JsonPage, JsonTextItem, MarkupData, OcrData, ParseResult,
    ParseResultJson, ParsedPage, ProjectionTextBox, ScreenshotResult, Snap, TextItem,
};
