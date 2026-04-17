//! OCR engine abstraction + implementations.
//!
//! Port target: `liteparse/src/engines/ocr/*`.

#![warn(clippy::all)]

mod engine;
#[cfg(feature = "http")]
mod http;
#[cfg(feature = "tesseract")]
mod tesseract_engine;

pub use engine::{OcrEngine, OcrOptions, OcrResult};

#[cfg(feature = "http")]
pub use http::HttpOcrEngine;

#[cfg(feature = "tesseract")]
pub use tesseract_engine::TesseractEngine;
