//! OCR engine abstraction + implementations.
//!
//! - [`NeuralOcrEngine`] (feature `neural`): built-in PP-OCRv5 detection +
//!   recognition running on the GPU through Burn/wgpu, CPU fallback.
//! - [`HttpOcrEngine`] (feature `http`): remote OCR server.
//! - `TesseractEngine` (feature `tesseract`): system libtesseract.

#![warn(clippy::all)]

mod engine;
#[cfg(feature = "http")]
mod http;
#[cfg(feature = "neural")]
pub mod neural;
#[cfg(feature = "tesseract")]
mod tesseract_engine;

pub use engine::{OcrEngine, OcrOptions, OcrResult};

#[cfg(feature = "http")]
pub use http::HttpOcrEngine;

#[cfg(feature = "neural")]
pub use neural::NeuralOcrEngine;

#[cfg(feature = "tesseract")]
pub use tesseract_engine::TesseractEngine;
