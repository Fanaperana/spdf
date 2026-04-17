//! PDF engine abstraction + PDFium implementation.
//!
//! Port target: `liteparse/src/engines/pdf/*`.

#![warn(clippy::all)]

mod engine;
#[cfg(feature = "pdfium")]
mod pdfium_engine;

pub use engine::{ExtractOptions, PageData, PdfDocumentHandle, PdfEngine};

#[cfg(feature = "pdfium")]
pub use pdfium_engine::PdfiumEngine;
