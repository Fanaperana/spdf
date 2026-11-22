//! PDF engine abstraction + PDFium implementation.
//!
//! Port target: `liteparse/src/engines/pdf/*`.

#![warn(clippy::all)]

mod engine;
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
mod form;
#[cfg(feature = "pdfium")]
mod pdfium_engine;
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
mod rtl;

pub use engine::{ExtractOptions, PageData, PdfDocumentHandle, PdfEngine};

#[cfg(feature = "pdfium")]
pub use pdfium_engine::PdfiumEngine;
