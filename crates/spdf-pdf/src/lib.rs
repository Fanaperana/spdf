//! PDF engine abstraction + PDFium implementation.
//!
//! Port target: `liteparse/src/engines/pdf/*`.

#![warn(clippy::all)]

#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
mod cidfont;
mod engine;
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
mod form;
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
mod frame;
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
mod gutter;
#[cfg(feature = "pdfium")]
mod pdfium_engine;
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
mod rtl;
mod segtext;

pub use engine::{ExtractOptions, PageData, PdfDocumentHandle, PdfEngine};

#[cfg(feature = "pdfium")]
pub use pdfium_engine::PdfiumEngine;
