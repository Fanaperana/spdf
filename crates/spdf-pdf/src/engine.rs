//! Engine trait and shared data types.

use spdf_types::{Image, SpdfResult, TextItem};

/// Options controlling page extraction.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtractOptions {
    /// Whether to extract embedded image bounds (needed for selective OCR).
    pub extract_images: bool,
}

/// Minimal page data required by the projection/OCR pipeline.
#[derive(Debug, Clone)]
pub struct PageData {
    pub page_num: u32,
    pub width: f64,
    pub height: f64,
    pub text_items: Vec<TextItem>,
    pub images: Vec<Image>,
}

/// An opaque handle to a loaded PDF document.
pub trait PdfDocumentHandle: Send {
    fn num_pages(&self) -> u32;
}

/// Synchronous PDF engine interface. Engines own their own locking internally;
/// spdf-core will wrap a single instance behind a `Mutex` or route calls
/// through a dedicated thread if needed.
pub trait PdfEngine: Send + Sync {
    type Doc: PdfDocumentHandle;

    fn name(&self) -> &'static str;

    fn load_bytes(&self, bytes: &[u8], password: Option<&str>) -> SpdfResult<Self::Doc>;
    fn load_path(&self, path: &std::path::Path, password: Option<&str>) -> SpdfResult<Self::Doc>;

    fn extract_page(
        &self,
        doc: &Self::Doc,
        page_num: u32,
        options: ExtractOptions,
    ) -> SpdfResult<PageData>;

    /// Render a page to a PNG byte buffer at the given DPI.
    fn render_page_png(&self, doc: &Self::Doc, page_num: u32, dpi: u32) -> SpdfResult<Vec<u8>>;

    fn close(&self, doc: Self::Doc) -> SpdfResult<()> {
        drop(doc);
        Ok(())
    }
}
