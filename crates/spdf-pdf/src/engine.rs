//! Engine trait and shared data types.

use spdf_types::{Image, Ruling, SpdfResult, TextItem};

/// Options controlling page extraction.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtractOptions {
    /// Whether to extract embedded image bounds (needed for selective OCR).
    pub extract_images: bool,
    /// Whether to collect straight line rules (table borders) from vector
    /// graphics. Only layout-aware output (markdown tables) needs them.
    pub extract_rulings: bool,
}

/// Minimal page data required by the projection/OCR pipeline.
#[derive(Debug, Clone)]
pub struct PageData {
    pub page_num: u32,
    pub width: f64,
    pub height: f64,
    pub text_items: Vec<TextItem>,
    pub images: Vec<Image>,
    /// Axis-aligned rules drawn on the page (empty unless requested).
    pub rulings: Vec<Ruling>,
}

/// A rendered page as packed 8-bit RGB pixels, row-major, no padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbPixels {
    pub width: usize,
    pub height: usize,
    /// `width × height × 3` bytes.
    pub data: Vec<u8>,
}

impl RgbPixels {
    /// Drop the alpha channel of packed RGBA pixels.
    pub fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Self {
        let data = rgba
            .chunks_exact(4)
            .take(width * height)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        Self {
            width,
            height,
            data,
        }
    }
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

    /// Extract several pages, in the order given. Engines that pay a
    /// per-call cost to open the document should override this to pay it
    /// once per batch.
    fn extract_pages(
        &self,
        doc: &Self::Doc,
        page_nums: &[u32],
        options: ExtractOptions,
    ) -> SpdfResult<Vec<PageData>> {
        page_nums
            .iter()
            .map(|&n| self.extract_page(doc, n, options))
            .collect()
    }

    /// Render a page to a PNG byte buffer at the given DPI.
    fn render_page_png(&self, doc: &Self::Doc, page_num: u32, dpi: u32) -> SpdfResult<Vec<u8>>;

    /// Render a page to RGB pixels at the given DPI, for consumers that
    /// work on pixels (OCR) and would otherwise decode the PNG again.
    /// Default: decode [`PdfEngine::render_page_png`].
    fn render_page_rgb(&self, doc: &Self::Doc, page_num: u32, dpi: u32) -> SpdfResult<RgbPixels> {
        let png = self.render_page_png(doc, page_num, dpi)?;
        let img = image::load_from_memory(&png)
            .map_err(|e| spdf_types::SpdfError::Pdf(format!("decode rendered page: {e}")))?
            .to_rgb8();
        Ok(RgbPixels {
            width: img.width() as usize,
            height: img.height() as usize,
            data: img.into_raw(),
        })
    }

    fn close(&self, doc: Self::Doc) -> SpdfResult<()> {
        drop(doc);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Doc(u32);
    impl PdfDocumentHandle for Doc {
        fn num_pages(&self) -> u32 {
            self.0
        }
    }

    #[derive(Default)]
    struct Counting {
        calls: AtomicUsize,
    }

    impl PdfEngine for Counting {
        type Doc = Doc;
        fn name(&self) -> &'static str {
            "counting"
        }
        fn load_bytes(&self, _: &[u8], _: Option<&str>) -> SpdfResult<Doc> {
            Ok(Doc(3))
        }
        fn load_path(&self, _: &std::path::Path, _: Option<&str>) -> SpdfResult<Doc> {
            Ok(Doc(3))
        }
        fn extract_page(&self, doc: &Doc, n: u32, _: ExtractOptions) -> SpdfResult<PageData> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 || n > doc.0 {
                return Err(spdf_types::SpdfError::InvalidInput("no such page".into()));
            }
            Ok(PageData {
                page_num: n,
                width: 1.0,
                height: 1.0,
                text_items: Vec::new(),
                images: Vec::new(),
                rulings: Vec::new(),
            })
        }
        fn render_page_png(&self, _: &Doc, _: u32, _: u32) -> SpdfResult<Vec<u8>> {
            // A 2×1 RGBA PNG: one red, one half-transparent blue pixel.
            let img = image::RgbaImage::from_raw(2, 1, vec![255, 0, 0, 255, 0, 0, 255, 128])
                .expect("buffer size");
            let mut buf = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .expect("encode");
            Ok(buf)
        }
    }

    #[test]
    fn default_rgb_render_decodes_the_png() {
        let e = Counting::default();
        let doc = e.load_bytes(&[], None).unwrap();
        let px = e.render_page_rgb(&doc, 1, 72).unwrap();
        assert_eq!((px.width, px.height), (2, 1));
        assert_eq!(px.data, vec![255, 0, 0, 0, 0, 255]);
    }

    #[test]
    fn rgba_to_rgb_drops_alpha() {
        let px = RgbPixels::from_rgba(2, 1, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(px.data, vec![1, 2, 3, 5, 6, 7]);
    }

    #[test]
    fn default_extract_pages_keeps_requested_order() {
        let e = Counting::default();
        let doc = e.load_bytes(&[], None).unwrap();
        let pages = e
            .extract_pages(&doc, &[3, 1, 2], ExtractOptions::default())
            .unwrap();
        let nums: Vec<u32> = pages.iter().map(|p| p.page_num).collect();
        assert_eq!(nums, vec![3, 1, 2]);
        assert_eq!(e.calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn default_extract_pages_propagates_errors() {
        let e = Counting::default();
        let doc = e.load_bytes(&[], None).unwrap();
        assert!(
            e.extract_pages(&doc, &[1, 9], ExtractOptions::default())
                .is_err()
        );
    }
}
