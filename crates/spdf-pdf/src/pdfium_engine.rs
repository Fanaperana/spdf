//! PDFium-backed implementation of [`crate::PdfEngine`].
//!
//! Port target: `liteparse/src/engines/pdf/pdfjs.ts` (text extraction) +
//! `pdfium-renderer.ts` (rasterization). We unify both under PDFium so the
//! Rust build has a single native dependency.
//!
//! ## Library discovery
//!
//! We try, in order:
//! 1. `Pdfium::bind_to_library(path)` where `path` is `$PDFIUM_LIB_PATH` (an
//!    explicit file path).
//! 2. `Pdfium::bind_to_library(path)` with a path inferred from the `xtask
//!    pdfium-download` layout — `./pdfium/<triple>/lib/libpdfium.{dylib,so}`
//!    or `./pdfium/<triple>/bin/pdfium.dll`.
//! 3. `Pdfium::bind_to_system_library()` for system-wide installs.
//!
//! ## Threading
//!
//! `pdfium-render`'s bindings are not thread-safe by default; the
//! `thread_safe` feature adds an internal mutex. We still serialise calls
//! through the engine to be defensive against library misuse.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use pdfium_render::prelude::{PdfPageObjectsCommon, PdfRenderConfig, Pdfium};
use spdf_types::{Image, SpdfError, SpdfResult, TextItem};
use tracing::{debug, warn};

use crate::engine::{ExtractOptions, PageData, PdfDocumentHandle, PdfEngine};

fn candidate_library_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(explicit) = std::env::var("PDFIUM_LIB_PATH") {
        out.push(PathBuf::from(explicit));
    }
    let triple = current_triple();
    let (subdir, name) = if cfg!(target_os = "windows") {
        ("bin", "pdfium.dll")
    } else if cfg!(target_os = "macos") {
        ("lib", "libpdfium.dylib")
    } else {
        ("lib", "libpdfium.so")
    };
    for root in ["pdfium", "../pdfium", "../../pdfium"] {
        out.push(PathBuf::from(root).join(triple).join(subdir).join(name));
    }
    out
}

fn current_triple() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("windows", "x86_64") => "x86_64-pc-windows-msvc",
        _ => "unknown",
    }
}

/// Global PDFium library handle. PDFium's C API keeps internal global state;
/// a single process-wide instance is the safest usage.
fn pdfium() -> SpdfResult<&'static Mutex<Pdfium>> {
    static LIB: OnceLock<Result<Mutex<Pdfium>, String>> = OnceLock::new();
    let entry = LIB.get_or_init(|| {
        for candidate in candidate_library_paths() {
            if candidate.is_file() {
                let path_str = candidate.to_string_lossy().into_owned();
                match Pdfium::bind_to_library(&path_str) {
                    Ok(b) => {
                        debug!(path = %path_str, "pdfium: bound to explicit library");
                        return Ok(Mutex::new(Pdfium::new(b)));
                    }
                    Err(e) => warn!(path = %path_str, error = %e, "pdfium: explicit bind failed"),
                }
            }
        }
        match Pdfium::bind_to_system_library() {
            Ok(b) => {
                debug!("pdfium: bound to system library");
                Ok(Mutex::new(Pdfium::new(b)))
            }
            Err(e) => Err(format!(
                "no pdfium library found; set PDFIUM_LIB_PATH, run `cargo run -p xtask -- pdfium-download`, or install system pdfium. last error: {e}"
            )),
        }
    });
    entry
        .as_ref()
        .map_err(|e| SpdfError::Pdf(format!("pdfium unavailable: {e}")))
}

#[derive(Debug)]
pub struct PdfiumDoc {
    /// We keep the PDF bytes alive for the lifetime of the handle because
    /// `PdfDocument` borrows them internally. Simpler than juggling lifetimes
    /// in a trait object.
    bytes: Vec<u8>,
    num_pages: u32,
    password: Option<String>,
}

impl PdfDocumentHandle for PdfiumDoc {
    fn num_pages(&self) -> u32 {
        self.num_pages
    }
}

#[derive(Debug, Default)]
pub struct PdfiumEngine;

impl PdfiumEngine {
    pub fn new() -> Self {
        Self
    }

    fn with_doc<R>(
        &self,
        doc: &PdfiumDoc,
        f: impl FnOnce(&pdfium_render::prelude::PdfDocument<'_>) -> SpdfResult<R>,
    ) -> SpdfResult<R> {
        let mutex = pdfium()?;
        let guard = mutex
            .lock()
            .map_err(|_| SpdfError::Pdf("pdfium mutex poisoned".into()))?;
        let password = doc.password.as_deref();
        let pdf = guard
            .load_pdf_from_byte_slice(&doc.bytes, password)
            .map_err(|e| SpdfError::Pdf(format!("load_pdf_from_byte_slice: {e:?}")))?;
        f(&pdf)
    }
}

impl PdfEngine for PdfiumEngine {
    type Doc = PdfiumDoc;

    fn name(&self) -> &'static str {
        "pdfium"
    }

    fn load_bytes(&self, bytes: &[u8], password: Option<&str>) -> SpdfResult<Self::Doc> {
        let mutex = pdfium()?;
        let guard = mutex
            .lock()
            .map_err(|_| SpdfError::Pdf("pdfium mutex poisoned".into()))?;
        let pdf = guard
            .load_pdf_from_byte_slice(bytes, password)
            .map_err(|e| SpdfError::Pdf(format!("load_pdf_from_byte_slice: {e:?}")))?;
        let num_pages = pdf.pages().len() as u32;
        debug!(pages = num_pages, "pdfium: loaded document");
        drop(pdf);
        drop(guard);
        Ok(PdfiumDoc {
            bytes: bytes.to_vec(),
            num_pages,
            password: password.map(str::to_owned),
        })
    }

    fn load_path(&self, path: &Path, password: Option<&str>) -> SpdfResult<Self::Doc> {
        let bytes = std::fs::read(path)?;
        self.load_bytes(&bytes, password)
    }

    fn extract_page(
        &self,
        doc: &Self::Doc,
        page_num: u32,
        options: ExtractOptions,
    ) -> SpdfResult<PageData> {
        self.with_doc(doc, |pdf| {
            let page_idx: u16 = (page_num
                .checked_sub(1)
                .ok_or_else(|| SpdfError::InvalidInput("page numbers are 1-indexed".into()))?)
                as u16;
            let page = pdf
                .pages()
                .get(page_idx.into())
                .map_err(|e| SpdfError::Pdf(format!("page {page_num}: {e:?}")))?;

            let width = page.width().value as f64;
            let height = page.height().value as f64;

            let text_page = page
                .text()
                .map_err(|e| SpdfError::Pdf(format!("page {page_num} text: {e:?}")))?;

            let mut items: Vec<TextItem> = Vec::new();
            for segment in text_page.segments().iter() {
                let rect = segment.bounds();
                let str_content = segment.text();
                if str_content.is_empty() {
                    continue;
                }
                // PDFium gives points with bottom-left origin; flip Y to match
                // liteparse's top-left convention.
                let top = rect.top().value as f64;
                let bottom = rect.bottom().value as f64;
                let left = rect.left().value as f64;
                let right = rect.right().value as f64;
                let y_top = height - top;
                let item_h = top - bottom;
                let item_w = right - left;
                let mut ti = TextItem::new(
                    str_content,
                    left,
                    y_top,
                    item_w.max(0.0),
                    item_h.max(0.0),
                );
                ti.font_size = Some(item_h.max(1.0));
                items.push(ti);
            }

            let images = if options.extract_images {
                extract_images(&page, height)
            } else {
                Vec::new()
            };

            Ok(PageData {
                page_num,
                width,
                height,
                text_items: items,
                images,
            })
        })
    }

    fn render_page_png(&self, doc: &Self::Doc, page_num: u32, dpi: u32) -> SpdfResult<Vec<u8>> {
        self.with_doc(doc, |pdf| {
            let page_idx: u16 = (page_num
                .checked_sub(1)
                .ok_or_else(|| SpdfError::InvalidInput("page numbers are 1-indexed".into()))?)
                as u16;
            let page = pdf
                .pages()
                .get(page_idx.into())
                .map_err(|e| SpdfError::Pdf(format!("page {page_num}: {e:?}")))?;

            // 72 points per inch; convert DPI to a pixel scale.
            let scale = dpi as f32 / 72.0;
            let cfg = PdfRenderConfig::new().scale_page_by_factor(scale);

            let bitmap = page
                .render_with_config(&cfg)
                .map_err(|e| SpdfError::Pdf(format!("render page {page_num}: {e:?}")))?;

            let img = bitmap
                .as_image()
                .map_err(|e| SpdfError::Pdf(format!("bitmap to image: {e:?}")))?;
            let mut buf: Vec<u8> = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .map_err(|e| SpdfError::Pdf(format!("encode png: {e}")))?;
            Ok(buf)
        })
    }
}

/// Iterate the page's objects and collect every image object.
///
/// For each image, we capture PDF-space bounds (top-left origin, Y flipped)
/// and attempt to decode the raw pixel buffer to PNG via the `image` crate.
/// Decode failures are non-fatal: the image entry is still emitted with
/// `data = None` so downstream OCR can re-render the region from the page.
fn extract_images(page: &pdfium_render::prelude::PdfPage<'_>, page_height: f64) -> Vec<Image> {
    use pdfium_render::prelude::PdfPageObjectCommon;
    let mut out = Vec::new();
    for object in page.objects().iter() {
        let Some(image_obj) = object.as_image_object() else {
            continue;
        };
        let Ok(bounds) = object.bounds() else {
            continue;
        };
        let left = bounds.left().value as f64;
        let right = bounds.right().value as f64;
        let top = bounds.top().value as f64;
        let bottom = bounds.bottom().value as f64;
        let w = (right - left).max(0.0);
        let h = (top - bottom).max(0.0);
        if w <= 0.0 || h <= 0.0 {
            continue;
        }

        let png_bytes = encode_image_object_to_png(image_obj);

        out.push(Image {
            x: left,
            y: page_height - top,
            width: w,
            height: h,
            data: png_bytes,
            scale_factor: None,
            original_orientation_angle: None,
            image_type: Some("embedded".into()),
        });
    }
    out
}

/// Try to pull the decoded bitmap for an image object and re-encode it as PNG.
/// Returns `None` on any failure.
fn encode_image_object_to_png(
    image_obj: &pdfium_render::prelude::PdfPageImageObject<'_>,
) -> Option<Vec<u8>> {
    // `get_raw_image` returns a DynamicImage if pdfium could decode the stream.
    let dynamic = image_obj.get_raw_image().ok()?;
    let rgba = dynamic.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    if w == 0 || h == 0 {
        return None;
    }
    let mut buf = std::io::Cursor::new(Vec::with_capacity((w * h * 4) as usize));
    use image::ImageEncoder;
    image::codecs::png::PngEncoder::new(&mut buf)
        .write_image(rgba.as_raw(), w, h, image::ExtendedColorType::Rgba8)
        .ok()?;
    Some(buf.into_inner())
}
