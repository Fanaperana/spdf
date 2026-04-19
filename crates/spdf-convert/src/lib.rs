//! Format conversion. Port target: `liteparse/src/conversion/convertToPdf.ts`.
//!
//! Shells out to LibreOffice (office + spreadsheet docs) and ImageMagick
//! (raster/vector images). Plain-text formats are returned as
//! [`ConversionResult::PlainText`] so the orchestrator can short-circuit.

#![warn(clippy::all)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use spdf_types::{SpdfError, SpdfResult};
use tempfile::TempDir;
use tracing::debug;

/// Outcome of a conversion attempt.
#[derive(Debug)]
pub enum ConversionResult {
    /// The input is already a PDF (or we produced one). The temp directory
    /// (if any) is kept alive via `_tempdir` so callers don't have to worry
    /// about cleanup ordering.
    Pdf {
        pdf_path: PathBuf,
        original_extension: String,
        #[allow(dead_code)]
        _tempdir: Option<TempDir>,
    },
    /// The input is a plain text format; its contents can be used directly.
    PlainText { content: String },
}

const OFFICE_EXTS: &[&str] = &[
    "doc", "docx", "docm", "dot", "dotm", "dotx", "odt", "ott", "ppt", "pptx", "pptm", "pot",
    "potm", "potx", "odp", "otp", "rtf", "pages", "key",
];
const SPREADSHEET_EXTS: &[&str] = &[
    "xls", "xlsx", "xlsm", "xlsb", "ods", "ots", "csv", "tsv", "numbers",
];
const IMAGE_EXTS: &[&str] = &[
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "svg",
];
const HTML_EXTS: &[&str] = &["htm", "html", "xhtml"];
const PLAIN_TEXT_EXTS: &[&str] = &["txt", "md", "log"];
const GS_REQUIRED_EXTS: &[&str] = &["svg", "eps", "ps", "ai"];

/// Guess a file extension from raw bytes.
pub fn guess_extension_from_buffer(bytes: &[u8]) -> Option<&'static str> {
    infer::get(bytes).map(|t| t.extension())
}

/// Convert a path to PDF. Office docs use LibreOffice, images use ImageMagick,
/// plain text is returned verbatim. PDF inputs pass through unchanged.
pub fn convert_path_to_pdf(path: &Path, password: Option<&str>) -> SpdfResult<ConversionResult> {
    if !path.exists() {
        return Err(SpdfError::InvalidInput(format!(
            "file not found: {}",
            path.display()
        )));
    }

    let ext = guess_extension(path)
        .unwrap_or_default()
        .to_ascii_lowercase();

    if ext == "pdf" {
        return Ok(ConversionResult::Pdf {
            pdf_path: path.to_path_buf(),
            original_extension: ext,
            _tempdir: None,
        });
    }

    if PLAIN_TEXT_EXTS.contains(&ext.as_str()) {
        let content = std::fs::read_to_string(path)?;
        return Ok(ConversionResult::PlainText { content });
    }

    if OFFICE_EXTS.contains(&ext.as_str()) || SPREADSHEET_EXTS.contains(&ext.as_str()) {
        let tmp = tempdir()?;
        let pdf = convert_office_document(path, tmp.path(), password)?;
        return Ok(ConversionResult::Pdf {
            pdf_path: pdf,
            original_extension: ext,
            _tempdir: Some(tmp),
        });
    }

    if IMAGE_EXTS.contains(&ext.as_str()) {
        let tmp = tempdir()?;
        let pdf = convert_image_to_pdf(path, tmp.path())?;
        return Ok(ConversionResult::Pdf {
            pdf_path: pdf,
            original_extension: ext,
            _tempdir: Some(tmp),
        });
    }

    if HTML_EXTS.contains(&ext.as_str()) {
        let tmp = tempdir()?;
        let pdf = convert_office_document(path, tmp.path(), password)?;
        return Ok(ConversionResult::Pdf {
            pdf_path: pdf,
            original_extension: ext,
            _tempdir: Some(tmp),
        });
    }

    // Unknown extension: try to read as UTF-8 text, matching liteparse's
    // fallthrough for unknown content.
    match std::fs::read_to_string(path) {
        Ok(content) => Ok(ConversionResult::PlainText { content }),
        Err(_) => Err(SpdfError::UnsupportedFormat(ext)),
    }
}

fn guess_extension(path: &Path) -> Option<String> {
    if let Some(e) = path.extension().and_then(|s| s.to_str()) {
        if !e.is_empty() {
            return Some(e.to_string());
        }
    }
    let mut f = std::fs::File::open(path).ok()?;
    use std::io::Read;
    let mut buf = vec![0u8; 4096];
    let n = f.read(&mut buf).ok()?;
    buf.truncate(n);
    infer::get(&buf).map(|t| t.extension().to_string())
}

fn tempdir() -> SpdfResult<TempDir> {
    tempfile::Builder::new()
        .prefix("spdf-")
        .tempdir()
        .map_err(|e| SpdfError::Conversion(format!("tempdir: {e}")))
}

fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> SpdfResult<String> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| SpdfError::Conversion(format!("spawn: {e}")))?;

    let start = std::time::Instant::now();
    loop {
        match child
            .try_wait()
            .map_err(|e| SpdfError::Conversion(format!("try_wait: {e}")))?
        {
            Some(status) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut o) = child.stdout.take() {
                    use std::io::Read;
                    o.read_to_string(&mut stdout).ok();
                }
                if let Some(mut e) = child.stderr.take() {
                    use std::io::Read;
                    e.read_to_string(&mut stderr).ok();
                }
                if status.success() {
                    return Ok(stdout);
                }
                return Err(SpdfError::Conversion(format!(
                    "command exited {}: {}",
                    status.code().unwrap_or(-1),
                    stderr.trim()
                )));
            }
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    return Err(SpdfError::Conversion(format!(
                        "command timeout after {:?}",
                        timeout
                    )));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn which(bin: &str) -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let (prog, args) = ("where", vec![bin]);
    #[cfg(not(target_os = "windows"))]
    let (prog, args) = ("which", vec![bin]);

    let out = Command::new(prog)
        .args(args)
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(PathBuf::from(line))
    }
}

fn is_path_executable(p: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(m) = std::fs::metadata(p) {
            return m.permissions().mode() & 0o111 != 0;
        }
        false
    }
    #[cfg(not(unix))]
    {
        p.exists()
    }
}

/// Resolve the LibreOffice binary.
pub fn find_libreoffice() -> Option<PathBuf> {
    for name in ["libreoffice", "soffice"] {
        if let Some(p) = which(name) {
            return Some(p);
        }
    }
    let candidates: &[&str] = &[
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/libreoffice",
        "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
    ];
    for c in candidates {
        let p = Path::new(c);
        if is_path_executable(p) {
            return Some(p.to_path_buf());
        }
    }
    None
}

/// Resolve an ImageMagick binary (`magick` then fall back to `convert`).
pub fn find_imagemagick() -> Option<PathBuf> {
    for name in ["magick", "convert"] {
        if let Some(p) = which(name) {
            #[cfg(target_os = "windows")]
            {
                if name == "convert" {
                    let normalized = p.to_string_lossy().to_ascii_lowercase();
                    if normalized.ends_with("system32\\convert.exe") {
                        continue;
                    }
                }
            }
            let ok = Command::new(&p)
                .arg("-version")
                .output()
                .map(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .to_ascii_lowercase()
                        .contains("imagemagick")
                })
                .unwrap_or(false);
            if ok {
                return Some(p);
            }
        }
    }
    None
}

fn convert_office_document(
    file_path: &Path,
    output_dir: &Path,
    password: Option<&str>,
) -> SpdfResult<PathBuf> {
    let bin = find_libreoffice().ok_or_else(|| {
        SpdfError::Conversion(
            "LibreOffice not found. Install with `brew install --cask libreoffice`, \
             `apt-get install libreoffice`, or `choco install libreoffice-fresh`."
                .into(),
        )
    })?;
    debug!(bin = %bin.display(), "spdf: converting via LibreOffice");

    let mut cmd = Command::new(&bin);
    cmd.args([
        "--headless",
        "--invisible",
        "--convert-to",
        "pdf",
        "--outdir",
    ])
    .arg(output_dir);
    if let Some(pw) = password {
        cmd.arg(format!("--infilter=:{pw}"));
    }
    cmd.arg(file_path);

    run_with_timeout(&mut cmd, Duration::from_secs(120))?;

    let base = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| SpdfError::Conversion("no file stem".into()))?;
    let pdf = output_dir.join(format!("{base}.pdf"));
    if !pdf.exists() {
        return Err(SpdfError::Conversion(
            "LibreOffice reported success but output PDF is missing".into(),
        ));
    }
    Ok(pdf)
}

fn convert_image_to_pdf(file_path: &Path, output_dir: &Path) -> SpdfResult<PathBuf> {
    let bin = find_imagemagick().ok_or_else(|| {
        SpdfError::Conversion(
            "ImageMagick not found. Install with `brew install imagemagick`, \
             `apt-get install imagemagick`, or `choco install imagemagick.app`."
                .into(),
        )
    })?;
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if GS_REQUIRED_EXTS.contains(&ext.as_str()) && which("gs").is_none() {
        return Err(SpdfError::Conversion(format!(
            "Ghostscript is required to convert .{ext} files. \
             Install with `brew install ghostscript` / `apt-get install ghostscript`."
        )));
    }

    let base = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| SpdfError::Conversion("no file stem".into()))?;
    let pdf = output_dir.join(format!("{base}.pdf"));

    let mut cmd = Command::new(&bin);
    cmd.arg(file_path)
        .args(["-density", "150", "-units", "PixelsPerInch"])
        .arg(&pdf);
    run_with_timeout(&mut cmd, Duration::from_secs(60))?;

    if !pdf.exists() {
        return Err(SpdfError::Conversion(
            "ImageMagick reported success but output PDF is missing".into(),
        ));
    }
    Ok(pdf)
}

/// Write `bytes` to a temp file and convert to PDF.
pub fn convert_bytes_to_pdf(
    bytes: &[u8],
    extension_hint: Option<&str>,
    password: Option<&str>,
) -> SpdfResult<ConversionResult> {
    let ext = extension_hint
        .map(|s| s.trim_start_matches('.').to_ascii_lowercase())
        .or_else(|| guess_extension_from_buffer(bytes).map(|s| s.to_string()))
        .unwrap_or_else(|| "bin".into());

    let tmp = tempdir()?;
    let path = tmp.path().join(format!("input.{ext}"));
    std::fs::write(&path, bytes)?;
    match convert_path_to_pdf(&path, password)? {
        ConversionResult::Pdf {
            pdf_path,
            original_extension,
            _tempdir,
        } => Ok(ConversionResult::Pdf {
            pdf_path,
            original_extension,
            _tempdir: Some(_tempdir.unwrap_or(tmp)),
        }),
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sample.txt");
        std::fs::write(&p, "hello").unwrap();
        match convert_path_to_pdf(&p, None).unwrap() {
            ConversionResult::PlainText { content } => assert_eq!(content, "hello"),
            _ => panic!("expected PlainText"),
        }
    }

    #[test]
    fn pdf_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("foo.pdf");
        std::fs::write(&p, b"%PDF-1.4\n").unwrap();
        match convert_path_to_pdf(&p, None).unwrap() {
            ConversionResult::Pdf {
                pdf_path,
                original_extension,
                ..
            } => {
                assert_eq!(pdf_path, p);
                assert_eq!(original_extension, "pdf");
            }
            _ => panic!("expected Pdf"),
        }
    }

    #[test]
    fn missing_file_is_error() {
        let err = convert_path_to_pdf(Path::new("/no/such/file.pdf"), None).unwrap_err();
        match err {
            SpdfError::InvalidInput(_) => {}
            _ => panic!("expected InvalidInput"),
        }
    }
}
