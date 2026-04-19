//! Python bindings for spdf.
//!
//! Exposes a minimal, idiomatic Python API over `spdf-core`. Result is
//! returned as a plain `dict` that matches the CLI's JSON schema, so
//! downstream Python code doesn't need Rust-specific glue.
#![allow(unsafe_op_in_unsafe_fn)]
//!
//! Build with [maturin](https://github.com/PyO3/maturin):
//!
//! ```shell
//! maturin develop --release -m crates/spdf-py/Cargo.toml
//! # or to build a wheel:
//! maturin build --release -m crates/spdf-py/Cargo.toml
//! ```
//!
//! Usage:
//!
//! ```python
//! import spdf
//! result = spdf.parse("doc.pdf", ocr=True, detect_tables=True)
//! print(result["text"])
//! for page in result["pages"]:
//!     print(page["pageNum"], len(page["textItems"]))
//! ```

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList, PyString};

use spdf_core::SpdfParser;
use spdf_types::{OutputFormat, ParseConfig, ParseInput};

/// Parse a PDF from a path or raw bytes and return the result as a dict.
///
/// Positional args:
///   * `source` — either a string / path-like (treated as a file path)
///     or `bytes` / `bytearray`.
///
/// Keyword args (all optional):
///   * `ocr` (bool, default True) — run OCR on pages with sparse text
///     or embedded images.
///   * `ocr_language` (str, default "eng") — Tesseract language code.
///     Pass comma-separated codes for multi-language.
///   * `ocr_server_url` (str) — HTTP OCR server endpoint.
///   * `tessdata_path` (str) — filesystem path to Tesseract language data.
///   * `max_pages` (int, default 1000) — hard cap on pages processed.
///   * `target_pages` (str) — `"1-5,10,15-20"` style selector.
///   * `dpi` (int, default 150) — rendering DPI for OCR / screenshots.
///   * `password` (str) — decryption password for the PDF.
///   * `precise_bounding_box` (bool, default True) — include `boundingBoxes`.
///   * `preserve_small_text` (bool, default False) — keep sub-2pt glyphs.
///   * `detect_tables` (bool, default False) — attach structured tables.
///   * `timeout_secs` (int) — abort if wall-clock exceeds this many seconds.
#[pyfunction]
#[pyo3(signature = (source, **kwargs))]
fn parse(py: Python<'_>, source: &Bound<'_, PyAny>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyObject> {
    let mut config = ParseConfig::default();
    // Force JSON output — we always return a dict.
    config.output_format = OutputFormat::Json;

    if let Some(opts) = kwargs {
        apply_kwargs(&mut config, opts)?;
    }

    let input = coerce_source(source)?;
    let parser = SpdfParser::new(config);

    let result = py
        .allow_threads(|| parser.parse(input))
        .map_err(|e| PyRuntimeError::new_err(format!("spdf parse: {e}")))?;

    let json_text = spdf_output::to_json_string(&result)
        .map_err(|e| PyRuntimeError::new_err(format!("spdf serialize: {e}")))?;
    let json_mod = py.import_bound("json")?;
    let loaded = json_mod.call_method1("loads", (json_text,))?;
    Ok(loaded.into())
}

/// Return the spdf-core crate version (semver string).
#[pyfunction]
fn version(py: Python<'_>) -> PyResult<PyObject> {
    Ok(PyString::new_bound(py, env!("CARGO_PKG_VERSION")).into())
}

fn coerce_source(obj: &Bound<'_, PyAny>) -> PyResult<ParseInput> {
    // bytes / bytearray → owned Vec<u8>
    if let Ok(bytes) = obj.downcast::<PyBytes>() {
        return Ok(ParseInput::Bytes(bytes.as_bytes().to_vec()));
    }
    if let Ok(bytearr) = obj.extract::<Vec<u8>>() {
        return Ok(ParseInput::Bytes(bytearr));
    }
    // Fallback: treat as a path (covers str, pathlib.Path via __fspath__).
    let fspath = obj
        .call_method0("__fspath__")
        .or_else(|_| obj.call_method0("__str__"))?;
    let path: String = fspath.extract()?;
    Ok(ParseInput::Path(path.into()))
}

fn apply_kwargs(cfg: &mut ParseConfig, opts: &Bound<'_, PyDict>) -> PyResult<()> {
    for (k, v) in opts.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "ocr" => cfg.ocr_enabled = v.extract()?,
            "ocr_language" => {
                let s: String = v.extract()?;
                cfg.ocr_language = parse_language(&s);
            }
            "ocr_server_url" => cfg.ocr_server_url = Some(v.extract()?),
            "tessdata_path" => cfg.tessdata_path = Some(v.extract()?),
            "max_pages" => cfg.max_pages = v.extract()?,
            "target_pages" => cfg.target_pages = Some(v.extract()?),
            "dpi" => cfg.dpi = v.extract()?,
            "password" => cfg.password = Some(v.extract()?),
            "precise_bounding_box" => cfg.precise_bounding_box = v.extract()?,
            "preserve_small_text" => cfg.preserve_very_small_text = v.extract()?,
            "detect_tables" => cfg.detect_tables = v.extract()?,
            "timeout_secs" => cfg.timeout_secs = Some(v.extract()?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown keyword argument: {other}"
                )));
            }
        }
    }
    Ok(())
}

fn parse_language(spec: &str) -> spdf_types::Language {
    let codes: Vec<String> = spec
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    match codes.len() {
        0 => spdf_types::Language::default(),
        1 => spdf_types::Language::Single(codes.into_iter().next().unwrap()),
        _ => spdf_types::Language::Multiple(codes),
    }
}

/// Module init. Exposes `parse` and `version` plus the `__version__`
/// dunder so users can do `spdf.__version__`.
#[pymodule]
fn spdf(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Silence unused-import warnings when the full PyList / PyAny
    // surface isn't touched directly.
    let _ = PyList::empty_bound(py);
    Ok(())
}
