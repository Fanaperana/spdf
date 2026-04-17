//! C-ABI bindings for `spdf-core`.
//!
//! Exposes a minimal surface so Node.js (via `bun:ffi`, `ffi-napi`, `koffi`),
//! Deno, Python `ctypes`, and other FFI consumers can parse PDFs without
//! going through the CLI.
//!
//! # Memory model
//!
//! All functions that return `*mut c_char` transfer ownership to the caller.
//! Free the string with [`spdf_string_free`]. Never free with the system
//! allocator directly.
//!
//! # API summary
//!
//! ```text
//! int32_t  spdf_version(char** out);                     // semver string
//! int32_t  spdf_parse_path_json(const char* path,        // parse file on disk
//!                               const char* opts_json,   // optional JSON opts or NULL
//!                               char** out_json);        // JSON result
//! int32_t  spdf_parse_bytes_json(const uint8_t* data,    // parse bytes
//!                                size_t len,
//!                                const char* opts_json,
//!                                char** out_json);
//! void     spdf_string_free(char* s);                    // free returned string
//! ```
//!
//! Return codes: `0` = success, `< 0` = error (the string output then holds a
//! JSON error envelope of the form `{"error": "..."}`).

#![warn(clippy::all)]

use std::ffi::{c_char, c_int, CStr, CString};
use std::path::PathBuf;
use std::slice;

use spdf_core::SpdfParser;
use spdf_types::{OutputFormat, ParseConfig};

/// Release a string previously returned by any `spdf_*` function.
///
/// # Safety
/// `s` must be a pointer returned from this library or null. Passing any other
/// pointer is undefined behaviour.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spdf_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: contract documented above.
    drop(unsafe { CString::from_raw(s) });
}

/// Write the crate version into `*out`. Caller owns the returned string.
///
/// # Safety
/// `out` must be a valid non-null pointer to a writable `*mut c_char`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spdf_version(out: *mut *mut c_char) -> c_int {
    if out.is_null() {
        return -1;
    }
    let s = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
    unsafe {
        *out = s.into_raw();
    }
    0
}

/// Parse a PDF/office/image file at `path` and return a JSON document.
///
/// # Safety
/// `path` must be a NUL-terminated UTF-8 string. `opts_json` may be null; if
/// non-null it must be a NUL-terminated UTF-8 JSON object. `out_json` must be
/// a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spdf_parse_path_json(
    path: *const c_char,
    opts_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    if path.is_null() || out_json.is_null() {
        return write_error(out_json, "null pointer argument");
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => return write_error(out_json, &format!("path is not valid utf-8: {e}")),
    };
    let cfg = match parse_opts(opts_json) {
        Ok(c) => c,
        Err(e) => return write_error(out_json, &e),
    };
    let parser = match SpdfParser::new(cfg) {
        Ok(p) => p,
        Err(e) => return write_error(out_json, &format!("init: {e}")),
    };
    match parser.parse(PathBuf::from(path_str)) {
        Ok(result) => write_ok_json(out_json, &result),
        Err(e) => write_error(out_json, &format!("parse: {e}")),
    }
}

/// Parse a PDF from an in-memory buffer and return a JSON document.
///
/// # Safety
/// `data` must point to at least `len` bytes. `opts_json` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spdf_parse_bytes_json(
    data: *const u8,
    len: usize,
    opts_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    if data.is_null() || out_json.is_null() {
        return write_error(out_json, "null pointer argument");
    }
    let bytes = unsafe { slice::from_raw_parts(data, len) }.to_vec();
    let cfg = match parse_opts(opts_json) {
        Ok(c) => c,
        Err(e) => return write_error(out_json, &e),
    };
    let parser = match SpdfParser::new(cfg) {
        Ok(p) => p,
        Err(e) => return write_error(out_json, &format!("init: {e}")),
    };
    match parser.parse(bytes) {
        Ok(result) => write_ok_json(out_json, &result),
        Err(e) => write_error(out_json, &format!("parse: {e}")),
    }
}

fn parse_opts(opts_json: *const c_char) -> Result<ParseConfig, String> {
    if opts_json.is_null() {
        return Ok(ParseConfig {
            output_format: OutputFormat::Json,
            ..ParseConfig::default()
        });
    }
    let raw = unsafe { CStr::from_ptr(opts_json) }
        .to_str()
        .map_err(|e| format!("opts_json is not valid utf-8: {e}"))?;
    // The accepted schema is a strict subset of ParseConfig fields.
    let mut cfg: ParseConfig = serde_json::from_str(raw)
        .map_err(|e| format!("opts_json is not valid ParseConfig: {e}"))?;
    cfg.output_format = OutputFormat::Json;
    Ok(cfg)
}

fn write_ok_json(
    out: *mut *mut c_char,
    result: &spdf_types::ParseResult,
) -> c_int {
    let json = result
        .json
        .clone()
        .unwrap_or_else(|| spdf_core::to_json(result));
    let s = match serde_json::to_string(&json) {
        Ok(s) => s,
        Err(e) => return write_error(out, &format!("serialize: {e}")),
    };
    match CString::new(s) {
        Ok(cs) => unsafe {
            *out = cs.into_raw();
            0
        },
        Err(_) => write_error(out, "result contained interior NUL"),
    }
}

fn write_error(out: *mut *mut c_char, message: &str) -> c_int {
    if !out.is_null() {
        let envelope = serde_json::json!({ "error": message }).to_string();
        if let Ok(cs) = CString::new(envelope) {
            unsafe {
                *out = cs.into_raw();
            }
        }
    }
    -1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_returns_non_empty_string() {
        let mut out: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { spdf_version(&mut out) };
        assert_eq!(rc, 0);
        let s = unsafe { CStr::from_ptr(out) }.to_str().unwrap().to_owned();
        assert!(!s.is_empty());
        unsafe { spdf_string_free(out) };
    }

    #[test]
    fn null_path_returns_error_envelope() {
        let mut out: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { spdf_parse_path_json(std::ptr::null(), std::ptr::null(), &mut out) };
        assert_eq!(rc, -1);
        assert!(!out.is_null());
        let s = unsafe { CStr::from_ptr(out) }.to_str().unwrap();
        assert!(s.contains("error"));
        unsafe { spdf_string_free(out) };
    }
}
