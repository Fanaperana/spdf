//! Build script for `spdf-pdf`.
//!
//! When the `bundled-pdfium` feature is enabled (default), this downloads a
//! prebuilt `libpdfium` for the host target from `bblanchon/pdfium-binaries`
//! and generates `$OUT_DIR/pdfium_bundle.rs` which the crate includes via
//! `include!`. The binary then ships the library bytes in its `.rodata` and
//! extracts them to a user cache dir on first run.
//!
//! Opt-outs / overrides:
//! - `SPDF_PDFIUM_SKIP_BUNDLE=1`  — disable bundling (equivalent to turning
//!   off the feature; emits an empty bundle module).
//! - `SPDF_PDFIUM_LIB_FILE=/abs/path/libpdfium.so` — use a pre-staged library
//!   file instead of downloading.
//! - `SPDF_PDFIUM_RELEASE=chromium/6996` — pin a specific upstream release
//!   (default: `latest`).

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SPDF_PDFIUM_SKIP_BUNDLE");
    println!("cargo:rerun-if-env-changed=SPDF_PDFIUM_LIB_FILE");
    println!("cargo:rerun-if-env-changed=SPDF_PDFIUM_RELEASE");

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    let bundle_rs = out_dir.join("pdfium_bundle.rs");

    let feature_on = std::env::var_os("CARGO_FEATURE_BUNDLED_PDFIUM").is_some();
    let skip = std::env::var("SPDF_PDFIUM_SKIP_BUNDLE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    // docs.rs has no network access — skip the download so `cargo doc`
    // succeeds there even though the resulting crate can't load
    // pdfium at runtime.
    let on_docs_rs = std::env::var_os("DOCS_RS").is_some();

    if !feature_on || skip || on_docs_rs {
        write_empty_bundle(&bundle_rs);
        return;
    }

    match prepare_bundle(&out_dir) {
        Ok((lib_path, lib_name)) => write_bundle(&bundle_rs, &lib_path, &lib_name),
        Err(e) => {
            println!("cargo:warning=spdf-pdf: bundling pdfium disabled: {e}");
            write_empty_bundle(&bundle_rs);
        }
    }
}

fn write_empty_bundle(path: &Path) {
    let src = "pub const BYTES: &[u8] = &[];\npub const LIB_NAME: &str = \"\";\n";
    fs::write(path, src).expect("write empty pdfium_bundle.rs");
}

fn write_bundle(path: &Path, lib_path: &Path, lib_name: &str) {
    // Use `include_bytes!` with an absolute path so the library is embedded
    // without copying it into OUT_DIR again.
    let lit = lib_path.to_string_lossy().replace('\\', "\\\\");
    let src = format!(
        "pub const BYTES: &[u8] = include_bytes!(\"{lit}\");\n\
         pub const LIB_NAME: &str = \"{lib_name}\";\n"
    );
    fs::write(path, src).expect("write pdfium_bundle.rs");
    println!(
        "cargo:warning=spdf-pdf: embedding pdfium from {} ({} bytes)",
        lib_path.display(),
        fs::metadata(lib_path).map(|m| m.len()).unwrap_or(0)
    );
}

/// Returns (absolute path to the extracted shared library, library filename).
fn prepare_bundle(out_dir: &Path) -> Result<(PathBuf, String), String> {
    let (asset, dylib_rel, lib_name) = host_asset()?;

    // Allow a user-provided library to short-circuit the download.
    if let Ok(user_path) = std::env::var("SPDF_PDFIUM_LIB_FILE") {
        let p = PathBuf::from(user_path);
        if !p.is_file() {
            return Err(format!(
                "SPDF_PDFIUM_LIB_FILE set but not a file: {}",
                p.display()
            ));
        }
        return Ok((p, lib_name.to_string()));
    }

    let release = std::env::var("SPDF_PDFIUM_RELEASE").unwrap_or_else(|_| "latest".into());
    let url = if release == "latest" {
        format!("https://github.com/bblanchon/pdfium-binaries/releases/latest/download/{asset}")
    } else {
        format!("https://github.com/bblanchon/pdfium-binaries/releases/download/{release}/{asset}")
    };

    let stage = out_dir.join("pdfium-bundle");
    fs::create_dir_all(&stage).map_err(|e| format!("mkdir {}: {e}", stage.display()))?;
    let dylib_path = stage.join(dylib_rel);

    // Idempotent: only re-download if the extracted library is missing.
    if dylib_path.is_file() {
        return Ok((dylib_path, lib_name.to_string()));
    }

    let archive = stage.join(asset);
    if !archive.is_file() {
        download(&url, &archive)?;
    }
    extract_tgz(&archive, &stage)?;

    if !dylib_path.is_file() {
        return Err(format!(
            "extracted {} but expected library missing: {}",
            archive.display(),
            dylib_path.display()
        ));
    }
    Ok((dylib_path, lib_name.to_string()))
}

fn host_asset() -> Result<(&'static str, &'static str, &'static str), String> {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    match (os.as_str(), arch.as_str()) {
        ("macos", "aarch64") => Ok((
            "pdfium-mac-arm64.tgz",
            "lib/libpdfium.dylib",
            "libpdfium.dylib",
        )),
        ("macos", "x86_64") => Ok((
            "pdfium-mac-x64.tgz",
            "lib/libpdfium.dylib",
            "libpdfium.dylib",
        )),
        ("linux", "x86_64") => Ok(("pdfium-linux-x64.tgz", "lib/libpdfium.so", "libpdfium.so")),
        ("linux", "aarch64") => Ok(("pdfium-linux-arm64.tgz", "lib/libpdfium.so", "libpdfium.so")),
        ("windows", "x86_64") => Ok(("pdfium-win-x64.tgz", "bin/pdfium.dll", "pdfium.dll")),
        _ => Err(format!("unsupported host target: {os}/{arch}")),
    }
}

fn download(url: &str, dest: &Path) -> Result<(), String> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("GET {url}: {e}"))?;
    if resp.status() != 200 {
        return Err(format!("unexpected status {} for {url}", resp.status()));
    }
    let mut reader = resp.into_reader();
    let mut f = File::create(dest).map_err(|e| format!("create {}: {e}", dest.display()))?;
    io::copy(&mut reader, &mut f).map_err(|e| format!("write {}: {e}", dest.display()))?;
    f.flush().ok();
    Ok(())
}

fn extract_tgz(archive: &Path, dest: &Path) -> Result<(), String> {
    let f = File::open(archive).map_err(|e| format!("open {}: {e}", archive.display()))?;
    let gz = flate2::read::GzDecoder::new(f);
    tar::Archive::new(gz)
        .unpack(dest)
        .map_err(|e| format!("unpack {}: {e}", archive.display()))?;
    Ok(())
}
