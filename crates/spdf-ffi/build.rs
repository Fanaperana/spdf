//! Build script for the C-ABI crate.
//!
//! On macOS the neural OCR stack pulls in `buildid` (via burn -> cubecl-runtime),
//! which references `__mh_execute_header`. That symbol only exists in executables,
//! so linking this crate as a `cdylib` fails with "Undefined symbols". Telling the
//! linker to allow it to stay undefined (it is resolved, or never touched, at load
//! time) fixes the dylib link without affecting other targets.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-U,__mh_execute_header");
    }
}
