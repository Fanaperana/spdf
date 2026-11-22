//! Build script for the `spdf` binary.
//!
//! The built-in neural OCR engine links the wgpu GPU stack, which imports
//! graphics / shell DLLs (opengl32, user32, gdi32, dxgi, ...). Windows would
//! load and initialise all of them on every launch, adding ~50 ms to each
//! run even when no page needs OCR. Delay-loading defers that cost to the
//! first GPU call, so born-digital PDFs keep their start-up time.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if !(msvc && windows) || std::env::var_os("CARGO_FEATURE_NEURAL_OCR").is_none() {
        return;
    }
    const DELAYED: &[&str] = &[
        "opengl32.dll",
        "user32.dll",
        "gdi32.dll",
        "dxgi.dll",
        "setupapi.dll",
        "oleaut32.dll",
        "combase.dll",
        "shell32.dll",
        "userenv.dll",
    ];
    for dll in DELAYED {
        println!("cargo:rustc-link-arg-bins=/DELAYLOAD:{dll}");
    }
    println!("cargo:rustc-link-arg-bins=delayimp.lib");
}
