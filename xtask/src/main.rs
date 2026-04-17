//! Internal workspace tooling.
//!
//! Subcommands:
//! - `pdfium-download` — fetch a prebuilt PDFium shared library for the current
//!   host into `pdfium/<triple>/` and print the env vars needed to load it.
//! - `parity` — placeholder for the liteparse/spdf JSON diff harness (Phase 11).
//! - `bench` — placeholder for the criterion benchmark driver (Phase 11).

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "xtask", version, about = "Internal spdf tooling")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Download the pdfium shared library for the current host.
    PdfiumDownload {
        /// Output directory (default: `pdfium/`).
        #[arg(long, default_value = "pdfium")]
        out: PathBuf,
        /// Release tag to fetch (default: `latest`, which follows the GitHub
        /// `releases/latest/download/<asset>` redirect).
        #[arg(long, default_value = "latest")]
        release: String,
    },
    /// Run the liteparse-vs-spdf parity harness.
    Parity {
        /// Directory containing fixture PDFs.
        #[arg(long)]
        corpus: PathBuf,
        /// Command to invoke liteparse (defaults to `lit parse --format json`).
        #[arg(long, default_value = "lit")]
        lit_bin: String,
        /// Command to invoke spdf (defaults to the just-built `spdf`).
        #[arg(long, default_value = "./target/debug/spdf")]
        spdf_bin: String,
    },
    /// Run simple timing benchmarks on a corpus.
    Bench {
        /// Directory containing PDF fixtures.
        #[arg(long)]
        corpus: PathBuf,
        /// spdf binary to time.
        #[arg(long, default_value = "./target/release/spdf")]
        spdf_bin: String,
        /// Iterations per file.
        #[arg(long, default_value_t = 3)]
        iterations: u32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::PdfiumDownload { out, release } => pdfium_download(&out, &release),
        Commands::Parity {
            corpus,
            lit_bin,
            spdf_bin,
        } => run_parity(&corpus, &lit_bin, &spdf_bin),
        Commands::Bench {
            corpus,
            spdf_bin,
            iterations,
        } => run_bench(&corpus, &spdf_bin, iterations),
    }
}

/// Download a prebuilt PDFium tarball from bblanchon/pdfium-binaries, extract
/// it, and print the env var(s) required to load the library.
fn pdfium_download(out_dir: &Path, release: &str) -> Result<()> {
    let (asset, dylib_rel) = current_host_asset()?;
    let url = if release == "latest" {
        format!("https://github.com/bblanchon/pdfium-binaries/releases/latest/download/{asset}")
    } else {
        format!("https://github.com/bblanchon/pdfium-binaries/releases/download/{release}/{asset}")
    };
    eprintln!("xtask: fetching {url}");

    fs::create_dir_all(out_dir)?;
    let target_dir = out_dir.join(current_triple());
    fs::create_dir_all(&target_dir)?;

    let resp = ureq::get(&url)
        .call()
        .with_context(|| format!("HTTP GET {url}"))?;
    if resp.status() != 200 {
        return Err(anyhow!("unexpected status {} for {url}", resp.status()));
    }
    let mut reader = resp.into_reader();
    let archive_path = target_dir.join(asset);
    let mut f = File::create(&archive_path)?;
    io::copy(&mut reader, &mut f)?;
    f.flush()?;
    drop(f);
    eprintln!("xtask: downloaded to {}", archive_path.display());

    extract_tgz(&archive_path, &target_dir)?;

    let dylib_path = target_dir.join(dylib_rel);
    if !dylib_path.exists() {
        return Err(anyhow!(
            "extracted archive but {} is missing — check asset contents",
            dylib_path.display()
        ));
    }

    let var = if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "windows") {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let dir = dylib_path
        .parent()
        .ok_or_else(|| anyhow!("no parent dir for {}", dylib_path.display()))?
        .canonicalize()?;
    println!("{var}={}", dir.display());
    eprintln!(
        "xtask: ready. Export {var}={} to load libpdfium.",
        dir.display()
    );
    Ok(())
}

fn current_host_asset() -> Result<(&'static str, &'static str)> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Ok(("pdfium-mac-arm64.tgz", "lib/libpdfium.dylib")),
        ("macos", "x86_64") => Ok(("pdfium-mac-x64.tgz", "lib/libpdfium.dylib")),
        ("linux", "x86_64") => Ok(("pdfium-linux-x64.tgz", "lib/libpdfium.so")),
        ("linux", "aarch64") => Ok(("pdfium-linux-arm64.tgz", "lib/libpdfium.so")),
        ("windows", "x86_64") => Ok(("pdfium-win-x64.tgz", "bin/pdfium.dll")),
        (os, arch) => Err(anyhow!("unsupported host: {os}/{arch}")),
    }
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

fn extract_tgz(archive: &Path, dest: &Path) -> Result<()> {
    let f = File::open(archive)?;
    let gz = flate2::read::GzDecoder::new(f);
    let mut tar = tar::Archive::new(gz);
    tar.unpack(dest)
        .with_context(|| format!("unpack {} into {}", archive.display(), dest.display()))?;
    Ok(())
}

/// Parity harness: parse every `*.pdf` in `corpus` with both liteparse and
/// spdf, then compare extracted text length and character overlap.
fn run_parity(corpus: &Path, lit_bin: &str, spdf_bin: &str) -> Result<()> {
    use std::process::Command;

    if !corpus.is_dir() {
        return Err(anyhow!("corpus not a directory: {}", corpus.display()));
    }

    let mut pdfs: Vec<PathBuf> = fs::read_dir(corpus)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("pdf"))
        .collect();
    pdfs.sort();

    if pdfs.is_empty() {
        eprintln!("xtask: no PDFs found in {}", corpus.display());
        return Ok(());
    }

    let mut total = 0usize;
    let mut matched = 0usize;
    println!("{:<40}  {:>10}  {:>10}  {:>6}", "file", "lit_chars", "spdf_chars", "%match");

    for pdf in pdfs {
        total += 1;
        let name = pdf.file_name().unwrap().to_string_lossy().into_owned();

        let lit_out = Command::new(lit_bin)
            .args(["parse", "--format", "text"])
            .arg(&pdf)
            .output();
        let spdf_out = Command::new(spdf_bin)
            .args(["parse", "--format", "text", "--no-ocr"])
            .arg(&pdf)
            .output();

        let lit_text = lit_out
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();
        let spdf_text = spdf_out
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();

        let pct = char_overlap_percent(&lit_text, &spdf_text);
        if pct >= 80.0 {
            matched += 1;
        }
        println!(
            "{:<40}  {:>10}  {:>10}  {:>5.1}%",
            truncate(&name, 40),
            lit_text.chars().count(),
            spdf_text.chars().count(),
            pct
        );
    }
    println!(
        "\nparity summary: {}/{} files ≥ 80% char overlap",
        matched, total
    );
    Ok(())
}

/// Bag-of-words character overlap: fraction of liteparse's non-whitespace
/// characters that also appear in spdf's output, counted with multiplicities.
fn char_overlap_percent(a: &str, b: &str) -> f64 {
    use std::collections::HashMap;
    let mut wanted: HashMap<char, usize> = HashMap::new();
    for c in a.chars().filter(|c| !c.is_whitespace()) {
        *wanted.entry(c).or_insert(0) += 1;
    }
    let total: usize = wanted.values().sum();
    if total == 0 {
        return 100.0;
    }
    let mut have: HashMap<char, usize> = HashMap::new();
    for c in b.chars().filter(|c| !c.is_whitespace()) {
        *have.entry(c).or_insert(0) += 1;
    }
    let mut matched = 0usize;
    for (c, n) in &wanted {
        matched += (*n).min(*have.get(c).unwrap_or(&0));
    }
    matched as f64 / total as f64 * 100.0
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max - 1).collect::<String>() + "…"
    }
}

/// Simple wall-clock benchmark: time `spdf parse` across a corpus and print
/// per-file averages.
fn run_bench(corpus: &Path, spdf_bin: &str, iterations: u32) -> Result<()> {
    use std::process::Command;
    use std::time::Instant;

    if !corpus.is_dir() {
        return Err(anyhow!("corpus not a directory: {}", corpus.display()));
    }
    let mut pdfs: Vec<PathBuf> = fs::read_dir(corpus)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("pdf"))
        .collect();
    pdfs.sort();

    println!("{:<40}  {:>10}  {:>10}", "file", "mean_ms", "min_ms");
    for pdf in pdfs {
        let name = pdf.file_name().unwrap().to_string_lossy().into_owned();
        let mut samples: Vec<u128> = Vec::with_capacity(iterations as usize);
        for _ in 0..iterations {
            let t0 = Instant::now();
            let status = Command::new(spdf_bin)
                .args(["parse", "--format", "text", "--no-ocr", "-q"])
                .arg(&pdf)
                .output();
            if status.is_err() || !status.as_ref().unwrap().status.success() {
                eprintln!("bench: {name} failed");
                continue;
            }
            samples.push(t0.elapsed().as_millis());
        }
        if samples.is_empty() {
            continue;
        }
        let mean = samples.iter().sum::<u128>() / samples.len() as u128;
        let min = *samples.iter().min().unwrap();
        println!("{:<40}  {:>10}  {:>10}", truncate(&name, 40), mean, min);
    }
    Ok(())
}
