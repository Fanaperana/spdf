//! `spdf` CLI entry point.
//!
//! Mirrors the command surface of `lit` in `liteparse/cli/parse.ts`.
//! Phase 9 adds every advanced flag; today we ship the essential subset so
//! the pipeline is exercisable end-to-end.

use std::io::{self, Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use spdf_core::{OutputFormat, SpdfParser};
use spdf_types::{Language, ParseInput};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Debug, Parser)]
#[command(name = "spdf", version, about = "Fast PDF parsing and OCR")]
struct Cli {
    /// Suppress progress output on stderr.
    #[arg(short = 'q', long, global = true)]
    quiet: bool,

    /// Verbose logging. `-v` = info, `-vv` = debug, `-vvv` = trace. Overrides RUST_LOG.
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Parse a document and emit text or JSON.
    Parse(ParseArgs),
    /// Render pages to PNG images on disk.
    Screenshot(ScreenshotArgs),
    /// Parse every document in a directory and write outputs side-by-side.
    BatchParse(BatchArgs),
}

#[derive(Debug, Parser)]
struct ParseArgs {
    /// Input file path. Use `-` to read bytes from stdin.
    file: String,

    /// Output file path. Defaults to stdout.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = CliFormat::Text)]
    format: CliFormat,

    /// Disable OCR even if the document needs it.
    #[arg(long = "no-ocr", default_value_t = false)]
    no_ocr: bool,

    /// HTTP OCR server URL (matches `--ocr-server-url` in liteparse).
    #[arg(long)]
    ocr_server_url: Option<String>,

    /// OCR language code(s). Pass multiple times or comma-separate for
    /// multi-language Tesseract (e.g. `--ocr-language eng --ocr-language fra`
    /// or `--ocr-language eng,fra`). Defaults to `en`.
    #[arg(long = "ocr-language", value_delimiter = ',')]
    ocr_language: Vec<String>,

    /// Path to a tessdata directory (Tesseract language data).
    #[arg(long)]
    tessdata_path: Option<PathBuf>,

    /// Max pages to process.
    #[arg(long, default_value_t = 10_000)]
    max_pages: u32,

    /// Target pages (e.g. `1-5,10,15-20`).
    #[arg(long)]
    target_pages: Option<String>,

    /// DPI for rendering.
    #[arg(long, default_value_t = 150)]
    dpi: u32,

    /// Password for encrypted PDFs.
    #[arg(long)]
    password: Option<String>,

    /// Enable grid-projection debug tracing.
    #[arg(long = "trace-grid", default_value_t = false)]
    trace_grid: bool,

    /// Visualize grid rows/columns to this PNG path.
    #[arg(long = "visualize-grid")]
    visualize_grid: Option<PathBuf>,
}

#[derive(Debug, Parser)]
struct ScreenshotArgs {
    file: String,

    #[arg(short = 'o', long, default_value = "./screenshots")]
    output_dir: PathBuf,

    #[arg(long)]
    target_pages: Option<String>,

    #[arg(long, default_value_t = 150)]
    dpi: u32,

    #[arg(long)]
    password: Option<String>,
}

#[derive(Debug, Parser)]
struct BatchArgs {
    /// Directory containing input documents.
    input_dir: PathBuf,
    /// Directory where outputs will be written (created if missing).
    output_dir: PathBuf,

    #[arg(long, value_enum, default_value_t = CliFormat::Text)]
    format: CliFormat,

    #[arg(long = "no-ocr", default_value_t = false)]
    no_ocr: bool,

    #[arg(long)]
    ocr_server_url: Option<String>,

    /// OCR language code(s). Repeat or comma-separate for multi-language.
    #[arg(long = "ocr-language", value_delimiter = ',')]
    ocr_language: Vec<String>,

    /// Path to a tessdata directory.
    #[arg(long)]
    tessdata_path: Option<PathBuf>,

    #[arg(long, default_value_t = 10_000)]
    max_pages: u32,

    #[arg(long, default_value_t = 150)]
    dpi: u32,

    #[arg(long)]
    password: Option<String>,

    /// Filter by extension (e.g. `pdf`). Defaults to all files.
    #[arg(long)]
    extension: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliFormat {
    Text,
    Json,
}

impl From<CliFormat> for OutputFormat {
    fn from(f: CliFormat) -> Self {
        match f {
            CliFormat::Text => OutputFormat::Text,
            CliFormat::Json => OutputFormat::Json,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    // `-v` levels override RUST_LOG; otherwise respect the env.
    let filter = match cli.verbose {
        0 => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        1 => EnvFilter::new("info"),
        2 => EnvFilter::new("debug"),
        _ => EnvFilter::new("trace"),
    };
    fmt().with_env_filter(filter).with_writer(io::stderr).init();

    let result = match cli.command {
        Commands::Parse(args) => run_parse(args, cli.quiet),
        Commands::Screenshot(args) => run_screenshot(args, cli.quiet),
        Commands::BatchParse(args) => run_batch(args, cli.quiet),
    };

    // Print any error ourselves so it reaches the user BEFORE we redirect
    // stderr below. If we returned `Result` from main, the Rust runtime would
    // print after we've already muted stderr.
    let exit_code = match &result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {e:#}");
            1
        }
    };

    // Suppress libtesseract's `ObjectCache(...)~ObjectCache(): WARNING! LEAK!`
    // messages that its C++ destructors print to stderr during process
    // teardown. Our tracing warnings have already been flushed.
    suppress_late_stderr();

    std::process::exit(exit_code);
}

#[cfg(unix)]
fn suppress_late_stderr() {
    use std::io::Write as _;
    let _ = io::stderr().flush();
    // Safety: we're about to exit; overwriting fd 2 only affects writes that
    // happen after this point (i.e. C++ static destructors in libtesseract).
    unsafe {
        let devnull = libc::open(b"/dev/null\0".as_ptr() as *const _, libc::O_WRONLY);
        if devnull >= 0 {
            libc::dup2(devnull, 2);
            libc::close(devnull);
        }
    }
}

#[cfg(not(unix))]
fn suppress_late_stderr() {}

fn run_parse(args: ParseArgs, quiet: bool) -> Result<()> {
    let parser = SpdfParser::builder()
        .ocr_enabled(!args.no_ocr)
        .output_format(args.format.into())
        .max_pages(args.max_pages)
        .dpi(args.dpi)
        .build();
    let mut cfg = parser.config().clone();
    cfg.target_pages = args.target_pages.clone();
    cfg.password = args.password.clone();
    cfg.ocr_server_url = args.ocr_server_url.clone();
    if let Some(lang) = parse_language(&args.ocr_language) {
        cfg.ocr_language = lang;
    }
    if let Some(p) = &args.tessdata_path {
        cfg.tessdata_path = Some(p.to_string_lossy().into_owned());
    }
    if args.trace_grid || args.visualize_grid.is_some() {
        let mut debug = spdf_types::DebugConfig::default();
        debug.enabled = true;
        debug.trace = Some(args.trace_grid);
        if let Some(p) = &args.visualize_grid {
            debug.visualize = Some(true);
            debug.visualize_path = Some(p.to_string_lossy().into_owned());
        }
        cfg.debug = Some(debug);
    }
    let parser = SpdfParser::new(cfg);

    let input: ParseInput = if args.file == "-" {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf).context("reading stdin")?;
        if buf.is_empty() {
            anyhow::bail!("no data received from stdin");
        }
        ParseInput::Bytes(buf)
    } else {
        ParseInput::Path(PathBuf::from(&args.file))
    };

    if !quiet {
        eprintln!("Parsing {}...", args.file);
    }

    let result = parser.parse(input)?;
    let rendered = parser.format(&result);

    match args.output {
        Some(path) => std::fs::write(&path, rendered)?,
        None => {
            let mut out = io::stdout().lock();
            out.write_all(rendered.as_bytes())?;
            out.write_all(b"\n")?;
        }
    }
    Ok(())
}

fn run_screenshot(args: ScreenshotArgs, quiet: bool) -> Result<()> {
    let mut cfg = spdf_types::ParseConfig::default();
    cfg.dpi = args.dpi;
    cfg.password = args.password;
    cfg.target_pages = args.target_pages.clone();
    let parser = SpdfParser::new(cfg);

    let input = ParseInput::Path(PathBuf::from(&args.file));
    let shots = parser.screenshot(input, None)?;
    std::fs::create_dir_all(&args.output_dir)?;

    for s in shots {
        let path = args.output_dir.join(format!("page-{}.png", s.page_num));
        std::fs::write(&path, &s.image_buffer)?;
        if !quiet {
            eprintln!("wrote {}", path.display());
        }
    }
    Ok(())
}

fn run_batch(args: BatchArgs, quiet: bool) -> Result<()> {
    if !args.input_dir.is_dir() {
        anyhow::bail!("not a directory: {}", args.input_dir.display());
    }
    std::fs::create_dir_all(&args.output_dir)?;

    let want_ext = args.extension.as_deref().map(str::to_ascii_lowercase);
    let output_format: OutputFormat = args.format.into();
    let out_ext = match output_format {
        OutputFormat::Json => "json",
        OutputFormat::Text => "txt",
    };

    let mut files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&args.input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if let Some(want) = &want_ext {
            let have = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase());
            if have.as_deref() != Some(want.as_str()) {
                continue;
            }
        }
        files.push(path);
    }
    files.sort();

    for file in files {
        let stem = file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "output".into());
        let out_path = args.output_dir.join(format!("{stem}.{out_ext}"));

        let mut cfg = spdf_types::ParseConfig::default();
        cfg.output_format = output_format;
        cfg.ocr_enabled = !args.no_ocr;
        cfg.ocr_server_url = args.ocr_server_url.clone();
        cfg.max_pages = args.max_pages;
        cfg.dpi = args.dpi;
        cfg.password = args.password.clone();
        if let Some(lang) = parse_language(&args.ocr_language) {
            cfg.ocr_language = lang;
        }
        if let Some(p) = &args.tessdata_path {
            cfg.tessdata_path = Some(p.to_string_lossy().into_owned());
        }

        let parser = SpdfParser::new(cfg);
        match parser.parse(ParseInput::Path(file.clone())) {
            Ok(result) => {
                std::fs::write(&out_path, parser.format(&result))?;
                if !quiet {
                    eprintln!("parsed {} -> {}", file.display(), out_path.display());
                }
            }
            Err(e) => eprintln!("failed {}: {e}", file.display()),
        }
    }
    Ok(())
}

/// Build a `Language` from CLI values. Returns `None` when the user didn't
/// pass `--ocr-language`, so the default is preserved.
fn parse_language(values: &[String]) -> Option<Language> {
    let cleaned: Vec<String> = values
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    match cleaned.len() {
        0 => None,
        1 => Some(Language::Single(cleaned.into_iter().next().unwrap())),
        _ => Some(Language::Multiple(cleaned)),
    }
}
