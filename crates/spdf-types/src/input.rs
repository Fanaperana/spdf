//! Accepted inputs for [`crate::ParseConfig`]-driven operations.

use std::path::PathBuf;

/// A document source: either a file path or raw bytes.
///
/// Mirrors `LiteParseInput` in the TypeScript codebase.
#[derive(Debug, Clone)]
pub enum ParseInput {
    /// Path to a file on disk.
    Path(PathBuf),
    /// In-memory bytes. PDFs go straight to the parser with zero disk I/O;
    /// non-PDF bytes are written to a temp file for format conversion.
    Bytes(Vec<u8>),
}

impl From<PathBuf> for ParseInput {
    fn from(p: PathBuf) -> Self {
        ParseInput::Path(p)
    }
}

impl From<&std::path::Path> for ParseInput {
    fn from(p: &std::path::Path) -> Self {
        ParseInput::Path(p.to_path_buf())
    }
}

impl From<String> for ParseInput {
    fn from(s: String) -> Self {
        ParseInput::Path(PathBuf::from(s))
    }
}

impl From<&str> for ParseInput {
    fn from(s: &str) -> Self {
        ParseInput::Path(PathBuf::from(s))
    }
}

impl From<Vec<u8>> for ParseInput {
    fn from(b: Vec<u8>) -> Self {
        ParseInput::Bytes(b)
    }
}
