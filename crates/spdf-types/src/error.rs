//! Error types shared across the workspace.

use thiserror::Error;

pub type SpdfResult<T> = Result<T, SpdfError>;

#[derive(Debug, Error)]
pub enum SpdfError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("document is password-protected")]
    PasswordRequired,

    #[error("invalid password for protected document")]
    InvalidPassword,

    #[error("unsupported input format: {0}")]
    UnsupportedFormat(String),

    #[error("PDF parse error: {0}")]
    Pdf(String),

    #[error("OCR error: {0}")]
    Ocr(String),

    #[error("conversion failed: {0}")]
    Conversion(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}
