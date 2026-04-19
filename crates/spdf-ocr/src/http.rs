//! HTTP OCR client. See `liteparse/OCR_API_SPEC.md`.
//!
//! Uses a shared tokio runtime so the rest of the spdf pipeline can stay
//! synchronous. Creating a full runtime per request is wasteful, so we keep a
//! lazily-initialised `current_thread` runtime internally.

use std::sync::OnceLock;
use std::time::Duration;

use serde::Deserialize;
use spdf_types::{SpdfError, SpdfResult};
use tokio::runtime::Runtime;

use crate::engine::{OcrEngine, OcrOptions, OcrResult};

fn rt() -> SpdfResult<&'static Runtime> {
    static RT: OnceLock<Runtime> = OnceLock::new();
    if let Some(r) = RT.get() {
        return Ok(r);
    }
    let r = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| SpdfError::Ocr(format!("tokio runtime: {e}")))?;
    // Racing threads may both try to init; OnceLock handles it.
    Ok(RT.get_or_init(|| r))
}

/// HTTP OCR client for servers implementing the LiteParse OCR API.
#[derive(Debug, Clone)]
pub struct HttpOcrEngine {
    url: String,
    client: reqwest::Client,
}

impl HttpOcrEngine {
    pub fn new(url: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest client builds");
        Self {
            url: url.into(),
            client,
        }
    }
}

#[derive(Debug, Deserialize)]
struct Response {
    results: Vec<OcrResult>,
}

impl OcrEngine for HttpOcrEngine {
    fn name(&self) -> &'static str {
        "http"
    }

    fn recognize(&self, image: &[u8], options: &OcrOptions) -> SpdfResult<Vec<OcrResult>> {
        let url = self.url.clone();
        let client = self.client.clone();
        let language = options
            .languages
            .first()
            .cloned()
            .unwrap_or_else(|| "en".into());
        let image = image.to_vec();

        rt()?.block_on(async move {
            let form = reqwest::multipart::Form::new()
                .part(
                    "file",
                    reqwest::multipart::Part::bytes(image)
                        .file_name("page.png")
                        .mime_str("image/png")
                        .map_err(|e| SpdfError::Ocr(format!("mime: {e}")))?,
                )
                .text("language", language);

            let resp = client
                .post(&url)
                .multipart(form)
                .send()
                .await
                .map_err(|e| SpdfError::Ocr(format!("http send: {e}")))?;

            if !resp.status().is_success() {
                return Err(SpdfError::Ocr(format!(
                    "OCR server returned HTTP {}",
                    resp.status()
                )));
            }
            let body: Response = resp
                .json()
                .await
                .map_err(|e| SpdfError::Ocr(format!("http decode: {e}")))?;
            Ok(body.results)
        })
    }
}
