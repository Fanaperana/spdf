//! Build script for `spdf-ocr`.
//!
//! With the `neural` feature on, this fetches the pretrained PaddleOCR
//! PP-OCRv5 mobile text detection (DBNet) and recognition (SVTR-LCNet, CTC)
//! ONNX models, plus the PP-OCRv5 server recogniser used for CJK lines and the
//! Latin recogniser fused in on Latin-script lines,, verifies them by SHA-256, and turns them into native Burn code
//! with `burn-onnx`. The weights end up embedded in the final binary, so a
//! plain `cargo build` produces a self-contained OCR engine: no Python, no
//! ONNX runtime, no system Tesseract.
//!
//! Model cache: `SPDF_OCR_MODEL_DIR` if set, else `%LOCALAPPDATA%\spdf\models`
//! (Windows) / `$XDG_CACHE_HOME/spdf/models` / `~/.cache/spdf/models`.
//! Files already present with the right hash are never re-downloaded.
//!
//! When the models cannot be obtained (offline build, docs.rs) the crate still
//! compiles; the neural engine then reports "models unavailable" at runtime.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SPDF_OCR_MODEL_DIR");
    println!("cargo:rerun-if-env-changed=SPDF_OCR_SKIP_MODELS");
    println!("cargo::rustc-check-cfg=cfg(spdf_neural_models)");
    #[cfg(feature = "neural")]
    neural::run();
}

#[cfg(feature = "neural")]
mod neural {
    use std::fs;
    use std::io::Read;
    use std::path::{Path, PathBuf};

    /// (file name, URL, SHA-256). PP-OCRv5 mobile models, Apache-2.0
    /// (PaddlePaddle/PaddleOCR), ONNX conversions published by RapidAI.
    const MODELS: &[(&str, &str, &str)] = &[
        (
            "ch_PP-OCRv5_det_mobile.onnx",
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
            "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
        ),
        (
            "ch_PP-OCRv5_rec_mobile.onnx",
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
            "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
        ),
        (
            "ppocrv5_dict.txt",
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/ppocrv5_dict.txt",
            "d1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b",
        ),
        // Larger PP-OCRv5 server recogniser (PP-HGNetV2), same dictionary.
        // Only used to re-read lines the mobile model decoded as CJK, where
        // the mobile model drops simple glyphs (一, 二, 入, 几).
        (
            "ch_PP-OCRv5_rec_server.onnx",
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx",
            "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
        ),
        // PP-OCRv5 Latin recogniser + dictionary: accented Latin letters
        // (ï, Ø, Ł, ź, ...) the Chinese dictionary lacks. Fused with the
        // primary decode on non-CJK lines.
        (
            "latin_PP-OCRv5_rec_mobile.onnx",
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx",
            "b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a",
        ),
        (
            "ppocrv5_latin_dict.txt",
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/paddle/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile/ppocrv5_latin_dict.txt",
            "3c0a8a79b612653c25f765271714f71281e4e955962c153e272b7b8c1d2b13ff",
        ),
    ];

    pub fn run() {
        let skip = std::env::var("SPDF_OCR_SKIP_MODELS")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        if skip || std::env::var_os("DOCS_RS").is_some() {
            return;
        }
        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
        match fetch_all() {
            Ok(dir) => {
                let det = dir.join(MODELS[0].0);
                let rec = dir.join(MODELS[1].0);
                let rec_server = dir.join(MODELS[3].0);
                let rec_latin = dir.join(MODELS[4].0);
                burn_onnx::ModelGen::new()
                    .input(det.to_str().expect("utf-8 model path"))
                    .input(rec.to_str().expect("utf-8 model path"))
                    .input(rec_server.to_str().expect("utf-8 model path"))
                    .input(rec_latin.to_str().expect("utf-8 model path"))
                    .out_dir("model/")
                    .load_strategy(burn_onnx::LoadStrategy::Bytes)
                    .run_from_script();
                for dict in [MODELS[2].0, MODELS[5].0] {
                    fs::copy(dir.join(dict), out_dir.join("model").join(dict))
                        .expect("copy OCR dictionary");
                }
                println!("cargo:rustc-cfg=spdf_neural_models");
            }
            Err(e) => {
                println!("cargo:warning=spdf-ocr: neural OCR models unavailable: {e}");
            }
        }
    }

    fn cache_dir() -> PathBuf {
        if let Some(d) = std::env::var_os("SPDF_OCR_MODEL_DIR") {
            return PathBuf::from(d);
        }
        if let Some(d) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(d).join("spdf").join("models");
        }
        if let Some(d) = std::env::var_os("XDG_CACHE_HOME") {
            return PathBuf::from(d).join("spdf").join("models");
        }
        if let Some(d) = std::env::var_os("HOME") {
            return PathBuf::from(d).join(".cache").join("spdf").join("models");
        }
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("models")
    }

    fn fetch_all() -> Result<PathBuf, String> {
        let dir = cache_dir();
        fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        for (name, url, sha) in MODELS {
            let path = dir.join(name);
            if path.exists() && sha256_file(&path)? == *sha {
                continue;
            }
            let bytes = download(url)?;
            let got = sha256_hex(&bytes);
            if got != *sha {
                return Err(format!("{name}: SHA-256 mismatch (got {got}, want {sha})"));
            }
            let tmp = dir.join(format!("{name}.part"));
            fs::write(&tmp, &bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
            fs::rename(&tmp, &path).map_err(|e| format!("rename {}: {e}", path.display()))?;
        }
        Ok(dir)
    }

    fn download(url: &str) -> Result<Vec<u8>, String> {
        let resp = ureq::get(url)
            .timeout(std::time::Duration::from_secs(300))
            .call()
            .map_err(|e| format!("GET {url}: {e}"))?;
        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| format!("read {url}: {e}"))?;
        Ok(buf)
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Ok(sha256_hex(&bytes))
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::Digest;
        let digest = sha2::Sha256::digest(bytes);
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}
