//! Build script for `spdf-ocr`.
//!
//! With the `neural` feature on, this fetches the pretrained PaddleOCR
//! PP-OCRv5 mobile text detection (DBNet) and recognition (SVTR-LCNet, CTC)
//! ONNX models, plus the PP-OCRv5 server recogniser used for CJK lines and the
//! Latin recogniser fused in on Latin-script lines, verifies them by SHA-256,
//! simplifies their graphs (`src/neural/fold.rs`: element-wise ops folded into
//! the convolutions, wide convolutions split by input channel) and turns them
//! into native Burn code with `burn-onnx`. The weights end up embedded in the final binary, so a
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
    println!("cargo:rerun-if-changed=src/neural/fold.rs");
    println!("cargo:rerun-if-env-changed=SPDF_OCR_MODEL_DIR");
    println!("cargo:rerun-if-env-changed=SPDF_OCR_SKIP_MODELS");
    println!("cargo::rustc-check-cfg=cfg(spdf_neural_models)");
    #[cfg(feature = "neural")]
    neural::run();
}

#[cfg(feature = "neural")]
#[path = "src/neural/fold.rs"]
mod fold;

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
                // Fold the element-wise ops around each convolution into
                // its weights before code generation (see `fold.rs`).
                let opt_dir = out_dir.join("onnx-opt");
                fs::create_dir_all(&opt_dir).expect("create onnx-opt dir");
                let opt = |i: usize| {
                    let dst = opt_dir.join(MODELS[i].0);
                    match optimize_onnx(&dir.join(MODELS[i].0), &dst) {
                        Ok(()) => dst,
                        Err(e) => {
                            println!(
                                "cargo:warning=spdf-ocr: {}: graph folding skipped: {e}",
                                MODELS[i].0
                            );
                            dir.join(MODELS[i].0)
                        }
                    }
                };
                let det = opt(0);
                let rec = opt(1);
                let rec_server = opt(3);
                let rec_latin = opt(4);
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

    /// Rewrite `src` with [`crate::fold`] applied and write it to `dst`.
    fn optimize_onnx(src: &Path, dst: &Path) -> Result<(), String> {
        use crate::fold;
        use onnx_ir::{Message, ModelProto};
        use std::collections::HashSet;
        const FLOAT: i32 = 1;

        let bytes = fs::read(src).map_err(|e| format!("read {}: {e}", src.display()))?;
        let mut model = ModelProto::parse_from_bytes(&bytes).map_err(|e| format!("parse: {e}"))?;
        let graph = model.graph.as_mut().ok_or("model has no graph")?;

        let mut g = fold::Graph::default();
        for t in &graph.initializer {
            if t.data_type != FLOAT {
                continue;
            }
            let data: Vec<f32> = if !t.raw_data.is_empty() {
                t.raw_data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            } else {
                t.float_data.clone()
            };
            let numel: i64 = t.dims.iter().product();
            if numel as usize != data.len() {
                continue; // external data or malformed: leave it alone
            }
            g.consts.insert(
                t.name.clone(),
                fold::Const {
                    dims: t.dims.clone(),
                    data,
                },
            );
        }
        for (i, n) in graph.node.iter().enumerate() {
            let attr = |name: &str| n.attribute.iter().find(|a| a.name == name);
            let zero_pad = attr("pads").is_none_or(|a| a.ints.iter().all(|&p| p == 0))
                && attr("auto_pad").is_none_or(|a| matches!(&a.s[..], b"" | b"NOTSET" | b"VALID"));
            let mut node = fold::Node::new(&n.op_type, &[], &[]);
            node.inputs = n.input.clone();
            node.outputs = n.output.clone();
            node.zero_pad = zero_pad;
            node.epsilon = attr("epsilon").map_or(1e-5, |a| a.f);
            node.group = attr("group").map_or(1, |a| a.i);
            node.attrs_from = Some(i);
            g.nodes.push(node);
        }
        g.outputs = graph.output.iter().map(|v| v.name.clone()).collect();
        let stats = g.optimize();

        let protos = std::mem::take(&mut graph.node);
        let mut names = HashSet::new();
        graph.node = g
            .nodes
            .iter()
            .filter(|n| !n.removed)
            .map(|n| {
                let mut p = match n.attrs_from {
                    Some(j) => protos[j].clone(),
                    None => onnx_ir::NodeProto::new(),
                };
                p.op_type = n.op.clone();
                // Nodes split from one source must not share its name.
                if !names.insert(p.name.clone()) || p.name.is_empty() {
                    p.name = format!("{}__node", n.outputs[0]);
                    names.insert(p.name.clone());
                }
                p.input = n.inputs.clone();
                p.output = n.outputs.clone();
                p
            })
            .collect();
        let used: HashSet<String> = graph
            .node
            .iter()
            .flat_map(|n| n.input.iter().cloned())
            .chain(graph.output.iter().map(|v| v.name.clone()))
            .collect();
        let had: HashSet<String> = graph.initializer.iter().map(|t| t.name.clone()).collect();
        graph.initializer.retain(|t| used.contains(&t.name));
        // Old exporters also list initializers as graph inputs.
        graph
            .input
            .retain(|v| !had.contains(&v.name) || used.contains(&v.name));
        let mut fresh: Vec<&String> = g
            .live_consts()
            .into_iter()
            .filter(|n| !had.contains(n))
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|n| g.consts.get_key_value(&n).expect("live const").0)
            .collect();
        fresh.sort();
        for name in fresh {
            let c = &g.consts[name];
            let mut t = onnx_ir::TensorProto::new();
            t.name = name.clone();
            t.dims = c.dims.clone();
            t.data_type = FLOAT;
            t.raw_data = c
                .data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
                .into();
            graph.initializer.push(t);
        }
        let mut ints: Vec<(&String, &Vec<i64>)> = g
            .int_consts
            .iter()
            .filter(|(n, _)| used.contains(*n))
            .collect();
        ints.sort();
        for (name, v) in ints {
            const INT64: i32 = 7;
            let mut t = onnx_ir::TensorProto::new();
            t.name = name.clone();
            t.dims = vec![v.len() as i64];
            t.data_type = INT64;
            t.raw_data = v
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>()
                .into();
            graph.initializer.push(t);
        }
        let out = model
            .write_to_bytes()
            .map_err(|e| format!("serialize: {e}"))?;
        fs::write(dst, out).map_err(|e| format!("write {}: {e}", dst.display()))?;
        // Build-script stdout is kept in the build log (`cargo -vv`).
        println!(
            "spdf-ocr: {}: folded {} identity, {} post-conv, {} pre-conv ops; split {} convs",
            src.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            stats.identity,
            stats.after_conv,
            stats.before_conv,
            stats.split_conv
        );
        Ok(())
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
