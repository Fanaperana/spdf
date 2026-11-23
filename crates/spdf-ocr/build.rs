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
    println!("cargo:rerun-if-changed=src/neural/head.rs");
    println!("cargo:rerun-if-changed=src/neural/staged.rs");
    println!("cargo:rerun-if-changed=src/neural/layers.rs");
    println!("cargo:rerun-if-env-changed=SPDF_OCR_MODEL_DIR");
    println!("cargo:rerun-if-env-changed=SPDF_OCR_SKIP_MODELS");
    println!("cargo::rustc-check-cfg=cfg(spdf_neural_models)");
    println!("cargo::rustc-check-cfg=cfg(spdf_rec_logits)");
    println!("cargo::rustc-check-cfg=cfg(spdf_rec_head)");
    #[cfg(feature = "neural")]
    neural::run();
}

#[cfg(feature = "neural")]
#[path = "src/neural/fold.rs"]
mod fold;

#[cfg(feature = "neural")]
#[path = "src/neural/head.rs"]
mod head;

#[cfg(feature = "neural")]
#[path = "src/neural/staged.rs"]
mod staged;

#[cfg(feature = "neural")]
#[path = "src/neural/layers.rs"]
mod layers;

#[cfg(feature = "neural")]
mod neural {
    use std::fs;
    use std::io::Read;
    use std::path::{Path, PathBuf};

    /// How `burn-onnx` spells an ONNX `HardSwish` in generated code, and the
    /// fused in-place kernel that replaces it (`src/neural/ops.rs`).
    const BURN_HARD_SWISH: &str = "burn::tensor::activation::hard_swish(";
    const FUSED_HARD_SWISH: &str = "crate::neural::ops::hard_swish(";

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
                let model_dir = out_dir.join("model");
                fs::create_dir_all(&model_dir).expect("create model dir");
                // Recognisers also lose their output softmax (see
                // `strip_output_softmax`) and, when all three allow it, their
                // output classifier, which the runtime then applies itself
                // (`split_output_linear`, `head.rs`). Both are all-or-nothing
                // across the three, which share the runtime decoding path.
                let opt = |i: usize, rec: bool, split: bool| -> (PathBuf, usize, bool) {
                    let dst = opt_dir.join(MODELS[i].0);
                    let head = split.then(|| model_dir.join(MODELS[i].0.replace(".onnx", ".head")));
                    match optimize_onnx(&dir.join(MODELS[i].0), &dst, rec, head.as_deref()) {
                        Ok((stripped, split)) => (dst, stripped, split),
                        Err(e) => {
                            println!(
                                "cargo:warning=spdf-ocr: {}: graph folding skipped: {e}",
                                MODELS[i].0
                            );
                            (dir.join(MODELS[i].0), 0, false)
                        }
                    }
                };
                let det = opt(0, false, false).0;
                let mut recs: Vec<(PathBuf, usize, bool)> =
                    [1, 3, 4].iter().map(|&i| opt(i, true, true)).collect();
                let heads = recs.iter().filter(|r| r.2).count();
                if heads == 3 {
                    println!("cargo:rustc-cfg=spdf_rec_head");
                } else {
                    if heads > 0 {
                        recs = [1, 3, 4].iter().map(|&i| opt(i, true, false)).collect();
                    }
                    println!("cargo:warning=spdf-ocr: recogniser heads kept ({heads}/3 split)");
                }
                let logits = recs.iter().filter(|r| r.1 == 1).count();
                if logits == 3 {
                    println!("cargo:rustc-cfg=spdf_rec_logits");
                } else {
                    println!(
                        "cargo:warning=spdf-ocr: recogniser softmax kept ({logits}/3 stripped)"
                    );
                }
                let [rec, rec_server, rec_latin] = [0, 1, 2].map(|i| recs[i].0.clone());
                burn_onnx::ModelGen::new()
                    .input(det.to_str().expect("utf-8 model path"))
                    .input(rec.to_str().expect("utf-8 model path"))
                    .input(rec_server.to_str().expect("utf-8 model path"))
                    .input(rec_latin.to_str().expect("utf-8 model path"))
                    .out_dir("model/")
                    .load_strategy(burn_onnx::LoadStrategy::Bytes)
                    .run_from_script();
                // Every model's hard-swish goes through the fused in-place
                // kernel (see `ops.rs`); the recognisers also get a
                // stage-by-stage forward pass (see `staged.rs`).
                for path in [&det, &rec, &rec_server, &rec_latin] {
                    let stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .expect("model name");
                    let rs = model_dir.join(format!("{stem}.rs"));
                    let mut src = fs::read_to_string(&rs).expect("read generated model");
                    src = src.replace(BURN_HARD_SWISH, FUSED_HARD_SWISH);
                    // In large models, each layer's weight pages go back
                    // to the OS as soon as the layer has run (see
                    // `layers.rs`).
                    let model = if path == &det {
                        "Det"
                    } else if path == &rec {
                        "Mobile"
                    } else if path == &rec_server {
                        "Server"
                    } else {
                        "Latin"
                    };
                    let bpk =
                        fs::read(model_dir.join(format!("{stem}.bpk"))).expect("read weights");
                    match crate::layers::bpk_spans(&bpk) {
                        _ if bpk.len() < crate::layers::LAYER_RELEASE_MIN_WEIGHTS => {}
                        Ok(spans) => {
                            src = crate::layers::release_after_layers(&src, &spans, model).0;
                        }
                        Err(e) => println!(
                            "cargo:warning=spdf-ocr: {stem}: per-layer weight release skipped: {e}"
                        ),
                    }
                    if path != &det {
                        let staged = crate::staged::staged_impl(&src).unwrap_or_else(|| {
                            panic!("{}: no Model::forward to stage", rs.display())
                        });
                        src.push_str(&staged);
                    }
                    fs::write(&rs, src).expect("write generated model");
                }
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

    /// Fold a model's graph (see `fold.rs`) into `dst`; with
    /// `strip_softmax`, also drop its output softmax, and with `head`, also
    /// detach its output classifier into that file (see `head.rs`). Returns
    /// the number of softmax nodes stripped and whether the classifier was
    /// detached.
    fn optimize_onnx(
        src: &Path,
        dst: &Path,
        strip_softmax: bool,
        head: Option<&Path>,
    ) -> Result<(usize, bool), String> {
        use crate::fold;
        use onnx_ir::{Message, ModelProto};
        use std::collections::HashSet;
        const FLOAT: i32 = 1;

        let bytes = fs::read(src).map_err(|e| format!("read {}: {e}", src.display()))?;
        let mut model = ModelProto::parse_from_bytes(&bytes).map_err(|e| format!("parse: {e}"))?;
        // Softmax's default axis is -1 since opset 13, 1 before.
        let opset = model
            .opset_import
            .iter()
            .filter(|o| o.domain.is_empty() || o.domain == "ai.onnx")
            .map(|o| o.version)
            .max()
            .unwrap_or(13);
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
            node.axis = attr("axis").map_or(if opset >= 13 { -1 } else { 1 }, |a| a.i);
            node.attrs_from = Some(i);
            g.nodes.push(node);
        }
        g.outputs = graph.output.iter().map(|v| v.name.clone()).collect();
        for v in &graph.output {
            let rank = v
                .type_
                .as_ref()
                .and_then(|t| t.tensor_type().shape.as_ref())
                .map(|s| s.dim.len());
            if let Some(r) = rank.filter(|&r| r > 0) {
                g.output_ranks.insert(v.name.clone(), r);
            }
        }
        let stats = g.optimize();
        let stripped = if strip_softmax {
            g.strip_output_softmax()
        } else {
            0
        };
        let split = match head {
            Some(path) if stripped == 1 => match g.split_output_linear() {
                Some(h) => {
                    fs::write(path, crate::head::pack(h.k, h.n, &h.w, &h.b))
                        .map_err(|e| format!("write {}: {e}", path.display()))?;
                    // The graph now returns the classifier's input features.
                    for v in &mut graph.output {
                        let shape = v
                            .type_
                            .mut_or_insert_default()
                            .mut_tensor_type()
                            .shape
                            .mut_or_insert_default();
                        if let Some(d) = shape.dim.last_mut() {
                            d.set_dim_value(h.k as i64);
                        }
                    }
                    true
                }
                None => false,
            },
            _ => false,
        };

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
            "spdf-ocr: {}: folded {} identity, {} post-conv, {} pre-conv ops; split {} convs; stripped {stripped} softmax; head split: {split}",
            src.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            stats.identity,
            stats.after_conv,
            stats.before_conv,
            stats.split_conv
        );
        Ok((stripped, split))
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
