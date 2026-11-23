//! Release each layer's embedded weights right after the layer has run.
//!
//! A model's weights are `include_bytes!` statics that tensors alias without
//! a copy (see `resident.rs`): each weight page joins the working set when a
//! layer first reads it and stays there until it is released. Releasing per
//! stage (`run_rec`) still lets a whole stage's weights pile up during that
//! stage; for the server recogniser that is 24-27 MB of pages, most of them
//! belonging to layers that are already done.
//!
//! The build script calls [`release_after_layers`] on each generated model
//! source. After every `let <out> = self.<field>.forward(..);` whose field
//! is a leaf layer with tensors in the model's `.bpk` (a convolution or a
//! linear layer), it inserts
//! `crate::neural::layer_done(crate::neural::OcrModel::<M>, start, end);`
//! with the absolute byte range of that layer's tensors in the `.bpk`. The
//! runtime then drops the whole pages inside that range. A concurrent batch
//! still inside the same layer faults the pages back in from the executable
//! (a soft fault), so the output is unchanged either way.
//!
//! Only models of at least [`LAYER_RELEASE_MIN_WEIGHTS`] bytes get the
//! calls: those run stage-major on the CPU, so their line batches move
//! through each stage together and a released layer is rarely needed again.
//! The ~8 MB mobile models run their batches independently, and re-faulting
//! each layer once per batch measured 3-5% slower for a 3-4 MB smaller peak.
//!
//! [`bpk_spans`] reads the tensor offsets from the `.bpk` header (CBOR
//! metadata, see the `burn-pack` crate docs) with a minimal CBOR reader.
//!
//! No dependencies: the build script includes this file directly.

#![allow(dead_code)]

use std::collections::HashMap;

/// Models with at least this many bytes of weights release them layer by
/// layer; the same bound as the runtime's stage-major threshold.
pub const LAYER_RELEASE_MIN_WEIGHTS: usize = 32 << 20;

/// `burnpack` header: magic (4) + version (2) + metadata size (4).
const HEADER_SIZE: usize = 10;
/// The tensor data section starts at this alignment after the metadata.
const TENSOR_ALIGNMENT: usize = 256;
/// "BURN", little-endian.
const MAGIC: [u8; 4] = *b"NRUB";

/// A decoded CBOR value (only what burnpack metadata uses).
#[derive(Debug, Clone, PartialEq)]
enum Cbor {
    Uint(u64),
    Neg(u64),
    Bytes(Vec<u8>),
    Text(String),
    Array(Vec<Cbor>),
    Map(Vec<(Cbor, Cbor)>),
    Simple(u64),
}

impl Cbor {
    fn get(&self, key: &str) -> Option<&Cbor> {
        match self {
            Cbor::Map(kv) => kv
                .iter()
                .find(|(k, _)| matches!(k, Cbor::Text(t) if t == key))
                .map(|(_, v)| v),
            _ => None,
        }
    }
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn byte(&mut self) -> Result<u8, String> {
        let v = *self.b.get(self.pos).ok_or("cbor: truncated")?;
        self.pos += 1;
        Ok(v)
    }

    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.pos.checked_add(n).ok_or("cbor: overflow")?;
        let s = self.b.get(self.pos..end).ok_or("cbor: truncated")?;
        self.pos = end;
        Ok(s)
    }

    /// The argument of an initial byte with additional info `info`.
    fn arg(&mut self, info: u8) -> Result<u64, String> {
        Ok(match info {
            0..=23 => info as u64,
            24 => self.byte()? as u64,
            25 => u16::from_be_bytes(self.take(2)?.try_into().unwrap()) as u64,
            26 => u32::from_be_bytes(self.take(4)?.try_into().unwrap()) as u64,
            27 => u64::from_be_bytes(self.take(8)?.try_into().unwrap()),
            _ => return Err(format!("cbor: unsupported length info {info}")),
        })
    }

    fn value(&mut self, depth: usize) -> Result<Cbor, String> {
        if depth > 64 {
            return Err("cbor: too deep".into());
        }
        let ib = self.byte()?;
        let (major, info) = (ib >> 5, ib & 0x1f);
        if info == 31 {
            return Err("cbor: indefinite lengths unsupported".into());
        }
        let n = self.arg(info)?;
        let len = |n: u64| usize::try_from(n).map_err(|_| "cbor: length".to_string());
        Ok(match major {
            0 => Cbor::Uint(n),
            1 => Cbor::Neg(n),
            2 => Cbor::Bytes(self.take(len(n)?)?.to_vec()),
            3 => Cbor::Text(
                String::from_utf8(self.take(len(n)?)?.to_vec()).map_err(|_| "cbor: bad utf-8")?,
            ),
            4 => {
                let n = len(n)?;
                let mut v = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    v.push(self.value(depth + 1)?);
                }
                Cbor::Array(v)
            }
            5 => {
                let n = len(n)?;
                let mut v = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    let k = self.value(depth + 1)?;
                    v.push((k, self.value(depth + 1)?));
                }
                Cbor::Map(v)
            }
            // A tag: its value follows.
            6 => self.value(depth + 1)?,
            _ => Cbor::Simple(n),
        })
    }
}

/// Absolute byte range of every tensor in a `.bpk` file, by tensor name
/// (`submodule1.conv2d1.weight`, ...).
pub fn bpk_spans(bpk: &[u8]) -> Result<HashMap<String, (usize, usize)>, String> {
    if bpk.len() < HEADER_SIZE || bpk[..4] != MAGIC {
        return Err("not a burnpack file".into());
    }
    let meta_len = u32::from_le_bytes(bpk[6..10].try_into().unwrap()) as usize;
    let meta_end = HEADER_SIZE
        .checked_add(meta_len)
        .filter(|&e| e <= bpk.len())
        .ok_or("burnpack: truncated metadata")?;
    let data = meta_end.div_ceil(TENSOR_ALIGNMENT) * TENSOR_ALIGNMENT;
    let meta = Reader {
        b: &bpk[HEADER_SIZE..meta_end],
        pos: 0,
    }
    .value(0)?;
    let Some(Cbor::Map(tensors)) = meta.get("tensors") else {
        return Err("burnpack: no tensor table".into());
    };
    let mut out = HashMap::new();
    for (name, desc) in tensors {
        let Cbor::Text(name) = name else {
            return Err("burnpack: tensor name is not text".into());
        };
        let Some(Cbor::Array(off)) = desc.get("data_offsets") else {
            return Err(format!("burnpack: {name}: no data_offsets"));
        };
        let [Cbor::Uint(a), Cbor::Uint(b)] = off.as_slice() else {
            return Err(format!("burnpack: {name}: bad data_offsets"));
        };
        let (a, b) = (data + *a as usize, data + *b as usize);
        if a > b || b > bpk.len() {
            return Err(format!("burnpack: {name}: data out of range"));
        }
        out.insert(name.clone(), (a, b));
    }
    Ok(out)
}

/// Field name → type name of every `pub struct` in a generated source, so a
/// submodule's `impl` block can be mapped to its path in the parameter tree.
fn struct_fields(src: &str) -> HashMap<String, Vec<(String, String)>> {
    let mut out: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut cur: Option<String> = None;
    for line in src.lines() {
        if let Some(rest) = line.strip_prefix("pub struct ") {
            cur = rest
                .strip_suffix(" {")
                .map(|n| n.trim().to_string())
                .filter(|n| !n.contains(['<', ' ']));
            continue;
        }
        if line.starts_with('}') {
            cur = None;
            continue;
        }
        if let Some(name) = &cur {
            let l = line.trim().trim_end_matches(',');
            if let Some((f, t)) = l.split_once(": ")
                && !f.starts_with('#')
                && is_ident(f)
            {
                out.entry(name.clone())
                    .or_default()
                    .push((f.to_string(), t.trim().to_string()));
            }
        }
    }
    out
}

fn is_ident(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The parameter path prefix (`""` for the root, `"submodule3."`) of every
/// struct reachable from `Model`.
fn struct_paths(fields: &HashMap<String, Vec<(String, String)>>) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut todo = vec![("Model".to_string(), String::new())];
    while let Some((ty, path)) = todo.pop() {
        if out.contains_key(&ty) || out.len() > 10_000 {
            continue;
        }
        if let Some(fs) = fields.get(&ty) {
            for (f, t) in fs {
                if fields.contains_key(t) {
                    todo.push((t.clone(), format!("{path}{f}.")));
                }
            }
        }
        out.insert(ty, path);
    }
    out
}

/// Insert a `layer_done` call (see the module docs) after each leaf layer
/// call of the generated source `src`. `model` is the `OcrModel` variant
/// the calls name. Layers whose tensors do not form one contiguous run of
/// the `.bpk` are left alone (each call releases one byte range). Returns
/// the new source and the number of calls inserted.
pub fn release_after_layers(
    src: &str,
    spans: &HashMap<String, (usize, usize)>,
    model: &str,
) -> (String, usize) {
    let fields = struct_fields(src);
    let paths = struct_paths(&fields);
    // Tensors in file order, to check a layer's run is contiguous.
    let mut ordered: Vec<(&str, (usize, usize))> =
        spans.iter().map(|(n, &s)| (n.as_str(), s)).collect();
    ordered.sort_by_key(|&(n, s)| (s, n));
    let layer_range = |prefix: &str| -> Option<(usize, usize)> {
        let idx: Vec<usize> = ordered
            .iter()
            .enumerate()
            .filter(|(_, (n, _))| {
                n.strip_prefix(prefix)
                    .is_some_and(|rest| !rest.is_empty() && !rest.contains('.'))
            })
            .map(|(i, _)| i)
            .collect();
        let (&first, &last) = (idx.first()?, idx.last()?);
        (last - first + 1 == idx.len()).then(|| (ordered[first].1.0, ordered[last].1.1))
    };
    let mut out = String::with_capacity(src.len() + src.len() / 8);
    let mut path: Option<&String> = None;
    let mut n = 0;
    for line in src.split_inclusive('\n') {
        out.push_str(line);
        if let Some(rest) = line.strip_prefix("impl ") {
            path = rest
                .strip_suffix(" {\n")
                .or_else(|| rest.strip_suffix(" {\r\n"))
                .and_then(|t| paths.get(t.trim()));
            continue;
        }
        let Some(prefix) = path else { continue };
        let t = line.trim();
        let Some(call) = t
            .strip_prefix("let ")
            .and_then(|r| r.split_once(" = self."))
            .map(|(_, r)| r)
        else {
            continue;
        };
        let Some((field, _)) = call.split_once(".forward(") else {
            continue;
        };
        if !is_ident(field) || !t.ends_with(");") {
            continue;
        }
        if let Some((a, b)) = layer_range(&format!("{prefix}{field}.")) {
            let indent = &line[..line.len() - line.trim_start().len()];
            out.push_str(&format!(
                "{indent}crate::neural::layer_done(crate::neural::OcrModel::{model}, {a}, {b});\n"
            ));
            n += 1;
        }
    }
    (out, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CBOR encoding helpers (definite lengths, as ciborium writes them).
    fn head(major: u8, n: u64, out: &mut Vec<u8>) {
        let m = major << 5;
        match n {
            0..=23 => out.push(m | n as u8),
            24..=0xff => out.extend([m | 24, n as u8]),
            0x100..=0xffff => {
                out.push(m | 25);
                out.extend((n as u16).to_be_bytes());
            }
            _ => {
                out.push(m | 26);
                out.extend((n as u32).to_be_bytes());
            }
        }
    }
    fn text(s: &str, out: &mut Vec<u8>) {
        head(3, s.len() as u64, out);
        out.extend(s.as_bytes());
    }

    /// A burnpack file holding `tensors` (name, data start, data end).
    fn bpk(tensors: &[(&str, u64, u64)]) -> Vec<u8> {
        let mut m = Vec::new();
        head(5, 2, &mut m);
        text("tensors", &mut m);
        head(5, tensors.len() as u64, &mut m);
        for &(name, a, b) in tensors {
            text(name, &mut m);
            head(5, 3, &mut m);
            text("dtype", &mut m);
            text("F32", &mut m);
            text("shape", &mut m);
            head(4, 1, &mut m);
            head(0, (b - a) / 4, &mut m);
            text("data_offsets", &mut m);
            head(4, 2, &mut m);
            head(0, a, &mut m);
            head(0, b, &mut m);
        }
        text("metadata", &mut m);
        head(5, 0, &mut m);
        let mut f = b"NRUB".to_vec();
        f.extend(1u16.to_le_bytes());
        f.extend((m.len() as u32).to_le_bytes());
        f.extend(&m);
        let data = f.len().div_ceil(256) * 256;
        let end = tensors.iter().map(|t| t.2).max().unwrap_or(0) as usize;
        f.resize(data + end, 0);
        f
    }

    #[test]
    fn spans_are_absolute() {
        let f = bpk(&[("a.weight", 0, 400), ("a.bias", 512, 520)]);
        let data = f.len() - 520;
        assert_eq!(data % 256, 0);
        let s = bpk_spans(&f).unwrap();
        assert_eq!(s["a.weight"], (data, data + 400));
        assert_eq!(s["a.bias"], (data + 512, data + 520));
    }

    #[test]
    fn rejects_garbage() {
        assert!(bpk_spans(b"nope").is_err());
        let mut f = bpk(&[("a.weight", 0, 4)]);
        f.truncate(12);
        assert!(bpk_spans(&f).is_err());
        // Offsets past the end of the file.
        let mut f = bpk(&[("a.weight", 0, 4096)]);
        f.truncate(f.len() - 1);
        assert!(bpk_spans(&f).is_err());
    }

    const SRC: &str = "\
pub struct Submodule1 {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule1 {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let conv2d1_out1 = self.conv2d1.forward(x);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        conv2d2_out1
    }
}
pub struct Model {
    submodule1: Submodule1,
    linear1: Linear,
}
impl Model {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<3> {
        let submodule1_out1 = self.submodule1.forward(x);
        let linear1_out1 = self.linear1.forward(submodule1_out1);
        linear1_out1
    }
}
";

    #[test]
    fn inserts_after_leaf_layers_only() {
        let spans: HashMap<String, (usize, usize)> = [
            ("submodule1.conv2d1.bias", (1000, 1010)),
            ("submodule1.conv2d1.weight", (1024, 2048)),
            ("submodule1.conv2d2.bias", (2048, 2060)),
            ("submodule1.conv2d2.weight", (2304, 4096)),
            ("linear1.bias", (4096, 4100)),
            ("linear1.weight", (4352, 8000)),
        ]
        .into_iter()
        .map(|(n, s)| (n.to_string(), s))
        .collect();
        let (out, n) = release_after_layers(SRC, &spans, "Server");
        assert_eq!(n, 3);
        assert!(out.contains(
            "        let conv2d1_out1 = self.conv2d1.forward(x);\n        \
             crate::neural::layer_done(crate::neural::OcrModel::Server, 1000, 2048);\n"
        ));
        assert!(out.contains("OcrModel::Server, 2048, 4096);"));
        assert!(out.contains("OcrModel::Server, 4096, 8000);"));
        // The submodule call is not a leaf layer.
        assert!(!out.contains("self.submodule1.forward(x);\n        crate::neural::layer_done"));
    }

    #[test]
    fn skips_layers_split_across_the_file() {
        let spans: HashMap<String, (usize, usize)> = [
            ("submodule1.conv2d1.bias", (1000, 1010)),
            ("submodule1.conv2d2.weight", (1024, 2048)),
            ("submodule1.conv2d1.weight", (2304, 4096)),
        ]
        .into_iter()
        .map(|(n, s)| (n.to_string(), s))
        .collect();
        let (out, n) = release_after_layers(SRC, &spans, "Det");
        assert_eq!(n, 1);
        assert!(out.contains("OcrModel::Det, 1024, 2048);"));
    }
}
