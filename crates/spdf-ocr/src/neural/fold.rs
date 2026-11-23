//! Build-time graph simplification for the OCR models: fold the linear
//! element-wise ops that surround each convolution into the convolution's
//! own weights and bias.
//!
//! PaddleOCR's networks wrap every convolution in several full-tensor
//! passes: `BatchNormalization`, a separate per-channel bias `Add`, and
//! PP-LCNetV3's "learnable affine blocks" (a scalar `Mul` + `Add` before and
//! after each activation). Each pass streams the whole activation through
//! memory, and on a CPU those passes cost more than the convolutions. All
//! of them are affine, so they fold exactly:
//!
//! - after a conv, `y = s·conv(x) + c` is a conv with weights `s·W` and bias
//!   `s·b + c` (any padding);
//! - before a conv *without padding*, `conv(a·x + c)` is a conv with weights
//!   `a·W` and bias `b + c·ΣW` (with zero padding the border taps would see
//!   `0` instead of `c`, so padded convs are left alone).
//!
//! This file has no dependencies: the build script converts ONNX protobufs
//! to and from [`Graph`], and the crate's unit tests include it directly.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// A float constant (ONNX initializer).
#[derive(Clone, Debug, PartialEq)]
pub struct Const {
    pub dims: Vec<i64>,
    pub data: Vec<f32>,
}

/// One graph node, with just what the folds need to know.
#[derive(Clone, Debug)]
pub struct Node {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// `Conv`: true when every pad is zero (and `auto_pad` is unset/VALID).
    pub zero_pad: bool,
    /// `BatchNormalization` epsilon.
    pub epsilon: f32,
    /// `Conv` group count.
    pub group: i64,
    /// `Softmax` axis (may be negative).
    pub axis: i64,
    /// Index of the source node whose attributes this node carries (`None`
    /// for nodes the passes created that need no attributes).
    pub attrs_from: Option<usize>,
    /// Set once the node has been folded away.
    pub removed: bool,
}

impl Node {
    pub fn new(op: &str, inputs: &[&str], outputs: &[&str]) -> Self {
        Self {
            op: op.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            zero_pad: false,
            epsilon: 1e-5,
            group: 1,
            axis: -1,
            attrs_from: None,
            removed: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub consts: HashMap<String, Const>,
    /// Integer (`INT64`) constants the passes created.
    pub int_consts: HashMap<String, Vec<i64>>,
    /// Names the graph exposes as outputs; never folded away.
    pub outputs: HashSet<String>,
    /// Rank of each graph output, where the model declares it.
    pub output_ranks: HashMap<String, usize>,
}

/// A dense layer detached from a graph output by
/// [`Graph::split_output_linear`]: `y = x · w + b`, `w` row-major `[k, n]`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearHead {
    pub k: usize,
    pub n: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

/// Fold counts, for the build log.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct FoldStats {
    pub identity: usize,
    pub after_conv: usize,
    pub before_conv: usize,
    pub split_conv: usize,
}

/// Convolutions whose im2col matrix would have more rows than this
/// (`in_channels × kh × kw`) are split along their input channels.
pub const SPLIT_MAX_ROWS: usize = 256;

impl Graph {
    /// Run every fold to a fixed point.
    pub fn optimize(&mut self) -> FoldStats {
        let mut st = FoldStats {
            identity: self.drop_identities(),
            ..FoldStats::default()
        };
        loop {
            let a = self.fold_after_conv();
            let b = self.fold_before_conv();
            st.after_conv += a;
            st.before_conv += b;
            if a + b == 0 {
                break;
            }
        }
        st.split_conv = self.split_wide_convs(SPLIT_MAX_ROWS, 2);
        st
    }

    /// Split every dense convolution with at least `min_taps` kernel taps
    /// whose im2col matrix would have more than `max_rows` rows into
    /// convolutions over slices of its input channels, summed:
    /// `conv(x, W) = Σ conv(x[c₀..c₁], W[:, c₀..c₁])`. Exact for any
    /// padding, stride or input size, and it bounds the lowering buffer
    /// (`rows × output pixels`), which for a 3×3 conv over 96 channels at a
    /// quarter of the page size is >150 MB. (1×1 convs need no lowering.)
    fn split_wide_convs(&mut self, max_rows: usize, min_taps: usize) -> usize {
        let mut split = 0;
        let old = std::mem::take(&mut self.nodes);
        for node in old {
            let pieces = self.split_conv(&node, max_rows, min_taps);
            if pieces.is_empty() {
                self.nodes.push(node);
            } else {
                split += 1;
                self.nodes.extend(pieces);
            }
        }
        split
    }

    fn split_conv(&mut self, conv: &Node, max_rows: usize, min_taps: usize) -> Vec<Node> {
        if conv.removed || conv.op != "Conv" || conv.group != 1 || conv.inputs.len() < 2 {
            return Vec::new();
        }
        let Some(w) = self
            .consts
            .get(&conv.inputs[1])
            .filter(|w| w.dims.len() == 4)
        else {
            return Vec::new();
        };
        let (cout, cin, kh, kw) = (
            w.dims[0] as usize,
            w.dims[1] as usize,
            w.dims[2] as usize,
            w.dims[3] as usize,
        );
        let k = kh * kw;
        if k < min_taps || cin * k <= max_rows {
            return Vec::new();
        }
        let parts = (cin * k).div_ceil(max_rows);
        let step = cin.div_ceil(parts);
        if step >= cin {
            return Vec::new(); // a single channel: nothing to split
        }
        let w = w.clone();
        let base = format!("{}__split", conv.outputs[0]);
        let mut out = Vec::new();
        let mut acc: Option<String> = None;
        let mut c0 = 0;
        let mut p = 0;
        while c0 < cin {
            let c1 = (c0 + step).min(cin);
            let tag = format!("{base}{p}");
            // Input channels c0..c1 (a strided view, no copy, on Burn).
            let (starts, ends, axes) = (format!("{tag}_s"), format!("{tag}_e"), format!("{tag}_a"));
            self.int_consts.insert(starts.clone(), vec![c0 as i64]);
            self.int_consts.insert(ends.clone(), vec![c1 as i64]);
            self.int_consts.insert(axes.clone(), vec![1]);
            let xs = format!("{tag}_x");
            out.push(Node::new(
                "Slice",
                &[&conv.inputs[0], &starts, &ends, &axes],
                &[&xs],
            ));
            // Weights W[:, c0..c1].
            let mut data = Vec::with_capacity(cout * (c1 - c0) * k);
            for o in 0..cout {
                data.extend_from_slice(&w.data[(o * cin + c0) * k..(o * cin + c1) * k]);
            }
            let wn = format!("{tag}_w");
            self.consts.insert(
                wn.clone(),
                Const {
                    dims: vec![cout as i64, (c1 - c0) as i64, kh as i64, kw as i64],
                    data,
                },
            );
            let mut inputs: Vec<&str> = vec![&xs, &wn];
            if p == 0 {
                if let Some(b) = conv.inputs.get(2).filter(|b| !b.is_empty()) {
                    inputs.push(b);
                }
            }
            let y = format!("{tag}_y");
            let mut piece = Node::new("Conv", &inputs, &[&y]);
            piece.attrs_from = conv.attrs_from;
            piece.zero_pad = conv.zero_pad;
            out.push(piece);
            acc = Some(match acc {
                None => y,
                Some(prev) => {
                    let sum = if c1 == cin {
                        conv.outputs[0].clone()
                    } else {
                        format!("{tag}_sum")
                    };
                    out.push(Node::new("Add", &[&prev, &y], &[&sum]));
                    sum
                }
            });
            c0 = c1;
            p += 1;
        }
        out
    }

    fn consumers(&self, name: &str) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&i| !self.nodes[i].removed && self.nodes[i].inputs.iter().any(|s| s == name))
            .collect()
    }

    fn producer(&self, name: &str) -> Option<usize> {
        (0..self.nodes.len())
            .find(|&i| !self.nodes[i].removed && self.nodes[i].outputs.iter().any(|s| s == name))
    }

    /// The one live consumer of `name`, if it has exactly one and is not a
    /// graph output.
    fn sole_consumer(&self, name: &str) -> Option<usize> {
        if self.outputs.contains(name) {
            return None;
        }
        match self.consumers(name).as_slice() {
            [i] => Some(*i),
            _ => None,
        }
    }

    fn rename_uses(&mut self, from: &str, to: &str) {
        for n in self.nodes.iter_mut().filter(|n| !n.removed) {
            for s in n.inputs.iter_mut().filter(|s| *s == from) {
                *s = to.to_string();
            }
        }
    }

    fn drop_identities(&mut self) -> usize {
        let mut n = 0;
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            if node.removed || node.op != "Identity" || node.inputs.len() != 1 {
                continue;
            }
            let (src, dst) = (node.inputs[0].clone(), node.outputs[0].clone());
            if self.outputs.contains(&dst) {
                continue;
            }
            self.nodes[i].removed = true;
            self.rename_uses(&dst, &src);
            n += 1;
        }
        n
    }

    /// A constant that is a single scalar or one value per channel (shape
    /// `[C]`, `[C,1,1]` or `[1,C,1,1]`), expanded to `cout` values.
    fn channel_const(&self, name: &str, cout: usize, allow_flat: bool) -> Option<Vec<f32>> {
        let c = self.consts.get(name)?;
        if c.data.len() == 1 {
            return Some(vec![c.data[0]; cout]);
        }
        if c.data.len() != cout {
            return None;
        }
        let d: Vec<i64> = c.dims.clone();
        let per_channel = matches!(d.as_slice(), [1, k, 1, 1] if *k as usize == cout)
            || matches!(d.as_slice(), [k, 1, 1] if *k as usize == cout)
            || (allow_flat && matches!(d.as_slice(), [k] if *k as usize == cout));
        per_channel.then(|| c.data.clone())
    }

    /// The non-constant operand of a binary op whose other operand is the
    /// constant at `const_name`.
    fn split_binary<'a>(&self, n: &'a Node, x: &str) -> Option<&'a str> {
        if n.inputs.len() != 2 {
            return None;
        }
        if n.inputs[0] == x {
            Some(&n.inputs[1])
        } else if n.inputs[1] == x {
            Some(&n.inputs[0])
        } else {
            None
        }
    }

    /// Fresh weight/bias constants for conv `ci` (they may be shared with
    /// another conv in theory, so never edit them in place).
    fn conv_params(&mut self, ci: usize) -> Option<(String, String, usize, usize)> {
        let conv = &self.nodes[ci];
        let w = self.consts.get(conv.inputs.get(1)?)?.clone();
        if w.dims.len() != 4 {
            return None;
        }
        let cout = w.dims[0] as usize;
        let per_out = w.data.len() / cout.max(1);
        let b = match conv.inputs.get(2).filter(|s| !s.is_empty()) {
            Some(name) => self.consts.get(name)?.data.clone(),
            None => vec![0.0; cout],
        };
        if b.len() != cout {
            return None;
        }
        let tag = format!("{}__fold{ci}", conv.outputs[0]);
        let (wn, bn) = (format!("{tag}_w"), format!("{tag}_b"));
        self.consts.insert(wn.clone(), w);
        self.consts.insert(
            bn.clone(),
            Const {
                dims: vec![cout as i64],
                data: b,
            },
        );
        let conv = &mut self.nodes[ci];
        conv.inputs.truncate(1);
        conv.inputs.push(wn.clone());
        conv.inputs.push(bn.clone());
        Some((wn, bn, cout, per_out))
    }

    fn fold_after_conv(&mut self) -> usize {
        let mut folded = 0;
        for ci in 0..self.nodes.len() {
            loop {
                let conv = &self.nodes[ci];
                if conv.removed || conv.op != "Conv" || conv.inputs.len() < 2 {
                    break;
                }
                let Some(cout) = self
                    .consts
                    .get(&conv.inputs[1])
                    .filter(|w| w.dims.len() == 4)
                    .map(|w| w.dims[0] as usize)
                else {
                    break;
                };
                let out = conv.outputs[0].clone();
                let Some(ni) = self.sole_consumer(&out) else {
                    break;
                };
                let next = self.nodes[ni].clone();
                // (scale, shift) per output channel: y = scale·x + shift.
                let affine = match next.op.as_str() {
                    "Add" => self
                        .split_binary(&next, &out)
                        .and_then(|k| self.channel_const(k, cout, false))
                        .map(|c| (vec![1.0; cout], c)),
                    "Mul" => self
                        .split_binary(&next, &out)
                        .and_then(|k| self.channel_const(k, cout, false))
                        .map(|s| (s, vec![0.0; cout])),
                    "BatchNormalization" if next.inputs.len() == 5 && next.inputs[0] == out => {
                        let get = |i: usize| self.channel_const(&next.inputs[i], cout, true);
                        match (get(1), get(2), get(3), get(4)) {
                            (Some(g), Some(beta), Some(mean), Some(var)) => {
                                let scale: Vec<f32> = (0..cout)
                                    .map(|k| g[k] / (var[k] + next.epsilon).sqrt())
                                    .collect();
                                let shift =
                                    (0..cout).map(|k| beta[k] - mean[k] * scale[k]).collect();
                                Some((scale, shift))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };
                let Some((scale, shift)) = affine else {
                    break;
                };
                let Some((wn, bn, cout, per_out)) = self.conv_params(ci) else {
                    break;
                };
                let w = self.consts.get_mut(&wn).expect("just inserted");
                for (k, chunk) in w.data.chunks_mut(per_out).enumerate().take(cout) {
                    chunk.iter_mut().for_each(|v| *v *= scale[k]);
                }
                let b = self.consts.get_mut(&bn).expect("just inserted");
                for k in 0..cout {
                    b.data[k] = b.data[k] * scale[k] + shift[k];
                }
                // The conv now produces the folded node's output.
                self.nodes[ci].outputs[0] = next.outputs[0].clone();
                self.nodes[ni].removed = true;
                folded += 1;
            }
        }
        folded
    }

    fn fold_before_conv(&mut self) -> usize {
        let mut folded = 0;
        for ci in 0..self.nodes.len() {
            loop {
                let conv = &self.nodes[ci];
                if conv.removed || conv.op != "Conv" || !conv.zero_pad || conv.inputs.len() < 2 {
                    break;
                }
                let Some(cin_g) = self
                    .consts
                    .get(&conv.inputs[1])
                    .filter(|w| w.dims.len() == 4)
                    .map(|w| w.dims[1] as usize)
                else {
                    break;
                };
                let x = conv.inputs[0].clone();
                let Some(pi) = self.producer(&x) else {
                    break;
                };
                if self.sole_consumer(&x) != Some(ci) {
                    break;
                }
                let prev = self.nodes[pi].clone();
                if prev.outputs.len() != 1 {
                    break;
                }
                // Only scalars: a per-channel input affine would have to be
                // matched against the grouping of the weights.
                let scalar = |k: &str| {
                    self.consts
                        .get(k)
                        .filter(|c| c.data.len() == 1)
                        .map(|c| c.data[0])
                };
                let (op, src, v) = match prev.op.as_str() {
                    "Add" | "Mul" => {
                        let (k, src) = if scalar(&prev.inputs[0]).is_some() {
                            (&prev.inputs[0], &prev.inputs[1])
                        } else {
                            (&prev.inputs[1], &prev.inputs[0])
                        };
                        match scalar(k) {
                            Some(v) if prev.inputs.len() == 2 => (prev.op.clone(), src.clone(), v),
                            _ => break,
                        }
                    }
                    _ => break,
                };
                let Some((wn, bn, cout, per_out)) = self.conv_params(ci) else {
                    break;
                };
                debug_assert_eq!(per_out % cin_g.max(1), 0);
                if op == "Mul" {
                    let w = self.consts.get_mut(&wn).expect("just inserted");
                    w.data.iter_mut().for_each(|w| *w *= v);
                } else {
                    let sums: Vec<f32> = self.consts[&wn]
                        .data
                        .chunks(per_out)
                        .take(cout)
                        .map(|c| c.iter().sum())
                        .collect();
                    let b = self.consts.get_mut(&bn).expect("just inserted");
                    for k in 0..cout {
                        b.data[k] += v * sums[k];
                    }
                }
                self.nodes[ci].inputs[0] = src;
                self.nodes[pi].removed = true;
                folded += 1;
            }
        }
        folded
    }

    /// Drop a `Softmax` over the last axis that produces a graph output, so
    /// the graph returns the logits under the same output name. For a CTC
    /// recogniser the softmax is the costliest op of the head (it writes a
    /// second steps × classes tensor, ~18k classes per step) while greedy
    /// decoding only needs each step's arg-max and its probability, which
    /// the caller can derive from the logits. Returns the number stripped.
    pub fn strip_output_softmax(&mut self) -> usize {
        let mut stripped = 0;
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.removed || n.op != "Softmax" || n.inputs.len() != 1 {
                continue;
            }
            let (src, out) = (n.inputs[0].clone(), n.outputs[0].clone());
            if !self.outputs.contains(&out) || self.outputs.contains(&src) {
                continue;
            }
            let last = match self.output_ranks.get(&out) {
                Some(&rank) => n.axis == -1 || n.axis == rank as i64 - 1,
                None => n.axis == -1,
            };
            let users = self
                .nodes
                .iter()
                .filter(|m| !m.removed && m.inputs.contains(&src))
                .count();
            let producer = self
                .nodes
                .iter()
                .position(|m| !m.removed && m.outputs.contains(&src));
            let (true, 1, Some(p)) = (last, users, producer) else {
                continue;
            };
            self.nodes[i].removed = true;
            for o in &mut self.nodes[p].outputs {
                if *o == src {
                    *o = out.clone();
                }
            }
            stripped += 1;
        }
        stripped
    }

    /// Detach a graph output's final dense layer, `out = x · W + b` (a
    /// `MatMul` by a constant `[K, N]` matrix then an `Add` of a constant
    /// `[N]` bias), so the graph returns `x` (the `K`-wide features) under
    /// the old output name and the caller applies the layer itself. A CTC
    /// recogniser's classifier maps 120 features to ~18k classes per step:
    /// materialising those logits is the bulk of the recogniser's memory,
    /// while greedy decoding only needs each step's arg-max and softmax
    /// mass, which the caller can stream. Only applies when the graph has
    /// exactly one output and the layer's intermediates have no other use.
    pub fn split_output_linear(&mut self) -> Option<LinearHead> {
        if self.outputs.len() != 1 {
            return None;
        }
        let out = self.outputs.iter().next()?.clone();
        let add = self.producer(&out)?;
        let a = &self.nodes[add];
        if a.op != "Add" || a.inputs.len() != 2 {
            return None;
        }
        let (m, bias) = match (self.consts.get(&a.inputs[0]), self.consts.get(&a.inputs[1])) {
            (None, Some(b)) => (a.inputs[0].clone(), b.clone()),
            (Some(b), None) => (a.inputs[1].clone(), b.clone()),
            _ => return None,
        };
        let mm = self.producer(&m)?;
        if self.sole_consumer(&m) != Some(add) {
            return None;
        }
        let mmn = &self.nodes[mm];
        if mmn.op != "MatMul" || mmn.inputs.len() != 2 {
            return None;
        }
        let x = mmn.inputs[0].clone();
        let w = self.consts.get(&mmn.inputs[1])?.clone();
        let [k, n] = w.dims[..] else {
            return None;
        };
        let (k, n) = (k as usize, n as usize);
        if bias.data.len() != n || bias.dims.iter().product::<i64>() as usize != n {
            return None;
        }
        // `x` must be computed by the graph (not a constant or the graph
        // input) and feed nothing but this layer.
        let xp = self.producer(&x)?;
        if self.sole_consumer(&x) != Some(mm) {
            return None;
        }
        self.nodes[add].removed = true;
        self.nodes[mm].removed = true;
        for o in &mut self.nodes[xp].outputs {
            if *o == x {
                *o = out.clone();
            }
        }
        Some(LinearHead {
            k,
            n,
            w: w.data,
            b: bias.data,
        })
    }

    /// Names of constants still referenced by a live node.
    pub fn live_consts(&self) -> HashSet<String> {
        self.nodes
            .iter()
            .filter(|n| !n.removed)
            .flat_map(|n| n.inputs.iter())
            .filter(|s| self.consts.contains_key(*s))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(dims: &[i64], data: &[f32]) -> Const {
        Const {
            dims: dims.to_vec(),
            data: data.to_vec(),
        }
    }

    /// Reference evaluation of a tiny graph: 1x1 convs and element-wise ops
    /// on a `[C, N]` activation (N spatial positions).
    fn eval(g: &Graph, input: &[Vec<f32>], out: &str) -> Vec<Vec<f32>> {
        let mut vals: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        vals.insert("x".into(), input.to_vec());
        let bcast = |c: &Const, ch: usize| {
            if c.data.len() == 1 {
                c.data[0]
            } else {
                c.data[ch]
            }
        };
        for n in g.nodes.iter().filter(|n| !n.removed) {
            let a = |i: usize| vals.get(&n.inputs[i]).cloned();
            let v: Vec<Vec<f32>> = match n.op.as_str() {
                "Identity" => a(0).unwrap(),
                "Slice" => {
                    let x = a(0).unwrap();
                    let (s0, e0) = (
                        g.int_consts[&n.inputs[1]][0] as usize,
                        g.int_consts[&n.inputs[2]][0] as usize,
                    );
                    assert_eq!(g.int_consts[&n.inputs[3]], vec![1]);
                    x[s0..e0].to_vec()
                }
                "Add" if a(0).is_some() && a(1).is_some() => {
                    let (x, y) = (a(0).unwrap(), a(1).unwrap());
                    x.iter()
                        .zip(&y)
                        .map(|(r, q)| r.iter().zip(q).map(|(u, v)| u + v).collect())
                        .collect()
                }
                "Relu" => a(0)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.into_iter().map(|v| v.max(0.0)).collect())
                    .collect(),
                "Add" | "Mul" => {
                    let (x, c) = match a(0) {
                        Some(x) => (x, &g.consts[&n.inputs[1]]),
                        None => (a(1).unwrap(), &g.consts[&n.inputs[0]]),
                    };
                    x.into_iter()
                        .enumerate()
                        .map(|(ch, r)| {
                            r.into_iter()
                                .map(|v| {
                                    if n.op == "Add" {
                                        v + bcast(c, ch)
                                    } else {
                                        v * bcast(c, ch)
                                    }
                                })
                                .collect()
                        })
                        .collect()
                }
                "BatchNormalization" => {
                    let x = a(0).unwrap();
                    let p = |i: usize| &g.consts[&n.inputs[i]].data;
                    x.into_iter()
                        .enumerate()
                        .map(|(ch, r)| {
                            r.into_iter()
                                .map(|v| {
                                    (v - p(3)[ch]) / (p(4)[ch] + n.epsilon).sqrt() * p(1)[ch]
                                        + p(2)[ch]
                                })
                                .collect()
                        })
                        .collect()
                }
                "Conv" => {
                    let x = a(0).unwrap();
                    let w = &g.consts[&n.inputs[1]];
                    let (cout, cin) = (w.dims[0] as usize, w.dims[1] as usize);
                    let b = n.inputs.get(2).map(|s| g.consts[s].data.clone());
                    (0..cout)
                        .map(|o| {
                            (0..x[0].len())
                                .map(|p| {
                                    let s: f32 =
                                        (0..cin).map(|i| w.data[o * cin + i] * x[i][p]).sum();
                                    s + b.as_ref().map_or(0.0, |b| b[o])
                                })
                                .collect()
                        })
                        .collect()
                }
                op => panic!("eval: {op}"),
            };
            vals.insert(n.outputs[0].clone(), v);
        }
        vals.remove(out).unwrap()
    }

    fn conv(inp: &str, w: &str, out: &str) -> Node {
        let mut n = Node::new("Conv", &[inp, w], &[out]);
        n.zero_pad = true;
        n
    }

    /// PP-LCNetV3-style block: conv → bias add → affine → relu → affine →
    /// conv, plus a BatchNorm and Identity hops.
    fn lcnet_block() -> Graph {
        let mut g = Graph::default();
        g.consts
            .insert("w1".into(), k(&[2, 2, 1, 1], &[0.5, -1.0, 2.0, 0.25]));
        g.consts
            .insert("bias".into(), k(&[1, 2, 1, 1], &[0.1, -0.2]));
        g.consts.insert("s1".into(), k(&[1], &[1.5]));
        g.consts.insert("c1".into(), k(&[1], &[-0.3]));
        g.consts.insert("s2".into(), k(&[1], &[0.7]));
        g.consts.insert("c2".into(), k(&[1], &[0.4]));
        g.consts
            .insert("w2".into(), k(&[2, 2, 1, 1], &[1.0, 2.0, -0.5, 0.3]));
        g.consts.insert("g".into(), k(&[2], &[1.2, 0.8]));
        g.consts.insert("be".into(), k(&[2], &[0.05, -0.1]));
        g.consts.insert("mu".into(), k(&[2], &[0.3, -0.4]));
        g.consts.insert("var".into(), k(&[2], &[0.9, 1.6]));
        g.nodes = vec![
            conv("x", "w1", "c1o"),
            Node::new("Identity", &["c1o"], &["c1i"]),
            Node::new("Add", &["c1i", "bias"], &["a1"]),
            Node::new("Mul", &["s1", "a1"], &["m1"]),
            Node::new("Add", &["m1", "c1"], &["a2"]),
            Node::new("Relu", &["a2"], &["r"]),
            Node::new("Mul", &["s2", "r"], &["m2"]),
            Node::new("Add", &["m2", "c2"], &["a3"]),
            conv("a3", "w2", "c2o"),
            Node::new(
                "BatchNormalization",
                &["c2o", "g", "be", "mu", "var"],
                &["y"],
            ),
        ];
        g.outputs.insert("y".into());
        g
    }

    #[test]
    fn folds_lcnet_block_exactly() {
        let x = vec![vec![1.0, -2.0, 0.5, 3.0], vec![0.25, 4.0, -1.0, 0.0]];
        let g0 = lcnet_block();
        let want = eval(&g0, &x, "y");
        let mut g = g0.clone();
        let st = g.optimize();
        assert_eq!(st.identity, 1);
        assert_eq!(st.after_conv, 4, "{st:?}"); // bias, mul, add, batchnorm
        assert_eq!(st.before_conv, 2, "{st:?}"); // mul, add before conv 2
        let live: Vec<&str> = g
            .nodes
            .iter()
            .filter(|n| !n.removed)
            .map(|n| n.op.as_str())
            .collect();
        assert_eq!(live, ["Conv", "Relu", "Conv"]);
        let got = eval(&g, &x, "y");
        for (a, b) in want.iter().flatten().zip(got.iter().flatten()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        // The original constants are untouched (folds write fresh ones).
        assert_eq!(g.consts["w1"], g0.consts["w1"]);
    }

    fn softmax_head(axis: i64) -> Graph {
        let mut softmax = Node::new("Softmax", &["logits"], &["probs"]);
        softmax.axis = axis;
        let mut g = Graph {
            nodes: vec![Node::new("MatMul", &["x", "w"], &["logits"]), softmax],
            ..Graph::default()
        };
        g.outputs.insert("probs".into());
        g.output_ranks.insert("probs".into(), 3);
        g
    }

    #[test]
    fn strips_last_axis_output_softmax() {
        for axis in [-1, 2] {
            let mut g = softmax_head(axis);
            assert_eq!(g.strip_output_softmax(), 1);
            let live: Vec<&Node> = g.nodes.iter().filter(|n| !n.removed).collect();
            assert_eq!(live.len(), 1);
            // The logits now leave the graph under the old output name.
            assert_eq!(live[0].outputs, ["probs"]);
        }
    }

    /// `x → Transpose → MatMul(w) → Add(b)` with `y` as the graph output.
    fn linear_head(bias_first: bool) -> Graph {
        let mut g = Graph::default();
        g.consts
            .insert("w".into(), k(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        g.consts.insert("b".into(), k(&[3], &[0.1, 0.2, 0.3]));
        let add = if bias_first {
            Node::new("Add", &["b", "mm"], &["y"])
        } else {
            Node::new("Add", &["mm", "b"], &["y"])
        };
        g.nodes = vec![
            Node::new("Transpose", &["x"], &["feat"]),
            Node::new("MatMul", &["feat", "w"], &["mm"]),
            add,
        ];
        g.outputs.insert("y".into());
        g
    }

    #[test]
    fn splits_output_linear_layer() {
        for bias_first in [false, true] {
            let mut g = linear_head(bias_first);
            let head = g.split_output_linear().expect("split");
            assert_eq!((head.k, head.n), (2, 3));
            assert_eq!(head.w, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            assert_eq!(head.b, vec![0.1, 0.2, 0.3]);
            let live: Vec<&Node> = g.nodes.iter().filter(|n| !n.removed).collect();
            assert_eq!(live.len(), 1);
            // The features now leave the graph under the old output name.
            assert_eq!(live[0].op, "Transpose");
            assert_eq!(live[0].outputs, ["y"]);
            assert!(!g.live_consts().contains("w"));
        }
    }

    #[test]
    fn keeps_linear_layer_that_cannot_be_detached() {
        // Matrix product feeds something else too.
        let mut g = linear_head(false);
        g.nodes.push(Node::new("Relu", &["mm"], &["z"]));
        assert!(g.split_output_linear().is_none());
        // Features used elsewhere.
        let mut g = linear_head(false);
        g.nodes.push(Node::new("Relu", &["feat"], &["z"]));
        assert!(g.split_output_linear().is_none());
        // Weight is not a constant.
        let mut g = linear_head(false);
        g.consts.remove("w");
        assert!(g.split_output_linear().is_none());
        // Bias length does not match the weight's columns.
        let mut g = linear_head(false);
        g.consts.insert("b".into(), k(&[2], &[0.1, 0.2]));
        assert!(g.split_output_linear().is_none());
        // Two graph outputs: ambiguous.
        let mut g = linear_head(false);
        g.outputs.insert("feat".into());
        assert!(g.split_output_linear().is_none());
        // The input feeds the product directly: nothing would be left.
        let mut g = linear_head(false);
        g.nodes.remove(0);
        g.nodes[0].inputs[0] = "x".into();
        assert!(g.split_output_linear().is_none());
    }

    #[test]
    fn keeps_softmax_that_is_not_a_last_axis_output() {
        // Softmax over another axis changes the arg-max: keep it.
        assert_eq!(softmax_head(1).strip_output_softmax(), 0);
        // Rank unknown: only an explicit -1 is known to be the last axis.
        let mut g = softmax_head(2);
        g.output_ranks.clear();
        assert_eq!(g.strip_output_softmax(), 0);
        // Inner softmax (attention): not a graph output.
        let mut g = softmax_head(-1);
        g.nodes.push(Node::new("Relu", &["probs"], &["y"]));
        g.outputs = ["y".to_string()].into();
        assert_eq!(g.strip_output_softmax(), 0);
        // Logits also used elsewhere: the producer must keep its name.
        let mut g = softmax_head(-1);
        g.nodes.push(Node::new("Relu", &["logits"], &["z"]));
        assert_eq!(g.strip_output_softmax(), 0);
    }

    #[test]
    fn wide_conv_split_is_exact() {
        // The reference evaluator only does 1x1 convs, so split one with a
        // budget of 2 rows: 5 input channels -> slices of 2, 2 and 1.
        let mut g = Graph::default();
        let w: Vec<f32> = (0..15).map(|i| (i as f32 * 0.37).sin()).collect();
        g.consts.insert("w".into(), k(&[3, 5, 1, 1], &w));
        g.consts.insert("b".into(), k(&[3], &[0.5, -1.0, 2.0]));
        let mut c = Node::new("Conv", &["x", "w", "b"], &["y"]);
        c.attrs_from = Some(0);
        g.nodes = vec![c];
        g.outputs.insert("y".into());
        let x: Vec<Vec<f32>> = (0..5)
            .map(|c| (0..4).map(|p| (c * 4 + p) as f32 * 0.1 - 0.7).collect())
            .collect();
        let want = eval(&g, &x, "y");
        // 1x1 kernels are not split in production (no im2col to bound).
        assert_eq!(g.clone().split_wide_convs(2, 2), 0);
        assert_eq!(g.split_wide_convs(2, 1), 1);
        let ops: Vec<&str> = g.nodes.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(
            ops,
            [
                "Slice", "Conv", "Slice", "Conv", "Add", "Slice", "Conv", "Add"
            ]
        );
        assert_eq!(g.nodes.last().unwrap().outputs[0], "y");
        let convs: Vec<&Node> = g.nodes.iter().filter(|n| n.op == "Conv").collect();
        assert!(convs.iter().all(|n| n.attrs_from == Some(0)));
        // Only the first piece carries the bias.
        assert_eq!(convs[0].inputs.len(), 3);
        assert_eq!(convs[1].inputs.len(), 2);
        let got = eval(&g, &x, "y");
        for (a, b) in want.iter().flatten().zip(got.iter().flatten()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn wide_three_by_three_conv_is_split_into_bounded_pieces() {
        let mut g = Graph::default();
        g.consts
            .insert("w".into(), k(&[24, 96, 3, 3], &vec![0.01; 24 * 96 * 9]));
        g.nodes = vec![Node::new("Conv", &["x", "w"], &["y"])];
        assert_eq!(g.split_wide_convs(SPLIT_MAX_ROWS, 2), 1);
        let convs: Vec<&Node> = g.nodes.iter().filter(|n| n.op == "Conv").collect();
        // 864 rows -> 4 slices of 24 channels (216 rows each).
        assert_eq!(convs.len(), 4);
        for c in &convs {
            assert_eq!(g.consts[&c.inputs[1]].dims, vec![24, 24, 3, 3]);
        }
        // Grouped (depthwise) convs are left alone.
        let mut g = Graph::default();
        g.consts
            .insert("w".into(), k(&[96, 1, 3, 3], &vec![0.01; 96 * 9]));
        let mut c = Node::new("Conv", &["x", "w"], &["y"]);
        c.group = 96;
        g.nodes = vec![c];
        assert_eq!(g.split_wide_convs(SPLIT_MAX_ROWS, 2), 0);
    }

    #[test]
    fn padded_conv_keeps_input_affine() {
        let mut g = lcnet_block();
        g.nodes[8].zero_pad = false;
        let st = g.optimize();
        assert_eq!(st.before_conv, 0);
        let live = g.nodes.iter().filter(|n| !n.removed).count();
        assert_eq!(live, 5); // conv, relu, mul, add, conv (+BN folded)
    }

    #[test]
    fn shared_activation_is_not_folded() {
        let mut g = lcnet_block();
        // A second consumer of the conv output blocks the fold.
        g.nodes.push(Node::new("Relu", &["c1i"], &["side"]));
        g.outputs.insert("side".into());
        let st = g.optimize();
        let first_add = &g.nodes[2];
        assert!(!first_add.removed, "{st:?}");
    }

    #[test]
    fn graph_output_is_preserved() {
        let mut g = Graph::default();
        g.consts.insert("w".into(), k(&[1, 1, 1, 1], &[2.0]));
        g.consts.insert("c".into(), k(&[1], &[1.0]));
        g.nodes = vec![conv("x", "w", "co"), Node::new("Add", &["co", "c"], &["y"])];
        g.outputs.insert("co".into());
        g.outputs.insert("y".into());
        assert_eq!(g.optimize(), FoldStats::default());
    }
}
