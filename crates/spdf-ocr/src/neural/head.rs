//! Streaming CTC classifier head.
//!
//! A PP-OCRv5 recogniser ends in one dense layer mapping 120 features per
//! CTC step to ~18k classes. Evaluated as a tensor op, those logits (and
//! their copies: matrix product, bias add, host transfer) are the bulk of
//! the recogniser's memory: ~9 KB per input pixel column, so a long line
//! needs tens of MB. Greedy CTC decoding only needs, per step, the arg-max
//! class and its softmax probability. The build script therefore detaches
//! the layer from the graph (`fold.rs`, `split_output_linear`) and packs its
//! weights with [`pack`]; [`CtcHead::decode`] applies it tile by tile with
//! an online softmax (running maximum, rescaled sum), so no logits buffer
//! larger than one tile ever exists.
//!
//! No dependencies: the build script includes this file for [`pack`].

#![allow(dead_code)]

use std::borrow::Cow;

/// Classes per weight tile (two 8-lane vectors).
pub const TILE: usize = 16;
/// Steps evaluated together against one weight tile.
const MS: usize = 4;
/// Steps per pass over the weights: each tile is reused this many times
/// while it is hot in the L1 cache.
const SB: usize = 64;
/// Logits more than this below a step's running maximum add under
/// `e^-20` each to the softmax denominator and are skipped.
pub const SOFTMAX_CUTOFF: f32 = 20.0;
const MAGIC: u32 = u32::from_le_bytes(*b"CTCH");
const HEADER: usize = 16;

/// Serialise a dense layer `y = x · w + b` (`w` row-major `[k, n]`) in the
/// layout [`CtcHead::parse`] reads: a 16-byte header (magic, `k`, `n`,
/// [`TILE`]), the bias padded to whole tiles with `-inf` (padding classes
/// never win), then the weights tile by tile, each tile `[k][TILE]`.
pub fn pack(k: usize, n: usize, w: &[f32], b: &[f32]) -> Vec<u8> {
    assert_eq!(w.len(), k * n, "weight shape");
    assert_eq!(b.len(), n, "bias length");
    let np = n.div_ceil(TILE) * TILE;
    let mut out = Vec::with_capacity(HEADER + 4 * (np + k * np));
    for v in [MAGIC, k as u32, n as u32, TILE as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for c in 0..np {
        let v = b.get(c).copied().unwrap_or(f32::NEG_INFINITY);
        out.extend_from_slice(&v.to_le_bytes());
    }
    for t in 0..np / TILE {
        for kk in 0..k {
            for j in 0..TILE {
                let c = t * TILE + j;
                let v = if c < n { w[kk * n + c] } else { 0.0 };
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

/// A packed dense classifier, borrowed from (typically static) bytes when
/// their alignment allows, else decoded into owned memory.
pub struct CtcHead<'a> {
    k: usize,
    n: usize,
    bias: Cow<'a, [f32]>,
    w: Cow<'a, [f32]>,
}

impl<'a> CtcHead<'a> {
    /// Read the layout written by [`pack`].
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        let word = |i: usize| -> Result<usize, String> {
            let b = bytes
                .get(i * 4..i * 4 + 4)
                .ok_or("CTC head: truncated header")?;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        };
        if word(0)? as u32 != MAGIC {
            return Err("CTC head: bad magic".into());
        }
        let (k, n, tile) = (word(1)?, word(2)?, word(3)?);
        if tile != TILE || k == 0 || n == 0 {
            return Err(format!(
                "CTC head: unsupported shape k={k} n={n} tile={tile}"
            ));
        }
        let np = n.div_ceil(TILE) * TILE;
        let body = &bytes[HEADER..];
        if body.len() != 4 * (np + k * np) {
            return Err("CTC head: size does not match its header".into());
        }
        let (b, w) = body.split_at(4 * np);
        Ok(Self {
            k,
            n,
            bias: floats(b),
            w: floats(w),
        })
    }

    /// Input features per step.
    pub fn features(&self) -> usize {
        self.k
    }

    /// Output classes.
    pub fn classes(&self) -> usize {
        self.n
    }

    /// Apply the layer to `steps` feature rows (`feats` is `[steps][k]`)
    /// and return, per step, the arg-max class and its softmax probability
    /// `1 / Σ exp(l - max)` (terms below `max - SOFTMAX_CUTOFF` skipped).
    pub fn decode(&self, feats: &[f32], steps: usize) -> (Vec<usize>, Vec<f32>) {
        let k = self.k;
        assert!(feats.len() >= steps * k, "CTC head: short feature buffer");
        let padded = steps.div_ceil(MS) * MS;
        let mut f = feats[..steps * k].to_vec();
        f.resize(padded * k, 0.0);
        let mut st = vec![Online::new(); padded];
        let fast = simd_available();
        let mut acc = [[0f32; TILE]; MS];
        for s0 in (0..padded).step_by(SB) {
            let s1 = (s0 + SB).min(padded);
            for t in 0..self.bias.len() / TILE {
                let w = &self.w[t * k * TILE..(t + 1) * k * TILE];
                let b = &self.bias[t * TILE..(t + 1) * TILE];
                for s in (s0..s1).step_by(MS) {
                    tile(fast, &f[s * k..(s + MS) * k], k, w, b, &mut acc);
                    for (o, row) in st[s..s + MS].iter_mut().zip(&acc) {
                        o.update(t * TILE, row);
                    }
                }
            }
        }
        st.truncate(steps);
        st.iter()
            .map(|o| {
                let p = if o.sum > 0.0 { 1.0 / o.sum } else { 0.0 };
                (o.arg, p)
            })
            .unzip()
    }
}

/// Little-endian `f32`s, borrowed when aligned (zero-copy for weights
/// embedded in the binary), else copied.
fn floats(b: &[u8]) -> Cow<'_, [f32]> {
    let aligned = b.as_ptr().align_offset(std::mem::align_of::<f32>()) == 0;
    if aligned && cfg!(target_endian = "little") {
        // SAFETY: the pointer is aligned for f32, the length is a whole
        // number of f32s within `b`, every bit pattern is a valid f32, and
        // the result borrows `b`.
        Cow::Borrowed(unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len() / 4) })
    } else {
        Cow::Owned(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }
}

/// Running arg-max and softmax denominator of one step's logits.
#[derive(Clone, Copy, Debug)]
struct Online {
    max: f32,
    arg: usize,
    /// `Σ exp(l - max)` over the logits seen so far.
    sum: f32,
}

impl Online {
    const fn new() -> Self {
        Self {
            max: f32::NEG_INFINITY,
            arg: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn update(&mut self, base: usize, l: &[f32; TILE]) {
        let (mut tm, mut ti) = (f32::NEG_INFINITY, 0);
        for (j, &v) in l.iter().enumerate() {
            if v > tm {
                (tm, ti) = (v, j);
            }
        }
        if tm > self.max {
            // Rescale what was summed relative to the old maximum.
            self.sum = if self.max.is_finite() {
                self.sum * (self.max - tm).exp()
            } else {
                0.0
            };
            self.max = tm;
            self.arg = base + ti;
        }
        let floor = self.max - SOFTMAX_CUTOFF;
        if tm >= floor {
            for &v in l {
                if v >= floor {
                    self.sum += (v - self.max).exp();
                }
            }
        }
    }
}

fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `acc[i] = b + f[i] · w` for `MS` feature rows `f` (`[MS][k]`) against one
/// weight tile `w` (`[k][TILE]`).
#[inline]
fn tile(fast: bool, f: &[f32], k: usize, w: &[f32], b: &[f32], acc: &mut [[f32; TILE]; MS]) {
    assert!(f.len() >= MS * k && w.len() >= k * TILE && b.len() >= TILE);
    #[cfg(target_arch = "x86_64")]
    if fast {
        // SAFETY: the CPU supports AVX2 and FMA (checked by the caller via
        // `simd_available`), and the slice lengths were asserted above.
        unsafe { tile_avx2(f, k, w, b, acc) };
        return;
    }
    let _ = fast;
    tile_portable(f, k, w, b, acc);
}

#[inline(always)]
fn tile_portable(f: &[f32], k: usize, w: &[f32], b: &[f32], acc: &mut [[f32; TILE]; MS]) {
    for row in acc.iter_mut() {
        row.copy_from_slice(&b[..TILE]);
    }
    for kk in 0..k {
        let wr = &w[kk * TILE..kk * TILE + TILE];
        for (i, row) in acc.iter_mut().enumerate() {
            let x = f[i * k + kk];
            for (a, &wv) in row.iter_mut().zip(wr) {
                *a += x * wv;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_avx2(f: &[f32], k: usize, w: &[f32], b: &[f32], acc: &mut [[f32; TILE]; MS]) {
    use std::arch::x86_64::*;
    const _: () = assert!(TILE == 16 && MS == 4);
    // SAFETY: the caller guarantees AVX2 + FMA and the slice lengths
    // (`f`: MS·k, `w`: k·TILE, `b`: TILE); every access below stays inside.
    unsafe {
        let (fp, wp) = (f.as_ptr(), w.as_ptr());
        let b0 = _mm256_loadu_ps(b.as_ptr());
        let b1 = _mm256_loadu_ps(b.as_ptr().add(8));
        let (mut a00, mut a01, mut a10, mut a11) = (b0, b1, b0, b1);
        let (mut a20, mut a21, mut a30, mut a31) = (b0, b1, b0, b1);
        for kk in 0..k {
            let w0 = _mm256_loadu_ps(wp.add(kk * TILE));
            let w1 = _mm256_loadu_ps(wp.add(kk * TILE + 8));
            let x = _mm256_set1_ps(*fp.add(kk));
            a00 = _mm256_fmadd_ps(x, w0, a00);
            a01 = _mm256_fmadd_ps(x, w1, a01);
            let x = _mm256_set1_ps(*fp.add(k + kk));
            a10 = _mm256_fmadd_ps(x, w0, a10);
            a11 = _mm256_fmadd_ps(x, w1, a11);
            let x = _mm256_set1_ps(*fp.add(2 * k + kk));
            a20 = _mm256_fmadd_ps(x, w0, a20);
            a21 = _mm256_fmadd_ps(x, w1, a21);
            let x = _mm256_set1_ps(*fp.add(3 * k + kk));
            a30 = _mm256_fmadd_ps(x, w0, a30);
            a31 = _mm256_fmadd_ps(x, w1, a31);
        }
        for (row, (lo, hi)) in acc
            .iter_mut()
            .zip([(a00, a01), (a10, a11), (a20, a21), (a30, a31)])
        {
            _mm256_storeu_ps(row.as_mut_ptr(), lo);
            _mm256_storeu_ps(row.as_mut_ptr().add(8), hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values in [-scale, scale).
    fn noise(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Plain logits, arg-max and softmax of the maximum.
    fn reference(k: usize, n: usize, w: &[f32], b: &[f32], f: &[f32]) -> (Vec<usize>, Vec<f32>) {
        f.chunks(k)
            .map(|x| {
                let l: Vec<f32> = (0..n)
                    .map(|c| b[c] + (0..k).map(|i| x[i] * w[i * n + c]).sum::<f32>())
                    .collect();
                let (mut arg, mut max) = (0, f32::NEG_INFINITY);
                for (c, &v) in l.iter().enumerate() {
                    if v > max {
                        (arg, max) = (c, v);
                    }
                }
                let sum: f32 = l.iter().map(|v| (v - max).exp()).sum();
                (arg, 1.0 / sum)
            })
            .unzip()
    }

    #[test]
    fn decode_matches_dense_softmax() {
        // Odd sizes: classes not a whole number of tiles, steps not a whole
        // number of micro-tiles, more steps than one weight pass.
        for (k, n, steps) in [(7, 37, 5), (120, 200, 70), (3, 16, 1), (5, 1, 9)] {
            let w = noise(k * n, 1, 1.0);
            let b = noise(n, 2, 0.5);
            let f = noise(steps * k, 3, 2.0);
            let packed = pack(k, n, &w, &b);
            let head = CtcHead::parse(&packed).unwrap();
            assert_eq!((head.features(), head.classes()), (k, n));
            let (idx, p) = head.decode(&f, steps);
            let (want_idx, want_p) = reference(k, n, &w, &b, &f);
            assert_eq!(idx, want_idx, "k={k} n={n}");
            for (a, b) in p.iter().zip(&want_p) {
                assert!((a - b).abs() < 1e-4 * b.max(1e-3), "{a} vs {b}");
            }
        }
    }

    #[test]
    fn simd_and_portable_tiles_agree() {
        let k = 33;
        let (f, w, b) = (
            noise(MS * k, 4, 1.0),
            noise(k * TILE, 5, 1.0),
            noise(TILE, 6, 1.0),
        );
        let mut a = [[0f32; TILE]; MS];
        let mut c = [[0f32; TILE]; MS];
        tile(false, &f, k, &w, &b, &mut a);
        tile(simd_available(), &f, k, &w, &b, &mut c);
        for (x, y) in a.iter().flatten().zip(c.iter().flatten()) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn peaked_steps_skip_far_logits_without_changing_the_result() {
        // One dominant class: probability ~1, arg-max exact.
        let (k, n) = (2, 40);
        let mut w = vec![0.0; k * n];
        w[17] = 50.0; // feature 0 → class 17
        let b = vec![0.0; n];
        let head_bytes = pack(k, n, &w, &b);
        let head = CtcHead::parse(&head_bytes).unwrap();
        let (idx, p) = head.decode(&[1.0, 0.0], 1);
        assert_eq!(idx, vec![17]);
        assert!((p[0] - 1.0).abs() < 1e-6);
        // Rising maxima across tiles rescale the running sum correctly.
        let mut w = vec![0.0; k * n];
        // Feature 0's row: class c scores 0.1·c.
        for (c, v) in w.iter_mut().take(n).enumerate() {
            *v = c as f32 * 0.1;
        }
        let head_bytes = pack(k, n, &w, &b);
        let head = CtcHead::parse(&head_bytes).unwrap();
        let (idx, p) = head.decode(&[1.0, 0.0], 1);
        let (want_idx, want_p) = reference(k, n, &w, &b, &[1.0, 0.0]);
        assert_eq!(idx, want_idx);
        assert!((p[0] - want_p[0]).abs() < 1e-5);
    }

    #[test]
    fn parse_rejects_malformed_bytes() {
        let packed = pack(2, 3, &[0.0; 6], &[0.0; 3]);
        assert!(CtcHead::parse(&packed[..10]).is_err());
        assert!(CtcHead::parse(&packed[..packed.len() - 4]).is_err());
        let mut bad = packed.clone();
        bad[0] ^= 1;
        assert!(CtcHead::parse(&bad).is_err());
        // Misaligned input still parses (by copying).
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&packed);
        let head = CtcHead::parse(&shifted[1..]).unwrap();
        assert_eq!(head.classes(), 3);
    }
}
