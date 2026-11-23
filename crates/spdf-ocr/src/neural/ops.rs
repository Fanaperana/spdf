//! Fused element-wise kernels substituted into the generated models.
//!
//! `burn-onnx` lowers every ONNX `HardSwish` to burn's
//! `activation::hard_swish`, which is `x.clone() * hard_sigmoid(x)`: the
//! clone makes `x` shared, so `hard_sigmoid` cannot run in place and
//! allocates a second activation-sized buffer (single-threaded), then a
//! binary `mul` makes a third pass. PP-LCNetV3 (the mobile recognisers) and
//! the detector wrap almost every conv in one, so on the CPU these passes
//! were ~30% of a mobile recognition pass and doubled the live activation at
//! each of them. The build script points the generated code at
//! [`hard_swish`] here instead: on the Flex (CPU) backend it rewrites the
//! tensor in place in one parallel pass, computing bit-for-bit the same
//! values as burn's version.

use burn::prelude::*;
use burn_flex::{Flex, FlexTensor};

/// Elements per parallel task of an in-place element-wise pass: large
/// enough that task overhead is noise, small enough that a line's
/// activations spread over the pool.
const PAR_CHUNK: usize = 1 << 15;

/// burn's `hard_sigmoid` slope for `hard_swish`, as burn rounds it to `f32`.
const ALPHA: f32 = (1.0f64 / 6.0) as f32;

/// `hard_swish(x) = x · clamp(x/6 + 1/2, 0, 1)`, exactly as
/// `burn::tensor::activation::hard_swish` computes it, but in place and in
/// one parallel pass when `x` is a uniquely owned, contiguous `f32` tensor
/// on the Flex backend; anything else takes burn's path.
pub(crate) fn hard_swish<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    if x.device() != Device::flex() {
        return burn::tensor::activation::hard_swish(x);
    }
    let t: FlexTensor = x
        .try_into_primitive::<Flex>()
        .expect("a tensor on the Flex device is a Flex tensor");
    match hard_swish_in_place(t) {
        Ok(t) => Tensor::from_primitive::<Flex>(t),
        Err(t) => burn::tensor::activation::hard_swish(Tensor::from_primitive::<Flex>(t)),
    }
}

/// Apply hard-swish to `t`'s storage in place. Returns the tensor in `Err`
/// (unchanged) when it is not a uniquely owned, contiguous `f32` buffer.
fn hard_swish_in_place(mut t: FlexTensor) -> Result<FlexTensor, FlexTensor> {
    let layout = t.layout();
    if t.dtype() != burn::tensor::DType::F32
        || !t.is_unique()
        || !layout.is_contiguous()
        || layout.start_offset() != 0
    {
        return Err(t);
    }
    let n = layout.num_elements();
    hard_swish_slice(&mut t.storage_mut::<f32>()[..n]);
    Ok(t)
}

/// One hard-swish step, with the same `f32` operations, in the same order,
/// as burn's `hard_sigmoid` followed by its `mul`.
#[inline(always)]
fn hard_swish_f32(x: f32) -> f32 {
    x * (ALPHA * x + 0.5).clamp(0.0, 1.0)
}

/// Hard-swish every element of `v` in place, in parallel on the current
/// rayon pool for large buffers.
fn hard_swish_slice(v: &mut [f32]) {
    use rayon::prelude::*;
    if v.len() <= PAR_CHUNK {
        v.iter_mut().for_each(|x| *x = hard_swish_f32(*x));
    } else {
        v.par_chunks_mut(PAR_CHUNK)
            .for_each(|c| c.iter_mut().for_each(|x| *x = hard_swish_f32(*x)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values around the kinks (±3), signed zeros, large magnitudes.
    fn samples(n: usize) -> Vec<f32> {
        let mut v: Vec<f32> = vec![
            -1e6, -3.5, -3.0, -2.999, -0.0, 0.0, 1e-7, 2.999, 3.0, 3.5, 1e6,
        ];
        v.extend((0..n).map(|i| (i as f32 * 0.7331).sin() * 8.0));
        v
    }

    fn burn_ref(v: &[f32]) -> Vec<f32> {
        let x = Tensor::<1>::from_data(TensorData::new(v.to_vec(), [v.len()]), &Device::flex());
        burn::tensor::activation::hard_swish(x)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn fused_kernel_matches_burn_bit_for_bit() {
        // Below and above the parallel threshold.
        for n in [100, 3 * PAR_CHUNK + 17] {
            let v = samples(n);
            let mut fused = v.clone();
            hard_swish_slice(&mut fused);
            assert_eq!(bits(&fused), bits(&burn_ref(&v)), "n = {n}");
        }
    }

    #[test]
    fn tensor_entry_point_matches_burn() {
        let v = samples(500);
        let x = Tensor::<1>::from_data(TensorData::new(v.clone(), [v.len()]), &Device::flex());
        let y = hard_swish(x).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(bits(&y), bits(&burn_ref(&v)));
    }

    #[test]
    fn shared_tensor_is_not_modified_in_place() {
        let v = samples(64);
        let x = Tensor::<1>::from_data(TensorData::new(v.clone(), [v.len()]), &Device::flex());
        let keep = x.clone();
        let y = hard_swish(x).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(bits(&y), bits(&burn_ref(&v)));
        let kept = keep.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(bits(&kept), bits(&v), "the other owner's data must survive");
    }

    #[test]
    fn strided_view_takes_burns_path() {
        let v = samples(61);
        let x = Tensor::<2>::from_data(TensorData::new(v.clone(), [8, 9]), &Device::flex());
        let t = x.clone().transpose();
        let want = burn::tensor::activation::hard_swish(t.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        drop(x);
        let got = hard_swish(t).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(bits(&got), bits(&want));
    }
}
