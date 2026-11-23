//! Flush subnormal floats to zero on the inference threads.
//!
//! The PP-OCRv5 weights hold tens of thousands of subnormal `f32` values
//! (tiny leftovers of training and batch-norm folding), and products with
//! them stay subnormal. x86 cores take a microcode assist on every SIMD
//! operation that reads or produces a subnormal, which made the large 1x1
//! convolutions of the server recogniser run ~20-35x slower than the same
//! GEMM on ordinary data (a 1408-px line: 790 ms → 250 ms with this mode).
//!
//! Subnormals are below `1.2e-38`, far under anything a network output
//! depends on: recognised text is byte-identical with and without the mode.
//! It is what inference runtimes (ONNX Runtime, TensorFlow, PyTorch's
//! `set_flush_denormal`) do too. The mode is per thread, so it is set only on
//! spdf's own inference pool and never leaks into a caller's threads.

/// Make the current thread treat subnormal inputs as zero and flush
/// subnormal results to zero (x86-64: MXCSR DAZ + FTZ; AArch64: FPCR FZ).
/// Returns whether the mode is supported (and now on) for this thread.
pub(crate) fn flush_subnormals() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        /// MXCSR flush-to-zero (bit 15) and denormals-are-zero (bit 6).
        const FTZ_DAZ: u32 = (1 << 15) | (1 << 6);
        let mut csr: u32 = 0;
        // SAFETY: STMXCSR / LDMXCSR only read and write the SSE control
        // register of this thread (SSE is baseline on x86-64), through a
        // pointer to a live local.
        unsafe {
            std::arch::asm!("stmxcsr [{}]", in(reg) &mut csr, options(nostack, preserves_flags));
            csr |= FTZ_DAZ;
            std::arch::asm!("ldmxcsr [{}]", in(reg) &csr, options(nostack, preserves_flags, readonly));
        }
        true
    }
    #[cfg(target_arch = "aarch64")]
    {
        /// FPCR flush-to-zero (bit 24), covering inputs and outputs.
        const FZ: u64 = 1 << 24;
        let mut fpcr: u64;
        // SAFETY: reading and writing FPCR only changes this thread's
        // floating-point mode.
        unsafe {
            std::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags));
            fpcr |= FZ;
            std::arch::asm!("msr fpcr, {}", in(reg) fpcr, options(nomem, nostack, preserves_flags));
        }
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    #[test]
    fn subnormals_flush_only_on_threads_that_ask() {
        // A fresh thread each, so the test harness's threads keep IEEE mode.
        let tiny = || black_box(f32::MIN_POSITIVE) * black_box(0.25);
        let plain = std::thread::spawn(tiny).join().unwrap();
        assert!(plain > 0.0 && plain.is_subnormal());
        let (supported, flushed, normal) = std::thread::spawn(move || {
            let on = flush_subnormals();
            (on, tiny(), black_box(1.5f32) * black_box(2.0))
        })
        .join()
        .unwrap();
        assert_eq!(normal, 3.0);
        if supported {
            assert_eq!(flushed, 0.0);
        } else {
            assert_eq!(flushed, plain);
        }
    }
}
