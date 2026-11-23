//! Hand the resident pages of embedded model weights back to the OS once a
//! model's stage of the page is done.
//!
//! The weights are `include_bytes!` statics that tensors alias without a
//! copy, so they are read-only, file-backed pages of the executable: once
//! touched they count towards the process's resident set until the OS trims
//! them. Dropping them after use costs nothing but a soft page fault when
//! the next page needs the model again (the file stays in the page cache),
//! and keeps one stage's weights from stacking on top of the next stage's
//! activations: the detector's and the mobile recogniser's pages would
//! otherwise still be resident while the server or Latin recogniser runs.
//!
//! Supported on Windows (`VirtualUnlock` on unlocked pages removes them from
//! the working set) and Linux (`madvise(MADV_DONTNEED)`, which re-reads
//! unmodified private file pages on the next access). Elsewhere it is a
//! no-op.

// Builds without embedded models have no weights to release.
#![cfg_attr(not(spdf_neural_models), allow(dead_code))]

/// Only whole blocks of this many bytes inside the data are released, so
/// the range is page-aligned for every page size in use (4 KiB to 64 KiB).
const GRANULE: usize = 64 * 1024;

/// The largest `GRANULE`-aligned sub-range of `[addr, addr + len)`, as
/// `(start, len)`; `None` when the data spans no whole granule.
fn inner_range(addr: usize, len: usize, granule: usize) -> Option<(usize, usize)> {
    let start = addr.checked_add(granule - 1)? & !(granule - 1);
    let end = addr.checked_add(len)? & !(granule - 1);
    (end > start).then(|| (start, end - start))
}

/// Ask the OS to drop the resident pages of read-only embedded data. The
/// data stays valid: the next read faults the pages back in from the
/// executable. Only `'static` data qualifies, since only statics are
/// guaranteed to be file-backed and never written.
pub(crate) fn release(data: &'static [u8]) {
    if let Some((start, len)) = inner_range(data.as_ptr() as usize, data.len(), GRANULE) {
        // SAFETY: `[start, start + len)` lies inside `data`, a live static
        // that is never written, so dropping its pages loses nothing.
        unsafe { os::release(start as *const u8, len) };
    }
}

#[cfg(windows)]
mod os {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualUnlock(addr: *const u8, len: usize) -> i32;
    }

    /// Unlocking pages that were never locked fails with `ERROR_NOT_LOCKED`
    /// but removes them from the working set, which is the point.
    pub(super) unsafe fn release(addr: *const u8, len: usize) {
        // SAFETY: the caller passes a mapped range.
        unsafe { VirtualUnlock(addr, len) };
    }
}

#[cfg(target_os = "linux")]
mod os {
    const MADV_DONTNEED: i32 = 4;

    unsafe extern "C" {
        fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    }

    /// On a private file mapping that was never written, `MADV_DONTNEED`
    /// drops the pages and the next access re-reads them from the file.
    pub(super) unsafe fn release(addr: *const u8, len: usize) {
        // SAFETY: the caller passes a page-aligned, mapped, read-only range.
        unsafe { madvise(addr as *mut u8, len, MADV_DONTNEED) };
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod os {
    pub(super) unsafe fn release(_addr: *const u8, _len: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_range_rounds_inwards() {
        assert_eq!(
            inner_range(0x10000, 0x30000, 0x10000),
            Some((0x10000, 0x30000))
        );
        assert_eq!(
            inner_range(0x10001, 0x30000, 0x10000),
            Some((0x20000, 0x20000))
        );
        assert_eq!(
            inner_range(0xFFFF, 0x10002, 0x10000),
            Some((0x10000, 0x10000))
        );
        // No whole granule inside.
        assert_eq!(inner_range(0x10001, 0x10000, 0x10000), None);
        assert_eq!(inner_range(0x10000, 0, 0x10000), None);
        assert_eq!(inner_range(usize::MAX - 10, 5, 0x10000), None);
    }

    /// Unaligned (COFF sections cap alignment at 8 KiB), but long enough
    /// to contain whole granules wherever it lands.
    static BLOB: [u8; 5 * GRANULE] = [0xA5; 5 * GRANULE];

    #[test]
    fn released_static_data_reads_back_unchanged() {
        let data: &'static [u8] = &BLOB;
        assert!(inner_range(data.as_ptr() as usize, data.len(), GRANULE).is_some());
        let intact = |d: &[u8]| std::hint::black_box(d).iter().all(|&v| v == 0xA5);
        assert!(intact(data));
        release(data);
        assert!(intact(data));
        // Twice in a row, and on a sub-slice that spans no whole granule.
        release(data);
        release(&data[1..GRANULE]);
        assert!(intact(data));
    }
}
