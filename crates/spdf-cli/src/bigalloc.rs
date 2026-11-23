//! Global allocator: mimalloc for ordinary blocks, the OS page allocator
//! plus a small shared cache for large ones.
//!
//! Neural OCR churns through multi-megabyte activation tensors that are
//! allocated on one inference worker and often freed on another. mimalloc
//! parks such cross-thread frees on the owning thread's list until that
//! thread allocates again, and keeps freed pages committed per thread, so
//! a stage's working set sat 15-40 MB above its live heap. Blocks of
//! [`DIRECT_MIN`] bytes or more are therefore mapped straight from the OS
//! (rounded up to a size class) and kept, once freed, in one cache shared
//! by all threads and capped at [`CACHE_CAP`] bytes; anything beyond the
//! cap is unmapped at once. The working set then stays within the cap of
//! the live heap, while the recurring tensor shapes of consecutive layers
//! and batches still reuse warm pages instead of faulting fresh ones.
//! Fresh mappings come zeroed, so `alloc_zeroed` skips its memset.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Smallest block served by the OS page allocator.
pub const DIRECT_MIN: usize = 256 << 10;

/// Largest alignment the OS mapping guarantees (one page on every
/// supported platform; Windows reservations are even 64 KiB aligned).
const PAGE: usize = 4096;

/// Threshold in force (overridable with `SPDF_DIRECT_MIN` for tuning; 0
/// disables the direct path).
static THRESHOLD: AtomicUsize = AtomicUsize::new(DIRECT_MIN);

/// Byte cap of the shared free-block cache.
pub const CACHE_CAP: usize = 6 << 20;
static CAP: AtomicUsize = AtomicUsize::new(CACHE_CAP);

fn env_usize(k: &str) -> Option<usize> {
    std::env::var_os(k).and_then(|v| v.to_str().and_then(|s| s.parse::<usize>().ok()))
}

/// Read the tuning override. Call once at start-up, before threads exist.
pub fn init_from_env() {
    if let Some(v) = env_usize("SPDF_DIRECT_MIN") {
        THRESHOLD.store(if v == 0 { usize::MAX } else { v }, Ordering::Relaxed);
    }
    if let Some(v) = env_usize("SPDF_DIRECT_CACHE") {
        CAP.store(v, Ordering::Relaxed);
    }
}

/// Mapping size for a request: rounded up to a size class with 1/8 steps
/// per power of two, so tensors of nearly the same shape share cached
/// blocks. The unused tail is never touched, so it costs no memory.
fn class_size(size: usize) -> usize {
    let step = (size.next_power_of_two() / 16).max(PAGE);
    size.div_ceil(step) * step
}

const SLOTS: usize = 64;

/// Freed direct blocks kept for reuse, shared by all threads (unlike the
/// per-thread caches of a general allocator, a block freed on one worker
/// serves the next request on any other), oldest first.
struct Cache {
    blocks: [(usize, usize); SLOTS],
    len: usize,
    bytes: usize,
}

static CACHE: std::sync::Mutex<Cache> = std::sync::Mutex::new(Cache {
    blocks: [(0, 0); SLOTS],
    len: 0,
    bytes: 0,
});

fn take_cached(size: usize) -> *mut u8 {
    let Ok(mut c) = CACHE.lock() else {
        return std::ptr::null_mut();
    };
    // Newest first: its pages are the likeliest to still be warm.
    let len = c.len;
    if let Some(i) = (0..len).rev().find(|&i| c.blocks[i].1 == size) {
        let (p, s) = c.blocks[i];
        c.blocks.copy_within(i + 1..len, i);
        c.len -= 1;
        c.bytes -= s;
        return p as *mut u8;
    }
    std::ptr::null_mut()
}

/// Cache a freed block, evicting the oldest ones over the cap.
fn put_cached(ptr: *mut u8, size: usize) {
    let cap = CAP.load(Ordering::Relaxed);
    let mut evict = [(0usize, 0usize); SLOTS + 1];
    let mut n = 0;
    if size > cap {
        evict[0] = (ptr as usize, size);
        n = 1;
    } else if let Ok(mut c) = CACHE.lock() {
        while c.len > 0 && (c.len == SLOTS || c.bytes + size > cap) {
            evict[n] = c.blocks[0];
            n += 1;
            c.bytes -= c.blocks[0].1;
            let len = c.len;
            c.blocks.copy_within(1..len, 0);
            c.len -= 1;
        }
        let len = c.len;
        c.blocks[len] = (ptr as usize, size);
        c.len += 1;
        c.bytes += size;
    } else {
        evict[0] = (ptr as usize, size);
        n = 1;
    }
    for &(p, s) in &evict[..n] {
        unsafe { os::unmap(p as *mut u8, s) };
    }
}

/// Return every cached block to the OS (between OCR stages).
pub fn flush() {
    let mut evict = [(0usize, 0usize); SLOTS];
    let n = match CACHE.lock() {
        Ok(mut c) => {
            let n = c.len;
            evict[..n].copy_from_slice(&c.blocks[..n]);
            c.len = 0;
            c.bytes = 0;
            n
        }
        Err(_) => 0,
    };
    for &(p, s) in &evict[..n] {
        unsafe { os::unmap(p as *mut u8, s) };
    }
}

unsafe fn direct_alloc(size: usize, zero: bool) -> *mut u8 {
    let size = class_size(size);
    let p = take_cached(size);
    if p.is_null() {
        // Fresh OS pages are zero-filled.
        return unsafe { os::map(size) };
    }
    if zero {
        unsafe { std::ptr::write_bytes(p, 0, size) };
    }
    p
}

unsafe fn direct_free(ptr: *mut u8, size: usize) {
    put_cached(ptr, class_size(size));
}

pub struct Alloc;

/// Whether a block of this layout lives in its own OS mapping. A pure
/// function of the layout, so `dealloc` always agrees with `alloc`.
#[inline]
fn is_direct(size: usize, align: usize) -> bool {
    direct_at(size, align, THRESHOLD.load(Ordering::Relaxed))
}

/// Same decision with an explicit threshold (unit-testable).
#[inline]
fn direct_at(size: usize, align: usize, threshold: usize) -> bool {
    size >= threshold && align <= PAGE
}

// SAFETY: every block is either a mimalloc block (allocated, resized and
// freed by mimalloc) or an OS mapping of at least `size` bytes; `is_direct`
// is deterministic in the layout, so frees reach the allocator that made
// the block, and `realloc` moves blocks across the boundary by copying.
unsafe impl GlobalAlloc for Alloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if is_direct(layout.size(), layout.align()) {
            unsafe { direct_alloc(layout.size(), false) }
        } else {
            unsafe { mimalloc::MiMalloc.alloc(layout) }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if is_direct(layout.size(), layout.align()) {
            unsafe { direct_alloc(layout.size(), true) }
        } else {
            unsafe { mimalloc::MiMalloc.alloc_zeroed(layout) }
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if is_direct(layout.size(), layout.align()) {
            unsafe { direct_free(ptr, layout.size()) }
        } else {
            unsafe { mimalloc::MiMalloc.dealloc(ptr, layout) }
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old_direct = is_direct(layout.size(), layout.align());
        let new_direct = is_direct(new_size, layout.align());
        if !old_direct && !new_direct {
            return unsafe { mimalloc::MiMalloc.realloc(ptr, layout, new_size) };
        }
        if old_direct && new_direct && class_size(layout.size()) == class_size(new_size) {
            // Same mapping size: the block already fits.
            return ptr;
        }
        // SAFETY: the caller guarantees `new_size` with `layout.align()` is
        // a valid layout.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let new = unsafe { self.alloc(new_layout) };
        if !new.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
        }
        new
    }
}

#[cfg(windows)]
mod os {
    use std::ffi::c_void;

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_READWRITE: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
    }

    pub unsafe fn map(size: usize) -> *mut u8 {
        unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
            .cast()
        }
    }

    pub unsafe fn unmap(ptr: *mut u8, _size: usize) {
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(unix)]
mod os {
    pub unsafe fn map(size: usize) -> *mut u8 {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            std::ptr::null_mut()
        } else {
            p.cast()
        }
    }

    pub unsafe fn unmap(ptr: *mut u8, size: usize) {
        unsafe { libc::munmap(ptr.cast(), size) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_depends_only_on_layout() {
        assert!(!direct_at(DIRECT_MIN - 1, 8, DIRECT_MIN));
        assert!(direct_at(DIRECT_MIN, 8, DIRECT_MIN));
        assert!(direct_at(DIRECT_MIN, PAGE, DIRECT_MIN));
        // Over-aligned blocks stay with mimalloc: a mapping is only
        // page-aligned.
        assert!(!direct_at(DIRECT_MIN, 2 * PAGE, DIRECT_MIN));
        assert!(!direct_at(DIRECT_MIN, 8, usize::MAX));
    }

    #[test]
    fn size_classes_cover_the_request_within_an_eighth() {
        for size in [
            DIRECT_MIN,
            DIRECT_MIN + 1,
            300_000,
            1 << 20,
            (1 << 20) + 1,
            7_777_777,
        ] {
            let c = class_size(size);
            assert!(c >= size);
            assert!(c - size <= size / 8 + PAGE, "{size} -> {c}");
            assert_eq!(c % PAGE, 0);
            // Every size of a class maps to the same class.
            assert_eq!(class_size(c), c);
        }
    }

    #[test]
    fn cache_reuses_freed_blocks_and_respects_the_cap() {
        // A size no other test (or the test harness) is likely to use.
        let size = class_size(3 * DIRECT_MIN + 4321);
        unsafe {
            let p = direct_alloc(size, false);
            assert!(!p.is_null());
            *p = 42;
            direct_free(p, size);
            let q = direct_alloc(size, true);
            assert_eq!(q, p, "a freed block is reused for the same class");
            assert_eq!(*q, 0, "zeroed allocations are cleared on reuse");
            direct_free(q, size);
        }
        flush();
        let c = CACHE.lock().unwrap();
        assert!(c.bytes <= CAP.load(Ordering::Relaxed));
    }

    #[test]
    fn large_blocks_round_trip() {
        let a = Alloc;
        let layout = Layout::from_size_align(DIRECT_MIN + 123, 64).unwrap();
        unsafe {
            let p = a.alloc_zeroed(layout);
            assert!(!p.is_null());
            assert_eq!(p as usize % 64, 0);
            let s = std::slice::from_raw_parts_mut(p, layout.size());
            assert!(s.iter().all(|&b| b == 0));
            s[0] = 7;
            s[layout.size() - 1] = 9;
            // Grow within the direct path, then shrink across the boundary.
            let p = a.realloc(p, layout, 2 * DIRECT_MIN);
            assert!(!p.is_null());
            assert_eq!((*p, *p.add(layout.size() - 1)), (7, 9));
            let big = Layout::from_size_align(2 * DIRECT_MIN, 64).unwrap();
            let p = a.realloc(p, big, 100);
            assert!(!p.is_null());
            assert_eq!(*p, 7);
            // Small back up to large.
            let small = Layout::from_size_align(100, 64).unwrap();
            let p = a.realloc(p, small, DIRECT_MIN);
            assert_eq!(*p, 7);
            a.dealloc(p, Layout::from_size_align(DIRECT_MIN, 64).unwrap());
        }
    }
}
