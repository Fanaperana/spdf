//! Return an allocator's cached free memory to the OS between OCR stages.
//!
//! Each stage of a page (detection, recognition, ...) allocates and frees
//! tens of megabytes of activations, spread over the engine's worker
//! threads. Caching allocators such as mimalloc keep freed pages in the
//! freeing thread's heap and only return them to the OS when that heap is
//! used again; a worker that goes idle after a stage therefore keeps its
//! share of the stage's peak committed while the next stage, scheduled on
//! other workers, allocates fresh memory on top of it. Releasing the caches
//! at stage boundaries keeps the stages' peaks from stacking.
//!
//! The OCR crate does not choose the global allocator, so the binary that
//! does registers how to trim it with [`set_trim_hook`]. Without a hook,
//! trimming is a no-op.

use std::sync::OnceLock;

static HOOK: OnceLock<fn()> = OnceLock::new();

/// Register `hook`, called on every OCR worker thread at stage boundaries
/// to hand that thread's cached free memory back to the OS (e.g. mimalloc's
/// `mi_collect(true)`). It must be cheap when there is nothing to free and
/// safe to call from any thread. Only the first registration takes effect;
/// returns whether this one did.
pub fn set_trim_hook(hook: fn()) -> bool {
    HOOK.set(hook).is_ok()
}

/// Trim the calling thread's allocator caches.
pub(crate) fn trim_current() {
    if let Some(hook) = HOOK.get() {
        hook();
    }
}

/// Trim every thread of the rayon pool the caller runs in (the calling
/// thread included), or only the calling thread outside a pool. Never
/// starts rayon's global pool.
#[cfg(feature = "neural")]
pub(crate) fn trim_pool() {
    let Some(hook) = HOOK.get() else { return };
    if rayon::current_thread_index().is_some() {
        rayon::broadcast(|_| hook());
    } else {
        hook();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CALLS: AtomicUsize = AtomicUsize::new(0);

    fn count() {
        CALLS.fetch_add(1, Ordering::SeqCst);
    }

    // One test owns the process-wide hook, so the counts stay exact.
    #[test]
    fn hook_runs_on_the_caller_and_on_every_pool_thread() {
        trim_current(); // no hook yet: a no-op
        assert_eq!(CALLS.load(Ordering::SeqCst), 0);
        assert!(set_trim_hook(count));
        assert!(!set_trim_hook(count), "only the first hook registers");
        trim_current();
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        #[cfg(feature = "neural")]
        {
            // Outside a pool: just the caller.
            trim_pool();
            assert_eq!(CALLS.load(Ordering::SeqCst), 2);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(3)
                .build()
                .unwrap();
            pool.install(trim_pool);
            assert_eq!(CALLS.load(Ordering::SeqCst), 5);
        }
    }
}
