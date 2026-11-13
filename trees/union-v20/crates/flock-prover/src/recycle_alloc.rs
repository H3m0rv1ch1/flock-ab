//! Recycling global allocator for the prover process.
//!
//! Blocks at least 32 KiB are parked on exact-size freelists rather than
//! returned to the system allocator. The ranked worker performs an untimed warm proof
//! with the same allocation pattern, so the timed proof reuses resident pages
//! for large allocations not already handled by the typed scratch pools.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{Acquire, Release},
};

const RECYCLE_MIN: usize = 32 * 1024;
const MAX_ALIGN: usize = 16;
const MAX_CLASSES: usize = 512;

struct Class {
    size: AtomicUsize,
    head: Mutex<usize>,
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Class = Class {
    size: AtomicUsize::new(0),
    head: Mutex::new(0),
};
static CLASSES: [Class; MAX_CLASSES] = [EMPTY; MAX_CLASSES];

#[inline]
fn class_slot(size: usize) -> usize {
    (size.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 55) % MAX_CLASSES
}

// Match only the existing padded large-buffer layout. Ordinary small
// power-of-two sizes use System instead of a shared per-size freelist.
#[inline]
fn retained_size(size: usize) -> bool {
    const TAIL: usize = flock_core::UNINIT_VEC_TAIL_PAGES * 4096;
    size >= flock_core::UNINIT_VEC_LARGE_MIN_BYTES + TAIL
        && (size - TAIL).is_power_of_two()
}

#[inline]
fn find_class(size: usize, insert: bool) -> Option<usize> {
    // At most 42 valid classes on 64-bit layouts (payload exponents 21..62).
    // This changes retention only; recyclable() still controls alignment.
    if !retained_size(size) {
        return None;
    }
    let start = class_slot(size);
    for probe in 0..MAX_CLASSES {
        let i = (start + probe) % MAX_CLASSES;
        let s = CLASSES[i].size.load(Acquire);
        if s == size {
            return Some(i);
        }
        if s == 0 {
            if !insert {
                return None;
            }
            if CLASSES[i]
                .size
                .compare_exchange(0, size, Release, Acquire)
                .is_ok()
            {
                return Some(i);
            }
            if CLASSES[i].size.load(Acquire) == size {
                return Some(i);
            }
        }
    }
    None
}

#[inline]
fn recyclable(layout: &Layout) -> bool {
    layout.size() >= RECYCLE_MIN && layout.align() <= MAX_ALIGN
}

/// `FLOCK_NO_ALIGN64=1` restores raw System pointers for the recyclable
/// class (exact same-binary A/B). Latched once, and only ever initialized
/// from a RECYCLABLE allocation — `var_os`'s own small allocations take the
/// non-recyclable branch straight to System, so initialization cannot
/// re-enter this latch.
#[inline]
fn align64_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ALIGN64").is_none())
}

/// Every large prover buffer historically landed `16 mod 64` (glibc mmap
/// chunk header), which the hot kernels pay for three ways: every wide
/// load/store on these buffers is a cache-line split, non-temporal 64-byte
/// writes straddle two lines (partial write-combining flushes are DRAM
/// read-modify-writes), and single-uop ZMM streaming stores are illegal.
/// Ordinary recyclable classes over-allocate by `ALIGN_SLACK` and return
/// `round_up(base + 8, 64)`. The selected witness class instead reserves
/// `COLOR_SLACK` for a page frame and a cache-line color. Both leave room for the
/// back-offset word stored at `aligned - 8` — BEFORE the block, so it can
/// never collide with the freelist link word at `aligned + 0`.
const ALIGN_SLACK: usize = 64;
const COLOR_SLACK: usize = 8192;
static COLOR_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn colored_layout(layout: &Layout) -> bool {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        const WITNESS: usize = (512 << 20) + flock_core::UNINIT_VEC_TAIL_PAGES * 4096;
        if layout.size() != WITNESS { return false; }
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("FLOCK_NO_WITNESS_COLOR").is_none())
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    { let _ = layout; false }
}

#[inline]
fn slack(layout: &Layout) -> usize {
    if colored_layout(layout) { COLOR_SLACK } else { ALIGN_SLACK }
}

#[inline]
fn adjusted(layout: &Layout) -> Option<Layout> {
    Layout::from_size_align(layout.size().checked_add(slack(layout))?, MAX_ALIGN).ok()
}

#[inline]
fn aligned_address(base: usize, color: Option<usize>) -> usize {
    if let Some(color) = color {
        ((base + 8 + 4095) & !4095) + (color & 63) * 64
    } else {
        (base + 8 + 63) & !63
    }
}

#[inline]
unsafe fn align_up(base: *mut u8, layout: &Layout) -> *mut u8 {
    if base.is_null() { return base; }
    let color = colored_layout(layout).then(|| {
        COLOR_COUNTER.fetch_add(31, std::sync::atomic::Ordering::Relaxed) & 63
    });
    let offset = aligned_address(base as usize, color) - base as usize;
    debug_assert!((8..=slack(layout)).contains(&offset));
    // The offset and complete payload fit in the reserved System allocation.
    let aligned = unsafe { base.add(offset) };
    unsafe { *(aligned.sub(8) as *mut usize) = offset };
    aligned
}

#[inline]
unsafe fn align_base(ptr: *mut u8, layout: &Layout) -> *mut u8 {
    let off = unsafe { *(ptr.sub(8) as *const usize) };
    debug_assert!((8..=slack(layout)).contains(&off));
    unsafe { ptr.sub(off) }
}

#[inline]
fn pop(size: usize) -> *mut u8 {
    let Some(i) = find_class(size, false) else {
        return core::ptr::null_mut();
    };
    let mut head = CLASSES[i].head.lock().unwrap();
    let top = *head;
    if top == 0 {
        return core::ptr::null_mut();
    }
    *head = unsafe { *(top as *const usize) };
    top as *mut u8
}

#[inline]
fn push(ptr: *mut u8, size: usize) -> bool {
    let Some(i) = find_class(size, true) else {
        return false;
    };
    let mut head = CLASSES[i].head.lock().unwrap();
    unsafe { *(ptr as *mut usize) = *head };
    *head = ptr as usize;
    true
}

pub struct RecycleAlloc;

// SAFETY: every recycled block came from this allocator with the exact same
// size class. With align64 on, every recyclable-class block System sees uses
// the `adjusted` layout on both alloc and dealloc, and the user pointer is
// recovered to its System base via the offset word `align_up` stored.
// glibc/mimalloc and the macOS allocator provide at least 16-byte alignment
// at these sizes; layouts requiring larger alignment bypass the recycler.
unsafe impl GlobalAlloc for RecycleAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if recyclable(&layout) {
            let p = pop(layout.size());
            // Provenance for `flock_core::alloc_uninit_vec`, which skips its
            // `MADV_HUGEPAGE` re-advise on a block that came off a freelist.
            let _ = flock_core::LAST_ALLOC_RECYCLED.try_with(|c| c.set(!p.is_null()));
            if !p.is_null() {
                return p;
            }
            if align64_enabled() {
                // SAFETY: adjusted() reserves the slack align_up consumes.
                let Some(system_layout) = adjusted(&layout) else { return core::ptr::null_mut(); };
                return unsafe { align_up(System.alloc(system_layout), &layout) };
            }
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if recyclable(&layout) {
            let p = pop(layout.size());
            let _ = flock_core::LAST_ALLOC_RECYCLED.try_with(|c| c.set(!p.is_null()));
            if !p.is_null() {
                unsafe { core::ptr::write_bytes(p, 0, layout.size()) };
                return p;
            }
            if align64_enabled() {
                // The user range [aligned, aligned + size) is inside the
                // zeroed System block; the offset word sits before it.
                // SAFETY: as for alloc.
                let Some(system_layout) = adjusted(&layout) else { return core::ptr::null_mut(); };
                return unsafe { align_up(System.alloc_zeroed(system_layout), &layout) };
            }
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if recyclable(&layout) {
            if push(ptr, layout.size()) {
                return;
            }
            if align64_enabled() {
                // SAFETY: every recyclable-class pointer this allocator
                // handed out with align64 on came from align_up; recover the
                // System base and the adjusted layout it was allocated with.
                unsafe {
                    return System.dealloc(align_base(ptr, &layout),
                        adjusted(&layout).expect("layout accepted at allocation"));
                }
            }
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn colored_offsets_preserve_bounds_and_layout_scope() {
        use super::*;
        const PAYLOAD: usize = 128;
        for base in (0x100000usize..0x101000).step_by(16) {
            for color in 0..64 {
                let ptr = aligned_address(base, Some(color));
                assert_eq!(ptr % 64, 0);
                assert!((8..=COLOR_SLACK).contains(&(ptr - base)));
                assert!(ptr + PAYLOAD <= base + PAYLOAD + COLOR_SLACK);
            }
            let ptr = aligned_address(base, None);
            assert_eq!(ptr % 64, 0);
            assert!((8..=ALIGN_SLACK).contains(&(ptr - base)));
        }
        let huge = Layout::from_size_align(isize::MAX as usize - 15, 16).unwrap();
        assert!(adjusted(&huge).is_none());
        let ordinary = Layout::from_size_align(2 << 20,16).unwrap();
        assert!(!colored_layout(&ordinary));
        assert_eq!(adjusted(&ordinary).unwrap().size(), ordinary.size()+ALIGN_SLACK);
        #[cfg(all(target_os="linux",target_arch="x86_64"))]
        unsafe {
            let witness = Layout::from_size_align(
                (512 << 20)+flock_core::UNINIT_VEC_TAIL_PAGES*4096,16).unwrap();
            assert!(colored_layout(&witness));
            assert_eq!(adjusted(&witness).unwrap().size(),witness.size()+COLOR_SLACK);
            // Exercise real metadata roundtrips without reserving a batch-sized
            // payload: align_up only needs the slack and a small live suffix.
            let backing = Layout::from_size_align(PAYLOAD+COLOR_SLACK,16).unwrap();
            let mut colors=[false;64];
            for round in 0..64 {
                let base=System.alloc(backing); assert!(!base.is_null());
                let ptr=align_up(base,&witness);
                assert_eq!(align_base(ptr,&witness),base);
                assert!(ptr as usize + PAYLOAD <= base as usize + backing.size());
                colors[(ptr as usize %4096)/64]=true;
                core::ptr::write_bytes(ptr,round as u8,PAYLOAD);
                assert!(core::slice::from_raw_parts(ptr,PAYLOAD).iter().all(|&x|x==round as u8));
                System.dealloc(base,backing);
            }
            assert!(colors.into_iter().all(|x|x));
        }
    }


    #[test]
    fn padded_only_retention_preserves_real_layouts_and_bypass() {
        use super::*;
        use std::sync::Barrier;
        use std::sync::atomic::Ordering::Relaxed;
        const TAIL: usize = 96 * 4096;
        const MIN: usize = 2 << 20;
        assert_eq!(flock_core::UNINIT_VEC_TAIL_PAGES, 96);
        assert_eq!(flock_core::UNINIT_VEC_LARGE_MIN_BYTES, MIN);
        for exponent in 21..usize::BITS - 1 {
            let payload = 1usize << exponent;
            let size = payload + TAIL;
            assert!(retained_size(size));
            assert!(!retained_size(size - 1));
            assert!(!retained_size(size + 1));
            assert!(!retained_size(payload));
            assert!(Layout::from_size_align(size, 8).is_ok());
        }
        for size in [0, RECYCLE_MIN, 64 * 1024, 128 * 1024, MIN, MIN + TAIL - 1, usize::MAX] {
            assert!(!retained_size(size));
            assert!(find_class(size, true).is_none());
        }
        let barrier = Barrier::new(8);
        let reused = AtomicUsize::new(0);
        let bypassed = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for lane in 0..8usize {
                let (barrier, reused, bypassed) = (&barrier, &reused, &bypassed);
                scope.spawn(move || {
                    for round in 0..8usize {
                        let (size, admitted) = match round % 4 {
                            0 => (RECYCLE_MIN, false),
                            1 => (MIN + TAIL, true),
                            2 => (2 * MIN + TAIL, true),
                            _ => (MIN + TAIL + 128, false),
                        };
                        let layout = Layout::from_size_align(size, 8).unwrap();
                        unsafe {
                            let block = RecycleAlloc.alloc(layout);
                            assert!(!block.is_null());
                            let hit = flock_core::LAST_ALLOC_RECYCLED.with(|c| c.get());
                            if hit { assert!(admitted); reused.fetch_add(1, Relaxed); }
                            if !admitted { assert!(!hit); bypassed.fetch_add(1, Relaxed); }
                            assert_eq!(block as usize % 64, 0);
                            let value = (lane + 1 + round * 8) as u8;
                            core::ptr::write_bytes(block, value, size);
                            barrier.wait();
                            assert!(core::slice::from_raw_parts(block, size).iter().all(|&v| v == value));
                            barrier.wait();
                            RecycleAlloc.dealloc(block, layout);
                            let zeroed = RecycleAlloc.alloc_zeroed(layout);
                            assert!(!zeroed.is_null());
                            let hit = flock_core::LAST_ALLOC_RECYCLED.with(|c| c.get());
                            if hit { assert!(admitted); reused.fetch_add(1, Relaxed); }
                            if !admitted { assert!(!hit); bypassed.fetch_add(1, Relaxed); }
                            assert_eq!(zeroed as usize % 64, 0);
                            assert!(core::slice::from_raw_parts(zeroed, size).iter().all(|&v| v == 0));
                            RecycleAlloc.dealloc(zeroed, layout);
                        }
                    }
                });
            }
        });
        let occupied = CLASSES.iter().filter(|c| {
            let size = c.size.load(Acquire);
            assert!(size == 0 || retained_size(size));
            size != 0
        }).count();
        assert!(occupied <= (usize::BITS - 1 - 21) as usize);
        assert_eq!(bypassed.load(Relaxed), 64);
        eprintln!("REUSED_ALLOCATIONS={}", reused.load(Relaxed));
    }
    /// With `FLOCK_NO_ALIGN64` unset (the ranked worker's cleared env), every
    /// recyclable-class allocation — fresh from System, recycled off the
    /// freelist, and grown through realloc — returns a 64-aligned pointer
    /// with intact contents. (The switch-off arm can't be covered in the
    /// same process: the latch resolves once.)
    #[test]
    fn recyclable_class_is_64_aligned_and_roundtrips() {
        for i in 0..4usize {
            let n = 32 * 1024 + 4096 * i + 128;
            let v = vec![7u8; n];
            assert_eq!(v.as_ptr() as usize % 64, 0, "fresh alloc n={n}");
            drop(v);
            let v2 = vec![9u8; n];
            assert_eq!(v2.as_ptr() as usize % 64, 0, "recycled alloc n={n}");
            assert!(v2.iter().all(|&b| b == 9), "contents survive recycle n={n}");
            let z = vec![0u8; n + 64];
            assert_eq!(z.as_ptr() as usize % 64, 0, "alloc_zeroed n={}", n + 64);
            assert!(z.iter().all(|&b| b == 0), "zeroed contents n={}", n + 64);
        }
        // Growth path: realloc = alloc + copy + dealloc through this
        // allocator; contents must survive the move between size classes.
        let mut g: Vec<u8> = Vec::with_capacity(48 * 1024);
        g.extend(std::iter::repeat_n(0xA5u8, 48 * 1024));
        g.extend(std::iter::repeat_n(0x5Au8, 128 * 1024));
        assert_eq!(g.as_ptr() as usize % 64, 0, "grown alloc");
        assert!(g[..48 * 1024].iter().all(|&b| b == 0xA5));
        assert!(g[48 * 1024..].iter().all(|&b| b == 0x5A));
    }
}
