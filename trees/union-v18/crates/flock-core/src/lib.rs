//! `flock-core`: the protocol library and verifier for Flock's R1CS-over-GF(2)
//! sumcheck/zerocheck PIOP with a multilinear PCS.
//!
//! This crate carries everything the verifier needs. It is portable — the NEON
//! kernels in `field`, `ntt`, `lincheck`, `zerocheck`, and `merkle` have scalar
//! fallbacks — though it is tuned for Apple silicon. The end-to-end prover, the
//! hash R1CS encoders, and the CLI live in the `flock-prover` crate built on
//! top of this one.
//!
//! Protocol flow:
//!   1. Prover commits to the witness z ∈ GF(2)^n via a multilinear PCS.
//!   2. Prover computes the row-witnesses a = A·z, b = B·z, c = C·z.
//!   3. Zerocheck PIOP reduces a·b ⊕ c = 0 to evaluation claims on (â, b̂, ĉ) at ρ.
//!   4. Lincheck PIOP reduces those to a single evaluation claim ẑ(ρ') = v.
//!   5. PCS opens ẑ at ρ'.
//!
//! Workspace-wide Clippy `allow`s for the hand-tuned numeric kernels are
//! declared in `[workspace.lints.clippy]` at the repo root.

pub mod bits;
#[path = "stage_ir.rs"]
pub mod gf2_kernels;
pub mod challenger;
pub mod field;
pub mod gaptime;
pub mod rstrace;
pub mod gpu;
pub mod hash;
pub mod lincheck;
pub mod merkle;
pub mod ntt;
pub mod pcs;
pub mod permutation;
pub mod proof;
pub mod r1cs;
pub mod scratch;
pub mod verifier;
pub mod zerocheck;

/// Configure rayon's global thread pool to use only performance cores on
/// Apple silicon (excluding efficiency cores).
///
/// On M-series chips the 2 efficiency cores run at ~30-40% of perf-core
/// speed and become stragglers in compute-bound parallel work — the
/// work-stealing scheduler keeps assigning them tasks that hold up the perf
/// cores at synchronization barriers. Empirically, 8 threads beats 10 by
/// ~10-20% on `pcs::commit` and similar parallel-NTT workloads.
///
/// Call this **once** at program startup, before any other parallel flock
/// code runs (rayon's global pool is set on first use; if it's already
/// created, this call is a no-op).
///
/// Respects `RAYON_NUM_THREADS` — if that env var is set, this function
/// does nothing (so explicit user configuration always wins).
///
/// Returns the number of threads the pool was configured with, or `None`
/// if no change was made (either because the env var was set or because
/// rayon was already initialized).
pub fn init_perf_thread_pool() -> Option<usize> {
    // Before any worker thread (and its malloc arena) exists.
    pin_glibc_heap_thresholds();
    let built = init_global_pool();
    // Process start, on the (unpinned) main thread, after the global pool
    // exists. This is the only place the zerocheck round-1 half-pools may be
    // built: see `smt_split`.
    smt_split::init();
    built
}

/// Pin glibc's heap thresholds at process start: `M_MMAP_THRESHOLD` = 32 MiB
/// (the ceiling glibc's own dynamic threshold can reach, `HEAP_MAX_SIZE / 2`)
/// and `M_TRIM_THRESHOLD` = 1 GiB.
///
/// Mid-size (128 KiB – 32 MiB) prover transients that the typed scratch pools
/// and the recycling allocator do not cover go to glibc. Under the default
/// dynamic regime they are `mmap`/`munmap` pairs until a large enough free
/// has raised the threshold, and afterwards heap chunks whose arena top can
/// still be trimmed and re-faulted between proves. Pinning both keeps them
/// on arena heaps whose pages stay resident across proves.
///
/// ORDER MATTERS: any `mallopt` of either parameter disables the dynamic
/// threshold. Setting only the trim threshold freezes the mmap threshold at
/// its 128 KiB default and turns every mid-size buffer into an mmap/munmap
/// pair (measured worse, 02c77fd1). So the trim threshold is set only after
/// the mmap threshold was accepted. `FLOCK_NO_MALLOPT=1` skips both. Pure
/// allocator policy: no proof byte depends on it.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn pin_glibc_heap_thresholds() {
    unsafe extern "C" {
        fn mallopt(param: i32, value: i32) -> i32;
    }
    const M_TRIM_THRESHOLD: i32 = -1;
    const M_MMAP_THRESHOLD: i32 = -3;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("FLOCK_NO_MALLOPT").is_some() {
            return;
        }
        // SAFETY: plain glibc tunable setters; they take the main arena lock
        // and have no memory-safety preconditions.
        unsafe {
            if mallopt(M_MMAP_THRESHOLD, 32 << 20) == 1 {
                mallopt(M_TRIM_THRESHOLD, 1 << 30);
            }
        }
    });
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn pin_glibc_heap_thresholds() {}

fn init_global_pool() -> Option<usize> {
    if let Some(n) = topology_pool::try_build_global() {
        return Some(n);
    }
    if std::env::var("RAYON_NUM_THREADS").is_ok() {
        return None;
    }
    let n = perf_core_count();
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
    {
        Ok(()) => Some(n),
        Err(_) => None, // pool already built
    }
}

/// Topology-aware global rayon pool for the ranked x86 worker (Linux only).
///
/// The ranked runner exposes 16 vCPUs = 8 physical cores × 2 SMT threads and
/// the harness sets `RAYON_NUM_THREADS=16`, so rayon spins 16 workers that the
/// guest scheduler is free to migrate and to double up on one core while a
/// sibling core idles. This builds the global pool explicitly and pins worker
/// `i` to one logical CPU chosen from `/sys/devices/system/cpu/cpu*/topology/
/// thread_siblings_list`: mode `pin16` (default here) keeps the harness's
/// thread count but gives every worker a fixed logical CPU, alternating
/// physical cores first (worker 0 → core 0 sibling 0, worker 1 → core 1
/// sibling 0, … worker 8 → core 0 sibling 1, …), so a partially-idle pool
/// still spreads across cores; mode `phys8` uses one worker per physical
/// core. Selected at build time by `POOL_MODE`; `FLOCK_NO_TOPOLOGY_POOL=1`
/// disables it (rayon then builds its default pool from `RAYON_NUM_THREADS`).
/// Engages only when the machine has exactly 2 SMT threads per core and
/// `RAYON_NUM_THREADS` equals the logical CPU count (i.e. the ranked shape);
/// anything else falls through to the incumbent behaviour. Pure scheduling:
/// no arithmetic changes, proof bytes identical.
pub(crate) mod topology_pool {
    /// "pin16" | "phys8"
    pub(crate) const POOL_MODE: &str = "pin16";

    #[cfg(target_os = "linux")]
    fn sibling_groups() -> Option<Vec<Vec<usize>>> {
        let n = std::thread::available_parallelism().ok()?.get();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut seen = vec![false; n];
        for cpu in 0..n {
            if seen[cpu] {
                continue;
            }
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
            let list = std::fs::read_to_string(path).ok()?;
            let mut g = Vec::new();
            for part in list.trim().split(',') {
                if let Some((a, b)) = part.split_once('-') {
                    let (a, b): (usize, usize) = (a.parse().ok()?, b.parse().ok()?);
                    g.extend(a..=b);
                } else {
                    g.push(part.parse().ok()?);
                }
            }
            g.retain(|&c| c < n);
            g.sort_unstable();
            g.dedup();
            for &c in &g {
                seen[c] = true;
            }
            groups.push(g);
        }
        Some(groups)
    }

    #[cfg(target_os = "linux")]
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }

    #[cfg(target_os = "linux")]
    fn pin_to(cpu: usize) {
        let mut mask = [0u64; 16]; // 1024 CPUs
        if cpu >= 1024 {
            return;
        }
        mask[cpu / 64] |= 1u64 << (cpu % 64);
        // SAFETY: plain syscall on the calling thread with a valid mask buffer;
        // failure is ignored (pure hint).
        unsafe {
            let _ = sched_setaffinity(0, core::mem::size_of_val(&mask), mask.as_ptr());
        }
    }

    pub(crate) fn try_build_global() -> Option<usize> {
        #[cfg(not(target_os = "linux"))]
        {
            return None;
        }
        #[cfg(target_os = "linux")]
        {
            if std::env::var_os("FLOCK_NO_TOPOLOGY_POOL").is_some() {
                return None;
            }
            let want: usize = std::env::var("RAYON_NUM_THREADS").ok()?.parse().ok()?;
            let logical = std::thread::available_parallelism().ok()?.get();
            if want != logical {
                return None;
            }
            let groups = sibling_groups()?;
            let cores = groups.len();
            if cores == 0 || logical != 2 * cores || groups.iter().any(|g| g.len() != 2) {
                return None;
            }
            // Worker → CPU map: cores first (sibling 0 of each core), then the
            // second siblings; phys8 uses only the first half.
            let mut order: Vec<usize> = Vec::with_capacity(logical);
            for s in 0..2 {
                for g in &groups {
                    order.push(g[s]);
                }
            }
            let n = if POOL_MODE == "phys8" { cores } else { logical };
            let order = std::sync::Arc::new(order);
            let o2 = order.clone();
            match rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .stack_size(super::RANKED_WORKER_STACK)
                .start_handler(move |i| {
                    crate::hot_pool::record_top();
                    if let Some(&cpu) = o2.get(i) {
                        pin_to(cpu);
                    }
                })
                .build_global()
            {
                Ok(()) => {
                    super::RANKED_POOL_BUILT.store(true, std::sync::atomic::Ordering::Release);
                    Some(n)
                }
                Err(_) => None,
            }
        }
    }
}

/// Stack size of the ranked topology pool's workers. Large enough that a
/// whole prove may run on a worker (the ranked seed-pipe body does, see
/// `flock-prover/src/seed_pipe.rs`); the main thread that runs the untimed
/// warm-up proves has 8 MiB. Virtual reservation only.
pub const RANKED_WORKER_STACK: usize = 32 << 20;

static RANKED_POOL_BUILT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

std::thread_local! {
    static POOL_OWNER: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Mark the calling pool worker as the *owner* of the global pool: it runs a
/// whole prove at the top of its stack with no enclosing parallel region, so
/// the whole-pool `rayon::broadcast` pipelines (which otherwise require a
/// caller outside the pool) may engage from it. Set and cleared by the ranked
/// seed-pipe body only.
pub fn set_pool_owner(on: bool) {
    POOL_OWNER.with(|c| c.set(on));
}

/// Is the caller the designated pool owner (see [`set_pool_owner`])?
#[inline]
pub fn is_pool_owner() -> bool {
    POOL_OWNER.with(|c| c.get())
}

/// Is the caller outside the pool, or its designated owner (see
/// [`set_pool_owner`])?
#[inline]
pub fn owns_whole_pool() -> bool {
    rayon::current_thread_index().is_none() || POOL_OWNER.with(|c| c.get())
}

/// True once the ranked topology pool (workers with [`RANKED_WORKER_STACK`]
/// stacks) is the global pool.
pub fn ranked_pool_built() -> bool {
    RANKED_POOL_BUILT.load(std::sync::atomic::Ordering::Acquire)
}

/// Per-core SMT arm partition for zerocheck round 1 (Linux/x86-64).
///
/// Round 1 runs two independent halves — the AB shift-reduce and the
/// identity-C block-major fold — as `rayon::join`. The incumbent spreads
/// BOTH halves' inner `par_iter`s over the same 16 pinned workers, so each
/// physical core ends up running whichever chunks the work-stealer handed
/// it: sometimes AB next to AB, sometimes C next to C.
///
/// Measured on this machine (`zc1-notes/FINDINGS.md`, 144 value-valid runs):
/// the per-task cost of a co-resident SMT sibling, normalised against the
/// same task with the sibling idle, is **1.66** for AB beside AB and
/// **1.85** for C beside C (sum 3.50), against **1.69 / 1.64** for the
/// heterogeneous pairing (sum 3.33). The two kernels stress different parts
/// of a core, so pairing one of each on a core is ~5% cheaper than pairing
/// like with like. Giving each half its own 8-thread pool, pinned one thread
/// per physical core to opposite SMT siblings, makes every core run exactly
/// one AB thread beside one C thread.
///
/// The pools are built **here, at process start, on the unpinned main
/// thread**, from an explicit CPU list read out of
/// `/sys/devices/system/cpu/*/topology/thread_siblings_list`. They must
/// never be built lazily on a rayon worker: those are pinned to one logical
/// CPU, and `std::thread::available_parallelism()` honours the calling
/// thread's affinity mask, so asking it there returns 1 and both "8-thread"
/// pools silently degenerate to a single thread (this happened, and the
/// resulting 4-8x regression read as a clean refutation of the mechanism).
/// Nothing on this path calls `available_parallelism`.
///
/// Rollback is compile-time, because `run_trial` calls `env_clear()` and no
/// `FLOCK_*` variable reaches the ranked worker: set
/// [`ZC_R1_SMT_SPLIT`] to `false` and the incumbent `rayon::join` is
/// restored and no extra thread is spawned. Environment overrides are local
/// diagnostics only and are absent on the ranked runner, so the shipped
/// default is the split. Grouped AB leaves reduce the AB arm's work, so the
/// eight-core ranked host assigns seven workers to AB and nine to C. Local
/// diagnostics select an explicit AB count with `FLOCK_ZC_R1_AB_WORKERS` or
/// restore 9/7 with `FLOCK_ZC_R1_AB_REBALANCE=1`.
pub(crate) mod smt_split {
    use rayon::ThreadPool;
    use std::sync::OnceLock;

    /// Compile-time rollback. `false` restores the incumbent schedule and
    /// builds no pools at all.
    pub(crate) const ZC_R1_SMT_SPLIT: bool = true;

    /// Compile-time rollback of the eight-core 9/7 arm partition.
    const ZC_R1_AB_REBALANCE: bool = false;

    static POOLS: OnceLock<Option<(ThreadPool, ThreadPool)>> = OnceLock::new();

    // Separate topology planning from thread creation so disjointness and
    // the eight-core rebalance can be checked without spawning a pool.
    #[cfg(any(test, all(target_os = "linux", target_arch = "x86_64")))]
    fn arm_cpus(groups: &[Vec<usize>], rebalance: bool) -> (Vec<usize>, Vec<usize>) {
        let mut ab: Vec<usize> = groups.iter().map(|g| g[0]).collect();
        let mut c: Vec<usize> = groups.iter().map(|g| g[1]).collect();
        if rebalance && groups.len() == 8 {
            // Keep heterogeneous pairs on seven cores. The last core's two
            // siblings both serve the now-critical AB arm.
            ab.push(c.pop().expect("eight sibling groups"));
        }
        (ab, c)
    }

    // Keep the original planner as the exact eight/eight and nine/seven
    // controls. Only the already-validated eight-core production topology
    // selects an explicit partition.
    #[cfg(any(test, all(target_os = "linux", target_arch = "x86_64")))]
    fn arm_cpus_for_workers(groups: &[Vec<usize>], ab_workers: usize) -> (Vec<usize>, Vec<usize>) {
        assert!(ab_workers > 0 && ab_workers < 2 * groups.len());
        let (mut ab, mut c) = arm_cpus(groups, false);
        while ab.len() < ab_workers { ab.push(c.pop().unwrap()); }
        while ab.len() > ab_workers { c.push(ab.pop().unwrap()); }
        (ab, c)
    }

    #[cfg(test)]
    #[test]
    fn explicit_partitions_preserve_every_cpu_once() {
        let groups: Vec<Vec<usize>> = (0..8).map(|i| vec![100 + 3 * i, 101 + 3 * i]).collect();
        let mut expected: Vec<usize> = groups.iter().flatten().copied().collect();
        expected.sort_unstable();
        for workers in 1..16 {
            let (ab, c) = arm_cpus_for_workers(&groups, workers);
            assert_eq!(ab.len(), workers);
            assert_eq!(c.len(), 16 - workers);
            let mut actual = ab;
            actual.extend(c);
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn arm_partition_rebalances_eight_cores_without_losing_or_sharing_cpus() {
            for cores in [2usize, 4, 8, 16] {
                let groups: Vec<Vec<usize>> = (0..cores).map(|i| vec![i, i + cores]).collect();
                for rebalance in [false, true] {
                    let (ab, c) = super::arm_cpus(&groups, rebalance);
                    let moved = usize::from(rebalance && cores == 8);
                    assert_eq!((ab.len(), c.len()), (cores + moved, cores - moved));
                    let mut all: Vec<usize> = ab.iter().chain(&c).copied().collect();
                    all.sort_unstable();
                    assert_eq!(all, (0..2 * cores).collect::<Vec<_>>());
                    assert_eq!(&ab[..cores], &(0..cores).collect::<Vec<_>>());
                    assert_eq!(c, (cores..2 * cores - moved).collect::<Vec<_>>());
                    if moved != 0 { assert_eq!(ab.last(), Some(&(2 * cores - 1))); }
                }
            }
        }
    }

    /// The two arm pools, or `None` when this process is not the ranked
    /// shape (or the mechanism is off). Never builds anything: if [`init`]
    /// has not run, callers get the incumbent schedule.
    #[inline]
    pub(crate) fn zc_r1_pools() -> Option<&'static (ThreadPool, ThreadPool)> {
        POOLS.get().and_then(Option::as_ref)
    }

    /// Build the arm pools. Call ONCE, at process start, from an UNPINNED
    /// thread, and only after the global rayon pool exists.
    pub(crate) fn init() {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            if !ZC_R1_SMT_SPLIT {
                return;
            }
            let _ = POOLS.set(imp::build());
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    mod imp {
        use rayon::ThreadPool;

        unsafe extern "C" {
            fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
            fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u64) -> i32;
        }

        type Mask = [u64; 16]; // 1024 CPUs

        fn affinity() -> Mask {
            let mut m: Mask = [0; 16];
            // SAFETY: `m` is exactly the 1024-bit buffer the kernel is told about.
            unsafe {
                sched_getaffinity(0, core::mem::size_of::<Mask>(), m.as_mut_ptr());
            }
            m
        }

        fn pin_to(cpu: usize) {
            if cpu >= 1024 {
                return;
            }
            let mut mask: Mask = [0; 16];
            mask[cpu / 64] |= 1u64 << (cpu % 64);
            // SAFETY: plain syscall on the calling thread with a valid mask
            // buffer; failure is ignored (pure scheduling hint).
            unsafe {
                let _ = sched_setaffinity(0, core::mem::size_of_val(&mask), mask.as_ptr());
            }
        }

        /// Logical CPU count from `/sys/devices/system/cpu/present`.
        /// Deliberately NOT `available_parallelism()` — see the module doc.
        fn present_cpus() -> usize {
            let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/present") else {
                return 0;
            };
            let mut hi = 0usize;
            for part in s.trim().split(',') {
                let last = part.split('-').next_back().unwrap_or("");
                if let Ok(v) = last.parse::<usize>() {
                    hi = hi.max(v);
                }
            }
            hi + 1
        }

        /// One ascending group per physical core, from `thread_siblings_list`.
        fn sibling_groups() -> Vec<Vec<usize>> {
            let n = present_cpus();
            if n == 0 {
                return Vec::new();
            }
            let mut groups: Vec<Vec<usize>> = Vec::new();
            let mut seen = vec![false; n];
            for cpu in 0..n {
                if seen[cpu] {
                    continue;
                }
                let path =
                    format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
                let Ok(list) = std::fs::read_to_string(path) else {
                    return Vec::new();
                };
                let mut g = Vec::new();
                for part in list.trim().split(',') {
                    if let Some((a, b)) = part.split_once('-') {
                        let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) else {
                            return Vec::new();
                        };
                        g.extend(a..=b);
                    } else if let Ok(v) = part.parse::<usize>() {
                        g.push(v);
                    } else {
                        return Vec::new();
                    }
                }
                g.retain(|&c| c < n);
                g.sort_unstable();
                g.dedup();
                for &c in &g {
                    seen[c] = true;
                }
                groups.push(g);
            }
            groups
        }

        fn half(sibling: usize, cpus: Vec<usize>, debug: bool) -> ThreadPool {
            if debug {
                eprintln!("[smt_split] pool sibling{sibling} cpus {cpus:?}");
            }
            let n = cpus.len();
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(move |i| format!("zcr1s{sibling}-{i}"))
                .start_handler(move |i| {
                    if let Some(&c) = cpus.get(i) {
                        pin_to(c);
                    }
                })
                .build()
                .expect("zc_r1 half pool")
        }

        pub(super) fn build() -> Option<(ThreadPool, ThreadPool)> {
            let debug = std::env::var_os("FLOCK_SMT_SPLIT_DEBUG").is_some();
            if std::env::var_os("FLOCK_NO_ZC_R1_SMT_SPLIT").is_some() {
                if debug {
                    eprintln!("[smt_split] disabled by FLOCK_NO_ZC_R1_SMT_SPLIT");
                }
                return None;
            }
            let groups = sibling_groups();
            let cores = groups.len();
            // Ranked shape only: every core has exactly two SMT siblings, the
            // machine is fully described, and the global pool already owns one
            // worker per logical CPU (so the split reassigns workers rather
            // than adding oversubscription during the window itself).
            if cores < 2
                || groups.iter().any(|g| g.len() != 2)
                || present_cpus() != 2 * cores
                || rayon::current_num_threads() != 2 * cores
            {
                if debug {
                    eprintln!(
                        "[smt_split] declined: cores={cores} present={} rayon={}",
                        present_cpus(),
                        rayon::current_num_threads()
                    );
                }
                return None;
            }
            // Every CPU we would pin to must be inside this process's own
            // affinity set, or the pin silently lands the thread somewhere the
            // cpuset forbids.
            let m = affinity();
            for g in &groups {
                for &c in g {
                    if c >= 1024 || m[c / 64] & (1u64 << (c % 64)) == 0 {
                        if debug {
                            eprintln!("[smt_split] declined: cpu {c} outside affinity mask");
                        }
                        return None;
                    }
                }
            }
            let rebalance = (super::ZC_R1_AB_REBALANCE
                || std::env::var_os("FLOCK_ZC_R1_AB_REBALANCE").is_some())
                && std::env::var_os("FLOCK_NO_ZC_R1_AB_REBALANCE").is_none();
            let (s0, s1) = super::arm_cpus(&groups, rebalance);
            let (s0, s1) = if cores == 8 {
                match std::env::var("FLOCK_ZC_R1_AB_WORKERS").ok()
                    .and_then(|s| s.parse::<usize>().ok()).filter(|n| (1..16).contains(n)) {
                    Some(workers) => super::arm_cpus_for_workers(&groups, workers),
                    None if rebalance => (s0, s1),
                    None => super::arm_cpus_for_workers(&groups, 7),
                }
            } else { (s0, s1) };
            let p0 = half(0, s0, debug);
            let p1 = half(1, s1, debug);
            if std::env::var_os("FLOCK_ZC_R1_SMT_SPLIT_IDLE").is_some() {
                // Diagnostics: pools are built (and their threads spawned and
                // pinned) but never installed on, isolating the standing cost
                // of the extra threads from the value of the partition.
                if debug {
                    eprintln!("[smt_split] built but IDLE (FLOCK_ZC_R1_SMT_SPLIT_IDLE)");
                }
                // Leak them: `ThreadPool`'s `Drop` terminates its workers, and
                // this arm exists precisely to keep the extra threads alive.
                core::mem::forget(p0);
                core::mem::forget(p1);
                return None;
            }
            Some((p0, p1))
        }
    }
}

/// Best-effort `madvise(MADV_HUGEPAGE)` for multi-MB buffers. Ubuntu ships
/// THP in `madvise` mode, so without this hint the prover's 32 MB-1 GB
/// working buffers sit on 4 KiB pages through every strided sweep. Advising
/// at allocation time means the setup-phase prewarm faults the pool's pages
/// as 2 MiB pages once, before any timed proof. Raw syscall (no libc dep);
/// errors ignored. `FLOCK_NO_HUGEPAGES` is a local-diagnostics kill switch.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn advise_hugepages(ptr: *mut u8, bytes: usize) {
    const HUGE: usize = 1 << 21;
    if bytes < HUGE {
        return;
    }
    static DISABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_HUGEPAGES").is_some());
    if *DISABLED {
        return;
    }
    const PAGE: usize = 4096;
    let start = (ptr as usize).next_multiple_of(PAGE);
    let end = ptr as usize + bytes;
    if end <= start {
        return;
    }
    const SYS_MADVISE: usize = 28;
    const MADV_HUGEPAGE: usize = 14;
    // SAFETY: the advised range lies within the just-allocated buffer, and
    // MADV_HUGEPAGE never alters contents or mapping validity.
    unsafe {
        let ret: isize;
        core::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MADVISE as isize => ret,
            in("rdi") start,
            in("rsi") end - start,
            in("rdx") MADV_HUGEPAGE,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
        let _ = ret;
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
/// Prewarm marker ab
fn advise_hugepages(_ptr: *mut u8, _bytes: usize) {}

/// Round one's two arms run in the SMT-split pools while the calling worker waits for both.
///
/// `rayon::join(|| ab_pool.install(..), || c_pool.install(..))` leaves that worker in rayon's
/// sleeping wait. While the AB arm's thread shares its CPU it stays awake (each `sched_yield`
/// hands the CPU back), but once the shorter AB arm has ended its idle rounds run out and it is
/// asleep, its core halted, when the longer C arm finishes; when the join's second closure was
/// taken by another global worker the end of round one is even relayed through two such
/// wake-ups, and the C arm only starts once that worker has been woken.
///
/// Here the caller hands both closures to their pools itself (C first). It waits for the AB arm
/// with rayon's own wait, exactly as before, and only from the AB arm's end until the C arm's end
/// (about 0.4 ms, when no arm thread uses its CPU) does it stay on CPU with a bounded
/// `sched_yield` + PAUSE spin. The closures, their pools and their inputs are the same, so proof
/// bytes are unchanged. `FLOCK_NO_R1_OWNER_AWAKE=1` restores the join.
///
/// Second, separate switch (`FLOCK_NO_R1_EXIT_COREWARM=1`): when the AB arm's closure ends, the AB
/// pool's own threads (first hyperthread of cores 0-6) have nothing left to do and would go to
/// sleep; with the C threads on the sibling hyperthreads finishing shortly after, those cores are
/// fully halted when round two fans out, and its wake-ups pay the idle state's exit latency. The
/// AB pool's threads instead wait in a PAUSE loop (no system calls, no global-pool thread is woken,
/// nothing runs on a CPU the C arm uses) until the caller releases them at the end of round one,
/// bounded at 1.5 ms. The thread that shares the caller's CPU is left out.
pub mod r1_owner_awake {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::LazyLock;
    use std::time::{Duration, Instant};

    /// After this long the caller stops spinning and rayon's sleeping wait takes over.
    const SPIN_LIMIT: Duration = Duration::from_millis(5);

    pub fn enabled() -> bool {
        static ON: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("FLOCK_NO_R1_OWNER_AWAKE").is_none());
        *ON
    }

    /// Bumped by the caller when round one's last arm has ended; releases the holders.
    static HOLD_EPOCH: AtomicU64 = AtomicU64::new(0);
    const HOLD_LIMIT: Duration = Duration::from_micros(1500);

    pub fn corewarm_enabled() -> bool {
        static ON: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("FLOCK_NO_R1_EXIT_COREWARM").is_none());
        *ON
    }

    fn hold(epoch: u64) {
        hold_for(epoch, HOLD_LIMIT, false);
    }

    /// The holder loop: stay on CPU until [`HOLD_EPOCH`] moves past `epoch` or
    /// `limit` passes. `yielding` adds a `sched_yield` per lap so a holder never
    /// keeps another runnable thread (one still finishing the commit, or the
    /// serial path into zerocheck) off its CPU.
    fn hold_for(epoch: u64, limit: Duration, yielding: bool) {
        let t0 = Instant::now();
        while HOLD_EPOCH.load(Ordering::Relaxed) == epoch {
            side_poll();
            if yielding {
                std::thread::yield_now();
            }
            for _ in 0..32 {
                core::hint::spin_loop();
            }
            if t0.elapsed() > limit {
                break;
            }
        }
    }

    // ---- Lane rl item 3: round-one ENTRY pre-wake ----
    //
    // Round one's two SMT-split pools are asleep when the round starts, and
    // their threads come on CPU 0.10-0.15 ms after the work is injected. The
    // same holder as the exit core-warm is broadcast to both pools while the
    // commit's last deep-pass workers finish (`prewake_arrive`), so the
    // threads are already on CPU, yielding, when round one injects its two
    // arms; `run_both` releases them (one `HOLD_EPOCH` bump) right after both
    // arms are injected. Same closures, pools and inputs: proof bytes are
    // unchanged. `FLOCK_NO_RL_PREWAKE=1` disables it; `FLOCK_RL_PREWAKE_AT=k`
    // fires it when the k-th deep-pass worker has finished (default 4 of 16:
    // measured best of 1/4/12/16; the holders yield, so the twelve workers still
    // in the deep pass keep their CPUs).

    const PREWAKE_LIMIT: Duration = Duration::from_micros(1500);
    static PREWAKE_ARRIVED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static PREWAKE_LIVE: AtomicBool = AtomicBool::new(false);

    /// 0 = off, else the deep-pass arrival count that fires the pre-wake.
    /// Read once per process; callers resolve it once per pass.
    pub fn prewake_at() -> usize {
        static AT: LazyLock<usize> = LazyLock::new(|| {
            if std::env::var_os("FLOCK_NO_RL_PREWAKE").is_some() || !enabled() {
                return 0;
            }
            std::env::var("FLOCK_RL_PREWAKE_AT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&k| k >= 1)
                .unwrap_or(4)
        });
        *AT
    }

    /// Start of a commit pass that will report its deep-pass arrivals.
    pub fn prewake_reset() {
        PREWAKE_ARRIVED.store(0, Ordering::Relaxed);
    }

    /// One deep-pass worker has finished; the `at`-th arrival fires the pre-wake.
    pub fn prewake_arrive(at: usize) {
        if PREWAKE_ARRIVED.fetch_add(1, Ordering::AcqRel) + 1 == at {
            prewake();
        }
    }

    fn prewake() {
        let Some((ab_pool, c_pool)) = crate::smt_split::zc_r1_pools() else {
            return;
        };
        let epoch = HOLD_EPOCH.load(Ordering::Acquire);
        PREWAKE_LIVE.store(true, Ordering::Release);
        for pool in [ab_pool, c_pool] {
            pool.spawn_broadcast(move |_| hold_for(epoch, PREWAKE_LIMIT, true));
        }
    }

    // ---- Lane gl1: side jobs for round one's C-arm tail ----
    //
    // After its fold, the C arm's tail runs serially on one thread while the
    // caller (yield-spinning on the C arm) and the AB pool's core-warm
    // holders (PAUSE loop) are on CPU with nothing to do. `side_run` posts up
    // to two independent closures into fixed slots that those waiting loops
    // poll; whichever claims a slot first runs it. The poster runs its own
    // part, then takes back any slot nobody claimed and waits for the rest,
    // so the result is the same closures' values, only placed on other CPUs.
    // `FLOCK_NO_GL1_SIDE=1` runs all three in order on the poster.
    const SIDES: usize = 2;
    static SIDE_BODY: [AtomicU64; SIDES] = [AtomicU64::new(0), AtomicU64::new(0)];
    static SIDE_STATE: [std::sync::atomic::AtomicU32; SIDES] =
        [std::sync::atomic::AtomicU32::new(0), std::sync::atomic::AtomicU32::new(0)];
    const S_IDLE: u32 = 0;
    const S_POSTED: u32 = 1;
    const S_CLAIMED: u32 = 2;
    const S_DONE: u32 = 3;

    pub fn side_enabled() -> bool {
        static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FLOCK_NO_GL1_SIDE").is_none());
        *ON
    }

    #[inline]
    fn side_poll() {
        for i in 0..SIDES {
            if SIDE_STATE[i].load(Ordering::Acquire) == S_POSTED
                && SIDE_STATE[i]
                    .compare_exchange(S_POSTED, S_CLAIMED, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                // SAFETY: the poster keeps the body alive until it sees S_DONE.
                let body = unsafe { &**(SIDE_BODY[i].load(Ordering::Acquire) as usize as *const &(dyn Fn() + Sync)) };
                body();
                SIDE_STATE[i].store(S_DONE, Ordering::Release);
            }
        }
    }

    /// Run `main` here and `a`, `b` on whichever waiting round-one thread
    /// claims them first (or here, if none does). Single poster at a time.
    pub fn side_run<M, A, B, RM, RA, RB>(main: M, a: A, b: B) -> (RM, RA, RB)
    where
        M: FnOnce() -> RM,
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let a_in = std::sync::Mutex::new(Some(a));
        let b_in = std::sync::Mutex::new(Some(b));
        let a_out: std::sync::Mutex<Option<RA>> = std::sync::Mutex::new(None);
        let b_out: std::sync::Mutex<Option<RB>> = std::sync::Mutex::new(None);
        let ja = || {
            if let Some(f) = a_in.lock().unwrap().take() {
                let v = f();
                *a_out.lock().unwrap() = Some(v);
            }
        };
        let jb = || {
            if let Some(f) = b_in.lock().unwrap().take() {
                let v = f();
                *b_out.lock().unwrap() = Some(v);
            }
        };
        let fa: &(dyn Fn() + Sync) = &ja;
        let fb: &(dyn Fn() + Sync) = &jb;
        let bodies = [&fa as *const &(dyn Fn() + Sync) as usize, &fb as *const &(dyn Fn() + Sync) as usize];
        for i in 0..SIDES {
            debug_assert_eq!(SIDE_STATE[i].load(Ordering::Relaxed), S_IDLE);
            SIDE_BODY[i].store(bodies[i] as u64, Ordering::Release);
            SIDE_STATE[i].store(S_POSTED, Ordering::Release);
        }
        let rm = main();
        for i in 0..SIDES {
            if SIDE_STATE[i]
                .compare_exchange(S_POSTED, S_IDLE, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                if i == 0 { ja() } else { jb() }
            } else {
                while SIDE_STATE[i].load(Ordering::Acquire) != S_DONE {
                    core::hint::spin_loop();
                }
                SIDE_STATE[i].store(S_IDLE, Ordering::Release);
            }
        }
        (
            rm,
            a_out.into_inner().unwrap().expect("side job a ran"),
            b_out.into_inner().unwrap().expect("side job b ran"),
        )
    }

    // ---- Lane v1: round-one EXIT without two sleeping waits ----
    //
    // The caller (pinned to the CPU of AB worker 0) used to block inside
    // `ab_pool.in_place_scope` until the AB arm finished, and the AB arm's own
    // `rayon::broadcast` caller slept until the last band ended. Each wake-up
    // measured 0.15-0.30 ms in about a third of the proves. With the switch on,
    // the first AB thread that finds the band counter empty unparks the caller,
    // which then yield-spins on `AB_DONE`; the broadcast caller spin-waits for
    // the other AB threads (`ab_claim_spin`). Same closures, same inputs, same
    // result order: proof bytes are unchanged. `FLOCK_NO_V1_R1_SPIN=1` restores
    // the sleeping waits.
    pub fn v1_spin_enabled() -> bool {
        static ON: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("FLOCK_NO_V1_R1_SPIN").is_none());
        *ON
    }
    static AB_DRAIN: AtomicBool = AtomicBool::new(false);
    static AB_DONE: AtomicBool = AtomicBool::new(false);
    static OWNER: std::sync::Mutex<Option<std::thread::Thread>> = std::sync::Mutex::new(None);

    fn unpark_owner() {
        if let Some(t) = OWNER.lock().unwrap().as_ref() {
            t.unpark();
        }
    }

    /// An AB thread found the band counter exhausted (the arm's last bands are
    /// in flight): wake the caller so it is on CPU when the arm ends.
    pub fn ab_draining() {
        if !AB_DRAIN.swap(true, Ordering::AcqRel) {
            unpark_owner();
        }
    }

    struct Done<'a>(&'a AtomicBool);
    impl Drop for Done<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    pub fn run_both<A, B, RA, RB>(
        ab_pool: &rayon::ThreadPool,
        c_pool: &rayon::ThreadPool,
        ab: A,
        c: B,
    ) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let c_done = AtomicBool::new(false);
        let mut ra = None;
        let mut rb = None;
        let spin = v1_spin_enabled();
        if spin {
            AB_DRAIN.store(false, Ordering::Release);
            AB_DONE.store(false, Ordering::Release);
            *OWNER.lock().unwrap() = Some(std::thread::current());
        }
        // The AB pool's thread i runs on the same CPU as global worker i.
        let owner_idx = rayon::current_thread_index();
        {
            let (ra_slot, rb_slot, c_done_ref) = (&mut ra, &mut rb, &c_done);
            c_pool.in_place_scope(|sc| {
                // The longer arm first.
                sc.spawn(move |_| {
                    let _d = Done(c_done_ref);
                    *rb_slot = Some(c());
                });
                // rayon's own wait until the AB arm has finished.
                ab_pool.in_place_scope(|sa| {
                    sa.spawn(move |_| {
                        *ra_slot = Some(ab());
                        if corewarm_enabled() {
                            let epoch = HOLD_EPOCH.load(Ordering::Acquire);
                            ab_pool.spawn_broadcast(move |ctx| {
                                if Some(ctx.index()) != owner_idx {
                                    hold(epoch);
                                }
                            });
                        }
                        if spin {
                            AB_DONE.store(true, Ordering::Release);
                            unpark_owner();
                        }
                    });
                    // Both arms are injected: release any pre-woken holders
                    // so they pick the arms up without a wake-up.
                    if PREWAKE_LIVE.swap(false, Ordering::AcqRel) {
                        HOLD_EPOCH.fetch_add(1, Ordering::Release);
                    }
                    if spin {
                        let t0 = Instant::now();
                        while !AB_DONE.load(Ordering::Acquire) {
                            if AB_DRAIN.load(Ordering::Acquire) {
                                std::thread::yield_now();
                                for _ in 0..32 {
                                    core::hint::spin_loop();
                                }
                            } else {
                                std::thread::park_timeout(Duration::from_millis(2));
                            }
                            if t0.elapsed() > Duration::from_millis(50) {
                                break;
                            }
                        }
                    }
                });
                let t0 = Instant::now();
                while !c_done_ref.load(Ordering::Acquire) {
                    side_poll();
                    std::thread::yield_now();
                    for _ in 0..32 {
                        core::hint::spin_loop();
                    }
                    if t0.elapsed() > SPIN_LIMIT {
                        break;
                    }
                }
                HOLD_EPOCH.fetch_add(1, Ordering::Release);
            });
        }
        if spin {
            *OWNER.lock().unwrap() = None;
        }
        (
            ra.expect("round-one AB arm finished"),
            rb.expect("round-one C arm finished"),
        )
    }
}

/// Lane ru: flat claim-counter parallel map for short parallel passes.
///
/// `rayon`'s indexed `par_iter().map().reduce()` ends in a join tree: the
/// worker that finishes the last leaf sets a latch whose owner, having run
/// out of work long before, is usually parked in rayon's sleep and has to be
/// woken (futex + idle exit), often more than once up the tree. Traced at the
/// zerocheck tail's large levels that hand-off took 40-65 µs after the last
/// chunk ended. Here the caller spawns one claimer per other worker, claims
/// items itself, and then spins (no sleep) until every item is done; late
/// claimers that start after the counter is exhausted exit without touching
/// the borrowed closure. Each claimer folds its items into a private
/// accumulator (a per-item slot array read back serially from other cores'
/// caches cost 0.5 ms at 8192 items), so the result is the same sum of the
/// same per-item values in another grouping.
pub mod flat_par {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    struct Shared {
        body: usize, // thin pointer to a `&(dyn Fn() + Sync)`
        active: AtomicUsize,
        closed: AtomicBool,
    }

    fn enter(sh: &Shared) {
        sh.active.fetch_add(1, Ordering::SeqCst);
        if !sh.closed.load(Ordering::SeqCst) {
            // SAFETY: `map_reduce` keeps the body alive until `closed` is set
            // and `active` has drained; this claimer incremented `active` first.
            let body = unsafe { &**(sh.body as *const &(dyn Fn() + Sync)) };
            body();
        }
        sh.active.fetch_sub(1, Ordering::SeqCst);
    }

    /// Lane cmb1 (after mpjunior92 a6571e3f): each claimer publishes its
    /// completed item count once, after its last run, instead of one
    /// `done` RMW per claimed run. `FLOCK_NO_CMB_DONE1=1` restores per-run.
    #[inline]
    fn done_once() -> bool {
        static ON: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_CMB_DONE1").is_none());
        *ON
    }

    /// `FLOCK_NO_RU_FLAT_PAR=1` makes callers keep their rayon route.
    pub fn enabled() -> bool {
        static ON: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_RU_FLAT_PAR").is_none());
        *ON
    }

    /// `op`-reduction of `f(0..n)`: claimers (the caller plus one spawned per
    /// other worker of the current pool) take items from one counter and fold
    /// them into a private accumulator; the caller folds the claimers'
    /// accumulators. For an associative and commutative `op` (F128 sums) the
    /// result equals any other grouping. Must be called from a pool worker.
    pub fn map_reduce<T, F>(
        n: usize,
        f: F,
        id: impl Fn() -> T + Sync,
        op: impl Fn(T, T) -> T + Sync,
    ) -> T
    where
        F: Fn(usize) -> T + Sync,
    {
        map_reduce_idle(n, f, id, op, || ())
    }

    /// [`map_reduce`] that runs `idle` on the caller once its own claims are
    /// exhausted, while the other claimers drain their last runs (lane gl1:
    /// serial post-work that does not depend on the result hides there).
    pub fn map_reduce_idle<T, F>(
        n: usize,
        f: F,
        id: impl Fn() -> T + Sync,
        op: impl Fn(T, T) -> T + Sync,
        idle: impl FnOnce(),
    ) -> T
    where
        F: Fn(usize) -> T + Sync,
    {
        if n == 0 {
            idle();
            return id();
        }
        let helpers = rayon::current_num_threads().saturating_sub(1).min(n - 1);
        let claimers = helpers + 1;
        let next = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        let who = AtomicUsize::new(0);
        let mut slots: Vec<core::mem::MaybeUninit<T>> = Vec::with_capacity(claimers);
        // SAFETY: only slots flagged in `written` are read.
        unsafe { slots.set_len(claimers) };
        let written: Vec<AtomicBool> = (0..claimers).map(|_| AtomicBool::new(false)).collect();
        let slot_addr = slots.as_mut_ptr() as usize;
        let agg = done_once();
        let body = || {
            let me = who.fetch_add(1, Ordering::Relaxed);
            if me >= claimers {
                return;
            }
            let mut acc = id();
            let mut completed = 0usize;
            // Guided claims: contiguous runs of remaining/(2·claimers) items
            // (rayon's first split hands each worker a contiguous ~n/T range;
            // single-item claims interleaved the passes' streams and cost
            // +1.0/+1.3 ms in round two / r345).
            let mut cur = next.load(Ordering::Relaxed);
            loop {
                if cur >= n {
                    break;
                }
                let take = ((n - cur) / (2 * claimers)).max(1);
                match next.compare_exchange_weak(cur, cur + take, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => {
                        for i in cur..cur + take {
                            acc = op(acc, f(i));
                        }
                        if agg {
                            completed += take;
                        } else {
                            done.fetch_add(take, Ordering::Release);
                        }
                        cur = next.load(Ordering::Relaxed);
                    }
                    Err(now) => cur = now,
                }
            }
            // SAFETY: slot `me` is owned by this claimer.
            unsafe { (slot_addr as *mut core::mem::MaybeUninit<T>).add(me).write(core::mem::MaybeUninit::new(acc)) };
            written[me].store(true, Ordering::Release);
            if completed != 0 {
                done.fetch_add(completed, Ordering::Release);
            }
        };
        let fat: &(dyn Fn() + Sync) = &body;
        let sh = Arc::new(Shared {
            body: &fat as *const &(dyn Fn() + Sync) as usize,
            active: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
        });
        for _ in 0..helpers {
            let s = sh.clone();
            rayon::spawn(move || enter(&s));
        }
        enter(&sh);
        idle();
        while done.load(Ordering::Acquire) < n {
            core::hint::spin_loop();
        }
        sh.closed.store(true, Ordering::SeqCst);
        while sh.active.load(Ordering::SeqCst) != 0 {
            core::hint::spin_loop();
        }
        let mut acc = id();
        for (slot, w) in slots.iter().zip(written.iter()) {
            if w.load(Ordering::Acquire) {
                // SAFETY: flagged slots were fully written before the flag.
                acc = op(acc, unsafe { slot.assume_init_read() });
            }
        }
        acc
    }

    // ---- Lane gl1: persistent spin team over the hot pool's spinners ----
    //
    // The small zerocheck tail levels (n18 .. n6) and other short parallel
    // regions are fork/join-bound: rayon's split/steal and join-tree hand-off
    // cost 5-30 µs per region against 1-20 µs of work. While a `HotPool` is
    // live its spinners already run on every other worker; each spinner lap
    // polls `TEAM_SEQ`, so the owner publishes a region with two stores and
    // the claim loop below starts on every spinning worker within a lap. The
    // owner waits for the region by spinning on the done count (no latch, no
    // sleep). Results are folded per claimer exactly as in `map_reduce`, so
    // for an associative and commutative `op` (F128 sums, XOR) the value is
    // identical. `FLOCK_NO_GL1_TEAM=1` keeps the incumbent routes.
    static TEAM_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static TEAM_BODY: AtomicUsize = AtomicUsize::new(0);
    static TEAM_ACTIVE: AtomicUsize = AtomicUsize::new(0);
    pub(crate) static TEAM_OWNER: AtomicUsize = AtomicUsize::new(usize::MAX);

    /// Kill switch, read once per process; callers resolve it once per phase.
    pub fn team_enabled() -> bool {
        static ON: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GL1_TEAM").is_none());
        *ON
    }

    std::thread_local! {
        static TEAM_TOKEN: u8 = const { 0 };
    }

    /// This OS thread's identity for [`TEAM_OWNER`] (rayon indices are per
    /// pool, so an index alone would also match a worker of another pool).
    #[inline]
    pub(crate) fn team_token() -> usize {
        TEAM_TOKEN.with(|t| t as *const u8 as usize)
    }

    /// The calling thread owns a live hot pool (its spinners poll the team).
    #[inline]
    pub fn team_ready() -> bool {
        let o = TEAM_OWNER.load(Ordering::Relaxed);
        o != usize::MAX && o == team_token()
    }

    /// One spinner lap: join the live region if it is new to this spinner.
    #[inline]
    pub(crate) fn team_poll(last: &mut u64) {
        let s = TEAM_SEQ.load(Ordering::Acquire);
        if s & 1 == 0 || s == *last {
            return;
        }
        *last = s;
        TEAM_ACTIVE.fetch_add(1, Ordering::SeqCst);
        if TEAM_SEQ.load(Ordering::SeqCst) == s {
            // SAFETY: the owner keeps the body alive until it has closed the
            // region (`TEAM_SEQ` moved on) and seen `TEAM_ACTIVE` drain; this
            // helper raised `TEAM_ACTIVE` before re-checking the sequence.
            let body = unsafe { &**(TEAM_BODY.load(Ordering::Acquire) as *const &(dyn Fn() + Sync)) };
            body();
        }
        TEAM_ACTIVE.fetch_sub(1, Ordering::SeqCst);
    }

    /// Lane sl1 (`FLOCK_NO_SL1_AFF`): writer map of the last flat-claimed
    /// tail level (rayon index of the claimer that wrote each output chunk),
    /// so the next, cache-resident level can hand each chunk to the thread
    /// whose L2 holds it. Performance hint only: any item may still be run by
    /// any claimer, and the reduction is an XOR sum, so values are identical.
    pub const SL1_OWN_CAP: usize = 1024;
    pub static SL1_OWN: [core::sync::atomic::AtomicU8; SL1_OWN_CAP] =
        [const { core::sync::atomic::AtomicU8::new(u8::MAX) }; SL1_OWN_CAP];
    /// (a base address, b base address, chunk bytes, items) of the recorded level.
    pub static SL1_OWN_GEOM: std::sync::Mutex<(usize, usize, usize, usize)> =
        std::sync::Mutex::new((0, 0, 0, 0));

    pub fn sl1_aff_enabled() -> bool {
        static ON: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_SL1_AFF").is_none());
        *ON
    }

    /// [`team_map_reduce`] where claimer `t` first runs the items with
    /// `owner[i] == t` (rayon index), then sweeps every unclaimed item.
    pub fn team_map_reduce_aff<T, F>(
        n: usize,
        owner: &[u8],
        f: F,
        id: impl Fn() -> T + Sync,
        op: impl Fn(T, T) -> T + Sync,
    ) -> T
    where
        F: Fn(usize) -> T + Sync,
    {
        assert_eq!(owner.len(), n);
        if n <= 1 {
            return team_map_reduce(n, f, id, op);
        }
        const CAP: usize = 64;
        let claimers = rayon::current_num_threads().clamp(1, CAP);
        let claimed: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
        let done = AtomicUsize::new(0);
        let who = AtomicUsize::new(0);
        let mut slots: [core::mem::MaybeUninit<T>; CAP] = [const { core::mem::MaybeUninit::uninit() }; CAP];
        let written: [AtomicBool; CAP] = [const { AtomicBool::new(false) }; CAP];
        let slot_addr = slots.as_mut_ptr() as usize;
        let body = || {
            let me = who.fetch_add(1, Ordering::Relaxed);
            if me >= CAP || done.load(Ordering::Relaxed) >= n {
                return;
            }
            let tid = rayon::current_thread_index().map_or(u8::MAX - 1, |t| t.min(250) as u8);
            let mut acc = id();
            let mut completed = 0usize;
            for i in 0..n {
                if owner[i] == tid && !claimed[i].load(Ordering::Relaxed) && !claimed[i].swap(true, Ordering::AcqRel) {
                    acc = op(acc, f(i));
                    completed += 1;
                }
            }
            let start = me * n / claimers;
            for j in 0..n {
                let i = (start + j) % n;
                if !claimed[i].load(Ordering::Relaxed) && !claimed[i].swap(true, Ordering::AcqRel) {
                    acc = op(acc, f(i));
                    completed += 1;
                }
            }
            if completed != 0 {
                // SAFETY: slot `me` is owned by this claimer.
                unsafe { (slot_addr as *mut core::mem::MaybeUninit<T>).add(me).write(core::mem::MaybeUninit::new(acc)) };
                written[me].store(true, Ordering::Release);
                done.fetch_add(completed, Ordering::Release);
            }
        };
        let fat: &(dyn Fn() + Sync) = &body;
        let seq = TEAM_SEQ.load(Ordering::Relaxed);
        debug_assert!(seq & 1 == 0, "nested team region");
        TEAM_BODY.store(&fat as *const &(dyn Fn() + Sync) as usize, Ordering::Release);
        TEAM_SEQ.store(seq + 1, Ordering::SeqCst);
        body();
        while done.load(Ordering::Acquire) < n {
            core::hint::spin_loop();
        }
        TEAM_SEQ.store(seq + 2, Ordering::SeqCst);
        while TEAM_ACTIVE.load(Ordering::SeqCst) != 0 {
            core::hint::spin_loop();
        }
        let mut acc = id();
        for (slot, w) in slots.iter().zip(written.iter()) {
            if w.load(Ordering::Acquire) {
                // SAFETY: flagged slots were fully written before the flag.
                acc = op(acc, unsafe { slot.assume_init_read() });
            }
        }
        acc
    }

    /// [`map_reduce`] dispatched to the hot pool's spinning team instead of
    /// spawned claimers. Caller must satisfy [`team_ready`].
    pub fn team_map_reduce<T, F>(
        n: usize,
        f: F,
        id: impl Fn() -> T + Sync,
        op: impl Fn(T, T) -> T + Sync,
    ) -> T
    where
        F: Fn(usize) -> T + Sync,
    {
        if n == 0 {
            return id();
        }
        if n == 1 {
            return op(id(), f(0));
        }
        const CAP: usize = 64;
        let claimers = rayon::current_num_threads().clamp(1, CAP);
        let next = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        let who = AtomicUsize::new(0);
        let mut slots: [core::mem::MaybeUninit<T>; CAP] = [const { core::mem::MaybeUninit::uninit() }; CAP];
        let written: [AtomicBool; CAP] = [const { AtomicBool::new(false) }; CAP];
        let slot_addr = slots.as_mut_ptr() as usize;
        let agg = done_once();
        let body = || {
            let me = who.fetch_add(1, Ordering::Relaxed);
            if me >= CAP {
                return;
            }
            let mut cur = next.load(Ordering::Relaxed);
            if cur >= n {
                return;
            }
            let mut acc = id();
            let mut any = false;
            let mut completed = 0usize;
            loop {
                if cur >= n {
                    break;
                }
                let take = ((n - cur) / (2 * claimers)).max(1);
                match next.compare_exchange_weak(cur, cur + take, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => {
                        for i in cur..cur + take {
                            acc = op(acc, f(i));
                        }
                        any = true;
                        if agg {
                            completed += take;
                        } else {
                            done.fetch_add(take, Ordering::Release);
                        }
                        cur = next.load(Ordering::Relaxed);
                    }
                    Err(now) => cur = now,
                }
            }
            if any {
                // SAFETY: slot `me` is owned by this claimer.
                unsafe { (slot_addr as *mut core::mem::MaybeUninit<T>).add(me).write(core::mem::MaybeUninit::new(acc)) };
                written[me].store(true, Ordering::Release);
            }
            if completed != 0 {
                done.fetch_add(completed, Ordering::Release);
            }
        };
        let fat: &(dyn Fn() + Sync) = &body;
        let seq = TEAM_SEQ.load(Ordering::Relaxed);
        debug_assert!(seq & 1 == 0, "nested team region");
        TEAM_BODY.store(&fat as *const &(dyn Fn() + Sync) as usize, Ordering::Release);
        TEAM_SEQ.store(seq + 1, Ordering::SeqCst);
        body();
        while done.load(Ordering::Acquire) < n {
            core::hint::spin_loop();
        }
        TEAM_SEQ.store(seq + 2, Ordering::SeqCst);
        while TEAM_ACTIVE.load(Ordering::SeqCst) != 0 {
            core::hint::spin_loop();
        }
        let mut acc = id();
        for (slot, w) in slots.iter().zip(written.iter()) {
            if w.load(Ordering::Acquire) {
                // SAFETY: flagged slots were fully written before the flag.
                acc = op(acc, unsafe { slot.assume_init_read() });
            }
        }
        acc
    }
}

/// Lane gl2: a hook the publisher installs; the open runs it on an idle
/// hot-pool spinner shortly before the proof is assembled (see
/// `pcs::ligerito`), e.g. to bring the publish path's kernel state back into
/// cache. Must be cheap, side-effect free for the proof, and never block.
pub static GL2_TAIL_HOOK: std::sync::OnceLock<fn()> = std::sync::OnceLock::new();

pub mod hot_pool {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn enabled() -> bool {
        static OFF: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_HOT_POOL").is_some());
        !*OFF
    }

    std::thread_local! {
        static TOP_SP: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
        static CALLBACK_ACTIVE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    }

    const TOP_LEVEL_DEPTH: usize = 1024;

    #[inline(never)]
    pub(crate) fn record_top() {
        let marker = 0u8;
        TOP_SP.with(|t| t.set(core::hint::black_box(&marker) as *const u8 as usize));
    }

    #[inline(never)]
    fn at_top_level() -> bool {
        let marker = 0u8;
        let here = core::hint::black_box(&marker) as *const u8 as usize;
        let top = TOP_SP.with(|t| t.get());
        top != 0 && here <= top && top - here < TOP_LEVEL_DEPTH
    }

    /// Admission shared by all warming callbacks, including preseed/tail work.
    /// Stack distance alone can admit several optimized callback frames. Keep
    /// this guard across scheduler yields so their finite budgets cannot nest
    /// and multiply. Workers without a recorded ranked-pool stack are refused.
    #[must_use = "keep the guard alive for the whole warming callback"]
    pub struct CallbackGuard(core::marker::PhantomData<*mut ()>);

    impl CallbackGuard {
        pub fn enter() -> Option<Self> {
            if CALLBACK_ACTIVE.with(|active| active.get()) || !at_top_level() {
                return None;
            }
            CALLBACK_ACTIVE.with(|active| active.set(true));
            // A thread-local guard must not move to another worker.
            Some(Self(core::marker::PhantomData))
        }
    }

    impl Drop for CallbackGuard {
        fn drop(&mut self) {
            CALLBACK_ACTIVE.with(|active| active.set(false));
        }
    }

    fn spins_per_job() -> usize {
        static V: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
            std::env::var("FLOCK_HOT_POOL_SPINS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048)
        });
        *V
    }

    /// Lane gl2: side tasks run by the hot-pool spinners (never by the
    /// pool owner's own join waits, unlike a `rayon::spawn` onto its deque).
    /// A task is called repeatedly with `is_owner_sibling` and must return
    /// at once when it has nothing (left) to do; registration and removal
    /// bump a generation so each spinner re-reads the list only on change.
    pub mod side {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        pub type Task = Arc<dyn Fn(bool) + Send + Sync>;

        static GEN: AtomicUsize = AtomicUsize::new(0);
        static LIST: std::sync::Mutex<Vec<Task>> = std::sync::Mutex::new(Vec::new());

        std::thread_local! {
            static CACHE: core::cell::RefCell<(usize, Vec<Task>)> =
                const { core::cell::RefCell::new((0, Vec::new())) };
        }

        pub fn register(t: Task) {
            LIST.lock().unwrap_or_else(|e| e.into_inner()).push(t);
            GEN.fetch_add(1, Ordering::AcqRel);
        }

        /// Remove several tasks with one generation bump (each bump makes
        /// every spinner re-read the list).
        pub fn unregister_many(ts: &[Task]) {
            if ts.is_empty() {
                return;
            }
            LIST.lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|x| !ts.iter().any(|t| Arc::ptr_eq(x, t)));
            GEN.fetch_add(1, Ordering::AcqRel);
        }

        pub fn unregister(t: &Task) {
            LIST.lock().unwrap_or_else(|e| e.into_inner()).retain(|x| !Arc::ptr_eq(x, t));
            GEN.fetch_add(1, Ordering::AcqRel);
        }

        /// Run every registered task once (spinners only).
        #[inline]
        pub(crate) fn poll(is_owner_sibling: bool) {
            let g = GEN.load(Ordering::Acquire);
            CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.0 != g {
                    let list = LIST.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    *c = (g, list);
                }
                for t in c.1.iter() {
                    t(is_owner_sibling);
                }
            });
        }
    }

    fn spinner(run: Arc<AtomicBool>, owner: usize, backoff: usize) {
        if rayon::current_thread_index() == Some(owner) {
            return;
        }
        let Some(_callback) = CallbackGuard::enter() else {
            return;
        };
        let mut team_last = 0u64;
        for _ in 0..spins_per_job() {
            if !run.load(Ordering::Acquire) {
                return;
            }
            crate::flat_par::team_poll(&mut team_last);
            side::poll(backoff == 64);
            if !matches!(rayon::yield_now(), Some(rayon::Yield::Executed)) {
                for _ in 0..backoff {
                    core::hint::spin_loop();
                }
            }
        }
        rayon::spawn(move || spinner(run, owner, backoff));
    }

    pub struct HotPool(Arc<AtomicBool>, usize);

    impl HotPool {
        pub fn start() -> Option<Self> {
            if !enabled() {
                return None;
            }
            if TOP_SP.with(|top| top.get()) == 0 {
                return None;
            }
            let owner = rayon::current_thread_index()?;
            let n = rayon::current_num_threads();
            let sibling = (owner + n / 2) % n.max(1);
            let run = Arc::new(AtomicBool::new(true));
            let flag = run.clone();
            let bcast = move || {
                rayon::spawn_broadcast(move |ctx| {
                    let me = ctx.index();
                    if me != owner {
                        spinner(flag.clone(), owner, if me == sibling { 64 } else { 8 });
                    }
                })
            };
            // Lane gl1 (`FLOCK_NO_GL1_HOT_DEFER`): the broadcast wakes every
            // sleeping worker (one futex wake each, 0.02-0.03 ms on the owner,
            // which is on the serial path). Hand it to one woken worker
            // instead: the owner pays a single spawn + wake.
            static DEFER: std::sync::LazyLock<bool> =
                std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GL1_HOT_DEFER").is_none());
            if *DEFER {
                rayon::spawn(bcast);
            } else {
                bcast();
            }
            // Lane gl1: the owner may dispatch team regions to the spinners.
            let prev = crate::flat_par::TEAM_OWNER.swap(crate::flat_par::team_token(), Ordering::AcqRel);
            Some(Self(run, prev))
        }
    }

    impl Drop for HotPool {
        fn drop(&mut self) {
            crate::flat_par::TEAM_OWNER.store(self.1, Ordering::Release);
            self.0.store(false, Ordering::Release);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn callback_admission_requires_recorded_stack() {
            assert!(CallbackGuard::enter().is_none());
        }

        #[test]
        fn callback_admission_excludes_descendants_and_recovers_after_unwind() {
            record_top();
            {
                let _callback = CallbackGuard::enter().expect("recorded stack");
                assert!(CallbackGuard::enter().is_none(), "nested callback admitted");
            }
            let result = std::panic::catch_unwind(|| {
                let _callback = CallbackGuard::enter().expect("previous callback returned");
                panic!("exercise callback unwinding");
            });
            assert!(result.is_err());
            let _callback = CallbackGuard::enter().expect("unwinding must release admission");
        }
    }
}

/// Run `op` on a pool worker instead of the calling thread.
///
/// Every top-level parallel region opened from a NON-worker thread parks that
/// thread on the region's latch and wakes it again at the end; measured in
/// situ that round trip is ~15 µs, against ~5 µs for the same region opened
/// from inside a worker. A phase that opens dozens of small regions therefore
/// pays the difference dozens of times. Wrapping the phase body in a
/// `rayon::join` moves the body onto a worker, so each of its regions becomes
/// a nested join instead of a fresh top-level entry.
///
/// Schedule only: the closure runs to completion before this returns, on the
/// same pool, with the same work. `FLOCK_NO_IN_POOL=1` restores the direct
/// call, and a caller that is already a worker takes the direct call anyway.
/// Lane sl2 (`FLOCK_NO_SL2_ENV`): a per-site cached `var_os(name).is_some()`.
/// The first read of the environment after a large phase costs ~2 us of
/// cold misses on the serial path; the process environment never changes
/// after start-up in the ranked worker.
#[macro_export]
macro_rules! sl2_env_set {
    ($name:expr) => {{
        static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if $crate::sl2_env_live() {
            std::env::var_os($name).is_some()
        } else {
            *C.get_or_init(|| std::env::var_os($name).is_some())
        }
    }};
}

#[doc(hidden)]
pub fn sl2_env_live() -> bool {
    static LIVE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_SL2_ENV").is_some());
    *LIVE
}

/// Lane sl2: set by the ranked worker for the timed speculative prove, whose
/// process exits right after publishing (its tail may leak instead of free).
pub static SL2_TIMED_TAIL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `FLOCK_NO_SL2_LEAK=1` keeps every free in the timed prove's tail.
pub fn sl2_leak_tail() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_SL2_LEAK").is_none());
    *ON && SL2_TIMED_TAIL.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn in_pool<R: Send>(op: impl FnOnce() -> R + Send) -> R {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_IN_POOL").is_none());
    if !*ON || rayon::current_thread_index().is_some() {
        return op();
    }
    rayon::join(op, || ()).0
}

/// Best-effort `madvise(MADV_COLLAPSE)` over an already-touched buffer: where
/// the first-touch faults lost the THP lottery under fragmentation, this asks
/// the kernel to assemble the range into 2 MiB pages synchronously; where they
/// won, the calls are no-ops. Setup-phase only. `FLOCK_NO_MADV_COLLAPSE=1`
/// disables it.
///
/// Returns `true` when there is nothing left to collapse for this range —
/// the kernel reported the whole range assembled, or the range is below one
/// huge page — and `false` when the call was skipped or refused (typically
/// `ENOMEM` under fragmentation), so a caller that retries on later parks
/// can stop as soon as the range is known huge.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn collapse_hugepages(ptr: *mut u8, bytes: usize) -> bool {
    const HUGE: usize = 1 << 21;
    if bytes < HUGE {
        return true;
    }
    static DISABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_MADV_COLLAPSE").is_some());
    if *DISABLED {
        return false;
    }
    const PAGE: usize = 4096;
    let start = (ptr as usize).next_multiple_of(PAGE);
    let end = ptr as usize + bytes;
    if end <= start {
        return true;
    }
    const SYS_MADVISE: usize = 28;
    const MADV_COLLAPSE: usize = 25;
    // SAFETY: the advised range lies within the caller's live buffer, and
    // MADV_COLLAPSE never alters contents or mapping validity; failure is
    // ignored (pure hint).
    unsafe {
        let ret: isize;
        core::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MADVISE as isize => ret,
            in("rdi") start,
            in("rsi") end - start,
            in("rdx") MADV_COLLAPSE,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
        ret == 0
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub(crate) fn collapse_hugepages(_ptr: *mut u8, _bytes: usize) -> bool {
    true
}

/// Existing large uninitialized-allocation layout, shared with the prover recycler.
#[doc(hidden)]
pub const UNINIT_VEC_LARGE_MIN_BYTES: usize = 2 << 20;
#[doc(hidden)]
pub const UNINIT_VEC_TAIL_PAGES: usize = 96;

/// Allocate a `Vec<T>` of length `n` whose contents are NOT zero-initialized.
/// Caller MUST write every slot before reading it.
///
/// Used to skip the eager zero-init of large ping-pong buffers in hot prover
/// paths (PCS open, Round-2 fold, NTT scratch, lincheck packing). At m=29 the
/// zero-fill of a fresh 128 MB `vec![T::default(); n]` runs sequentially on
/// the main thread (~22 ms), which caps the parallel speedup of those phases.
///
/// `T: Copy` ensures `T` has no Drop impl, so the leaked uninitialized
/// elements are a no-op on drop.
///
/// # Safety contract
///
/// Reading uninitialized memory is UB per Rust's memory model regardless of
/// whether all bit patterns are valid for `T`. Caller must ensure every slot
/// is written before any read.
// `uninit_vec` flags exactly this pattern; here it is the deliberate purpose of
// the function (the safety contract above is what makes it sound).
#[allow(clippy::uninit_vec)]
pub(crate) fn alloc_uninit_vec<T: Copy>(n: usize) -> Vec<T> {
    // Layout-basin probe, gated to LARGE allocations only. The lever exists to
    // shift the phase of the 512 MiB pool buffers inside their 2 MiB pages, and
    // those already go straight to mmap. Applying it to small allocations would
    // push them over glibc's 128 KiB M_MMAP_THRESHOLD and turn heap bumps into
    // mmap syscalls — measured at -8% before this gate was added.
    const EY_TAIL_PAGES: usize = UNINIT_VEC_TAIL_PAGES;
    let bytes = n * core::mem::size_of::<T>();
    let tail = if bytes >= UNINIT_VEC_LARGE_MIN_BYTES {
        core::hint::black_box(EY_TAIL_PAGES) * 4096 / core::mem::size_of::<T>().max(1)
    } else {
        0
    };
    LAST_ALLOC_RECYCLED.with(|c| c.set(false));
    let mut v: Vec<T> = Vec::with_capacity(n + tail);
    // SAFETY:
    // - capacity >= n was just allocated, so set_len(n) is in bounds.
    // - T: Copy implies !Drop, so leaking uninit elements is a no-op.
    // - Caller upholds write-before-read.
    unsafe {
        v.set_len(n);
    }
    // A block handed back by the recycling allocator is the same mapping
    // that was advised when it was first allocated; `MADV_HUGEPAGE` is a
    // VMA flag, so re-advising it is a syscall for nothing. The ranked
    // prove takes three such blocks (8/16/4 MiB) inside the timed window.
    // `FLOCK_ALWAYS_ADVISE=1` restores the unconditional advise.
    if !LAST_ALLOC_RECYCLED.with(|c| c.get()) || always_advise() {
        advise_hugepages(v.as_mut_ptr().cast::<u8>(), n * core::mem::size_of::<T>());
    }
    // A fresh allocation may land on the address of a buffer that was
    // released by a plain `drop` while a scratch provenance tag was still
    // armed for it. This address now belongs to a new object, so that tag is
    // void — drop it before the new buffer can inherit it at its own
    // release. Lock-free (a short atomic scan) when nothing is armed.
    crate::scratch::void_pending_tag(v.as_ptr().cast::<crate::field::F128>());
    v
}

thread_local! {
    /// Provenance of the most recent recyclable-class allocation on this
    /// thread, set by the process's global allocator (`flock_prover::
    /// recycle_alloc`): `true` when the block came off a freelist, `false`
    /// when the system allocator produced a fresh mapping. Const-initialised
    /// `Cell<bool>`: no destructor, no lazy registration, safe to touch from
    /// inside `GlobalAlloc::alloc`. Builds without that allocator never set
    /// it, so they keep advising every allocation.
    pub static LAST_ALLOC_RECYCLED: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

fn always_advise() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_ALWAYS_ADVISE").is_some());
    *ON
}

/// Compatibility shim — same as `alloc_uninit_vec::<F128>(n)`.
pub(crate) fn alloc_uninit_f128_vec(n: usize) -> Vec<crate::field::F128> {
    alloc_uninit_vec::<crate::field::F128>(n)
}

/// Ranked default parallelizes a family of previously single-threaded
/// segments on the prover's Fiat–Shamir chain — pure per-element gathers,
/// bit-transposes, and eq-table builds that ran alone on one core while the
/// pool idled (ligerito query-phase gathers, zerocheck round-one table
/// preparation). Every gated site is byte-neutral: pure functions of
/// already-sampled values, order-preserving indexed parallel maps, and
/// exact-field factorizations — nothing the challenger observes changes.
/// Read once per process. `FLOCK_NO_SERIAL_PAR=1` restores every incumbent
/// sequential form (each site keeps it verbatim as its oracle).
pub(crate) fn serial_par_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_SERIAL_PAR").is_none());
    *ON
}

/// Cached [`perf_core_count`]. The uncached version may spawn `sysctl`; this
/// memoizes it so hot paths can cheaply ask "is the current rayon pool the
/// homogeneous P-core pool?" (i.e. `current_num_threads() <= this`).
#[cfg(target_arch = "aarch64")]
pub(crate) fn perf_core_count_cached() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(perf_core_count)
}

/// Best-effort count of **physical** performance cores used to size the
/// prover's thread pool. The hot phases are CLMUL-heavy and/or
/// memory-bandwidth-bound; SMT siblings share the core's execution ports and
/// add no DRAM bandwidth, so running 2 threads per physical core only adds
/// contention (on a 32C/64T Threadripper the prove is ~16% faster at 32 threads
/// than 64). On macOS, queries `hw.perflevel0.physicalcpu` (= P-core count on
/// Apple silicon, = physical CPU count on Intel). On Linux, `available_
/// parallelism()` counts SMT siblings, so derive physical cores from `/sys`
/// topology and clamp that host-wide count to the process's affinity/cgroup
/// availability. Elsewhere, falls back to `available_parallelism()`.
fn perf_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.physicalcpu"])
            .output()
            && let Ok(s) = std::str::from_utf8(&out.stdout)
            && let Ok(n) = s.trim().parse::<usize>()
            && n > 0
        {
            return n;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(n) = linux_physical_cores()
            && n > 0
        {
            let available = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            return n.min(available);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Count distinct physical cores via `/sys` topology: one entry per unique
/// `(physical_package_id, core_id)` over the online `cpuN` directories. Returns
/// `None` if the topology can't be read (caller falls back to logical count).
#[cfg(target_os = "linux")]
fn linux_physical_cores() -> Option<usize> {
    use std::collections::HashSet;
    let mut cores: HashSet<(String, String)> = HashSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let Some(rest) = name.strip_prefix("cpu") else {
            continue;
        };
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            continue; // skip "cpufreq", "cpuidle", etc.
        }
        let topo = path.join("topology");
        let core_id = std::fs::read_to_string(topo.join("core_id")).ok();
        let pkg = std::fs::read_to_string(topo.join("physical_package_id")).ok();
        if let (Some(c), Some(p)) = (core_id, pkg) {
            cores.insert((p.trim().to_owned(), c.trim().to_owned()));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}
