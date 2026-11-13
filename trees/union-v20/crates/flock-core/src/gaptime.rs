//! `FLOCK_GAP_TIMING`: wall-clock boundary instrumentation for the fast
//! prover. When the env var is set, [`mark`] prints one line per phase
//! boundary with the delta since the previous mark and the offset since
//! [`begin`] — the sum of the deltas accounts for the FULL prove wall time,
//! unlike per-phase stopwatches which silently drop the time BETWEEN phases
//! (pool installs, buffer drops, challenger hashing, allocator churn).
//!
//! Diagnostics only: the ranked worker's cleared env never sets the var, and
//! the disabled path is a single cached-bool load. Output goes to stderr so
//! it never mixes with bench stdout parsing.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Cached `FLOCK_GAP_TIMING` presence. Read once per process.
pub fn enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FLOCK_GAP_TIMING").is_some());
    *ON
}

/// Process-wide monotonic epoch all marks are measured against.
static PROC_EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);
/// Nanos-since-`PROC_EPOCH` of the current prove's `begin`.
static BEGIN_NS: AtomicU64 = AtomicU64::new(0);
/// Nanos-since-`PROC_EPOCH` of the previous mark.
static LAST_NS: AtomicU64 = AtomicU64::new(0);

fn now_ns() -> u64 {
    PROC_EPOCH.elapsed().as_nanos() as u64
}

/// Reset the epoch at the start of a prove and print a header line.
pub fn begin(label: &str) {
    if !enabled() {
        return;
    }
    let now = now_ns();
    BEGIN_NS.store(now, Ordering::Relaxed);
    LAST_NS.store(now, Ordering::Relaxed);
    eprintln!("[gap] ======== {label} ========");
}

/// Record a boundary: prints `+delta_ms @offset_ms label`.
pub fn mark(label: &'static str) {
    crate::rstrace::ev(label, 0);
    if !enabled() {
        return;
    }
    let now = now_ns();
    let last = LAST_NS.swap(now, Ordering::Relaxed);
    let begin = BEGIN_NS.load(Ordering::Relaxed);
    let d_ms = now.saturating_sub(last) as f64 / 1e6;
    let t_ms = now.saturating_sub(begin) as f64 / 1e6;
    if who_enabled() {
        let owner = crate::is_pool_owner();
        let name = std::thread::current().name().map(str::to_owned).unwrap_or_default();
        eprintln!("[gap] +{d_ms:9.3} ms  @{t_ms:9.3} ms  {label}  [who idx={:?} owner={owner} thr={name}]", rayon::current_thread_index());
        return;
    }
    eprintln!("[gap] +{d_ms:9.3} ms  @{t_ms:9.3} ms  {label}");
}

/// Lane hot1 diagnostics: `FLOCK_HOT1_WHO` appends the calling thread's rayon
/// index / pool-owner flag to every gap mark.
pub fn who_enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FLOCK_HOT1_WHO").is_some());
    *ON
}

// TEMP-ABS-TRACE
#[repr(C)]
struct Ts { s: i64, ns: i64 }
unsafe extern "C" { fn clock_gettime(clk: i32, ts: *mut Ts) -> i32; }
pub fn abs_enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FLOCK_ABS_TRACE").is_some());
    *ON
}
static ABS_BUF: std::sync::Mutex<Vec<(u64, &'static str)>> = std::sync::Mutex::new(Vec::new());
pub fn abs(label: &'static str) {
    crate::rstrace::ev(label, 1);
    if !abs_enabled() { return; }
    let mut t = Ts { s: 0, ns: 0 };
    unsafe { clock_gettime(1, &mut t) };
    let ns = t.s as u64 * 1_000_000_000 + t.ns as u64;
    ABS_BUF.lock().unwrap().push((ns, label));
}
pub fn abs_flush() {
    if !abs_enabled() { return; }
    let st = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let f: Vec<&str> = st.rsplit(')').next().unwrap_or("").split_whitespace().collect();
    let minflt = f.get(7).copied().unwrap_or("?");
    for (ns, l) in ABS_BUF.lock().unwrap().drain(..) {
        eprintln!("[abs] {ns} {l} minflt={minflt}");
    }
}
