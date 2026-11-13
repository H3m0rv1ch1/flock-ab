//! Elided witness: the ranked producer may skip z's DRAM stores when an
//! exact regenerator (built from witness-born root tiles) can reproduce every
//! z bit on demand. The regenerator is registered here, keyed by the z
//! buffer's address and length, and consumed exactly once:
//!
//! * the open's direct-fold8 L1-precommit route folds straight from the
//!   regenerator ([`take`]), never reading the stale buffer;
//! * every other z reader first calls [`ensure_dense`], which rewrites the
//!   complete canonical witness into the buffer.
//!
//! Field arithmetic is exact, so either route yields bit-identical outputs.

use crate::field::F128;
use std::sync::Mutex;

/// Exact regenerator of a canonical block-major witness.
pub trait ZRegen: Send + Sync {
    /// `out[t] = Σ_{i<64} w[i] · z[64 t + i]` for every `t < out.len()`,
    /// where `out.len() == z.len() / 64` of the registered witness.
    fn fold64(&self, out: &mut [F128], w: &[F128; 64]);
    /// Overwrite every element of `z` with the canonical witness.
    fn materialize(&self, z: &mut [F128]);
}

struct Entry {
    addr: usize,
    len: usize,
    regen: Box<dyn ZRegen>,
}

/// Registrations keyed by buffer; normally at most one (proofs are
/// serialized), but concurrent proofs never see each other's entries.
static SLOT: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

fn remove(addr: usize, len: usize) -> Option<Box<dyn ZRegen>> {
    let mut slot = SLOT.lock().unwrap();
    let i = slot.iter().position(|e| e.addr == addr && e.len == len)?;
    Some(slot.swap_remove(i).regen)
}

/// Register `regen` as the only valid source of the (stale) buffer `z`.
/// A previous registration of the same buffer is dropped.
pub fn install(z: &[F128], regen: Box<dyn ZRegen>) {
    let (addr, len) = (z.as_ptr() as usize, z.len());
    let old = remove(addr, len);
    SLOT.lock().unwrap().push(Entry { addr, len, regen });
    drop(old);
}

/// True iff `z` is a registered elided buffer.
pub fn is_elided(z: &[F128]) -> bool {
    let (addr, len) = (z.as_ptr() as usize, z.len());
    SLOT.lock().unwrap().iter().any(|e| e.addr == addr && e.len == len)
}

/// Remove and return the regenerator registered for exactly `z`.
pub fn take(z: &[F128]) -> Option<Box<dyn ZRegen>> {
    remove(z.as_ptr() as usize, z.len())
}

/// Make `z` dense: if it is the registered elided buffer, regenerate it in
/// place and drop the registration. No-op otherwise.
pub fn ensure_dense(z: &mut [F128], site: &str) {
    if let Some(regen) = take(z) {
        let t = std::time::Instant::now();
        regen.materialize(z);
        crate::gaptime::mark("elided z materialized");
        if std::env::var_os("FLOCK_ZR_TRACE").is_some() {
            eprintln!(
                "[zr] fallback materialize at {site}: {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

/// Drop the registration for the buffer at `addr` with `len` elements, if
/// any (end of a prove: the buffer is dense, recycled or dead).
pub fn forget(addr: usize, len: usize) {
    drop(remove(addr, len));
}
