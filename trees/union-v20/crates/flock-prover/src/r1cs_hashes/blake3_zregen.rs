//! Exact regeneration of the ranked BLAKE3 witness `z` from the witness-born
//! root tiles (`flock_core::lincheck::WitnessRootTiles`).
//!
//! The ranked producer may skip z's global DRAM stores for octas it seeds
//! from their compact copy (`StreamProj::elide_z`). Every z bit is then
//! still determined by the tiles:
//!
//! * root wires (11,249 of the 11,313 plan roots) are stored verbatim, one
//!   64-wire group per z qword (or per qword pair around a lin interval);
//! * the 64 omitted roots (wires 1089..1153) are the constant `block_len = 64`
//!   and `flags = 11` words every ranked compression carries;
//! * the 4,096 dependent wires (each G's `b_new`/`d_new`, `out_lo`, `out_hi`)
//!   are BLAKE3 state words, recomputed here from the stored `cv`, `m`,
//!   counter inputs.
//!
//! The expander works on one octa pair (16 compressions) at a time: BLAKE3 in
//! sixteen u32 lanes, then each octa's 242 live z qwords in qword-lane order
//! (one ZMM per qword, lane = compression) in an L1-resident stage. The fold
//! consumes that stage directly with carry-less multiplies, so z is never
//! transposed to block order and never written to DRAM.

use std::sync::atomic::{AtomicBool, Ordering};

/// Test-only override: force z elision off without touching the process env.
pub(crate) static FORCE_OFF: AtomicBool = AtomicBool::new(false);
/// Diagnostics: regenerator folds / full materializations performed.
pub(crate) static REGEN_FOLDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(crate) static REGEN_MATERIALIZES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Direct field folding for exact generated witness pairs; rollback keeps
/// the promoted full-stage generation and field-fold path.
pub(crate) fn fused_fold_enabled()->bool {
    static ON:std::sync::LazyLock<bool>=std::sync::LazyLock::new(||std::env::var_os("FLOCK_NO_FUSED_Z_FOLD").is_none());
    *ON
}

/// Production gate for skipping z's DRAM stores (`FLOCK_NO_Z_ELIDE=1`
/// restores the dense witness). Requires the exact regeneration plan.
pub(crate) fn z_elide_enabled() -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "avx512vbmi2",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("FLOCK_NO_Z_ELIDE").is_none() && imp::plan().is_some()
        });
        *ON && !FORCE_OFF.load(Ordering::Relaxed)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "avx512vbmi2",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )))]
    {
        let _ = &FORCE_OFF;
        false
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "avx512vbmi2",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
pub(crate) use imp::TileRegen;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "avx512vbmi2",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
pub(crate) mod imp {
    use super::super::{
        BLEN_BASE, FLAGS_BASE, G_LANES, G_MSG_IDX, G_STRIDE, GS_BASE, K, MSG_PERMUTATION,
        N_G, N_G_PER_ROUND, N_ROUNDS, OUT_HI_BASE, OUT_LO_BASE, T_HI_BASE, USEFUL_BITS,
        WORD_BITS, ADDS_PER_G, CARRY_BITS_PER_ADD,
    };
    use core::arch::x86_64::*;
    use flock_core::field::F128;
    use flock_core::lincheck::{RootFoldSource, WitnessRootTiles, WitnessTileView};
    use rayon::prelude::*;
    use std::sync::Arc;

    /// Staged z qwords per compression: 0..=240 are live, 241 is the zero
    /// high half of element 120. Qwords 242..256 are zero padding.
    pub(super) const NQ: usize = 242;
    const GROUPS: usize = 176;
    const GROUP_BYTES: usize = 512;
    const TILE_BYTES: usize = GROUPS * GROUP_BYTES;
    const BLOCK_LEN: u32 = 64;
    const FLAGS: u32 = 11;
    /// Fixed-cache group 13 carries wire 1088 in bit 0 and wires 1153..1216
    /// in bits 1..63 (see `RootTileCache::from_witness_storage`).
    const SPLICED_GROUP: usize = 13;
    const TAIL_GROUP: usize = 175;
    const TAIL_Q: usize = 235;
    const TAIL_BITS: u32 = 49;
    /// Constant bits of z qword 17 (wires 1088..1152): block_len and flags.
    const Q17_CONST: u64 = ((BLOCK_LEN as u64) << (BLEN_BASE - 1088))
        | ((FLAGS as u64) << (FLAGS_BASE - 1088));
    const LIN_OFFSET: usize = ADDS_PER_G * CARRY_BITS_PER_ADD;

    const fn sched() -> [[[usize; 2]; N_G_PER_ROUND]; N_ROUNDS] {
        let mut perm = [0usize; 16];
        let mut i = 0;
        while i < 16 {
            perm[i] = i;
            i += 1;
        }
        let mut out = [[[0usize; 2]; N_G_PER_ROUND]; N_ROUNDS];
        let mut r = 0;
        while r < N_ROUNDS {
            let mut g = 0;
            while g < N_G_PER_ROUND {
                out[r][g][0] = perm[G_MSG_IDX[g][0]];
                out[r][g][1] = perm[G_MSG_IDX[g][1]];
                g += 1;
            }
            let mut next = [0usize; 16];
            let mut i = 0;
            while i < 16 {
                next[i] = perm[MSG_PERMUTATION[i]];
                i += 1;
            }
            perm = next;
            r += 1;
        }
        out
    }
    const SCHED: [[[usize; 2]; N_G_PER_ROUND]; N_ROUNDS] = sched();

    /// z qword `q` = cache group `cg`.
    #[derive(Clone, Copy, Debug)]
    struct FullOp {
        q: u16,
        cg: u16,
    }

    /// z qwords `q`, `q+1` = cache group `cg` split around G `g`'s 64 lin
    /// wires; rotation and low mask live at the same index in `rot`/`mask`.
    #[derive(Clone, Copy, Debug)]
    struct PairOp {
        q: u16,
        cg: u16,
        g: u16,
    }

    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    struct Lanes([u64; 8]);

    pub(crate) struct Plan {
        full: Vec<FullOp>,
        pairs: Vec<PairOp>,
        rot: Vec<Lanes>,
        mask: Vec<Lanes>,
    }

    /// Derive (and fully check) the tile -> z qword map from the linear fold
    /// plan's root list. `None` on any geometry surprise: elision then stays
    /// off and z is stored as before.
    fn build_plan(roots: &[usize]) -> Option<Plan> {
        let lanes_ok = G_LANES
            == [
                [0, 4, 8, 12],
                [1, 5, 9, 13],
                [2, 6, 10, 14],
                [3, 7, 11, 15],
                [0, 5, 10, 15],
                [1, 6, 11, 12],
                [2, 7, 8, 13],
                [3, 4, 9, 14],
            ];
        if !lanes_ok
            || K != 16384
            || USEFUL_BITS != 15409
            || OUT_LO_BASE != 256
            || T_HI_BASE + WORD_BITS != 1089
            || BLEN_BASE != 1089
            || FLAGS_BASE + WORD_BITS != 1153
            || GS_BASE != 1153
            || N_G != 56
            || GS_BASE + G_STRIDE * (N_G - 1) + LIN_OFFSET != TAIL_Q * 64 + TAIL_BITS as usize
            || OUT_HI_BASE != (TAIL_Q + 1) * 64 + TAIL_BITS as usize
            || OUT_HI_BASE + 8 * WORD_BITS != USEFUL_BITS
            || roots.len() != 11313
            || roots.windows(2).any(|w| w[0] >= w[1])
        {
            return None;
        }
        let groups: Vec<&[usize]> = roots.chunks(64).collect();
        if groups.len() != GROUPS + 1 {
            return None;
        }
        let contiguous = |g: &[usize], first: usize| g.iter().enumerate().all(|(i, &c)| c == first + i);
        if groups[13].len() != 64 || !contiguous(groups[13], 1088) {
            return None;
        }
        if groups[14].len() != 64 || !contiguous(groups[14], 1152) {
            return None;
        }
        let mut covered = [0u8; NQ];
        for q in 4..8 {
            covered[q] += 1; // out_lo
        }
        covered[17] += 1;
        covered[18] += 1;
        for q in TAIL_Q..NQ {
            covered[q] += 1; // tail roots + G55 lin + out_hi + zero
        }
        let mut full = Vec::new();
        let mut pairs = Vec::new();
        let mut rot = Vec::new();
        let mut mask = Vec::new();
        let mut g_seen = [false; 55];
        for (r, grp) in groups.iter().enumerate() {
            if r == 13 || r == 14 {
                continue;
            }
            let cg = if r < 13 { r } else { r - 1 };
            let first = grp[0];
            if first % 64 != 0 {
                return None;
            }
            let q = first / 64;
            if grp.len() == TAIL_BITS as usize {
                if r != GROUPS || cg != TAIL_GROUP || q != TAIL_Q || !contiguous(grp, first) {
                    return None;
                }
                continue;
            }
            if grp.len() != 64 || q + 1 >= NQ {
                return None;
            }
            if contiguous(grp, first) {
                covered[q] += 1;
                full.push(FullOp { q: q as u16, cg: cg as u16 });
                continue;
            }
            let split = grp.iter().position(|&c| c >= first + 64)?;
            if split == 0
                || grp
                    .iter()
                    .enumerate()
                    .any(|(i, &c)| c != first + i + if i >= split { 64 } else { 0 })
            {
                return None;
            }
            let lin = first + split;
            let off = lin.checked_sub(GS_BASE + LIN_OFFSET)?;
            if off % G_STRIDE != 0 || off / G_STRIDE >= 55 {
                return None;
            }
            let g = off / G_STRIDE;
            if std::mem::replace(&mut g_seen[g], true) {
                return None;
            }
            covered[q] += 1;
            covered[q + 1] += 1;
            pairs.push(PairOp { q: q as u16, cg: cg as u16, g: g as u16 });
            rot.push(Lanes([split as u64; 8]));
            mask.push(Lanes([(1u64 << split) - 1; 8]));
        }
        if covered.iter().any(|&c| c != 1) || g_seen.iter().any(|&s| !s) {
            return None;
        }
        Some(Plan { full, pairs, rot, mask })
    }

    pub(crate) fn plan() -> Option<&'static Plan> {
        static PLAN: std::sync::LazyLock<Option<Plan>> = std::sync::LazyLock::new(|| {
            let lf = super::super::BLAKE3_ADJOINT_PLAN
                .linear_fold
                .get_or_init(|| super::super::BLAKE3_ADJOINT_PLAN.linear_fold_plan());
            build_plan(&lf.roots)
        });
        PLAN.as_ref()
    }

    #[repr(C, align(64))]
    pub(super) struct Stage(pub(super) [__m512i; NQ]);

    /// Per-worker scratch (~39 KiB): two octa stages, plus dependencies and
    /// outputs for the root expansion fallback. Direct generation writes both
    /// stages and leaves the fallback arrays cold. All storage is reused
    /// across proofs and completely overwritten before a consumer reads it.
    #[repr(C, align(64))]
    pub(super) struct Scratch {
        pub(super) stage: Stage,
        second_stage: Stage,
        deps: [[__m512i; N_G]; 2],
        out_lo: [[__m512i; 4]; 2],
        out_hi: [[__m512i; 4]; 2],
    }

    #[repr(align(64))]
    struct GeneratedTile([core::mem::MaybeUninit<u8>; TILE_BYTES]);

    impl Scratch {
        pub(super) fn new() -> Box<Self> {
            // SAFETY: all-zero bytes are a valid value of every field. Built
            // in place on the heap (it is too large for a worker's stack).
            unsafe { Box::<Self>::new_zeroed().assume_init() }
        }
    }

    /// BLAKE3 over octas `o0`, `o0 + 1` (sixteen u32 lanes) from their input
    /// words in the tile; stores every dependent qword in `scratch` (slots
    /// 0 and 1). Issues one `pf` step per odd G (28 lines).
    ///
    /// # Safety
    /// `tile` points at one complete, initialized, 64-byte-aligned witness
    /// tile (176 groups x 512 bytes) and `o0 + 1 < 8`.
    #[inline(never)]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,gfni,vpclmulqdq")]
    pub(super) unsafe fn blake3_pair(tile: *const u8, o0: usize, scratch: &mut Scratch, pf: &mut PfCur) {
        unsafe {
            macro_rules! stripe {
                ($cg:expr, $o:expr) => {
                    _mm512_load_si512(tile.add($cg * GROUP_BYTES + $o * 64).cast::<__m512i>())
                };
            }
            let idx_lo = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            let idx_hi = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
            // Word w (16 lanes: octa o0 then o0+1) from its cache group.
            macro_rules! word16 {
                ($cg:expr, $hi:expr) => {
                    _mm512_permutex2var_epi32(
                        stripe!($cg, o0),
                        if $hi { idx_hi } else { idx_lo },
                        stripe!($cg, o0 + 1),
                    )
                };
            }
            let cv: [__m512i; 8] = core::array::from_fn(|j| word16!(j / 2, j & 1 == 1));
            // z words 16..=34 (wires 512..1120); word 34 keeps wire 1088 in
            // bit 0, the only bit read from it.
            let zw: [__m512i; 19] = core::array::from_fn(|i| {
                let w = 16 + i;
                let q = w / 2;
                let cg = if q == 17 { SPLICED_GROUP } else { q - 4 };
                word16!(cg, w & 1 == 1)
            });
            let m: [__m512i; 16] = core::array::from_fn(|i| _mm512_shrdi_epi32::<1>(zw[i], zw[i + 1]));
            let t_lo = _mm512_shrdi_epi32::<1>(zw[16], zw[17]);
            let t_hi = _mm512_shrdi_epi32::<1>(zw[17], zw[18]);
            let iv = super::super::BLAKE3_IV;
            let mut v: [__m512i; 16] = [
                cv[0],
                cv[1],
                cv[2],
                cv[3],
                cv[4],
                cv[5],
                cv[6],
                cv[7],
                _mm512_set1_epi32(iv[0] as i32),
                _mm512_set1_epi32(iv[1] as i32),
                _mm512_set1_epi32(iv[2] as i32),
                _mm512_set1_epi32(iv[3] as i32),
                t_lo,
                t_hi,
                _mm512_set1_epi32(BLOCK_LEN as i32),
                _mm512_set1_epi32(FLAGS as i32),
            ];
            // Lane j of octa A/B as qword (b_new | d_new << 32).
            let il_a = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            let il_b = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            let deps = &mut scratch.deps;
            macro_rules! g {
                ($r:literal, $gi:literal, $a:literal, $b:literal, $c:literal, $d:literal) => {{
                    let mx = m[SCHED[$r][$gi][0]];
                    let my = m[SCHED[$r][$gi][1]];
                    v[$a] = _mm512_add_epi32(_mm512_add_epi32(v[$a], v[$b]), mx);
                    v[$d] = _mm512_ror_epi32::<16>(_mm512_xor_si512(v[$d], v[$a]));
                    v[$c] = _mm512_add_epi32(v[$c], v[$d]);
                    v[$b] = _mm512_ror_epi32::<12>(_mm512_xor_si512(v[$b], v[$c]));
                    v[$a] = _mm512_add_epi32(_mm512_add_epi32(v[$a], v[$b]), my);
                    v[$d] = _mm512_ror_epi32::<8>(_mm512_xor_si512(v[$d], v[$a]));
                    v[$c] = _mm512_add_epi32(v[$c], v[$d]);
                    v[$b] = _mm512_ror_epi32::<7>(_mm512_xor_si512(v[$b], v[$c]));
                    deps[0][8 * $r + $gi] = _mm512_permutex2var_epi32(v[$b], il_a, v[$d]);
                    deps[1][8 * $r + $gi] = _mm512_permutex2var_epi32(v[$b], il_b, v[$d]);
                    if $gi & 1 == 1 {
                        pf.step();
                    }
                }};
            }
            macro_rules! round {
                ($r:literal) => {
                    g!($r, 0, 0, 4, 8, 12);
                    g!($r, 1, 1, 5, 9, 13);
                    g!($r, 2, 2, 6, 10, 14);
                    g!($r, 3, 3, 7, 11, 15);
                    g!($r, 4, 0, 5, 10, 15);
                    g!($r, 5, 1, 6, 11, 12);
                    g!($r, 6, 2, 7, 8, 13);
                    g!($r, 7, 3, 4, 9, 14);
                };
            }
            round!(0);
            round!(1);
            round!(2);
            round!(3);
            round!(4);
            round!(5);
            round!(6);
            for k in 0..4 {
                let lo0 = _mm512_xor_si512(v[2 * k], v[2 * k + 8]);
                let lo1 = _mm512_xor_si512(v[2 * k + 1], v[2 * k + 9]);
                let hi0 = _mm512_xor_si512(v[2 * k + 8], cv[2 * k]);
                let hi1 = _mm512_xor_si512(v[2 * k + 9], cv[2 * k + 1]);
                scratch.out_lo[0][k] = _mm512_permutex2var_epi32(lo0, il_a, lo1);
                scratch.out_lo[1][k] = _mm512_permutex2var_epi32(lo0, il_b, lo1);
                scratch.out_hi[0][k] = _mm512_permutex2var_epi32(hi0, il_a, hi1);
                scratch.out_hi[1][k] = _mm512_permutex2var_epi32(hi0, il_b, hi1);
            }
        }
    }

    /// Fill `scratch.stage` with octa `o0 + x` (qword-lane z: one ZMM per z
    /// qword, lane = compression) from the tile (L2-resident by then) and
    /// dependent slot `x`. Spreads the current `pf` phase over its 174
    /// group ops.
    ///
    /// # Safety
    /// As for [`blake3_pair`]; `x < 2`; `prefetch` points into the tile span.
    #[inline(never)]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,gfni,vpclmulqdq")]
    pub(super) unsafe fn fill_stage(
        plan: &Plan,
        tile: *const u8,
        o: usize,
        x: usize,
        scratch: &mut Scratch,
        pf: &mut PfCur,
    ) {
        unsafe {
            let src = tile.add(o * 64);
            let idxq = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
            macro_rules! q64 {
                ($cg:expr) => {
                    _mm512_permutexvar_epi32(
                        idxq,
                        _mm512_load_si512(src.add($cg * GROUP_BYTES).cast::<__m512i>()),
                    )
                };
            }
            let stp = scratch.stage.0.as_mut_ptr();
            let dep = scratch.deps[x].as_ptr();
            for op in plan.full.iter() {
                pf.tick();
                *stp.add(op.q as usize) = q64!(op.cg as usize);
            }
            let rot = plan.rot.as_ptr();
            let mask = plan.mask.as_ptr();
            for (i, op) in plan.pairs.iter().enumerate() {
                pf.tick();
                let gq = q64!(op.cg as usize);
                let r = _mm512_rolv_epi64(
                    *dep.add(op.g as usize),
                    _mm512_load_si512((*rot.add(i)).0.as_ptr().cast()),
                );
                let mk = _mm512_load_si512((*mask.add(i)).0.as_ptr().cast());
                *stp.add(op.q as usize) = _mm512_ternarylogic_epi64::<0xCA>(mk, gq, r);
                *stp.add(op.q as usize + 1) = _mm512_ternarylogic_epi64::<0xCA>(mk, r, gq);
            }
            let ol = &scratch.out_lo[x];
            for k in 0..4 {
                *stp.add(4 + k) = ol[k];
            }
            let one = _mm512_set1_epi64(1);
            let c13 = q64!(SPLICED_GROUP);
            *stp.add(17) = _mm512_ternarylogic_epi64::<0xEA>(c13, one, _mm512_set1_epi64(Q17_CONST as i64));
            *stp.add(18) = _mm512_andnot_si512(one, c13);
            let oh = &scratch.out_hi[x];
            let d55 = *dep.add(N_G - 1);
            *stp.add(TAIL_Q) = _mm512_ternarylogic_epi64::<0xEA>(
                q64!(TAIL_GROUP),
                _mm512_set1_epi64(((1u64 << TAIL_BITS) - 1) as i64),
                _mm512_slli_epi64::<49>(d55),
            );
            *stp.add(TAIL_Q + 1) = _mm512_shrdi_epi64::<15>(d55, oh[0]);
            *stp.add(TAIL_Q + 2) = _mm512_shrdi_epi64::<15>(oh[0], oh[1]);
            *stp.add(TAIL_Q + 3) = _mm512_shrdi_epi64::<15>(oh[1], oh[2]);
            *stp.add(TAIL_Q + 4) = _mm512_shrdi_epi64::<15>(oh[2], oh[3]);
            *stp.add(TAIL_Q + 5) = _mm512_srli_epi64::<15>(oh[3]);
            *stp.add(TAIL_Q + 6) = _mm512_setzero_si512();
        }
    }

    /// Next-tile prefetch cursor. Each octa pair of a tile streams in the
    /// matching quarter (352 contiguous lines) of the next tile, paced evenly
    /// through the pair's BLAKE3 (`PF_BLAKE`), both staging passes
    /// (`PF_FILL` each) and both folds (one per unrolled iteration, 60
    /// each). Bursts would occupy the core's fill buffers while compute
    /// idles; paced, the stream overlaps the compute.
    pub(super) struct PfCur {
        cur: usize,
        lim: usize,
        end: usize,
        num: usize,
        den: usize,
        acc: usize,
    }

    const PF_BLAKE: usize = 28;
    const PF_FOLD: usize = 60;
    const PF_FILL: usize = (2 * GROUPS - PF_BLAKE - 2 * PF_FOLD) / 2;
    /// Ops of one staging pass (119 full groups + 55 split pairs).
    const FILL_OPS: usize = 174;

    impl PfCur {
        fn new(next: Option<usize>) -> Self {
            let (cur, end) = next.map_or((0, 0), |n| (n, n + 2 * (TILE_BYTES / 8)));
            Self { cur, lim: cur, end, num: 0, den: 1, acc: 0 }
        }

        /// Next phase: `lines` prefetches, spread over `ticks` ticks.
        #[inline(always)]
        fn phase(&mut self, lines: usize, ticks: usize) {
            self.lim = (self.cur + lines * 64).min(self.end);
            self.num = lines;
            self.den = ticks;
            self.acc = 0;
        }

        /// Prefetch the next line of the phase.
        #[inline(always)]
        fn step(&mut self) {
            if self.cur < self.lim {
                // SAFETY: a prefetch hint inside the tile span.
                unsafe { _mm_prefetch::<_MM_HINT_T1>(self.cur as *const i8) };
                self.cur += 64;
            }
        }

        /// One of the phase's `ticks`: issue its share of the lines.
        #[inline(always)]
        fn tick(&mut self) {
            self.acc += self.num;
            while self.acc >= self.den && self.cur < self.lim {
                // SAFETY: as in `step`.
                unsafe { _mm_prefetch::<_MM_HINT_T1>(self.cur as *const i8) };
                self.cur += 64;
                self.acc -= self.den;
            }
        }

        /// Issue whatever the phase has left.
        #[inline(always)]
        fn flush(&mut self) {
            while self.cur < self.lim {
                self.step();
            }
        }
    }

    /// Per-element fold weights: `[W.lo, W.hi]` and `[W.lo ^ W.hi, 0]` in
    /// every 128-bit lane.
    #[repr(C, align(64))]
    pub(super) struct Weights {
        lohi: [__m512i; 64],
        sum: [__m512i; 64],
    }

    impl Weights {
        pub(super) fn new(w: &[F128; 64]) -> Box<Self> {
            // SAFETY: AVX-512 is a cfg precondition of this module.
            unsafe {
                let mut out: Box<Self> = Box::new(core::mem::zeroed());
                for (i, wi) in w.iter().enumerate() {
                    out.lohi[i] = _mm512_broadcast_i32x4(_mm_set_epi64x(wi.hi as i64, wi.lo as i64));
                    out.sum[i] = _mm512_broadcast_i32x4(_mm_set_epi64x(0, (wi.lo ^ wi.hi) as i64));
                }
                out
            }
        }
    }

    /// Fold one staged octa: `[half][parity]`, each reduced ZMM holding the
    /// four F128 results of lanes `2l + parity`.
    #[inline(never)]
    #[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
    unsafe fn fold_stage(st: &Stage, w: &Weights, pf: &mut PfCur) -> [[__m512i; 2]; 2] {
        unsafe {
            let mut out = [[_mm512_setzero_si512(); 2]; 2];
            let st = st.0.as_ptr();
            let wl = w.lohi.as_ptr();
            let ws = w.sum.as_ptr();
            for (h, slot) in out.iter_mut().enumerate() {
                // Element pairs per half: 64 and 57 (qword 241 is zero).
                let ne = if h == 0 { 64 } else { (NQ - 128) / 2 };
                let base = st.add(128 * h);
                let z = _mm512_setzero_si512();
                let (mut le, mut lo, mut he, mut ho, mut me, mut mo) = (z, z, z, z, z, z);
                let mut e = 0;
                while e + 2 <= ne {
                    pf.step();
                    let a0 = *base.add(2 * e);
                    let b0 = *base.add(2 * e + 1);
                    let a1 = *base.add(2 * e + 2);
                    let b1 = *base.add(2 * e + 3);
                    let w0 = *wl.add(e);
                    let w1 = *wl.add(e + 1);
                    let s0 = *ws.add(e);
                    let s1 = *ws.add(e + 1);
                    let x0 = _mm512_xor_si512(a0, b0);
                    let x1 = _mm512_xor_si512(a1, b1);
                    le = _mm512_ternarylogic_epi64::<0x96>(
                        le,
                        _mm512_clmulepi64_epi128::<0x00>(a0, w0),
                        _mm512_clmulepi64_epi128::<0x00>(a1, w1),
                    );
                    lo = _mm512_ternarylogic_epi64::<0x96>(
                        lo,
                        _mm512_clmulepi64_epi128::<0x01>(a0, w0),
                        _mm512_clmulepi64_epi128::<0x01>(a1, w1),
                    );
                    he = _mm512_ternarylogic_epi64::<0x96>(
                        he,
                        _mm512_clmulepi64_epi128::<0x10>(b0, w0),
                        _mm512_clmulepi64_epi128::<0x10>(b1, w1),
                    );
                    ho = _mm512_ternarylogic_epi64::<0x96>(
                        ho,
                        _mm512_clmulepi64_epi128::<0x11>(b0, w0),
                        _mm512_clmulepi64_epi128::<0x11>(b1, w1),
                    );
                    me = _mm512_ternarylogic_epi64::<0x96>(
                        me,
                        _mm512_clmulepi64_epi128::<0x00>(x0, s0),
                        _mm512_clmulepi64_epi128::<0x00>(x1, s1),
                    );
                    mo = _mm512_ternarylogic_epi64::<0x96>(
                        mo,
                        _mm512_clmulepi64_epi128::<0x01>(x0, s0),
                        _mm512_clmulepi64_epi128::<0x01>(x1, s1),
                    );
                    e += 2;
                }
                if e < ne {
                    let a = *base.add(2 * e);
                    let b = *base.add(2 * e + 1);
                    let x = _mm512_xor_si512(a, b);
                    let w0 = *wl.add(e);
                    let s0 = *ws.add(e);
                    le = _mm512_xor_si512(le, _mm512_clmulepi64_epi128::<0x00>(a, w0));
                    lo = _mm512_xor_si512(lo, _mm512_clmulepi64_epi128::<0x01>(a, w0));
                    he = _mm512_xor_si512(he, _mm512_clmulepi64_epi128::<0x10>(b, w0));
                    ho = _mm512_xor_si512(ho, _mm512_clmulepi64_epi128::<0x11>(b, w0));
                    me = _mm512_xor_si512(me, _mm512_clmulepi64_epi128::<0x00>(x, s0));
                    mo = _mm512_xor_si512(mo, _mm512_clmulepi64_epi128::<0x01>(x, s0));
                }
                // Karatsuba middle term: (a_lo + a_hi)(w_lo + w_hi) + lo + hi.
                let me = _mm512_ternarylogic_epi64::<0x96>(me, le, he);
                let mo = _mm512_ternarylogic_epi64::<0x96>(mo, lo, ho);
                slot[0] = flock_core::field::gf2_128::x86_64::ghash_reduce_acc_x4(le, me, he);
                slot[1] = flock_core::field::gf2_128::x86_64::ghash_reduce_acc_x4(lo, mo, ho);
            }
            out
        }
    }

    /// Original compression index of physical slot `p` in a tile view.
    #[inline(always)]
    fn compression(cross_quarter: bool, p: usize) -> usize {
        if cross_quarter {
            let g = p / 256;
            let kpair = (p % 256) / 8;
            let lane = p % 8;
            (lane / 2) * 65536 + (2 * kpair + lane % 2) * 1024 + g
        } else {
            p
        }
    }

    #[derive(Clone, Copy)]
    struct SendPtr<T>(*mut T);
    // SAFETY: each worker writes a disjoint set of destination slots.
    unsafe impl<T> Send for SendPtr<T> {}
    unsafe impl<T> Sync for SendPtr<T> {}

    /// Visit every octa of every tile with its z staged in qword-lane order,
    /// eight tiles per task (the cross-quarter outputs of compressions 2i and
    /// 2i+1 then stay within one task). Each task streams its 704 KiB of
    /// tiles sequentially, prefetching one tile ahead (also across task
    /// boundaries: rayon runs adjacent tasks on the same worker).
    fn for_each_octa(view: &WitnessTileView, f: impl Fn(&Stage, usize, usize, &mut PfCur) + Sync) {
        let plan = plan().expect("z regeneration plan");
        assert!(view.tiles > 0);
        if view.cross_quarter {
            assert_eq!(view.tiles * 64, 1 << 18);
        }
        let unit = if view.tiles % 8 == 0 { 8 } else { 1 };
        let base = view.base;
        let tiles = view.tiles;
        #[cfg(test)]
        let mode = TEST_MODE.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let mode = 0u8;
        thread_local! {
            // Reused across proofs: no per-task allocation/zeroing.
            static SCRATCH: core::cell::RefCell<Option<(Box<Scratch>, Box<GeneratedTile>)>> =
                const { core::cell::RefCell::new(None) };
        }
        assert_eq!(GROUPS * 8, TILE_BYTES / 64);
        (0..view.tiles / unit).into_par_iter().for_each(|u| {
            SCRATCH.with(|cell| {
                let mut guard = cell.borrow_mut();
                let (scratch, generated_tile) = guard.get_or_insert_with(|| {
                    // Separate allocations keep the read-only tile disjoint
                    // from the &mut Scratch passed to the expansion kernels.
                    (Scratch::new(), unsafe { Box::<GeneratedTile>::new_uninit().assume_init() })
                });
                for t in u * unit..(u + 1) * unit {
                    let tile = if let Some((seed, generate)) = view.generator {
                        let dst = generated_tile.0.as_mut_ptr().cast::<u8>();
                        if view.pair_generator.is_none() {
                            unsafe { generate(seed, t, dst); }
                        }
                        dst.cast_const()
                    } else {
                        (base + t * TILE_BYTES) as *const u8
                    };
                    let next = (view.generator.is_none() && t + 1 < tiles)
                        .then(|| base + (t + 1) * TILE_BYTES);
                    for pair in 0..4 {
                        if mode == 0 && let Some(generate) = view.stage_generator {
                            let seed=view.generator.expect("stage generator seed").0;
                            unsafe {
                                generate(seed,t,pair,scratch.stage.0.as_mut_ptr().cast(),scratch.second_stage.0.as_mut_ptr().cast(),core::ptr::null());
                            }
                            let mut pf=PfCur::new(None);
                            f(&scratch.stage,t,2*pair,&mut pf);
                            f(&scratch.second_stage,t,2*pair+1,&mut pf);
                            continue;
                        }

                        if let Some(generate_pair) = view.pair_generator {
                            let seed = view.generator.expect("paired generator has a seed").0;
                            unsafe {
                                generate_pair(seed, t, pair,
                                    tile.cast_mut(),
                                    scratch.deps.as_mut_ptr().cast(),
                                    scratch.out_lo.as_mut_ptr().cast(),
                                    scratch.out_hi.as_mut_ptr().cast());
                            }
                        }
                        // While octa pair `pair` of this tile is processed,
                        // the same quarter of the next tile streams into L2.
                        let mut pf = PfCur::new(next.map(|n| n + 2 * pair * (TILE_BYTES / 8)));
                        // SAFETY: tiles are complete, initialized, 64-byte
                        // aligned; prefetch hints stay within the tile span.
                        // Mode bits (tests only) skip phases.
                        unsafe {
                            pf.phase(PF_BLAKE, 1);
                            if mode & 1 == 0 && view.pair_generator.is_none() {
                                blake3_pair(tile, 2 * pair, scratch, &mut pf);
                            }
                            for x in 0..2 {
                                let o = 2 * pair + x;
                                pf.flush();
                                pf.phase(PF_FILL, FILL_OPS);
                                if mode & 2 == 0 {
                                    fill_stage(plan, tile, o, x, scratch, &mut pf);
                                }
                                pf.flush();
                                pf.phase(PF_FOLD, 1);
                                f(&scratch.stage, t, o, &mut pf);
                            }
                            pf.flush();
                        }
                    }
                }
            })
        });
    }

    /// Tests only: bit 0 skips BLAKE3, bit 1 skips staging.
    #[cfg(test)]
    pub(crate) static TEST_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

    #[cfg(test)]
    pub(crate) fn check_generated_pair(
        seed: u64, tile: usize,
        generate: unsafe fn(u64, usize, *mut u8),
        generate_pair: flock_core::lincheck::RootPairGenerator,
    ) {
        let mut reference = Scratch::new();
        let mut fused = Scratch::new();
        let mut source = unsafe { Box::<GeneratedTile>::new_uninit().assume_init() };
        let mut rebuilt = unsafe { Box::<GeneratedTile>::new_uninit().assume_init() };
        unsafe {
            let src = source.0.as_mut_ptr().cast::<u8>();
            let dst = rebuilt.0.as_mut_ptr().cast::<u8>();
            generate(seed, tile, src);
            for pair in 0..4 {
                generate_pair(seed, tile, pair, dst, fused.deps.as_mut_ptr().cast(),
                    fused.out_lo.as_mut_ptr().cast(), fused.out_hi.as_mut_ptr().cast());
                blake3_pair(src, 2 * pair, &mut reference, &mut PfCur::new(None));
                for (a, b, size) in [
                    (reference.deps.as_ptr().cast::<u8>(), fused.deps.as_ptr().cast::<u8>(), core::mem::size_of_val(&fused.deps)),
                    (reference.out_lo.as_ptr().cast::<u8>(), fused.out_lo.as_ptr().cast::<u8>(), core::mem::size_of_val(&fused.out_lo)),
                    (reference.out_hi.as_ptr().cast::<u8>(), fused.out_hi.as_ptr().cast::<u8>(), core::mem::size_of_val(&fused.out_hi)),
                ] {
                    assert_eq!(core::slice::from_raw_parts(a, size), core::slice::from_raw_parts(b, size),
                        "generated dependencies seed={seed:x} tile={tile} pair={pair}");
                }
                for group in 0..GROUPS {
                    let at = group * GROUP_BYTES + pair * 128;
                    assert_eq!(core::slice::from_raw_parts(src.add(at), 128), core::slice::from_raw_parts(dst.add(at), 128));
                }
            }
        }
    }

    /// `out[t] = Σ_{i<64} w[i] · z[64 t + i]` straight from the tiles.
    fn fold64_generated(
        view: &WitnessTileView,
        out: &mut [F128],
        weights: &Weights,
        generate: unsafe fn(u64, usize, usize, *mut u8, *mut u8, *const u8),
    ) {
        const {
            assert!(core::mem::size_of::<Weights>() == 8192);
            assert!(core::mem::align_of::<Weights>() == 64);
        }
        assert!(view.cross_quarter && view.tiles == 4096);
        let seed = view.generator.expect("fold generator seed").0;
        let dst = SendPtr(out.as_mut_ptr());
        #[repr(C, align(64))]
        struct FoldOut([__m512i; 8]);
        thread_local! {static OUTPUT:core::cell::RefCell<Option<Box<FoldOut>>>=const {core::cell::RefCell::new(None)};}
        (0..view.tiles / 8).into_par_iter().for_each(|u| {
            let dst = dst;
            flock_core::rstrace::evd("z:task", u as u32);
            OUTPUT.with(|cell| {
                let mut guard = cell.borrow_mut();
                let output = guard
                    .get_or_insert_with(|| unsafe { Box::<FoldOut>::new_zeroed().assume_init() });
                for tile in u * 8..(u + 1) * 8 {
                    for pair in 0..4 {
                        unsafe {
                            generate(
                                seed,
                                tile,
                                pair,
                                output.0.as_mut_ptr().cast(),
                                core::ptr::null_mut(),
                                (weights as *const Weights).cast(),
                            );
                            let base = 64 * tile + 16 * pair;
                            for octa in 0..2 {
                                for parity in 0..2 {
                                    let lo = output.0[2 * octa + parity];
                                    let hi = output.0[4 + 2 * octa + parity];
                                    macro_rules! store {
                                        ($l:literal) => {{
                                            let c = compression(
                                                true,
                                                base + 8 * octa + 2 * $l + parity,
                                            );
                                            let value = _mm256_inserti128_si256::<1>(
                                                _mm256_castsi128_si256(
                                                    _mm512_extracti32x4_epi32::<$l>(lo),
                                                ),
                                                _mm512_extracti32x4_epi32::<$l>(hi),
                                            );
                                            _mm256_storeu_si256(dst.0.add(2 * c).cast(), value);
                                        }};
                                    }
                                    store!(0);
                                    store!(1);
                                    store!(2);
                                    store!(3);
                                }
                            }
                        }
                    }
                }
            });
        });
    }

    pub(crate) fn fold64(view: &WitnessTileView, out: &mut [F128], w: &[F128; 64]) {
        assert_eq!(out.len(), view.tiles * 64 * 2);
        let weights = Weights::new(w);
        if super::fused_fold_enabled() && let Some(generate)=view.stage_generator {
            fold64_generated(view,out,&weights,generate);
            return;
        }

        let dst = SendPtr(out.as_mut_ptr());
        let cq = view.cross_quarter;
        for_each_octa(view, |stage, t, o, pf| {
            let dst = dst;
            // SAFETY: AVX-512/VPCLMULQDQ are cfg preconditions; every
            // (compression, half) slot is written by exactly one octa; the
            // prefetch hints stay within the tile span.
            unsafe {
                let r = fold_stage(stage, &weights, pf);
                let base = 64 * t + 8 * o;
                for (h, halves) in r.iter().enumerate() {
                    for (parity, v) in halves.iter().enumerate() {
                        let mut lanes = [F128::ZERO; 4];
                        _mm512_storeu_si512(lanes.as_mut_ptr().cast(), *v);
                        for (l, value) in lanes.into_iter().enumerate() {
                            let c = compression(cq, base + 2 * l + parity);
                            *dst.0.add(2 * c + h) = value;
                        }
                    }
                }
            }
        });
    }

    /// Write the complete canonical witness into `z` from the tiles.
    pub(crate) fn materialize(view: &WitnessTileView, z: &mut [F128]) {
        assert_eq!(z.len(), view.tiles * 64 * (K / 128));
        let dst = SendPtr(z.as_mut_ptr().cast::<u64>());
        let cq = view.cross_quarter;
        for_each_octa(view, |stage, t, o, _| {
            let dst = dst;
            // SAFETY: Stage is NQ ZMMs = NQ x 8 qwords; each compression's
            // 256 qwords are written by exactly one octa lane.
            unsafe {
                let words = &*(stage.0.as_ptr().cast::<[[u64; 8]; NQ]>());
                for j in 0..8 {
                    let c = compression(cq, 64 * t + 8 * o + j);
                    let row = dst.0.add(c * (K / 64));
                    for (q, w) in words.iter().enumerate() {
                        *row.add(q) = w[j];
                    }
                    for q in NQ..K / 64 {
                        *row.add(q) = 0;
                    }
                }
            }
        });
    }

    /// Timing decomposition hooks (tests only).
    #[cfg(test)]
    pub(crate) fn expand_only(view: &WitnessTileView) -> u64 {
        let acc = std::sync::atomic::AtomicU64::new(0);
        for_each_octa(view, |stage, t, o, _| {
            if t % 64 == 0 && o == 7 {
                // SAFETY: plain read of the initialized stage.
                let w = unsafe { *(stage.0.as_ptr().add(NQ - 2).cast::<u64>()) };
                acc.fetch_xor(w, std::sync::atomic::Ordering::Relaxed);
            }
        });
        acc.into_inner()
    }

    #[cfg(test)]
    pub(crate) fn read_only(view: &WitnessTileView) -> u64 {
        if let Some((seed, generate)) = view.generator {
            #[repr(align(64))]
            struct Tile([core::mem::MaybeUninit<u8>; TILE_BYTES]);
            return (0..view.tiles).into_par_iter().map_init(
                || unsafe { Box::<Tile>::new_uninit().assume_init() },
                |tile, index| unsafe {
                    let p = tile.0.as_mut_ptr().cast::<u8>();
                    generate(seed, index, p);
                    let mut x = _mm512_setzero_si512();
                    for line in 0..TILE_BYTES / 64 {
                        x = _mm512_xor_si512(x, _mm512_load_si512(p.add(line * 64).cast()));
                    }
                    _mm512_reduce_or_epi64(x) as u64
                },
            ).reduce(|| 0, |a, b| a ^ b);
        }
        let n = view.tiles * TILE_BYTES / 64;
        let base = view.base;
        (0..n / 4096)
            .into_par_iter()
            .map(|c| {
                // SAFETY: in-bounds aligned reads of the live tile span.
                unsafe {
                    let mut x = _mm512_setzero_si512();
                    for i in c * 4096..(c + 1) * 4096 {
                        x = _mm512_xor_si512(x, _mm512_load_si512((base + i * 64) as *const __m512i));
                    }
                    _mm512_reduce_or_epi64(x) as u64
                }
            })
            .reduce(|| 0, |a, b| a ^ b)
    }

    enum Tiles {
        Installed(Arc<RootFoldSource>),
        Owned(WitnessRootTiles),
    }

    /// [`flock_core::pcs::elided_z::ZRegen`] over witness root tiles.
    pub(crate) struct TileRegen {
        tiles: Tiles,
    }

    impl TileRegen {
        /// Regenerator over tiles installed in `source` (shared with
        /// zerocheck/lincheck; kept alive until the open consumes it).
        pub(crate) fn installed(source: Arc<RootFoldSource>) -> Self {
            assert!(source.witness_root_tiles().and_then(|t| t.witness_view()).is_some());
            Self { tiles: Tiles::Installed(source) }
        }

        pub(crate) fn owned(tiles: WitnessRootTiles) -> Self {
            assert!(tiles.witness_view().is_some());
            Self { tiles: Tiles::Owned(tiles) }
        }

        pub(crate) fn view(&self) -> WitnessTileView {
            match &self.tiles {
                Tiles::Installed(s) => s.witness_root_tiles().and_then(|t| t.witness_view()),
                Tiles::Owned(t) => t.witness_view(),
            }
            .expect("witness tile view")
        }
    }

    impl flock_core::pcs::elided_z::ZRegen for TileRegen {
        fn fold64(&self, out: &mut [F128], w: &[F128; 64]) {
            fold64(&self.view(), out, w);
            super::REGEN_FOLDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        fn materialize(&self, z: &mut [F128]) {
            materialize(&self.view(), z);
            super::REGEN_MATERIALIZES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "avx512vbmi2",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
mod tests {
    use super::imp::TileRegen;
    use super::*;
    use flock_core::field::F128;
    use flock_core::pcs::elided_z::ZRegen;
    use rayon::prelude::*;

    fn weights(seed: u64) -> [F128; 64] {
        let mut s = seed;
        std::array::from_fn(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let lo = s;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            F128::new(lo, s)
        })
    }

    fn reference_fold64(z: &[F128], w: &[F128; 64]) -> Vec<F128> {
        z.par_chunks(64)
            .map(|c| {
                let mut v = F128::ZERO;
                for (a, b) in c.iter().zip(w) {
                    v += *a * *b;
                }
                v
            })
            .collect()
    }

    fn check(z: &[F128], regen: &TileRegen, tag: &str) {
        let mut zr = vec![F128::new(0xdead, 0xbeef); z.len()];
        regen.materialize(&mut zr);
        let bad = z.iter().zip(&zr).position(|(a, b)| a != b);
        assert!(bad.is_none(), "{tag}: regenerated z differs at element {bad:?} (block {:?})", bad.map(|i| i / 128));
        for seed in [3u64, 0x5eed_0f_f01d] {
            let w = weights(seed);
            let mut got = vec![F128::ZERO; z.len() / 64];
            regen.fold64(&mut got, &w);
            assert!(got == reference_fold64(z, &w), "{tag}: fold64 mismatch (seed {seed})");
        }
    }

    /// Non-pipelined ranked witness (block-order tiles, dense z): the
    /// regenerated z and its fold64 match the witness bit for bit.
    #[test]
    #[ignore = "ranked-scale (~3 GiB); run serially on the benchmark host"]
    fn zregen_matches_ranked_witness_block_order() {
        use super::super::generate_witness_with_ab_packed_and_round1_inner_from;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        pool.install(|| {
            let mut checked = 0;
            for (pass, seed) in [0x953a_81c7u64, 0x953a_81c7, 0x1234_5678_9abc, 7].into_iter().enumerate() {
                let blocks = crate::seed_pipe::BlockSource::closed(18, seed);
                let (z, a, b, mut ab) = generate_witness_with_ab_packed_and_round1_inner_from(blocks, 18);
                if let Some(tiles) = ab.take_witness_root_tiles() {
                    assert!(!tiles.z_elided());
                    let regen = TileRegen::owned(tiles);
                    check(&z, &regen, &format!("block-order pass={pass} seed={seed:#x}"));
                    checked += 1;
                }
                flock_core::scratch::give_f128(z);
                flock_core::scratch::give_f128(a);
                flock_core::scratch::give_f128(b);
            }
            assert!(checked >= 3, "warm ranked witness must publish root tiles");
            println!("ZREGEN_BLOCK_ORDER_OK passes={checked}");
        });
    }

    /// Pipelined (committed) ranked witness: cross-quarter tiles. With
    /// elision forced off z is dense and must equal the regeneration; with
    /// elision on, the tiles are flagged and still regenerate the dense z.
    #[test]
    #[ignore = "ranked-scale (~5 GiB); run serially on the benchmark host"]
    fn zregen_matches_ranked_witness_cross_quarter() {
        let mut setup = super::super::Blake3Setup::new(1 << 18);
        // The ranked worker's transcript and Merkle hash.
        setup.pcs_params.merkle_hash = flock_core::hash::HashKind::Blake3;
        let params = setup.pcs_params.clone();
        let padding = setup.r1cs.padding_spec();
        let odd_tail = flock_core::ntt::additive_ntt_f128::ZeroOddTailLanes::lanes_for_padding(
            params.num_ntts(),
            padding.k_log,
            padding.useful_bits_per_block,
        );
        // Run from the test thread on the global pool: the paired seed
        // schedule (the only elidable producer) requires whole-pool custody.
        {
            let mut dense_checked = 0;
            let mut elided_checked = 0;
            for seed in [0xC0FFEEu64, 0xC0FFEE, 0x51DE_5EED] {
                let run = |elide: bool| {
                    FORCE_OFF.store(!elide, Ordering::Relaxed);
                    let codeword = vec![F128::ZERO; 1 << 26];
                    let (cw, (witness, a, b, mut ab)) = super::super::generate_witness_and_commit_from(
                        crate::seed_pipe::BlockSource::closed(18, seed),
                        18,
                        &params,
                        odd_tail,
                        Some(codeword),
                    );
                    FORCE_OFF.store(false, Ordering::Relaxed);
                    drop(cw);
                    let (z, committed) = witness.into_parts();
                    let root = committed.as_ref().map(|(c, _)| c.clone());
                    drop(committed);
                    flock_core::scratch::give_f128(a);
                    flock_core::scratch::give_f128(b);
                    (z, ab.take_witness_root_tiles(), root)
                };
                let (z, tiles, root_dense) = run(false);
                let Some(tiles) = tiles else {
                    flock_core::scratch::give_f128(z);
                    continue;
                };
                assert!(!tiles.z_elided());
                let cq = tiles.witness_view().unwrap().cross_quarter;
                check(&z, &TileRegen::owned(tiles), &format!("dense seed={seed:#x} cq={cq}"));
                dense_checked += usize::from(cq);
                let (mut z2, tiles2, root_elided) = run(true);
                if let Some(tiles2) = tiles2 {
                    if tiles2.z_elided() {
                        assert_eq!(format!("{root_dense:?}"), format!("{root_elided:?}"), "commitment changed under elision");
                        let regen = TileRegen::owned(tiles2);
                        let mut got = vec![F128::ZERO; z.len() / 64];
                        let w = weights(seed);
                        regen.fold64(&mut got, &w);
                        assert!(got == reference_fold64(&z, &w), "elided fold64 mismatch seed={seed:#x}");
                        regen.materialize(&mut z2);
                        assert!(z2 == z, "elided materialize mismatch seed={seed:#x}");
                        elided_checked += 1;
                    }
                }
                flock_core::scratch::give_f128(z);
                flock_core::scratch::give_f128(z2);
            }
            assert!(dense_checked >= 1, "no cross-quarter tiles were produced");
            assert!(elided_checked >= 1, "elision never engaged");
            println!("ZREGEN_CROSS_QUARTER_OK dense={dense_checked} elided={elided_checked}");
        }
    }

    /// End to end: the ranked prove with z elision produces the same proof
    /// bytes as the dense witness, and actually folds from the regenerator.
    #[test]
    #[ignore = "ranked-scale prove; run serially on the benchmark host"]
    fn zregen_full_proof_bytes_identical() {
        use std::sync::atomic::Ordering::Relaxed;
        let mut setup = super::super::Blake3Setup::new(1 << 18);
        // The ranked worker's transcript and Merkle hash.
        setup.pcs_params.merkle_hash = flock_core::hash::HashKind::Blake3;
        {
            let prove = |elide: bool, seed: u64| {
                FORCE_OFF.store(!elide, Relaxed);
                let mut ch = crate::challenger::FsChallenger::with_hash(
                    b"zr-proof-identity",
                    flock_core::hash::HashKind::Blake3,
                );
                let (proof, commitment, _claim) =
                    setup.prove_fast_inner(crate::seed_pipe::BlockSource::closed(18, seed), &mut ch);
                FORCE_OFF.store(false, Relaxed);
                crate::proof_io::R1csProofBundleLigerito { commitment, proof }.to_bytes()
            };
            // Warm the scratch pool so the pipelined (elidable) commit runs.
            let _ = prove(false, 1);
            let _ = prove(true, 2);
            let mut engaged = 0;
            for seed in [0x5eed_0001u64, 0x5eed_0002, 0x5eed_0003] {
                let dense = prove(false, seed);
                let folds = REGEN_FOLDS.load(Relaxed);
                let mats = REGEN_MATERIALIZES.load(Relaxed);
                let elided = prove(true, seed);
                let used_fold = REGEN_FOLDS.load(Relaxed) - folds;
                let used_mat = REGEN_MATERIALIZES.load(Relaxed) - mats;
                assert!(dense == elided, "proof bytes differ under z elision (seed {seed:#x}, folds {used_fold}, materializes {used_mat})");
                println!("ZREGEN_PROOF seed={seed:#x} bytes={} regen_folds={used_fold} fallback_materializes={used_mat}", dense.len());
                engaged += used_fold;
            }
            assert!(engaged >= 1, "z elision never reached the regenerator fold");
            println!("ZREGEN_PROOF_OK engaged={engaged}");
        }
    }

    /// Timing decomposition on the production (cross-quarter) tile layout:
    /// dense fold of z vs regenerated fold, expansion only, tile read only.
    #[test]
    #[ignore = "ranked-scale timing; run alone on the benchmark host"]
    fn zregen_timing() {
        let mut setup = super::super::Blake3Setup::new(1 << 18);
        setup.pcs_params.merkle_hash = flock_core::hash::HashKind::Blake3;
        let params = setup.pcs_params.clone();
        let padding = setup.r1cs.padding_spec();
        let odd_tail = flock_core::ntt::additive_ntt_f128::ZeroOddTailLanes::lanes_for_padding(
            params.num_ntts(),
            padding.k_log,
            padding.useful_bits_per_block,
        );
        let mut found = None;
        for seed in 11u64..19 {
            FORCE_OFF.store(true, Ordering::Relaxed);
            let (cw, (witness, a, b, mut ab)) = super::super::generate_witness_and_commit_from(
                crate::seed_pipe::BlockSource::closed(18, seed),
                18,
                &params,
                odd_tail,
                Some(vec![F128::ZERO; 1 << 26]),
            );
            FORCE_OFF.store(false, Ordering::Relaxed);
            drop(cw);
            let (z, committed) = witness.into_parts();
            drop(committed);
            flock_core::scratch::give_f128(a);
            flock_core::scratch::give_f128(b);
            if let Some(t) = ab.take_witness_root_tiles() {
                if t.witness_view().unwrap().cross_quarter {
                    found = Some((z, t));
                    break;
                }
            }
            // Return z to the tagged pool so the warm exact producer engages.
            flock_core::scratch::give_f128(z);
        }
        let (z, tiles) = found.expect("cross-quarter tiles");
        let regen = TileRegen::owned(tiles);
        let view = regen.view();
        let w16: [F128; 16] = std::array::from_fn(|i| weights(5)[i]);
        let (r4, r5) = (F128::new(0x1234, 0x5678), F128::new(0x9abc, 0xdef0));
        let e4 = [F128::ONE + r4, r4];
        let e5 = [F128::ONE + r5, r5];
        let w: [F128; 64] = std::array::from_fn(|i| w16[i & 15] * e4[(i >> 4) & 1] * e5[i >> 5]);
        let mut dense = vec![F128::ZERO; z.len() / 64];
        let mut regd = vec![F128::ZERO; z.len() / 64];
        let time = |f: &mut dyn FnMut()| {
            let mut v: Vec<f64> = (0..9)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[0], v[4])
        };
        let mut sink = 0u64;
        let td = time(&mut || flock_core::pcs::ligerito::bench_direct_fold8_f_dense(&z, &mut dense, &w16, r4, r5));
        let tr = time(&mut || regen.fold64(&mut regd, &w));
        let te = time(&mut || sink ^= super::imp::expand_only(&view));
        let tt = time(&mut || sink ^= super::imp::read_only(&view));
        assert!(dense == regd, "timing fold mismatch");
        println!(
            "ZREGEN_TIMING dense_fold min={:.3} med={:.3} | regen_fold min={:.3} med={:.3} | expand_only min={:.3} med={:.3} | tile_read min={:.3} med={:.3} ms (sink {sink:x})",
            td.0, td.1, tr.0, tr.1, te.0, te.1, tt.0, tt.1
        );
        let mut parts = Vec::new();
        for (mode, name) in [(1u8, "no_blake3"), (2, "no_stage"), (3, "fold_only")] {
            super::imp::TEST_MODE.store(mode, Ordering::Relaxed);
            let tf = time(&mut || regen.fold64(&mut regd, &w));
            let tx = time(&mut || sink ^= super::imp::expand_only(&view));
            parts.push(format!("{name}: fold {:.3}/{:.3} expand {:.3}/{:.3}", tf.0, tf.1, tx.0, tx.1));
        }
        super::imp::TEST_MODE.store(0, Ordering::Relaxed);
        println!("ZREGEN_PARTS {} (min/med ms, sink {sink:x})", parts.join(" | "));
    }
}
