//! Three-variable composed fold (ZC-3VAR): round two's sweep also carries the
//! deferred **round-four** message as a biquadratic in `(ρ₁, ρ₂)`, so the
//! packed composed pass folds `X₁, X₂, X₃` at once (`ρ₁, ρ₂, ρ₃`) and writes
//! an eighth-size table instead of a quarter-size one. Every later cascade
//! level then works on half the data it used to.
//!
//! Value identity: with `A''(ρ₁, ρ₂)` the two-level fold of an eight-row group
//! (rows `8w + i + 2j + 4t`), the round-four product `A''·B''` has degree ≤ 2
//! in each of `ρ₁, ρ₂`, so it is pinned by its values on `{0, 1, ∞}²`
//! (`∞` = leading coefficient, i.e. the XOR of the two endpoint rows). The
//! sweep accumulates those nine eq-weighted values for `t = 1` and `t = ∞`;
//! [`eval_round4_lookahead`] re-expands them at the sampled challenges. F128
//! arithmetic is exact, so the round-four message is the same field element
//! the incumbent derives from its folded tables, and the transcript is
//! byte-identical. `FLOCK_NO_ZC_3VAR=1` restores the two-variable route.

use super::*;

/// Deferred round-four message. `v1[i + 3j]` / `vinf[i + 3j]` are the
/// eq₄-weighted sums of the products at `(ρ₁, ρ₂) = (P_i, P_j)`, `P = (0, 1, ∞)`,
/// for the round-four points `t = 1` and `t = ∞`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Round4Lookahead {
    pub v1: [F128; 9],
    pub vinf: [F128; 9],
}

impl Round4Lookahead {
    pub const ZERO: Self = Self {
        v1: [F128::ZERO; 9],
        vinf: [F128::ZERO; 9],
    };

    pub fn add(&mut self, o: &Self) {
        for k in 0..9 {
            self.v1[k] += o.v1[k];
            self.vinf[k] += o.vinf[k];
        }
    }

    pub fn scaled(&self, s: F128) -> Self {
        Self {
            v1: self.v1.map(|x| s * x),
            vinf: self.vinf.map(|x| s * x),
        }
    }

    /// From the 18-entry kernel layout `[v1; 9] ++ [vinf; 9]`.
    pub fn from_flat(f: &[F128; 18]) -> Self {
        Self {
            v1: std::array::from_fn(|k| f[k]),
            vinf: std::array::from_fn(|k| f[9 + k]),
        }
    }
}

/// Lagrange-style basis on `{0, 1, ∞}` for a quadratic `q`:
/// `q(ρ) = q(0)(1+ρ) + q(1)ρ + q(∞)(ρ+ρ²)` (characteristic two).
#[inline]
fn basis3(r: F128) -> [F128; 3] {
    [F128::ONE + r, r, r + r * r]
}

/// Evaluate the deferred round-four message at the sampled `(ρ₁, ρ₂)`.
/// `r_next[0] = ONE` (Convention A) exactly as for round three.
pub fn eval_round4_lookahead(la: &Round4Lookahead, rho1: F128, rho2: F128) -> (F128, F128) {
    let e1 = basis3(rho1);
    let e2 = basis3(rho2);
    let mut g1 = F128::ZERO;
    let mut ginf = F128::ZERO;
    for j in 0..3 {
        let mut s1 = F128::ZERO;
        let mut sinf = F128::ZERO;
        for i in 0..3 {
            s1 += e1[i] * la.v1[i + 3 * j];
            sinf += e1[i] * la.vinf[i + 3 * j];
        }
        g1 += e2[j] * s1;
        ginf += e2[j] * sinf;
    }
    (g1, ginf)
}

/// The nine `{0,1,∞}²` points of a bilinear form given its four corner rows
/// `x[i + 2j]`, in `[i + 3j]` order.
#[inline]
pub(crate) fn corners9(x: &[F128]) -> [F128; 9] {
    let (c0, c1, c2, c3) = (x[0], x[1], x[2], x[3]);
    [
        c0,
        c1,
        c0 + c1,
        c2,
        c3,
        c2 + c3,
        c0 + c2,
        c1 + c3,
        c0 + c1 + c2 + c3,
    ]
}

/// Portable full-pass reference of the round-four lookahead straight from the
/// packed witness (test oracle, `FLOCK_ZC_3VAR_CHECK=1` cross-check, and the
/// fallback when the AVX-512 sweep is not available).
pub(crate) fn round4_lookahead_ref(
    a_packed: &[u8],
    b_packed: &[u8],
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
    k_skip: usize,
) -> Round4Lookahead {
    use rayon::prelude::*;
    let n_chunks = table.n_chunks;
    let n_rows = a_packed.len() / n_chunks;
    let (mask, useful) = round2_pair_skip(padding, k_skip);
    let eq4 = build_eq(&mlv_challenges[3..]);
    assert_eq!(eq4.len() * 8, n_rows);
    let row = |p: &[u8], r: usize| -> F128 {
        if ((r / 2) & mask) >= useful {
            F128::ZERO
        } else {
            table.fold_one_row(&p[r * n_chunks..(r + 1) * n_chunks])
        }
    };
    (0..n_rows / 8)
        .into_par_iter()
        .with_min_len(1024)
        .map(|w| {
            let a: [F128; 8] = std::array::from_fn(|i| row(a_packed, 8 * w + i));
            let b: [F128; 8] = std::array::from_fn(|i| row(b_packed, 8 * w + i));
            let c: [F128; 4] = std::array::from_fn(|k| a[k] + a[k + 4]);
            let d: [F128; 4] = std::array::from_fn(|k| b[k] + b[k + 4]);
            let (pa, pb) = (corners9(&a[4..8]), corners9(&b[4..8]));
            let (pc, pd) = (corners9(&c), corners9(&d));
            let e = eq4[w];
            Round4Lookahead {
                v1: std::array::from_fn(|k| e * (pa[k] * pb[k])),
                vinf: std::array::from_fn(|k| e * (pc[k] * pd[k])),
            }
        })
        .reduce(
            || Round4Lookahead::ZERO,
            |mut x, y| {
                x.add(&y);
                x
            },
        )
}

/// Round-four contribution of the two canonical windows the ranked sweep
/// elides (see [`round2_canonical_window_deltas`] for the brand layout):
///
/// * ones rows `0..16` (`B` folds to the constant `v0`): only the `t = 1`
///   corners with `i, j ∈ {0,1}` survive (every `∞` point is a `B`
///   difference, and every `t = ∞` row is `B + B = 0`):
///   `ΔV1[i+3j] = v0 · Σ_{w'<2} eq₅(w')·F(8w' + 4 + i + 2j)`;
/// * sparse row 240 (`B = v1`, rows 241..256 zero) is local row 0 of group
///   `w' = 30`: only `t = ∞` points that include corner `(0,0)` survive:
///   `ΔVinf[(0,0),(∞,0),(0,∞),(∞,∞)] = v1 · eq₅(30) · F(240)`;
///
/// with `eq₅ = eq(mlv[3..8])` over the in-block group index.
pub(crate) fn round4_canonical_window_deltas(
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    rows: &[F128],
) -> Round4Lookahead {
    assert_eq!(table.n_chunks, 8);
    assert_eq!(rows.len(), CANON_ROWS * 64);
    assert!(mlv_challenges.len() >= 8);
    let cols: [F128; 64] =
        std::array::from_fn(|s| table.data[(s / 8) * 256 + (1usize << (s % 8))]);
    let fold = |slot: usize| -> F128 {
        crate::pcs::ring_switch::inner_product(&cols, &rows[slot * 64..(slot + 1) * 64])
    };
    let mut v1 = F128::ZERO;
    for &col in &cols[..49] {
        v1 += col;
    }
    let mut v0 = v1;
    for &col in &cols[49..] {
        v0 += col;
    }
    let r3 = mlv_challenges[3];
    let top4 = (F128::ONE + mlv_challenges[4])
        * (F128::ONE + mlv_challenges[5])
        * (F128::ONE + mlv_challenges[6])
        * (F128::ONE + mlv_challenges[7]);
    let hi4 = mlv_challenges[4] * mlv_challenges[5] * mlv_challenges[6] * mlv_challenges[7];
    let eq5 = [(F128::ONE + r3) * top4, r3 * top4];
    let eq5_30 = (F128::ONE + r3) * hi4;
    let mut d = Round4Lookahead::ZERO;
    for j in 0..2 {
        for i in 0..2 {
            let mut s = F128::ZERO;
            for (wp, e) in eq5.iter().enumerate() {
                s += *e * fold(8 * wp + 4 + i + 2 * j);
            }
            d.v1[i + 3 * j] = v0 * s;
        }
    }
    let sp = v1 * (eq5_30 * fold(16));
    for k in [0usize, 2, 6, 8] {
        d.vinf[k] = sp;
    }
    d
}

/// Remove the three least-significant coordinates from an eq tensor (each
/// eight-entry group of a rank-one tensor XORs to its suffix weight).
/// Lane gl1 (`FLOCK_NO_GL1_R345_PRE`): the rounds 3+4+5 pass's challenge-
/// independent tables from round two's low eq tensor — the marginalized
/// low tensor and its `(w, w·x⁶⁴)` pair table — so round one's idle AB
/// thread can build them. Same functions as the pass's own setup.
pub(crate) fn r345_pre_tables(peq_lo: &[F128]) -> (Vec<F128>, Option<Vec<F128>>) {
    let lo = marginalize_eq_low3(peq_lo);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    let w = lo.len().is_multiple_of(8).then(|| build_w_pair_table(&lo));
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "vpclmulqdq")))]
    let w = None;
    (lo, w)
}

pub(crate) fn marginalize_eq_low3(eq: &[F128]) -> Vec<F128> {
    assert!(eq.len().is_power_of_two() && eq.len() >= 8);
    eq.chunks_exact(8)
        .map(|v| v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7])
        .collect()
}

/// Three-variable folds of the four static B lines of a ranked block, in line
/// order `[0, 1, 30, 31]` (a line is eight packed rows, one output of the
/// pass).
///
/// Ranked witness brand, per 256-row block (the same facts
/// [`round2_canonical_window_deltas`] uses): B rows `0..16` are all ones, row
/// 240 is the static word `0x0001_ffff_ffff_ffff`, rows `241..256` are zero.
/// With `v0 = Σ_s col_s` and `v1 = Σ_{s<49} col_s` the images of those two
/// row words, the composed coefficients of an all-ones line sum to one, so
/// lines 0 and 1 fold to `v0`; line 30 holds one live row at coefficient
/// `(1+ρ₁)(1+ρ₂)(1+ρ₃)`; line 31 is zero. Exact field identities, so these
/// are the elements the sweep would compute from the rows.
pub(crate) fn fold3_b_canonical_lines(table: &UniSkipFoldTable, rho: [F128; 3]) -> [F128; 4] {
    assert_eq!(table.n_chunks, 8);
    let col = |s: usize| table.data[(s / 8) * 256 + (1usize << (s % 8))];
    let mut v1 = F128::ZERO;
    for s in 0..49 {
        v1 += col(s);
    }
    let mut v0 = v1;
    for s in 49..64 {
        v0 += col(s);
    }
    let one = F128::ONE;
    let row240 = ((one + rho[0]) * (one + rho[1])) * v1;
    [v0, v0, row240 + rho[2] * row240, F128::ZERO]
}

/// Portable leaf of the three-variable packed pass for one worker chunk:
/// output `x` ← packed rows `8x..8x+8` folded at `ρ₁, ρ₂, ρ₃`, then the
/// round-five message and round-six lookahead partials in the same eight-sum
/// odd-lane layout as [`fold2_from_packed_lookahead_scalar`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold3_from_packed_lookahead_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    table: &UniSkipFoldTable,
    out_base: usize,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho: [F128; 3],
    eq_lo: &[F128],
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [F128; 8] {
    let n_chunks = table.n_chunks;
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());
    let fold_row = |p: &[u8], r: usize| -> F128 {
        if ((r / 2) & pair_in_block_mask) >= useful_pairs_inclusive {
            F128::ZERO
        } else {
            table.fold_one_row(&p[r * n_chunks..(r + 1) * n_chunks])
        }
    };
    let fold8 = |p: &[u8], x: usize| -> F128 {
        let r: [F128; 8] = std::array::from_fn(|i| fold_row(p, 8 * x + i));
        let t: [F128; 4] = std::array::from_fn(|h| r[2 * h] + rho[0] * (r[2 * h] + r[2 * h + 1]));
        let u0 = t[0] + rho[1] * (t[0] + t[1]);
        let u1 = t[2] + rho[1] * (t[2] + t[3]);
        u0 + rho[2] * (u0 + u1)
    };
    let mut acc = [F256Unreduced::ZERO; 8];
    for u in 0..eq_lo.len() / 2 {
        let o = 4 * u;
        let a: [F128; 4] = std::array::from_fn(|k| fold8(a_packed, out_base + o + k));
        let b: [F128; 4] = std::array::from_fn(|k| fold8(b_packed, out_base + o + k));
        a_out[o..o + 4].copy_from_slice(&a);
        b_out[o..o + 4].copy_from_slice(&b);
        let [a0, a1, a2, a3] = a;
        let [b0, b1, b2, b3] = b;
        let wt = eq_lo[2 * u + 1];
        let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
        acc[0] ^= a1w.mul_unreduced(b1);
        acc[1] ^= (a0w + a1w).mul_unreduced(b0 + b1);
        acc[2] ^= a3w.mul_unreduced(b3);
        acc[3] ^= (a2w + a3w).mul_unreduced(b2 + b3);
        acc[4] ^= a2w.mul_unreduced(b2);
        let (e_aw, e_b) = (a0w + a2w, b0 + b2);
        let (o_aw, o_b) = (a1w + a3w, b1 + b3);
        acc[5] ^= e_aw.mul_unreduced(e_b);
        acc[6] ^= o_aw.mul_unreduced(o_b);
        acc[7] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
    }
    acc.map(|x| x.reduce())
}

/// Three-variable composed pass from the packed witness: folds `X₁..X₃` at
/// `ρ₁..ρ₃` into `a_out`/`b_out` (length `2^(m−k_skip−3)`), and returns the
/// round-five message `(r_next5[0]·G₅(1), G₅(∞))` plus the deferred round-six
/// quadratic. `r_next5 = [1, mlv[4..]]`, `r_next5[1] ≠ 0`.
///
/// The AVX-512 route reuses the incumbent rounds-3+4 c4 register fold
/// (`ρ₁, ρ₂` baked into the GFNI matrices) and applies the `ρ₃` level in
/// registers before the store.
///
/// `b_canonical` asserts the ranked witness brand on `b_packed` (see
/// [`fold3_b_canonical_lines`]); the AVX-512 route then takes B's four
/// static lines per block from constants instead of folding them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold3_from_packed_and_round_pair_lookahead_into_with_eq(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    padding: &PaddingSpec,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho: [F128; 3],
    r_next5: &[F128],
    eq_override: Option<(&[F128], &[F128])>,
    out_lm: bool,
    b_canonical: bool,
) -> (F128, F128, Round3Lookahead) {
    let (m1, mi, la, _) = fold3_from_packed_and_round_pair_lookahead_casc(
        a_packed, b_packed, m, k_skip, table, padding, a_out, b_out, rho, r_next5, eq_override,
        out_lm, b_canonical, false, None,
    );
    (m1, mi, la)
}

/// Whether the AVX-512 b-canonical route of the rounds 3+4+5 pass (the only
/// one that implements the deeper cascade) is switched on in this process.
/// Shape conditions (`n % 256`, `lo_size % 16`) are checked by the pass.
pub(crate) fn casc_route_enabled() -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    {
        kernels::x86_64::zc_c3_enabled()
            && zc_gfni_enabled()
            && cfold_bake_enabled()
            && kernels::x86_64::zc_wtab_enabled()
            && kernels::x86_64::zc_c4_lm_enabled()
            && kernels::x86_64::zc_r345_b_canon_enabled()
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )))]
    {
        false
    }
}

/// [`fold3_from_packed_and_round_pair_lookahead_into_with_eq`] plus, when
/// `casc` and the b-canonical AVX-512 route runs, the deferred **round-seven**
/// message as a biquadratic in `(ρ₄, ρ₅)` (returned `Some`), in which case the
/// tables are stored in the octet layout of
/// [`fold3_from_packed_lookahead_c4_lm_wtab_b_canon_casc_x86`]. Needs
/// `r_next5[1], r_next5[2] ≠ 0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold3_from_packed_and_round_pair_lookahead_casc(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    padding: &PaddingSpec,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho: [F128; 3],
    r_next5: &[F128],
    eq_override: Option<(&[F128], &[F128])>,
    out_lm: bool,
    b_canonical: bool,
    casc: bool,
    pre_wtab: Option<&[F128]>,
) -> (F128, F128, Round3Lookahead, Option<Round4Lookahead>) {
    use rayon::prelude::*;
    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n = 1usize << (m - k_skip);
    assert!(n >= 32);
    assert_eq!(a_packed.len(), n * n_chunks);
    assert_eq!(b_packed.len(), n * n_chunks);
    let eighth = n / 8;
    assert_eq!(a_out.len(), eighth);
    assert_eq!(b_out.len(), eighth);
    assert_eq!(r_next5.len(), m - k_skip - 3);
    let r = r_next5[1];
    assert_ne!(r, F128::ZERO, "3var lookahead requires a non-zero parity weight");

    let eq_owned;
    let (eq_lo, eq_hi) = if let Some(eq) = eq_override {
        eq
    } else {
        let n_vars = r_next5.len() - 1;
        eq_owned = SplitEqGhash::with_n_hi(&r_next5[1..], packed_split_n_hi(n_vars));
        (&eq_owned.lo[..], &eq_owned.hi[..])
    };
    let lo_size = eq_lo.len();
    let hi_size = eq_hi.len();
    assert!(lo_size >= 2);
    assert_eq!(lo_size * hi_size * 2, eighth);
    crate::rstrace::ev("r345:s0", 0);
    let (kappa, r_inv) = lookahead_inv_factors(r);
    let chunk_out = 2 * lo_size;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);
    crate::rstrace::ev("r345:s1", 0);

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let fast = {
        let cfold = (zc_gfni_enabled() && cfold_bake_enabled()).then(|| {
            let one = F128::ONE;
            let (r1, r2) = (rho[0], rho[1]);
            kernels::x86_64::build_cfold_mats(
                &table.data,
                [(one + r1) * (one + r2), r1 * (one + r2), (one + r1) * r2, r1 * r2],
            )
        });
        let wtab = (kernels::x86_64::zc_wtab_enabled() && lo_size.is_multiple_of(8)).then(|| {
            match pre_wtab {
                // Lane gl1: built from the same eq_lo during round one.
                Some(w) if w.len() == eq_lo.len() => w.to_vec(),
                _ => build_w_pair_table(eq_lo),
            }
        });
        let ok = kernels::x86_64::fold2_packed_c4_lm_wtab_route_ok(
            cfold.is_some(),
            kernels::x86_64::zc_c4_lm_enabled(),
            lo_size,
            cfold.is_some(),
            wtab.is_some(),
        );
        let nt_out = kernels::x86_64::zc_fold_nt_enabled()
            && eighth >= (1 << 23)
            && (a_out.as_ptr() as usize) % 16 == 0
            && (b_out.as_ptr() as usize) % 16 == 0;
        if ok { Some((cfold.unwrap(), wtab.unwrap(), nt_out)) } else { None }
    };
    crate::rstrace::ev("r345:s2", 0);
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )))]
    let fast: Option<((), (), bool)> = None;
    assert!(
        !out_lm || fast.is_some(),
        "lane-major 3var output needs the AVX-512 route"
    );
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let b_canon = (b_canonical
        && fast.is_some()
        && n.is_multiple_of(256)
        && kernels::x86_64::zc_r345_b_canon_enabled())
    .then(|| fold3_b_canonical_lines(table, rho));
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let casc = casc && b_canon.is_some() && kernels::x86_64::zc_c3_enabled();
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )))]
    let casc = {
        let _ = casc;
        false
    };
    if casc {
        assert_ne!(r_next5[2], F128::ZERO, "round-seven lookahead needs r6 != 0");
    }
    // Deeper cascade: both lanes of an eight-group ride the odd lane's
    // weight; κ₆ restores the even lanes' own weight in the message sums.
    crate::rstrace::ev("r345:s3", 0);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_casc = casc.then(|| build_w_pair_table_casc(eq_lo));
    let kappa6 = if casc { (F128::ONE + r_next5[2]) * r_next5[2].inv() } else { F128::ZERO };
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )))]
    let _ = b_canonical;

    crate::rstrace::ev("r345:par0", 0);
    let leaf = |(x_hi, (a_out, b_out)): (usize, (&mut [F128], &mut [F128]))| {
            crate::rstrace::item_start();
            let out_base = x_hi * chunk_out;
            let mut la7 = [F128::ZERO; 18];
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512vbmi",
                target_feature = "gfni",
                target_feature = "vpclmulqdq"
            ))]
            let out = if let Some((c, wt, nt)) = fast.as_ref() {
                let (ap, bp) = (a_packed.as_ptr(), b_packed.as_ptr());
                // SAFETY: the packed inputs expose 8 readable bytes for every
                // row 8·out_base .. 8·(out_base + chunk_out); the cfg gate
                // supplies every feature; lo_size is a multiple of 16.
                unsafe {
                    if casc && let Some(bc) = b_canon.as_ref() {
                        macro_rules! casc {
                            ($nt:literal) => {
                                kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_b_canon_casc_x86::<$nt, true, true>(
                                    ap, bp, out_base, a_out, b_out, lo_size, c, rho[2],
                                    wtab_casc.as_deref().unwrap(), bc, &mut la7, kappa6,
                                )
                            };
                        }
                        let _ = wt;
                        if *nt { casc!(true) } else { casc!(false) }
                    } else if let Some(bc) = b_canon.as_ref() {
                        match (*nt, out_lm) {
                            (true, true) => kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_b_canon_x86::<true, true>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, rho[2], wt, bc,
                            ),
                            (true, false) => kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_b_canon_x86::<true, false>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, rho[2], wt, bc,
                            ),
                            (false, true) => kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_b_canon_x86::<false, true>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, rho[2], wt, bc,
                            ),
                            (false, false) => kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_b_canon_x86::<false, false>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, rho[2], wt, bc,
                            ),
                        }
                    } else {
                        match (*nt, out_lm) {
                            (true, true) => kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_x86::<true, true>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, rho[2], wt,
                            ),
                            (true, false) => kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_x86::<true, false>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, rho[2], wt,
                            ),
                            (false, true) => kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_x86::<false, true>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, rho[2], wt,
                            ),
                            (false, false) => kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_x86::<false, false>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, rho[2], wt,
                            ),
                        }
                    }
                }
            } else {
                fold3_from_packed_lookahead_scalar(
                    a_packed,
                    b_packed,
                    table,
                    out_base,
                    a_out,
                    b_out,
                    rho,
                    eq_lo,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512vbmi",
                target_feature = "gfni",
                target_feature = "vpclmulqdq"
            )))]
            let out = {
                let _ = &fast;
                fold3_from_packed_lookahead_scalar(
                    a_packed,
                    b_packed,
                    table,
                    out_base,
                    a_out,
                    b_out,
                    rho,
                    eq_lo,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            let eq_h = eq_hi[x_hi];
            crate::rstrace::item_end();
            (
                eq_h * out[0],
                eq_h * out[1],
                [
                    eq_h * out[2],
                    eq_h * out[3],
                    eq_h * out[4],
                    eq_h * out[5],
                    eq_h * out[6],
                    eq_h * out[7],
                ],
                if casc { la7.map(|v| eq_h * v) } else { la7 },
            )
    };
    type R345Sums = (F128, F128, [F128; 6], [F128; 18]);
    let id = || -> R345Sums { (F128::ZERO, F128::ZERO, [F128::ZERO; 6], [F128::ZERO; 18]) };
    let op = |(s1, sinf, sa, sl): R345Sums, (c1, cinf, ca, cl): R345Sums| -> R345Sums {
                let mut a = sa;
                for (x, y) in a.iter_mut().zip(ca.iter()) {
                    *x += *y;
                }
                let mut l = sl;
                for (x, y) in l.iter_mut().zip(cl.iter()) {
                    *x += *y;
                }
                (s1 + c1, sinf + cinf, a, l)
            };
    // Lane ru flat claim route (`FLOCK_NO_RU_FLAT_PAR`), see `super::flat_tail`.
    let (sum1_even, sum_inf_even, agg, la7_acc) =
        if crate::flat_par::enabled() && rayon::current_thread_index().is_some() {
            super::flat_chunks(a_out, b_out, chunk_out, leaf, id, op)
        } else {
            a_out
                .par_chunks_mut(chunk_out)
                .zip(b_out.par_chunks_mut(chunk_out))
                .enumerate()
                .map(leaf)
                .reduce(id, op)
        };
    crate::rstrace::ev("r345:par1", 0);
    crate::rstrace::flush_items("r345:first", "r345:last");
    let sum1 = kappa * sum1_even + agg[0];
    let sum_inf = kappa * sum_inf_even + agg[1];
    let la = lookahead_from_odd_weighted(&agg, r_inv);
    // Every round-seven term rides the x5 = x6 = 1 weight r5·r6·ω.
    let la7 = casc.then(|| {
        Round4Lookahead::from_flat(&la7_acc).scaled((r_next5[1] * r_next5[2]).inv())
    });
    (r_next5[0] * sum1, sum_inf, la, la7)
}
