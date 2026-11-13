//! Multilinear sumcheck — rounds 2..(m − k_skip + 1) of the zerocheck protocol.
//!
//! After the round-1 URM and the verifier's univariate-skip fold-point `z`, the
//! protocol enters a standard multilinear sumcheck over `n = m − k_skip` variables.
//! For the **extract_c** variant, only AB participate (C was pinned down at round
//! 1 as `res_C_lifted`), so the polynomial we sumcheck is
//!
//!   `Σ_x eq(r_rest, x) · a_mlv(x) · b_mlv(x)`
//!
//! with claim `P^{AB}(z)` from round 1. Each subsequent round sends `(P_r(1),
//! P_r(∞))` via the Karatsuba ∞-trick.
//!
//! This module begins with the **naive reference** (separately compute the
//! Lagrange-weighted fold, then a direct sum for the round-2 message). The
//! optimized fused-fold-plus-round-2 implementation (`uni_skip_fold_and_compute
//! _round_pair_ghash` in the C++) will be added next and cross-checked against
//! these naive functions.
//!
//! **Index convention** (matches the C++ extract_c pipeline's `sumcheck_round_pair`
//! and the NEON `fold_in_place_pair`): the **low bit** of the multilinear index
//! is bound first. So `a_mlv[2k]` is the X=0 value and `a_mlv[2k+1]` is the X=1
//! value, paired by the round message and the fold.
//!
//! For `mlv_challenges = [r_0, …, r_{n-1}]` (one per round) built so `build_eq`
//! places `r_i` at bit i, **round r=2 uses `mlv_challenges[0]`** for the
//! variable being bound, with eq over `mlv_challenges[1..]` for the remaining
//! variables. Subsequent rounds peel off `mlv_challenges[1]`, etc.
//!
//! **Round message format** (matches the C++): returns `(r_now · G(1), G(∞))`
//! where `r_now` is the challenge for the variable being bound *this* round.
//! The protocol polynomial sent is `Π(X) = eq(r_now, X) · G(X)` of degree 3;
//! at X=1 it equals `r_now · G(1)`, and the leading coefficient is `G(∞)`.
//! Verifier reconstructs `G(0)` from the running claim via
//! `current_claim = (1+r_now)·G(0) + r_now·G(1)`.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use crate::field::gf2_128::x86_64::{WideGhashX4, f128x4_loadu, f128x4_set, ghash_mul_x4};
use crate::field::{F128, F256Unreduced, PHI_8_TABLE};
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::univariate_skip::{SplitEqGhash, build_eq, pack_bits};

pub(crate) mod kernels;
pub(crate) mod three_var;
pub use three_var::{Round4Lookahead, eval_round4_lookahead};

#[cfg(all(test, target_arch = "aarch64"))]
use kernels::aarch64::fold_one_row_neon_unchecked_8;
#[cfg(target_arch = "aarch64")]
use kernels::aarch64::{round2_chunk_raw_neon, round2_chunk_raw_neon_q};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use kernels::x86_64::{fold_and_message_x86_avx512, fold_round2_pair_x86_unchecked_8};

/// Returns `(pair_in_block_mask, useful_pairs_inclusive)` for the round-2
/// fused-fold kernel. A pair (post-URM chunks `2k`, `2k+1`) is fully inside
/// padding iff `(k & pair_in_block_mask) >= useful_pairs_inclusive` — those
/// pairs contribute zero to both the message and the folded output (which is
/// already zero-initialized), so the kernel can `continue` past them.
///
/// `useful_pairs_inclusive` is the index AFTER the last pair that has any
/// useful chunk. The boundary "mixed" pair (one useful + one padding chunk,
/// when `useful_bits` is odd in chunk units) is INSIDE the useful range and
/// processed normally — its padding side has value 0 so the message
/// contribution is naturally correct.
fn round2_pair_skip(padding: &PaddingSpec, k_skip: usize) -> (usize, usize) {
    if padding.k_log <= k_skip + 1 {
        return (0, usize::MAX);
    }
    let pairs_per_block = 1usize << (padding.k_log - k_skip - 1);
    let chunk_bits = 1usize << k_skip;
    let useful_pairs = padding.useful_bits_per_block.div_ceil(2 * chunk_bits);
    if useful_pairs >= pairs_per_block {
        return (0, usize::MAX);
    }
    (pairs_per_block - 1, useful_pairs)
}

/// Exact same-binary rollback for sharing the parity-weight inverse between
/// `kappa = (1 + r) / r` and the deferred-lookahead rescale. Only the literal
/// value `1` restores the incumbent two independent `F128::inv` calls.
pub const ENV_NO_ZC_DUP_INV_ELIDE: &str = "FLOCK_NO_ZC_DUP_INV_ELIDE";

#[inline]
fn dup_inv_elide_disabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

#[cfg(test)]
std::thread_local! {
    /// Thread-local so exact tests can exercise both arms without mutating the
    /// process environment or racing unrelated tests. The selector is resolved
    /// before any Rayon work is launched.
    static DUP_INV_ELIDE_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[inline]
fn dup_inv_elide_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = DUP_INV_ELIDE_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }

    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !dup_inv_elide_disabled_value(std::env::var_os(ENV_NO_ZC_DUP_INV_ELIDE).as_deref())
    });
    *ON
}

/// Cold incumbent arm. Keeping it out of line leaves one inversion loop in
/// each ranked caller while the kill switch still executes the original two
/// independent inversions in the same binary.
#[cold]
#[inline(never)]
fn duplicate_lookahead_inv_factors(r: F128) -> (F128, F128) {
    ((F128::ONE + r) * r.inv(), r.inv())
}

#[inline(always)]
fn lookahead_inv_factors(r: F128) -> (F128, F128) {
    if dup_inv_elide_enabled() {
        let r_inv = r.inv();
        (r_inv + F128::ONE, r_inv)
    } else {
        duplicate_lookahead_inv_factors(r)
    }
}

// ---------------------------------------------------------------------------
// GFNI prefold dead-line plan (x86 round-2 / cascade-L1 batch folds).
// ---------------------------------------------------------------------------

/// `FLOCK_PREFOLD_ROW_SKIP=1` selects the predicated 64-row GFNI prefold, which
/// skips tile lines whose rows are all padding. The default loads every tile
/// line. Output is bit-identical either way.
#[cfg_attr(
    not(all(target_feature = "avx512vbmi", target_feature = "gfni")),
    allow(dead_code)
)]
fn prefold_row_skip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_PREFOLD_ROW_SKIP").as_deref() == Some(std::ffi::OsStr::new("1"))
    })
}

/// **Exact** provably-zero post-URM row set implied by the [`PaddingSpec`]:
/// returns `(row_in_block_mask, first_dead_row)` such that post-URM row `r`
/// holds an all-zero packed chunk iff `(r & row_in_block_mask) >=
/// first_dead_row`. `None` when the shape has no dead rows.
///
/// A post-URM row is one `2^k_skip`-bit chunk of a padding block. Row `c`
/// of a block covers witness bits `c·2^k_skip .. (c+1)·2^k_skip`, so it is
/// entirely inside the block's zero tail iff `c·2^k_skip >=
/// useful_bits_per_block`, i.e. `c >= ceil(useful_bits_per_block / 2^k_skip)`.
///
/// At the ranked BLAKE3 shape (`k_log = 14`, `useful_bits_per_block = 15409`,
/// `k_skip = 6`): 256 rows per block, `first_dead_row = ceil(15409/64) = 241`,
/// so rows 241..=255 — **15 rows of every 256** — are provably zero.
///
/// This is the *tightest* statement of the dead set and is the reference the
/// line-granular plan below is proved against; the hot path derives its plan
/// from [`round2_pair_skip`]'s already-trusted pair predicate instead, which
/// is provably equivalent at 8-row granularity (see
/// `prefold_line_plan_matches_exact_padding_derivation`).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn round2_row_zero(padding: &PaddingSpec, k_skip: usize) -> Option<(usize, usize)> {
    if padding.k_log <= k_skip {
        return None;
    }
    let rows_per_block = 1usize << (padding.k_log - k_skip);
    let chunk_bits = 1usize << k_skip;
    let first_dead_row = padding.useful_bits_per_block.div_ceil(chunk_bits);
    if first_dead_row >= rows_per_block {
        return None;
    }
    Some((rows_per_block - 1, first_dead_row))
}

/// Dead-line plan for one 64-row GFNI prefold tile: bit `i` of the result is
/// set iff the tile's `i`-th 64-byte line (packed rows `tile_base_row + 8i
/// ..= tile_base_row + 8i + 7`) is entirely inside padding, so the kernel may
/// substitute an all-zero register for that load.
///
/// **Value-derived, refusing off-shape.** The two inputs are exactly what
/// [`round2_pair_skip`] derived from the [`PaddingSpec`] at runtime; every
/// premise the plan needs is re-checked here and a violation returns `0`
/// (= load every line, the incumbent behaviour):
///
/// 1. `useful_pairs_inclusive == usize::MAX` (or `pair_in_block_mask == 0`)
///    is `round2_pair_skip`'s "nothing is dead" sentinel.
/// 2. `pairs_per_block = pair_in_block_mask + 1` must be a power of two —
///    otherwise the mask is not a block-position mask at all.
/// 3. `rows_per_block = 2 · pairs_per_block` must be a multiple of 64, and
///    `tile_base_row` a multiple of 64. Together these put the whole 64-row
///    tile inside one padding block, so `tile_base_row & (rows_per_block − 1)`
///    is the tile's block-local start and no line straddles a block edge.
/// 4. `first_dead_row = 2 · useful_pairs_inclusive` must be `< rows_per_block`.
///
/// **Soundness.** `round2_pair_skip` establishes that a pair `p` with
/// `(p & pair_in_block_mask) >= useful_pairs_inclusive` lies wholly in
/// padding; its two rows are `2p` and `2p+1`, so every block-local row
/// `>= 2 · useful_pairs_inclusive` is a padding row. A tile line whose
/// block-local start `local + 8i` is `>= first_dead_row` therefore contains
/// only padding rows, and — because rows only ever reach an accumulator
/// through the pair they belong to — *every* pair owning a row of that line
/// is already `continue`d by the consuming loop. The folded values the
/// kernel would have computed there are never read, so zeroing them is
/// unobservable regardless of what bytes the padding region actually holds.
///
/// At the ranked shape (`pair_in_block_mask = 127`,
/// `useful_pairs_inclusive = 121`): `rows_per_block = 256`,
/// `first_dead_row = 242`, and only the tile at block-local start 192 has a
/// dead line — its last one, rows 248..=255. That is **1 of the 32 lines of
/// every 256-row block**.
#[cfg_attr(
    not(all(target_feature = "avx512vbmi", target_feature = "gfni")),
    allow(dead_code)
)]
pub(crate) fn prefold_dead_line_mask(
    tile_base_row: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> u8 {
    if useful_pairs_inclusive == usize::MAX || pair_in_block_mask == 0 {
        return 0;
    }
    let pairs_per_block = pair_in_block_mask + 1;
    if !pairs_per_block.is_power_of_two() {
        return 0;
    }
    let rows_per_block = 2 * pairs_per_block;
    if !rows_per_block.is_multiple_of(64) || !tile_base_row.is_multiple_of(64) {
        return 0;
    }
    let first_dead_row = useful_pairs_inclusive.saturating_mul(2);
    if first_dead_row >= rows_per_block {
        return 0;
    }
    let local = tile_base_row & (rows_per_block - 1);
    let mut mask = 0u8;
    for i in 0..8 {
        if local + 8 * i >= first_dead_row {
            mask |= 1 << i;
        }
    }
    mask
}

/// [`prefold_dead_line_mask`] behind the kill switch — what the kernels call.
/// The predicate itself stays pure so its proof tests are independent of the
/// environment they run in.
#[cfg_attr(
    not(all(target_feature = "avx512vbmi", target_feature = "gfni")),
    allow(dead_code)
)]
#[inline]
pub(crate) fn prefold_dead_line_mask_gated(
    tile_base_row: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> u8 {
    if !prefold_row_skip_enabled() {
        return 0;
    }
    prefold_dead_line_mask(tile_base_row, pair_in_block_mask, useful_pairs_inclusive)
}

// ---------------------------------------------------------------------------
// Lagrange weights for the univariate-skip fold at z.
// ---------------------------------------------------------------------------

/// Lagrange weights `L_i(z)` for `i ∈ 0..2^k_skip` at the fold point `z`.
///
/// `L_i(z) = ∏_{j ≠ i} (z + φ_8(j)) / (φ_8(i) + φ_8(j))` — the standard Lagrange
/// formula, with the nodes being the F_8 elements `0..2^k_skip` embedded into
/// F_{2^128} via `φ_8`. Subtraction is XOR in characteristic 2.
///
/// See [`lagrange_weights_on_nodes`] for how the weights are formed.
pub fn lagrange_weights_naive(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    assert!(ell <= 256, "k_skip > 8 would exceed PHI_8_TABLE");
    lagrange_weights_on_nodes(k_skip, z, 0)
}

/// Cached `1 / ∏_{j ≠ i} (s_i + s_j)` for one φ_8 node set, indexed by
/// `k_skip`. The denominators depend only on the nodes, so a node set's
/// inverses are computed once per process and shared by every fold point.
/// `node_base` is the node set's first index into `PHI_8_TABLE` (`0` for the
/// S domain, `2^k_skip` for Λ).
fn lagrange_denominator_inv(k_skip: usize, node_base: usize) -> &'static [F128] {
    static CACHE: [[std::sync::OnceLock<Vec<F128>>; 9]; 2] =
        [const { [const { std::sync::OnceLock::new() }; 9] }; 2];
    let domain = usize::from(node_base != 0);
    CACHE[domain][k_skip].get_or_init(|| {
        let ell = 1usize << k_skip;
        let mut out = vec![F128::ZERO; ell];
        for (i, slot) in out.iter_mut().enumerate() {
            let si = PHI_8_TABLE[node_base + i];
            let mut den = F128::ONE;
            for j in 0..ell {
                if j == i {
                    continue;
                }
                den *= si + PHI_8_TABLE[node_base + j];
            }
            *slot = den.inv();
        }
        out
    })
}

/// `FLOCK_NO_LAGRANGE_BATCH_INV=1` restores the per-node numerator product
/// and per-node denominator inversion for the Lagrange weights. The ranked
/// worker's cleared environment never sets it.
#[inline]
fn lagrange_batch_inv_off() -> bool {
    static OFF: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LAGRANGE_BATCH_INV").is_some());
    *OFF
}

/// Lagrange weights at `z` over the φ_8 node set starting at `node_base`.
///
/// `L_i(z) = ∏_{j ≠ i} (z + s_j) / ∏_{j ≠ i} (s_i + s_j)`. The numerators are
/// all cofactors of the same product `P(z) = ∏_j (z + s_j)`, so one batch
/// inversion of the `z + s_i` terms yields every cofactor as `P · (z + s_i)⁻¹`,
/// and the denominators come from the cached node-set table. When `z` lands
/// exactly on a node the cofactor form is unavailable and the per-node
/// products are used instead.
fn lagrange_weights_on_nodes(k_skip: usize, z: F128, node_base: usize) -> Vec<F128> {
    let ell = 1usize << k_skip;
    let mut weights = vec![F128::ZERO; ell];

    if !lagrange_batch_inv_off() {
        let mut terms = vec![F128::ZERO; ell];
        let mut on_node = false;
        for (i, slot) in terms.iter_mut().enumerate() {
            let t = z + PHI_8_TABLE[node_base + i];
            on_node |= t.is_zero();
            *slot = t;
        }
        if !on_node {
            let den_inv = lagrange_denominator_inv(k_skip, node_base);
            // Montgomery batch inverse: prefix[i] = ∏_{j < i} terms[j].
            let mut prefix = vec![F128::ZERO; ell];
            let mut acc = F128::ONE;
            for (i, slot) in prefix.iter_mut().enumerate() {
                *slot = acc;
                acc *= terms[i];
            }
            let total = acc;
            let mut suffix = total.inv();
            for i in (0..ell).rev() {
                weights[i] = total * (prefix[i] * suffix) * den_inv[i];
                suffix *= terms[i];
            }
            return weights;
        }
    }

    for i in 0..ell {
        let si = PHI_8_TABLE[node_base + i];
        let mut num = F128::ONE;
        let mut den = F128::ONE;
        for j in 0..ell {
            if j == i {
                continue;
            }
            let sj = PHI_8_TABLE[node_base + j];
            num *= z + sj;
            den *= si + sj;
        }
        weights[i] = num * den.inv();
    }
    weights
}

/// Lagrange weights `L_i^Λ(z)` for `i ∈ 0..2^k_skip` at the fold point `z`,
/// where the nodes are the **extension domain** `Λ = {2^k_skip, …, 2^(k_skip+1) − 1}`
/// embedded via `φ_8` (offset by `2^k_skip` from the S-domain nodes).
///
/// Used to interpolate the extract_c round-1 output `round1_c` (which carries
/// the polynomial `P^C` as its 2^k_skip evaluations on Λ) at the URM challenge `z`.
pub fn lagrange_weights_lambda_naive(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    lagrange_weights_on_nodes(k_skip, z, ell)
}

/// Interpolate a degree-`< 2^k_skip` polynomial at z, given its `2^k_skip`
/// evaluations on Λ. Returns `Σ_i L_i^Λ(z) · values[i]`.
///
/// In the extract_c protocol the prover ships `round1_c` (the `P^C` polynomial
/// in Λ-form) and the verifier (or higher-level prover) needs `P^C(z) = ĉ(z, r_rest)`.
/// That value is *the c-claim* at the bound point `(z, r_rest)`.
pub fn interpolate_at_z_on_lambda(values: &[F128], k_skip: usize, z: F128) -> F128 {
    let ell = 1usize << k_skip;
    assert_eq!(values.len(), ell);
    let weights = lagrange_weights_lambda_naive(k_skip, z);
    let mut acc = F128::ZERO;
    for i in 0..ell {
        acc += weights[i] * values[i];
    }
    acc
}

/// Fused interpolation on `Λ` (`interpolate_at_z_on_lambda`) and construction
/// of [`UniSkipFoldTable`] on `S` at the same fold challenge `z`.
///
/// Both domains (`Λ` at `node_base = 2^k_skip` and `S` at `node_base = 0`)
/// evaluate `2^k_skip` Lagrange weights at `z`, each requiring a Montgomery
/// batch inversion of its `2^k_skip` node-difference terms `z + φ_8(j)`.
/// Batching all `2 · 2^k_skip` terms into a single prefix-product sweep replaces
/// two `F128::inv()` calls with one while keeping every cofactor product and
/// the `UniSkipFoldTable` allocation identical.
pub fn interpolate_lambda_and_fold_table(
    values_on_lambda: &[F128],
    k_skip: usize,
    z: F128,
) -> (F128, UniSkipFoldTable) {
    let ell = 1usize << k_skip;
    assert_eq!(values_on_lambda.len(), ell);
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    assert_eq!(ell % 8, 0, "k_skip must be ≥ 3 (need ell divisible by 8)");

    if !lagrange_batch_inv_off() && ell == 64 {
        let mut terms_s = [F128::ZERO; 64];
        let mut terms_l = [F128::ZERO; 64];
        let mut on_node = false;
        for i in 0..64 {
            let ts = z + PHI_8_TABLE[i];
            let tl = z + PHI_8_TABLE[64 + i];
            on_node |= ts.is_zero() | tl.is_zero();
            terms_s[i] = ts;
            terms_l[i] = tl;
        }
        if !on_node {
            let den_s = lagrange_denominator_inv(k_skip, 0);
            let den_l = lagrange_denominator_inv(k_skip, 64);

            let mut prefix_s = [F128::ZERO; 64];
            let mut acc_s = F128::ONE;
            for i in 0..64 {
                prefix_s[i] = acc_s;
                acc_s *= terms_s[i];
            }
            let total_s = acc_s;

            let mut prefix_l = [F128::ZERO; 64];
            let mut acc_l = F128::ONE;
            for i in 0..64 {
                prefix_l[i] = acc_l;
                acc_l *= terms_l[i];
            }
            let total_l = acc_l;

            let inv_both = (total_s * total_l).inv();
            let mut suffix_s = inv_both * total_l;
            let mut suffix_l = inv_both * total_s;

            let mut c_eval = F128::ZERO;
            for i in (0..64).rev() {
                let w_l = total_l * (prefix_l[i] * suffix_l) * den_l[i];
                suffix_l *= terms_l[i];
                c_eval += w_l * values_on_lambda[i];
            }

            let mut weights_s = [F128::ZERO; 64];
            for i in (0..64).rev() {
                weights_s[i] = total_s * (prefix_s[i] * suffix_s) * den_s[i];
                suffix_s *= terms_s[i];
            }

            return (c_eval, UniSkipFoldTable::from_weights(8, &weights_s));
        }
    }

    let c_eval = interpolate_at_z_on_lambda(values_on_lambda, k_skip, z);
    let table = UniSkipFoldTable::new(k_skip, z);
    (c_eval, table)
}

/// Interpolate a degree-`< 2·2^k_skip` polynomial at z, given its `2^k_skip`
/// evaluations on Λ and the assumption that it equals **zero on S**.
///
/// This is the verifier's round-1 reconstruction trick: for an honest prover
/// the combined polynomial `P = P^{AB} + P^C` satisfies `P(λ) = 0` for every
/// `λ ∈ S` (the zerocheck identity at S). Together with the `2^k_skip`
/// evaluations on Λ that the prover sends, that's `2·2^k_skip` evaluations —
/// enough to interpolate the degree-`< 2·2^k_skip` polynomial uniquely.
///
/// Cost: `2·ell × (2·ell − 1)` F128 muls + `ell` inversions for the Lagrange
/// weights. At ell=64 that's ~16K muls + 64 inversions. Sub-millisecond
/// one-time cost in the verifier.
pub fn interpolate_at_z_combined(values_on_lambda: &[F128], k_skip: usize, z: F128) -> F128 {
    let ell = 1usize << k_skip;
    assert_eq!(values_on_lambda.len(), ell);
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    let n_total = 2 * ell;
    let mut acc = F128::ZERO;
    for i in 0..ell {
        // i-th Λ node = node index `ell + i` in PHI_8_TABLE.
        let node_idx = ell + i;
        let si = PHI_8_TABLE[node_idx];
        let mut num = F128::ONE;
        let mut den = F128::ONE;
        for j in 0..n_total {
            if j == node_idx {
                continue;
            }
            let sj = PHI_8_TABLE[j];
            num *= z + sj;
            den *= si + sj;
        }
        let weight = num * den.inv();
        acc += weight * values_on_lambda[i];
    }
    acc
}

/// Evaluate the multilinear eq polynomial at a point: `eq(r, x) = Π_i (1 + r_i + x_i)`
/// for `r, x ∈ F_{2^128}^n` (char-2 simplification of `(1-r)(1-x) + r·x`).
pub fn eq_eval(r: &[F128], x: &[F128]) -> F128 {
    assert_eq!(r.len(), x.len());
    let mut acc = F128::ONE;
    for i in 0..r.len() {
        acc *= F128::ONE + r[i] + x[i];
    }
    acc
}

/// Specialized variant of [`eq_eval`] for the case where `x` is binary,
/// encoded as a bitmask. Each factor reduces to `r_i` (bit=1) or `1 + r_i`
/// (bit=0), saving one F128 add per coord.
pub fn eq_eval_binary_x(r: &[F128], x_bits: u32) -> F128 {
    debug_assert!(r.len() <= 32, "x_bits is u32; r > 32 dims not supported");
    let mut acc = F128::ONE;
    for (i, &r_i) in r.iter().enumerate() {
        let factor = if (x_bits >> i) & 1 == 1 {
            r_i
        } else {
            F128::ONE + r_i
        };
        acc *= factor;
    }
    acc
}

// ---------------------------------------------------------------------------
// Fold a Boolean witness at z.
// ---------------------------------------------------------------------------

/// Evaluate the univariate-skip polynomial at the fold point `z`, given the
/// precomputed Lagrange `weights`. Returns the multilinear extension table
/// `a_mlv` of length `2^(m − k_skip)` over F_{2^128}.
///
///   `a_mlv[x_rest] = Σ_s a(s, x_rest) · L_s(z)`
///
/// `a(s, x_rest)` is the witness bit at index `x_rest * 2^k_skip + s` (low
/// bits = skip variable, high bits = rest variables).
pub fn fold_at_z_naive(witness: &[bool], m: usize, k_skip: usize, weights: &[F128]) -> Vec<F128> {
    assert!(k_skip <= m);
    let ell = 1usize << k_skip;
    let n_rest = 1usize << (m - k_skip);
    assert_eq!(witness.len(), 1usize << m);
    assert_eq!(weights.len(), ell);

    let mut folded = vec![F128::ZERO; n_rest];
    for x_rest in 0..n_rest {
        let base = x_rest * ell;
        let mut acc = F128::ZERO;
        for s in 0..ell {
            if witness[base + s] {
                acc += weights[s];
            }
        }
        folded[x_rest] = acc;
    }
    folded
}

// ---------------------------------------------------------------------------
// Naive round-2 prover message (AB-pair multilinear sumcheck).
// ---------------------------------------------------------------------------

/// Round-2 (and any subsequent round) prover message for the AB-pair
/// multilinear sumcheck.
///
/// Inputs:
/// - `a_mlv`, `b_mlv`: F128 vectors of length `2^n` for some `n ≥ 1`.
/// - `r`: full eq challenges, length `n`. `r[0]` is the challenge for the
///   variable being bound *this* round; `r[1..]` is for the remaining `n − 1`
///   variables.
///
/// Output: `(r[0] · G(1), G(∞))` for the round polynomial `G(X) = Σ_{x'} eq(r[1..], x')
/// · a_mlv(X, x') · b_mlv(X, x')`, where `a_mlv(0, x') = a_mlv[2x']` and
/// `a_mlv(1, x') = a_mlv[2x' + 1]` (low bit bound).
///
/// The `r[0]` prefactor matches the C++ `sumcheck_round_pair` convention: the
/// quantity sent on the wire is `Π(1) = eq(r[0], 1) · G(1) = r[0] · G(1)`,
/// where `Π(X) = eq(r[0], X) · G(X)` is the actual round polynomial.
pub fn round_pair_naive(a_mlv: &[F128], b_mlv: &[F128], r: &[F128]) -> (F128, F128) {
    let n = a_mlv.len();
    assert_eq!(b_mlv.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r.len(), log_n);

    let eq_remaining = build_eq(&r[1..]);
    assert_eq!(eq_remaining.len(), half);

    let mut g_one = F128::ZERO;
    let mut g_inf = F128::ZERO;
    for x_prime in 0..half {
        let a0 = a_mlv[2 * x_prime];
        let a1 = a_mlv[2 * x_prime + 1];
        let b0 = b_mlv[2 * x_prime];
        let b1 = b_mlv[2 * x_prime + 1];
        let eq_x = eq_remaining[x_prime];
        g_one += eq_x * a1 * b1;
        // Char-2: (a_1 − a_0)(b_1 − b_0) = (a_0 + a_1)(b_0 + b_1).
        g_inf += eq_x * (a0 + a1) * (b0 + b1);
    }
    (r[0] * g_one, g_inf)
}

// ---------------------------------------------------------------------------
// Naive fused (fold at z + round-2 message) for AB-pair.
// ---------------------------------------------------------------------------

/// Naive fold (at the univariate-skip challenge `z`) of `a` and `b`, plus the
/// round-2 prover message on the resulting multilinear polynomials.
///
/// `mlv_challenges` is of length `m − k_skip` — one challenge per multilinear
/// round. `mlv_challenges[0]` is for the variable bound in round 2 (this
/// round's message uses it as the `r_now` multiplier); `mlv_challenges[1..]`
/// is for subsequent rounds (eq table).
///
/// This is the *unfused* reference: it computes the fold and the round-2
/// message in two separate passes. The optimized version (next) will do both
/// in one pass through the witness.
///
/// Returns `(a_mlv, b_mlv, mlv_challenges[0] · G(1), G(∞))`.
pub fn uni_skip_fold_and_round_pair_naive(
    a: &[bool],
    b: &[bool],
    m: usize,
    k_skip: usize,
    z: F128,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    assert!(
        m > k_skip,
        "need at least one multilinear variable past the skip"
    );
    assert_eq!(mlv_challenges.len(), m - k_skip);

    let weights = lagrange_weights_naive(k_skip, z);
    let a_mlv = fold_at_z_naive(a, m, k_skip, &weights);
    let b_mlv = fold_at_z_naive(b, m, k_skip, &weights);
    let (msg_1, msg_inf) = round_pair_naive(&a_mlv, &b_mlv, mlv_challenges);
    (a_mlv, b_mlv, msg_1, msg_inf)
}

// ---------------------------------------------------------------------------
// Optimized fused fold + round-2 message.
// ---------------------------------------------------------------------------

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn zc_gfni_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_NO_ZC_GFNI").as_deref() != Some(std::ffi::OsStr::new("1"))
    })
}

/// `FLOCK_NO_ZC_CFOLD_BAKE=1` restores the incumbent rounds-3+4 composed
/// fold: three constant `(ρ₁, ρ₂)` multiplies per output on top of the plain
/// byte-table fold, instead of carrying those constants inside the fold's own
/// bit matrices. Exact same-binary A/B — the baked form is the same field
/// element, so the transcript and proof bytes are unchanged. Read once per
/// process; the ranked worker's cleared environment never sets it.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
fn cfold_bake_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_CFOLD_BAKE").is_none());
    *ON
}

/// GFNI bit-matrix form of the fold table for the batched row folds
/// (`FLOCK_NO_ZC_GFNI=1` keeps the gather kernels; bit-identical output).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn r2_gfni_mats(table: &UniSkipFoldTable) -> Option<[u64; 128]> {
    (table.n_chunks == 8 && zc_gfni_enabled())
        .then(|| kernels::x86_64::build_row_fold_mats(&table.data))
}

/// Precomputed fold table for the univariate-skip fold at a fixed `z`.
///
/// Storage: `n_chunks × 256` F128 entries (32 KB at `k_skip=6`). For each
/// byte-chunk `j ∈ 0..n_chunks` and byte value `v ∈ 0..256`:
///
///   `data[j * 256 + v] = Σ_{b : bit b of v set} weights[8j + b]`
///
/// where `weights = lagrange_weights_naive(k_skip, z)`. Built incrementally by
/// XOR-composition over the set bits of `v` (one XOR per non-power-of-2 entry).
///
/// Per-row fold then becomes one table lookup + XOR per byte (n_chunks lookups
/// total instead of `ell` Lagrange multiplications).
#[derive(Clone, Debug)]
pub struct UniSkipFoldTable {
    pub n_chunks: usize,
    pub data: Vec<F128>,
}

impl UniSkipFoldTable {
    pub fn new(k_skip: usize, z: F128) -> Self {
        let ell = 1usize << k_skip;
        assert_eq!(ell % 8, 0, "k_skip must be ≥ 3 (need ell divisible by 8)");
        let n_chunks = ell / 8;
        let weights = lagrange_weights_naive(k_skip, z);
        Self::from_weights(n_chunks, &weights)
    }

    pub(crate) fn from_weights(n_chunks: usize, weights: &[F128]) -> Self {
        assert_eq!(weights.len(), n_chunks * 8);
        let mut data = vec![F128::ZERO; n_chunks * 256];
        for j in 0..n_chunks {
            let basis = &weights[8 * j..8 * j + 8];
            // v = 0: zero (already initialized).
            for b in 0..8 {
                data[j * 256 + (1 << b)] = basis[b];
            }
            // Non-powers-of-2: composed by XOR of (v ^ lo_bit) and lo_bit entries.
            for v in 3usize..256 {
                if (v & (v - 1)) == 0 {
                    continue; // skip powers of 2 (already written)
                }
                let lo_bit = v.isolate_lowest_one();
                let parent = v ^ lo_bit;
                data[j * 256 + v] = data[j * 256 + parent] + data[j * 256 + lo_bit];
            }
        }
        Self { n_chunks, data }
    }

    /// Scalar one-row fold: `Σ_j table[j][bytes[j]]`. Ports the NEON
    /// `uni_skip_fold_one_output_ghash` in scalar form.
    #[inline]
    pub fn fold_one_row(&self, bytes: &[u8]) -> F128 {
        assert_eq!(bytes.len(), self.n_chunks);
        let mut acc = F128::ZERO;
        for j in 0..self.n_chunks {
            acc += self.data[j * 256 + bytes[j] as usize];
        }
        acc
    }
}

/// Optimized fused fold (at the URM challenge `z`, baked into `table`) plus
/// round-2 prover message. **Packed input** (LSB-first bit packing). **Parallel
/// by default** via rayon — the outer x_hi loop is distributed across workers,
/// each writing to a disjoint chunk of `a_folded`/`b_folded` via `par_chunks_mut`
/// and accumulating its own `(sum1_contrib, sum_inf_contrib)`. The final
/// reduce sums the per-worker contributions (commutative + associative F128
/// XOR/multiply).
///
/// Algorithm (per worker, one x_hi):
/// 1. For each `(x0, x1) = (2k, 2k+1)` pair (k within this x_hi's range),
///    fold the four rows `a[x0], b[x0], a[x1], b[x1]` via the table.
/// 2. Accumulate `eq_lo · a1·b1` and `eq_lo · (a0+a1)·(b0+b1)` with deferred
///    256-bit reduction, reduced once at the end of the worker's x_lo loop.
/// 3. Outer fold by `eq.hi[x_hi]` into the worker's `(sum1_contrib, sum_inf_contrib)`.
///
/// Returns `(a_folded, b_folded, mlv_challenges[0] · G(1), G(∞))` — same
/// convention as `uni_skip_fold_and_round_pair_naive`.
///
/// To run single-threaded for debugging, set `RAYON_NUM_THREADS=1`.
///
/// `k_skip = 6` is currently hardcoded (the protocol headline).
pub fn uni_skip_fold_and_round_pair_optimized_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    uni_skip_fold_and_round_pair_optimized_packed_padded(
        a_packed,
        b_packed,
        m,
        k_skip,
        table,
        mlv_challenges,
        &PaddingSpec::dense(m),
    )
}

/// Padding-aware variant of [`uni_skip_fold_and_round_pair_optimized_packed`].
/// Skips pairs whose post-URM chunk indices both fall in the per-block zero
/// padding: the fold output is already zero-initialized and the message
/// contribution would be zero, so we can `continue` past those pairs.
pub fn uni_skip_fold_and_round_pair_optimized_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    use rayon::prelude::*;

    assert_eq!(
        k_skip, 6,
        "optimized fold-and-round_pair variant is k_skip=6 only"
    );
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);

    // Uninit alloc — the parallel loop below writes every slot (dense path)
    // or explicitly writes F128::ZERO at padding holes (padded path).
    // Saves ~22 ms of sequential zero-fill at m=29 (256 MB total) that would
    // otherwise cap the parallel speedup of this phase at ~2.5× on 8 cores.
    let mut a_folded: Vec<F128> = crate::scratch::take_f128(n_out);
    let mut b_folded: Vec<F128> = crate::scratch::take_f128(n_out);

    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_out);

    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);
    // q-form message gate: the scalar leaf extracts every folded value to
    // GPRs for the message muls; the q leaf keeps them in NEON registers
    // (bit-identical output). `FLOCK_NO_R2_QMSG` is a local-diagnostics kill
    // switch for one-process A/B runs; the ranked worker never sets it.
    #[cfg(target_arch = "aarch64")]
    let r2_qmsg = !crate::sl2_env_set!("FLOCK_NO_R2_QMSG");

    // Parallel: each worker writes one disjoint chunk of a_folded/b_folded
    // and returns its (sum1, sum_inf) contribution. Reduce by F128 XOR.
    let (sum1, sum_inf) = a_folded
        .par_chunks_mut(chunk_size)
        .zip(b_folded.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(x_hi, (a_chunk, b_chunk))| {
            #[cfg(not(target_arch = "aarch64"))]
            let mut p1_acc = F256Unreduced::ZERO;
            #[cfg(not(target_arch = "aarch64"))]
            let mut pinf_acc = F256Unreduced::ZERO;
            let pair_idx_base = x_hi * lo_size;

            #[cfg(target_arch = "aarch64")]
            let (p1, pinf) = unsafe {
                let base = x_hi * chunk_size;
                let kernel = if r2_qmsg {
                    round2_chunk_raw_neon_q
                } else {
                    round2_chunk_raw_neon
                };
                kernel(
                    table.data.as_ptr() as *const u8,
                    a_packed.as_ptr().add(base * 8),
                    b_packed.as_ptr().add(base * 8),
                    a_chunk.as_mut_ptr(),
                    b_chunk.as_mut_ptr(),
                    eq_lo.as_ptr(),
                    lo_size,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            unsafe {
                let table_ptr = table.data.as_ptr();
                let a_pkt_ptr = a_packed.as_ptr();
                let b_pkt_ptr = b_packed.as_ptr();
                let base = x_hi * chunk_size;
                let mut p1_wide = WideGhashX4::zero();
                let mut pinf_wide = WideGhashX4::zero();
                let mut x_lo = 0;

                while x_lo + 4 <= lo_size {
                    let mut a0 = [F128::ZERO; 4];
                    let mut a1 = [F128::ZERO; 4];
                    let mut b0 = [F128::ZERO; 4];
                    let mut b1 = [F128::ZERO; 4];

                    for lane in 0..4 {
                        let pair = x_lo + lane;
                        let x0l = 2 * pair;
                        let x1l = x0l + 1;
                        if ((pair_idx_base + pair) & pair_in_block_mask) >= useful_pairs_inclusive {
                            a_chunk[x0l] = F128::ZERO;
                            a_chunk[x1l] = F128::ZERO;
                            b_chunk[x0l] = F128::ZERO;
                            b_chunk[x1l] = F128::ZERO;
                            continue;
                        }

                        let x0g = base + x0l;
                        let x1g = x0g + 1;
                        let folded = fold_round2_pair_x86_unchecked_8(
                            table_ptr,
                            a_pkt_ptr.add(x0g * 8),
                            a_pkt_ptr.add(x1g * 8),
                            b_pkt_ptr.add(x0g * 8),
                            b_pkt_ptr.add(x1g * 8),
                        );
                        [a0[lane], a1[lane], b0[lane], b1[lane]] = folded;
                        a_chunk[x0l] = a0[lane];
                        a_chunk[x1l] = a1[lane];
                        b_chunk[x0l] = b0[lane];
                        b_chunk[x1l] = b1[lane];
                    }

                    let a1x4 = f128x4_loadu(a1.as_ptr());
                    let b1x4 = f128x4_loadu(b1.as_ptr());
                    let a_sum_x4 =
                        f128x4_set(a0[0] + a1[0], a0[1] + a1[1], a0[2] + a1[2], a0[3] + a1[3]);
                    let b_sum_x4 =
                        f128x4_set(b0[0] + b1[0], b0[1] + b1[1], b0[2] + b1[2], b0[3] + b1[3]);
                    let g1x4 = ghash_mul_x4(a1x4, b1x4);
                    let g_inf_x4 = ghash_mul_x4(a_sum_x4, b_sum_x4);
                    let eqx4 = f128x4_loadu(eq_lo[x_lo..].as_ptr());
                    p1_wide.mul_acc(eqx4, g1x4);
                    pinf_wide.mul_acc(eqx4, g_inf_x4);
                    x_lo += 4;
                }

                // Small instances can leave a 1- or 2-pair tail.
                while x_lo < lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        x_lo += 1;
                        continue;
                    }

                    let x0g = base + x0l;
                    let x1g = x0g + 1;
                    let [a0, a1, b0, b1] = fold_round2_pair_x86_unchecked_8(
                        table_ptr,
                        a_pkt_ptr.add(x0g * 8),
                        a_pkt_ptr.add(x1g * 8),
                        b_pkt_ptr.add(x0g * 8),
                        b_pkt_ptr.add(x1g * 8),
                    );
                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;
                    let eq_l = eq_lo[x_lo];
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                    x_lo += 1;
                }

                p1_acc ^= p1_wide.fold();
                pinf_acc ^= pinf_wide.fold();
            }
            #[cfg(not(any(
                target_arch = "aarch64",
                all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "vpclmulqdq"
                )
            )))]
            {
                let base = x_hi * chunk_size;
                for x_lo in 0..lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        // See aarch64 branch above for why this zero write is needed.
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        continue;
                    }
                    let x0g = base + 2 * x_lo;
                    let x1g = x0g + 1;
                    let a0 = table.fold_one_row(&a_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let b0 = table.fold_one_row(&b_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let a1 = table.fold_one_row(&a_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    let b1 = table.fold_one_row(&b_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;
                    let eq_l = eq_lo[x_lo];
                    let g1 = a1 * b1;
                    p1_acc ^= eq_l.mul_unreduced(g1);
                    let g_inf = (a0 + a1) * (b0 + b1);
                    pinf_acc ^= eq_l.mul_unreduced(g_inf);
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            let (p1, pinf) = (p1_acc.reduce(), pinf_acc.reduce());
            let eq_h = eq_hi[x_hi];
            (eq_h * p1, eq_h * pinf)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
        );

    (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf)
}

// ---------------------------------------------------------------------------
// Two-challenge lookahead: round-3 message deferred as a quadratic in ρ₁, so
// rounds 3+4 collapse into one composed double-fold pass (deletes the first
// tail iteration — the largest DRAM pass of the ladder). Pure reassociation
// of exact F128 arithmetic; every wire byte is identical to the incumbent.
// ---------------------------------------------------------------------------

/// Deferred round-three message: `G₃(1)` coefficients in `c[0..3]`, `G₃(∞)`
/// in `c[3..6]`, each in the basis `{1, ρ, ρ²}`.
///
/// Mirrors the six-coefficient shape of `pcs::ligerito::eval_lookahead`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Round3Lookahead {
    pub c: [F128; 6],
}

/// Evaluate the deferred round-three message at the sampled ρ₁.
///
/// The tail's `r_next[0] = ONE` in Convention A, so there is no prefactor:
/// the two returned values are exactly what the incumbent
/// [`fold_and_compute_round_pair_into`] would have sent for round three.
#[inline]
pub fn eval_round3_lookahead(la: &Round3Lookahead, rho1: F128) -> (F128, F128) {
    let rho_sq = rho1 * rho1;
    (
        la.c[0] + la.c[1] * rho1 + la.c[2] * rho_sq,
        la.c[3] + la.c[4] * rho1 + la.c[5] * rho_sq,
    )
}

/// eq split for the lookahead round-two sweep and the composed passes.
/// Identical to [`SplitEqGhash::new`] at the ranked shape (`n_vars ≥ 8`);
/// clamped so the lo half always keeps at least one variable, because the
/// sweep consumes pairs two at a time (one round-three group) inside a
/// chunk. Any admissible split is value-identical (exact tensor
/// factorization of eq).
#[inline]
fn lookahead_n_hi(n_vars: usize) -> usize {
    SplitEqGhash::MAX_N_HI.min(n_vars.saturating_sub(1))
}

/// Chunk-count cap for the plain composed tail levels: `2^TAIL_SPLIT_MAX_N_HI`
/// rayon chunks at most.
const TAIL_SPLIT_MAX_N_HI: usize = 11;

/// Per-chunk `eq_lo` floor for the plain composed tail levels, in log2 F128
/// entries. A chunk covers `2·lo_size` outputs and reads `8·lo_size` inputs;
/// shrinking `eq_lo` below 2^10 stops paying — measured at the ranked shape a
/// 2^6 `eq_lo` at `log_n = 20` took the chunk span from 0.53 ms to 0.88 ms and
/// in-region occupancy from 12.5/16 to 7.0/16, because the leaf loop no longer
/// amortizes the per-chunk prologue (eight `eq_hi` multiplies plus a
/// reduce-tree node).
const TAIL_SPLIT_MIN_LO_LOG: usize = 10;

/// Fan-out for the **plain** composed tail levels (`fold2_plain_*`).
///
/// The incumbent `lookahead_n_hi` cap of `MAX_N_HI = 7` is a constant 128
/// rayon chunks at *every* level. With 16 workers that is 8 chunks per worker,
/// so the region's drain costs a whole chunk. Per-chunk start/end timestamps
/// at the ranked shape (15 proves per arm, interleaved) put the drain
/// (`span − busy/16`) at a median 0.33 ms on the `log_n = 24` level and
/// 0.21 ms on `log_n = 22`. Splitting to 2^11 / 2^9 chunks while holding
/// `lo_size ≥ 2^10` cuts those to 0.20 ms and 0.10 ms, lifting in-region
/// occupancy from 15.36/16 to 15.61/16 and from 14.38/16 to 15.24/16, with
/// `busy` (total core-time) unchanged — the same work, spread flatter. Levels
/// whose `eq` cannot keep `lo_size ≥ 2^10` (at the ranked shape `log_n ≤ 20`)
/// keep the incumbent 128-chunk split.
///
/// Every admissible `n_hi` is value-identical: `eq` factors exactly as
/// `eq(x) = eq_hi[x_hi] · eq_lo[x_lo]`, the per-chunk accumulators are
/// deferred-reduction sums whose `reduce` is F2-linear, and the cross-chunk
/// combine is XOR — so `Σ_hi eq_hi · reduce(Σ_lo eq_lo ⊗ g)` is the same
/// field element for every split. Nothing about the transcript, the message
/// values, or their absorption order moves; the proof is byte-identical.
/// `FLOCK_NO_ZC_TAIL_SPLIT=1` restores the incumbent 128-chunk fan-out
/// (same-binary A/B control and emergency fallback); the ranked worker's
/// cleared environment never sets it.
#[inline]
fn tail_split_n_hi(n_vars: usize) -> usize {
    let base = lookahead_n_hi(n_vars);
    static OFF: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_TAIL_SPLIT").is_some());
    if *OFF {
        return base;
    }
    let n_hi = base.max(TAIL_SPLIT_MAX_N_HI.min(n_vars.saturating_sub(TAIL_SPLIT_MIN_LO_LOG)));
    // Lane gl1 (`FLOCK_NO_GL1_SPLIT`, floor `FLOCK_GL1_SPLIT_LO`, default 6):
    // the small levels (n14 and below) kept 64-128 chunks of 4-32 inputs,
    // where the per-chunk prologue and claim traffic cost 0.5-1 µs each
    // against ~0.2 µs of work. Keep every chunk's `eq_lo` at ≥ 2^lo (fewer,
    // larger chunks; down to one chunk). Same XOR/tensor identity as above.
    let lo = gl1_split_lo();
    if lo != 0 && n_vars <= 11 {
        return n_hi.min(n_vars.saturating_sub(lo));
    }
    n_hi
}

/// 0 = off. Read once per process (per call site: once per tail level).
fn gl1_split_lo() -> usize {
    static LO: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        if std::env::var_os("FLOCK_NO_GL1_SPLIT").is_some() {
            return 0;
        }
        std::env::var("FLOCK_GL1_SPLIT_LO").ok().and_then(|v| v.parse().ok()).unwrap_or(6)
    });
    *LO
}

/// Per-chunk `eq_lo` length of the plain composed tail level whose input has
/// `n` elements — the split [`fold2_plain_and_round_pair_lookahead_into`] and
/// [`fold2_plain_and_round4_into`] build over `r_next[1..]` (`log₂(n) − 3`
/// coordinates).
pub(crate) fn plain_cascade_lo_size(n: usize) -> usize {
    assert!(n.is_power_of_two() && n >= 16);
    let n_vars = n.trailing_zeros() as usize - 3;
    1usize << (n_vars - tail_split_n_hi(n_vars).min(n_vars))
}

/// Qword index vector for a two-source 512-bit lane select (`vpermi2q`):
/// output lane `i` is lane `sel[i].1` of the first source when `sel[i].0 == 0`
/// and of the second source when it is 1.
#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )),
    allow(dead_code)
)]
pub(crate) const fn lane_select(sel: [(u8, u8); 4]) -> [u64; 8] {
    let mut idx = [0u64; 8];
    let mut i = 0;
    while i < 4 {
        let (src, lane) = sel[i];
        assert!(src < 2 && lane < 4);
        idx[2 * i] = (8 * src + 2 * lane) as u64;
        idx[2 * i + 1] = (8 * src + 2 * lane + 1) as u64;
        i += 1;
    }
    idx
}

/// Lane ru: the flat claim route (see `crate::flat_par`) for the large
/// passes, where rayon's join-tree hand-off after the last chunk was traced at
/// 20-65 µs. Small tail levels keep rayon (its tree spreads tiny levels faster
/// than one-by-one spawns). Needs a pool worker caller.
pub(crate) fn flat_tail(n: usize) -> bool {
    n >= (1usize << 21) && crate::flat_par::enabled() && rayon::current_thread_index().is_some()
}

/// Chunked [`crate::flat_par::map_reduce`] over paired output tables.
pub(crate) fn flat_chunks<T>(
    a_out: &mut [F128],
    b_out: &mut [F128],
    chunk: usize,
    leaf: impl Fn((usize, (&mut [F128], &mut [F128]))) -> T + Sync,
    id: impl Fn() -> T + Sync,
    op: impl Fn(T, T) -> T + Sync,
) -> T {
    assert_eq!(a_out.len(), b_out.len());
    assert!(chunk > 0 && a_out.len().is_multiple_of(chunk));
    let n = a_out.len() / chunk;
    let (ao, bo) = (a_out.as_mut_ptr() as usize, b_out.as_mut_ptr() as usize);
    let rec = crate::flat_par::sl1_aff_enabled() && n <= crate::flat_par::SL1_OWN_CAP;
    if rec {
        *crate::flat_par::SL1_OWN_GEOM.lock().unwrap() = (ao, bo, chunk * 16, n);
    }
    crate::flat_par::map_reduce(
        n,
        |i| {
            if rec {
                let t = rayon::current_thread_index().map_or(u8::MAX, |t| t.min(250) as u8);
                crate::flat_par::SL1_OWN[i].store(t, core::sync::atomic::Ordering::Relaxed);
            }
            // SAFETY: item `i` owns output chunk `i` of both tables; chunks are
            // disjoint and both borrows outlive `map_reduce`.
            let (ac, bc) = unsafe {
                (
                    core::slice::from_raw_parts_mut((ao as *mut F128).add(i * chunk), chunk),
                    core::slice::from_raw_parts_mut((bo as *mut F128).add(i * chunk), chunk),
                )
            };
            leaf((i, (ac, bc)))
        },
        id,
        op,
    )
}

/// Lane sl1: [`team_chunks`] that hands item `i` (input chunk `i` of `a`,
/// `chunk_in` elements) first to the thread that wrote it in the last
/// flat-claimed level, when `a` is that level's output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn team_chunks_aff<T>(
    a_in: &[F128],
    chunk_in: usize,
    a_out: &mut [F128],
    b_out: &mut [F128],
    chunk: usize,
    leaf: impl Fn((usize, (&mut [F128], &mut [F128]))) -> T + Sync,
    id: impl Fn() -> T + Sync,
    op: impl Fn(T, T) -> T + Sync,
) -> T {
    let n = a_out.len() / chunk;
    let owner: Option<Vec<u8>> = if crate::flat_par::sl1_aff_enabled() {
        let (ga, _gb, gbytes, gn) = *crate::flat_par::SL1_OWN_GEOM.lock().unwrap();
        let base = a_in.as_ptr() as usize;
        if gn > 0 && base == ga && a_in.len() * 16 == gbytes * gn {
            Some((0..n).map(|i| {
                let off = i * chunk_in * 16;
                crate::flat_par::SL1_OWN[(off / gbytes).min(gn - 1)].load(core::sync::atomic::Ordering::Relaxed)
            }).collect())
        } else {
            None
        }
    } else {
        None
    };
    let Some(owner) = owner else { return team_chunks(a_out, b_out, chunk, leaf, id, op) };
    assert_eq!(a_out.len(), b_out.len());
    assert!(chunk > 0 && a_out.len().is_multiple_of(chunk));
    let (ao, bo) = (a_out.as_mut_ptr() as usize, b_out.as_mut_ptr() as usize);
    crate::flat_par::team_map_reduce_aff(
        n,
        &owner,
        |i| {
            // SAFETY: item `i` owns output chunk `i` of both tables; chunks are
            // disjoint and both borrows outlive `team_map_reduce_aff`.
            let (ac, bc) = unsafe {
                (
                    core::slice::from_raw_parts_mut((ao as *mut F128).add(i * chunk), chunk),
                    core::slice::from_raw_parts_mut((bo as *mut F128).add(i * chunk), chunk),
                )
            };
            leaf((i, (ac, bc)))
        },
        id,
        op,
    )
}

/// [`flat_chunks`] on the hot pool's spinning team (lane gl1).
pub(crate) fn team_chunks<T>(
    a_out: &mut [F128],
    b_out: &mut [F128],
    chunk: usize,
    leaf: impl Fn((usize, (&mut [F128], &mut [F128]))) -> T + Sync,
    id: impl Fn() -> T + Sync,
    op: impl Fn(T, T) -> T + Sync,
) -> T {
    assert_eq!(a_out.len(), b_out.len());
    assert!(chunk > 0 && a_out.len().is_multiple_of(chunk));
    let n = a_out.len() / chunk;
    let (ao, bo) = (a_out.as_mut_ptr() as usize, b_out.as_mut_ptr() as usize);
    crate::flat_par::team_map_reduce(
        n,
        |i| {
            // SAFETY: item `i` owns output chunk `i` of both tables; chunks are
            // disjoint and both borrows outlive `team_map_reduce`.
            let (ac, bc) = unsafe {
                (
                    core::slice::from_raw_parts_mut((ao as *mut F128).add(i * chunk), chunk),
                    core::slice::from_raw_parts_mut((bo as *mut F128).add(i * chunk), chunk),
                )
            };
            leaf((i, (ac, bc)))
        },
        id,
        op,
    )
}

/// Whether cascade level buffers may be handed over lane-major (see
/// `kernels::x86_64::zc_lm_handoff_enabled`). The lane-major reader is the
/// deferred composed fold, so the handoff also needs that route.
pub(crate) fn zc_lane_major_handoff_on() -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    {
        kernels::x86_64::zc_lm_handoff_enabled() && kernels::x86_64::zc_fold_defer_enabled()
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    {
        false
    }
}

#[cfg(test)]
thread_local! {
    static PACKED_SPLIT_N_HI_OVERRIDE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Packed kernels amortize their prologue at a smaller low half than the
/// plain tail, so keep splitting two bits beyond the plain-tail cap.
const PACKED_SPLIT_MAX_N_HI: usize = 13;

/// Fan-out for the **packed** round-two lookahead sweep and the packed
/// composed n26 pass. Same occupancy identity as [`tail_split_n_hi`], applied
/// to the two remaining `lookahead_n_hi` sites the promoted plain-tail hop
/// left on 128 chunks:
///
/// * ranked r2 (`m=32`, `k_skip=6`): `n_vars = 25` → 8192 chunks of
///   `lo_size = 2^12` instead of 128 chunks of `lo_size = 2^18`
/// * ranked n26: `n_vars = 23` → 8192 chunks of `lo_size = 2^10` instead of
///   128 chunks of `lo_size = 2^16`
///
/// The packed kernels already accept any even `lo_size ≥ 2` (and the
/// residue-major / cfold-bake fast paths need `lo_size` a multiple of 32,
/// which both ranked sizes keep). Per-chunk prologue (eight `eq_hi`
/// multiplies plus a reduce-tree node) is noise against a 30 ms inner walk;
/// the drain of one 128-way chunk is not. Every admissible `n_hi` is
/// value-identical for the same XOR/tensor reason as the plain tail.
/// `FLOCK_NO_ZC_PACKED_SPLIT=1` restores the incumbent 128-chunk split.
#[inline]
/// Per-pass table of the message block's `(w, w·x⁶⁴)` ZMM pairs: entry
/// group `g` holds `[eq_lo[8g+1], eq_lo[8g+3], eq_lo[8g+5], eq_lo[8g+7]]`
/// followed by the same four values multiplied by `x⁶⁴` (mod p) — the split
/// multiplier companion. Both are pure functions of `eq_lo`, so the sweep
/// loads two ZMMs instead of deriving them (a lane permute plus a CLMUL of
/// pure latency on the head of the accumulate chain) every iteration.
/// `eq_lo.len()` must be a multiple of 8.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
/// [`build_w_pair_table`] for the deeper-cascade (`CASC`) passes: lanes `2h`
/// and `2h + 1` of a tile (the two halves of an eight-group) both carry the
/// odd lane's weight `eq_lo[8g + 4h + 3]`.
fn build_w_pair_table_casc(eq_lo: &[F128]) -> Vec<F128> {
    debug_assert!(eq_lo.len().is_multiple_of(8));
    let x64 = F128::new(0, 1);
    let mut t = crate::alloc_uninit_f128_vec(eq_lo.len());
    for g in 0..eq_lo.len() / 8 {
        for k in 0..4 {
            let w = eq_lo[8 * g + 2 * (k | 1) + 1];
            t[8 * g + k] = w;
            t[8 * g + 4 + k] = w * x64;
        }
    }
    t
}

fn build_w_pair_table(eq_lo: &[F128]) -> Vec<F128> {
    debug_assert!(eq_lo.len().is_multiple_of(8));
    let x64 = F128::new(0, 1);
    let mut t = crate::alloc_uninit_f128_vec(eq_lo.len());
    for g in 0..eq_lo.len() / 8 {
        for k in 0..4 {
            let w = eq_lo[8 * g + 2 * k + 1];
            t[8 * g + k] = w;
            t[8 * g + 4 + k] = w * x64;
        }
    }
    t
}

/// Build the per-pass eq bake for the round-two lookahead sweep.
///
/// `r_lo` are the low split's coordinates in `build_eq` order — index bit `t`
/// of `eq_lo` selects `r_lo[t]` — i.e. `mlv_challenges[1..][..n_lo]`.
///
/// The sweep's odd-lane weight for the group at `x_lo = 32·B + 8·g`, lane
/// `lane`, is `eq_lo[32·B + 8·g + 2·lane + 1]`. Those index bits split three
/// ways: bits 0-2 are `(1, lane & 1, lane >> 1)`, bits 3-4 are `(g & 1,
/// g >> 1)`, bits 5.. are `B`. `eq_lo` is the rank-one tensor of `r_lo`, so
///
/// ```text
/// w = LV(lane) · S(g) · Etop(B)
/// ```
///
/// exactly (one reassociation of the same field product). `S` is baked into
/// four A-side matrix sets: the broadcast prefold picks its matrix per
/// eight-row input octet `i`, and octet `i` of a 64-row batch is the four
/// pairs `4i..4i+4`, all inside group `g = i >> 1`. `Etop` is baked into one
/// B-side set per 32-pair batch. `LV` survives as a four-lane vector applied
/// to the lane-reduced accumulators once per worker chunk.
///
/// `eq_lo` is the rank-one `eq` of `r_lo` at the only production call site, so
/// the odd-entry factorization re-check runs only when
/// `FLOCK_NO_ZC_R2_SETUP_TRIM=1` restores it; see [`build_r2_eq_bake_checked`].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn build_r2_eq_bake(
    table: &UniSkipFoldTable,
    eq_lo: &[F128],
    r_lo: &[F128],
) -> Option<kernels::x86_64::R2EqBake> {
    build_r2_eq_bake_checked(table, eq_lo, r_lo, !zc_r2_setup_trim_enabled())
}

/// [`build_r2_eq_bake`] with the factorization re-check explicit: when
/// `verify`, every odd `eq_lo` entry is re-derived as `LV·S·Etop` and any
/// mismatch returns `None`, so the sweep never trusts a foreign low tensor.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn build_r2_eq_bake_checked(
    table: &UniSkipFoldTable,
    eq_lo: &[F128],
    r_lo: &[F128],
    verify: bool,
) -> Option<kernels::x86_64::R2EqBake> {
    use crate::zerocheck::univariate_skip::build_eq;
    use rayon::prelude::*;
    let lo_size = eq_lo.len();
    if r_lo.len() < 5 || (1usize << r_lo.len()) != lo_size || !lo_size.is_multiple_of(32) {
        return None;
    }
    let n_batches = lo_size / 32;
    // f₀·f₁·f₂ at the always-set bit 0: the odd entries of the three-
    // coordinate eq, i.e. lane `l` at index `1 + 2·l`.
    let eq3 = build_eq(&r_lo[..3]);
    let lv: [F128; 4] = std::array::from_fn(|l| eq3[1 + 2 * l]);
    // f₃·f₄, indexed by the 16-row group `g` of the prefold batch.
    let s = build_eq(&r_lo[3..5]);
    // f₅ ⋯ f_{n_lo−1}, indexed by the 32-pair prefold batch.
    let etop = build_eq(&r_lo[5..]);
    if s.len() != 4 || etop.len() != n_batches {
        return None;
    }
    // `eq_lo` is the rank-one tensor of `r_lo`, so its odd entries factor
    // exactly as `LV·S·Etop` (pinned by `r2_eq_bake_factorization_is_exact`).
    // The 4,096-multiply re-verification runs only behind
    // `FLOCK_NO_ZC_R2_SETUP_TRIM=1`.
    if verify {
        for b in 0..n_batches {
            for g in 0..4 {
                for lane in 0..4 {
                    let w = lv[lane] * s[g] * etop[b];
                    if w != eq_lo[32 * b + 8 * g + 2 * lane + 1] {
                        return None;
                    }
                }
            }
        }
    }
    // Scaling a fold table's 64 basis columns scales every XOR-composed image
    // by the same factor, so the matrices of `c · fold` are the matrices of
    // the `c`-scaled basis and the prefold emits `c · fold(row)` for free.
    let cols: [F128; 64] =
        std::array::from_fn(|i| table.data[(i / 8) * 256 + (1usize << (i % 8))]);
    let scaled = |c: F128| {
        let sc: [F128; 64] = std::array::from_fn(|i| c * cols[i]);
        kernels::x86_64::R2FoldMats(kernels::x86_64::build_row_fold_mats_from_cols(&sc))
    };
    let a_mats: [kernels::x86_64::R2FoldMats; 4] = std::array::from_fn(|g| scaled(s[g]));
    // Lane gl1 (`FLOCK_NO_GL1_BAKE_TEAM`): the 128 per-batch matrix sets on
    // the hot pool's spinning team (fork/join-bound under rayon). Same values.
    static GL1_BAKE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GL1_BAKE_TEAM").is_none());
    let b_mats: Vec<kernels::x86_64::R2FoldMats> =
        if *GL1_BAKE && crate::flat_par::team_enabled() && crate::flat_par::team_ready() && etop.len() >= 16 {
            let n = etop.len();
            let mut v: Vec<core::mem::MaybeUninit<kernels::x86_64::R2FoldMats>> = Vec::with_capacity(n);
            // SAFETY: every slot is written exactly once below before the cast.
            unsafe { v.set_len(n) };
            let vp = v.as_mut_ptr() as usize;
            const PER: usize = 4;
            crate::flat_par::team_map_reduce(
                n.div_ceil(PER),
                |i| {
                    for j in i * PER..((i + 1) * PER).min(n) {
                        // SAFETY: item `i` owns slots [i*PER, (i+1)*PER).
                        unsafe {
                            (vp as *mut core::mem::MaybeUninit<kernels::x86_64::R2FoldMats>)
                                .add(j)
                                .write(core::mem::MaybeUninit::new(scaled(etop[j])))
                        };
                    }
                },
                || (),
                |_, _| (),
            );
            // SAFETY: all `n` slots initialized; MaybeUninit<T> has T's layout.
            unsafe { core::mem::transmute::<_, Vec<kernels::x86_64::R2FoldMats>>(v) }
        } else {
            etop.par_iter().map(|e| scaled(*e)).collect()
        };
    Some(kernels::x86_64::R2EqBake {
        a_mats,
        b_mats,
        b_scale: etop,
        lane: lv,
        canonical_elide: false,
    })
}

/// `FLOCK_NO_ZC_R2_SETUP_TRIM=1` restores two pieces of dead round-two setup:
/// the eq-bake factorization re-check and, on the baked route (which never
/// reads it), the `(w, w·x⁶⁴)` pair table. Read once per process.
#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )),
    allow(dead_code)
)]
fn zc_r2_setup_trim_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R2_SETUP_TRIM").is_none());
    *ON
}

fn packed_split_n_hi(n_vars: usize) -> usize {
    let base = lookahead_n_hi(n_vars);
    #[cfg(test)]
    if let Some(over) = PACKED_SPLIT_N_HI_OVERRIDE.with(|c| c.get()) {
        // Leave at least one lo variable (and therefore even lo_size ≥ 2).
        return over.min(n_vars.saturating_sub(1));
    }
    static OFF: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_PACKED_SPLIT").is_some());
    if *OFF {
        return base;
    }
    base.max(PACKED_SPLIT_MAX_N_HI.min(n_vars.saturating_sub(TAIL_SPLIT_MIN_LO_LOG)))
}

/// Build the split used by the packed round-two sweep. The no-materialize
/// caller retains this tiny pair of tensors for the packed rounds-3+4 pass:
/// its high coordinates are identical, while its low tensor is round two's
/// low tensor with the first two coordinates marginalized out.
pub(crate) fn packed_round2_split_eq(mlv_challenges: &[F128]) -> SplitEqGhash {
    let n_vars = mlv_challenges.len() - 1;
    SplitEqGhash::with_n_hi(&mlv_challenges[1..], packed_split_n_hi(n_vars))
}

/// Remove the two least-significant coordinates from an eq tensor. In
/// characteristic two, the four weights of those coordinates XOR to one, so
/// each four-entry group collapses to the untouched suffix weight exactly.
pub(crate) fn marginalize_eq_low2(eq: &[F128]) -> Vec<F128> {
    assert!(eq.len().is_power_of_two() && eq.len() >= 4);
    eq.chunks_exact(4)
        .map(|v| v[0] + v[1] + v[2] + v[3])
        .collect()
}

/// Round-two fused fold **plus** the deferred round-three coefficients.
///
/// The folded tables and the round-two wire message are bit-identical to
/// [`uni_skip_fold_and_round_pair_optimized_packed_padded`]; the sweep merely
/// also accumulates six aggregates over round-three groups `y = x'/2` (raw
/// post-URM rows `a0..a3 = A[4y..4y+4]`, likewise `b`):
///
/// ```text
/// W0 = Σ_y eq₃(y)·a2b2   W1 = Σ eq₃·a3b3        W2 = Σ eq₃·(a2+a3)(b2+b3)
/// W3 = Σ_y eq₃(y)·e_a e_b   W4 = Σ eq₃·o_a o_b   W5 = Σ eq₃·(e+o)_a (e+o)_b
/// e = a0+a2,  o = a1+a3
/// ```
///
/// With the fold convention `A'[x'] = A[2x'] + ρ₁(A[2x']+A[2x'+1])`, both
/// round-three message points expand over `{1, ρ₁, ρ₁²}`:
///
/// ```text
/// G₃(1) = W0 + ρ₁(W0+W1+W2) + ρ₁²W2      G₃(∞) = W3 + ρ₁(W3+W4+W5) + ρ₁²W5
/// ```
///
/// `W1` and `W2` cost **zero extra multiplies**: they are the odd-parity half
/// of the two round-two accumulators, because `eq₂(2y+1) = r₁·eq₃(y)` and
/// `eq₂(2y) = (1+r₁)·eq₃(y)` where `r₁ = mlv_challenges[1]`. The kernel uses
/// the odd lane's weight `w = eq₂_lo[2u+1]` for the whole group (four
/// reduced pre-scalings of the `a` rows, then every product is one unreduced
/// multiply); the driver restores the even lane with `κ = (1+r₁)·r₁⁻¹` and
/// puts all six aggregates back on `eq₃` with one `r₁⁻¹` — exact field
/// arithmetic, so the round-two message is bit-identical to the incumbent.
///
/// Requires `r₁ ≠ 0`; the caller falls back to the incumbent route otherwise.
///
/// Returns `(a_folded, b_folded, mlv_challenges[0]·G₂(1), G₂(∞), lookahead)`.
pub fn uni_skip_fold_and_round_pair_optimized_packed_padded_lookahead(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, F128, F128, Round3Lookahead) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let r2_mats = {
        #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
        {
            r2_gfni_mats(table)
        }
        #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
        {
            None::<[u64; 128]>
        }
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[allow(unused_variables)]
    let r2_mats_arg = r2_mats.as_ref();

    use rayon::prelude::*;
    assert_eq!(k_skip, 6, "lookahead round two is k_skip=6 only");
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);
    let r1 = mlv_challenges[1];
    assert_ne!(r1, F128::ZERO, "lookahead requires a non-zero r[k_skip+1]");

    // Uninit alloc — every slot is written below (see the incumbent).
    let mut a_folded: Vec<F128> = crate::scratch::take_f128(n_out);
    let mut b_folded: Vec<F128> = crate::scratch::take_f128(n_out);

    let n_vars = mlv_challenges.len() - 1;
    let eq = SplitEqGhash::with_n_hi(&mlv_challenges[1..], packed_split_n_hi(n_vars));
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_out);
    assert!(lo_size >= 2, "lookahead sweep pairs two x_lo per group");
    // `eq₂(2y) = (1+r₁)·eq₃(y)` and `eq₂(2y+1) = r₁·eq₃(y)`: the sweep uses
    // the odd lane as the group's single weight and the two constants below
    // put every aggregate back on its own scale, once, off the hot path.
    let (kappa, r1_inv) = lookahead_inv_factors(r1);
    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    // Per-pass (w, w·x⁶⁴) pair table for the message block: both are pure
    // functions of the odd eq_lo lanes, hoisted out of the sweep (the
    // companion CLMUL sat on the head of the chain feeding all eight
    // accumulates). Interleaved per 8-lo group: [w×4, w·x⁶⁴×4].
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_vec = if kernels::x86_64::zc_wtab_enabled() && lo_size.is_multiple_of(8) {
        Some(build_w_pair_table(eq_lo))
    } else {
        None
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_arg = wtab_vec.as_deref();
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);

    // Per-chunk: (even-parity pair, six lookahead partials), all already
    // scaled by eq_hi[x_hi]. Reduced by F128 XOR (commutative, associative).
    let (sum1_even, sum_inf_even, agg) = a_folded
        .par_chunks_mut(chunk_size)
        .zip(b_folded.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(x_hi, (a_chunk, b_chunk))| {
            let row_base = x_hi * chunk_size;
            let pair_idx_base = x_hi * lo_size;

            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            // SAFETY: the packed inputs expose 8 readable bytes per post-URM
            // row of this chunk, the table has the protocol-fixed 8 × 256
            // shape, and the cfg gate supplies every intrinsic feature.
            let out = unsafe {
                kernels::x86_64::round2_lookahead_chunk_x86_avx512::<true, false>(
                    table.data.as_ptr(),
                    r2_mats_arg,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    row_base,
                    a_chunk,
                    b_chunk,
                    eq_lo,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                    wtab_arg,
                    None,
                )
            };
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            let out = round2_lookahead_chunk_scalar::<true>(
                a_packed,
                b_packed,
                table,
                a_chunk,
                b_chunk,
                eq_lo,
                row_base,
                pair_idx_base,
                pair_in_block_mask,
                useful_pairs_inclusive,
            );

            let eq_h = eq_hi[x_hi];
            // `out[0..2]` carry the even lane on the odd lane's weight; κ
            // restores `eq₂(2y)` after the chunk reduction below.
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
            )
        })
        .reduce(
            || (F128::ZERO, F128::ZERO, [F128::ZERO; 6]),
            |(s1, sinf, sa), (c1, cinf, ca)| {
                let mut a = sa;
                for (x, y) in a.iter_mut().zip(ca.iter()) {
                    *x += *y;
                }
                (s1 + c1, sinf + cinf, a)
            },
        );

    // The odd sums already ride in agg[0..2]. Apply the shared parity
    // factor twice for the whole pass rather than twice per chunk.
    let sum1 = kappa * sum1_even + agg[0];
    let sum_inf = kappa * sum_inf_even + agg[1];
    let la = lookahead_from_odd_weighted(&agg, r1_inv);
    (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf, la)
}

/// Assemble the deferred quadratic from the six aggregates
/// `[r·W1, r·W2, r·W0, r·W3, r·W4, r·W5]` (the order the chunk kernels report
/// them in), all accumulated on the odd lane's weight `r·eq_next`; one `r⁻¹`
/// puts them back on `eq_next`.
#[inline]
fn lookahead_from_odd_weighted(agg: &[F128; 6], r_inv: F128) -> Round3Lookahead {
    let w1 = r_inv * agg[0];
    let w2 = r_inv * agg[1];
    let w0 = r_inv * agg[2];
    let w3 = r_inv * agg[3];
    let w4 = r_inv * agg[4];
    let w5 = r_inv * agg[5];
    Round3Lookahead {
        c: [w0, w0 + w1 + w2, w2, w3, w3 + w4 + w5, w5],
    }
}

/// Rows of the unbound identity-C inner fold the round-two canonical-window
/// elision consumes: rows `0..16`, then row `240`, 64 `F128` each.
pub(crate) const CANON_ROWS: usize = 17;

/// `FLOCK_NO_ZC_R2_CANON_ELIDE=1` restores the two canonical B windows in the
/// round-two sweep and skips their round-one capture. Read once per process.
pub(crate) fn round2_canonical_elide_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R2_CANON_ELIDE").is_none());
    *ON
}

/// Exact round-two message and deferred round-three contribution of the two
/// canonical B windows of every ranked block, from the identity-C inner fold.
///
/// Ranked witness brand, per 128-pair block: rows `0..16` have `B = 1` and
/// `A = z` (static windows 0/1); row 240 has the static `B` word
/// `0x0001_ffff_ffff_ffff` and `A = z`, and rows `241..256` are zero on both
/// sides (window 30/31). `rows[slot·64 + s] = Σ_blocks eq(r_blk, blk)·z(blk,
/// row, s)` with `r_blk = mlv_challenges[8..]` (slot = row, or 16 for row 240).
/// With `col_s` the fold table's basis column (`L_s(z)`), `F(row) = Σ_s col_s ·
/// rows[slot·64 + s]` is the eq-weighted fold of the A row over all blocks,
/// `v0 = Σ_s col_s` the image of the all-ones B row and `v1 = Σ_{s<49} col_s`
/// the image of row 240's B word. Expanding the sweep's accumulators
/// (`round2_lookahead_chunk_scalar`) over those pairs, with `eq7 =
/// eq(mlv[1..8])` over a pair's in-block index and `eq6 = eq(mlv[2..8])` over
/// a group's (`eq(pair 2y+1) = mlv[1]·eq6(y)` absorbs the odd-lane weighting):
///
/// ```text
/// ones  (pairs 0..8):  ΔG(1)  = v0·Σ_{u<8} eq7(u)·F(2u+1)
///                      ΔW0 = v0·Σ_{y<4} eq6(y)·F(4y+2)   ΔW1 = v0·Σ_{y<4} eq6(y)·F(4y+3)
///                      (every other term multiplies a B difference or an even sum of ones)
/// sparse (pair 120):   ΔG(∞) = v1·eq7(120)·F(240)
///                      ΔW3 = ΔW5 = v1·eq6(60)·F(240)
/// ```
///
/// Returns `(ΔG(1), ΔG(∞), Δc)` with `Δc` in [`Round3Lookahead`]'s layout
/// `[W0, W0+W1+W2, W2, W3, W3+W4+W5, W5]`. Every term is an exact field sum,
/// so adding these to the sweep's sums over the remaining pairs reproduces
/// the incumbent values bit for bit.
#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )),
    allow(dead_code)
)]
pub(crate) fn round2_canonical_window_deltas(
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    rows: &[F128],
) -> (F128, F128, [F128; 6]) {
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

    // Build eq6[0..4] = eq(mlv_challenges[2..8], 0..4) and eq6_60 = eq(mlv_challenges[2..8], 60)
    // on the stack without allocating 64- and 128-element heap vectors.
    // Since indices 0..4 have bits 2..5 = 0, their common high-bit factor is
    // top4 = ∏_{k=4..8} (1 + mlv_challenges[k]).
    // Index 60 = 0b111100 has bits 2..5 = 1 and bits 0..1 = 0, so its high-bit factor is
    // hi4 = ∏_{k=4..8} mlv_challenges[k].
    let r1 = mlv_challenges[1];
    let r2 = mlv_challenges[2];
    let r3 = mlv_challenges[3];
    let top4 = (F128::ONE + mlv_challenges[4])
        * (F128::ONE + mlv_challenges[5])
        * (F128::ONE + mlv_challenges[6])
        * (F128::ONE + mlv_challenges[7]);
    let hi4 = mlv_challenges[4] * mlv_challenges[5] * mlv_challenges[6] * mlv_challenges[7];
    let mut eq6_low = [F128::ZERO; 4];
    eq6_low[0] = top4;
    {
        let hi = eq6_low[0] * r2;
        eq6_low[1] = hi;
        eq6_low[0] += hi;
    }
    for x in (0..2).rev() {
        let hi = eq6_low[x] * r3;
        eq6_low[x | 2] = hi;
        eq6_low[x] += hi;
    }
    let eq6_60 = hi4 * (F128::ONE + r2) * (F128::ONE + r3);
    // Index 120 in eq7 (challenges[1..8]) is 2 * 60 = 0b1111000, which has bit 0 (challenge[1]) = 0
    // and bits 1..6 matching index 60 of eq6.
    let eq7_120 = eq6_60 * (F128::ONE + r1);

    let mut g1 = F128::ZERO;
    let (mut w0, mut w1) = (F128::ZERO, F128::ZERO);
    for y in 0..4 {
        let eq6_y = eq6_low[y];
        let eq7_2y_1 = eq6_y * r1;
        let eq7_2y_0 = eq6_y + eq7_2y_1;
        let f_4y_1 = fold(4 * y + 1);
        let f_4y_2 = fold(4 * y + 2);
        let f_4y_3 = fold(4 * y + 3);
        g1 += eq7_2y_0 * f_4y_1 + eq7_2y_1 * f_4y_3;
        w0 += eq6_y * f_4y_2;
        w1 += eq6_y * f_4y_3;
    }
    let f240 = fold(16);
    let d_g1 = v0 * g1;
    let d_ginf = v1 * (eq7_120 * f240);
    let dw0 = v0 * w0;
    let dw1 = v0 * w1;
    let dw3 = v1 * (eq6_60 * f240);
    let dw5 = dw3;
    (
        d_g1,
        d_ginf,
        [dw0, dw0 + dw1, F128::ZERO, dw3, dw3 + dw5, dw5],
    )
}

/// No-materialize round-two sweep: the round-two message and the deferred
/// round-three coefficients, computed exactly as
/// [`uni_skip_fold_and_round_pair_optimized_packed_padded_lookahead`] does,
/// but **without writing the two folded tables** (2 GiB of stores at the
/// ranked shape). The composed rounds-3+4 pass re-derives the folded rows it
/// needs straight from the packed witness
/// ([`fold2_from_packed_and_round_pair_lookahead_into`]).
///
/// Returns `(mlv_challenges[0]·G₂(1), G₂(∞), lookahead)`.
pub fn uni_skip_round_pair_lookahead_nomat_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (F128, F128, Round3Lookahead) {
    uni_skip_round_pair_lookahead_nomat_packed_padded_with_eq(
        a_packed,
        b_packed,
        m,
        k_skip,
        table,
        mlv_challenges,
        padding,
        None,
        None,
        false,
    )
    .0
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn uni_skip_round_pair_lookahead_nomat_packed_padded_with_eq(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
    eq_override: Option<&SplitEqGhash>,
    canon: Option<&[F128]>,
    want_r4: bool,
) -> ((F128, F128, Round3Lookahead), Option<Round4Lookahead>) {
    use rayon::prelude::*;
    crate::rstrace::ev("r2:in", 0);
    assert_eq!(k_skip, 6, "lookahead round two is k_skip=6 only");
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);
    let r1 = mlv_challenges[1];
    assert_ne!(r1, F128::ZERO, "lookahead requires a non-zero r[k_skip+1]");

    let eq_owned;
    let eq = if let Some(eq) = eq_override {
        eq
    } else {
        let n_vars = mlv_challenges.len() - 1;
        eq_owned = SplitEqGhash::with_n_hi(&mlv_challenges[1..], packed_split_n_hi(n_vars));
        &eq_owned
    };
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_out);
    assert!(lo_size >= 2, "lookahead sweep pairs two x_lo per group");
    let (kappa, r1_inv) = lookahead_inv_factors(r1);
    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let gfni_ok = table.n_chunks == 8 && zc_gfni_enabled();
    // Per-pass eq bake: the odd-lane weight factored into the prefold's
    // matrix sets (A per broadcast octet, B per 32-pair batch) plus a
    // period-two lane vector, so the message block consumes pre-weighted rows
    // and its four `ghash_mul_x4_split` prescales per four-group iteration
    // disappear. `FLOCK_NO_R2_EQ_BAKE=1` rolls back to the prescales.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let mut bake_vec = {
        #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
        {
            if gfni_ok
                && kernels::x86_64::zc_r2_bake_route_ok(lo_size)
                && mlv_challenges.len() > eq.n_lo
            {
                build_r2_eq_bake(table, eq_lo, &mlv_challenges[1..1 + eq.n_lo])
            } else {
                None
            }
        }
        #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
        {
            None::<kernels::x86_64::R2EqBake>
        }
    };
    // Ranked canonical-window elision: `canon` (rows of the identity-C inner
    // fold, see `round2_canonical_window_deltas`) arrives only on the ranked
    // witness brand; the sweep skips the two canonical B windows only on the
    // bake route, and exactly then the driver adds their contribution back.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let canon_elided = {
        let rows = canon.filter(|rows| {
            rows.len() == CANON_ROWS * 64
                && padding.k_log == 14
                && padding.useful_bits_per_block == 15_409
                && m >= k_skip + 8
                && round2_canonical_elide_enabled()
        });
        if rows.is_some()
            && let Some(bk) = bake_vec.as_mut()
        {
            bk.canonical_elide = true;
        }
        rows.filter(|_| bake_vec.is_some())
    };
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    let canon_elided: Option<&[F128]> = {
        let _ = canon;
        None
    };
    crate::rstrace::ev("r2:bake", 0);
    // Per-pass (w, w·x⁶⁴) pair table for the message block: both are pure
    // functions of the odd eq_lo lanes, hoisted out of the sweep (the
    // companion CLMUL sat on the head of the chain feeding all eight
    // accumulates). Interleaved per 8-lo group: [w×4, w·x⁶⁴×4].
    // The baked message block never reads the `(w, w·x⁶⁴)` pair table, so it
    // is built only for the prescale route (unless the setup trim is off).
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_vec = if kernels::x86_64::zc_wtab_enabled()
        && lo_size.is_multiple_of(8)
        && !(zc_r2_setup_trim_enabled() && bake_vec.is_some())
    {
        Some(build_w_pair_table(eq_lo))
    } else {
        None
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_arg = wtab_vec.as_deref();
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let bake_arg = bake_vec.as_ref();
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let fast_r2 = kernels::x86_64::round2_baked_elide_route_ok(gfni_ok, lo_size, bake_arg);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let r2_mats = {
        #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
        {
            if fast_r2 && (lo_size & 127) == 0 && (chunk_size & 255) == 0 {
                None
            } else {
                r2_gfni_mats(table)
            }
        }
        #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
        {
            None::<[u64; 128]>
        }
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[allow(unused_variables)]
    let r2_mats_arg = r2_mats.as_ref();
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);

    // ZC-3VAR: the ranked fast kernel also carries the deferred round-four
    // biquadratic. Only that route computes it in the sweep; elsewhere the
    // caller falls back to the two-variable route (tests use the reference).
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let r4_fast = want_r4
        && fast_r2
        && (lo_size & 127) == 0
        && bake_arg.is_some()
        && mlv_challenges.len() >= 4;
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )))]
    let r4_fast = false;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let r4_lv = if r4_fast {
        let r3 = mlv_challenges[3];
        let one = F128::ONE;
        Some(kernels::x86_64::R4LaneWeights {
            lv4: [one + r3, r3, one + r3, r3],
            lv4_odd: [F128::ZERO, one + r3, F128::ZERO, r3],
        })
    } else {
        None
    };

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let lo1_ys = kernels::x86_64::lo1_ys_enabled();
    crate::rstrace::ev("r2:par0", hi_size as u32);
    let leaf = |x_hi: usize| {
            crate::rstrace::item_start();
            #[allow(unused_mut)]
            let mut r4_out = [F128::ZERO; 18];
            let x_in = x_hi;
            let row_base = x_in * chunk_size;
            let pair_idx_base = x_in * lo_size;
            let mut none_a: [F128; 0] = [];
            let mut none_b: [F128; 0] = [];

            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            // SAFETY: the packed inputs expose 8 readable bytes per post-URM
            // row of this chunk, the table has the protocol-fixed 8 × 256
            // shape, WRITE=false touches no chunk, and the cfg gate supplies
            // every intrinsic feature.
            let out = unsafe {
                #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
                if fast_r2
                    && (pair_idx_base & 127) == 0
                    && (row_base & 255) == 0
                    && let Some(bk) = bake_arg
                {
                    if let Some(lw) = r4_lv.as_ref() {
                        let (o, r4) = kernels::x86_64::round2_lookahead_chunk_baked_elide_r4_x86(
                            a_packed.as_ptr(),
                            b_packed.as_ptr(),
                            row_base,
                            lo_size,
                            bk,
                            lw,
                            lo1_ys,
                        );
                        r4_out = r4;
                        o
                    } else {
                        kernels::x86_64::round2_lookahead_chunk_baked_elide_x86(
                            a_packed.as_ptr(),
                            b_packed.as_ptr(),
                            row_base,
                            lo_size,
                            bk,
                        )
                    }
                } else if let Some(bk) = bake_arg {
                    kernels::x86_64::round2_lookahead_chunk_x86_avx512::<false, true>(
                        table.data.as_ptr(),
                        r2_mats_arg,
                        a_packed.as_ptr(),
                        b_packed.as_ptr(),
                        row_base,
                        &mut none_a,
                        &mut none_b,
                        eq_lo,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        wtab_arg,
                        Some(bk),
                    )
                } else {
                    kernels::x86_64::round2_lookahead_chunk_x86_avx512::<false, false>(
                        table.data.as_ptr(),
                        r2_mats_arg,
                        a_packed.as_ptr(),
                        b_packed.as_ptr(),
                        row_base,
                        &mut none_a,
                        &mut none_b,
                        eq_lo,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        wtab_arg,
                        None,
                    )
                }
                #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
                if let Some(bk) = bake_arg {
                    kernels::x86_64::round2_lookahead_chunk_x86_avx512::<false, true>(
                        table.data.as_ptr(),
                        r2_mats_arg,
                        a_packed.as_ptr(),
                        b_packed.as_ptr(),
                        row_base,
                        &mut none_a,
                        &mut none_b,
                        eq_lo,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        wtab_arg,
                        Some(bk),
                    )
                } else {
                    kernels::x86_64::round2_lookahead_chunk_x86_avx512::<false, false>(
                        table.data.as_ptr(),
                        r2_mats_arg,
                        a_packed.as_ptr(),
                        b_packed.as_ptr(),
                        row_base,
                        &mut none_a,
                        &mut none_b,
                        eq_lo,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        wtab_arg,
                        None,
                    )
                }
            };
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            let out = round2_lookahead_chunk_scalar::<false>(
                a_packed,
                b_packed,
                table,
                &mut none_a,
                &mut none_b,
                eq_lo,
                row_base,
                pair_idx_base,
                pair_in_block_mask,
                useful_pairs_inclusive,
            );

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
                if r4_fast { r4_out.map(|x| eq_h * x) } else { r4_out },
            )
    };
    type R2Sums = (F128, F128, [F128; 6], [F128; 18]);
    let id = || -> R2Sums { (F128::ZERO, F128::ZERO, [F128::ZERO; 6], [F128::ZERO; 18]) };
    let op = |(s1, sinf, sa, sr): R2Sums, (c1, cinf, ca, cr): R2Sums| -> R2Sums {
        let mut a = sa;
        for (x, y) in a.iter_mut().zip(ca.iter()) {
            *x += *y;
        }
        let mut r4 = sr;
        for (x, y) in r4.iter_mut().zip(cr.iter()) {
            *x += *y;
        }
        (s1 + c1, sinf + cinf, a, r4)
    };
    // Lane ru flat claim route (`FLOCK_NO_RU_FLAT_PAR`), see `flat_tail`.
    // Lane gl1 (`FLOCK_NO_GL1_R2_DELTA_IDLE`): the canonical-window deltas
    // (pure functions of the fold table, challenges and rows) are computed by
    // the caller while the other claimers drain the sweep. Same values.
    static GL1_DIDLE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GL1_R2_DELTA_IDLE").is_none());
    let mut early_d: Option<(F128, F128, [F128; 6])> = None;
    let mut early_d4: Option<Round4Lookahead> = None;
    let (sum1_even, sum_inf_even, agg, r4_flat) =
        if crate::flat_par::enabled() && rayon::current_thread_index().is_some() {
            if *GL1_DIDLE && canon_elided.is_some() {
                let rows = canon_elided.unwrap();
                crate::flat_par::map_reduce_idle(hi_size, leaf, id, op, || {
                    early_d = Some(round2_canonical_window_deltas(table, mlv_challenges, rows));
                    if r4_fast {
                        early_d4 = Some(three_var::round4_canonical_window_deltas(table, mlv_challenges, rows));
                    }
                })
            } else {
                crate::flat_par::map_reduce(hi_size, leaf, id, op)
            }
        } else {
            (0..hi_size).into_par_iter().map(leaf).reduce(id, op)
        };
    crate::rstrace::ev("r2:par1", 0);
    crate::rstrace::flush_items("r2:first", "r2:last");

    crate::rstrace::ev("r2:red", 0);
    let sum1 = kappa * sum1_even + agg[0];
    let sum_inf = kappa * sum_inf_even + agg[1];
    let mut la = lookahead_from_odd_weighted(&agg, r1_inv);
    let (mut sum1, mut sum_inf) = (sum1, sum_inf);
    if let Some(rows) = canon_elided {
        let (d_g1, d_ginf, d_c) =
            early_d.unwrap_or_else(|| round2_canonical_window_deltas(table, mlv_challenges, rows));
        sum1 += d_g1;
        sum_inf += d_ginf;
        for (c, d) in la.c.iter_mut().zip(d_c) {
            *c += d;
        }
    }
    let r4 = if r4_fast {
        let mut r4 = Round4Lookahead::from_flat(&r4_flat);
        if let Some(rows) = canon_elided {
            match early_d4.take() {
                Some(d4) => r4.add(&d4),
                None => r4.add(&three_var::round4_canonical_window_deltas(
                    table,
                    mlv_challenges,
                    rows,
                )),
            }
        }
        Some(r4)
    } else if want_r4 && cfg!(test) {
        Some(three_var::round4_lookahead_ref(
            a_packed,
            b_packed,
            table,
            mlv_challenges,
            padding,
            k_skip,
        ))
    } else {
        let _ = r4_flat;
        None
    };
    crate::rstrace::ev("r2:out", 0);
    ((mlv_challenges[0] * sum1, sum_inf, la), r4)
}

/// Composed rounds-3+4 pass **from the packed witness**: re-derives the
/// round-two folded rows through the same byte-table gathers the sweep uses
/// (`fold_z` is linear over the packed bytes, so the same bytes give the same
/// `F128`s the materializing sweep would have stored, padded pairs included),
/// then applies exactly the composed fold of
/// [`fold2_plain_and_round_pair_lookahead_into`] and emits the round-four
/// message plus the deferred round-five coefficients.
///
/// Reads 1 GiB of packed bytes instead of 2 GiB of dense `F128` at the ranked
/// shape, and lets the sweep skip its 2 GiB store stream entirely.
///
/// `a_out`/`b_out` have length `2^(m-k_skip-2)`; `r_next4.len() = m − k_skip
/// − 2`, `r_next4[1] ≠ 0` (parity weight of the round-five lookahead).
#[allow(clippy::too_many_arguments)]
pub fn fold2_from_packed_and_round_pair_lookahead_into(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    padding: &PaddingSpec,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho1: F128,
    rho2: F128,
    r_next4: &[F128],
) -> (F128, F128, Round3Lookahead) {
    fold2_from_packed_and_round_pair_lookahead_into_with_eq(
        a_packed, b_packed, m, k_skip, table, padding, a_out, b_out, rho1, rho2, r_next4, None,
        false,
    )
}

/// [`fold2_from_packed_and_round_pair_lookahead_into`] with an optional
/// retained eq split and the output tile layout: `out_lm` stores lane-major
/// sixteen-element tiles for a lane-major next cascade level (see
/// [`zc_lane_major_handoff_on`]); it requires eight-group chunks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_from_packed_and_round_pair_lookahead_into_with_eq(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    padding: &PaddingSpec,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho1: F128,
    rho2: F128,
    r_next4: &[F128],
    eq_override: Option<(&[F128], &[F128])>,
    out_lm: bool,
) -> (F128, F128, Round3Lookahead) {
    // Composed-fold coefficients: expanding the two pair-fold levels (char 2)
    // gives `out = Σ_k c_k · fold(row 4x+k)` with
    //   c₀=(1+ρ₁)(1+ρ₂)  c₁=ρ₁(1+ρ₂)  c₂=(1+ρ₁)ρ₂  c₃=ρ₁ρ₂,
    // exactly the coefficients of `fold16_to_4_deferred`'s expansion. The fold
    // is `F128`-linear in the table, so scaling the table by `c_k` and folding
    // is the same field element as folding and then multiplying — the
    // multiplies move into the batch's bit matrices and vanish from the sweep.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let cfold_mats = {
        #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
        {
            (table.n_chunks == 8 && zc_gfni_enabled() && cfold_bake_enabled()).then(|| {
                let one = F128::ONE;
                let coeffs = [
                    (one + rho1) * (one + rho2),
                    rho1 * (one + rho2),
                    (one + rho1) * rho2,
                    rho1 * rho2,
                ];
                kernels::x86_64::build_cfold_mats(&table.data, coeffs)
            })
        }
        #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
        {
            None::<kernels::x86_64::CFoldMats>
        }
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[allow(unused_variables)]
    let cfold_arg = cfold_mats.as_ref();

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let r2_mats = {
        #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
        {
            if cfold_arg.is_some() && kernels::x86_64::zc_regfold_enabled() {
                None
            } else {
                r2_gfni_mats(table)
            }
        }
        #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
        {
            None::<[u64; 128]>
        }
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[allow(unused_variables)]
    let r2_mats_arg = r2_mats.as_ref();

    use rayon::prelude::*;
    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n = 1usize << (m - k_skip);
    assert!(n >= 16);
    assert_eq!(a_packed.len(), n * n_chunks);
    assert_eq!(b_packed.len(), n * n_chunks);
    let quarter = n / 4;
    assert_eq!(a_out.len(), quarter);
    assert_eq!(b_out.len(), quarter);
    assert_eq!(r_next4.len(), m - k_skip - 2);
    let r = r_next4[1];
    assert_ne!(
        r,
        F128::ZERO,
        "cascade lookahead requires a non-zero parity weight"
    );

    let eq_owned;
    let (eq_lo, eq_hi) = if let Some(eq) = eq_override {
        eq
    } else {
        let n_vars = r_next4.len() - 1;
        eq_owned = SplitEqGhash::with_n_hi(&r_next4[1..], packed_split_n_hi(n_vars));
        (&eq_owned.lo[..], &eq_owned.hi[..])
    };
    let lo_size = eq_lo.len();
    let hi_size = eq_hi.len();
    assert!(lo_size >= 2, "composed lookahead requires lo_size ≥ 2");
    assert_eq!(lo_size * hi_size * 2, quarter);
    assert!(
        !out_lm || (lo_size.is_multiple_of(8) && zc_lane_major_handoff_on()),
        "lane-major composed output needs eight-group AVX-512 chunks"
    );
    let (kappa, r_inv) = lookahead_inv_factors(r);
    let chunk_out = 2 * lo_size;
    // Per-pass (w, w·x⁶⁴) pair table for the message block: both are pure
    // functions of the odd eq_lo lanes, hoisted out of the sweep (the
    // companion CLMUL sat on the head of the chain feeding all eight
    // accumulates). Interleaved per 8-lo group: [w×4, w·x⁶⁴×4].
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_vec = if kernels::x86_64::zc_wtab_enabled() && lo_size.is_multiple_of(8) {
        Some(build_w_pair_table(eq_lo))
    } else {
        None
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_arg = wtab_vec.as_deref();
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);

    // NT publish of the fold outputs: only when the outputs are too large to
    // be LLC-resident when the next cascade level reads them (2^23 F128 =
    // 128 MiB per array selects the rounds-3+4 level alone at the ranked
    // shape; later levels' outputs ARE cache-resident and NT would hurt).
    // The message terms come from registers, so the stores are write-once.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let nt_out = kernels::x86_64::zc_fold_nt_enabled()
        && quarter >= (1 << 23)
        && (a_out.as_ptr() as usize) % 16 == 0
        && (b_out.as_ptr() as usize) % 16 == 0;
    // Lane-major composed prefold (see `zc_c4_lm_enabled`), once per pass.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let c4_lm = kernels::x86_64::zc_c4_lm_enabled();
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    let fast_r34 = kernels::x86_64::fold2_packed_c4_lm_wtab_route_ok(
        r2_mats_arg.is_some() || cfold_arg.is_some(),
        c4_lm,
        lo_size,
        cfold_arg.is_some(),
        wtab_arg.is_some(),
    );

    let (sum1_even, sum_inf_even, agg) = a_out
        .par_chunks_mut(chunk_out)
        .zip(b_out.par_chunks_mut(chunk_out))
        .enumerate()
        .map(|(x_hi, (a_out, b_out))| {
            // Output x of this chunk ← packed rows 4x..4x+4 (pairs 2x, 2x+1).
            let out_base = x_hi * chunk_out;

            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            // SAFETY: the packed inputs expose 8 readable bytes for every row
            // 4·out_base .. 4·(out_base + chunk_out); the table has the
            // protocol-fixed shape; the cfg gate supplies every feature.
            let out = unsafe {
                #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
                if fast_r34
                    && let (Some(c), Some(wt)) = (cfold_arg, wtab_arg)
                {
                    let ap = a_packed.as_ptr();
                    let bp = b_packed.as_ptr();
                    match (nt_out, out_lm) {
                        (true, true) => {
                            kernels::x86_64::fold2_from_packed_lookahead_c4_lm_wtab_x86::<true, true>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, wt,
                            )
                        }
                        (true, false) => {
                            kernels::x86_64::fold2_from_packed_lookahead_c4_lm_wtab_x86::<true, false>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, wt,
                            )
                        }
                        (false, true) => {
                            kernels::x86_64::fold2_from_packed_lookahead_c4_lm_wtab_x86::<false, true>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, wt,
                            )
                        }
                        (false, false) => {
                            kernels::x86_64::fold2_from_packed_lookahead_c4_lm_wtab_x86::<false, false>(
                                ap, bp, out_base, a_out, b_out, lo_size, c, wt,
                            )
                        }
                    }
                } else {
                    kernels::x86_64::fold2_from_packed_lookahead_x86_avx512(
                        table.data.as_ptr(),
                        r2_mats_arg,
                        a_packed.as_ptr(),
                        b_packed.as_ptr(),
                        out_base,
                        a_out,
                        b_out,
                        rho1,
                        rho2,
                        eq_lo,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        nt_out,
                        cfold_arg,
                        wtab_arg,
                        c4_lm,
                        out_lm,
                    )
                }
                #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
                kernels::x86_64::fold2_from_packed_lookahead_x86_avx512(
                    table.data.as_ptr(),
                    r2_mats_arg,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    out_base,
                    a_out,
                    b_out,
                    rho1,
                    rho2,
                    eq_lo,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                    nt_out,
                    cfold_arg,
                    wtab_arg,
                    c4_lm,
                    out_lm,
                )
            };
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            let out = fold2_from_packed_lookahead_scalar(
                a_packed,
                b_packed,
                table,
                out_base,
                a_out,
                b_out,
                rho1,
                rho2,
                eq_lo,
                pair_in_block_mask,
                useful_pairs_inclusive,
            );

            let eq_h = eq_hi[x_hi];
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
            )
        })
        .reduce(
            || (F128::ZERO, F128::ZERO, [F128::ZERO; 6]),
            |(s1, sinf, sa), (c1, cinf, ca)| {
                let mut a = sa;
                for (x, y) in a.iter_mut().zip(ca.iter()) {
                    *x += *y;
                }
                (s1 + c1, sinf + cinf, a)
            },
        );

    let sum1 = kappa * sum1_even + agg[0];
    let sum_inf = kappa * sum_inf_even + agg[1];
    let la = lookahead_from_odd_weighted(&agg, r_inv);
    (r_next4[0] * sum1, sum_inf, la)
}

/// Portable leaf of [`fold2_from_packed_and_round_pair_lookahead_into`] for
/// one worker chunk (`out_base` = first output index of the chunk). Returns
/// the eight per-chunk sums on the odd lane's weight, each reduced. Also the
/// oracle for the AVX-512 kernel.
#[cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ),
    allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_from_packed_lookahead_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    table: &UniSkipFoldTable,
    out_base: usize,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho1: F128,
    rho2: F128,
    eq_lo: &[F128],
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [F128; 8] {
    let n_chunks = table.n_chunks;
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());
    debug_assert_eq!(b_out.len(), 2 * eq_lo.len());
    debug_assert!(eq_lo.len().is_multiple_of(2));
    // Composed output `x` (global) from packed rows 4x..4x+4.
    let out_at = |x: usize| -> (F128, F128) {
        let mut ta = [F128::ZERO; 2];
        let mut tb = [F128::ZERO; 2];
        for half in 0..2 {
            let pair = 2 * x + half;
            if (pair & pair_in_block_mask) >= useful_pairs_inclusive {
                continue; // the sweep wrote zeros here: t = 0
            }
            let r0 = 2 * pair;
            let r1 = r0 + 1;
            let a0 = table.fold_one_row(&a_packed[r0 * n_chunks..(r0 + 1) * n_chunks]);
            let a1 = table.fold_one_row(&a_packed[r1 * n_chunks..(r1 + 1) * n_chunks]);
            let b0 = table.fold_one_row(&b_packed[r0 * n_chunks..(r0 + 1) * n_chunks]);
            let b1 = table.fold_one_row(&b_packed[r1 * n_chunks..(r1 + 1) * n_chunks]);
            ta[half] = a0 + rho1 * (a0 + a1);
            tb[half] = b0 + rho1 * (b0 + b1);
        }
        (
            ta[0] + rho2 * (ta[0] + ta[1]),
            tb[0] + rho2 * (tb[0] + tb[1]),
        )
    };
    let mut acc = [F256Unreduced::ZERO; 8];
    for u in 0..eq_lo.len() / 2 {
        let o = 4 * u;
        let (a0, b0) = out_at(out_base + o);
        let (a1, b1) = out_at(out_base + o + 1);
        let (a2, b2) = out_at(out_base + o + 2);
        let (a3, b3) = out_at(out_base + o + 3);
        a_out[o] = a0;
        a_out[o + 1] = a1;
        a_out[o + 2] = a2;
        a_out[o + 3] = a3;
        b_out[o] = b0;
        b_out[o + 1] = b1;
        b_out[o + 2] = b2;
        b_out[o + 3] = b3;
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

/// Portable reference for one lookahead round-two chunk. Returns the eight
/// per-chunk sums `[p1_even, pinf_even, p1_odd, pinf_odd, W0', W3', W4', W5']`
/// (all on the odd lane's weight `r₁·eq₃`, see the driver), each reduced.
///
/// Also the oracle for the AVX-512 kernel.
#[cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ),
    allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn round2_lookahead_chunk_scalar<const WRITE: bool>(
    a_packed: &[u8],
    b_packed: &[u8],
    table: &UniSkipFoldTable,
    a_chunk: &mut [F128],
    b_chunk: &mut [F128],
    eq_lo: &[F128],
    row_base: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [F128; 8] {
    let n_chunks = table.n_chunks;
    let lo_size = eq_lo.len();
    debug_assert!(!WRITE || a_chunk.len() == 2 * lo_size);
    debug_assert!(!WRITE || b_chunk.len() == 2 * lo_size);
    debug_assert!(lo_size.is_multiple_of(2));
    let mut acc = [F256Unreduced::ZERO; 8];

    let fold_pair = |x_lo: usize, a_chunk: &mut [F128], b_chunk: &mut [F128]| {
        let x0l = 2 * x_lo;
        let x1l = x0l + 1;
        if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
            if WRITE {
                a_chunk[x0l] = F128::ZERO;
                a_chunk[x1l] = F128::ZERO;
                b_chunk[x0l] = F128::ZERO;
                b_chunk[x1l] = F128::ZERO;
            }
            return None;
        }
        let x0g = row_base + x0l;
        let x1g = x0g + 1;
        let a0 = table.fold_one_row(&a_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
        let a1 = table.fold_one_row(&a_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
        let b0 = table.fold_one_row(&b_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
        let b1 = table.fold_one_row(&b_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
        if WRITE {
            a_chunk[x0l] = a0;
            a_chunk[x1l] = a1;
            b_chunk[x0l] = b0;
            b_chunk[x1l] = b1;
        }
        Some((a0, a1, b0, b1))
    };

    for u in 0..lo_size / 2 {
        let even = fold_pair(2 * u, a_chunk, b_chunk);
        let odd = fold_pair(2 * u + 1, a_chunk, b_chunk);
        if even.is_none() && odd.is_none() {
            continue;
        }
        let z = (F128::ZERO, F128::ZERO, F128::ZERO, F128::ZERO);
        let (a0, a1, b0, b1) = even.unwrap_or(z);
        let (a2, a3, b2, b3) = odd.unwrap_or(z);
        // One weight per group: the odd lane's. See the driver's doc.
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

// ---------------------------------------------------------------------------
// Subsequent multilinear rounds (3..(m−k_skip+1)): fold + next message.
// ---------------------------------------------------------------------------

/// In-place fold of a single multilinear polynomial table at `challenge`.
/// Pairs `(a[2x], a[2x+1])` collapse to `a[x] = a[2x] + challenge · (a[2x+1] + a[2x])`.
/// After the call, `a.len()` is halved.
pub fn fold_in_place_single(a: &mut Vec<F128>, challenge: F128) {
    let n = a.len();
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    for x in 0..half {
        let a0 = a[2 * x];
        let a1 = a[2 * x + 1];
        a[x] = a0 + challenge * (a1 + a0);
    }
    a.truncate(half);
}

/// Fold a packed boolean witness at the univariate-skip challenge `z`,
/// producing the multilinear table `f_mlv` of length `2^(m − k_skip)` over
/// F_{2^128}. Uses the precomputed [`UniSkipFoldTable`] so each row costs
/// `n_chunks` lookups + XORs.
///
/// Useful for the prover's `ĉ` track: extract_c handles `c` outside the
/// multilinear sumcheck, but the prover still needs `ĉ` at the final point
/// for the claim. This is the per-row fold (Σ_s L_s(z) · c(s, x_rest)) in
/// packed form.
pub fn fold_packed_witness_at_z(
    witness_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
) -> Vec<F128> {
    use rayon::prelude::*;
    assert_eq!(witness_packed.len(), (1usize << m) / 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let mut out = vec![F128::ZERO; n_out];
    out.par_iter_mut().enumerate().for_each(|(x_rest, slot)| {
        *slot = table.fold_one_row(&witness_packed[x_rest * n_chunks..(x_rest + 1) * n_chunks]);
    });
    out
}

/// In-place fold of a pair `(a, b)` of multilinear polynomial tables at
/// `challenge`. Binds the lowest bit of the index: pairs `(a[2x], a[2x+1])`
/// collapse to `a[x] = a[2x] + challenge · (a[2x+1] + a[2x])` (and same for b).
/// After the call, `a.len()` and `b.len()` are halved.
///
/// Used at the tail of the multilinear-round sequence where the polynomial is
/// small enough that parallel/fusion overhead outweighs benefit.
pub fn fold_in_place_pair(a: &mut Vec<F128>, b: &mut Vec<F128>, challenge: F128) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    for x in 0..half {
        let a0 = a[2 * x];
        let a1 = a[2 * x + 1];
        let b0 = b[2 * x];
        let b1 = b[2 * x + 1];
        a[x] = a0 + challenge * (a1 + a0);
        b[x] = b0 + challenge * (b1 + b0);
    }
    a.truncate(half);
    b.truncate(half);
}

/// Fused: bind one variable at `r_fold` AND compute the *next* round's prover
/// message. Returns the new (folded) `a, b` vectors (half the input size) and
/// `(r_next[0] · G(1), G(∞))` for the next round.
///
/// Parallelized via rayon: each worker reads one disjoint 4·lo_size chunk of
/// the input and writes the corresponding 2·lo_size chunk of the output.
///
/// Requires `a.len() = b.len() ≥ 8` so the post-fold polynomial has at least
/// one bit of x_lo (lo_size ≥ 2). Smaller polynomials should use the
/// unfused `fold_in_place_pair + round_pair_naive` pair.
pub fn fold_and_compute_round_pair_optimized(
    a: &[F128],
    b: &[F128],
    r_fold: F128,
    r_next: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    let half = a.len() / 2;
    // Uninit alloc — `_into` writes every slot of a_new/b_new.
    let mut a_new = crate::alloc_uninit_f128_vec(half);
    let mut b_new = crate::alloc_uninit_f128_vec(half);
    let (m1, mi) = fold_and_compute_round_pair_into(a, b, &mut a_new, &mut b_new, r_fold, r_next);
    (a_new, b_new, m1, mi)
}

/// Per-chunk output floor for the generic tail rounds, in log2 F128 elements.
///
/// The tail's rayon fan-out is `hi_size = 2^n_hi` chunks; with the fixed
/// `SplitEqGhash::MAX_N_HI = 7` split that is 128 chunks at *every* round, so
/// once the table is small each chunk carries a handful of outputs and the
/// fork/join tree costs more than the arithmetic. Measured on a 16-thread
/// Zen 5 the generic tail rounds sat at a flat ~1.4 ms each from log_n 20 all
/// the way down to log_n 10 even though the work halves every round — that
/// floor is pure dispatch. Capping the chunk count so each chunk keeps at
/// least `2^ZC_TAIL_CHUNK_MIN_OUT_LOG` outputs collapses it.
///
/// Regrouping across chunk boundaries is exact, so the wire bytes do not
/// move: `eq` splits as an exact tensor product (`eq(x) = eq_hi[x_hi] ·
/// eq_lo[x_lo]`), F128 multiplication is associative, and the deferred
/// reduction is F2-linear, so `Σ_hi eq_hi · reduce(Σ_lo eq_lo ⊗ g)` is the
/// same field element for every choice of `n_hi`.
const ZC_TAIL_CHUNK_MIN_OUT_LOG: usize = 9;

/// `FLOCK_NO_ZC_TAIL_FANOUT=1` restores the fixed `MAX_N_HI` fan-out for the
/// generic tail rounds (same-binary A/B control and emergency fallback);
/// `FLOCK_ZC_TAIL_CHUNK_LOG=<n>` overrides the floor for threshold sweeps.
/// The ranked worker's cleared environment never sets either, so the shipped
/// behavior is the tuned default.
fn zc_tail_chunk_min_out_log() -> usize {
    static V: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        // Default OFF on this lineage: the calling-thread/fan-out route measured
        // -0.66% on the ranked runner (rayon regions are cheap there);
        // FLOCK_ZC_TAIL_FANOUT=1 opts back in for local A/B.
        if std::env::var_os("FLOCK_NO_ZC_TAIL_FANOUT").is_some()
            || std::env::var_os("FLOCK_ZC_TAIL_FANOUT").is_none()
        {
            return 0;
        }
        std::env::var("FLOCK_ZC_TAIL_CHUNK_LOG")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(ZC_TAIL_CHUNK_MIN_OUT_LOG)
    });
    *V
}

/// Size-adaptive hi-split for the generic tail rounds. `half` is the folded
/// output length. `n_hi = 0` (a single chunk) is deliberately reachable — the
/// caller then skips the rayon region entirely.
fn tail_n_hi_for(half: usize) -> usize {
    let floor = zc_tail_chunk_min_out_log();
    if floor == 0 {
        return SplitEqGhash::MAX_N_HI;
    }
    let log_half = half.trailing_zeros() as usize;
    SplitEqGhash::MAX_N_HI.min(log_half.saturating_sub(floor))
}

/// Output-length ceiling (in F128 elements) below which a generic tail round
/// runs entirely on the calling thread — one chunk, the same AVX-512/portable
/// kernel, and no rayon region at all. Entering a rayon region from the prove
/// thread costs a job injection plus a latch wait on every worker; for the
/// smallest tail rounds that is the whole round.
const ZC_SERIAL_TAIL_MAX_OUT_LOG: usize = 17;

/// `FLOCK_NO_ZC_SERIAL_TAIL=1` keeps every generic tail round on rayon;
/// `FLOCK_ZC_SERIAL_TAIL_LOG=<n>` moves the crossover (`n = 0` also disables).
/// The ranked worker's cleared environment never sets either.
fn zc_serial_tail_max_out() -> usize {
    static V: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        // Default OFF (see zc_tail_chunk_min_out_log); FLOCK_ZC_SERIAL_TAIL=1 opts in.
        if std::env::var_os("FLOCK_NO_ZC_SERIAL_TAIL").is_some()
            || std::env::var_os("FLOCK_ZC_SERIAL_TAIL").is_none()
        {
            return 0;
        }
        let log = std::env::var("FLOCK_ZC_SERIAL_TAIL_LOG")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(ZC_SERIAL_TAIL_MAX_OUT_LOG)
            .min(usize::BITS as usize - 1);
        if log == 0 { 0 } else { 1usize << log }
    });
    *V
}

/// Buffer-reusing variant of [`fold_and_compute_round_pair_optimized`]: writes
/// the folded `a`/`b` into the caller-provided `a_out`/`b_out` (each length
/// `a.len() / 2`) instead of allocating. Returns `(r_next[0] · G(1), G(∞))`.
///
/// Lets the multilinear-sumcheck tail ping-pong between two persistent scratch
/// buffers, so the ~22 decreasing-size buffers are allocated/freed once rather
/// than per round. The per-round `munmap` of the old buffer (64 MB at m=29)
/// runs single-threaded and otherwise caps the tail's parallel speedup.
pub fn fold_and_compute_round_pair_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    r_next: &[F128],
) -> (F128, F128) {
    // Chunk-count policy: cap the fan-out so each rayon chunk keeps a useful
    // amount of work (see `tail_n_hi_for`), and drop to a single chunk on the
    // calling thread once the round is smaller than the region entry cost
    // (see `zc_serial_tail_max_out`). Both only change how the identical
    // per-element products are grouped for the deferred reduction, which is
    // exact — the wire bytes are unchanged.
    let half = a.len() / 2;
    let n_hi = if half <= zc_serial_tail_max_out() {
        0
    } else {
        tail_n_hi_for(half)
    };
    fold_and_compute_round_pair_into_with_n_hi(a, b, a_out, b_out, r_fold, r_next, n_hi)
}

/// Split-explicit implementation behind [`fold_and_compute_round_pair_into`]'s
/// policy wrapper. `n_hi = 0` means one chunk, executed on the calling thread
/// with no rayon region. Every `n_hi` produces bit-identical output (see
/// [`ZC_TAIL_CHUNK_MIN_OUT_LOG`]); the regrouping-identity test pins that.
pub(crate) fn fold_and_compute_round_pair_into_with_n_hi(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    r_next: &[F128],
    n_hi: usize,
) -> (F128, F128) {
    use rayon::prelude::*;

    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 8);
    let half = n / 2;
    assert_eq!(a_out.len(), half);
    assert_eq!(b_out.len(), half);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 1);

    let eq = SplitEqGhash::with_n_hi(&r_next[1..], n_hi);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 2, "fold_and_compute requires lo_size ≥ 2");
    // Total non-bound multilinear vars is log_n - 1; eq covers log_n - 2 of those.
    assert_eq!(lo_size * hi_size * 2, half);

    let chunk_in = 4 * lo_size; // read chunk per worker
    let chunk_out = 2 * lo_size; // write chunk per worker
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    // Non-temporal tail path gate: only when the folded output per array is
    // ≥ 32 MB (64 MB for the a/b pair) — beyond L2+SLC reach, so the next
    // round's reads are DRAM-cold regardless and the regular stores'
    // write-allocate is one pure hidden DRAM read per output line. Below the
    // threshold the outputs may still be cache-resident when the next round
    // reads them, and regular stores win (NT on cache-reachable data inverts).
    // `FLOCK_NO_NT_TAIL` is a local-diagnostics kill switch for A/B runs;
    // the ranked worker's cleared environment never sets it.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let use_nt_stores = half >= (1usize << 21) && std::env::var_os("FLOCK_NO_NT_TAIL").is_none();
    // q-form (all-NEON) kernel gate. The scalar-struct leaf keeps F128 halves
    // in GPRs and pays a fmov per PMULL operand; the q-form leaf keeps every
    // value in NEON registers (bit-identical output). Unlike the v1 leaf it
    // also covers the sub-NT mid rounds (regular stores). `FLOCK_NO_TAIL_QNEON`
    // is a local-diagnostics kill switch for one-process A/B runs (reverts to
    // v1 leaf + generic reload path); `FLOCK_TAIL_LDNP` opts the DRAM rounds
    // into `ldnp` no-allocate input streaming. The ranked worker's cleared
    // environment never sets either.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let use_qneon = std::env::var_os("FLOCK_NO_TAIL_QNEON").is_none();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let use_ldnp = std::env::var_os("FLOCK_TAIL_LDNP").is_some();

    // Per-chunk body, shared by the rayon route and the calling-thread route.
    let chunk_body = |x_hi: usize, a_out: &mut [F128], b_out: &mut [F128]| {
        let a_in = &a[x_hi * chunk_in..(x_hi + 1) * chunk_in];
        let b_in = &b[x_hi * chunk_in..(x_hi + 1) * chunk_in];

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        // SAFETY: chunk geometry supplies two inputs per output and two
        // outputs per eq_lo value; features are guaranteed by the cfg.
        let (p1, pinf) =
            unsafe { fold_and_message_x86_avx512(a_in, b_in, a_out, b_out, r_fold, eq_lo) };

        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )))]
        #[allow(unused_labels)] // Used by the aarch64-only early exits below.
        let (p1, pinf) = 'msg: {
            // Large rounds: fused fold + register-sourced message with
            // `stnp` output stores — no write-allocate, no reload. Value-
            // and byte-identical to the generic path below (same vec2
            // fold, same reduced message products, same unreduced
            // accumulation schedule).
            #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
            {
                // SAFETY: aes is cfg-guaranteed; the chunk geometry
                // supplies 4·lo_size inputs and 2·lo_size outputs per
                // worker and lo_size eq weights, all in-bounds slices.
                if use_qneon {
                    break 'msg unsafe {
                        match (use_nt_stores, use_ldnp) {
                            (true, true) => kernels::aarch64::tail_fold_chunk_q::<true, true>(
                                a_in.as_ptr(),
                                b_in.as_ptr(),
                                a_out.as_mut_ptr(),
                                b_out.as_mut_ptr(),
                                eq_lo.as_ptr(),
                                lo_size,
                                r_fold,
                            ),
                            (true, false) => kernels::aarch64::tail_fold_chunk_q::<true, false>(
                                a_in.as_ptr(),
                                b_in.as_ptr(),
                                a_out.as_mut_ptr(),
                                b_out.as_mut_ptr(),
                                eq_lo.as_ptr(),
                                lo_size,
                                r_fold,
                            ),
                            (false, _) => kernels::aarch64::tail_fold_chunk_q::<false, false>(
                                a_in.as_ptr(),
                                b_in.as_ptr(),
                                a_out.as_mut_ptr(),
                                b_out.as_mut_ptr(),
                                eq_lo.as_ptr(),
                                lo_size,
                                r_fold,
                            ),
                        }
                    };
                }
                if use_nt_stores {
                    break 'msg unsafe {
                        kernels::aarch64::tail_fold_chunk_nt_neon(
                            a_in.as_ptr(),
                            b_in.as_ptr(),
                            a_out.as_mut_ptr(),
                            b_out.as_mut_ptr(),
                            eq_lo.as_ptr(),
                            lo_size,
                            r_fold,
                        )
                    };
                }
            }
            // Fold a_in→a_out and b_in→b_out at r_fold. The field layer
            // selects the architecture kernel; this loop only consumes
            // the resulting values to build the message.
            crate::field::f128_slice::fold_pairs(a_in, 0, a_out, r_fold);
            crate::field::f128_slice::fold_pairs(b_in, 0, b_out, r_fold);

            let mut p1_acc = F256Unreduced::ZERO;
            let mut pinf_acc = F256Unreduced::ZERO;
            // x86: 4-wide deferred-reduction accumulators for the unrolled loop;
            // the 2-wide tail still uses the scalar `*_acc` above, folded in
            // before the final reduce.
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            // SAFETY: vpclmulqdq+avx512f guaranteed by the cfg gate.
            let (mut p1_wide, mut pinf_wide) =
                unsafe { (WideGhashX4::zero(), WideGhashX4::zero()) };

            // Unroll 4 x_lo's per iteration when lo_size % 4 == 0 (the common
            // case for the fused path; falls back to 2-wide for lo_size==2 at
            // the smallest fused round). 16 independent r_fold muls and 8
            // independent msg muls in flight gives the M4 OoO engine and
            // 2/cy PMULL throughput maximum ILP.
            assert!(lo_size & 1 == 0, "lo_size must be even");
            let mut x_lo = 0;
            if lo_size.is_multiple_of(4) {
                while x_lo + 4 <= lo_size {
                    let x_lo_a = x_lo;
                    // Read the just-folded pairs: (a0,a1) = (a_out[2·x_lo], a_out[2·x_lo+1]).
                    let o = 2 * x_lo;
                    let a0_a = a_out[o];
                    let a1_a = a_out[o + 1];
                    let b0_a = b_out[o];
                    let b1_a = b_out[o + 1];
                    let a0_b = a_out[o + 2];
                    let a1_b = a_out[o + 3];
                    let b0_b = b_out[o + 2];
                    let b1_b = b_out[o + 3];
                    let a0_c = a_out[o + 4];
                    let a1_c = a_out[o + 5];
                    let b0_c = b_out[o + 4];
                    let b1_c = b_out[o + 5];
                    let a0_d = a_out[o + 6];
                    let a1_d = a_out[o + 7];
                    let b0_d = b_out[o + 6];
                    let b1_d = b_out[o + 7];

                    // 8 reduced msg muls (g1 = a1·b1, g_inf = (a0+a1)(b0+b1)).
                    let g1_a = a1_a * b1_a;
                    let g1_b = a1_b * b1_b;
                    let g1_c = a1_c * b1_c;
                    let g1_d = a1_d * b1_d;
                    let g_inf_a = (a0_a + a1_a) * (b0_a + b1_a);
                    let g_inf_b = (a0_b + a1_b) * (b0_b + b1_b);
                    let g_inf_c = (a0_c + a1_c) * (b0_c + b1_c);
                    let g_inf_d = (a0_d + a1_d) * (b0_d + b1_d);
                    // Deferred-reduction accumulate: on x86 widen all 8 products
                    // 4 lanes at a time (eq_lo[x_lo_a..x_lo_a+4] is contiguous),
                    // reduced once after the loop; else scalar mul_unreduced.
                    #[cfg(all(
                        target_arch = "x86_64",
                        target_feature = "avx512f",
                        target_feature = "vpclmulqdq"
                    ))]
                    // SAFETY: vpclmulqdq+avx512f guaranteed by the cfg gate; the
                    // four eq values eq_lo[x_lo_a..x_lo_a+4] are in bounds (the
                    // 4-wide loop runs only while x_lo + 4 <= lo_size == eq_lo.len()).
                    unsafe {
                        let eq4 = f128x4_loadu(eq_lo[x_lo_a..].as_ptr());
                        p1_wide.mul_acc(eq4, f128x4_set(g1_a, g1_b, g1_c, g1_d));
                        pinf_wide.mul_acc(eq4, f128x4_set(g_inf_a, g_inf_b, g_inf_c, g_inf_d));
                    }
                    #[cfg(not(all(
                        target_arch = "x86_64",
                        target_feature = "avx512f",
                        target_feature = "vpclmulqdq"
                    )))]
                    {
                        let eq_l_a = eq_lo[x_lo_a];
                        let eq_l_b = eq_lo[x_lo_a + 1];
                        let eq_l_c = eq_lo[x_lo_a + 2];
                        let eq_l_d = eq_lo[x_lo_a + 3];
                        p1_acc ^= eq_l_a.mul_unreduced(g1_a);
                        p1_acc ^= eq_l_b.mul_unreduced(g1_b);
                        p1_acc ^= eq_l_c.mul_unreduced(g1_c);
                        p1_acc ^= eq_l_d.mul_unreduced(g1_d);
                        pinf_acc ^= eq_l_a.mul_unreduced(g_inf_a);
                        pinf_acc ^= eq_l_b.mul_unreduced(g_inf_b);
                        pinf_acc ^= eq_l_c.mul_unreduced(g_inf_c);
                        pinf_acc ^= eq_l_d.mul_unreduced(g_inf_d);
                    }

                    x_lo += 4;
                }
            }
            // 2-wide tail (handles lo_size == 2 case and any remainder when
            // 4-wide loop is skipped or doesn't cover everything).
            while x_lo + 2 <= lo_size {
                let x_lo_a = x_lo;
                let x_lo_b = x_lo + 1;
                let o = 2 * x_lo;
                let a0_a = a_out[o];
                let a1_a = a_out[o + 1];
                let b0_a = b_out[o];
                let b1_a = b_out[o + 1];
                let a0_b = a_out[o + 2];
                let a1_b = a_out[o + 3];
                let b0_b = b_out[o + 2];
                let b1_b = b_out[o + 3];

                let eq_l_a = eq_lo[x_lo_a];
                let eq_l_b = eq_lo[x_lo_b];
                let g1_a = a1_a * b1_a;
                let g1_b = a1_b * b1_b;
                let g_inf_a = (a0_a + a1_a) * (b0_a + b1_a);
                let g_inf_b = (a0_b + a1_b) * (b0_b + b1_b);
                p1_acc ^= eq_l_a.mul_unreduced(g1_a);
                p1_acc ^= eq_l_b.mul_unreduced(g1_b);
                pinf_acc ^= eq_l_a.mul_unreduced(g_inf_a);
                pinf_acc ^= eq_l_b.mul_unreduced(g_inf_b);

                x_lo += 2;
            }

            // Merge the 4-wide deferred accumulators with the scalar tail, then
            // reduce once (reduction is F2-linear, so this equals the scalar
            // Σ mul_unreduced then reduce).
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            // SAFETY: vpclmulqdq+avx512f+sse4.1 guaranteed by the cfg gate.
            unsafe {
                p1_acc ^= p1_wide.fold();
                pinf_acc ^= pinf_wide.fold();
            }
            let p1 = p1_acc.reduce();
            let pinf = pinf_acc.reduce();
            (p1, pinf)
        };
        let eq_h = eq_hi[x_hi];
        (eq_h * p1, eq_h * pinf)
    };

    let (sum1, sum_inf) = if hi_size == 1 {
        // One chunk: run it right here. Entering a rayon region costs a job
        // injection plus a latch round-trip on every worker, which for the
        // small tail rounds is more than the whole round's arithmetic.
        chunk_body(0, a_out, b_out)
    } else {
        a_out
            .par_chunks_mut(chunk_out)
            .zip(b_out.par_chunks_mut(chunk_out))
            .enumerate()
            .map(|(x_hi, (a_out, b_out))| {
                crate::rstrace::evd("fr:b", x_hi as u32);
                let r = chunk_body(x_hi, a_out, b_out);
                crate::rstrace::evd("fr:e", 0);
                r
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
            )
    };
    crate::rstrace::ev("fr:red", hi_size as u32);

    (r_next[0] * sum1, sum_inf)
}

/// Bind ρ₁ **and** ρ₂ in one pass over the round-two tables and emit the
/// round-four message — replacing tail iterations `i = 0` and `i = 1`.
///
/// `a`/`b` have length `n = 2^k`; `a_out`/`b_out` have length `n/4`. Per
/// composed output `x`, the four inputs `v[4x..4x+4]` fold exactly as two
/// sequential pair folds would:
///
/// ```text
/// t0 = v[4x]   + ρ₁(v[4x]   + v[4x+1])
/// t1 = v[4x+2] + ρ₁(v[4x+2] + v[4x+3])
/// out[x] = t0 + ρ₂(t0 + t1)
/// ```
///
/// so the tables are bit-identical to `fold(ρ₁)` then `fold(ρ₂)`, and the
/// message `(r_next4[0]·G₄(1), G₄(∞))` over `r_next4[1..]` is the same sum
/// the incumbent's second tail iteration computes. Multiply count is the same
/// as two sequential folds; the win is one deleted 1.5×-size DRAM pass.
///
/// Requires `n ≥ 16` and `r_next4.len() = log₂(n) − 2`.
pub fn fold2_plain_and_round4_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho1: F128,
    rho2: F128,
    r_next4: &[F128],
) -> (F128, F128) {
    use rayon::prelude::*;
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 16);
    let quarter = n / 4;
    assert_eq!(a_out.len(), quarter);
    assert_eq!(b_out.len(), quarter);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next4.len(), log_n - 2);

    let n_vars = r_next4.len() - 1;
    let eq = SplitEqGhash::with_n_hi(&r_next4[1..], tail_split_n_hi(n_vars));
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 2, "composed fold requires lo_size ≥ 2");
    assert_eq!(lo_size * hi_size * 2, quarter);

    let chunk_in = 8 * lo_size; // four inputs per composed output
    let chunk_out = 2 * lo_size;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    let (sum1, sum_inf) = a_out
        .par_chunks_mut(chunk_out)
        .zip(b_out.par_chunks_mut(chunk_out))
        .enumerate()
        .map(|(x_hi, (a_out, b_out))| {
            crate::rstrace::evd("p4:b", x_hi as u32);
            let a_in = &a[x_hi * chunk_in..(x_hi + 1) * chunk_in];
            let b_in = &b[x_hi * chunk_in..(x_hi + 1) * chunk_in];

            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            // SAFETY: chunk geometry supplies four inputs per output and two
            // outputs per eq_lo value; features are guaranteed by the cfg.
            let (p1, pinf) = unsafe {
                kernels::x86_64::fold2_and_message_x86_avx512(
                    a_in, b_in, a_out, b_out, rho1, rho2, eq_lo,
                )
            };
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            let (p1, pinf) = fold2_and_message_scalar(a_in, b_in, a_out, b_out, rho1, rho2, eq_lo);

            let eq_h = eq_hi[x_hi];
            crate::rstrace::evd("p4:e", 0);
            (eq_h * p1, eq_h * pinf)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
        );
    crate::rstrace::ev("p4:red", 0);

    (r_next4[0] * sum1, sum_inf)
}

/// Composed double fold (ρ_a then ρ_b) **plus** the round message **plus**
/// the deferred quadratic for the round after it — the cascade step.
///
/// Tables and message are bit-identical to [`fold2_plain_and_round4_into`];
/// additionally the six aggregates over next-round groups `y` (composed
/// outputs `out[4y..4y+4]`) are accumulated with the same parity trick as the
/// round-two sweep, against `r = r_next[1]` (the eq weight of the variable
/// the emitted message binds): `eq_R(2y+1) = r·eq_{R+1}(y)`. Requires
/// `r ≠ 0`; the caller falls back to the plain composed pass otherwise.
///
/// Returns `(r_next[0]·G_R(1), G_R(∞), lookahead for round R+1)`.
pub fn fold2_plain_and_round_pair_lookahead_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    r_next: &[F128],
) -> (F128, F128, Round3Lookahead) {
    fold2_plain_and_round_pair_lookahead_into_lm(
        a, b, a_out, b_out, rho_a, rho_b, r_next, false, false,
    )
}

/// [`fold2_plain_and_round_pair_lookahead_into`] with explicit tile layouts:
/// `in_lm` reads `a`/`b` as lane-major sixteen-element tiles and `out_lm`
/// writes `a_out`/`b_out` that way (ZMM `k` of a tile = element `k` of each of
/// its four groups; see [`zc_lane_major_handoff_on`]). Values, message and
/// lookahead are identical for every layout; only the element order inside
/// each tile of the buffers differs. Either flag requires eight-group chunks
/// and the AVX-512 kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_plain_and_round_pair_lookahead_into_lm(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    r_next: &[F128],
    in_lm: bool,
    out_lm: bool,
) -> (F128, F128, Round3Lookahead) {
    let (m1, mi, la, _) = fold2_plain_and_round_pair_lookahead_into_lm_casc(
        a, b, a_out, b_out, rho_a, rho_b, r_next, in_lm, out_lm, false, false,
    );
    (m1, mi, la)
}

/// Whether the plain two-variable level can run the deeper cascade (`casc`)
/// for an input of `n` elements: the AVX-512 deferred-fold route with eight-
/// group chunks.
pub(crate) fn fold2_plain_casc_route_ok(n: usize) -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    {
        let lo_size = plain_cascade_lo_size(n);
        kernels::x86_64::zc_c3_enabled()
            && kernels::x86_64::zc_wtab_enabled()
            && kernels::x86_64::fold2_plain_defer_wtab_route_ok(
                kernels::x86_64::zc_defer_split_enabled(),
                lo_size,
                true,
            )
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    {
        let _ = n;
        false
    }
}

/// [`fold2_plain_and_round_pair_lookahead_into_lm`] plus, with `casc`, the
/// deeper tail cascade: the tables are stored in octet layout (for
/// [`fold3_oct_plain_and_round_pair_lookahead_into`]; `out_lm` ignored) and
/// the round two ahead of this level's message is returned as a biquadratic in
/// `(ρ_c, ρ_d)` — this level's message variable and the next one. Needs
/// [`fold2_plain_casc_route_ok`] and `r_next[1], r_next[2] ≠ 0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_plain_and_round_pair_lookahead_into_lm_casc(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    r_next: &[F128],
    in_lm: bool,
    out_lm: bool,
    casc: bool,
    team: bool,
) -> (F128, F128, Round3Lookahead, Option<Round4Lookahead>) {
    use rayon::prelude::*;
    assert!(!casc || fold2_plain_casc_route_ok(a.len()), "casc route unavailable");
    let out_lm = out_lm && !casc;
    if casc {
        assert_ne!(r_next[2], F128::ZERO, "deeper cascade needs a non-zero second weight");
    }
    // Deeper cascade: both lanes of an eight-group ride the odd lane's weight;
    // κ restores the even lanes' own weight in the message sums.
    let kappa_c = if casc { (F128::ONE + r_next[2]) * r_next[2].inv() } else { F128::ZERO };
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 16);
    let quarter = n / 4;
    assert_eq!(a_out.len(), quarter);
    assert_eq!(b_out.len(), quarter);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 2);
    let r = r_next[1];
    assert_ne!(
        r,
        F128::ZERO,
        "cascade lookahead requires a non-zero parity weight"
    );

    let n_vars = r_next.len() - 1;
    let eq = SplitEqGhash::with_n_hi(&r_next[1..], tail_split_n_hi(n_vars));
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 2, "composed lookahead requires lo_size ≥ 2");
    assert_eq!(lo_size * hi_size * 2, quarter);
    assert!(
        !(in_lm || out_lm) || (lo_size.is_multiple_of(8) && zc_lane_major_handoff_on()),
        "lane-major tiles need eight-group AVX-512 chunks"
    );
    let (kappa, r_inv) = lookahead_inv_factors(r);

    let chunk_in = 8 * lo_size;
    let chunk_out = 2 * lo_size;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;
    // Hoist the `(w, w*x^64)` pairs once for the whole plain cascade, just as
    // the packed composed-fold paths do. The AVX-512 kernel otherwise derives
    // the companion on every high chunk's accumulator dependency chain.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_vec = if kernels::x86_64::zc_wtab_enabled() && lo_size.is_multiple_of(8) {
        Some(build_w_pair_table(eq_lo))
    } else {
        None
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_arg = wtab_vec.as_deref();
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let wtab_casc = casc.then(|| build_w_pair_table_casc(eq_lo));

    // Non-temporal publish of the fold outputs, for the levels whose outputs
    // are too large to still be cache-resident when the next level reads
    // them. The message terms come from registers, so the stores are
    // write-once.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let nt_out = kernels::x86_64::zc_tail_nt_enabled()
        && quarter >= (1usize << kernels::x86_64::ZC_TAIL_NT_LOG)
        && (a_out.as_ptr() as usize) % 16 == 0
        && (b_out.as_ptr() as usize) % 16 == 0;
    // Split-companion deferred fold (see `zc_defer_split_enabled`), once per pass.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let split = kernels::x86_64::zc_defer_split_enabled();
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let fast_plain = kernels::x86_64::fold2_plain_defer_wtab_route_ok(
        split,
        lo_size,
        wtab_arg.is_some(),
    );

    let leaf = |(x_hi, (a_out, b_out)): (usize, (&mut [F128], &mut [F128]))| {
            crate::rstrace::evd("zt:c0", x_hi as u32);
            let mut la7 = [F128::ZERO; 18];
            let a_in = &a[x_hi * chunk_in..(x_hi + 1) * chunk_in];
            let b_in = &b[x_hi * chunk_in..(x_hi + 1) * chunk_in];

            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            // SAFETY: chunk geometry supplies four inputs per output and two
            // outputs per eq_lo value; features are guaranteed by the cfg.
            let out = unsafe {
                if casc && let Some(wt) = wtab_arg {
                    let ap = a_in.as_ptr();
                    let bp = b_in.as_ptr();
                    let aop = a_out.as_mut_ptr();
                    let bop = b_out.as_mut_ptr();
                    let l7 = &mut la7;
                    let _ = wt;
                    let wt = wtab_casc.as_deref().unwrap();
                    match (nt_out, in_lm) {
                        (true, true) => kernels::x86_64::fold2_and_message_lookahead_defer_wtab_casc_x86::<true, true, false, true>(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt, l7, kappa_c),
                        (true, false) => kernels::x86_64::fold2_and_message_lookahead_defer_wtab_casc_x86::<true, false, false, true>(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt, l7, kappa_c),
                        (false, true) => kernels::x86_64::fold2_and_message_lookahead_defer_wtab_casc_x86::<false, true, false, true>(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt, l7, kappa_c),
                        (false, false) => kernels::x86_64::fold2_and_message_lookahead_defer_wtab_casc_x86::<false, false, false, true>(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt, l7, kappa_c),
                    }
                } else if fast_plain && let Some(wt) = wtab_arg {
                    let ap = a_in.as_ptr();
                    let bp = b_in.as_ptr();
                    let aop = a_out.as_mut_ptr();
                    let bop = b_out.as_mut_ptr();
                    match (nt_out, in_lm, out_lm) {
                        (false, true, true) => {
                            kernels::x86_64::fold2_and_message_lookahead_defer_wtab_x86::<
                                false,
                                true,
                                true,
                            >(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt)
                        }
                        (false, false, true) => {
                            kernels::x86_64::fold2_and_message_lookahead_defer_wtab_x86::<
                                false,
                                false,
                                true,
                            >(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt)
                        }
                        (false, true, false) => {
                            kernels::x86_64::fold2_and_message_lookahead_defer_wtab_x86::<
                                false,
                                true,
                                false,
                            >(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt)
                        }
                        (false, false, false) => {
                            kernels::x86_64::fold2_and_message_lookahead_defer_wtab_x86::<
                                false,
                                false,
                                false,
                            >(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt)
                        }
                        (true, true, true) => {
                            kernels::x86_64::fold2_and_message_lookahead_defer_wtab_x86::<
                                true,
                                true,
                                true,
                            >(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt)
                        }
                        (true, false, true) => {
                            kernels::x86_64::fold2_and_message_lookahead_defer_wtab_x86::<
                                true,
                                false,
                                true,
                            >(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt)
                        }
                        (true, true, false) => {
                            kernels::x86_64::fold2_and_message_lookahead_defer_wtab_x86::<
                                true,
                                true,
                                false,
                            >(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt)
                        }
                        (true, false, false) => {
                            kernels::x86_64::fold2_and_message_lookahead_defer_wtab_x86::<
                                true,
                                false,
                                false,
                            >(ap, bp, aop, bop, rho_a, rho_b, lo_size, wt)
                        }
                    }
                } else {
                    kernels::x86_64::fold2_and_message_lookahead_x86_avx512(
                        a_in, b_in, a_out, b_out, rho_a, rho_b, eq_lo, wtab_arg, nt_out, in_lm,
                        out_lm, split,
                    )
                }
            };
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            let out =
                fold2_and_message_lookahead_scalar(a_in, b_in, a_out, b_out, rho_a, rho_b, eq_lo);

            let eq_h = eq_hi[x_hi];
            crate::rstrace::evd("zt:c1", x_hi as u32);
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
    type Sums = (F128, F128, [F128; 6], [F128; 18]);
    let id = || -> Sums { (F128::ZERO, F128::ZERO, [F128::ZERO; 6], [F128::ZERO; 18]) };
    let op = |(s1, sinf, sa, sl): Sums, (c1, cinf, ca, cl): Sums| -> Sums {
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
    let (sum1_even, sum_inf_even, agg, la7_acc) =
        if flat_tail(n) {
            // Lane ru flat claim route (`FLOCK_NO_RU_FLAT_PAR`): same leaves,
            // per-chunk results summed (XOR) on the caller.
            flat_chunks(a_out, b_out, chunk_out, leaf, id, op)
        } else if team && crate::flat_par::team_ready() {
            // Lane gl1 (`FLOCK_NO_GL1_TEAM`): the hot pool's spinning team.
            team_chunks_aff(a, chunk_in, a_out, b_out, chunk_out, leaf, id, op)
        } else {
            a_out
                .par_chunks_mut(chunk_out)
                .zip(b_out.par_chunks_mut(chunk_out))
                .enumerate()
                .map(leaf)
                .reduce(id, op)
        };

    let sum1 = kappa * sum1_even + agg[0];
    let sum_inf = kappa * sum_inf_even + agg[1];
    let la = lookahead_from_odd_weighted(&agg, r_inv);
    let la7 = casc.then(|| {
        assert_ne!(r_next[2], F128::ZERO, "deeper cascade needs a non-zero second weight");
        Round4Lookahead::from_flat(&la7_acc).scaled((r_next[1] * r_next[2]).inv())
    });
    (r_next[0] * sum1, sum_inf, la, la7)
}

/// Deeper tail cascade level (`FLOCK_NO_ZC_C3`): folds three variables
/// (`rho = [ρa, ρb, ρc]`) of the rounds 3+4+5 pass's **octet-layout** tables
/// (see `fold3_from_packed_lookahead_c4_lm_wtab_b_canon_casc_x86`) into
/// eighth-size tables, and returns the next round's message plus the deferred
/// quadratic for the round after it, with exactly the conventions of
/// [`fold2_plain_and_round_pair_lookahead_into_lm`] (which equals folding
/// `X_a` first and then running that two-variable level). `out_lm` stores
/// lane-major tiles for a following lookahead level.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold3_oct_plain_and_round_pair_lookahead_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho: [F128; 3],
    r_next: &[F128],
    out_lm: bool,
) -> (F128, F128, Round3Lookahead) {
    use rayon::prelude::*;
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 32);
    let eighth = n / 8;
    assert_eq!(a_out.len(), eighth);
    assert_eq!(b_out.len(), eighth);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 3);
    let r = r_next[1];
    assert_ne!(r, F128::ZERO, "cascade lookahead requires a non-zero parity weight");

    let n_vars = r_next.len() - 1;
    let eq = SplitEqGhash::with_n_hi(&r_next[1..], tail_split_n_hi(n_vars));
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 2);
    assert_eq!(lo_size * hi_size * 2, eighth);
    assert!(
        !out_lm || (lo_size.is_multiple_of(8) && zc_lane_major_handoff_on()),
        "lane-major tiles need eight-group chunks"
    );
    let (kappa, r_inv) = lookahead_inv_factors(r);
    let chunk_in = 16 * lo_size;
    let chunk_out = 2 * lo_size;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let fast = (lo_size >= 16 && lo_size.is_multiple_of(16)).then(|| {
        let nt = kernels::x86_64::zc_tail_nt_enabled()
            && eighth >= (1usize << kernels::x86_64::ZC_TAIL_NT_LOG)
            && (a_out.as_ptr() as usize) % 16 == 0
            && (b_out.as_ptr() as usize) % 16 == 0;
        (build_w_pair_table(eq_lo), nt)
    });

    let leaf = |(x_hi, (a_out, b_out)): (usize, (&mut [F128], &mut [F128]))| {
            crate::rstrace::evd("zt:c0", x_hi as u32);
            let a_in = &a[x_hi * chunk_in..(x_hi + 1) * chunk_in];
            let b_in = &b[x_hi * chunk_in..(x_hi + 1) * chunk_in];
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            let out = if let Some((wt, nt)) = fast.as_ref() {
                let (ap, bp) = (a_in.as_ptr(), b_in.as_ptr());
                let (aop, bop) = (a_out.as_mut_ptr(), b_out.as_mut_ptr());
                // SAFETY: the chunk holds eight inputs per output and two
                // outputs per eq_lo entry; lo_size is a multiple of 16.
                unsafe {
                    match (*nt, out_lm) {
                        (true, true) => kernels::x86_64::fold3_oct_and_message_lookahead_wtab_x86::<true, true>(ap, bp, aop, bop, rho, lo_size, wt),
                        (true, false) => kernels::x86_64::fold3_oct_and_message_lookahead_wtab_x86::<true, false>(ap, bp, aop, bop, rho, lo_size, wt),
                        (false, true) => kernels::x86_64::fold3_oct_and_message_lookahead_wtab_x86::<false, true>(ap, bp, aop, bop, rho, lo_size, wt),
                        (false, false) => kernels::x86_64::fold3_oct_and_message_lookahead_wtab_x86::<false, false>(ap, bp, aop, bop, rho, lo_size, wt),
                    }
                }
            } else {
                fold3_oct_and_message_lookahead_scalar(a_in, b_in, a_out, b_out, rho, eq_lo, out_lm)
            };
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            let out =
                fold3_oct_and_message_lookahead_scalar(a_in, b_in, a_out, b_out, rho, eq_lo, out_lm);
            let eq_h = eq_hi[x_hi];
            crate::rstrace::evd("zt:c1", x_hi as u32);
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
            )
    };
    type Sums3 = (F128, F128, [F128; 6]);
    let id = || -> Sums3 { (F128::ZERO, F128::ZERO, [F128::ZERO; 6]) };
    let op = |(s1, sinf, sa): Sums3, (c1, cinf, ca): Sums3| -> Sums3 {
        let mut a = sa;
        for (x, y) in a.iter_mut().zip(ca.iter()) {
            *x += *y;
        }
        (s1 + c1, sinf + cinf, a)
    };
    let (sum1_even, sum_inf_even, agg) =
        if flat_tail(n) {
            // Lane ru flat claim route (`FLOCK_NO_RU_FLAT_PAR`): same leaves,
            // per-chunk results summed (XOR) on the caller.
            flat_chunks(a_out, b_out, chunk_out, leaf, id, op)
        } else {
            a_out
                .par_chunks_mut(chunk_out)
                .zip(b_out.par_chunks_mut(chunk_out))
                .enumerate()
                .map(leaf)
                .reduce(id, op)
        };
    let sum1 = kappa * sum1_even + agg[0];
    let sum_inf = kappa * sum_inf_even + agg[1];
    let la = lookahead_from_odd_weighted(&agg, r_inv);
    (r_next[0] * sum1, sum_inf, la)
}

/// Portable leaf of [`fold3_oct_plain_and_round_pair_lookahead_into`]: octet
/// input element `32B + 8g + j` sits at `32B + 4j + g`.
#[cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ),
    allow(dead_code)
)]
pub(crate) fn fold3_oct_and_message_lookahead_scalar(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho: [F128; 3],
    eq_lo: &[F128],
    out_lm: bool,
) -> [F128; 8] {
    let fold8 = |v: &[F128], o: usize| -> F128 {
        let x: [F128; 8] = std::array::from_fn(|j| v[32 * (o / 4) + 4 * j + o % 4]);
        let t: [F128; 4] = std::array::from_fn(|h| x[2 * h] + rho[0] * (x[2 * h] + x[2 * h + 1]));
        let u0 = t[0] + rho[1] * (t[0] + t[1]);
        let u1 = t[2] + rho[1] * (t[2] + t[3]);
        u0 + rho[2] * (u0 + u1)
    };
    let n_out = a_out.len();
    let a_rm: Vec<F128> = (0..n_out).map(|o| fold8(a_in, o)).collect();
    let b_rm: Vec<F128> = (0..n_out).map(|o| fold8(b_in, o)).collect();
    for o in 0..n_out {
        // Lane-major tiles: tile element 4k + g holds group g's element k.
        let dst = if out_lm { (o & !15) + 4 * (o % 4) + (o % 16) / 4 } else { o };
        a_out[dst] = a_rm[o];
        b_out[dst] = b_rm[o];
    }
    let mut acc = [F256Unreduced::ZERO; 8];
    for u in 0..eq_lo.len() / 2 {
        let o = 4 * u;
        let (a0, a1, a2, a3) = (a_rm[o], a_rm[o + 1], a_rm[o + 2], a_rm[o + 3]);
        let (b0, b1, b2, b3) = (b_rm[o], b_rm[o + 1], b_rm[o + 2], b_rm[o + 3]);
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

/// Portable leaf of [`fold2_plain_and_round_pair_lookahead_into`] for one
/// worker chunk. Returns the eight per-chunk sums
/// `[p1_even, pinf_even, p1_odd, pinf_odd, W0', W3', W4', W5']` on the odd
/// lane's weight, each reduced. Also the oracle for the AVX-512 kernel.
#[cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ),
    allow(dead_code)
)]
pub(crate) fn fold2_and_message_lookahead_scalar(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
) -> [F128; 8] {
    debug_assert_eq!(a_in.len(), 4 * a_out.len());
    debug_assert_eq!(b_in.len(), 4 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());
    debug_assert!(eq_lo.len().is_multiple_of(2));
    let fold4 = |v: &[F128], i: usize| {
        let t0 = v[i] + rho_a * (v[i] + v[i + 1]);
        let t1 = v[i + 2] + rho_a * (v[i + 2] + v[i + 3]);
        t0 + rho_b * (t0 + t1)
    };
    let mut acc = [F256Unreduced::ZERO; 8];
    for u in 0..eq_lo.len() / 2 {
        // Group u: pairs 2u (even) and 2u+1 (odd) = outputs 4u..4u+4.
        let o = 4 * u;
        let i = 16 * u;
        let a0 = fold4(a_in, i);
        let a1 = fold4(a_in, i + 4);
        let a2 = fold4(a_in, i + 8);
        let a3 = fold4(a_in, i + 12);
        let b0 = fold4(b_in, i);
        let b1 = fold4(b_in, i + 4);
        let b2 = fold4(b_in, i + 8);
        let b3 = fold4(b_in, i + 12);
        a_out[o] = a0;
        a_out[o + 1] = a1;
        a_out[o + 2] = a2;
        a_out[o + 3] = a3;
        b_out[o] = b0;
        b_out[o + 1] = b1;
        b_out[o + 2] = b2;
        b_out[o + 3] = b3;
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

/// Portable leaf of [`fold2_plain_and_round4_into`] for one worker chunk.
/// Also the oracle for the AVX-512 kernel.
#[cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ),
    allow(dead_code)
)]
pub(crate) fn fold2_and_message_scalar(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho1: F128,
    rho2: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    debug_assert_eq!(a_in.len(), 4 * a_out.len());
    debug_assert_eq!(b_in.len(), 4 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());
    let fold4 = |v: &[F128], i: usize| {
        let t0 = v[i] + rho1 * (v[i] + v[i + 1]);
        let t1 = v[i + 2] + rho1 * (v[i + 2] + v[i + 3]);
        t0 + rho2 * (t0 + t1)
    };
    let mut p1_acc = F256Unreduced::ZERO;
    let mut pinf_acc = F256Unreduced::ZERO;
    for (x_lo, &eq_l) in eq_lo.iter().enumerate() {
        let i = 8 * x_lo;
        let o = 2 * x_lo;
        let a0 = fold4(a_in, i);
        let a1 = fold4(a_in, i + 4);
        let b0 = fold4(b_in, i);
        let b1 = fold4(b_in, i + 4);
        a_out[o] = a0;
        a_out[o + 1] = a1;
        b_out[o] = b0;
        b_out[o + 1] = b1;
        p1_acc ^= eq_l.mul_unreduced(a1 * b1);
        pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
    }
    (p1_acc.reduce(), pinf_acc.reduce())
}

/// Serial reference — identical I/O contract to
/// [`uni_skip_fold_and_round_pair_optimized_packed`], no rayon. Kept under
/// `#[cfg(test)]` as the cross-check oracle for the parallel version.
#[cfg(test)]
fn uni_skip_fold_and_round_pair_optimized_packed_serial(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let mut a_folded = vec![F128::ZERO; n_out];
    let mut b_folded = vec![F128::ZERO; n_out];
    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let mut sum1 = F128::ZERO;
    let mut sum_inf = F128::ZERO;
    for x_hi in 0..hi_size {
        let mut p1_acc = F256Unreduced::ZERO;
        let mut pinf_acc = F256Unreduced::ZERO;
        let k_base = x_hi << eq.n_lo;
        for x_lo in 0..lo_size {
            let k = k_base | x_lo;
            let x0 = 2 * k;
            let x1 = x0 + 1;
            let a0 = table.fold_one_row(&a_packed[x0 * n_chunks..(x0 + 1) * n_chunks]);
            let b0 = table.fold_one_row(&b_packed[x0 * n_chunks..(x0 + 1) * n_chunks]);
            let a1 = table.fold_one_row(&a_packed[x1 * n_chunks..(x1 + 1) * n_chunks]);
            let b1 = table.fold_one_row(&b_packed[x1 * n_chunks..(x1 + 1) * n_chunks]);
            a_folded[x0] = a0;
            b_folded[x0] = b0;
            a_folded[x1] = a1;
            b_folded[x1] = b1;
            let eq_l = eq.lo[x_lo];
            let g1 = a1 * b1;
            p1_acc ^= eq_l.mul_unreduced(g1);
            let g_inf = (a0 + a1) * (b0 + b1);
            pinf_acc ^= eq_l.mul_unreduced(g_inf);
        }
        let p1 = p1_acc.reduce();
        let pinf = pinf_acc.reduce();
        sum1 += eq.hi[x_hi] * p1;
        sum_inf += eq.hi[x_hi] * pinf;
    }
    (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf)
}

/// `&[bool]` convenience wrapper around
/// [`uni_skip_fold_and_round_pair_optimized_packed`]. Packs internally, builds
/// the fold table from `z`.
pub fn uni_skip_fold_and_round_pair_optimized(
    a: &[bool],
    b: &[bool],
    m: usize,
    k_skip: usize,
    z: F128,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    let a_packed = pack_bits(a);
    let b_packed = pack_bits(b);
    let table = UniSkipFoldTable::new(k_skip, z);
    uni_skip_fold_and_round_pair_optimized_packed(
        &a_packed,
        &b_packed,
        m,
        k_skip,
        &table,
        mlv_challenges,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    /// The re-verification `FLOCK_NO_ZC_R2_SETUP_TRIM` keeps is an identity:
    /// the packed round-two split's low eq tensor factors as
    /// `LV(lane)·S(g)·Etop(B)` on every odd entry the eq bake reads.
    #[test]
    fn r2_eq_bake_factorization_is_exact() {
        use crate::challenger::Challenger;
        use crate::zerocheck::univariate_skip::build_eq;
        let mut rng = crate::challenger::RandomChallenger::new(0x2EB4_0001);
        let mut checked = 0usize;
        for n_vars in [10usize, 12, 16, 20, 25] {
            let challenges = rng.sample_f128_vec(n_vars + 1);
            let eq = packed_round2_split_eq(&challenges);
            let lo_size = 1usize << eq.n_lo;
            if eq.n_lo < 5 || !lo_size.is_multiple_of(32) {
                continue;
            }
            let r_lo = &challenges[1..1 + eq.n_lo];
            let eq3 = build_eq(&r_lo[..3]);
            let lv: [F128; 4] = std::array::from_fn(|l| eq3[1 + 2 * l]);
            let s = build_eq(&r_lo[3..5]);
            let etop = build_eq(&r_lo[5..]);
            for b in 0..lo_size / 32 {
                for g in 0..4 {
                    for lane in 0..4 {
                        assert_eq!(
                            lv[lane] * s[g] * etop[b],
                            eq.lo[32 * b + 8 * g + 2 * lane + 1],
                            "n_vars={n_vars} b={b} g={g} lane={lane}"
                        );
                    }
                }
            }
            checked += 1;
        }
        assert!(checked > 0, "no bake-shaped split exercised");
    }

    /// The fast composed-fold matrix build is the scalar bit loop's matrices.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    #[test]
    fn cfold_mats_from_cols_match_scaled_bit_loop() {
        use crate::challenger::Challenger;
        let mut rng = crate::challenger::RandomChallenger::new(0xCF01_D4A7);
        for _ in 0..8 {
            let data = rng.sample_f128_vec(8 * 256);
            let c = rng.sample_f128();
            let cols: [F128; 64] =
                std::array::from_fn(|i| c * data[(i / 8) * 256 + (1usize << (i % 8))]);
            assert_eq!(
                kernels::x86_64::build_row_fold_mats_from_cols(&cols),
                kernels::x86_64::build_row_fold_mats_scaled(&data, c)
            );
        }
    }

    use super::*;

    /// Independent restatement of the Lagrange weight formula:
    /// `L_i(z) = ∏_{j≠i} (z + s_j) / ∏_{j≠i} (s_i + s_j)` over the φ_8 node
    /// set beginning at `node_base`.
    fn lagrange_weights_reference(k_skip: usize, z: F128, node_base: usize) -> Vec<F128> {
        let ell = 1usize << k_skip;
        let mut weights = vec![F128::ZERO; ell];
        for i in 0..ell {
            let si = PHI_8_TABLE[node_base + i];
            let mut num = F128::ONE;
            let mut den = F128::ONE;
            for j in 0..ell {
                if j == i {
                    continue;
                }
                let sj = PHI_8_TABLE[node_base + j];
                num *= z + sj;
                den *= si + sj;
            }
            weights[i] = num * den.inv();
        }
        weights
    }

    /// **Lagrange weight oracle**: the shipped cofactor route must equal the
    /// per-node product-and-invert formula exactly, on both the S and Λ node
    /// sets, for every `k_skip` the protocol can present.
    #[test]
    fn lagrange_weights_match_per_node_formula() {
        let mut rng = Rng::new(0x1EAF_0F17);
        for k_skip in 0..=6usize {
            let ell = 1usize << k_skip;
            for _ in 0..8 {
                let z = rng.f128();
                assert_eq!(
                    lagrange_weights_naive(k_skip, z),
                    lagrange_weights_reference(k_skip, z, 0),
                    "S domain k_skip={k_skip}"
                );
                assert_eq!(
                    lagrange_weights_lambda_naive(k_skip, z),
                    lagrange_weights_reference(k_skip, z, ell),
                    "lambda domain k_skip={k_skip}"
                );
            }
        }
    }

    /// **On-node degeneracy**: a fold point sitting exactly on a node must
    /// still produce that node's indicator vector.
    #[test]
    fn lagrange_weights_on_a_node_are_the_indicator() {
        for k_skip in 0..=6usize {
            let ell = 1usize << k_skip;
            for m in 0..ell {
                let z = PHI_8_TABLE[m];
                let w = lagrange_weights_naive(k_skip, z);
                assert_eq!(w, lagrange_weights_reference(k_skip, z, 0));
                for (i, wi) in w.iter().enumerate() {
                    let expect = if i == m { F128::ONE } else { F128::ZERO };
                    assert_eq!(*wi, expect, "k_skip={k_skip} node={m} i={i}");
                }
                let zl = PHI_8_TABLE[ell + m];
                let wl = lagrange_weights_lambda_naive(k_skip, zl);
                assert_eq!(wl, lagrange_weights_reference(k_skip, zl, ell));
                for (i, wi) in wl.iter().enumerate() {
                    let expect = if i == m { F128::ONE } else { F128::ZERO };
                    assert_eq!(*wi, expect, "lambda k_skip={k_skip} node={m} i={i}");
                }
            }
        }
    }

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bit(&mut self) -> bool {
            (self.next_u64() & 1) != 0
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.bit()).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    // ----------------------------------------------------------------------
    // Lagrange weights — algebraic properties.
    // ----------------------------------------------------------------------

    /// `Σ_i L_i(z) = 1` for all z. The polynomial `1` interpolates to constant
    /// `1` at every node, so its evaluation at z is `Σ_i 1·L_i(z) = Σ_i L_i(z)`.
    #[test]
    fn lagrange_weights_sum_to_one() {
        let mut rng = Rng::new(1);
        for &k_skip in &[1usize, 2, 3, 4, 5, 6] {
            for _ in 0..4 {
                let z = rng.f128();
                let weights = lagrange_weights_naive(k_skip, z);
                let sum: F128 = weights.iter().copied().fold(F128::ZERO, |a, b| a + b);
                assert_eq!(sum, F128::ONE, "Σ L_i ≠ 1 at k_skip={k_skip}");
            }
        }
    }

    /// `L_i(s_j) = δ_{ij}` — Kronecker delta. At a node, exactly one weight is 1.
    #[test]
    fn lagrange_at_node_is_indicator() {
        for k_skip in [2usize, 3, 4, 5] {
            let ell = 1usize << k_skip;
            for i in 0..ell {
                let z = PHI_8_TABLE[i];
                let weights = lagrange_weights_naive(k_skip, z);
                for j in 0..ell {
                    let expected = if j == i { F128::ONE } else { F128::ZERO };
                    assert_eq!(weights[j], expected, "k_skip={k_skip}, z=node{i}, j={j}");
                }
            }
        }
    }

    // ----------------------------------------------------------------------
    // Fold — algebraic properties.
    // ----------------------------------------------------------------------

    /// At a node `z = φ_8(i)`, fold reduces to the witness restricted to s=i:
    /// `a_mlv[x_rest] = a[x_rest · 2^k_skip + i]` (lifted to F_128).
    #[test]
    fn fold_at_node_recovers_witness_slice() {
        let m = 8;
        let k_skip = 3;
        let ell = 1usize << k_skip;
        let n_rest = 1usize << (m - k_skip);
        let mut rng = Rng::new(7);
        let a = rng.bits(1 << m);
        for i in 0..ell {
            let z = PHI_8_TABLE[i];
            let weights = lagrange_weights_naive(k_skip, z);
            let a_mlv = fold_at_z_naive(&a, m, k_skip, &weights);
            for x_rest in 0..n_rest {
                let expected = if a[x_rest * ell + i] {
                    F128::ONE
                } else {
                    F128::ZERO
                };
                assert_eq!(
                    a_mlv[x_rest], expected,
                    "fold at node {i} mismatch at x_rest={x_rest}"
                );
            }
        }
    }

    /// Fold is linear in the input witness: fold(a ⊕ a') = fold(a) + fold(a').
    /// (XOR-linearity is the defining property of the multilinear extension.)
    #[test]
    fn fold_is_xor_linear() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(11);
        let a = rng.bits(1 << m);
        let aprime = rng.bits(1 << m);
        let a_xor: Vec<bool> = a.iter().zip(&aprime).map(|(x, y)| x ^ y).collect();
        let z = rng.f128();
        let weights = lagrange_weights_naive(k_skip, z);

        let fa = fold_at_z_naive(&a, m, k_skip, &weights);
        let fap = fold_at_z_naive(&aprime, m, k_skip, &weights);
        let fxor = fold_at_z_naive(&a_xor, m, k_skip, &weights);
        for i in 0..fa.len() {
            assert_eq!(fa[i] + fap[i], fxor[i], "linearity broken at i={i}");
        }
    }

    // ----------------------------------------------------------------------
    // Round-2 message — properties + cross-checks.
    // ----------------------------------------------------------------------

    /// All-zero witness ⇒ a_mlv = b_mlv = 0 ⇒ G(1) = G(∞) = 0, so the message
    /// elements (r[0]·G(1), G(∞)) are also both zero.
    #[test]
    fn zero_witness_gives_zero_round_message() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(20);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let zeros = vec![false; 1 << m];
        let (a_mlv, b_mlv, msg_1, msg_inf) =
            uni_skip_fold_and_round_pair_naive(&zeros, &zeros, m, k_skip, z, &mlv_challenges);
        assert!(a_mlv.iter().all(|v| v.is_zero()));
        assert!(b_mlv.iter().all(|v| v.is_zero()));
        assert_eq!(msg_1, F128::ZERO);
        assert_eq!(msg_inf, F128::ZERO);
    }

    #[test]
    fn deterministic() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(33);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let o1 = uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        let o2 = uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        assert_eq!(o1, o2);
    }

    /// Round-pair message is symmetric in a, b: swapping a↔b gives the same
    /// message. `a · b = b · a` is built-in, and the `r[0]` multiplier doesn't
    /// distinguish AB.
    #[test]
    fn round_pair_symmetric_in_ab() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(40);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let (_, _, m1_ab, minf_ab) =
            uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        let (_, _, m1_ba, minf_ba) =
            uni_skip_fold_and_round_pair_naive(&b, &a, m, k_skip, z, &mlv_challenges);
        assert_eq!(m1_ab, m1_ba);
        assert_eq!(minf_ab, minf_ba);
    }

    // ----------------------------------------------------------------------
    // Optimized fused — UniSkipFoldTable + fold_one_row, then naive cross-check.
    // ----------------------------------------------------------------------

    /// NEON `fold_one_row_neon_unchecked_8` matches scalar `fold_one_row`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fold_one_row_neon_matches_scalar() {
        let k_skip = 6;
        let mut rng = Rng::new(70);
        let z = rng.f128();
        let table = UniSkipFoldTable::new(k_skip, z);

        let mut cases = vec![[0u8; 8], [0xffu8; 8]];
        for _ in 0..256 {
            let mut bytes = [0u8; 8];
            for byte in bytes.iter_mut() {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            cases.push(bytes);
        }
        for bytes in cases {
            let scalar = table.fold_one_row(&bytes);
            // SAFETY: on aarch64; bytes has 8 entries; table has 8 chunks.
            let neon = unsafe {
                fold_one_row_neon_unchecked_8(table.data.as_ptr() as *const u8, bytes.as_ptr())
            };
            assert_eq!(scalar, neon, "fold mismatch bytes={bytes:02x?}");
        }
    }

    /// The NT tail leaf (fused fold + register-sourced message + `stnp`)
    /// produces bit-identical folded outputs AND message partial sums to the
    /// generic path (`fold_pairs` + reload loop) it replaces on large rounds.
    /// The NT hint changes cache allocation only, so the equality must hold at
    /// any size — tested here across several lo_size shapes including the
    /// odd/2-wide tails the generic path special-cases.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn tail_fold_nt_leaf_matches_generic() {
        let mut rng = Rng::new(72);
        for lo_size in [2usize, 4, 6, 8, 64, 256] {
            let n_in = 4 * lo_size;
            let a_in = rng.f128_vec(n_in);
            let b_in = rng.f128_vec(n_in);
            let eq_lo = rng.f128_vec(lo_size);
            let r_fold = rng.f128();

            // Generic reference: fold_pairs then the reload message loop.
            let mut a_ref = vec![F128::ZERO; 2 * lo_size];
            let mut b_ref = vec![F128::ZERO; 2 * lo_size];
            crate::field::f128_slice::fold_pairs(&a_in, 0, &mut a_ref, r_fold);
            crate::field::f128_slice::fold_pairs(&b_in, 0, &mut b_ref, r_fold);
            let mut p1_acc = F256Unreduced::ZERO;
            let mut pinf_acc = F256Unreduced::ZERO;
            for x_lo in 0..lo_size {
                let o = 2 * x_lo;
                let (a0, a1, b0, b1) = (a_ref[o], a_ref[o + 1], b_ref[o], b_ref[o + 1]);
                p1_acc ^= eq_lo[x_lo].mul_unreduced(a1 * b1);
                pinf_acc ^= eq_lo[x_lo].mul_unreduced((a0 + a1) * (b0 + b1));
            }
            let (p1_ref, pinf_ref) = (p1_acc.reduce(), pinf_acc.reduce());

            let mut a_nt = vec![F128::ZERO; 2 * lo_size];
            let mut b_nt = vec![F128::ZERO; 2 * lo_size];
            // SAFETY: aes is cfg-guaranteed; buffers sized 4·lo_size in /
            // 2·lo_size out / lo_size eq exactly as the contract requires.
            let (p1_nt, pinf_nt) = unsafe {
                kernels::aarch64::tail_fold_chunk_nt_neon(
                    a_in.as_ptr(),
                    b_in.as_ptr(),
                    a_nt.as_mut_ptr(),
                    b_nt.as_mut_ptr(),
                    eq_lo.as_ptr(),
                    lo_size,
                    r_fold,
                )
            };

            assert_eq!(a_ref, a_nt, "folded a mismatch at lo_size={lo_size}");
            assert_eq!(b_ref, b_nt, "folded b mismatch at lo_size={lo_size}");
            assert_eq!(p1_ref, p1_nt, "p1 mismatch at lo_size={lo_size}");
            assert_eq!(pinf_ref, pinf_nt, "pinf mismatch at lo_size={lo_size}");
        }
    }
    /// The q-form tail leaf (all-NEON registers, no GPR round trips) is
    /// bit-identical to the scalar-struct NT leaf AND the generic path across
    /// shapes, including a ranked-adjacent lo_size (the ranked worker's first
    /// tail round runs lo_size = 4096).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn tail_fold_qneon_leaf_matches_v1_and_generic() {
        let mut rng = Rng::new(73);
        for lo_size in [2usize, 4, 6, 8, 64, 256, 4096] {
            let n_in = 4 * lo_size;
            let a_in = rng.f128_vec(n_in);
            let b_in = rng.f128_vec(n_in);
            let eq_lo = rng.f128_vec(lo_size);
            let r_fold = rng.f128();

            // Generic reference: fold_pairs then the reload message loop.
            let mut a_ref = vec![F128::ZERO; 2 * lo_size];
            let mut b_ref = vec![F128::ZERO; 2 * lo_size];
            crate::field::f128_slice::fold_pairs(&a_in, 0, &mut a_ref, r_fold);
            crate::field::f128_slice::fold_pairs(&b_in, 0, &mut b_ref, r_fold);
            let mut p1_acc = F256Unreduced::ZERO;
            let mut pinf_acc = F256Unreduced::ZERO;
            for x_lo in 0..lo_size {
                let o = 2 * x_lo;
                let (a0, a1, b0, b1) = (a_ref[o], a_ref[o + 1], b_ref[o], b_ref[o + 1]);
                p1_acc ^= eq_lo[x_lo].mul_unreduced(a1 * b1);
                pinf_acc ^= eq_lo[x_lo].mul_unreduced((a0 + a1) * (b0 + b1));
            }
            let (p1_ref, pinf_ref) = (p1_acc.reduce(), pinf_acc.reduce());

            let mut a_v1 = vec![F128::ZERO; 2 * lo_size];
            let mut b_v1 = vec![F128::ZERO; 2 * lo_size];
            let mut a_q = vec![F128::ZERO; 2 * lo_size];
            let mut b_q = vec![F128::ZERO; 2 * lo_size];
            // SAFETY: aes is cfg-guaranteed; buffers sized 4·lo_size in /
            // 2·lo_size out / lo_size eq exactly as the contract requires.
            let ((p1_v1, pinf_v1), (p1_q, pinf_q)) = unsafe {
                (
                    kernels::aarch64::tail_fold_chunk_nt_neon(
                        a_in.as_ptr(),
                        b_in.as_ptr(),
                        a_v1.as_mut_ptr(),
                        b_v1.as_mut_ptr(),
                        eq_lo.as_ptr(),
                        lo_size,
                        r_fold,
                    ),
                    kernels::aarch64::tail_fold_chunk_q::<true, false>(
                        a_in.as_ptr(),
                        b_in.as_ptr(),
                        a_q.as_mut_ptr(),
                        b_q.as_mut_ptr(),
                        eq_lo.as_ptr(),
                        lo_size,
                        r_fold,
                    ),
                )
            };

            assert_eq!(a_ref, a_q, "folded a mismatch at lo_size={lo_size}");
            assert_eq!(b_ref, b_q, "folded b mismatch at lo_size={lo_size}");
            assert_eq!(p1_ref, p1_q, "p1 mismatch at lo_size={lo_size}");
            assert_eq!(pinf_ref, pinf_q, "pinf mismatch at lo_size={lo_size}");
            assert_eq!(
                (a_v1, b_v1),
                (a_q, b_q),
                "v1/q fold divergence at lo_size={lo_size}"
            );
            assert_eq!(
                (p1_v1, pinf_v1),
                (p1_q, pinf_q),
                "v1/q message divergence at lo_size={lo_size}"
            );

            // Store/load-hint variants change cache behavior only, never bits.
            let mut a_v = vec![F128::ZERO; 2 * lo_size];
            let mut b_v = vec![F128::ZERO; 2 * lo_size];
            // SAFETY: same contract as above.
            let m_nt_ldnp = unsafe {
                kernels::aarch64::tail_fold_chunk_q::<true, true>(
                    a_in.as_ptr(),
                    b_in.as_ptr(),
                    a_v.as_mut_ptr(),
                    b_v.as_mut_ptr(),
                    eq_lo.as_ptr(),
                    lo_size,
                    r_fold,
                )
            };
            assert_eq!((&a_ref, &b_ref), (&a_v, &b_v), "ldnp variant fold mismatch");
            assert_eq!(
                m_nt_ldnp,
                (p1_ref, pinf_ref),
                "ldnp variant message mismatch"
            );
            // SAFETY: same contract as above.
            let m_reg = unsafe {
                kernels::aarch64::tail_fold_chunk_q::<false, false>(
                    a_in.as_ptr(),
                    b_in.as_ptr(),
                    a_v.as_mut_ptr(),
                    b_v.as_mut_ptr(),
                    eq_lo.as_ptr(),
                    lo_size,
                    r_fold,
                )
            };
            assert_eq!(
                (&a_ref, &b_ref),
                (&a_v, &b_v),
                "regular-store variant fold mismatch"
            );
            assert_eq!(
                m_reg,
                (p1_ref, pinf_ref),
                "regular-store variant message mismatch"
            );
        }
    }
    /// The q-form round-2 message leaf is bit-identical to the scalar-struct
    /// leaf across shapes, exercising the `b ≡ 1` constant-fiber shortcut
    /// (forced all-ones b rows), all-zero rows, and padding-skip pairs.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn round2_qmsg_leaf_matches_v1() {
        let mut rng = Rng::new(74);
        let table = UniSkipFoldTable::new(6, rng.f128());
        for lo_size in [2usize, 8, 64, 512] {
            // 2·lo_size packed rows of 8 bytes each per array.
            let mut a_packed: Vec<u8> = (0..2 * lo_size * 8)
                .map(|_| (rng.next_u64() & 0xff) as u8)
                .collect();
            let mut b_packed = a_packed.clone();
            for byte in &mut b_packed {
                *byte ^= (rng.next_u64() & 0xff) as u8;
            }
            // Force constant fibers: some all-ones b rows (shortcut), some
            // all-zero a rows (row-fold early exit).
            for row in 0..2 * lo_size {
                match row % 7 {
                    0 | 1 => b_packed[row * 8..(row + 1) * 8].fill(0xff),
                    2 => a_packed[row * 8..(row + 1) * 8].fill(0),
                    _ => {}
                }
            }
            let eq_lo = rng.f128_vec(lo_size);
            // Padding skip: mark the top quarter of pairs as padding.
            let pair_in_block_mask = lo_size - 1;
            let useful_pairs_inclusive = lo_size - lo_size / 4;

            let mut a_v1 = vec![F128::ZERO; 2 * lo_size];
            let mut b_v1 = vec![F128::ZERO; 2 * lo_size];
            let mut a_q = vec![F128::ZERO; 2 * lo_size];
            let mut b_q = vec![F128::ZERO; 2 * lo_size];
            // SAFETY: aes is cfg-guaranteed; 2·lo_size packed rows / output
            // elements and lo_size eq weights as the contract requires.
            let (m_v1, m_q) = unsafe {
                (
                    kernels::aarch64::round2_chunk_raw_neon(
                        table.data.as_ptr() as *const u8,
                        a_packed.as_ptr(),
                        b_packed.as_ptr(),
                        a_v1.as_mut_ptr(),
                        b_v1.as_mut_ptr(),
                        eq_lo.as_ptr(),
                        lo_size,
                        0,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                    ),
                    kernels::aarch64::round2_chunk_raw_neon_q(
                        table.data.as_ptr() as *const u8,
                        a_packed.as_ptr(),
                        b_packed.as_ptr(),
                        a_q.as_mut_ptr(),
                        b_q.as_mut_ptr(),
                        eq_lo.as_ptr(),
                        lo_size,
                        0,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                    ),
                )
            };
            assert_eq!(a_v1, a_q, "folded a mismatch at lo_size={lo_size}");
            assert_eq!(b_v1, b_q, "folded b mismatch at lo_size={lo_size}");
            assert_eq!(m_v1, m_q, "message mismatch at lo_size={lo_size}");
        }
    }

    /// Four-row x86 lookup fold matches four independent scalar folds.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold_round2_pair_x86_matches_scalar() {
        let mut rng = Rng::new(71);
        let table = UniSkipFoldTable::new(6, rng.f128());

        for _ in 0..256 {
            let mut rows = [[0u8; 8]; 4];
            for row in &mut rows {
                for byte in row {
                    *byte = (rng.next_u64() & 0xff) as u8;
                }
            }
            let expected = rows.map(|row| table.fold_one_row(&row));
            // SAFETY: each row has 8 bytes and the table has 8 × 256 entries.
            let actual = unsafe {
                fold_round2_pair_x86_unchecked_8(
                    table.data.as_ptr(),
                    rows[0].as_ptr(),
                    rows[1].as_ptr(),
                    rows[2].as_ptr(),
                    rows[3].as_ptr(),
                )
            };
            assert_eq!(actual, expected);
        }
    }

    /// `fold_in_place_pair` correctness: post-fold a[x] = a[2x] + X·(a[2x+1]+a[2x]).
    #[test]
    fn fold_in_place_pair_matches_formula() {
        let mut rng = Rng::new(300);
        for &log_n in &[1usize, 2, 3, 4, 6] {
            let n = 1usize << log_n;
            let a_orig: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b_orig: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let challenge = rng.f128();

            let mut a = a_orig.clone();
            let mut b = b_orig.clone();
            fold_in_place_pair(&mut a, &mut b, challenge);

            assert_eq!(a.len(), n / 2);
            assert_eq!(b.len(), n / 2);
            for x in 0..(n / 2) {
                let a0 = a_orig[2 * x];
                let a1 = a_orig[2 * x + 1];
                let b0 = b_orig[2 * x];
                let b1 = b_orig[2 * x + 1];
                assert_eq!(a[x], a0 + challenge * (a1 + a0), "log_n={log_n}, x={x}");
                assert_eq!(b[x], b0 + challenge * (b1 + b0), "log_n={log_n}, x={x}");
            }
        }
    }

    /// **The c-claim identity**: `C_s · interpolate(round1_c, k_skip, z)` equals
    /// `ĉ(z, r_rest)` computed by direct folding (Lagrange at z, then bind each
    /// `r_rest` value). This is the math identity that lets the extract_c
    /// prover skip per-round c tracking entirely.
    #[test]
    fn c_eval_from_round1_c_matches_direct_fold() {
        use crate::field::F8;
        use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
        use crate::zerocheck::univariate_skip_optimized::{
            c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed,
            small_challenges_ghash,
        };

        const K_SKIP: usize = 6;
        const N_INNER: usize = 7;

        for &m in &[14usize, 15, 16] {
            let mut rng = Rng::new(500 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);

            // Build r with protocol-fixed constants in the middle 7 dims,
            // matching how `prove` constructs it.
            let mut r = vec![F128::ZERO; m];
            for slot in r[..K_SKIP].iter_mut() {
                *slot = rng.f128();
            }
            for (i, v) in small_challenges_ghash().iter().enumerate() {
                r[K_SKIP + i] = *v;
            }
            for (i, v) in medium_challenges_ghash().iter().enumerate() {
                r[K_SKIP + 3 + i] = *v;
            }
            for slot in r[K_SKIP + N_INNER..].iter_mut() {
                *slot = rng.f128();
            }
            let z = rng.f128();

            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);
            let c_packed = pack_bits(&c);

            let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
            let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            let (_round1_ab, round1_c) = round1_shift_reduce_extract_c_packed(
                &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &inv_table,
            );

            // Path A: interpolate round1_c at z, scale by C_s.
            let c_eval_via_interpolation =
                c_s_f128() * interpolate_at_z_on_lambda(&round1_c, K_SKIP, z);

            // Path B: direct fold of c at z (Lagrange) then bind each
            // r_rest = r[K_SKIP..m] element with fold_in_place_single.
            let weights = lagrange_weights_naive(K_SKIP, z);
            let mut c_mlv = fold_at_z_naive(&c, m, K_SKIP, &weights);
            for &r_val in &r[K_SKIP..] {
                fold_in_place_single(&mut c_mlv, r_val);
            }
            assert_eq!(c_mlv.len(), 1);
            let c_eval_via_fold = c_mlv[0];

            assert_eq!(
                c_eval_via_interpolation, c_eval_via_fold,
                "c-claim identity broken at m={m}"
            );
        }
    }

    /// **Tail chunk-split regrouping identity**: the generic tail round is
    /// bit-identical for every `n_hi`, including `n_hi = 0` (one chunk on the
    /// calling thread, no rayon region) — which is what lets the shipped
    /// policy pick the split by size. `eq` factors exactly as a tensor
    /// product and the deferred reduction is F2-linear, so regrouping the
    /// per-element products across chunk boundaries cannot move a bit. Also
    /// cross-checks against the unfused `fold_in_place_pair` +
    /// `round_pair_naive` oracle.
    #[test]
    fn tail_round_identical_across_chunk_splits() {
        let mut rng = Rng::new(9137);
        for &log_n in &[10usize, 11, 13, 15] {
            let n = 1usize << log_n;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let r_fold = rng.f128();
            let r_next = rng.f128_vec(log_n - 1);

            // Oracle: unfused two-pass route.
            let mut a_orc = a.clone();
            let mut b_orc = b.clone();
            fold_in_place_pair(&mut a_orc, &mut b_orc, r_fold);
            let (m1_orc, minf_orc) = round_pair_naive(&a_orc, &b_orc, &r_next);

            // `n_hi` must leave n_lo ≥ 1 (eq is over log_n − 2 vars).
            for n_hi in 0..=(log_n - 3).min(SplitEqGhash::MAX_N_HI) {
                // POISON-prefill: any output slot the kernel fails to write
                // shows up as a mismatch rather than a stale zero.
                let poison = F128 {
                    lo: 0xDEAD_BEEF_DEAD_BEEF,
                    hi: 0xFEED_FACE_FEED_FACE,
                };
                let mut a_out = vec![poison; n / 2];
                let mut b_out = vec![poison; n / 2];
                let (m1, minf) = fold_and_compute_round_pair_into_with_n_hi(
                    &a, &b, &mut a_out, &mut b_out, r_fold, &r_next, n_hi,
                );
                assert_eq!(a_out, a_orc, "a mismatch log_n={log_n} n_hi={n_hi}");
                assert_eq!(b_out, b_orc, "b mismatch log_n={log_n} n_hi={n_hi}");
                assert_eq!(m1, m1_orc, "msg_1 mismatch log_n={log_n} n_hi={n_hi}");
                assert_eq!(minf, minf_orc, "msg_inf mismatch log_n={log_n} n_hi={n_hi}");
            }
        }
    }

    /// The shipped size policy never asks for an out-of-range split: `n_lo`
    /// stays ≥ 1 (the kernel needs `lo_size ≥ 2`) and `n_hi ≤ MAX_N_HI`.
    #[test]
    fn tail_split_policy_stays_in_range() {
        for log_n in 10usize..=28 {
            let half = 1usize << (log_n - 1);
            let n_hi = if half <= zc_serial_tail_max_out() {
                0
            } else {
                tail_n_hi_for(half)
            };
            assert!(n_hi <= SplitEqGhash::MAX_N_HI, "log_n={log_n}");
            assert!(n_hi < log_n - 2, "log_n={log_n} n_hi={n_hi}");
        }
    }

    /// Packed occupancy split stays inside the kernel contract: `n_lo ≥ 1`
    /// (`lo_size ≥ 2`, even) and never below the incumbent `lookahead_n_hi`.
    #[test]
    fn packed_split_policy_stays_in_range() {
        // Ranked r2: n_vars = (m − k_skip) − 1 = 25. Ranked n26: 23.
        for n_vars in 2usize..=28 {
            let n_hi = packed_split_n_hi(n_vars);
            let base = lookahead_n_hi(n_vars);
            assert!(n_hi >= base, "n_vars={n_vars} n_hi={n_hi} base={base}");
            assert!(n_hi < n_vars, "n_vars={n_vars} n_hi={n_hi}");
            let n_lo = n_vars - n_hi;
            assert!(n_lo >= 1, "n_vars={n_vars} n_lo={n_lo}");
            let lo_size = 1usize << n_lo;
            assert!(
                lo_size.is_multiple_of(2),
                "n_vars={n_vars} lo_size={lo_size}"
            );
        }
        assert_eq!(packed_split_n_hi(25), 13, "ranked r2");
        assert_eq!(packed_split_n_hi(23), 13, "ranked n26");
        assert_eq!(packed_split_n_hi(12), lookahead_n_hi(12), "below lo floor");
    }

    /// Packed n26 (and the r2 lookahead that feeds it) is bit-identical
    /// across `n_hi` — the occupancy split only regroups XOR reductions.
    #[test]
    fn packed_fold2_identical_across_chunk_splits() {
        const K_SKIP: usize = 6;
        let m = 16usize;
        let mut rng = Rng::new(0xA11C);
        let (a_packed, b_packed, padding) = lookahead_witness(&mut rng, m, true);
        let z = rng.f128();
        let table = UniSkipFoldTable::new(K_SKIP, z);
        let mut mlv = rng.f128_vec(m - K_SKIP);
        mlv[0] = F128::ONE;
        if mlv[1] == F128::ZERO {
            mlv[1] = F128::ONE;
        }
        let rho1 = rng.f128();
        let rho2 = rng.f128();
        let mut r_next4 = vec![F128::ONE; m - K_SKIP - 2];
        r_next4[1..].copy_from_slice(&mlv[3..]);
        if r_next4[1] == F128::ZERO {
            r_next4[1] = F128::ONE;
        }
        let n = 1usize << (m - K_SKIP);
        let n_vars = r_next4.len() - 1;
        let poison = F128 {
            lo: 0xDEAD_BEEF_DEAD_BEEF,
            hi: 0xFEED_FACE_FEED_FACE,
        };

        let mut a_ref = vec![poison; n / 4];
        let mut b_ref = vec![poison; n / 4];
        // n_hi = 1 is the coarsest split that still leaves lo_size ≥ 2.
        PACKED_SPLIT_N_HI_OVERRIDE.with(|c| c.set(Some(1)));
        let (m1_ref, mi_ref, la_ref) = fold2_from_packed_and_round_pair_lookahead_into(
            &a_packed, &b_packed, m, K_SKIP, &table, &padding, &mut a_ref, &mut b_ref, rho1, rho2,
            &r_next4,
        );

        for n_hi in 2..=n_vars.saturating_sub(1) {
            let mut a_out = vec![poison; n / 4];
            let mut b_out = vec![poison; n / 4];
            PACKED_SPLIT_N_HI_OVERRIDE.with(|c| c.set(Some(n_hi)));
            let (m1, mi, la) = fold2_from_packed_and_round_pair_lookahead_into(
                &a_packed, &b_packed, m, K_SKIP, &table, &padding, &mut a_out, &mut b_out, rho1,
                rho2, &r_next4,
            );
            assert_eq!(a_out, a_ref, "a n_hi={n_hi}");
            assert_eq!(b_out, b_ref, "b n_hi={n_hi}");
            assert_eq!((m1, mi), (m1_ref, mi_ref), "msg n_hi={n_hi}");
            assert_eq!(la, la_ref, "lookahead n_hi={n_hi}");
        }
        PACKED_SPLIT_N_HI_OVERRIDE.with(|c| c.set(None));

        // r2 materialize vs nomat, both on the packed occupancy split.
        let (a2, b2, m1_mat, mi_mat, la_mat) =
            uni_skip_fold_and_round_pair_optimized_packed_padded_lookahead(
                &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
            );
        let (m1_nm, mi_nm, la_nm) = uni_skip_round_pair_lookahead_nomat_packed_padded(
            &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
        );
        assert_eq!(
            (m1_mat, mi_mat),
            (m1_nm, mi_nm),
            "r2 msg across packed split"
        );
        assert_eq!(la_mat, la_nm, "r2 lookahead across packed split");
        assert_eq!(a2.len(), n);
        assert_eq!(b2.len(), n);
    }

    /// **The big cross-check**: fused `fold_and_compute_round_pair_optimized`
    /// produces the same output as the unfused sequence
    /// `fold_in_place_pair` → `round_pair_naive`.
    #[test]
    fn fused_round_matches_unfused() {
        let mut rng = Rng::new(310);
        // fold_and_compute requires lo_size ≥ 2 in SplitEqGhash. eq is over
        // r_next[1..] (size log_n − 2); with MAX_N_HI = 7, n_lo ≥ 1 needs
        // eq size ≥ 8 ⇒ log_n ≥ 10. Smaller cases use the unfused path.
        for &log_n in &[10usize, 11, 12] {
            let n = 1usize << log_n;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let r_fold = rng.f128();
            let r_next = rng.f128_vec(log_n - 1);

            // Fused path.
            let (a_fused, b_fused, m1_fused, minf_fused) =
                fold_and_compute_round_pair_optimized(&a, &b, r_fold, &r_next);

            // Unfused path: clone, in-place fold, naive message.
            let mut a_unf = a.clone();
            let mut b_unf = b.clone();
            fold_in_place_pair(&mut a_unf, &mut b_unf, r_fold);
            let (m1_unf, minf_unf) = round_pair_naive(&a_unf, &b_unf, &r_next);

            assert_eq!(a_fused, a_unf, "a mismatch at log_n={log_n}");
            assert_eq!(b_fused, b_unf, "b mismatch at log_n={log_n}");
            assert_eq!(m1_fused, m1_unf, "msg_1 mismatch at log_n={log_n}");
            assert_eq!(minf_fused, minf_unf, "msg_inf mismatch at log_n={log_n}");
        }
    }

    /// Parallel `uni_skip_fold_and_round_pair_optimized_packed` produces
    /// byte-identical output to the serial version. F128 XOR + multiply sum
    /// is commutative + associative, so worker scheduling order doesn't
    /// affect the result.
    #[test]
    fn parallel_matches_serial() {
        for &m in &[7usize, 8, 9, 10] {
            let k_skip = 6;
            if m <= k_skip {
                continue;
            }
            let mut rng = Rng::new(200 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - k_skip);
            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);
            let table = UniSkipFoldTable::new(k_skip, z);

            let par = uni_skip_fold_and_round_pair_optimized_packed(
                &a_packed,
                &b_packed,
                m,
                k_skip,
                &table,
                &mlv_challenges,
            );
            let ser = uni_skip_fold_and_round_pair_optimized_packed_serial(
                &a_packed,
                &b_packed,
                m,
                k_skip,
                &table,
                &mlv_challenges,
            );

            assert_eq!(par.0, ser.0, "a_mlv mismatch at m={m}");
            assert_eq!(par.1, ser.1, "b_mlv mismatch at m={m}");
            assert_eq!(par.2, ser.2, "msg_1 mismatch at m={m}");
            assert_eq!(par.3, ser.3, "msg_inf mismatch at m={m}");
        }
    }

    /// **Padding skip is byte-identical to the dense round-2 kernel.** Builds
    /// witnesses with bits `[useful_bits, 2^k_log)` of every block honestly
    /// zero, then asserts the `_padded` kernel produces the same
    /// `(a_mlv, b_mlv, msg_1, msg_inf)` as the dense path.
    ///
    /// Covers all three hash padding shapes: BLAKE3 (k_log=14, useful=15409),
    /// SHA-2 (k_log=15, useful=31401), Keccak (k_log=16, useful=42560).
    #[test]
    fn uni_skip_fold_round_pair_padded_matches_dense() {
        const K_SKIP: usize = 6;
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xFADE_F00D_u64.wrapping_add((k_log * 31 + m) as u64));
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;
            let n_blocks = 1usize << (m - k_log);

            // Random witness, then zero bits [useful_bits, block_size) of each
            // block in both a and b (matches honestly-padded hash R1CS).
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    a[blk * block_size + j] = false;
                    b[blk * block_size + j] = false;
                }
            }
            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);

            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - K_SKIP);
            let table = UniSkipFoldTable::new(K_SKIP, z);
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };

            let dense = uni_skip_fold_and_round_pair_optimized_packed(
                &a_packed,
                &b_packed,
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
            );
            let padded = uni_skip_fold_and_round_pair_optimized_packed_padded(
                &a_packed,
                &b_packed,
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
                &padding,
            );
            assert_eq!(
                dense.0, padded.0,
                "a_mlv: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.1, padded.1,
                "b_mlv: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.2, padded.2,
                "msg_1: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.3, padded.3,
                "msg_inf: m={m}, k_log={k_log}, useful={useful_bits}"
            );
        }
    }

    /// `fold_one_row` via the table equals direct-Lagrange fold.
    #[test]
    fn fold_table_one_row_matches_direct_lagrange() {
        let m = 8;
        let k_skip = 3;
        let mut rng = Rng::new(60);
        let z = rng.f128();
        let a = rng.bits(1 << m);
        let weights = lagrange_weights_naive(k_skip, z);
        let table = UniSkipFoldTable::new(k_skip, z);
        let a_packed = pack_bits(&a);

        let n_chunks = 1usize << (k_skip / 8);
        let _ = n_chunks; // ell/8 = (1<<k_skip)/8
        let n_chunks = table.n_chunks;

        for x_rest in 0..(1usize << (m - k_skip)) {
            let direct = {
                let mut acc = F128::ZERO;
                for s in 0..(1usize << k_skip) {
                    if a[x_rest * (1usize << k_skip) + s] {
                        acc += weights[s];
                    }
                }
                acc
            };
            let via_table =
                table.fold_one_row(&a_packed[x_rest * n_chunks..(x_rest + 1) * n_chunks]);
            assert_eq!(via_table, direct, "x_rest={x_rest}");
        }
    }

    /// **The full cross-check**: optimized fused output matches naive
    /// byte-for-byte at the headline `k_skip = 6` (and other small m). Same eq
    /// weights, same z, same r — so a_mlv, b_mlv, and the two message values
    /// must all agree exactly.
    #[test]
    fn optimized_matches_naive() {
        for &m in &[7usize, 8, 9, 10] {
            let k_skip = 6;
            if m <= k_skip {
                continue;
            }
            let mut rng = Rng::new(100 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - k_skip);

            let (a_n, b_n, m1_n, minf_n) =
                uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
            let (a_o, b_o, m1_o, minf_o) =
                uni_skip_fold_and_round_pair_optimized(&a, &b, m, k_skip, z, &mlv_challenges);

            assert_eq!(a_n, a_o, "a_mlv mismatch at m={m}");
            assert_eq!(b_n, b_o, "b_mlv mismatch at m={m}");
            assert_eq!(m1_n, m1_o, "msg_1 mismatch at m={m}");
            assert_eq!(minf_n, minf_o, "msg_inf mismatch at m={m}");
        }
    }

    /// Strong cross-check: compute G(0), G(1), G(∞) by direct sum (using the
    /// LSB-first index convention `a_mlv(0, x') = a[2x']`, `a_mlv(1, x') = a[2x'+1]`),
    /// then verify that G interpolated through those three values agrees with
    /// the direct multilinear evaluation at a fresh random X — confirming G
    /// genuinely has degree ≤ 2.
    ///
    /// Also verifies `round_pair_naive` returns `(r[0] · G(1), G(∞))`.
    #[test]
    fn round_pair_message_has_degree_two() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(55);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let r = rng.f128_vec(m - k_skip);

        let weights = lagrange_weights_naive(k_skip, z);
        let a_mlv = fold_at_z_naive(&a, m, k_skip, &weights);
        let b_mlv = fold_at_z_naive(&b, m, k_skip, &weights);

        let n = a_mlv.len();
        let half = n / 2;
        let eq_remaining = build_eq(&r[1..]);

        // G(0), G(1), G(∞) by direct definition.
        let mut g0 = F128::ZERO;
        let mut g1 = F128::ZERO;
        let mut g_inf = F128::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let eq_x = eq_remaining[x_prime];
            g0 += eq_x * a0 * b0;
            g1 += eq_x * a1 * b1;
            g_inf += eq_x * (a0 + a1) * (b0 + b1);
        }

        // round_pair_naive returns (r[0] · g1, g_inf).
        let (msg_1, msg_inf) = round_pair_naive(&a_mlv, &b_mlv, &r);
        assert_eq!(msg_1, r[0] * g1);
        assert_eq!(msg_inf, g_inf);

        // Degree-2 check: G(X) reconstructed through (G(0), G(1), G(∞)) must
        // agree with the direct multilinear evaluation at a fresh point X.
        // Char-2 interpolation: G(X) = G(0) + X·(G(0)+G(1)) + X·(X+1)·G(∞).
        let x = rng.f128();
        let g_via_poly = g0 + x * (g0 + g1) + x * (x + F128::ONE) * g_inf;
        let mut g_via_sum = F128::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let a_x = a0 + x * (a0 + a1);
            let b_x = b0 + x * (b0 + b1);
            g_via_sum += eq_remaining[x_prime] * a_x * b_x;
        }
        assert_eq!(g_via_poly, g_via_sum);
    }

    // ----------------------------------------------------------------------
    // Two-challenge lookahead (rounds 2–4).
    // ----------------------------------------------------------------------

    /// Honestly padded packed witness pair for `m` (BLAKE3-shaped padding when
    /// `padded`), plus a padding spec.
    fn lookahead_witness(rng: &mut Rng, m: usize, padded: bool) -> (Vec<u8>, Vec<u8>, PaddingSpec) {
        let total_bits = 1usize << m;
        let mut a = rng.bits(total_bits);
        let mut b = rng.bits(total_bits);
        let padding = if padded {
            let k_log = 14.min(m);
            let useful_bits = if k_log == 14 {
                15_409
            } else {
                (1usize << k_log) - 37
            };
            let block_size = 1usize << k_log;
            for blk in 0..(total_bits / block_size) {
                for j in useful_bits..block_size {
                    a[blk * block_size + j] = false;
                    b[blk * block_size + j] = false;
                }
            }
            PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            }
        } else {
            PaddingSpec::dense(m)
        };
        (pack_bits(&a), pack_bits(&b), padding)
    }

    /// The lookahead sweep emits bit-identical folded tables and round-two
    /// message, and its six coefficients reproduce the incumbent round-three
    /// message (`fold_and_compute_round_pair_optimized` on the same tables)
    /// at several ρ₁ — dense and padded, small and mid shapes.
    #[test]
    fn lookahead_round3_matches_incumbent() {
        const K_SKIP: usize = 6;
        for &(m, padded) in &[
            (13usize, false),
            (14, false),
            (15, true),
            (16, false),
            (17, true),
            (18, true),
        ] {
            let mut rng = Rng::new(0x1A00 + m as u64 + padded as u64 * 100);
            let (a_packed, b_packed, padding) = lookahead_witness(&mut rng, m, padded);
            let z = rng.f128();
            let table = UniSkipFoldTable::new(K_SKIP, z);
            let mut mlv = rng.f128_vec(m - K_SKIP);
            mlv[0] = F128::ONE; // Convention A, as the prover passes it.
            assert_ne!(mlv[1], F128::ZERO);

            let (a_ref, b_ref, m1_ref, mi_ref) =
                uni_skip_fold_and_round_pair_optimized_packed_padded(
                    &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
                );
            let (a_la, b_la, m1_la, mi_la, la) =
                uni_skip_fold_and_round_pair_optimized_packed_padded_lookahead(
                    &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
                );
            assert_eq!(a_ref, a_la, "a tables m={m} padded={padded}");
            assert_eq!(b_ref, b_la, "b tables m={m} padded={padded}");
            assert_eq!(
                (m1_ref, mi_ref),
                (m1_la, mi_la),
                "round-2 msg m={m} padded={padded}"
            );

            // Incumbent round three at three challenges (incl. 0 and 1).
            let mut r_next3 = vec![F128::ONE; m - K_SKIP - 1];
            r_next3[1..].copy_from_slice(&mlv[2..]);
            for &rho1 in &[F128::ZERO, F128::ONE, rng.f128(), rng.f128()] {
                let (mut a3, mut b3) = (a_ref.clone(), b_ref.clone());
                fold_in_place_pair(&mut a3, &mut b3, rho1);
                let (m1, mi) = round_pair_naive(&a3, &b3, &r_next3);
                assert_eq!(
                    eval_round3_lookahead(&la, rho1),
                    (m1, mi),
                    "round-3 msg m={m} padded={padded} rho1={rho1:?}"
                );
            }
        }
    }

    /// The composed pass equals fold(ρ₁) then fold(ρ₂) elementwise (outputs
    /// poison-prefilled so an unwritten slot is caught) and reproduces the
    /// incumbent round-four message.
    #[test]
    fn fold2_plain_matches_fold_then_fold() {
        for &log_n in &[4usize, 5, 6, 7, 9, 10, 12, 13] {
            let mut rng = Rng::new(0x2B00 + log_n as u64);
            let n = 1usize << log_n;
            let a = rng.f128_vec(n);
            let b = rng.f128_vec(n);
            let rho1 = rng.f128();
            let rho2 = rng.f128();
            let mut r_next4 = vec![F128::ONE; log_n - 2];
            for v in r_next4[1..].iter_mut() {
                *v = rng.f128();
            }
            let (mut a4, mut b4) = (a.clone(), b.clone());
            fold_in_place_pair(&mut a4, &mut b4, rho1);
            fold_in_place_pair(&mut a4, &mut b4, rho2);
            let (m4_1, m4_i) = round_pair_naive(&a4, &b4, &r_next4);

            let poison = F128 {
                lo: 0xDEAD_BEEF_DEAD_BEEF,
                hi: 0xFEED_FACE_FEED_FACE,
            };
            let mut a_out = vec![poison; n / 4];
            let mut b_out = vec![poison; n / 4];
            let (c4_1, c4_i) =
                fold2_plain_and_round4_into(&a, &b, &mut a_out, &mut b_out, rho1, rho2, &r_next4);
            assert_eq!(a_out, a4, "a tables log_n={log_n}");
            assert_eq!(b_out, b4, "b tables log_n={log_n}");
            assert_eq!((c4_1, c4_i), (m4_1, m4_i), "round-4 msg log_n={log_n}");
        }
    }

    /// The eq bake reproduces the portable reference exactly on the route it
    /// replaces, with random rows and with both canonical B windows live.
    /// A non-tensor `eq_lo` must be rejected by the builder rather than
    /// silently mis-weighted.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn round2_lookahead_bake_matches_scalar() {
        use crate::zerocheck::univariate_skip::build_eq;
        const K_SKIP: usize = 6;
        // Every shape runs under both round-two body arms: the 16-pair body
        // (`unroll16 = true`, the default) and the incumbent 8-pair body the
        // `FLOCK_NO_R2_UNROLL16` kill switch restores. The canonical windows
        // (all-ones at pair 0, sparse at pair 120) sit inside the sweep so
        // the pairing must refuse them and fall back to the 8-pair arms.
        let shapes = [
            (32usize, false),
            (64, false),
            (128, false),
            (32, true),
            (64, true),
            (128, true),
        ];
        for (&(lo_size, canon), unroll16) in
            shapes.iter().flat_map(|s| [(s, true), (s, false)])
        {
            let _arm = kernels::x86_64::R2Unroll16Override::set(unroll16);
            let mut rng = Rng::new(0x5B00 + lo_size as u64 + u64::from(canon));
            let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
            let r2_mats = r2_gfni_mats(&table);
            let n_lo = lo_size.trailing_zeros() as usize;
            let r_lo: Vec<F128> = (0..n_lo).map(|_| rng.f128()).collect();
            let eq_lo = build_eq(&r_lo);
            let n_rows = 2 * lo_size;
            let a_packed: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
            let mut b_packed: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
            if canon {
                // All-ones B window over pairs 0..8 and (when the shape has
                // the rows) the sparse window over pairs 120..128: the two
                // raw guards of the canonical prefold.
                b_packed[0..128].fill(0xff);
                if n_rows >= 256 {
                    b_packed[240 * 8..256 * 8].fill(0);
                    b_packed[240 * 8..240 * 8 + 8]
                        .copy_from_slice(&0x0001_ffff_ffff_ffffu64.to_le_bytes());
                }
            }
            let bake =
                build_r2_eq_bake(&table, &eq_lo, &r_lo).expect("tensor eq_lo must factor");
            let mut a_s: [F128; 0] = [];
            let mut b_s: [F128; 0] = [];
            let out_s = round2_lookahead_chunk_scalar::<false>(
                &a_packed,
                &b_packed,
                &table,
                &mut a_s,
                &mut b_s,
                &eq_lo,
                0,
                0,
                0,
                usize::MAX,
            );
            let mut a_e: [F128; 0] = [];
            let mut b_e: [F128; 0] = [];
            // SAFETY: the packed rows cover row_base..row_base+2*lo_size, the
            // table has the protocol shape, WRITE=false touches no chunk, and
            // the bake was built for this exact `eq_lo` and fold table.
            let out_b = unsafe {
                kernels::x86_64::round2_lookahead_chunk_x86_avx512::<false, true>(
                    table.data.as_ptr(),
                    r2_mats.as_ref(),
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    0,
                    &mut a_e,
                    &mut b_e,
                    &eq_lo,
                    0,
                    0,
                    usize::MAX,
                    None,
                    Some(&bake),
                )
            };
            assert_eq!(
                out_s, out_b,
                "bake lo_size={lo_size} canon={canon} unroll16={unroll16}"
            );
        }
        // Guard: an eq table that is not the tensor of `r_lo` cannot be
        // factored, and the builder must say so.
        let mut rng = Rng::new(0x5BFF);
        let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
        let r_lo: Vec<F128> = (0..5).map(|_| rng.f128()).collect();
        let mut eq_lo = build_eq(&r_lo);
        eq_lo[7] += F128::ONE;
        assert!(build_r2_eq_bake_checked(&table, &eq_lo, &r_lo, true).is_none());
    }

    /// Packed a/b with the ranked witness brand on the canonical rows of every
    /// 256-row block: B rows 0..16 all ones, B row 240 = the static 49-bit
    /// word, A row 240 a 49-bit word, rows 241..256 zero on both sides.
    fn ranked_canonical_witness(rng: &mut Rng, n_rows: usize) -> (Vec<u8>, Vec<u8>) {
        let mut a: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
        let mut b: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
        for blk in 0..n_rows / 256 {
            let row = |r: usize| (blk * 256 + r) * 8;
            b[row(0)..row(16)].fill(0xff);
            b[row(240)..row(256)].fill(0);
            b[row(240)..row(241)].copy_from_slice(&0x0001_ffff_ffff_ffffu64.to_le_bytes());
            let w = u64::from_le_bytes(a[row(240)..row(241)].try_into().unwrap())
                & 0x0001_ffff_ffff_ffff;
            a[row(240)..row(241)].copy_from_slice(&w.to_le_bytes());
            a[row(241)..row(256)].fill(0);
        }
        (a, b)
    }

    /// `Σ_blocks eq(mlv[8..], blk) · bit(A row, s)` for rows 0..16 and 240 —
    /// the identity-C inner-fold rows the elision consumes (A = z there).
    fn canonical_rows_direct(a: &[u8], mlv: &[F128], n_blocks: usize) -> Vec<F128> {
        use crate::zerocheck::univariate_skip::build_eq;
        let eq_blocks = build_eq(&mlv[8..]);
        assert_eq!(eq_blocks.len(), n_blocks);
        let mut rows = vec![F128::ZERO; CANON_ROWS * 64];
        for (slot, row) in (0..16).chain([240usize]).enumerate() {
            for (blk, e) in eq_blocks.iter().enumerate() {
                let off = (blk * 256 + row) * 8;
                let word = u64::from_le_bytes(a[off..off + 8].try_into().unwrap());
                for s in 0..64 {
                    if (word >> s) & 1 == 1 {
                        rows[slot * 64 + s] += *e;
                    }
                }
            }
        }
        rows
    }

    /// The delta formulas are exactly the two canonical windows: the
    /// incumbent sweep over the full witness equals the sweep over the same
    /// witness with both windows zeroed, plus the deltas. Runs on every
    /// target (portable or AVX-512 sweep, no elision).
    #[test]
    fn round2_canonical_window_deltas_are_the_windows() {
        const K_SKIP: usize = 6;
        let padding = PaddingSpec {
            k_log: 14,
            useful_bits_per_block: 15_409,
        };
        for blocks_log in [3usize, 4] {
            let m = K_SKIP + 8 + blocks_log;
            let n_rows = 1usize << (m - K_SKIP);
            let mut rng = Rng::new(0xCA70 + blocks_log as u64);
            let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
            let mut mlv: Vec<F128> = (0..m - K_SKIP).map(|_| rng.f128()).collect();
            mlv[0] = F128::ONE;
            let (mut a, mut b) = ranked_canonical_witness(&mut rng, n_rows);
            let rows = canonical_rows_direct(&a, &mlv, n_rows / 256);
            let (full, full4) = uni_skip_round_pair_lookahead_nomat_packed_padded_with_eq(
                &a, &b, m, K_SKIP, &table, &mlv, &padding, None, None, true,
            );
            for blk in 0..n_rows / 256 {
                let row = |r: usize| (blk * 256 + r) * 8;
                a[row(0)..row(16)].fill(0);
                b[row(0)..row(16)].fill(0);
                a[row(240)..row(256)].fill(0);
                b[row(240)..row(256)].fill(0);
            }
            let (rest, rest4) = uni_skip_round_pair_lookahead_nomat_packed_padded_with_eq(
                &a, &b, m, K_SKIP, &table, &mlv, &padding, None, None, true,
            );
            let (d_g1, d_ginf, d_c) = round2_canonical_window_deltas(&table, &mlv, &rows);
            {
                let mut want = rest4.expect("test route computes round four");
                want.add(&three_var::round4_canonical_window_deltas(&table, &mlv, &rows));
                assert_eq!(full4.expect("round four"), want, "r4 deltas blocks_log={blocks_log}");
            }
            assert_eq!(full.0, rest.0 + mlv[0] * d_g1, "G(1) blocks_log={blocks_log}");
            assert_eq!(full.1, rest.1 + d_ginf, "G(inf) blocks_log={blocks_log}");
            for i in 0..6 {
                assert_eq!(full.2.c[i], rest.2.c[i] + d_c[i], "c[{i}] blocks_log={blocks_log}");
            }
        }
    }

    /// AVX-512 sweep with the canonical windows elided (and their deltas added)
    /// is bit-identical to the incumbent sweep, on both round-two body arms.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn round2_canonical_elide_matches_incumbent() {
        const K_SKIP: usize = 6;
        let padding = PaddingSpec {
            k_log: 14,
            useful_bits_per_block: 15_409,
        };
        for &(blocks_log, n_hi) in &[(4usize, 2usize), (5, 1), (5, 4), (6, 5)] {
            for unroll16 in [true, false] {
                // The driver sweeps its chunks on rayon workers, so the arm
                // override must be installed on every worker of a dedicated
                // pool; a test-thread override alone never reaches the kernel.
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(4)
                    .start_handler(move |_| {
                        std::mem::forget(kernels::x86_64::R2Unroll16Override::set(unroll16));
                    })
                    .build()
                    .expect("unroll16 override pool");
                pool.install(|| {
                    let m = K_SKIP + 8 + blocks_log;
                    let n_rows = 1usize << (m - K_SKIP);
                    let mut rng = Rng::new(0xCA71 + (blocks_log * 16 + n_hi) as u64);
                    let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
                    let mut mlv: Vec<F128> = (0..m - K_SKIP).map(|_| rng.f128()).collect();
                    mlv[0] = F128::ONE;
                    let (a, b) = ranked_canonical_witness(&mut rng, n_rows);
                    let rows = canonical_rows_direct(&a, &mlv, n_rows / 256);
                    let eq = SplitEqGhash::with_n_hi(&mlv[1..], n_hi);
                    assert!((1usize << eq.n_lo) >= 32 && eq.n_lo < mlv.len());
                    let (incumbent, inc4) = uni_skip_round_pair_lookahead_nomat_packed_padded_with_eq(
                        &a, &b, m, K_SKIP, &table, &mlv, &padding, Some(&eq), None, true,
                    );
                    let hits = kernels::x86_64::R2_CANON_ELIDE_HITS
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let (elided, eli4) = uni_skip_round_pair_lookahead_nomat_packed_padded_with_eq(
                        &a, &b, m, K_SKIP, &table, &mlv, &padding, Some(&eq), Some(&rows), true,
                    );
                    let ref4 = three_var::round4_lookahead_ref(&a, &b, &table, &mlv, &padding, K_SKIP);
                    assert_eq!(inc4, Some(ref4), "r4 incumbent route blocks_log={blocks_log} n_hi={n_hi}");
                    assert_eq!(eli4, Some(ref4), "r4 fast elided route blocks_log={blocks_log} n_hi={n_hi} unroll16={unroll16}");
                    assert!(
                        kernels::x86_64::R2_CANON_ELIDE_HITS.load(std::sync::atomic::Ordering::Relaxed)
                            > hits,
                        "elision never engaged"
                    );
                    assert_eq!(
                        incumbent, elided,
                        "blocks_log={blocks_log} n_hi={n_hi} unroll16={unroll16}"
                    );
                });
            }
        }
    }

    /// The four B constants of the three-variable pass are the portable
    /// leaf's own folds of the ranked brand's static lines, in every block.
    /// Runs on every target.
    #[test]
    fn fold3_b_canonical_lines_are_the_static_line_folds() {
        const K_SKIP: usize = 6;
        let padding = PaddingSpec {
            k_log: 14,
            useful_bits_per_block: 15_409,
        };
        let (mask, useful) = round2_pair_skip(&padding, K_SKIP);
        let n_rows = 4 * 256;
        let mut rng = Rng::new(0x3CA0);
        let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
        let rho = [rng.f128(), rng.f128(), rng.f128()];
        let (a, b) = ranked_canonical_witness(&mut rng, n_rows);
        let n_out = n_rows / 8;
        let mut a_out = vec![F128::ZERO; n_out];
        let mut b_out = vec![F128::ZERO; n_out];
        let eq_lo = vec![F128::ONE; n_out / 2];
        three_var::fold3_from_packed_lookahead_scalar(
            &a, &b, &table, 0, &mut a_out, &mut b_out, rho, &eq_lo, mask, useful,
        );
        let consts = three_var::fold3_b_canonical_lines(&table, rho);
        assert_ne!(consts[0], F128::ZERO);
        assert_ne!(consts[2], F128::ZERO);
        for blk in 0..n_rows / 256 {
            for (k, line) in [0usize, 1, 30, 31].into_iter().enumerate() {
                assert_eq!(b_out[32 * blk + line], consts[k], "block {blk} line {line}");
            }
        }
    }

    /// Deeper cascade, r345 side: same message and lookahead as the incumbent
    /// pass, octet tables = the row-major tables permuted, and the round-seven
    /// biquadratic equal to its definition over the row-major tables.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn fold3_casc_round7_lookahead_matches_definition() {
        const K_SKIP: usize = 6;
        let padding = PaddingSpec {
            k_log: 14,
            useful_bits_per_block: 15_409,
        };
        for &(blocks_log, n_hi) in &[(4usize, 2usize), (4, 4), (5, 1), (6, 5)] {
            let m = K_SKIP + 8 + blocks_log;
            let n_rows = 1usize << (m - K_SKIP);
            let n_out = n_rows / 8;
            let mut rng = Rng::new(0xC3C3 + (blocks_log * 16 + n_hi) as u64);
            let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
            let rho = [rng.f128(), rng.f128(), rng.f128()];
            let mut r_next5: Vec<F128> = (0..m - K_SKIP - 3).map(|_| rng.f128()).collect();
            r_next5[0] = F128::ONE;
            let (a, b) = ranked_canonical_witness(&mut rng, n_rows);
            let eq = SplitEqGhash::with_n_hi(&r_next5[1..], n_hi);
            let case = format!("blocks_log={blocks_log} n_hi={n_hi}");
            let run = |casc: bool| {
                let mut a_out = vec![F128::ZERO; n_out];
                let mut b_out = vec![F128::ZERO; n_out];
                let msg = three_var::fold3_from_packed_and_round_pair_lookahead_casc(
                    &a, &b, m, K_SKIP, &table, &padding, &mut a_out, &mut b_out, rho, &r_next5,
                    Some((&eq.lo[..], &eq.hi[..])), false, true, casc, None,
                );
                (msg, a_out, b_out)
            };
            let (inc, a_rm, b_rm) = run(false);
            let (got, a_oct, b_oct) = run(true);
            assert!(inc.3.is_none());
            assert_eq!((inc.0, inc.1, inc.2), (got.0, got.1, got.2), "message {case}");
            for e in 0..n_out {
                let pos = 32 * (e / 32) + 4 * (e % 8) + (e % 32) / 8;
                assert_eq!(a_oct[pos], a_rm[e], "a oct {case} e={e}");
                assert_eq!(b_oct[pos], b_rm[e], "b oct {case} e={e}");
            }
            let w = build_eq(&r_next5[3..]);
            assert_eq!(w.len() * 8, n_out);
            let mut want = Round4Lookahead::ZERO;
            for (q, &om) in w.iter().enumerate() {
                let x = &a_rm[8 * q..8 * q + 8];
                let y = &b_rm[8 * q..8 * q + 8];
                let (pa, pb) = (three_var::corners9(&x[4..8]), three_var::corners9(&y[4..8]));
                let c: [F128; 4] = std::array::from_fn(|k| x[k] + x[k + 4]);
                let d: [F128; 4] = std::array::from_fn(|k| y[k] + y[k + 4]);
                let (pc, pd) = (three_var::corners9(&c), three_var::corners9(&d));
                for k in 0..9 {
                    want.v1[k] += om * (pa[k] * pb[k]);
                    want.vinf[k] += om * (pc[k] * pd[k]);
                }
            }
            assert_eq!(got.3.unwrap(), want, "round-seven lookahead {case}");
        }
    }

    /// Deeper cascade emitted by a plain two-variable level: same message and
    /// lookahead as the incumbent level, octet tables = its row-major tables
    /// permuted, biquadratic equal to its definition.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold2_plain_casc_matches_definition() {
        for &log_n in &[16usize, 17, 18] {
            let n = 1usize << log_n;
            if !fold2_plain_casc_route_ok(n) {
                continue;
            }
            let mut rng = Rng::new(0x2C30 + log_n as u64);
            let a = rng.f128_vec(n);
            let b = rng.f128_vec(n);
            let (ra, rb) = (rng.f128(), rng.f128());
            let mut r_next = rng.f128_vec(log_n - 2);
            r_next[0] = F128::ONE;
            let mut a_rm = vec![F128::ZERO; n / 4];
            let mut b_rm = vec![F128::ZERO; n / 4];
            let inc = fold2_plain_and_round_pair_lookahead_into_lm_casc(
                &a, &b, &mut a_rm, &mut b_rm, ra, rb, &r_next, false, false, false, false,
            );
            let mut a_oct = vec![F128::ZERO; n / 4];
            let mut b_oct = vec![F128::ZERO; n / 4];
            let got = fold2_plain_and_round_pair_lookahead_into_lm_casc(
                &a, &b, &mut a_oct, &mut b_oct, ra, rb, &r_next, false, false, true, false,
            );
            assert!(inc.3.is_none());
            assert_eq!((inc.0, inc.1, inc.2), (got.0, got.1, got.2), "message log_n={log_n}");
            for e in 0..n / 4 {
                let pos = 32 * (e / 32) + 4 * (e % 8) + (e % 32) / 8;
                assert_eq!(a_oct[pos], a_rm[e], "a oct log_n={log_n} e={e}");
                assert_eq!(b_oct[pos], b_rm[e], "b oct log_n={log_n} e={e}");
            }
            let w = build_eq(&r_next[3..]);
            assert_eq!(w.len() * 8, n / 4);
            let mut want = Round4Lookahead::ZERO;
            for (q, &om) in w.iter().enumerate() {
                let x = &a_rm[8 * q..8 * q + 8];
                let y = &b_rm[8 * q..8 * q + 8];
                let (pa, pb) = (three_var::corners9(&x[4..8]), three_var::corners9(&y[4..8]));
                let c: [F128; 4] = std::array::from_fn(|k| x[k] + x[k + 4]);
                let d: [F128; 4] = std::array::from_fn(|k| y[k] + y[k + 4]);
                let (pc, pd) = (three_var::corners9(&c), three_var::corners9(&d));
                for k in 0..9 {
                    want.v1[k] += om * (pa[k] * pb[k]);
                    want.vinf[k] += om * (pc[k] * pd[k]);
                }
            }
            assert_eq!(got.3.unwrap(), want, "biquadratic log_n={log_n}");
        }
    }

    /// Deeper cascade, plain side: the three-variable octet level equals
    /// folding X_a and then running the two-variable lookahead level.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold3_oct_level_matches_fold_then_fold2() {
        for &log_n in &[10usize, 16, 17, 18] {
            let mut rng = Rng::new(0x0C70 + log_n as u64);
            let n = 1usize << log_n;
            let a = rng.f128_vec(n);
            let b = rng.f128_vec(n);
            let rho = [rng.f128(), rng.f128(), rng.f128()];
            let mut r_next = rng.f128_vec(log_n - 3);
            r_next[0] = F128::ONE;
            let oct = |v: &[F128]| {
                let mut o = vec![F128::ZERO; v.len()];
                for e in 0..v.len() {
                    o[32 * (e / 32) + 4 * (e % 8) + (e % 32) / 8] = v[e];
                }
                o
            };
            let fold1 = |v: &[F128]| -> Vec<F128> {
                (0..v.len() / 2).map(|i| v[2 * i] + rho[0] * (v[2 * i] + v[2 * i + 1])).collect()
            };
            let (a1, b1) = (fold1(&a), fold1(&b));
            let mut a_ref = vec![F128::ZERO; n / 8];
            let mut b_ref = vec![F128::ZERO; n / 8];
            let want = fold2_plain_and_round_pair_lookahead_into(
                &a1, &b1, &mut a_ref, &mut b_ref, rho[1], rho[2], &r_next,
            );
            let (ao, bo) = (oct(&a), oct(&b));
            for out_lm in [false, true] {
                if out_lm && !(zc_lane_major_handoff_on() && log_n >= 16) {
                    continue;
                }
                let mut a_got = vec![F128::ZERO; n / 8];
                let mut b_got = vec![F128::ZERO; n / 8];
                let got = fold3_oct_plain_and_round_pair_lookahead_into(
                    &ao, &bo, &mut a_got, &mut b_got, rho, &r_next, out_lm,
                );
                assert_eq!(got, want, "message log_n={log_n} lm={out_lm}");
                if out_lm {
                    assert!(lane_major_tiles(&a_got) == a_ref, "a lm log_n={log_n}");
                    assert!(lane_major_tiles(&b_got) == b_ref, "b lm log_n={log_n}");
                } else {
                    assert!(a_got == a_ref, "a log_n={log_n}");
                    assert!(b_got == b_ref, "b log_n={log_n}");
                }
            }
        }
    }

    /// `lane_select` against the `vpermi2q` definition: qword `i` of the
    /// result is `idx[i] & 7` of the source `idx[i] >> 3` picks.
    #[test]
    fn lane_select_indexes_whole_lanes() {
        let a: [u64; 8] = std::array::from_fn(|i| 100 + i as u64);
        let b: [u64; 8] = std::array::from_fn(|i| 200 + i as u64);
        let sel = [(1u8, 0u8), (0, 3), (1, 2), (0, 1)];
        let idx = lane_select(sel);
        let got: [u64; 8] =
            std::array::from_fn(|i| if idx[i] & 8 == 0 { a } else { b }[(idx[i] & 7) as usize]);
        for (i, &(src, lane)) in sel.iter().enumerate() {
            let from = if src == 0 { a } else { b };
            let lane = lane as usize;
            assert_eq!(got[2 * i..2 * i + 2], from[2 * lane..2 * lane + 2], "lane {i}");
        }
    }

    /// Three-variable pass with B's static lines taken from constants is
    /// bit-identical to the incumbent AVX-512 pass (driver and leaf, every
    /// store variant), and its row-major tables are the portable leaf's.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn fold3_b_canon_elide_matches_incumbent() {
        use std::sync::atomic::Ordering::Relaxed;
        const K_SKIP: usize = 6;
        let padding = PaddingSpec {
            k_log: 14,
            useful_bits_per_block: 15_409,
        };
        let (mask, useful) = round2_pair_skip(&padding, K_SKIP);
        for &(blocks_log, n_hi) in &[(4usize, 2usize), (4, 4), (5, 1), (6, 5)] {
            let m = K_SKIP + 8 + blocks_log;
            let n_rows = 1usize << (m - K_SKIP);
            let n_out = n_rows / 8;
            let mut rng = Rng::new(0x3CA1 + (blocks_log * 16 + n_hi) as u64);
            let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
            let rho = [rng.f128(), rng.f128(), rng.f128()];
            let mut r_next5: Vec<F128> = (0..m - K_SKIP - 3).map(|_| rng.f128()).collect();
            r_next5[0] = F128::ONE;
            let (a, b) = ranked_canonical_witness(&mut rng, n_rows);
            let eq = SplitEqGhash::with_n_hi(&r_next5[1..], n_hi);
            let lo = eq.lo.len();
            assert!(lo >= 16 && lo.is_multiple_of(16));
            let case = format!("blocks_log={blocks_log} n_hi={n_hi}");

            let mut a_ref = vec![F128::ZERO; n_out];
            let mut b_ref = vec![F128::ZERO; n_out];
            three_var::fold3_from_packed_lookahead_scalar(
                &a, &b, &table, 0, &mut a_ref, &mut b_ref, rho,
                &vec![F128::ONE; n_out / 2], mask, useful,
            );

            for out_lm in [false, true] {
                let run = |b_canonical: bool| {
                    let mut a_out = vec![F128::ZERO; n_out];
                    let mut b_out = vec![F128::ZERO; n_out];
                    let msg = three_var::fold3_from_packed_and_round_pair_lookahead_into_with_eq(
                        &a, &b, m, K_SKIP, &table, &padding, &mut a_out, &mut b_out, rho,
                        &r_next5, Some((&eq.lo[..], &eq.hi[..])), out_lm, b_canonical,
                    );
                    (msg, a_out, b_out)
                };
                let incumbent = run(false);
                let hits = kernels::x86_64::R345_B_CANON_HITS.load(Relaxed);
                let elided = run(true);
                assert!(
                    kernels::x86_64::R345_B_CANON_HITS.load(Relaxed) > hits,
                    "elision never engaged {case}"
                );
                assert!(incumbent == elided, "driver {case} out_lm={out_lm}");
                if !out_lm {
                    assert!(elided.1 == a_ref, "a table vs portable {case}");
                    assert!(elided.2 == b_ref, "b table vs portable {case}");
                }
            }

            // Leaf level, so the streaming-store variants run too (the driver
            // only streams at 2^23 outputs).
            let one = F128::ONE;
            let cfold = kernels::x86_64::build_cfold_mats(
                &table.data,
                [
                    (one + rho[0]) * (one + rho[1]),
                    rho[0] * (one + rho[1]),
                    (one + rho[0]) * rho[1],
                    rho[0] * rho[1],
                ],
            );
            let wt = build_w_pair_table(&eq.lo);
            let consts = three_var::fold3_b_canonical_lines(&table, rho);
            macro_rules! leaf_pair {
                ($nt:literal, $lm:literal) => {{
                    let mut want = (vec![F128::ZERO; n_out], vec![F128::ZERO; n_out]);
                    let mut got = (vec![F128::ZERO; n_out], vec![F128::ZERO; n_out]);
                    let aligned = [&want.0, &want.1, &got.0, &got.1]
                        .iter()
                        .all(|v| (v.as_ptr() as usize).is_multiple_of(16));
                    if !$nt || aligned {
                        for (x_hi, out_base) in (0..n_out).step_by(2 * lo).enumerate() {
                            let r = out_base..out_base + 2 * lo;
                            // SAFETY: both leaves read rows 8·out_base .. 8·(out_base + 2·lo)
                            // of the packed inputs and write exactly these output chunks.
                            let (w, g) = unsafe {
                                (
                                    kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_x86::<$nt, $lm>(
                                        a.as_ptr(), b.as_ptr(), out_base,
                                        &mut want.0[r.clone()], &mut want.1[r.clone()],
                                        lo, &cfold, rho[2], &wt,
                                    ),
                                    kernels::x86_64::fold3_from_packed_lookahead_c4_lm_wtab_b_canon_x86::<$nt, $lm>(
                                        a.as_ptr(), b.as_ptr(), out_base,
                                        &mut got.0[r.clone()], &mut got.1[r],
                                        lo, &cfold, rho[2], &wt, &consts,
                                    ),
                                )
                            };
                            assert_eq!(g, w, "leaf sums {case} nt={} lm={} chunk={x_hi}", $nt, $lm);
                        }
                        assert!(got == want, "leaf tables {case} nt={} lm={}", $nt, $lm);
                    }
                }};
            }
            leaf_pair!(false, false);
            leaf_pair!(false, true);
            leaf_pair!(true, false);
            leaf_pair!(true, true);
        }
    }

    /// AVX-512 lookahead sweep kernel vs the portable reference on one chunk,
    /// with and without padded pairs, at several `lo_size` (incl. the scalar
    /// tail sizes 2 and 4).
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn round2_lookahead_chunk_x86_matches_scalar() {
        const K_SKIP: usize = 6;
        let shapes = [
            (2usize, 0usize, usize::MAX),
            (4, 0, usize::MAX),
            (8, 0, usize::MAX),
            (16, 0, usize::MAX),
            (32, 0, usize::MAX),
            (64, 0, usize::MAX),
            (64, 7, 5), // every block of 8 pairs keeps 5 (mixed group + zero groups)
            (128, 15, 12),
            (128, 127, 121),
        ];
        // Both round-two body arms (16-pair default, 8-pair kill-switch).
        for (&(lo_size, mask, useful), unroll16) in
            shapes.iter().flat_map(|s| [(s, true), (s, false)])
        {
            let _arm = kernels::x86_64::R2Unroll16Override::set(unroll16);
            let mut rng = Rng::new(0x3C00 + lo_size as u64 + mask as u64);
            let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
            #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
            let r2_mats = r2_gfni_mats(&table);
            #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
            let r2_mats_arg = r2_mats.as_ref();
            #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
            let r2_mats_arg: Option<&[u64; 128]> = None;
            let n_rows = 4 * lo_size + 16; // slack so row_base can be non-zero
            let mut a_packed: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
            let mut b_packed: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
            let eq_lo = rng.f128_vec(lo_size);
            let row_base = 8;
            let pair_idx_base = 3 * lo_size;
            // Production shape: rows of masked (padded) pairs are ZERO in
            // memory (r1cs zero padding). The batch register arm folds cached
            // rows unconditionally and relies on that invariant; the scalar
            // reference zeroes those outputs explicitly — equal only on
            // production-shaped inputs.
            for pair in 0..lo_size {
                if ((pair_idx_base + pair) & mask) >= useful {
                    let r0 = (row_base + 2 * pair) * 8;
                    a_packed[r0..r0 + 16].fill(0);
                    b_packed[r0..r0 + 16].fill(0);
                }
            }

            let mut a_s = vec![F128::ZERO; 2 * lo_size];
            let mut b_s = vec![F128::ZERO; 2 * lo_size];
            let out_s = round2_lookahead_chunk_scalar::<true>(
                &a_packed,
                &b_packed,
                &table,
                &mut a_s,
                &mut b_s,
                &eq_lo,
                row_base,
                pair_idx_base,
                mask,
                useful,
            );
            let mut a_v = vec![F128::ONE; 2 * lo_size];
            let mut b_v = vec![F128::ONE; 2 * lo_size];
            // SAFETY: rows/table/chunk lengths satisfy the kernel's contract.
            let out_v = unsafe {
                kernels::x86_64::round2_lookahead_chunk_x86_avx512::<true, false>(
                    table.data.as_ptr(),
                    r2_mats_arg,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    row_base,
                    &mut a_v,
                    &mut b_v,
                    &eq_lo,
                    pair_idx_base,
                    mask,
                    useful,
                    None,
                    None,
                )
            };
            assert_eq!(a_s, a_v, "a chunk lo_size={lo_size} mask={mask}");
            assert_eq!(b_s, b_v, "b chunk lo_size={lo_size} mask={mask}");
            assert_eq!(out_s, out_v, "sums lo_size={lo_size} mask={mask}");
            // No-store variant: same sums, nothing written — and it
            // exercises the hoisted (w, w·x⁶⁴) table against the scalar
            // oracle whenever the shape allows one.
            let wtab_test = if lo_size % 8 == 0 {
                Some(build_w_pair_table(&eq_lo))
            } else {
                None
            };
            let mut a_e: Vec<F128> = Vec::new();
            let mut b_e: Vec<F128> = Vec::new();
            // SAFETY: as above; WRITE=false ignores the (empty) chunks.
            let out_n = unsafe {
                kernels::x86_64::round2_lookahead_chunk_x86_avx512::<false, false>(
                    table.data.as_ptr(),
                    r2_mats_arg,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    row_base,
                    &mut a_e,
                    &mut b_e,
                    &eq_lo,
                    pair_idx_base,
                    mask,
                    useful,
                    wtab_test.as_deref(),
                    None,
                )
            };
            assert_eq!(out_s, out_n, "no-store sums lo_size={lo_size} mask={mask}");
            let out_ns = round2_lookahead_chunk_scalar::<false>(
                &a_packed,
                &b_packed,
                &table,
                &mut a_e,
                &mut b_e,
                &eq_lo,
                row_base,
                pair_idx_base,
                mask,
                useful,
            );
            assert_eq!(
                out_s, out_ns,
                "scalar no-store sums lo_size={lo_size} mask={mask}"
            );
        }
    }

    /// Transpose every sixteen-element tile of `v` between the group-major and
    /// lane-major layouts (ZMM `k` of a tile = element `k` of each of its four
    /// groups). The permutation is an involution.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    fn lane_major_tiles(v: &[F128]) -> Vec<F128> {
        assert!(v.len().is_multiple_of(16));
        let mut out = v.to_vec();
        for t in (0..v.len()).step_by(16) {
            for g in 0..4 {
                for k in 0..4 {
                    out[t + 4 * k + g] = v[t + 4 * g + k];
                }
            }
        }
        out
    }

    /// Lane-major handoff across two plain cascade levels: the writer's
    /// lane-major tables are the row-major tables with every tile transposed,
    /// and the lane-major reader reproduces the row-major level's tables,
    /// message and lookahead from them.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn cascade_lane_major_handoff_matches_row_major() {
        if !zc_lane_major_handoff_on() {
            return;
        }
        for &log_n in &[16usize, 17, 18] {
            let mut rng = Rng::new(0x1A00 + log_n as u64);
            let n = 1usize << log_n;
            assert!(plain_cascade_lo_size(n).is_multiple_of(8), "log_n={log_n}");
            assert!(plain_cascade_lo_size(n / 4).is_multiple_of(8), "log_n={log_n}");
            let a = rng.f128_vec(n);
            let b = rng.f128_vec(n);
            let (rho_a, rho_b, rho_c, rho_d) = (rng.f128(), rng.f128(), rng.f128(), rng.f128());
            let mut r1 = vec![F128::ONE; log_n - 2];
            for v in r1[1..].iter_mut() {
                *v = rng.f128();
            }
            let mut r2 = vec![F128::ONE; log_n - 4];
            for v in r2[1..].iter_mut() {
                *v = rng.f128();
            }
            let probe = rng.f128();

            let mut a1 = vec![F128::ZERO; n / 4];
            let mut b1 = vec![F128::ZERO; n / 4];
            let (m1, mi1, la1) =
                fold2_plain_and_round_pair_lookahead_into(&a, &b, &mut a1, &mut b1, rho_a, rho_b, &r1);
            let mut a1l = vec![F128::ZERO; n / 4];
            let mut b1l = vec![F128::ZERO; n / 4];
            let (m1l, mi1l, la1l) = fold2_plain_and_round_pair_lookahead_into_lm(
                &a, &b, &mut a1l, &mut b1l, rho_a, rho_b, &r1, false, true,
            );
            assert_eq!((m1, mi1), (m1l, mi1l), "writer msg log_n={log_n}");
            assert_eq!(lane_major_tiles(&a1), a1l, "writer a log_n={log_n}");
            assert_eq!(lane_major_tiles(&b1), b1l, "writer b log_n={log_n}");
            assert_eq!(
                eval_round3_lookahead(&la1, probe),
                eval_round3_lookahead(&la1l, probe),
                "writer lookahead log_n={log_n}"
            );

            let mut a2 = vec![F128::ZERO; n / 16];
            let mut b2 = vec![F128::ZERO; n / 16];
            let (m2, mi2, la2) = fold2_plain_and_round_pair_lookahead_into(
                &a1, &b1, &mut a2, &mut b2, rho_c, rho_d, &r2,
            );
            for out_lm in [false, true] {
                let mut a2l = vec![F128::ZERO; n / 16];
                let mut b2l = vec![F128::ZERO; n / 16];
                let (m2l, mi2l, la2l) = fold2_plain_and_round_pair_lookahead_into_lm(
                    &a1l, &b1l, &mut a2l, &mut b2l, rho_c, rho_d, &r2, true, out_lm,
                );
                let (a_want, b_want) = if out_lm {
                    (lane_major_tiles(&a2), lane_major_tiles(&b2))
                } else {
                    (a2.clone(), b2.clone())
                };
                assert_eq!((m2, mi2), (m2l, mi2l), "reader msg log_n={log_n} out_lm={out_lm}");
                assert_eq!(a_want, a2l, "reader a log_n={log_n} out_lm={out_lm}");
                assert_eq!(b_want, b2l, "reader b log_n={log_n} out_lm={out_lm}");
                assert_eq!(
                    eval_round3_lookahead(&la2, probe),
                    eval_round3_lookahead(&la2l, probe),
                    "reader lookahead log_n={log_n} out_lm={out_lm}"
                );
            }
        }
    }

    /// AVX-512 composed-fold kernel vs the portable reference on one chunk.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold2_and_message_x86_matches_scalar() {
        for &lo_size in &[2usize, 4, 6, 8, 16, 64] {
            let mut rng = Rng::new(0x4D00 + lo_size as u64);
            let a_in = rng.f128_vec(8 * lo_size);
            let b_in = rng.f128_vec(8 * lo_size);
            let rho1 = rng.f128();
            let rho2 = rng.f128();
            let eq_lo = rng.f128_vec(lo_size);
            let mut a_s = vec![F128::ZERO; 2 * lo_size];
            let mut b_s = vec![F128::ZERO; 2 * lo_size];
            let (p1_s, pi_s) =
                fold2_and_message_scalar(&a_in, &b_in, &mut a_s, &mut b_s, rho1, rho2, &eq_lo);
            let mut a_v = vec![F128::ONE; 2 * lo_size];
            let mut b_v = vec![F128::ONE; 2 * lo_size];
            // SAFETY: lengths satisfy the kernel's contract.
            let (p1_v, pi_v) = unsafe {
                kernels::x86_64::fold2_and_message_x86_avx512(
                    &a_in, &b_in, &mut a_v, &mut b_v, rho1, rho2, &eq_lo,
                )
            };
            assert_eq!(a_s, a_v, "a lo_size={lo_size}");
            assert_eq!(b_s, b_v, "b lo_size={lo_size}");
            assert_eq!((p1_s, pi_s), (p1_v, pi_v), "msg lo_size={lo_size}");
        }
    }

    /// The cascade step (composed fold + message + next-round quadratic)
    /// emits tables and message identical to the plain composed pass, and its
    /// six coefficients reproduce the incumbent next-round message
    /// (fold then `round_pair_naive`) at several challenges.
    #[test]
    fn cascade_lookahead_matches_incumbent() {
        for &log_n in &[4usize, 5, 6, 8, 10, 13] {
            let mut rng = Rng::new(0x5E00 + log_n as u64);
            let n = 1usize << log_n;
            let a = rng.f128_vec(n);
            let b = rng.f128_vec(n);
            let rho_a = rng.f128();
            let rho_b = rng.f128();
            let mut r_next = vec![F128::ONE; log_n - 2];
            for v in r_next[1..].iter_mut() {
                *v = rng.f128();
            }
            assert_ne!(r_next[1], F128::ZERO);
            let mut a_ref = vec![F128::ZERO; n / 4];
            let mut b_ref = vec![F128::ZERO; n / 4];
            let (m1_ref, mi_ref) =
                fold2_plain_and_round4_into(&a, &b, &mut a_ref, &mut b_ref, rho_a, rho_b, &r_next);
            let poison = F128 {
                lo: 0xDEAD_BEEF_DEAD_BEEF,
                hi: 0xFEED_FACE_FEED_FACE,
            };
            let mut a_la = vec![poison; n / 4];
            let mut b_la = vec![poison; n / 4];
            let (m1_la, mi_la, la) = fold2_plain_and_round_pair_lookahead_into(
                &a, &b, &mut a_la, &mut b_la, rho_a, rho_b, &r_next,
            );
            assert_eq!(a_ref, a_la, "a tables log_n={log_n}");
            assert_eq!(b_ref, b_la, "b tables log_n={log_n}");
            assert_eq!((m1_ref, mi_ref), (m1_la, mi_la), "msg log_n={log_n}");

            let mut r_nn = vec![F128::ONE; log_n - 3];
            r_nn[1..].copy_from_slice(&r_next[2..]);
            for &rho in &[F128::ZERO, F128::ONE, rng.f128(), rng.f128()] {
                let (mut a5, mut b5) = (a_ref.clone(), b_ref.clone());
                fold_in_place_pair(&mut a5, &mut b5, rho);
                let (m1, mi) = round_pair_naive(&a5, &b5, &r_nn);
                assert_eq!(
                    eval_round3_lookahead(&la, rho),
                    (m1, mi),
                    "next msg log_n={log_n} rho={rho:?}"
                );
            }
        }
    }

    /// AVX-512 cascade kernel vs the portable reference on one chunk.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold2_and_message_lookahead_x86_matches_scalar() {
        for &lo_size in &[2usize, 4, 6, 8, 10, 16, 24, 64] {
            let mut rng = Rng::new(0x6F00 + lo_size as u64);
            let a_in = rng.f128_vec(8 * lo_size);
            let b_in = rng.f128_vec(8 * lo_size);
            let rho_a = rng.f128();
            let rho_b = rng.f128();
            let eq_lo = rng.f128_vec(lo_size);
            let mut a_s = vec![F128::ZERO; 2 * lo_size];
            let mut b_s = vec![F128::ZERO; 2 * lo_size];
            let out_s = fold2_and_message_lookahead_scalar(
                &a_in, &b_in, &mut a_s, &mut b_s, rho_a, rho_b, &eq_lo,
            );
            let mut a_v = vec![F128::ONE; 2 * lo_size];
            let mut b_v = vec![F128::ONE; 2 * lo_size];
            // SAFETY: lengths satisfy the kernel's contract.
            let out_v = unsafe {
                kernels::x86_64::fold2_and_message_lookahead_x86_avx512(
                    &a_in, &b_in, &mut a_v, &mut b_v, rho_a, rho_b, &eq_lo, None, false, false,
                    false, false,
                )
            };
            assert_eq!(a_s, a_v, "a lo_size={lo_size}");
            assert_eq!(b_s, b_v, "b lo_size={lo_size}");
            assert_eq!(out_s, out_v, "sums lo_size={lo_size}");

            if lo_size.is_multiple_of(8) {
                let wtab = build_w_pair_table(&eq_lo);
                let mut a_w = vec![F128::ONE; 2 * lo_size];
                let mut b_w = vec![F128::ONE; 2 * lo_size];
                // SAFETY: same checked geometry; `wtab` comes from this
                // invocation's exact `eq_lo` slice.
                let out_w = unsafe {
                    kernels::x86_64::fold2_and_message_lookahead_x86_avx512(
                        &a_in,
                        &b_in,
                        &mut a_w,
                        &mut b_w,
                        rho_a,
                        rho_b,
                        &eq_lo,
                        Some(&wtab),
                        false,
                        false,
                        false,
                        false,
                    )
                };
                assert_eq!(a_s, a_w, "wtab a lo_size={lo_size}");
                assert_eq!(b_s, b_w, "wtab b lo_size={lo_size}");
                assert_eq!(out_s, out_w, "wtab sums lo_size={lo_size}");

                // Lane-major input/output tiles and the split-companion
                // deferred fold, every combination against the same oracle
                // (layouts converted tile by tile).
                let a_in_lm = lane_major_tiles(&a_in);
                let b_in_lm = lane_major_tiles(&b_in);
                for in_lm in [false, true] {
                    for out_lm in [false, true] {
                        for split in [false, true] {
                            for use_wtab in [false, true] {
                                let (ai, bi) = if in_lm {
                                    (&a_in_lm, &b_in_lm)
                                } else {
                                    (&a_in, &b_in)
                                };
                                let mut a_c = vec![F128::ONE; 2 * lo_size];
                                let mut b_c = vec![F128::ONE; 2 * lo_size];
                                // SAFETY: same checked geometry; `lo_size` is
                                // a multiple of eight, as both layouts need.
                                let out_c = unsafe {
                                    kernels::x86_64::fold2_and_message_lookahead_x86_avx512(
                                        ai,
                                        bi,
                                        &mut a_c,
                                        &mut b_c,
                                        rho_a,
                                        rho_b,
                                        &eq_lo,
                                        use_wtab.then_some(&wtab[..]),
                                        false,
                                        in_lm,
                                        out_lm,
                                        split,
                                    )
                                };
                                let (a_want, b_want) = if out_lm {
                                    (lane_major_tiles(&a_s), lane_major_tiles(&b_s))
                                } else {
                                    (a_s.clone(), b_s.clone())
                                };
                                let tag = format!(
                                    "lo_size={lo_size} in_lm={in_lm} out_lm={out_lm} split={split} wtab={use_wtab}"
                                );
                                assert_eq!(a_want, a_c, "a {tag}");
                                assert_eq!(b_want, b_c, "b {tag}");
                                assert_eq!(out_s, out_c, "sums {tag}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// Local cost-model probe for the exact ranked cascade geometry. This does
    /// not claim Sapphire Rapids kernel timing: it measures the portable work
    /// ledger on hosts where AVX-512 execution is unavailable. Run explicitly
    /// with `--ignored --nocapture` in an optimized test build.
    #[test]
    #[ignore]
    fn w_pair_table_hoist_cost_model() {
        use std::hint::black_box;
        use std::time::Instant;

        const LO_SIZE: usize = 1 << 10;
        const HI_SIZE: usize = 1 << 11;
        const SAMPLES: usize = 9;
        let eq_lo: Vec<F128> = (0..LO_SIZE)
            .map(|i| F128::new(i as u64 * 0x9E37_79B9 + 1, (i as u64).rotate_left(29) + 3))
            .collect();
        let x64 = F128::new(0, 1);

        let incumbent = || {
            let mut acc = F128::ZERO;
            for _ in 0..HI_SIZE {
                for g in 0..LO_SIZE / 8 {
                    for k in 0..4 {
                        let w = black_box(eq_lo[8 * g + 2 * k + 1]);
                        acc += w;
                        acc += black_box(w * x64);
                    }
                }
            }
            black_box(acc)
        };
        let candidate = || {
            let mut t = crate::alloc_uninit_f128_vec(LO_SIZE);
            for g in 0..LO_SIZE / 8 {
                for k in 0..4 {
                    let w = black_box(eq_lo[8 * g + 2 * k + 1]);
                    t[8 * g + k] = w;
                    t[8 * g + 4 + k] = black_box(w * x64);
                }
            }
            let mut acc = F128::ZERO;
            for _ in 0..HI_SIZE {
                for &v in &t {
                    acc += black_box(v);
                }
            }
            black_box(acc)
        };

        black_box(incumbent());
        black_box(candidate());
        let mut incumbent_ns = Vec::with_capacity(SAMPLES);
        let mut candidate_ns = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            if sample % 2 == 0 {
                let t0 = Instant::now();
                black_box(incumbent());
                incumbent_ns.push(t0.elapsed().as_nanos());
                let t0 = Instant::now();
                black_box(candidate());
                candidate_ns.push(t0.elapsed().as_nanos());
            } else {
                let t0 = Instant::now();
                black_box(candidate());
                candidate_ns.push(t0.elapsed().as_nanos());
                let t0 = Instant::now();
                black_box(incumbent());
                incumbent_ns.push(t0.elapsed().as_nanos());
            }
        }
        eprintln!("incumbent_ns={incumbent_ns:?}");
        eprintln!("candidate_ns={candidate_ns:?}");
        assert_eq!(black_box(incumbent()), black_box(candidate()));
    }

    /// The no-materialize route equals the materializing lookahead route:
    /// same round-2 message and coefficients from the store-free sweep, and
    /// the packed→composed pass reproduces the materialized composed tables
    /// (POISON-prefilled outputs), round-4 message and round-5 coefficients
    /// — dense and padded.
    #[test]
    fn nomat_route_matches_materialized() {
        const K_SKIP: usize = 6;
        for &(m, padded) in &[
            (13usize, false),
            (14, true),
            (15, true),
            (16, false),
            (17, true),
            (18, true),
        ] {
            let mut rng = Rng::new(0x7A00 + m as u64 + padded as u64 * 100);
            let (a_packed, b_packed, padding) = lookahead_witness(&mut rng, m, padded);
            let z = rng.f128();
            let table = UniSkipFoldTable::new(K_SKIP, z);
            let mut mlv = rng.f128_vec(m - K_SKIP);
            mlv[0] = F128::ONE;
            let (a2, b2, m1_ref, mi_ref, la_ref) =
                uni_skip_fold_and_round_pair_optimized_packed_padded_lookahead(
                    &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
                );
            let (m1_nm, mi_nm, la_nm) = uni_skip_round_pair_lookahead_nomat_packed_padded(
                &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
            );
            assert_eq!(
                (m1_ref, mi_ref),
                (m1_nm, mi_nm),
                "round-2 msg m={m} padded={padded}"
            );
            assert_eq!(la_ref, la_nm, "round-3 coeffs m={m} padded={padded}");

            let rho1 = rng.f128();
            let rho2 = rng.f128();
            let mut r_next4 = vec![F128::ONE; m - K_SKIP - 2];
            r_next4[1..].copy_from_slice(&mlv[3..]);
            let n = 1usize << (m - K_SKIP);
            let mut a4 = vec![F128::ZERO; n / 4];
            let mut b4 = vec![F128::ZERO; n / 4];
            let (m4_ref, mi4_ref, la5_ref) = fold2_plain_and_round_pair_lookahead_into(
                &a2, &b2, &mut a4, &mut b4, rho1, rho2, &r_next4,
            );
            let poison = F128 {
                lo: 0xDEAD_BEEF_DEAD_BEEF,
                hi: 0xFEED_FACE_FEED_FACE,
            };
            let mut a4n = vec![poison; n / 4];
            let mut b4n = vec![poison; n / 4];
            let (m4_nm, mi4_nm, la5_nm) = fold2_from_packed_and_round_pair_lookahead_into(
                &a_packed, &b_packed, m, K_SKIP, &table, &padding, &mut a4n, &mut b4n, rho1, rho2,
                &r_next4,
            );
            assert_eq!(a4, a4n, "a4 m={m} padded={padded}");
            assert_eq!(b4, b4n, "b4 m={m} padded={padded}");
            assert_eq!(
                (m4_ref, mi4_ref),
                (m4_nm, mi4_nm),
                "round-4 msg m={m} padded={padded}"
            );
            assert_eq!(la5_ref, la5_nm, "round-5 coeffs m={m} padded={padded}");
        }
    }

    #[test]
    fn duplicate_inv_elide_kill_switch_parser() {
        use std::ffi::OsStr;

        assert!(dup_inv_elide_disabled_value(Some(OsStr::new("1"))));
        for value in [
            None,
            Some(OsStr::new("")),
            Some(OsStr::new("0")),
            Some(OsStr::new("01")),
            Some(OsStr::new("true")),
        ] {
            assert!(!dup_inv_elide_disabled_value(value));
        }
    }

    /// Exercise the incumbent two-inversion arm and the shared-inverse arm on
    /// each production-ranked caller: no-materialize round two, packed level
    /// zero, and the plain cascade. Every table, message and lookahead
    /// coefficient must remain byte-for-byte identical.
    #[test]
    fn duplicate_inv_elide_plain_packed_nomat_exact_oracle() {
        const K_SKIP: usize = 6;
        let m = 16usize;
        let mut rng = Rng::new(0xD09_1A5E);
        let (a_packed, b_packed, padding) = lookahead_witness(&mut rng, m, true);
        let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
        let mut mlv = rng.f128_vec(m - K_SKIP);
        mlv[0] = F128::ONE;
        if mlv[1] == F128::ZERO {
            mlv[1] = F128::ONE;
        }

        // Materialize once only to supply the plain cascade's honest input;
        // this fallback-only producer deliberately retains its two inversions.
        let (a2, b2, _, _, _) = uni_skip_fold_and_round_pair_optimized_packed_padded_lookahead(
            &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
        );
        let n = 1usize << (m - K_SKIP);
        let rho1 = rng.f128();
        let rho2 = rng.f128();
        let mut r_next4 = vec![F128::ONE; m - K_SKIP - 2];
        r_next4[1..].copy_from_slice(&mlv[3..]);
        if r_next4[1] == F128::ZERO {
            r_next4[1] = F128::ONE;
        }

        DUP_INV_ELIDE_TEST_OVERRIDE.with(|slot| slot.set(Some(false)));
        let nomat_incumbent = uni_skip_round_pair_lookahead_nomat_packed_padded(
            &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
        );
        let mut packed_a_incumbent = vec![F128::ZERO; n / 4];
        let mut packed_b_incumbent = vec![F128::ZERO; n / 4];
        let packed_incumbent = fold2_from_packed_and_round_pair_lookahead_into(
            &a_packed,
            &b_packed,
            m,
            K_SKIP,
            &table,
            &padding,
            &mut packed_a_incumbent,
            &mut packed_b_incumbent,
            rho1,
            rho2,
            &r_next4,
        );
        let mut plain_a_incumbent = vec![F128::ZERO; n / 4];
        let mut plain_b_incumbent = vec![F128::ZERO; n / 4];
        let plain_incumbent = fold2_plain_and_round_pair_lookahead_into(
            &a2,
            &b2,
            &mut plain_a_incumbent,
            &mut plain_b_incumbent,
            rho1,
            rho2,
            &r_next4,
        );

        DUP_INV_ELIDE_TEST_OVERRIDE.with(|slot| slot.set(Some(true)));
        let nomat_shared = uni_skip_round_pair_lookahead_nomat_packed_padded(
            &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
        );
        let mut packed_a_shared = vec![F128::ZERO; n / 4];
        let mut packed_b_shared = vec![F128::ZERO; n / 4];
        let packed_shared = fold2_from_packed_and_round_pair_lookahead_into(
            &a_packed,
            &b_packed,
            m,
            K_SKIP,
            &table,
            &padding,
            &mut packed_a_shared,
            &mut packed_b_shared,
            rho1,
            rho2,
            &r_next4,
        );
        let mut plain_a_shared = vec![F128::ZERO; n / 4];
        let mut plain_b_shared = vec![F128::ZERO; n / 4];
        let plain_shared = fold2_plain_and_round_pair_lookahead_into(
            &a2,
            &b2,
            &mut plain_a_shared,
            &mut plain_b_shared,
            rho1,
            rho2,
            &r_next4,
        );
        DUP_INV_ELIDE_TEST_OVERRIDE.with(|slot| slot.set(None));

        assert_eq!(nomat_shared, nomat_incumbent, "nomat round-two output");
        assert_eq!(packed_a_shared, packed_a_incumbent, "packed a output");
        assert_eq!(packed_b_shared, packed_b_incumbent, "packed b output");
        assert_eq!(packed_shared, packed_incumbent, "packed message/lookahead");
        assert_eq!(plain_a_shared, plain_a_incumbent, "plain a output");
        assert_eq!(plain_b_shared, plain_b_incumbent, "plain b output");
        assert_eq!(plain_shared, plain_incumbent, "plain message/lookahead");
    }


    /// Pins the active NT/lane-major route and 64-byte output alignment.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn c4_register_return_callers_match_scalar() {
        #[repr(C, align(64))]
        struct Aligned([F128; 4]);
        for lo_size in [16usize, 64, 256, 1024] {
            let mut rng = Rng::new(0x7c4011 + lo_size as u64);
            let table = UniSkipFoldTable::new(6, rng.f128());
            let out_base = 4 * lo_size;
            let n_rows = 8 * (out_base + 2 * lo_size);
            let ap: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
            let bp: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
            let rho1 = rng.f128();
            let rho2 = rng.f128();
            let one = F128::ONE;
            let eq = rng.f128_vec(lo_size);
            let wt = build_w_pair_table(&eq);
            let cfold = kernels::x86_64::build_cfold_mats(
                &table.data,
                [
                    (one + rho1) * (one + rho2),
                    rho1 * (one + rho2),
                    (one + rho1) * rho2,
                    rho1 * rho2,
                ],
            );
            let mut a_s = vec![F128::ZERO; 2 * lo_size];
            let mut b_s = a_s.clone();
            let want = fold2_from_packed_lookahead_scalar(
                &ap,
                &bp,
                &table,
                out_base,
                &mut a_s,
                &mut b_s,
                rho1,
                rho2,
                &eq,
                0,
                usize::MAX,
            );
            macro_rules! check {
                ($nt:literal,$lm:literal) => {{
                    let mut aa: Vec<Aligned> =
                        (0..lo_size / 2 + 2).map(|_| Aligned([one; 4])).collect();
                    let mut bb: Vec<Aligned> =
                        (0..lo_size / 2 + 2).map(|_| Aligned([one; 4])).collect();
                    let a = unsafe {
                        core::slice::from_raw_parts_mut(
                            aa.as_mut_ptr().add(1).cast::<F128>(),
                            2 * lo_size,
                        )
                    };
                    let b = unsafe {
                        core::slice::from_raw_parts_mut(
                            bb.as_mut_ptr().add(1).cast::<F128>(),
                            2 * lo_size,
                        )
                    };
                    assert_eq!(a.as_ptr() as usize % 64, 0);
                    assert_eq!(b.as_ptr() as usize % 64, 0);
                    let got = unsafe {
                        kernels::x86_64::fold2_from_packed_lookahead_c4_lm_wtab_x86::<$nt, $lm>(
                            ap.as_ptr(),
                            bp.as_ptr(),
                            out_base,
                            a,
                            b,
                            lo_size,
                            &cfold,
                            &wt,
                        )
                    };
                    assert_eq!(got, want, "message lo={lo_size}");
                    assert_eq!(
                        a,
                        if $lm {
                            lane_major_tiles(&a_s)
                        } else {
                            a_s.clone()
                        }
                    );
                    assert_eq!(
                        b,
                        if $lm {
                            lane_major_tiles(&b_s)
                        } else {
                            b_s.clone()
                        }
                    );
                    assert_eq!(aa.first().unwrap().0, [one; 4]);
                    assert_eq!(aa.last().unwrap().0, [one; 4]);
                    assert_eq!(bb.first().unwrap().0, [one; 4]);
                    assert_eq!(bb.last().unwrap().0, [one; 4]);
                }};
            }
            check!(false, false);
            check!(false, true);
            check!(true, false);
            check!(true, true);
        }
        println!(
            "16 actual C4 caller size/layout/publication cases match scalar messages and outputs with guards"
        );
    }

    /// AVX-512 packed→composed kernel vs the portable reference on one chunk,
    /// with and without padded pairs, at several `lo_size`.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold2_from_packed_lookahead_x86_matches_scalar() {
        const K_SKIP: usize = 6;
        for &(lo_size, mask, useful) in &[
            (2usize, 0usize, usize::MAX),
            (4, 0, usize::MAX),
            (6, 0, usize::MAX),
            (8, 0, usize::MAX),
            // 10: main loop AND the regular-store tail in the SAME call, so
            // the nt arm's mixed NT/plain stores to one buffer are covered.
            (10, 0, usize::MAX),
            (16, 0, usize::MAX),
            (64, 0, usize::MAX),
            (16, 7, 5),
            (32, 15, 12),
            (64, 127, 121),
        ] {
            let mut rng = Rng::new(0x8B00 + lo_size as u64 + mask as u64);
            let table = UniSkipFoldTable::new(K_SKIP, rng.f128());
            #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
            let r2_mats = r2_gfni_mats(&table);
            #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
            let r2_mats_arg = r2_mats.as_ref();
            #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
            let r2_mats_arg: Option<&[u64; 128]> = None;
            let out_base = 4 * lo_size; // exercise a non-zero chunk offset
            let n_rows = 8 * (out_base + 2 * lo_size);
            let mut a_packed: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
            let mut b_packed: Vec<u8> = (0..n_rows * 8).map(|_| rng.next_u64() as u8).collect();
            // Production shape: masked (padded) pairs' packed rows are zero
            // in memory; the batch register arm folds cached rows
            // unconditionally and relies on that invariant.
            for x0 in out_base..out_base + 2 * lo_size {
                for p in 0..2usize {
                    let pair = 2 * x0 + p;
                    if (pair & mask) >= useful {
                        let r0 = 2 * pair * 8;
                        a_packed[r0..r0 + 16].fill(0);
                        b_packed[r0..r0 + 16].fill(0);
                    }
                }
            }
            let rho1 = rng.f128();
            let rho2 = rng.f128();
            let eq_lo = rng.f128_vec(lo_size);
            let wtab_test = if lo_size % 8 == 0 {
                Some(build_w_pair_table(&eq_lo))
            } else {
                None
            };
            let mut a_s = vec![F128::ZERO; 2 * lo_size];
            let mut b_s = vec![F128::ZERO; 2 * lo_size];
            let out_s = fold2_from_packed_lookahead_scalar(
                &a_packed, &b_packed, &table, out_base, &mut a_s, &mut b_s, rho1, rho2, &eq_lo,
                mask, useful,
            );
            // Both composed-fold routes: the incumbent (ρ₁, ρ₂) multiplies
            // and the baked-coefficient batch, each against the same oracle.
            #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
            let cfold_built = {
                let one = F128::ONE;
                Some(kernels::x86_64::build_cfold_mats(
                    &table.data,
                    [
                        (one + rho1) * (one + rho2),
                        rho1 * (one + rho2),
                        (one + rho1) * rho2,
                        rho1 * rho2,
                    ],
                ))
            };
            #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "gfni")))]
            let cfold_built: Option<kernels::x86_64::CFoldMats> = None;
            let out_lm_arms: &[bool] = if lo_size % 8 == 0 {
                &[false, true]
            } else {
                &[false]
            };
            for cfold_arg in [None, cfold_built.as_ref()] {
                for (nt_out, c4_lm, out_lm) in [false, true]
                    .into_iter()
                    .flat_map(|nt| [false, true].into_iter().map(move |c4| (nt, c4)))
                    .flat_map(|(nt, c4)| out_lm_arms.iter().map(move |&lm| (nt, c4, lm)))
                {
                    let mut a_v = vec![F128::ONE; 2 * lo_size];
                    let mut b_v = vec![F128::ONE; 2 * lo_size];
                    // SAFETY: rows/table/output lengths satisfy the kernel's
                    // contract; `F128` is `repr(C, align(16))`, so every
                    // `Vec<F128>` base is 16-byte aligned by the allocation
                    // layout — the nt arm's requirement is a language guarantee.
                    let out_v = unsafe {
                        kernels::x86_64::fold2_from_packed_lookahead_x86_avx512(
                            table.data.as_ptr(),
                            r2_mats_arg,
                            a_packed.as_ptr(),
                            b_packed.as_ptr(),
                            out_base,
                            &mut a_v,
                            &mut b_v,
                            rho1,
                            rho2,
                            &eq_lo,
                            mask,
                            useful,
                            nt_out,
                            cfold_arg,
                            wtab_test.as_deref(),
                            c4_lm,
                            out_lm,
                        )
                    };
                    let baked = cfold_arg.is_some();
                    let (a_want, b_want) = if out_lm {
                        (lane_major_tiles(&a_s), lane_major_tiles(&b_s))
                    } else {
                        (a_s.clone(), b_s.clone())
                    };
                    let tag = format!(
                        "lo_size={lo_size} mask={mask} nt={nt_out} baked={baked} c4_lm={c4_lm} out_lm={out_lm}"
                    );
                    assert_eq!(a_want, a_v, "a {tag}");
                    assert_eq!(b_want, b_v, "b {tag}");
                    assert_eq!(out_s, out_v, "sums {tag}");
                }
            }
        }
    }

    // ----------------------------------------------------------------------
    // GFNI prefold dead-line skip.
    // ----------------------------------------------------------------------

    /// The ranked BLAKE3 padding shape as `R1CS::padding_spec` (RowMajor)
    /// hands it to the zerocheck: `2^14`-bit blocks, 15409 useful bits each.
    fn ranked_padding() -> PaddingSpec {
        PaddingSpec {
            k_log: 14,
            useful_bits_per_block: 15_409,
        }
    }

    /// The derived dead set at the ranked shape, end to end: 15 provably-zero
    /// post-URM rows per 256-row block (241..=255), of which exactly one
    /// aligned 8-row group — one 64-byte line, rows 248..=255 — is fully dead.
    #[test]
    fn prefold_ranked_dead_set_is_one_line_of_thirty_two() {
        let padding = ranked_padding();
        let k_skip = 6;

        // Exact derivation from the PaddingSpec.
        let (row_in_block_mask, first_dead_row) = round2_row_zero(&padding, k_skip).unwrap();
        assert_eq!(row_in_block_mask, 255, "2^(14-6) = 256 post-URM rows/block");
        assert_eq!(first_dead_row, 241, "ceil(15409 / 64)");
        assert_eq!(256 - first_dead_row, 15, "15 dead rows per block");
        // Row 240 straddles the useful/padding boundary; row 241 is the first
        // whose whole 64-bit chunk sits past `useful_bits_per_block`.
        assert!(240 * 64 < padding.useful_bits_per_block);
        assert!(241 * 64 >= padding.useful_bits_per_block);

        // The pair predicate the kernels already carry.
        let (pair_mask, useful_pairs) = round2_pair_skip(&padding, k_skip);
        assert_eq!((pair_mask, useful_pairs), (127, 121));

        // Line plan: only the block's last 64-row tile has a dead line, and
        // only its last one. 1 of the block's 32 lines = 3.125% of the
        // prefold's packed-row read volume.
        let masks: Vec<u8> = (0..4)
            .map(|t| prefold_dead_line_mask(64 * t, pair_mask, useful_pairs))
            .collect();
        assert_eq!(masks, vec![0, 0, 0, 0b1000_0000]);
        let dead_lines: u32 = masks.iter().map(|m| m.count_ones()).sum();
        assert_eq!(dead_lines, 1, "1 dead 64-byte line of the block's 32");

        // It is the block position that decides, not the absolute row.
        for block in 0..5usize {
            for t in 0..4usize {
                let base = 256 * block + 64 * t;
                let want = if t == 3 { 0b1000_0000 } else { 0 };
                assert_eq!(
                    prefold_dead_line_mask(base, pair_mask, useful_pairs),
                    want,
                    "base={base}"
                );
            }
        }
    }

    /// At 64-byte-line granularity the pair-conservative plan the hot path
    /// uses is **identical** to the exact `useful_bits` derivation, for every
    /// padding shape — so the skip needs no premise beyond the pair predicate
    /// `round2_pair_skip` already established.
    ///
    /// Proof (checked exhaustively over `first_dead_row` below): with
    /// `f = ceil(useful_bits / 2^k_skip)` the exact first dead row, the pair
    /// form uses `2·ceil(f/2)`, which is `f` when `f` is even and `f + 1` when
    /// `f` is odd. A line start `q` is a multiple of 8, hence even, so
    /// `q >= f` and `q >= f + 1` are the same condition when `f` is odd.
    #[test]
    fn prefold_line_plan_matches_exact_padding_derivation() {
        for k_log in 7..=16usize {
            for k_skip in [3usize, 4, 5, 6] {
                if k_log <= k_skip + 1 {
                    continue;
                }
                let chunk_bits = 1usize << k_skip;
                let rows_per_block = 1usize << (k_log - k_skip);
                for f in 1..=rows_per_block {
                    for useful in [(f - 1) * chunk_bits + 1, f * chunk_bits] {
                        let padding = PaddingSpec {
                            k_log,
                            useful_bits_per_block: useful,
                        };
                        assert_eq!(
                            round2_row_zero(&padding, k_skip).map(|(_, r)| r),
                            (f < rows_per_block).then_some(f),
                            "k_log={k_log} k_skip={k_skip} useful={useful}"
                        );
                        let (pair_mask, useful_pairs) = round2_pair_skip(&padding, k_skip);
                        for t in 0..rows_per_block.div_ceil(64) {
                            let base = 64 * t;
                            let got = prefold_dead_line_mask(base, pair_mask, useful_pairs);
                            let want = if !rows_per_block.is_multiple_of(64) {
                                // Tiles do not nest inside a block: refused.
                                0
                            } else {
                                let mut m = 0u8;
                                for i in 0..8 {
                                    if f < rows_per_block && base + 8 * i >= f {
                                        m |= 1 << i;
                                    }
                                }
                                m
                            };
                            assert_eq!(
                                got, want,
                                "k_log={k_log} k_skip={k_skip} useful={useful} tile={t}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Every premise the plan needs is re-checked at runtime; a violation
    /// returns "load every line", never a wrong skip.
    #[test]
    fn prefold_line_plan_refuses_off_shape() {
        // Dense padding: `round2_pair_skip`'s "nothing is dead" sentinel.
        let (pm, up) = round2_pair_skip(&PaddingSpec::dense(32), 6);
        assert_eq!((pm, up), (0, usize::MAX));
        assert_eq!(prefold_dead_line_mask(0, pm, up), 0);
        assert_eq!(prefold_dead_line_mask(192, pm, up), 0);
        assert_eq!(round2_row_zero(&PaddingSpec::dense(32), 6), None);

        // Sentinel and degenerate inputs.
        assert_eq!(prefold_dead_line_mask(192, 127, usize::MAX), 0);
        assert_eq!(prefold_dead_line_mask(192, 0, 121), 0);
        // Non-power-of-two "block": not a block-position mask.
        assert_eq!(prefold_dead_line_mask(192, 100, 50), 0);
        // Block shorter than one tile (8 rows): tiles cannot nest.
        assert_eq!(prefold_dead_line_mask(0, 3, 2), 0);
        // Tile base not 64-row aligned: the block-local start is unusable.
        assert_eq!(prefold_dead_line_mask(200, 127, 121), 0);
        // Nothing dead at pair granularity.
        assert_eq!(prefold_dead_line_mask(192, 127, 128), 0);
        // Dead rows exist but no aligned group of 8 is fully dead.
        assert_eq!(prefold_dead_line_mask(192, 127, 126), 0);
    }

    /// The kill switch gates the plan the kernels consume, and nothing else:
    /// the predicate itself is pure, so every proof above holds in either
    /// switch state. Deterministic whichever way the suite is run.
    #[test]
    fn prefold_kill_switch_gates_the_plan() {
        let (pair_mask, useful_pairs) = round2_pair_skip(&ranked_padding(), 6);
        let pure = prefold_dead_line_mask(192, pair_mask, useful_pairs);
        assert_eq!(pure, 0b1000_0000);
        let want = if prefold_row_skip_enabled() { pure } else { 0 };
        assert_eq!(
            prefold_dead_line_mask_gated(192, pair_mask, useful_pairs),
            want
        );
        // Off-shape stays off-shape through the gate.
        assert_eq!(
            prefold_dead_line_mask_gated(200, pair_mask, useful_pairs),
            0
        );
    }

    /// Scalar model of one 64-row GFNI prefold tile applying the same
    /// dead-line predicate as `gfni_fold64_rows_masked`: a dead line yields
    /// `fold_one_row(0) = 0`, every other row the true table fold.
    fn prefold_tile_scalar(
        packed: &[u8],
        table: &UniSkipFoldTable,
        tile_base_row: usize,
        dead_lines: u8,
    ) -> Vec<F128> {
        let nc = table.n_chunks;
        (0..64)
            .map(|i| {
                if dead_lines & (1u8 << (i / 8)) != 0 {
                    return F128::ZERO;
                }
                let r = tile_base_row + i;
                table.fold_one_row(&packed[r * nc..(r + 1) * nc])
            })
            .collect()
    }

    /// Zero-propagation with teeth: garbage poured into the rows the plan
    /// declares dead is **invisible** with the skip ON (the tile is identical
    /// to the unpredicated fold of the genuinely-zero buffer) and **visible**
    /// with the skip OFF.
    #[test]
    fn prefold_dead_line_absorbs_poison() {
        let mut rng = Rng::new(0xF01D_5EED);
        let k_skip = 6;
        let table = UniSkipFoldTable::new(k_skip, rng.f128());
        let (pair_mask, useful_pairs) = round2_pair_skip(&ranked_padding(), k_skip);
        let nc = table.n_chunks;

        // One padding block: rows 0..241 carry witness bits, 241..=255 are the
        // zero tail the padding spec guarantees.
        let mut clean = vec![0u8; 256 * nc];
        for byte in clean[..241 * nc].iter_mut() {
            *byte = rng.next_u64() as u8;
        }
        let mut poisoned = clean.clone();
        for (i, byte) in poisoned[248 * nc..].iter_mut().enumerate() {
            *byte = 0xA5 ^ (i as u8);
        }

        let dead = prefold_dead_line_mask(192, pair_mask, useful_pairs);
        assert_eq!(dead, 0b1000_0000);

        let reference = prefold_tile_scalar(&clean, &table, 192, 0);
        assert!(
            reference[49..].iter().all(|v| *v == F128::ZERO),
            "rows 241..=255 fold to zero: T_j[0] = 0 in every byte table"
        );

        let on = prefold_tile_scalar(&poisoned, &table, 192, dead);
        assert_eq!(on, reference, "poison must be invisible with the skip ON");

        let off = prefold_tile_scalar(&poisoned, &table, 192, 0);
        assert_ne!(off, reference, "poison must be visible with the skip OFF");
    }

    /// Byte oracle for the GFNI output-plane split.  It models the complete
    /// 16-plane -> two qword transposes -> byte transpose -> lo/hi interleave
    /// reassembly and compares the explicit constant half schedule against
    /// the former full-array schedule across every one of the 1,024 bytes.
    #[test]
    fn gfni_split_plane_halves_preserve_all_output_bytes() {
        fn plane(seed: u8, k: usize) -> [u8; 64] {
            std::array::from_fn(|i| {
                seed.wrapping_add((k as u8).wrapping_mul(67))
                    .rotate_left((i & 7) as u32)
                    ^ (i as u8).wrapping_mul(29)
            })
        }

        fn qword_transpose(input: [[u8; 64]; 8]) -> [[u8; 64]; 8] {
            let mut out = [[0u8; 64]; 8];
            for src in 0..8 {
                for qword in 0..8 {
                    out[qword][8 * src..8 * src + 8]
                        .copy_from_slice(&input[src][8 * qword..8 * qword + 8]);
                }
            }
            out
        }

        fn byte_transpose(input: [u8; 64]) -> [u8; 64] {
            std::array::from_fn(|i| input[8 * (i & 7) + (i >> 3)])
        }

        fn reassemble(lo: [[u8; 64]; 8], hi: [[u8; 64]; 8]) -> [u8; 1024] {
            let lo = qword_transpose(lo);
            let hi = qword_transpose(hi);
            let mut out = [0u8; 1024];
            for row in 0..8 {
                let l = byte_transpose(lo[row]);
                let h = byte_transpose(hi[row]);
                for qword in 0..8 {
                    let dst = 128 * row + 16 * qword;
                    out[dst..dst + 8].copy_from_slice(&l[8 * qword..8 * qword + 8]);
                    out[dst + 8..dst + 16].copy_from_slice(&h[8 * qword..8 * qword + 8]);
                }
            }
            out
        }

        for seed in [0u8, 1, 0x5A, 0xA5, 0xFF] {
            let full: [[u8; 64]; 16] = std::array::from_fn(|k| plane(seed, k));
            let want = reassemble(full[..8].try_into().unwrap(), full[8..].try_into().unwrap());
            let got = reassemble(
                [
                    plane(seed, 0),
                    plane(seed, 1),
                    plane(seed, 2),
                    plane(seed, 3),
                    plane(seed, 4),
                    plane(seed, 5),
                    plane(seed, 6),
                    plane(seed, 7),
                ],
                [
                    plane(seed, 8),
                    plane(seed, 9),
                    plane(seed, 10),
                    plane(seed, 11),
                    plane(seed, 12),
                    plane(seed, 13),
                    plane(seed, 14),
                    plane(seed, 15),
                ],
            );
            assert_eq!(got, want, "all output bytes, seed={seed:#04x}");
        }
    }

    /// The c4 consumer XORs four residue chunks after the GFNI producer's
    /// final byte and qword transposes.  Those maps are F2-linear, so the
    /// producer may perform the same XOR before the transforms and emit only
    /// the four values the consumer observes.
    #[test]
    fn gfni_c4_pretranspose_compaction_matches_posttranspose_xor() {
        fn plane(seed: u8, k: usize) -> [u8; 64] {
            std::array::from_fn(|i| {
                seed.wrapping_add((k as u8).wrapping_mul(67))
                    .rotate_left((i & 7) as u32)
                    ^ (i as u8).wrapping_mul(29)
            })
        }

        fn qword_transpose(input: [[u8; 64]; 8]) -> [[u8; 64]; 8] {
            let mut out = [[0u8; 64]; 8];
            for src in 0..8 {
                for qword in 0..8 {
                    out[qword][8 * src..8 * src + 8]
                        .copy_from_slice(&input[src][8 * qword..8 * qword + 8]);
                }
            }
            out
        }

        fn byte_transpose(input: [u8; 64]) -> [u8; 64] {
            std::array::from_fn(|i| input[8 * (i & 7) + (i >> 3)])
        }

        fn interleave(lo: [u8; 64], hi: [u8; 64], upper: bool) -> [u8; 64] {
            let q0 = if upper { 4 } else { 0 };
            let mut out = [0u8; 64];
            for q in 0..4 {
                out[16 * q..16 * q + 8].copy_from_slice(&lo[8 * (q0 + q)..8 * (q0 + q + 1)]);
                out[16 * q + 8..16 * q + 16].copy_from_slice(&hi[8 * (q0 + q)..8 * (q0 + q + 1)]);
            }
            out
        }

        for seed in [0u8, 1, 0x5A, 0xA5, 0xFF] {
            let lo = qword_transpose(std::array::from_fn(|k| plane(seed, k)));
            let hi = qword_transpose(std::array::from_fn(|k| plane(seed, 8 + k)));
            let mut chunks = [[0u8; 64]; 16];
            for i in 0..8 {
                let l = byte_transpose(lo[i]);
                let h = byte_transpose(hi[i]);
                chunks[2 * i] = interleave(l, h, false);
                chunks[2 * i + 1] = interleave(l, h, true);
            }

            for group in 0..4 {
                let want = std::array::from_fn(|j| {
                    chunks[group][j]
                        ^ chunks[group + 4][j]
                        ^ chunks[group + 8][j]
                        ^ chunks[group + 12][j]
                });
                let parity = group >> 1;
                let lo_fold = std::array::from_fn(|j| {
                    lo[parity][j] ^ lo[parity + 2][j] ^ lo[parity + 4][j] ^ lo[parity + 6][j]
                });
                let hi_fold = std::array::from_fn(|j| {
                    hi[parity][j] ^ hi[parity + 2][j] ^ hi[parity + 4][j] ^ hi[parity + 6][j]
                });
                let got = interleave(
                    byte_transpose(lo_fold),
                    byte_transpose(hi_fold),
                    group & 1 != 0,
                );
                assert_eq!(got, want, "group={group} seed={seed:#04x}");
            }
        }
    }

    /// The residue reduction the c4 producer performs is a 4:1 XOR over
    /// qwords {i, i+2, i+4, i+6} of each plane, and the reassembly that
    /// follows it is a qword transpose.  Both are qword-granular and
    /// F2-linear, so pairing planes and halving the qword span twice
    /// (`pair`/`quad`) reaches the same two reduced registers per half that
    /// the full eight-plane transpose followed by the XOR reaches.
    #[test]
    fn gfni_c4_paired_qword_reduction_matches_transposed_xor() {
        fn plane(seed: u64, k: usize) -> [u64; 8] {
            std::array::from_fn(|q| {
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((k as u64) << 32)
                    .wrapping_add(q as u64)
                    .rotate_left(((k * 8 + q) % 61) as u32)
            })
        }

        /// `out[j][i] = in[i][j]`, the kernel's eight-plane qword transpose.
        fn qword_transpose(input: [[u64; 8]; 8]) -> [[u64; 8]; 8] {
            std::array::from_fn(|j| std::array::from_fn(|i| input[i][j]))
        }

        /// `_mm512_permutex2var_epi64`: index `< 8` picks `a`, else `b`.
        fn permx2(a: [u64; 8], idx: [usize; 8], b: [u64; 8]) -> [u64; 8] {
            std::array::from_fn(|n| if idx[n] < 8 { a[idx[n]] } else { b[idx[n] - 8] })
        }

        fn xor(a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
            std::array::from_fn(|n| a[n] ^ b[n])
        }

        const S2_LO: [usize; 8] = [0, 1, 8, 9, 2, 3, 10, 11];
        const S2_HI: [usize; 8] = [4, 5, 12, 13, 6, 7, 14, 15];
        const S3_LO: [usize; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
        const S3_HI: [usize; 8] = [4, 5, 6, 7, 12, 13, 14, 15];
        const Q_LO: [usize; 8] = [0, 2, 8, 10, 1, 3, 9, 11];
        const Q_HI: [usize; 8] = [4, 6, 12, 14, 5, 7, 13, 15];

        let pair = |a, b| xor(permx2(a, S2_LO, b), permx2(a, S2_HI, b));
        let quad = |a, b, c, d| {
            let z0 = pair(a, b);
            let z1 = pair(c, d);
            xor(permx2(z0, Q_LO, z1), permx2(z0, Q_HI, z1))
        };

        for seed in [0u64, 1, 0x5A5A_5A5A, 0xDEAD_BEEF, u64::MAX] {
            for half in 0..2 {
                let planes: [[u64; 8]; 8] = std::array::from_fn(|k| plane(seed, 8 * half + k));
                let t = qword_transpose(planes);
                let want_even = xor(xor(t[0], t[2]), xor(t[4], t[6]));
                let want_odd = xor(xor(t[1], t[3]), xor(t[5], t[7]));

                let w0 = quad(planes[0], planes[1], planes[2], planes[3]);
                let w1 = quad(planes[4], planes[5], planes[6], planes[7]);
                let got_even = permx2(w0, S3_LO, w1);
                let got_odd = permx2(w0, S3_HI, w1);

                assert_eq!(got_even, want_even, "even seed={seed:#x} half={half}");
                assert_eq!(got_odd, want_odd, "odd seed={seed:#x} half={half}");
            }
        }
    }

    /// The consumer-level claim that licenses the skip: no row of a dead line
    /// reaches any accumulator or any written table slot. Checked on the
    /// portable round-two chunk path (the same predicate the AVX-512 kernel
    /// applies), so it runs on every host.
    #[test]
    fn round2_chunk_never_reads_a_dead_line() {
        let mut rng = Rng::new(0xBEEF_2026);
        let k_skip = 6;
        let table = UniSkipFoldTable::new(k_skip, rng.f128());
        let (pair_mask, useful_pairs) = round2_pair_skip(&ranked_padding(), k_skip);
        let nc = table.n_chunks;
        // lo_size = 128 pairs = exactly one padding block.
        let lo_size = 128usize;
        let mut a = vec![0u8; 2 * lo_size * nc];
        let mut b = vec![0u8; 2 * lo_size * nc];
        for byte in a[..241 * nc].iter_mut() {
            *byte = rng.next_u64() as u8;
        }
        for byte in b[..241 * nc].iter_mut() {
            *byte = rng.next_u64() as u8;
        }
        let eq_lo = rng.f128_vec(lo_size);

        let run = |a: &[u8], b: &[u8]| {
            let mut a_chunk = vec![F128::ZERO; 2 * lo_size];
            let mut b_chunk = vec![F128::ZERO; 2 * lo_size];
            let out = round2_lookahead_chunk_scalar::<true>(
                a,
                b,
                &table,
                &mut a_chunk,
                &mut b_chunk,
                &eq_lo,
                0,
                0,
                pair_mask,
                useful_pairs,
            );
            (out, a_chunk, b_chunk)
        };
        let base = run(&a, &b);

        // Poison exactly the dead line (rows 248..=255).
        let (mut ap, mut bp) = (a.clone(), b.clone());
        for byte in ap[248 * nc..].iter_mut() {
            *byte = 0x5A;
        }
        for byte in bp[248 * nc..].iter_mut() {
            *byte = 0xC3;
        }
        assert_eq!(run(&ap, &bp), base, "dead-line rows must not be read");

        // Wider: every row of a skipped pair (242..=255) is equally unread.
        let (mut ap2, mut bp2) = (a.clone(), b.clone());
        for byte in ap2[242 * nc..].iter_mut() {
            *byte = 0x11;
        }
        for byte in bp2[242 * nc..].iter_mut() {
            *byte = 0x22;
        }
        assert_eq!(run(&ap2, &bp2), base);

        // Teeth: row 240 belongs to the boundary pair 120, which is inside the
        // useful range — poisoning it MUST change the result.
        let mut ap3 = a.clone();
        ap3[240 * nc] ^= 0xFF;
        assert_ne!(run(&ap3, &b), base, "row 240 is live");
    }

    /// Canonical B subgroups must match the complete dense map, including
    /// a linear map whose all-one input does not map to F128::ONE.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn gfni_b_canonical_prefold_matches_dense() {
        let mut rng = Rng::new(0xBC64_0049);
        let tail_bits = 0x0001_ffff_ffff_ffffu64.to_le_bytes();
        for altered_map in [false, true] {
            let mut table = UniSkipFoldTable::new(6, rng.f128());
            if altered_map {
                // Add a nonzero basis column to chunk zero. This preserves
                // F2-linearity and zero, but changes the all-one image.
                let delta = F128::new(0x1357_9BDF, 0x2468_ACE0);
                for v in (1..256).step_by(2) {
                    table.data[v] += delta;
                }
                assert_ne!(table.fold_one_row(&[0xff; 8]), F128::ONE);
            }
            let mats = kernels::x86_64::build_row_fold_mats(&table.data);
            for group in [0u8, 3] {
                let mut clean: Vec<u8> = (0..512).map(|_| rng.next_u64() as u8).collect();
                if group == 0 {
                    clean[..128].fill(0xff);
                } else {
                    clean[384..].fill(0);
                    clean[384..392].copy_from_slice(&tail_bits);
                }
                let folded = if group == 0 {
                    table.fold_one_row(&[0xff; 8])
                } else {
                    table.fold_one_row(&tail_bits)
                };
                for mutation in 0..18 {
                    let mut rows = clean.clone();
                    if (1..=16).contains(&mutation) {
                        // Every guarded qword gets an independent miss case.
                        let row = 16 * usize::from(group) + mutation - 1;
                        rows[row * 8 + mutation % 8] ^= 1;
                    } else if mutation == 17 {
                        // A mutation in the retained dense groups must not
                        // disable the shortcut or disappear from the output.
                        let row = if group == 0 { 63 } else { 0 };
                        rows[row * 8] ^= 0x40;
                    }
                    let mut want = [F128::ZERO; 64];
                    let poison = F128::new(0xA5A5_A5A5, 0x5A5A_5A5A);
                    let mut got = [poison; 64];
                    // SAFETY: 512 readable input bytes, 64 writable outputs,
                    // and folded is the exact image of the guarded pattern.
                    let hit = unsafe {
                        kernels::x86_64::gfni_fold64_rows_masked_tr_bcast(
                            rows.as_ptr(),
                            &mats,
                            want.as_mut_ptr(),
                            0,
                        );
                        kernels::x86_64::gfni_fold64_rows_tr_bcast_b_canonical(
                            rows.as_ptr(),
                            &mats,
                            got.as_mut_ptr(),
                            group,
                            folded,
                        )
                    };
                    assert_eq!(hit, mutation == 0 || mutation == 17);
                    assert_eq!(got, want, "map={altered_map} group={group} case={mutation}");
                    for unsupported in [1u8, 2, 255] {
                        got.fill(poison);
                        // SAFETY: unsupported groups use the same dense map.
                        let hit = unsafe {
                            kernels::x86_64::gfni_fold64_rows_tr_bcast_b_canonical(
                                rows.as_ptr(),
                                &mats,
                                got.as_mut_ptr(),
                                unsupported,
                                folded,
                            )
                        };
                        assert!(!hit);
                        assert_eq!(got, want);
                    }
                }
            }
        }
    }

    /// Check both tile phases and the downstream message, not only the
    /// local cache layout. Arbitrary A rows rule out A-side assumptions.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn round2_b_canonical_prefold_matches_scalar() {
        let mut rng = Rng::new(0xBC64_1200);
        for altered_map in [false, true] {
            for lo_size in [128usize, 256] {
                for pair_base in [0usize, 1, 32] {
                    let row_base = 2 * pair_base;
                    let n_rows = row_base + 2 * lo_size;
                    let mut table = UniSkipFoldTable::new(6, rng.f128());
                    if altered_map {
                        // Exercise the kernel's basis-derived replacement
                        // values as well as the direct helper's supplied ones.
                        let delta = F128::new(0x1357_9BDF, 0x2468_ACE0);
                        for v in (1..256).step_by(2) {
                            table.data[v] += delta;
                        }
                        assert_ne!(table.fold_one_row(&[0xff; 8]), F128::ONE);
                    }
                    let mats = kernels::x86_64::build_row_fold_mats(&table.data);
                    let eq_lo = rng.f128_vec(lo_size);
                    let wtab = build_w_pair_table(&eq_lo);
                    let a: Vec<u8> = (0..8 * n_rows).map(|_| rng.next_u64() as u8).collect();
                    let mut b: Vec<u8> = (0..8 * n_rows).map(|_| rng.next_u64() as u8).collect();
                    for base in (0..n_rows).step_by(256) {
                        b[base * 8..(base + 16).min(n_rows) * 8].fill(0xff);
                        if base + 256 <= n_rows {
                            b[(base + 240) * 8..(base + 256) * 8].fill(0);
                            b[(base + 240) * 8..(base + 241) * 8]
                                .copy_from_slice(&0x0001_ffff_ffff_ffffu64.to_le_bytes());
                        }
                    }
                    let mut a_ref = vec![F128::ZERO; 2 * lo_size];
                    let mut b_ref = vec![F128::ZERO; 2 * lo_size];
                    let want = round2_lookahead_chunk_scalar::<true>(
                        &a,
                        &b,
                        &table,
                        &mut a_ref,
                        &mut b_ref,
                        &eq_lo,
                        row_base,
                        pair_base,
                        0,
                        usize::MAX,
                    );
                    for weights in [None, Some(wtab.as_slice())] {
                        // SAFETY: all packed rows and eq entries exist. The
                        // no-materialize kernel does not touch empty outputs.
                        let got = unsafe {
                            kernels::x86_64::round2_lookahead_chunk_x86_avx512::<false, false>(
                                table.data.as_ptr(),
                                Some(&mats),
                                a.as_ptr(),
                                b.as_ptr(),
                                row_base,
                                &mut [],
                                &mut [],
                                &eq_lo,
                                pair_base,
                                0,
                                usize::MAX,
                                weights,
                                None,
                            )
                        };
                        assert_eq!(
                            got, want,
                            "nomat lo={lo_size} phase={pair_base} map={altered_map}"
                        );
                        let mut a_out = vec![F128::ONE; 2 * lo_size];
                        let mut b_out = vec![F128::ONE; 2 * lo_size];
                        // SAFETY: same valid inputs and full-sized destinations;
                        // WRITE=true retains the existing materialized path.
                        let got = unsafe {
                            kernels::x86_64::round2_lookahead_chunk_x86_avx512::<true, false>(
                                table.data.as_ptr(),
                                Some(&mats),
                                a.as_ptr(),
                                b.as_ptr(),
                                row_base,
                                &mut a_out,
                                &mut b_out,
                                &eq_lo,
                                pair_base,
                                0,
                                usize::MAX,
                                weights,
                                None,
                            )
                        };
                        assert_eq!(
                            got, want,
                            "materialized lo={lo_size} phase={pair_base} map={altered_map}"
                        );
                        assert_eq!(a_out, a_ref);
                        assert_eq!(b_out, b_ref);
                    }
                }
            }
        }
    }

    /// Compare the private preserving-ABI path to an independent output
    /// layout across random maps, unaligned inputs and non-ZMM-aligned output.
    #[cfg(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn gfni_c4_preserving_boundary_matches_rust() {
        #[repr(align(64))]
        struct AlignedOutput([F128; 20]);
        let mut rng = Rng::new(0xF128_C4AB_1C64_0020);
        for case in 0..64 {
            let table = UniSkipFoldTable::new(6, rng.f128());
            let coeffs = [rng.f128(), rng.f128(), rng.f128(), rng.f128()];
            let mats = kernels::x86_64::build_cfold_mats_ext(&table.data, coeffs, true);
            let offset = [0, 1, 7, 31][case % 4];
            let mut input = vec![0u8; 512 + offset];
            for b in &mut input {
                *b = rng.next_u64() as u8;
            }
            let guard = rng.f128();
            let mut output = AlignedOutput([guard; 20]);
            let got = &mut output.0;
            let mut want = [F128::ZERO; 16];
            // SAFETY: both get exactly 512 readable bytes; the composed map
            // writes sixteen fields. The extra got slots detect overstores.
            unsafe {
                kernels::x86_64::gfni_fold64_rows_masked_c4_bcast(
                    input.as_ptr().add(offset),
                    &mats,
                    want.as_mut_ptr(),
                    0,
                );
                kernels::x86_64::gfni_fold64_rows_masked_c4_bcast_lm(
                    input.as_ptr().add(offset),
                    &mats,
                    got.as_mut_ptr().add(1),
                    0,
                );
            }
            assert_eq!(got[1..17], lane_major_tiles(&want)[..], "case {case}");
            assert_eq!(got[0], guard);
            assert_eq!(got[17..], [guard; 3]);
        }
    }

    /// Byte-identity of the predicated prefold against the unpredicated
    /// kernel, on hardware. Compiles wherever the AVX-512/GFNI arms compile
    /// (e.g. `-C target-cpu=sapphirerapids`); on hosts without those features
    /// the portable oracles above carry the proof.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn gfni_masked_prefold_matches_unpredicated_kernel() {
        let mut rng = Rng::new(0x9E37_79B9);
        let table = UniSkipFoldTable::new(6, rng.f128());
        let mats = kernels::x86_64::build_row_fold_mats(&table.data);
        let nc = table.n_chunks;
        let mut clean = vec![0u8; 64 * nc];
        for byte in clean[..56 * nc].iter_mut() {
            *byte = rng.next_u64() as u8;
        }
        let mut poisoned = clean.clone();
        for byte in poisoned[56 * nc..].iter_mut() {
            *byte = 0xA5;
        }

        let mut want = [F128::ZERO; 64];
        let mut got = [F128::ZERO; 64];
        let mut off = [F128::ZERO; 64];
        // SAFETY: each call gets 512 readable input bytes and 64 writable
        // F128 outputs; the cfg gate supplies every intrinsic feature.
        unsafe {
            kernels::x86_64::gfni_fold64_rows(clean.as_ptr(), &mats, want.as_mut_ptr());
            kernels::x86_64::gfni_fold64_rows_masked(
                poisoned.as_ptr(),
                &mats,
                got.as_mut_ptr(),
                0b1000_0000,
            );
            kernels::x86_64::gfni_fold64_rows_masked(poisoned.as_ptr(), &mats, off.as_mut_ptr(), 0);
        }
        assert_eq!(got, want, "masked prefold must be byte-identical");
        // The residue-major emit and its broadcast factorisation compute the
        // same map: `out[16k + t] = fold(row 4t + k)`, byte for byte, from
        // the same inputs and the same 128 affine products.
        let mut tr = [F128::ZERO; 64];
        let mut trb = [F128::ZERO; 64];
        // SAFETY: as above.
        unsafe {
            kernels::x86_64::gfni_fold64_rows_masked_tr(
                poisoned.as_ptr(),
                &mats,
                tr.as_mut_ptr(),
                0b1000_0000,
            );
            kernels::x86_64::gfni_fold64_rows_masked_tr_bcast(
                poisoned.as_ptr(),
                &mats,
                trb.as_mut_ptr(),
                0b1000_0000,
            );
        }
        assert_eq!(trb, tr, "broadcast factorisation must be byte-identical");
        for r in 0..64 {
            assert_eq!(
                tr[16 * (r % 4) + r / 4],
                want[r],
                "residue-major slot, row {r}"
            );
        }
        // The composed rounds-3+4 fold and its broadcast factorisation
        // compute the same map, `out[t] = XOR_a c_a · fold(row 4t + a)`,
        // from the same 128 affine products — byte for byte, under every
        // dead-line mask (both substitute the same zeros).
        let coeffs = [rng.f128(), rng.f128(), rng.f128(), rng.f128()];
        let cm = kernels::x86_64::build_cfold_mats_ext(&table.data, coeffs, true);
        for dead in [0u8, 0b1000_0000, 0b1010_0101, 0b0000_0001, 0b1111_1110] {
            let mut c4 = [F128::ZERO; 64];
            let mut c4b = [F128::ZERO; 64];
            // SAFETY: as above; both kernels write sixteen F128s.
            unsafe {
                kernels::x86_64::gfni_fold64_rows_masked_c4(
                    poisoned.as_ptr(),
                    &cm,
                    c4.as_mut_ptr(),
                    dead,
                );
                kernels::x86_64::gfni_fold64_rows_masked_c4_bcast(
                    poisoned.as_ptr(),
                    &cm,
                    c4b.as_mut_ptr(),
                    dead,
                );
            }
            assert_eq!(c4b, c4, "c4 broadcast factorisation, dead={dead:#010b}");
            let mut c4l = [F128::ZERO; 64];
            // SAFETY: as above; the lane-major twin writes sixteen F128s.
            unsafe {
                kernels::x86_64::gfni_fold64_rows_masked_c4_bcast_lm(
                    poisoned.as_ptr(),
                    &cm,
                    c4l.as_mut_ptr(),
                    dead,
                );
            }
            assert_eq!(
                c4l[..16],
                lane_major_tiles(&c4[..16])[..],
                "c4 lane-major twin, dead={dead:#010b}"
            );
        }
        // ... and both against the scalar composed fold on live rows.
        let mut c4 = [F128::ZERO; 64];
        let mut c4b = [F128::ZERO; 64];
        // SAFETY: as above.
        unsafe {
            kernels::x86_64::gfni_fold64_rows_masked_c4(clean.as_ptr(), &cm, c4.as_mut_ptr(), 0);
            kernels::x86_64::gfni_fold64_rows_masked_c4_bcast(
                clean.as_ptr(),
                &cm,
                c4b.as_mut_ptr(),
                0,
            );
        }
        for t in 0..16 {
            let mut want_t = F128::ZERO;
            for (a, c) in coeffs.iter().enumerate() {
                want_t = want_t + *c * want[4 * t + a];
            }
            assert_eq!(c4[t], want_t, "composed rounds-3+4 fold, group {t}");
            assert_eq!(c4b[t], want_t, "composed broadcast fold, group {t}");
        }
        assert_ne!(off, want, "poison must be visible with the skip OFF");
        for r in 0..64 {
            assert_eq!(
                want[r],
                table.fold_one_row(&clean[r * nc..(r + 1) * nc]),
                "unpredicated kernel vs scalar table fold, row {r}"
            );
        }
    }
}
