//! Ranked-static B-complement projection for the two-image/offw round-1 path.
//!
//! For the caller/table-validated linear map T with T([0xff; 8]) = [ONE; 64],
//! T(B) = T(!B) + ONE. A known-one byte therefore needs no table load in
//! the complement. The caller must restrict this to the proven ranked BLAKE3
//! geometry; ordinary rows retain the incumbent two-image apply.

use core::arch::x86_64::*;

use super::x86_64_bstatic_plan::{BSTATIC_PLAN, BstaticRow};
use crate::ntt::inv_table::{
    IMAGE_STRIDE_64, apply_x86_avx512_register_2img_krow_at, apply_x86_avx512_register_img_krow_at,
    offw_krow_words,
};

#[derive(Clone, Copy)]
struct WindowPlan {
    /// Dense modes 0..15, with the K7 mode in the least-significant byte.
    modes: u64,
}

/// Do not use BstaticRow::vary: the old plan clears it for seven varying bytes.
const fn known_one_bytes(row: BstaticRow) -> u8 {
    let mut known = 0u8;
    let mut j = 0;
    while j < 8 {
        if (row.mask >> (8 * j)) & 0xff == 0xff && (row.expected >> (8 * j)) & 0xff == 0xff {
            known |= 1 << j;
        }
        j += 1;
    }
    known
}

/// 0: dense; 1..7: low n bytes known; 8..14: high n bytes known; 15: all known.
const fn mode_for_known_bytes(known: u8) -> u8 {
    if known == 0xff {
        return 15;
    }
    let mut n = 1;
    while n < 8 {
        if known == ((1u16 << n) - 1) as u8 {
            return n;
        }
        if known == (0xffu16 << (8 - n)) as u8 {
            return 7 + n;
        }
        n += 1;
    }
    // Any future non-prefix/suffix shape keeps the full original-B apply.
    0
}

const fn build_window_plans() -> [WindowPlan; 28] {
    let mut plans = [WindowPlan { modes: 0 }; 28];
    let mut blk = 2;
    while blk <= 29 {
        let mut k = 0;
        while k < 8 {
            let known = known_one_bytes(BSTATIC_PLAN[blk][k]);
            let mode = mode_for_known_bytes(known);
            plans[blk - 2].modes |= (mode as u64) << (8 * (7 - k));
            k += 1;
        }
        blk += 1;
    }
    plans
}

const WINDOW_PLANS: [WindowPlan; 28] = build_window_plans();

/// Project a complete ranked-static mixed-B window without changing the
/// producer's offset arena. The caller has established the ranked BLAKE3
/// geometry, so every byte represented by a nonzero mode is a structural
/// 0xff, not a sampled witness value.
///
/// `P` is the arena layout: byte order (`false`) or the `vpmaddubsw` parity
/// split (`true`, see [`crate::ntt::inv_table::apply_x86_avx512_register_2img_offp_at`]).
///
/// # Safety
/// - The CPU supports GFNI, AVX-512F and AVX-512BW.
/// - imgs are the base and sigma-8 images of the exact checked inverse-NTT
///   table, for which T([0xff; 8]) = [ONE; 64].
/// - op holds the original 128 pre-scaled offsets in the `P` layout and out
///   is 64-byte aligned. The caller performs the ordinary non-temporal
///   publish fence.
#[inline(always)]
pub(crate) unsafe fn shift_reduce_bcomplement_offw_nt2<const P: bool>(
    op: *const u16,
    out: &mut [u8; 64],
    imgs: (*const u8, *const u8),
    blk: usize,
) {
    debug_assert!((2..=29).contains(&blk));
    unsafe {
        let plan = WINDOW_PLANS[blk - 2];
        let mut modes = plan.modes;
        let apply = |p| apply_x86_avx512_register_2img_krow_at::<P>(imgs.0, imgs.1, p);
        let av = apply(op.add(offw_krow_words::<P>(7)));
        let (bv, correction) =
            apply_b_mode::<P>(imgs, op.add(64 + offw_krow_words::<P>(7)), modes as u8, av);
        let mut acc = krow_seed(av, bv, correction, modes as u8);
        let x = _mm512_set1_epi8(2);
        for k in (0..7usize).rev() {
            modes >>= 8;
            let av = apply(op.add(offw_krow_words::<P>(k)));
            let (bv, correction) =
                apply_b_mode::<P>(imgs, op.add(64 + offw_krow_words::<P>(k)), modes as u8, av);
            acc = krow_horner(acc, x, av, bv, correction, modes as u8);
        }
        _mm512_stream_si512(out.as_mut_ptr().cast::<__m512i>(), acc);
    }
}

#[inline(always)]
pub(crate) unsafe fn shift_reduce_bcomplement_offw_nt2_const<const BLK: usize, const P: bool>(
    op: *const u16,
    out: &mut [u8; 64],
    imgs: (*const u8, *const u8),
) {
    debug_assert!((3..=28).contains(&BLK));
    unsafe {
        let plan = WINDOW_PLANS[BLK - 2];
        let apply = |p| apply_x86_avx512_register_2img_krow_at::<P>(imgs.0, imgs.1, p);
        let av = apply(op.add(offw_krow_words::<P>(7)));
        let (bv, correction) = apply_b_mode::<P>(
            imgs,
            op.add(64 + offw_krow_words::<P>(7)),
            plan.modes as u8,
            av,
        );
        let mut acc = krow_seed(av, bv, correction, plan.modes as u8);
        let x = _mm512_set1_epi8(2);
        macro_rules! step {
            ($k:expr, $shift:expr) => {{
                let av = apply(op.add(offw_krow_words::<P>($k)));
                let (bv, correction) = apply_b_mode::<P>(
                    imgs,
                    op.add(64 + offw_krow_words::<P>($k)),
                    (plan.modes >> $shift) as u8,
                    av,
                );
                acc = krow_horner(acc, x, av, bv, correction, (plan.modes >> $shift) as u8);
            }};
        }
        step!(6, 8);
        step!(5, 16);
        step!(4, 24);
        step!(3, 32);
        step!(2, 40);
        step!(1, 48);
        step!(0, 56);
        _mm512_stream_si512(out.as_mut_ptr().cast::<__m512i>(), acc);
    }
}

/// Multi-image twin of [`shift_reduce_bcomplement_offw_nt2_const`]: the same
/// Horner projection, with every full apply and every pruned complement apply
/// addressing `I` σ-images (`4` or `8`, see
/// [`crate::ntt::inv_table::apply_x86_avx512_register_4img_krow_at`]). Same
/// rows, same XORs, fewer 128-bit-lane shuffles, identical bytes.
///
/// # Safety
/// As for [`shift_reduce_bcomplement_offw_nt2_const`], with `imgs.0` the base
/// of a table carrying at least `I` images.
#[inline(always)]
pub(crate) unsafe fn shift_reduce_bcomplement_offw_nt2_const_img<
    const BLK: usize,
    const P: bool,
    const I: u8,
>(
    op: *const u16,
    out: &mut [u8; 64],
    imgs: (*const u8, *const u8),
) {
    debug_assert!((3..=28).contains(&BLK));
    unsafe {
        let plan = WINDOW_PLANS[BLK - 2];
        let apply = |p| apply_x86_avx512_register_img_krow_at::<I, P>(imgs.0, imgs.1, p);
        let av = apply(op.add(offw_krow_words::<P>(7)));
        let (bv, correction) = apply_b_mode_img::<P, I>(
            imgs,
            op.add(64 + offw_krow_words::<P>(7)),
            plan.modes as u8,
            av,
        );
        let mut acc = krow_seed(av, bv, correction, plan.modes as u8);
        let x = _mm512_set1_epi8(2);
        macro_rules! step {
            ($k:expr, $shift:expr) => {{
                let av = apply(op.add(offw_krow_words::<P>($k)));
                let (bv, correction) = apply_b_mode_img::<P, I>(
                    imgs,
                    op.add(64 + offw_krow_words::<P>($k)),
                    (plan.modes >> $shift) as u8,
                    av,
                );
                acc = krow_horner(acc, x, av, bv, correction, (plan.modes >> $shift) as u8);
            }};
        }
        step!(6, 8);
        step!(5, 16);
        step!(4, 24);
        step!(3, 32);
        step!(2, 40);
        step!(1, 48);
        step!(0, 56);
        _mm512_stream_si512(out.as_mut_ptr().cast::<__m512i>(), acc);
    }
}

/// K-row modes of the two residual windows: block 2 (K2..K7 live) and block
/// 29 (K0..K3 live). Mode layout as in [`WindowPlan`].
const RESIDUAL_MODES_2: u64 = WINDOW_PLANS[0].modes;
const RESIDUAL_MODES_29: u64 = WINDOW_PLANS[27].modes;

/// One residual K-row product `T(A_k)·T(B_k)` over `I` σ-images. With `BC`,
/// a K-row whose B has structural 0xff bytes (`mode != 0`) uses the checked
/// complement identity `T(A)·T(B) = T(A)·T(!B) ⊕ T(A)` through
/// [`apply_b_mode_img`]; every other row keeps the full B apply.
#[inline(always)]
unsafe fn residual_krow_product<const P: bool, const I: u8, const BC: bool>(
    imgs: (*const u8, *const u8),
    op: *const u16,
    k: usize,
    mode: u8,
) -> __m512i {
    unsafe {
        let av = apply_x86_avx512_register_img_krow_at::<I, P>(
            imgs.0,
            imgs.1,
            op.add(offw_krow_words::<P>(k)),
        );
        let b = op.add(64 + offw_krow_words::<P>(k));
        if BC && mode != 0 {
            let (bv, correction) = apply_b_mode_img::<P, I>(imgs, b, mode, av);
            krow_seed(av, bv, correction, mode)
        } else {
            _mm512_gf2p8mul_epi8(
                av,
                apply_x86_avx512_register_img_krow_at::<I, P>(imgs.0, imgs.1, b),
            )
        }
    }
}

/// Residual windows 2 (`keep = 0xfc`: K2..K7) and 29 (`keep = 0x0f`: K0..K3)
/// over `I` σ-images, with the structural-B complement when `BC`. The same
/// `Σ_k x^k·T(A_k)·T(B_k)` as
/// [`super::x86_64::shift_reduce_inner_ab_x86_avx512_from_off_nt2_residual`],
/// in its explicit-scaling form and XOR order.
///
/// # Safety
/// As for that kernel. With `BC` the caller has also proved the ranked BLAKE3
/// static-one byte geometry and the checked table identity, exactly as for
/// [`shift_reduce_bcomplement_offw_nt2_const`]; `imgs.0` carries `I` images.
#[inline(always)]
pub(crate) unsafe fn shift_reduce_residual_offw_nt2_img<
    const P: bool,
    const I: u8,
    const BC: bool,
>(
    op: *const u16,
    out: &mut [u8; 64],
    imgs: (*const u8, *const u8),
    keep: u8,
) {
    unsafe {
        let mode = |modes: u64, k: usize| (modes >> (8 * (7 - k))) as u8;
        let scaled = |product: __m512i, k: usize| {
            _mm512_gf2p8mul_epi8(product, _mm512_set1_epi8((1u8 << k) as i8))
        };
        let acc = match keep {
            0xfc => {
                let p = |k: usize| {
                    scaled(
                        residual_krow_product::<P, I, BC>(imgs, op, k, mode(RESIDUAL_MODES_2, k)),
                        k,
                    )
                };
                let p2 = p(2);
                let p3 = p(3);
                let p4 = p(4);
                let p5 = p(5);
                let p6 = p(6);
                let p7 = p(7);
                _mm512_xor_si512(
                    _mm512_xor_si512(p2, p3),
                    _mm512_xor_si512(_mm512_xor_si512(p4, p5), _mm512_xor_si512(p6, p7)),
                )
            }
            0x0f => {
                let p0 = residual_krow_product::<P, I, BC>(imgs, op, 0, mode(RESIDUAL_MODES_29, 0));
                let p = |k: usize| {
                    scaled(
                        residual_krow_product::<P, I, BC>(imgs, op, k, mode(RESIDUAL_MODES_29, k)),
                        k,
                    )
                };
                _mm512_xor_si512(_mm512_xor_si512(p0, p(1)), _mm512_xor_si512(p(2), p(3)))
            }
            _ => core::hint::unreachable_unchecked(),
        };
        _mm512_stream_si512(out.as_mut_ptr().cast::<__m512i>(), acc);
    }
}

#[inline(always)]
unsafe fn krow_seed(av: __m512i, bv: __m512i, correction: __m512i, mode: u8) -> __m512i {
    unsafe {
        if mode == 15 {
            correction
        } else {
            _mm512_xor_si512(_mm512_gf2p8mul_epi8(av, bv), correction)
        }
    }
}

#[inline(always)]
unsafe fn krow_horner(
    acc: __m512i,
    x: __m512i,
    av: __m512i,
    bv: __m512i,
    correction: __m512i,
    mode: u8,
) -> __m512i {
    unsafe {
        let shifted = _mm512_gf2p8mul_epi8(acc, x);
        if mode == 15 {
            _mm512_xor_si512(shifted, correction)
        } else {
            _mm512_ternarylogic_epi64::<0x96>(shifted, _mm512_gf2p8mul_epi8(av, bv), correction)
        }
    }
}

/// Shared dispatch inside the existing consumer, not one call per K-row.
/// `op` addresses this K-row in the `P` layout.
#[inline(always)]
unsafe fn apply_b_mode<const P: bool>(
    imgs: (*const u8, *const u8),
    op: *const u16,
    mode: u8,
    av: __m512i,
) -> (__m512i, __m512i) {
    unsafe {
        match mode {
            1 => (apply_b_complement::<1, 8, P>(imgs, op), av),
            2 => (apply_b_complement::<2, 8, P>(imgs, op), av),
            3 => (apply_b_complement::<3, 8, P>(imgs, op), av),
            4 => (apply_b_complement::<4, 8, P>(imgs, op), av),
            5 => (apply_b_complement::<5, 8, P>(imgs, op), av),
            6 => (apply_b_complement::<6, 8, P>(imgs, op), av),
            7 => (apply_b_complement::<7, 8, P>(imgs, op), av),
            8 => (apply_b_complement::<0, 7, P>(imgs, op), av),
            9 => (apply_b_complement::<0, 6, P>(imgs, op), av),
            10 => (apply_b_complement::<0, 5, P>(imgs, op), av),
            11 => (apply_b_complement::<0, 4, P>(imgs, op), av),
            12 => (apply_b_complement::<0, 3, P>(imgs, op), av),
            13 => (apply_b_complement::<0, 2, P>(imgs, op), av),
            14 => (apply_b_complement::<0, 1, P>(imgs, op), av),
            15 => (_mm512_setzero_si512(), av),
            _ => (
                apply_x86_avx512_register_2img_krow_at::<P>(imgs.0, imgs.1, op),
                _mm512_setzero_si512(),
            ),
        }
    }
}

/// Multi-image twin of [`apply_b_mode`]: `I = 2` is exactly [`apply_b_mode`];
/// `I = 4`/`8` return the same pair from the multi-image applies.
#[inline(always)]
unsafe fn apply_b_mode_img<const P: bool, const I: u8>(
    imgs: (*const u8, *const u8),
    op: *const u16,
    mode: u8,
    av: __m512i,
) -> (__m512i, __m512i) {
    unsafe {
        if I == 2 {
            return apply_b_mode::<P>(imgs, op, mode, av);
        }
        match mode {
            1 => (apply_b_complement_img::<1, 8, P, I>(imgs, op), av),
            2 => (apply_b_complement_img::<2, 8, P, I>(imgs, op), av),
            3 => (apply_b_complement_img::<3, 8, P, I>(imgs, op), av),
            4 => (apply_b_complement_img::<4, 8, P, I>(imgs, op), av),
            5 => (apply_b_complement_img::<5, 8, P, I>(imgs, op), av),
            6 => (apply_b_complement_img::<6, 8, P, I>(imgs, op), av),
            7 => (apply_b_complement_img::<7, 8, P, I>(imgs, op), av),
            8 => (apply_b_complement_img::<0, 7, P, I>(imgs, op), av),
            9 => (apply_b_complement_img::<0, 6, P, I>(imgs, op), av),
            10 => (apply_b_complement_img::<0, 5, P, I>(imgs, op), av),
            11 => (apply_b_complement_img::<0, 4, P, I>(imgs, op), av),
            12 => (apply_b_complement_img::<0, 3, P, I>(imgs, op), av),
            13 => (apply_b_complement_img::<0, 2, P, I>(imgs, op), av),
            14 => (apply_b_complement_img::<0, 1, P, I>(imgs, op), av),
            15 => (_mm512_setzero_si512(), av),
            _ => (
                apply_x86_avx512_register_img_krow_at::<I, P>(imgs.0, imgs.1, op),
                _mm512_setzero_si512(),
            ),
        }
    }
}

/// Multi-image pruned complement apply: bytes `FIRST..END` of `!B` from `I`
/// σ-images (`4` or `8`) laid out at `imgs.0 + m·IMAGE_STRIDE_64`. Byte `b`
/// loads image `b` (`I = 8`, no shuffle) or image `b mod 4` with bytes 4..8
/// folded through one σ₃₂ shuffle (`I = 4`). Word reads, complement and field
/// extraction are [`apply_b_complement`]'s; absent rows and their XORs vanish
/// at compile time the same way.
#[inline(always)]
unsafe fn apply_b_complement_img<
    const FIRST: usize,
    const END: usize,
    const P: bool,
    const I: u8,
>(
    imgs: (*const u8, *const u8),
    op: *const u16,
) -> __m512i {
    const COMPLEMENT_OFFSETS: u64 = 0x3fc0_3fc0_3fc0_3fc0;
    unsafe {
        let (wa, wb) = if P {
            let has_even = FIRST == 0
                || (FIRST <= 2 && END > 2)
                || (FIRST <= 4 && END > 4)
                || (FIRST <= 6 && END > 6);
            let has_odd = (FIRST <= 1 && END > 1)
                || (FIRST <= 3 && END > 3)
                || (FIRST <= 5 && END > 5)
                || END == 8;
            (
                if has_even {
                    op.cast::<u64>().read_unaligned() ^ COMPLEMENT_OFFSETS
                } else {
                    0
                },
                if has_odd {
                    op.add(32).cast::<u64>().read_unaligned() ^ COMPLEMENT_OFFSETS
                } else {
                    0
                },
            )
        } else {
            (
                if FIRST < 4 {
                    op.cast::<u64>().read_unaligned() ^ COMPLEMENT_OFFSETS
                } else {
                    0
                },
                if END > 4 {
                    op.add(4).cast::<u64>().read_unaligned() ^ COMPLEMENT_OFFSETS
                } else {
                    0
                },
            )
        };
        let field = |b: usize| -> usize {
            let (w, shift) = if P {
                (if b & 1 == 0 { wa } else { wb }, 16 * (b >> 1))
            } else {
                (if b < 4 { wa } else { wb }, 16 * (b & 3))
            };
            (w >> shift) as u16 as usize
        };
        let base = imgs.0;
        let row = |image: usize, b: usize| {
            _mm512_loadu_si512(
                base.add(image * IMAGE_STRIDE_64 + field(b))
                    .cast::<__m512i>(),
            )
        };
        macro_rules! live {
            ($b:literal) => {
                FIRST <= $b && $b < END
            };
        }
        macro_rules! join {
            ($left:expr, $has_left:expr, $right:expr, $has_right:expr) => {
                if $has_left {
                    if $has_right {
                        _mm512_xor_si512($left, $right)
                    } else {
                        $left
                    }
                } else if $has_right {
                    $right
                } else {
                    _mm512_setzero_si512()
                }
            };
        }
        let lo_live = live!(0) || live!(1) || live!(2) || live!(3);
        let hi_live = live!(4) || live!(5) || live!(6) || live!(7);
        if I == 8 {
            let l01 = join!(row(0, 0), live!(0), row(1, 1), live!(1));
            let l23 = join!(row(2, 2), live!(2), row(3, 3), live!(3));
            let l45 = join!(row(4, 4), live!(4), row(5, 5), live!(5));
            let l67 = join!(row(6, 6), live!(6), row(7, 7), live!(7));
            let lo = join!(l01, live!(0) || live!(1), l23, live!(2) || live!(3));
            let hi = join!(l45, live!(4) || live!(5), l67, live!(6) || live!(7));
            join!(lo, lo_live, hi, hi_live)
        } else {
            debug_assert_eq!(I, 4);
            let a01 = join!(row(0, 0), live!(0), row(1, 1), live!(1));
            let a23 = join!(row(2, 2), live!(2), row(3, 3), live!(3));
            let b01 = join!(row(0, 4), live!(4), row(1, 5), live!(5));
            let b23 = join!(row(2, 6), live!(6), row(3, 7), live!(7));
            let v0 = join!(a01, live!(0) || live!(1), a23, live!(2) || live!(3));
            let v1 = join!(b01, live!(4) || live!(5), b23, live!(6) || live!(7));
            join!(v0, lo_live, _mm512_shuffle_i64x2::<0x4E>(v1, v1), hi_live)
        }
    }
}

/// Only bytes FIRST..END of !B vary. Retain the incumbent two-image butterfly
/// but remove every absent row and its XOR at compile time. FIRST/END are
/// literals in the fourteen prefix/suffix instantiations above.
///
/// Under the byte-order layout (`P = false`) the K-row is two words, bytes
/// 0..4 and 4..8; under the parity split (`P = true`) it is the even word
/// (bytes 0, 2, 4, 6) at `op` and the odd word (1, 3, 5, 7) at `op + 32`.
/// Either way a word is read only when one of its bytes is live, and every
/// byte is complemented, extracted and applied exactly as before.
#[inline(always)]
unsafe fn apply_b_complement<const FIRST: usize, const END: usize, const P: bool>(
    imgs: (*const u8, *const u8),
    op: *const u16,
) -> __m512i {
    const COMPLEMENT_OFFSETS: u64 = 0x3fc0_3fc0_3fc0_3fc0;
    unsafe {
        // An offset is byte << 6, so complementing a byte XORs exactly these
        // eight bits. Do not complement the u16 itself or touch the arena.
        // `wa`/`wb` are (bytes 0..4, bytes 4..8) in byte order and (even
        // bytes, odd bytes) under the parity split.
        let (wa, wb) = if P {
            let has_even = FIRST == 0
                || (FIRST <= 2 && END > 2)
                || (FIRST <= 4 && END > 4)
                || (FIRST <= 6 && END > 6);
            let has_odd = (FIRST <= 1 && END > 1)
                || (FIRST <= 3 && END > 3)
                || (FIRST <= 5 && END > 5)
                || END == 8;
            (
                if has_even {
                    op.cast::<u64>().read_unaligned() ^ COMPLEMENT_OFFSETS
                } else {
                    0
                },
                if has_odd {
                    op.add(32).cast::<u64>().read_unaligned() ^ COMPLEMENT_OFFSETS
                } else {
                    0
                },
            )
        } else {
            (
                if FIRST < 4 {
                    op.cast::<u64>().read_unaligned() ^ COMPLEMENT_OFFSETS
                } else {
                    0
                },
                if END > 4 {
                    op.add(4).cast::<u64>().read_unaligned() ^ COMPLEMENT_OFFSETS
                } else {
                    0
                },
            )
        };
        // Byte `b`'s complemented offset; `b` is a literal below, so the
        // selection and shift fold to one constant field extraction.
        let field = |b: usize| -> usize {
            let (w, shift) = if P {
                (if b & 1 == 0 { wa } else { wb }, 16 * (b >> 1))
            } else {
                (if b < 4 { wa } else { wb }, 16 * (b & 3))
            };
            (w >> shift) as u16 as usize
        };
        let row = |img: *const u8, o: usize| _mm512_loadu_si512(img.add(o).cast::<__m512i>());
        // The conditions use only const parameters. Each surviving component
        // is evaluated once, and a one-sided join has no XOR with zero.
        macro_rules! join {
            ($left:expr, $has_left:expr, $right:expr, $has_right:expr) => {
                if $has_left {
                    if $has_right {
                        _mm512_xor_si512($left, $right)
                    } else {
                        $left
                    }
                } else if $has_right {
                    $right
                } else {
                    _mm512_setzero_si512()
                }
            };
        }
        let u0 = join!(
            row(imgs.0, field(0)),
            FIRST == 0,
            row(imgs.1, field(1)),
            FIRST <= 1 && END > 1
        );
        let u1 = join!(
            row(imgs.0, field(2)),
            FIRST <= 2 && END > 2,
            row(imgs.1, field(3)),
            FIRST <= 3 && END > 3
        );
        let u2 = join!(
            row(imgs.0, field(4)),
            FIRST <= 4 && END > 4,
            row(imgs.1, field(5)),
            FIRST <= 5 && END > 5
        );
        let u3 = join!(
            row(imgs.0, field(6)),
            FIRST <= 6 && END > 6,
            row(imgs.1, field(7)),
            END == 8
        );
        let even = join!(
            u0,
            FIRST < 2,
            _mm512_shuffle_i64x2::<0x4E>(u2, u2),
            FIRST < 6 && END > 4
        );
        let odd = join!(
            u1,
            FIRST < 4 && END > 2,
            _mm512_shuffle_i64x2::<0x4E>(u3, u3),
            END > 6
        );
        let complement = join!(
            even,
            FIRST < 2 || (FIRST < 6 && END > 4),
            _mm512_shuffle_i64x2::<0xB1>(odd, odd),
            (FIRST < 4 && END > 2) || END > 6
        );
        complement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::F8;
    use crate::ntt::AdditiveNttGf8;
    use crate::ntt::inv_table::InvNttTableByteSingleGf8;

    #[repr(align(64))]
    struct Output([u8; 64]);

    /// Independent of the generated Bstatic table: these are the known-one
    /// BLAKE3 bit ranges in the ranked 15,409-bit witness layout.
    fn static_one_bit(bit: usize) -> bool {
        bit < 1153
            || (15153..15409).contains(&bit)
            || (0..56).any(|g| {
                let start = 1153 + g * 250;
                (start + 186..start + 250).contains(&bit)
            })
    }

    fn geometry_known_bytes(blk: usize, k: usize) -> u8 {
        let mut known = 0u8;
        for j in 0..8 {
            let start = blk * 512 + k * 64 + j * 8;
            if (start..start + 8).all(static_one_bit) {
                known |= 1 << j;
            }
        }
        known
    }

    fn fixed_bytes_for_mode(mode: u8) -> u8 {
        match mode {
            1..=7 => ((1u16 << mode) - 1) as u8,
            8..=14 => (0xffu16 << (15 - mode)) as u8,
            15 => 0xff,
            _ => 0,
        }
    }

    /// The static mode map and the complete Horner projection must both agree
    /// with the ordinary two-image offset kernel for every ranked mixed
    /// window. This is deliberately a whole-window oracle, not merely a
    /// test of an individual pruned table apply.
    #[test]
    fn ranked_static_bcomplement_matches_generic_offsets() {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(6, F8(64));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        let mut ones = [F8::ZERO; 64];
        table.apply(&[u8::MAX; 8], &mut ones);
        assert_eq!(ones, [F8::ONE; 64]);

        let mut seen_modes = 0u16;
        for blk in 2..=29usize {
            let plan = WINDOW_PLANS[blk - 2];
            let a: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(19));
            let mut b: [u8; 64] =
                core::array::from_fn(|i| (i as u8).wrapping_mul(73).wrapping_add(41));
            for k in 0..8 {
                let known = geometry_known_bytes(blk, k);
                let mode = (plan.modes >> (8 * (7 - k))) as u8;
                assert_eq!(
                    known_one_bytes(BSTATIC_PLAN[blk][k]),
                    known,
                    "blk={blk} k={k}"
                );
                assert_eq!(fixed_bytes_for_mode(mode), known, "blk={blk} k={k}");
                if mode != 0 {
                    seen_modes |= 1 << mode;
                }
                for j in 0..8 {
                    if known & (1 << j) != 0 {
                        b[k * 8 + j] = u8::MAX;
                    }
                }
            }
            let off: [u16; 128] =
                core::array::from_fn(|i| u16::from(if i < 64 { a[i] } else { b[i - 64] }) << 6);
            let mut got = Output([0xA5; 64]);
            let mut expected = Output([0x5A; 64]);
            unsafe {
                shift_reduce_bcomplement_offw_nt2::<false>(
                    off.as_ptr(),
                    &mut got.0,
                    table.image_ptrs(),
                    blk,
                );
                super::super::x86_64::shift_reduce_inner_ab_x86_avx512_from_off_nt2::<false>(
                    off.as_ptr(),
                    &mut expected.0,
                    table.image_ptrs(),
                );
                _mm_sfence();
            }
            assert_eq!(got.0, expected.0, "blk={blk}");
        }
        assert_eq!(seen_modes, 0xfffe, "all pruned prefix/suffix modes");
    }

    /// Scalar definition of the parity-split arena: offset `i` of a side
    /// lives at word `(i & 1) * 32 + (i >> 1)`.
    fn parity_layout(off: &[u16; 128]) -> [u16; 128] {
        let mut p = [0u16; 128];
        for side in 0..2 {
            for i in 0..64 {
                p[side * 64 + (i & 1) * 32 + (i >> 1)] = off[side * 64 + i];
            }
        }
        p
    }

    /// Every arena consumer must produce identical bytes from the byte-order
    /// arena (`P = false`) and the parity-split arena (`P = true`) built from
    /// the same window: generic Horner, both residual masks, the runtime- and
    /// const-window B-complement leaves, and the temporal generic store.
    #[test]
    fn parity_split_arena_matches_byte_order_for_every_consumer() {
        use super::super::x86_64::shift_reduce_inner_ab_x86_avx512_from_off as horner;
        use super::super::x86_64::shift_reduce_inner_ab_x86_avx512_from_off_nt2 as horner_nt2;
        use super::super::x86_64::shift_reduce_inner_ab_x86_avx512_from_off_nt2_residual as residual;
        use super::shift_reduce_bcomplement_offw_nt2 as bcomp;
        use super::shift_reduce_bcomplement_offw_nt2_const as bcomp_const;
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(6, F8(64));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        let imgs = table.image_ptrs();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 56) as u8
        };
        let mut got_w = Output([0; 64]);
        let mut got_p = Output([0; 64]);
        macro_rules! check {
            ($name:literal, $blk:expr, $w:expr, $p:expr) => {{
                got_w.0 = [0xA5; 64];
                got_p.0 = [0x5A; 64];
                unsafe {
                    $w;
                    $p;
                    _mm_sfence();
                }
                assert_eq!(got_w.0, got_p.0, "{} blk={}", $name, $blk);
            }};
        }
        for round in 0..4usize {
            for blk in 2..=29usize {
                let mut a = [0u8; 64];
                let mut b = [0u8; 64];
                for i in 0..64 {
                    a[i] = next();
                    b[i] = if round == 3 { 0xff } else { next() };
                }
                for k in 0..8 {
                    let known = geometry_known_bytes(blk, k);
                    for j in 0..8 {
                        if known & (1 << j) != 0 {
                            b[k * 8 + j] = u8::MAX;
                        }
                    }
                }
                let off_w: [u16; 128] =
                    core::array::from_fn(|i| u16::from(if i < 64 { a[i] } else { b[i - 64] }) << 6);
                let off_p = parity_layout(&off_w);
                let (pw, pp) = (off_w.as_ptr(), off_p.as_ptr());
                check!(
                    "horner nt2",
                    blk,
                    horner_nt2::<false>(pw, &mut got_w.0, imgs),
                    horner_nt2::<true>(pp, &mut got_p.0, imgs)
                );
                check!(
                    "horner temporal",
                    blk,
                    horner::<false>(pw, &mut got_w.0, 0, imgs),
                    horner::<true>(pp, &mut got_p.0, 0, imgs)
                );
                for keep in [0xfcu8, 0x0f] {
                    check!(
                        "residual",
                        blk,
                        residual::<false>(pw, &mut got_w.0, imgs, keep),
                        residual::<true>(pp, &mut got_p.0, imgs, keep)
                    );
                }
                check!(
                    "bcomplement runtime",
                    blk,
                    bcomp::<false>(pw, &mut got_w.0, imgs, blk),
                    bcomp::<true>(pp, &mut got_p.0, imgs, blk)
                );
                // The parity B-complement must also still equal the generic
                // kernel (the structural-0xff bytes are set above).
                check!(
                    "bcomplement parity vs generic",
                    blk,
                    horner_nt2::<false>(pw, &mut got_w.0, imgs),
                    bcomp::<true>(pp, &mut got_p.0, imgs, blk)
                );
                macro_rules! const_blk {
                    ($($n:literal),*) => {
                        match blk {
                            $($n => check!(
                                "bcomplement const",
                                blk,
                                bcomp_const::<$n, false>(pw, &mut got_w.0, imgs),
                                bcomp_const::<$n, true>(pp, &mut got_p.0, imgs)
                            ),)*
                            _ => {}
                        }
                    };
                }
                const_blk!(
                    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28
                );
            }
        }
    }

    /// Every multi-image leaf and the residual-window B complement must
    /// reproduce the incumbent two-image bytes for every ranked window and
    /// both arena layouts, with the structural 0xff B bytes set from the
    /// independent BLAKE3 geometry.
    #[test]
    fn image_variants_and_residual_bcomplement_match_incumbent() {
        use super::super::x86_64::shift_reduce_inner_ab_x86_avx512_from_off_nt2_residual as residual;
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(6, F8(64));
        crate::ntt::inv_table::force_table_images_for_test(8);
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        crate::ntt::inv_table::force_table_images_for_test(0);
        assert_eq!(table.table_image_count(), 8);
        let imgs = table.image_ptrs();
        let mut ones = [F8::ZERO; 64];
        table.apply(&[u8::MAX; 8], &mut ones);
        assert_eq!(ones, [F8::ONE; 64]);

        // The residual complement is not vacuous: some live residual K-row
        // carries structural B bytes, and each mode matches the geometry.
        let mut complemented = 0;
        for (blk, modes, rows) in [
            (2usize, RESIDUAL_MODES_2, 2..8usize),
            (29, RESIDUAL_MODES_29, 0..4),
        ] {
            for k in rows {
                let mode = (modes >> (8 * (7 - k))) as u8;
                assert_eq!(
                    fixed_bytes_for_mode(mode),
                    geometry_known_bytes(blk, k),
                    "blk={blk} k={k}"
                );
                complemented += usize::from(mode != 0);
            }
        }
        assert!(complemented > 0);

        let mut seed = 0x0DDB_1A5E_5BAD_5EEDu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 56) as u8
        };
        let mut expected = Output([0; 64]);
        let mut got = Output([0; 64]);
        for round in 0..4usize {
            for blk in 2..=29usize {
                let mut a = [0u8; 64];
                let mut b = [0u8; 64];
                for i in 0..64 {
                    a[i] = next();
                    b[i] = if round == 3 { 0xff } else { next() };
                }
                for k in 0..8 {
                    let known = geometry_known_bytes(blk, k);
                    for j in 0..8 {
                        if known & (1 << j) != 0 {
                            b[k * 8 + j] = u8::MAX;
                        }
                    }
                }
                let off_w: [u16; 128] =
                    core::array::from_fn(|i| u16::from(if i < 64 { a[i] } else { b[i - 64] }) << 6);
                let off_p = parity_layout(&off_w);
                for layout in [false, true] {
                    let op = if layout {
                        off_p.as_ptr()
                    } else {
                        off_w.as_ptr()
                    };
                    if blk == 2 || blk == 29 {
                        let keep = if blk == 2 { 0xfcu8 } else { 0x0f };
                        expected.0 = [0xA5; 64];
                        unsafe {
                            if layout {
                                residual::<true>(op, &mut expected.0, imgs, keep);
                            } else {
                                residual::<false>(op, &mut expected.0, imgs, keep);
                            }
                            _mm_sfence();
                        }
                        macro_rules! residual_variant {
                            ($i:literal, $bc:literal) => {{
                                got.0 = [0x5A; 64];
                                unsafe {
                                    if layout {
                                        shift_reduce_residual_offw_nt2_img::<true, $i, $bc>(
                                            op, &mut got.0, imgs, keep,
                                        );
                                    } else {
                                        shift_reduce_residual_offw_nt2_img::<false, $i, $bc>(
                                            op, &mut got.0, imgs, keep,
                                        );
                                    }
                                    _mm_sfence();
                                }
                                assert_eq!(
                                    got.0, expected.0,
                                    "residual I={} BC={} blk={blk} P={layout} round={round}",
                                    $i, $bc
                                );
                            }};
                        }
                        residual_variant!(2, false);
                        residual_variant!(2, true);
                        residual_variant!(4, false);
                        residual_variant!(4, true);
                        residual_variant!(8, false);
                        residual_variant!(8, true);
                    }
                    macro_rules! const_blk {
                        ($($n:literal),*) => {
                            match blk {
                                $($n => {
                                    expected.0 = [0xA5; 64];
                                    unsafe {
                                        if layout {
                                            shift_reduce_bcomplement_offw_nt2_const::<$n, true>(
                                                op, &mut expected.0, imgs,
                                            );
                                        } else {
                                            shift_reduce_bcomplement_offw_nt2_const::<$n, false>(
                                                op, &mut expected.0, imgs,
                                            );
                                        }
                                        _mm_sfence();
                                    }
                                    for images in [4u8, 8] {
                                        got.0 = [0x5A; 64];
                                        unsafe {
                                            match (layout, images) {
                                                (true, 4) => shift_reduce_bcomplement_offw_nt2_const_img::<$n, true, 4>(op, &mut got.0, imgs),
                                                (false, 4) => shift_reduce_bcomplement_offw_nt2_const_img::<$n, false, 4>(op, &mut got.0, imgs),
                                                (true, _) => shift_reduce_bcomplement_offw_nt2_const_img::<$n, true, 8>(op, &mut got.0, imgs),
                                                (false, _) => shift_reduce_bcomplement_offw_nt2_const_img::<$n, false, 8>(op, &mut got.0, imgs),
                                            }
                                            _mm_sfence();
                                        }
                                        assert_eq!(
                                            got.0, expected.0,
                                            "const I={images} blk={blk} P={layout} round={round}"
                                        );
                                    }
                                })*
                                _ => {}
                            }
                        };
                    }
                    const_blk!(
                        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28
                    );
                }
            }
        }
    }
}

/// Structural mode shared by the eight-row projection.
pub(crate) const fn window_mode(blk: usize, k: usize) -> u8 {
    (WINDOW_PLANS[blk - 2].modes >> (8 * (7 - k))) as u8
}
