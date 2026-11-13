use crate::field::F128;

/// `FLOCK_NO_NTT_MUL_DIET=1` restores the incumbent 6-CLMUL `ghash_mul_x4`
/// at DIET-controlled call sites, including the fused-two OUTER_LOW path.
/// The independent fused-three high-one specialization has its own rollback
/// switches. Read once, outside every lane loop.
#[inline]
fn mul_diet_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_NTT_MUL_DIET").is_some())
}

/// `FLOCK_NO_NTT_LOW_INNER_SPARSE=1` restores the general/diet product for
/// the sparse seed kernel's sole remaining twiddle (`right_twiddle` /
/// inner-b). Ranked seed-NT block 0 uses this kernel; the fused-two dense
/// outer LOW path and fused-three high-one path are independent sites.
#[inline]
fn low_inner_sparse_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_NTT_LOW_INNER_SPARSE").is_some())
}

/// A butterfly twiddle broadcast into all four 128-bit lanes, in the split
/// form [`crate::field::gf2_128::x86_64::ghash_mul_x4_split`] consumes:
/// `.0 = t`, `.1 = t·x^64 mod p`. The companion limb is only materialised
/// when the diet multiply will actually read it; otherwise it aliases `.0`
/// and is dropped by DCE.
type TwX4 = (core::arch::x86_64::__m512i, core::arch::x86_64::__m512i);

/// Broadcast `value` to all lanes and, when the split product will be used,
/// derive its `x^64` companion. Every caller does this OUTSIDE its lane loop,
/// so the extra CLMUL is amortised over the whole row set.
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq`.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn tw_x4<const LOW: bool, const DIET: bool>(value: F128) -> TwX4 {
    use crate::field::gf2_128::x86_64::ghash_shift64_x4;
    use core::arch::x86_64::*;

    // SAFETY: caller carries the required target features.
    unsafe {
        let t = _mm512_broadcast_i32x4(_mm_set_epi64x(value.hi as i64, value.lo as i64));
        // The LOW kernel needs no companion: its 3-CLMUL form is already the
        // short-product shape the split form generalises.
        let companion = if DIET && !LOW { ghash_shift64_x4(t) } else { t };
        (t, companion)
    }
}

/// Broadcast-twiddle product. `LOW` asserts the twiddle's high limb is zero
/// in every lane (3 CLMUL); otherwise `DIET` picks the 5-CLMUL split form
/// over the incumbent 6-CLMUL `ghash_mul_x4`. Monomorphized, so the choice
/// costs no branch inside the lane loop.
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq`; when `LOW`, every 128-bit lane of `t.0`
/// must have a zero high qword; when `DIET && !LOW`, `t.1` must be
/// `t.0·x^64 mod p`.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn mul_x4<const LOW: bool, const DIET: bool>(
    t: TwX4,
    v: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4, ghash_mul_x4_low_lhs, ghash_mul_x4_split};
    // SAFETY: caller carries the features and the twiddle-form preconditions.
    unsafe {
        if LOW {
            ghash_mul_x4_low_lhs(t.0, v)
        } else if DIET {
            ghash_mul_x4_split(v, t.0, t.1)
        } else {
            ghash_mul_x4(t.0, v)
        }
    }
}

/// Three-CLMUL product for a broadcast twiddle `t = a + x^64`.
/// Writing `v = b + c*x^64` gives
/// `t*v = a*b + x^64*(a*c + v)`, so the LOW product only needs one extra
/// XOR of `v` into its cross term. No x^64 companion is required.
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq` and a high qword of one in every
/// 128-bit lane of `t.0`.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn mul_x4_high_one(t: TwX4, v: core::arch::x86_64::__m512i) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;
    // SAFETY: caller establishes the features and the high-limb value.
    unsafe {
        let cross = _mm512_xor_si512(_mm512_clmulepi64_epi128::<0x10>(t.0, v), v);
        let lo = _mm512_clmulepi64_epi128::<0x00>(t.0, v);
        let poly = _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87);
        let reduced = _mm512_clmulepi64_epi128::<0x01>(cross, poly);
        _mm512_xor_si512(
            _mm512_xor_si512(lo, _mm512_bslli_epi128::<8>(cross)),
            reduced,
        )
    }
}

/// Unreduced split product for the diet forms: returns `(lo, hi)` with
/// `mul_x4::<LOW, true>(t, v) == lo ^ fold_x64(hi)`. `hi` is the
/// x^64-position half; callers may XOR several of them together and fold
/// the sum once, because [`fold_x64`] is linear.
///
/// # Safety
/// Same preconditions as `mul_x4::<LOW, true>`.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn mul_x4_parts<const LOW: bool>(
    t: TwX4,
    v: core::arch::x86_64::__m512i,
) -> (core::arch::x86_64::__m512i, core::arch::x86_64::__m512i) {
    use core::arch::x86_64::*;
    if LOW {
        (
            _mm512_clmulepi64_epi128::<0x00>(t.0, v),
            _mm512_clmulepi64_epi128::<0x10>(t.0, v),
        )
    } else {
        (
            _mm512_xor_si512(
                _mm512_clmulepi64_epi128::<0x00>(v, t.0),
                _mm512_clmulepi64_epi128::<0x01>(v, t.1),
            ),
            _mm512_xor_si512(
                _mm512_clmulepi64_epi128::<0x10>(v, t.0),
                _mm512_clmulepi64_epi128::<0x11>(v, t.1),
            ),
        )
    }
}

/// `x^64 * hi mod p` for four lanes: `(hi << 64) ^ clmul(hi.hi, 0x87)`.
/// Linear in `hi` and canonical on output.
///
/// # Safety
/// Requires `avx512f`, `avx512bw`, and `vpclmulqdq`.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn fold_x64(hi: core::arch::x86_64::__m512i) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;
    // SAFETY: the caller carries the AVX-512 byte-shift requirement.
    unsafe {
        let poly = _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87);
        _mm512_xor_si512(
            _mm512_bslli_epi128::<8>(hi),
            _mm512_clmulepi64_epi128::<0x01>(hi, poly),
        )
    }
}

/// Unreduced form of [`mul_x4_high_one`]: `(lo, hi)` with the product equal
/// to `lo ^ fold_x64(hi)`.
///
/// # Safety
/// Same preconditions as [`mul_x4_high_one`].
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn mul_x4_high_one_parts(
    t: TwX4,
    v: core::arch::x86_64::__m512i,
) -> (core::arch::x86_64::__m512i, core::arch::x86_64::__m512i) {
    use core::arch::x86_64::*;
    (
        _mm512_clmulepi64_epi128::<0x00>(t.0, v),
        _mm512_xor_si512(_mm512_clmulepi64_epi128::<0x10>(t.0, v), v),
    )
}


/// Store four F128 lanes. `NT` uses XMM `MOVNTDQ` so a cold codeword publish
/// skips write-allocate; dest must be 16-byte aligned. Temporal `storeu`
/// otherwise. Scalar twin: [`store_f128`].
///
/// # Safety
/// `avx512f`; `p` covers four F128; when `NT`, `p` is 16-byte aligned.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn store_row4<const NT: bool>(p: *mut F128, v: core::arch::x86_64::__m512i) {
    use core::arch::x86_64::*;
    // SAFETY: forwarded by the caller; SSE2 is x86_64 baseline.
    unsafe {
        if NT {
            let d = p as *mut __m128i;
            _mm_stream_si128(d, _mm512_castsi512_si128(v));
            _mm_stream_si128(d.add(1), _mm512_extracti32x4_epi32::<1>(v));
            _mm_stream_si128(d.add(2), _mm512_extracti32x4_epi32::<2>(v));
            _mm_stream_si128(d.add(3), _mm512_extracti32x4_epi32::<3>(v));
        } else {
            _mm512_storeu_si512(p as *mut __m512i, v);
        }
    }
}

/// Store one F128. `NT` uses XMM `MOVNTDQ`; dest must be 16-byte aligned.
///
/// # Safety
/// `p` is a valid F128; when `NT`, 16-byte aligned. SSE2 is x86_64 baseline.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn store_f128<const NT: bool>(p: *mut F128, v: F128) {
    use core::arch::x86_64::*;
    // SAFETY: forwarded by the caller.
    unsafe {
        if NT {
            _mm_stream_si128(p as *mut __m128i, _mm_set_epi64x(v.hi as i64, v.lo as i64));
        } else {
            *p = v;
        }
    }
}

#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    // SAFETY: forwarded caller contract.
    unsafe { butterfly_row_pair_gen::<false>(top, bot, twiddle) }
}

/// Broadcast-twiddle row pair; `LOW` asserts the twiddle's high limb is zero.
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq`; when `LOW`, `twiddle.hi == 0`.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_row_pair_gen<const LOW: bool>(
    top: &mut [F128],
    bot: &mut [F128],
    twiddle: F128,
) {
    // SAFETY: forwarded caller contract; the switch only picks the product
    // form, which is field-identical either way.
    unsafe {
        if mul_diet_disabled() {
            butterfly_row_pair_impl::<LOW, false>(top, bot, twiddle)
        } else {
            butterfly_row_pair_impl::<LOW, true>(top, bot, twiddle)
        }
    }
}

/// # Safety
/// Same contract as [`butterfly_row_pair_gen`].
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_row_pair_impl<const LOW: bool, const DIET: bool>(
    top: &mut [F128],
    bot: &mut [F128],
    twiddle: F128,
) {
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and equal slice lengths.
    unsafe {
        debug_assert!(!LOW || twiddle.hi == 0);
        let tw = tw_x4::<LOW, DIET>(twiddle);
        let lanes = top.len() & !3;
        let mut i = 0;
        while i + 8 <= lanes {
            let top0 = _mm512_loadu_si512(top.as_ptr().add(i) as *const __m512i);
            let bot0 = _mm512_loadu_si512(bot.as_ptr().add(i) as *const __m512i);
            let top1 = _mm512_loadu_si512(top.as_ptr().add(i + 4) as *const __m512i);
            let bot1 = _mm512_loadu_si512(bot.as_ptr().add(i + 4) as *const __m512i);

            let new_top0 = _mm512_xor_si512(top0, mul_x4::<LOW, DIET>(tw, bot0));
            let new_top1 = _mm512_xor_si512(top1, mul_x4::<LOW, DIET>(tw, bot1));
            let new_bot0 = _mm512_xor_si512(bot0, new_top0);
            let new_bot1 = _mm512_xor_si512(bot1, new_top1);

            _mm512_storeu_si512(top.as_mut_ptr().add(i) as *mut __m512i, new_top0);
            _mm512_storeu_si512(bot.as_mut_ptr().add(i) as *mut __m512i, new_bot0);
            _mm512_storeu_si512(top.as_mut_ptr().add(i + 4) as *mut __m512i, new_top1);
            _mm512_storeu_si512(bot.as_mut_ptr().add(i + 4) as *mut __m512i, new_bot1);
            i += 8;
        }
        while i < lanes {
            let top_lanes = _mm512_loadu_si512(top.as_ptr().add(i) as *const __m512i);
            let bot_lanes = _mm512_loadu_si512(bot.as_ptr().add(i) as *const __m512i);
            let new_top = _mm512_xor_si512(top_lanes, mul_x4::<LOW, DIET>(tw, bot_lanes));
            let new_bot = _mm512_xor_si512(bot_lanes, new_top);
            _mm512_storeu_si512(top.as_mut_ptr().add(i) as *mut __m512i, new_top);
            _mm512_storeu_si512(bot.as_mut_ptr().add(i) as *mut __m512i, new_bot);
            i += 4;
        }
        super::portable::butterfly_row_pair_gen::<LOW>(&mut top[i..], &mut bot[i..], twiddle);
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    // SAFETY: forwarded caller contract.
    unsafe { butterfly_fused_2layer_gen::<false, false>(a, b, c, d, t_outer, t_inner_a, t_inner_b) }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer`], plus: when `OUTER_LOW`,
/// `t_outer.hi == 0`; when `INNER_LOW`, both inner twiddles have a zero high
/// limb.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_gen<const OUTER_LOW: bool, const INNER_LOW: bool>(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_2layer_impl::<OUTER_LOW, INNER_LOW, false>(
                a, b, c, d, t_outer, t_inner_a, t_inner_b,
            )
        } else {
            butterfly_fused_2layer_impl::<OUTER_LOW, INNER_LOW, true>(
                a, b, c, d, t_outer, t_inner_a, t_inner_b,
            )
        }
    }
}

/// Evaluate two butterfly layers while sharing one pending outer product.
/// The inner multiplicands are canonical before their products; only the
/// common outer high part is carried into both final linear folds.
///
/// # Safety
/// Requires AVX-512F and VPCLMULQDQ. Each vector contains four F128 lanes.
/// Each non-LOW twiddle companion is its value multiplied by x^64. With
/// OUTER_LOW, the outer value has zero high qwords; with INNER_LOW, both
/// inner values have zero high qwords. Inputs are passed by value.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_2layer_diet_x4<const OUTER_LOW: bool, const INNER_LOW: bool>(
    outer: TwX4,
    inner_a: TwX4,
    inner_b: TwX4,
    va: core::arch::x86_64::__m512i,
    vb: core::arch::x86_64::__m512i,
    vc: core::arch::x86_64::__m512i,
    vd: core::arch::x86_64::__m512i,
) -> (
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
) {
    use core::arch::x86_64::*;
    unsafe {
        let (lo_ac, hi_ac) = mul_x4_parts::<OUTER_LOW>(outer, vc);
        let pa = _mm512_xor_si512(va, lo_ac);
        let pc = _mm512_xor_si512(vc, pa);

        let (lo_bd, hi_bd) = mul_x4_parts::<OUTER_LOW>(outer, vd);
        let pb = _mm512_xor_si512(_mm512_xor_si512(vb, lo_bd), fold_x64(hi_bd));
        let pd = _mm512_xor_si512(vd, pb);

        let (lo_ab, hi_ab) = mul_x4_parts::<INNER_LOW>(inner_a, pb);
        let na = _mm512_xor_si512(
            _mm512_xor_si512(pa, lo_ab),
            fold_x64(_mm512_xor_si512(hi_ac, hi_ab)),
        );
        let pb = _mm512_xor_si512(pb, na);

        let (lo_cd, hi_cd) = mul_x4_parts::<INNER_LOW>(inner_b, pd);
        let nc = _mm512_xor_si512(
            _mm512_xor_si512(pc, lo_cd),
            fold_x64(_mm512_xor_si512(hi_ac, hi_cd)),
        );
        let pd = _mm512_xor_si512(pd, nc);
        (na, pb, nc, pd)
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer_gen`].
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_impl<
    const OUTER_LOW: bool,
    const INNER_LOW: bool,
    const DIET: bool,
>(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and equal slice lengths.
    unsafe {
        let outer = tw_x4::<OUTER_LOW, DIET>(t_outer);
        let inner_a = tw_x4::<INNER_LOW, DIET>(t_inner_a);
        let inner_b = tw_x4::<INNER_LOW, DIET>(t_inner_b);
        let lanes = a.len() & !3;
        let mut i = 0;
        while i + 8 <= lanes {
            let mut va0 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let mut vb0 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            let mut vc0 = _mm512_loadu_si512(c.as_ptr().add(i) as *const __m512i);
            let mut vd0 = _mm512_loadu_si512(d.as_ptr().add(i) as *const __m512i);

            let mut va1 = _mm512_loadu_si512(a.as_ptr().add(i + 4) as *const __m512i);
            let mut vb1 = _mm512_loadu_si512(b.as_ptr().add(i + 4) as *const __m512i);
            let mut vc1 = _mm512_loadu_si512(c.as_ptr().add(i + 4) as *const __m512i);
            let mut vd1 = _mm512_loadu_si512(d.as_ptr().add(i + 4) as *const __m512i);

            if DIET {
                (va0, vb0, vc0, vd0) = butterfly_2layer_diet_x4::<OUTER_LOW, INNER_LOW>(
                    outer, inner_a, inner_b, va0, vb0, vc0, vd0,
                );
                (va1, vb1, vc1, vd1) = butterfly_2layer_diet_x4::<OUTER_LOW, INNER_LOW>(
                    outer, inner_a, inner_b, va1, vb1, vc1, vd1,
                );
            } else {
                let new_a0 = _mm512_xor_si512(va0, mul_x4::<OUTER_LOW, DIET>(outer, vc0));
                let new_a1 = _mm512_xor_si512(va1, mul_x4::<OUTER_LOW, DIET>(outer, vc1));
                vc0 = _mm512_xor_si512(vc0, new_a0);
                vc1 = _mm512_xor_si512(vc1, new_a1);
                va0 = new_a0;
                va1 = new_a1;

                let new_b0 = _mm512_xor_si512(vb0, mul_x4::<OUTER_LOW, DIET>(outer, vd0));
                let new_b1 = _mm512_xor_si512(vb1, mul_x4::<OUTER_LOW, DIET>(outer, vd1));
                vd0 = _mm512_xor_si512(vd0, new_b0);
                vd1 = _mm512_xor_si512(vd1, new_b1);
                vb0 = new_b0;
                vb1 = new_b1;

                let new_a0 = _mm512_xor_si512(va0, mul_x4::<INNER_LOW, DIET>(inner_a, vb0));
                let new_a1 = _mm512_xor_si512(va1, mul_x4::<INNER_LOW, DIET>(inner_a, vb1));
                vb0 = _mm512_xor_si512(vb0, new_a0);
                vb1 = _mm512_xor_si512(vb1, new_a1);
                va0 = new_a0;
                va1 = new_a1;

                let new_c0 = _mm512_xor_si512(vc0, mul_x4::<INNER_LOW, DIET>(inner_b, vd0));
                let new_c1 = _mm512_xor_si512(vc1, mul_x4::<INNER_LOW, DIET>(inner_b, vd1));
                vd0 = _mm512_xor_si512(vd0, new_c0);
                vd1 = _mm512_xor_si512(vd1, new_c1);
                vc0 = new_c0;
                vc1 = new_c1;
            }

            _mm512_storeu_si512(a.as_mut_ptr().add(i) as *mut __m512i, va0);
            _mm512_storeu_si512(b.as_mut_ptr().add(i) as *mut __m512i, vb0);
            _mm512_storeu_si512(c.as_mut_ptr().add(i) as *mut __m512i, vc0);
            _mm512_storeu_si512(d.as_mut_ptr().add(i) as *mut __m512i, vd0);

            _mm512_storeu_si512(a.as_mut_ptr().add(i + 4) as *mut __m512i, va1);
            _mm512_storeu_si512(b.as_mut_ptr().add(i + 4) as *mut __m512i, vb1);
            _mm512_storeu_si512(c.as_mut_ptr().add(i + 4) as *mut __m512i, vc1);
            _mm512_storeu_si512(d.as_mut_ptr().add(i + 4) as *mut __m512i, vd1);
            i += 8;
        }
        while i < lanes {
            let mut va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let mut vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            let mut vc = _mm512_loadu_si512(c.as_ptr().add(i) as *const __m512i);
            let mut vd = _mm512_loadu_si512(d.as_ptr().add(i) as *const __m512i);

            if DIET {
                (va, vb, vc, vd) = butterfly_2layer_diet_x4::<OUTER_LOW, INNER_LOW>(
                    outer, inner_a, inner_b, va, vb, vc, vd,
                );
            } else {
                let new_a = _mm512_xor_si512(va, mul_x4::<OUTER_LOW, DIET>(outer, vc));
                vc = _mm512_xor_si512(vc, new_a);
                va = new_a;
                let new_b = _mm512_xor_si512(vb, mul_x4::<OUTER_LOW, DIET>(outer, vd));
                vd = _mm512_xor_si512(vd, new_b);
                vb = new_b;

                let new_a = _mm512_xor_si512(va, mul_x4::<INNER_LOW, DIET>(inner_a, vb));
                vb = _mm512_xor_si512(vb, new_a);
                va = new_a;
                let new_c = _mm512_xor_si512(vc, mul_x4::<INNER_LOW, DIET>(inner_b, vd));
                vd = _mm512_xor_si512(vd, new_c);
                vc = new_c;
            }

            _mm512_storeu_si512(a.as_mut_ptr().add(i) as *mut __m512i, va);
            _mm512_storeu_si512(b.as_mut_ptr().add(i) as *mut __m512i, vb);
            _mm512_storeu_si512(c.as_mut_ptr().add(i) as *mut __m512i, vc);
            _mm512_storeu_si512(d.as_mut_ptr().add(i) as *mut __m512i, vd);
            i += 4;
        }
        super::portable::butterfly_fused_2layer_gen::<OUTER_LOW, INNER_LOW>(
            &mut a[i..],
            &mut b[i..],
            &mut c[i..],
            &mut d[i..],
            t_outer,
            t_inner_a,
            t_inner_b,
        );
    }
}

/// Stream one four-`F128` register to a merely 16-byte-aligned destination.
///
/// Large recyclable pool allocations may be 16 modulo 64, so an aligned ZMM
/// stream is not generally available. Four XMM streams preserve the no-RFO
/// publication contract for every permitted destination residue.
///
/// # Safety
/// Requires `avx512f`; `dst` must cover four writable `F128`s and be 16-byte
/// aligned. `ALIGNED_ZMM` additionally asserts 64-byte alignment.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn stream_f128x4<const ALIGNED_ZMM: bool>(
    dst: *mut F128,
    value: core::arch::x86_64::__m512i,
) {
    use core::arch::x86_64::*;
    // SAFETY: the caller provides four writable elements and the alignment
    // promised by the specialization.
    unsafe {
        if ALIGNED_ZMM {
            debug_assert_eq!(dst as usize % 64, 0);
            _mm512_stream_si512(dst as *mut __m512i, value);
            return;
        }
        debug_assert_eq!(dst as usize % 16, 0);
        let dst = dst as *mut __m128i;
        _mm_stream_si128(dst, _mm512_extracti32x4_epi32::<0>(value));
        _mm_stream_si128(dst.add(1), _mm512_extracti32x4_epi32::<1>(value));
        _mm_stream_si128(dst.add(2), _mm512_extracti32x4_epi32::<2>(value));
        _mm_stream_si128(dst.add(3), _mm512_extracti32x4_epi32::<3>(value));
    }
}

/// Direct-publish twin of [`butterfly_fused_2layer_gen`].
///
/// The arithmetic and dispatch are identical, but computed active lanes are
/// streamed to four disjoint codeword rows and never stored back to scratch.
/// The ranked odd-row optimization deliberately omits the known-zero scratch
/// suffix; that suffix is published from a zero register, without reloading it.
///
/// # Safety
/// Contract forwarded from the architecture dispatch wrapper. When
/// `OUTER_LOW`, `t_outer.hi == 0`; when `INNER_LOW`, both inner twiddles have a
/// zero high limb.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_publish_nt_gen<
    const OUTER_LOW: bool,
    const INNER_LOW: bool,
    const ALIGNED_ZMM: bool,
>(
    src: *const F128,
    src_step: usize,
    dst_a: *mut F128,
    dst_b: *mut F128,
    dst_c: *mut F128,
    dst_d: *mut F128,
    lanes: usize,
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    // SAFETY: forwarded caller contract. Match the incumbent's process-cached
    // multiply selection exactly; it remains outside the lane loop.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_2layer_publish_nt_cold::<OUTER_LOW, INNER_LOW, ALIGNED_ZMM>(
                src, src_step, dst_a, dst_b, dst_c, dst_d, lanes, t_outer, t_inner_a, t_inner_b,
            )
        } else {
            butterfly_fused_2layer_publish_nt_impl::<OUTER_LOW, INNER_LOW, true, ALIGNED_ZMM>(
                src, src_step, dst_a, dst_b, dst_c, dst_d, lanes, t_outer, t_inner_a, t_inner_b,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_publish_nt_cold<
    const OUTER_LOW: bool,
    const INNER_LOW: bool,
    const ALIGNED_ZMM: bool,
>(
    src: *const F128,
    src_step: usize,
    dst_a: *mut F128,
    dst_b: *mut F128,
    dst_c: *mut F128,
    dst_d: *mut F128,
    lanes: usize,
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    unsafe {
        butterfly_fused_2layer_publish_nt_impl::<OUTER_LOW, INNER_LOW, false, ALIGNED_ZMM>(
            src, src_step, dst_a, dst_b, dst_c, dst_d, lanes, t_outer, t_inner_a, t_inner_b,
        )
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer_publish_nt_gen`].
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_publish_nt_impl<
    const OUTER_LOW: bool,
    const INNER_LOW: bool,
    const DIET: bool,
    const ALIGNED_ZMM: bool,
>(
    src: *const F128,
    src_step: usize,
    dst_a: *mut F128,
    dst_b: *mut F128,
    dst_c: *mut F128,
    dst_d: *mut F128,
    lanes: usize,
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    use core::arch::x86_64::*;

    // SAFETY: the caller guarantees four readable scratch rows, four disjoint
    // 16-byte-aligned destinations, target features, and lanes in {60, 64}.
    unsafe {
        debug_assert!(lanes == 60 || lanes == 64);
        debug_assert!(!OUTER_LOW || t_outer.hi == 0);
        debug_assert!(!INNER_LOW || (t_inner_a.hi == 0 && t_inner_b.hi == 0));
        let a = src;
        let b = src.add(src_step);
        let c = src.add(2 * src_step);
        let d = src.add(3 * src_step);
        let outer = tw_x4::<OUTER_LOW, DIET>(t_outer);
        let inner_a = tw_x4::<INNER_LOW, DIET>(t_inner_a);
        let inner_b = tw_x4::<INNER_LOW, DIET>(t_inner_b);

        let mut i = 0;
        while i + 8 <= lanes {
            let mut va0 = _mm512_loadu_si512(a.add(i) as *const __m512i);
            let mut vb0 = _mm512_loadu_si512(b.add(i) as *const __m512i);
            let mut vc0 = _mm512_loadu_si512(c.add(i) as *const __m512i);
            let mut vd0 = _mm512_loadu_si512(d.add(i) as *const __m512i);

            let mut va1 = _mm512_loadu_si512(a.add(i + 4) as *const __m512i);
            let mut vb1 = _mm512_loadu_si512(b.add(i + 4) as *const __m512i);
            let mut vc1 = _mm512_loadu_si512(c.add(i + 4) as *const __m512i);
            let mut vd1 = _mm512_loadu_si512(d.add(i + 4) as *const __m512i);

            if DIET {
                (va0, vb0, vc0, vd0) = butterfly_2layer_diet_x4::<OUTER_LOW, INNER_LOW>(
                    outer, inner_a, inner_b, va0, vb0, vc0, vd0,
                );
                (va1, vb1, vc1, vd1) = butterfly_2layer_diet_x4::<OUTER_LOW, INNER_LOW>(
                    outer, inner_a, inner_b, va1, vb1, vc1, vd1,
                );
            } else {
                let new_a0 = _mm512_xor_si512(va0, mul_x4::<OUTER_LOW, DIET>(outer, vc0));
                let new_a1 = _mm512_xor_si512(va1, mul_x4::<OUTER_LOW, DIET>(outer, vc1));
                vc0 = _mm512_xor_si512(vc0, new_a0);
                vc1 = _mm512_xor_si512(vc1, new_a1);
                va0 = new_a0;
                va1 = new_a1;

                let new_b0 = _mm512_xor_si512(vb0, mul_x4::<OUTER_LOW, DIET>(outer, vd0));
                let new_b1 = _mm512_xor_si512(vb1, mul_x4::<OUTER_LOW, DIET>(outer, vd1));
                vd0 = _mm512_xor_si512(vd0, new_b0);
                vd1 = _mm512_xor_si512(vd1, new_b1);
                vb0 = new_b0;
                vb1 = new_b1;

                let new_a0 = _mm512_xor_si512(va0, mul_x4::<INNER_LOW, DIET>(inner_a, vb0));
                let new_a1 = _mm512_xor_si512(va1, mul_x4::<INNER_LOW, DIET>(inner_a, vb1));
                vb0 = _mm512_xor_si512(vb0, new_a0);
                vb1 = _mm512_xor_si512(vb1, new_a1);
                va0 = new_a0;
                va1 = new_a1;

                let new_c0 = _mm512_xor_si512(vc0, mul_x4::<INNER_LOW, DIET>(inner_b, vd0));
                let new_c1 = _mm512_xor_si512(vc1, mul_x4::<INNER_LOW, DIET>(inner_b, vd1));
                vd0 = _mm512_xor_si512(vd0, new_c0);
                vd1 = _mm512_xor_si512(vd1, new_c1);
                vc0 = new_c0;
                vc1 = new_c1;
            }

            stream_f128x4::<ALIGNED_ZMM>(dst_a.add(i), va0);
            stream_f128x4::<ALIGNED_ZMM>(dst_b.add(i), vb0);
            stream_f128x4::<ALIGNED_ZMM>(dst_c.add(i), vc0);
            stream_f128x4::<ALIGNED_ZMM>(dst_d.add(i), vd0);

            stream_f128x4::<ALIGNED_ZMM>(dst_a.add(i + 4), va1);
            stream_f128x4::<ALIGNED_ZMM>(dst_b.add(i + 4), vb1);
            stream_f128x4::<ALIGNED_ZMM>(dst_c.add(i + 4), vc1);
            stream_f128x4::<ALIGNED_ZMM>(dst_d.add(i + 4), vd1);
            i += 8;
        }
        while i < lanes {
            let mut va = _mm512_loadu_si512(a.add(i) as *const __m512i);
            let mut vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
            let mut vc = _mm512_loadu_si512(c.add(i) as *const __m512i);
            let mut vd = _mm512_loadu_si512(d.add(i) as *const __m512i);

            if DIET {
                (va, vb, vc, vd) = butterfly_2layer_diet_x4::<OUTER_LOW, INNER_LOW>(
                    outer, inner_a, inner_b, va, vb, vc, vd,
                );
            } else {
                let new_a = _mm512_xor_si512(va, mul_x4::<OUTER_LOW, DIET>(outer, vc));
                vc = _mm512_xor_si512(vc, new_a);
                va = new_a;
                let new_b = _mm512_xor_si512(vb, mul_x4::<OUTER_LOW, DIET>(outer, vd));
                vd = _mm512_xor_si512(vd, new_b);
                vb = new_b;

                let new_a = _mm512_xor_si512(va, mul_x4::<INNER_LOW, DIET>(inner_a, vb));
                vb = _mm512_xor_si512(vb, new_a);
                va = new_a;
                let new_c = _mm512_xor_si512(vc, mul_x4::<INNER_LOW, DIET>(inner_b, vd));
                vd = _mm512_xor_si512(vd, new_c);
                vc = new_c;
            }

            stream_f128x4::<ALIGNED_ZMM>(dst_a.add(i), va);
            stream_f128x4::<ALIGNED_ZMM>(dst_b.add(i), vb);
            stream_f128x4::<ALIGNED_ZMM>(dst_c.add(i), vc);
            stream_f128x4::<ALIGNED_ZMM>(dst_d.add(i), vd);
            i += 4;
        }

        // The exact-shape caller reaches i=60 only for odd_tail=4. That value is
        // published from the R1CS padding descriptor, so lanes 60..64 are
        // contractually zero. Publish a zero register: every destination byte
        // is initialized, with no dependency or reload from the scratch suffix.
        if i < 64 {
            debug_assert_eq!(i, 60);
            let zero = _mm512_setzero_si512();
            stream_f128x4::<ALIGNED_ZMM>(dst_a.add(i), zero);
            stream_f128x4::<ALIGNED_ZMM>(dst_b.add(i), zero);
            stream_f128x4::<ALIGNED_ZMM>(dst_c.add(i), zero);
            stream_f128x4::<ALIGNED_ZMM>(dst_d.add(i), zero);
        }
    }
}

/// Lane rs: one half of the rate-seed's fused-two output. Rows
/// `i·src_quarter + src_r` (i < 4) of the message pair as the seed does
/// (layer k: (0,2), (1,3) with `t_outer`; layer k+1 on the pair selected by
/// `h` with `t_inner`), and only that pair's two outputs are stored at
/// `dst_r` and `dst_quarter + dst_r`. Exactly the values the full seed kernel
/// writes to its rows 2h and 2h+1; 3 products per 2 rows.
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq`; valid, non-aliasing rows; `num_ntts`
/// a multiple of 4.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn seed_half_rows_from_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    t_outer: F128,
    t_inner: F128,
    h: usize,
) {
    use core::arch::x86_64::*;
    unsafe {
        let outer = tw_x4::<false, true>(t_outer);
        let inner = tw_x4::<false, true>(t_inner);
        let src_row = |i: usize| src.add((i * src_quarter + src_r) * num_ntts);
        let d0 = dst.add(dst_r * num_ntts);
        let d1 = dst.add((dst_quarter + dst_r) * num_ntts);
        let mut lane = 0;
        while lane < num_ntts {
            let m0 = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let m1 = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let m2 = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let m3 = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);
            let mut a = _mm512_xor_si512(m0, mul_x4::<false, true>(outer, m2));
            let mut b = _mm512_xor_si512(m1, mul_x4::<false, true>(outer, m3));
            if h == 1 {
                a = _mm512_xor_si512(m2, a);
                b = _mm512_xor_si512(m3, b);
            }
            let x0 = _mm512_xor_si512(a, mul_x4::<false, true>(inner, b));
            let x1 = _mm512_xor_si512(b, x0);
            _mm512_storeu_si512(d0.add(lane) as *mut __m512i, x0);
            _mm512_storeu_si512(d1.add(lane) as *mut __m512i, x1);
            lane += 4;
        }
    }
}

/// Out-of-place fused two-layer forward butterfly (layers 1–2 seed).
/// Same algebra as [`butterfly_fused_2layer`], loads from `src` and stores
/// to `dst`. Source and destination must not overlap.
///
/// # Safety
/// Caller guarantees target features, valid non-aliasing src/dst rows, and
/// disjoint destination row groups across concurrent calls.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 3],
) {
    // SAFETY: forwarded caller contract; identical geometry on both sides.
    unsafe {
        butterfly_fused_2layer_row_from_geo(src, quarter, r, dst, quarter, r, num_ntts, twiddles)
    }
}

/// [`butterfly_fused_2layer_row_from`] with independent source and
/// destination row geometry: source rows `(i·src_quarter + src_r)`,
/// destination rows `(i·dst_quarter + dst_r)`, `i ∈ 0..4`.
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    twiddles: &[F128; 3],
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        let diet_disabled = mul_diet_disabled();
        if !diet_disabled && twiddles[0].hi == 0 {
            butterfly_fused_2layer_row_from_geo_impl::<true, true, false>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                twiddles,
            );
        } else if diet_disabled {
            butterfly_fused_2layer_row_from_geo_cold::<false>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                twiddles,
            )
        } else {
            butterfly_fused_2layer_row_from_geo_impl::<false, true, false>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                twiddles,
            )
        }
    }
}

/// NT-publish twin of [`butterfly_fused_2layer_row_from_geo`]: dest stores
/// are XMM `MOVNTDQ`. Dest must be 16-byte aligned; `num_ntts` a multiple of
/// 4 so the scalar tail is empty (mixed NT/temporal on one line is illegal).
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from_geo`], plus the NT
/// alignment / lane-multiple constraints above.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_geo_nt(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    twiddles: &[F128; 3],
) {
    debug_assert_eq!(num_ntts % 4, 0);
    debug_assert_eq!(dst as usize % 16, 0);
    // SAFETY: forwarded caller contract.
    unsafe {
        let diet_disabled = mul_diet_disabled();
        if !diet_disabled && twiddles[0].hi == 0 {
            butterfly_fused_2layer_row_from_geo_impl::<true, true, true>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                twiddles,
            );
        } else if diet_disabled {
            butterfly_fused_2layer_row_from_geo_cold::<true>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                twiddles,
            )
        } else {
            butterfly_fused_2layer_row_from_geo_impl::<false, true, true>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                twiddles,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_row_from_geo_cold<const NT: bool>(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    twiddles: &[F128; 3],
) {
    unsafe {
        butterfly_fused_2layer_row_from_geo_impl::<false, false, NT>(
            src,
            src_quarter,
            src_r,
            dst,
            dst_quarter,
            dst_r,
            num_ntts,
            twiddles,
        )
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from_geo`]. `NT` requires
/// 16-byte dest alignment. `OUTER_LOW` additionally requires the outer
/// twiddle's high limb to be zero.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_row_from_geo_impl<
    const OUTER_LOW: bool,
    const DIET: bool,
    const NT: bool,
>(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    twiddles: &[F128; 3],
) {
    use core::arch::x86_64::*;

    let [t_outer, t_inner_a, t_inner_b] = *twiddles;
    // SAFETY: caller guarantees target features, pointer geometry, and
    // non-aliasing src/dst.
    unsafe {
        debug_assert!(!OUTER_LOW || t_outer.hi == 0);
        let outer = tw_x4::<OUTER_LOW, DIET>(t_outer);
        let inner_a = tw_x4::<false, DIET>(t_inner_a);
        let inner_b = tw_x4::<false, DIET>(t_inner_b);
        let src_row = |i: usize| src.add((i * src_quarter + src_r) * num_ntts);
        let dst_row = |i: usize| dst.add((i * dst_quarter + dst_r) * num_ntts);
        let lanes = num_ntts & !3;
        let mut lane = 0;
        while lane + 8 <= lanes {
            let mut va0 = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let mut vb0 = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let mut vc0 = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let mut vd0 = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);

            let mut va1 = _mm512_loadu_si512(src_row(0).add(lane + 4) as *const __m512i);
            let mut vb1 = _mm512_loadu_si512(src_row(1).add(lane + 4) as *const __m512i);
            let mut vc1 = _mm512_loadu_si512(src_row(2).add(lane + 4) as *const __m512i);
            let mut vd1 = _mm512_loadu_si512(src_row(3).add(lane + 4) as *const __m512i);

            if DIET {
                (va0, vb0, vc0, vd0) = butterfly_2layer_diet_x4::<OUTER_LOW, false>(
                    outer, inner_a, inner_b, va0, vb0, vc0, vd0,
                );
                (va1, vb1, vc1, vd1) = butterfly_2layer_diet_x4::<OUTER_LOW, false>(
                    outer, inner_a, inner_b, va1, vb1, vc1, vd1,
                );
            } else {
                let new_a0 = _mm512_xor_si512(va0, mul_x4::<OUTER_LOW, DIET>(outer, vc0));
                let new_a1 = _mm512_xor_si512(va1, mul_x4::<OUTER_LOW, DIET>(outer, vc1));
                vc0 = _mm512_xor_si512(vc0, new_a0);
                vc1 = _mm512_xor_si512(vc1, new_a1);
                va0 = new_a0;
                va1 = new_a1;

                let new_b0 = _mm512_xor_si512(vb0, mul_x4::<OUTER_LOW, DIET>(outer, vd0));
                let new_b1 = _mm512_xor_si512(vb1, mul_x4::<OUTER_LOW, DIET>(outer, vd1));
                vd0 = _mm512_xor_si512(vd0, new_b0);
                vd1 = _mm512_xor_si512(vd1, new_b1);
                vb0 = new_b0;
                vb1 = new_b1;

                let new_a0 = _mm512_xor_si512(va0, mul_x4::<false, DIET>(inner_a, vb0));
                let new_a1 = _mm512_xor_si512(va1, mul_x4::<false, DIET>(inner_a, vb1));
                vb0 = _mm512_xor_si512(vb0, new_a0);
                vb1 = _mm512_xor_si512(vb1, new_a1);
                va0 = new_a0;
                va1 = new_a1;

                let new_c0 = _mm512_xor_si512(vc0, mul_x4::<false, DIET>(inner_b, vd0));
                let new_c1 = _mm512_xor_si512(vc1, mul_x4::<false, DIET>(inner_b, vd1));
                vd0 = _mm512_xor_si512(vd0, new_c0);
                vd1 = _mm512_xor_si512(vd1, new_c1);
                vc0 = new_c0;
                vc1 = new_c1;
            }

            store_row4::<NT>(dst_row(0).add(lane), va0);
            store_row4::<NT>(dst_row(1).add(lane), vb0);
            store_row4::<NT>(dst_row(2).add(lane), vc0);
            store_row4::<NT>(dst_row(3).add(lane), vd0);

            store_row4::<NT>(dst_row(0).add(lane + 4), va1);
            store_row4::<NT>(dst_row(1).add(lane + 4), vb1);
            store_row4::<NT>(dst_row(2).add(lane + 4), vc1);
            store_row4::<NT>(dst_row(3).add(lane + 4), vd1);
            lane += 8;
        }
        while lane < lanes {
            let mut va = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let mut vb = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let mut vc = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let mut vd = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);

            if DIET {
                (va, vb, vc, vd) = butterfly_2layer_diet_x4::<OUTER_LOW, false>(
                    outer, inner_a, inner_b, va, vb, vc, vd,
                );
            } else {
                let new_a = _mm512_xor_si512(va, mul_x4::<OUTER_LOW, DIET>(outer, vc));
                vc = _mm512_xor_si512(vc, new_a);
                va = new_a;
                let new_b = _mm512_xor_si512(vb, mul_x4::<OUTER_LOW, DIET>(outer, vd));
                vd = _mm512_xor_si512(vd, new_b);
                vb = new_b;

                let new_a = _mm512_xor_si512(va, mul_x4::<false, DIET>(inner_a, vb));
                vb = _mm512_xor_si512(vb, new_a);
                va = new_a;
                let new_c = _mm512_xor_si512(vc, mul_x4::<false, DIET>(inner_b, vd));
                vd = _mm512_xor_si512(vd, new_c);
                vc = new_c;
            }

            store_row4::<NT>(dst_row(0).add(lane), va);
            store_row4::<NT>(dst_row(1).add(lane), vb);
            store_row4::<NT>(dst_row(2).add(lane), vc);
            store_row4::<NT>(dst_row(3).add(lane), vd);
            lane += 4;
        }
        while lane < num_ntts {
            let mut a = *src_row(0).add(lane);
            let mut b = *src_row(1).add(lane);
            let mut c = *src_row(2).add(lane);
            let mut d = *src_row(3).add(lane);

            let new_a = a + c * t_outer;
            c += new_a;
            a = new_a;
            let new_b = b + d * t_outer;
            d += new_b;
            b = new_b;

            let new_a = a + b * t_inner_a;
            b += new_a;
            a = new_a;
            let new_c = c + d * t_inner_b;
            d += new_c;
            c = new_c;

            store_f128::<NT>(dst_row(0).add(lane), a);
            store_f128::<NT>(dst_row(1).add(lane), b);
            store_f128::<NT>(dst_row(2).add(lane), c);
            store_f128::<NT>(dst_row(3).add(lane), d);
            lane += 1;
        }
    }
}

/// Sparse sibling: layer-1 and left layer-2 twiddles are zero, so `a` is
/// unchanged. Dense-with-zeros of [`butterfly_fused_2layer_row_from`].
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    right_twiddle: F128,
) {
    // SAFETY: forwarded caller contract; identical geometry on both sides.
    unsafe {
        butterfly_fused_2layer_row_from_sparse_geo(
            src,
            quarter,
            r,
            dst,
            quarter,
            r,
            num_ntts,
            right_twiddle,
        )
    }
}

/// [`butterfly_fused_2layer_row_from_sparse`] with independent source and
/// destination row geometry (see [`butterfly_fused_2layer_row_from_geo`]).
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_2layer_row_from_sparse_geo_cold::<false, false, false>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                right_twiddle,
                core::ptr::null(),
            )
        } else {
            butterfly_fused_2layer_row_from_sparse_geo_impl::<true, false, false, false>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                right_twiddle,
                core::ptr::null(),
            )
        }
    }
}

/// [`butterfly_fused_2layer_row_from_sparse_geo`] that also asks for one line
/// of each of the four rows starting at `pf_src` on every lane step, using the
/// same row geometry as `src`.
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from_sparse_geo`]; in
/// addition, the four rows `pf_src + i * src_quarter * num_ntts` must lie
/// inside the same source buffer.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_geo_pf(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
    pf_src: *const F128,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_2layer_row_from_sparse_geo_cold::<true, false, false>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                right_twiddle,
                pf_src,
            )
        } else {
            butterfly_fused_2layer_row_from_sparse_geo_impl::<true, true, false, false>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                right_twiddle,
                pf_src,
            )
        }
    }
}

/// NT-publish twin of [`butterfly_fused_2layer_row_from_sparse_geo`].
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from_geo_nt`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_geo_nt(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
) {
    debug_assert_eq!(num_ntts % 4, 0);
    debug_assert_eq!(dst as usize % 16, 0);
    let inner_low = !low_inner_sparse_disabled() && right_twiddle.hi == 0;
    // SAFETY: forwarded caller contract; `inner_low` proves the LOW-product
    // precondition by inspecting the actual twiddle.
    unsafe {
        match (mul_diet_disabled(), inner_low) {
            (true, false) => butterfly_fused_2layer_row_from_sparse_geo_cold::<false, true, false>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                right_twiddle,
                core::ptr::null(),
            ),
            (true, true) => butterfly_fused_2layer_row_from_sparse_geo_cold::<false, true, true>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                right_twiddle,
                core::ptr::null(),
            ),
            (false, false) => {
                butterfly_fused_2layer_row_from_sparse_geo_impl::<true, false, true, false>(
                    src,
                    src_quarter,
                    src_r,
                    dst,
                    dst_quarter,
                    dst_r,
                    num_ntts,
                    right_twiddle,
                    core::ptr::null(),
                )
            }
            (false, true) => {
                butterfly_fused_2layer_row_from_sparse_geo_impl::<true, false, true, true>(
                    src,
                    src_quarter,
                    src_r,
                    dst,
                    dst_quarter,
                    dst_r,
                    num_ntts,
                    right_twiddle,
                    core::ptr::null(),
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_row_from_sparse_geo_cold<
    const PF: bool,
    const NT: bool,
    const INNER_LOW: bool,
>(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
    pf_src: *const F128,
) {
    unsafe {
        butterfly_fused_2layer_row_from_sparse_geo_impl::<false, PF, NT, INNER_LOW>(
            src,
            src_quarter,
            src_r,
            dst,
            dst_quarter,
            dst_r,
            num_ntts,
            right_twiddle,
            pf_src,
        )
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from_sparse_geo`]. `NT`
/// requires 16-byte dest alignment.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_row_from_sparse_geo_impl<
    const DIET: bool,
    const PF: bool,
    const NT: bool,
    const INNER_LOW: bool,
>(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
    pf_src: *const F128,
) {
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees target features, pointer geometry, and
    // non-aliasing src/dst.
    unsafe {
        debug_assert!(!INNER_LOW || right_twiddle.hi == 0);
        let inner_b = tw_x4::<INNER_LOW, DIET>(right_twiddle);
        let src_row = |i: usize| src.add((i * src_quarter + src_r) * num_ntts);
        let dst_row = |i: usize| dst.add((i * dst_quarter + dst_r) * num_ntts);
        let pf_row = |i: usize| pf_src.add(i * src_quarter * num_ntts) as *const i8;
        let lanes = num_ntts & !3;
        let mut lane = 0;
        while lane + 8 <= lanes {
            if PF {
                let off0 = lane * core::mem::size_of::<F128>();
                let off1 = (lane + 4) * core::mem::size_of::<F128>();
                for i in 0..4 {
                    _mm_prefetch::<_MM_HINT_T0>(pf_row(i).add(off0));
                    _mm_prefetch::<_MM_HINT_T0>(pf_row(i).add(off1));
                }
            }
            let va0 = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let mut vb0 = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let mut vc0 = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let mut vd0 = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);

            let va1 = _mm512_loadu_si512(src_row(0).add(lane + 4) as *const __m512i);
            let mut vb1 = _mm512_loadu_si512(src_row(1).add(lane + 4) as *const __m512i);
            let mut vc1 = _mm512_loadu_si512(src_row(2).add(lane + 4) as *const __m512i);
            let mut vd1 = _mm512_loadu_si512(src_row(3).add(lane + 4) as *const __m512i);

            vc0 = _mm512_xor_si512(vc0, va0);
            vc1 = _mm512_xor_si512(vc1, va1);
            vd0 = _mm512_xor_si512(vd0, vb0);
            vd1 = _mm512_xor_si512(vd1, vb1);
            vb0 = _mm512_xor_si512(vb0, va0);
            vb1 = _mm512_xor_si512(vb1, va1);

            let new_c0 = _mm512_xor_si512(vc0, mul_x4::<INNER_LOW, DIET>(inner_b, vd0));
            let new_c1 = _mm512_xor_si512(vc1, mul_x4::<INNER_LOW, DIET>(inner_b, vd1));
            vd0 = _mm512_xor_si512(vd0, new_c0);
            vd1 = _mm512_xor_si512(vd1, new_c1);
            vc0 = new_c0;
            vc1 = new_c1;

            store_row4::<NT>(dst_row(0).add(lane), va0);
            store_row4::<NT>(dst_row(1).add(lane), vb0);
            store_row4::<NT>(dst_row(2).add(lane), vc0);
            store_row4::<NT>(dst_row(3).add(lane), vd0);

            store_row4::<NT>(dst_row(0).add(lane + 4), va1);
            store_row4::<NT>(dst_row(1).add(lane + 4), vb1);
            store_row4::<NT>(dst_row(2).add(lane + 4), vc1);
            store_row4::<NT>(dst_row(3).add(lane + 4), vd1);
            lane += 8;
        }
        while lane < lanes {
            if PF {
                let off = lane * core::mem::size_of::<F128>();
                for i in 0..4 {
                    _mm_prefetch::<_MM_HINT_T0>(pf_row(i).add(off));
                }
            }
            let va = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let mut vb = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let mut vc = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let mut vd = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);

            // t_outer = 0, t_inner_a = 0: a stays a.
            vc = _mm512_xor_si512(vc, va);
            vd = _mm512_xor_si512(vd, vb);
            vb = _mm512_xor_si512(vb, va);

            let new_c = _mm512_xor_si512(vc, mul_x4::<INNER_LOW, DIET>(inner_b, vd));
            vd = _mm512_xor_si512(vd, new_c);
            vc = new_c;

            store_row4::<NT>(dst_row(0).add(lane), va);
            store_row4::<NT>(dst_row(1).add(lane), vb);
            store_row4::<NT>(dst_row(2).add(lane), vc);
            store_row4::<NT>(dst_row(3).add(lane), vd);
            lane += 4;
        }
        while lane < num_ntts {
            let a = *src_row(0).add(lane);
            let mut b = *src_row(1).add(lane);
            let mut c = *src_row(2).add(lane);
            let mut d = *src_row(3).add(lane);

            c += a;
            d += b;
            b += a;
            let new_c = c + d * right_twiddle;
            d += new_c;
            c = new_c;

            store_f128::<NT>(dst_row(0).add(lane), a);
            store_f128::<NT>(dst_row(1).add(lane), b);
            store_f128::<NT>(dst_row(2).add(lane), c);
            store_f128::<NT>(dst_row(3).add(lane), d);
            lane += 1;
        }
    }
}

/// One four-row message load feeds both seed staging groups: sparse block-0
/// (`t_outer = t_inner_a = 0`) and dense block-1. Algebra of each group is
/// copied from the two-call form; destinations are the same. Live set is the
/// four original ZMMs plus three sparse working copies plus twiddle
/// broadcasts — not 32.
///
/// # Safety
/// Same physical geometry as the two-call form. `num_ntts` is the source and
/// destination row pitch; `active_lanes <= num_ntts` is the prefix this call
/// reads and writes. `dst_sparse` / `dst_dense` must not alias `src` or each
/// other. When `pf_src` is non-null, the four `pf_src` rows are valid through
/// `active_lanes`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_dense_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst_sparse: *mut F128,
    dst_dense: *mut F128,
    dst_quarter: usize,
    num_ntts: usize,
    active_lanes: usize,
    right_twiddle: F128,
    dense_tw: &[F128; 3],
    pf_src: *const F128,
) {
    debug_assert!(active_lanes <= num_ntts);
    debug_assert_eq!(active_lanes % 4, 0);
    unsafe {
        let diet_disabled = mul_diet_disabled();
        if !diet_disabled && dense_tw[0].hi == 0 {
            butterfly_fused_2layer_row_from_sparse_dense_geo_impl::<true, true>(
                src,
                src_quarter,
                src_r,
                dst_sparse,
                dst_dense,
                dst_quarter,
                num_ntts,
                active_lanes,
                right_twiddle,
                dense_tw,
                pf_src,
            );
        } else if diet_disabled {
            butterfly_fused_2layer_row_from_sparse_dense_geo_cold(
                src,
                src_quarter,
                src_r,
                dst_sparse,
                dst_dense,
                dst_quarter,
                num_ntts,
                active_lanes,
                right_twiddle,
                dense_tw,
                pf_src,
            )
        } else {
            butterfly_fused_2layer_row_from_sparse_dense_geo_impl::<false, true>(
                src,
                src_quarter,
                src_r,
                dst_sparse,
                dst_dense,
                dst_quarter,
                num_ntts,
                active_lanes,
                right_twiddle,
                dense_tw,
                pf_src,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_row_from_sparse_dense_geo_cold(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst_sparse: *mut F128,
    dst_dense: *mut F128,
    dst_quarter: usize,
    num_ntts: usize,
    active_lanes: usize,
    right_twiddle: F128,
    dense_tw: &[F128; 3],
    pf_src: *const F128,
) {
    unsafe {
        butterfly_fused_2layer_row_from_sparse_dense_geo_impl::<false, false>(
            src,
            src_quarter,
            src_r,
            dst_sparse,
            dst_dense,
            dst_quarter,
            num_ntts,
            active_lanes,
            right_twiddle,
            dense_tw,
            pf_src,
        )
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from_sparse_dense_geo`].
/// `OUTER_LOW` additionally requires the dense outer twiddle's high limb to be
/// zero.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_row_from_sparse_dense_geo_impl<
    const OUTER_LOW: bool,
    const DIET: bool,
>(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst_sparse: *mut F128,
    dst_dense: *mut F128,
    dst_quarter: usize,
    num_ntts: usize,
    active_lanes: usize,
    right_twiddle: F128,
    dense_tw: &[F128; 3],
    pf_src: *const F128,
) {
    #[cfg(target_feature = "avx512bw")]
    if DIET {
        // SAFETY: the shared implementation already carries the geometry,
        // feature and twiddle-form contract; the caller keeps buffers disjoint.
        unsafe {
            seed_sparse_dense_compact::<OUTER_LOW, false>(
                src, src_quarter, src_r, dst_sparse, dst_dense, dst_quarter,
                num_ntts, active_lanes, right_twiddle, dense_tw, pf_src, core::ptr::null(), num_ntts,
            );
        }
        return;
    }
    use core::arch::x86_64::*;
    let [t_outer, t_inner_a, t_inner_b] = *dense_tw;
    let pf = !pf_src.is_null();
    unsafe {
        debug_assert!(!OUTER_LOW || t_outer.hi == 0);
        let sparse_b = tw_x4::<false, DIET>(right_twiddle);
        let outer = tw_x4::<OUTER_LOW, DIET>(t_outer);
        let inner_a = tw_x4::<false, DIET>(t_inner_a);
        let inner_b = tw_x4::<false, DIET>(t_inner_b);
        let src_row = |i: usize| src.add((i * src_quarter + src_r) * num_ntts);
        let sp_row = |i: usize| dst_sparse.add((i * dst_quarter) * num_ntts);
        let dn_row = |i: usize| dst_dense.add((i * dst_quarter) * num_ntts);
        let pf_row = |i: usize| pf_src.add(i * src_quarter * num_ntts) as *const i8;
        let lanes = active_lanes & !3;
        let mut lane = 0;
        while lane + 8 <= lanes {
            if pf {
                let off0 = lane * core::mem::size_of::<F128>();
                let off1 = (lane + 4) * core::mem::size_of::<F128>();
                for i in 0..4 {
                    _mm_prefetch::<_MM_HINT_T0>(pf_row(i).add(off0));
                    _mm_prefetch::<_MM_HINT_T0>(pf_row(i).add(off1));
                }
            }
            let va0 = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let vb0 = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let vc0 = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let vd0 = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);

            let va1 = _mm512_loadu_si512(src_row(0).add(lane + 4) as *const __m512i);
            let vb1 = _mm512_loadu_si512(src_row(1).add(lane + 4) as *const __m512i);
            let vc1 = _mm512_loadu_si512(src_row(2).add(lane + 4) as *const __m512i);
            let vd1 = _mm512_loadu_si512(src_row(3).add(lane + 4) as *const __m512i);

            let mut sb0 = vb0;
            let mut sc0 = _mm512_xor_si512(vc0, va0);
            let mut sd0 = _mm512_xor_si512(vd0, vb0);
            sb0 = _mm512_xor_si512(sb0, va0);

            let mut sb1 = vb1;
            let mut sc1 = _mm512_xor_si512(vc1, va1);
            let mut sd1 = _mm512_xor_si512(vd1, vb1);
            sb1 = _mm512_xor_si512(sb1, va1);

            let new_sc0 = _mm512_xor_si512(sc0, mul_x4::<false, DIET>(sparse_b, sd0));
            let new_sc1 = _mm512_xor_si512(sc1, mul_x4::<false, DIET>(sparse_b, sd1));
            sd0 = _mm512_xor_si512(sd0, new_sc0);
            sd1 = _mm512_xor_si512(sd1, new_sc1);
            sc0 = new_sc0;
            sc1 = new_sc1;

            _mm512_storeu_si512(sp_row(0).add(lane) as *mut __m512i, va0);
            _mm512_storeu_si512(sp_row(1).add(lane) as *mut __m512i, sb0);
            _mm512_storeu_si512(sp_row(2).add(lane) as *mut __m512i, sc0);
            _mm512_storeu_si512(sp_row(3).add(lane) as *mut __m512i, sd0);

            _mm512_storeu_si512(sp_row(0).add(lane + 4) as *mut __m512i, va1);
            _mm512_storeu_si512(sp_row(1).add(lane + 4) as *mut __m512i, sb1);
            _mm512_storeu_si512(sp_row(2).add(lane + 4) as *mut __m512i, sc1);
            _mm512_storeu_si512(sp_row(3).add(lane + 4) as *mut __m512i, sd1);

            let (va0, vb0, vc0, vd0) = if DIET {
                butterfly_2layer_diet_x4::<OUTER_LOW, false>(
                    outer, inner_a, inner_b, va0, vb0, vc0, vd0,
                )
            } else {
                let new_a0 = _mm512_xor_si512(va0, mul_x4::<OUTER_LOW, DIET>(outer, vc0));
                let vc0 = _mm512_xor_si512(vc0, new_a0);
                let va0 = new_a0;
                let new_b0 = _mm512_xor_si512(vb0, mul_x4::<OUTER_LOW, DIET>(outer, vd0));
                let vd0 = _mm512_xor_si512(vd0, new_b0);
                let vb0 = new_b0;
                let new_a0 = _mm512_xor_si512(va0, mul_x4::<false, DIET>(inner_a, vb0));
                let vb0 = _mm512_xor_si512(vb0, new_a0);
                let va0 = new_a0;
                let new_c0 = _mm512_xor_si512(vc0, mul_x4::<false, DIET>(inner_b, vd0));
                let vd0 = _mm512_xor_si512(vd0, new_c0);
                let vc0 = new_c0;
                (va0, vb0, vc0, vd0)
            };
            let (va1, vb1, vc1, vd1) = if DIET {
                butterfly_2layer_diet_x4::<OUTER_LOW, false>(
                    outer, inner_a, inner_b, va1, vb1, vc1, vd1,
                )
            } else {
                let new_a1 = _mm512_xor_si512(va1, mul_x4::<OUTER_LOW, DIET>(outer, vc1));
                let vc1 = _mm512_xor_si512(vc1, new_a1);
                let va1 = new_a1;
                let new_b1 = _mm512_xor_si512(vb1, mul_x4::<OUTER_LOW, DIET>(outer, vd1));
                let vd1 = _mm512_xor_si512(vd1, new_b1);
                let vb1 = new_b1;
                let new_a1 = _mm512_xor_si512(va1, mul_x4::<false, DIET>(inner_a, vb1));
                let vb1 = _mm512_xor_si512(vb1, new_a1);
                let va1 = new_a1;
                let new_c1 = _mm512_xor_si512(vc1, mul_x4::<false, DIET>(inner_b, vd1));
                let vd1 = _mm512_xor_si512(vd1, new_c1);
                let vc1 = new_c1;
                (va1, vb1, vc1, vd1)
            };

            _mm512_storeu_si512(dn_row(0).add(lane) as *mut __m512i, va0);
            _mm512_storeu_si512(dn_row(1).add(lane) as *mut __m512i, vb0);
            _mm512_storeu_si512(dn_row(2).add(lane) as *mut __m512i, vc0);
            _mm512_storeu_si512(dn_row(3).add(lane) as *mut __m512i, vd0);

            _mm512_storeu_si512(dn_row(0).add(lane + 4) as *mut __m512i, va1);
            _mm512_storeu_si512(dn_row(1).add(lane + 4) as *mut __m512i, vb1);
            _mm512_storeu_si512(dn_row(2).add(lane + 4) as *mut __m512i, vc1);
            _mm512_storeu_si512(dn_row(3).add(lane + 4) as *mut __m512i, vd1);
            lane += 8;
        }
        while lane < lanes {
            if pf {
                let off = lane * core::mem::size_of::<F128>();
                for i in 0..4 {
                    _mm_prefetch::<_MM_HINT_T0>(pf_row(i).add(off));
                }
            }
            let va = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let vb = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let vc = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let vd = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);

            let mut sb = vb;
            let mut sc = _mm512_xor_si512(vc, va);
            let mut sd = _mm512_xor_si512(vd, vb);
            sb = _mm512_xor_si512(sb, va);
            let new_c = _mm512_xor_si512(sc, mul_x4::<false, DIET>(sparse_b, sd));
            sd = _mm512_xor_si512(sd, new_c);
            sc = new_c;
            _mm512_storeu_si512(sp_row(0).add(lane) as *mut __m512i, va);
            _mm512_storeu_si512(sp_row(1).add(lane) as *mut __m512i, sb);
            _mm512_storeu_si512(sp_row(2).add(lane) as *mut __m512i, sc);
            _mm512_storeu_si512(sp_row(3).add(lane) as *mut __m512i, sd);

            let (va, vb, vc, vd) = if DIET {
                butterfly_2layer_diet_x4::<OUTER_LOW, false>(
                    outer, inner_a, inner_b, va, vb, vc, vd,
                )
            } else {
                let new_a = _mm512_xor_si512(va, mul_x4::<OUTER_LOW, DIET>(outer, vc));
                let vc = _mm512_xor_si512(vc, new_a);
                let va = new_a;
                let new_b = _mm512_xor_si512(vb, mul_x4::<OUTER_LOW, DIET>(outer, vd));
                let vd = _mm512_xor_si512(vd, new_b);
                let vb = new_b;
                let new_a = _mm512_xor_si512(va, mul_x4::<false, DIET>(inner_a, vb));
                let vb = _mm512_xor_si512(vb, new_a);
                let va = new_a;
                let new_c = _mm512_xor_si512(vc, mul_x4::<false, DIET>(inner_b, vd));
                let vd = _mm512_xor_si512(vd, new_c);
                let vc = new_c;
                (va, vb, vc, vd)
            };
            _mm512_storeu_si512(dn_row(0).add(lane) as *mut __m512i, va);
            _mm512_storeu_si512(dn_row(1).add(lane) as *mut __m512i, vb);
            _mm512_storeu_si512(dn_row(2).add(lane) as *mut __m512i, vc);
            _mm512_storeu_si512(dn_row(3).add(lane) as *mut __m512i, vd);
            lane += 4;
        }
        // Ranked L0 / recursive seed pass `num_ntts ∈ {64, 8}`: this
        // remainder is dead. Keep it out of the 4-wide leaf's I-cache; other
        // geometries still enter the exact scalar remainder when needed.
        if lane < active_lanes {
            sparse_dense_geo_scalar_tail(
                src_row,
                sp_row,
                dn_row,
                right_twiddle,
                t_outer,
                t_inner_a,
                t_inner_b,
                lane,
                active_lanes,
            );
        }
    }
}

/// Scalar remainder of [`butterfly_fused_2layer_row_from_sparse_dense_geo_impl`].
/// Outlined so the ranked 4-wide body does not carry a 40-line never-taken
/// tail in I-cache. Same stores, same algebra.
#[cold]
#[inline(never)]
fn sparse_dense_geo_scalar_tail(
    src_row: impl Fn(usize) -> *const F128,
    sp_row: impl Fn(usize) -> *mut F128,
    dn_row: impl Fn(usize) -> *mut F128,
    right_twiddle: F128,
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    mut lane: usize,
    num_ntts: usize,
) {
    while lane < num_ntts {
        unsafe {
            let a = *src_row(0).add(lane);
            let b = *src_row(1).add(lane);
            let c = *src_row(2).add(lane);
            let d = *src_row(3).add(lane);
            let mut sb = b;
            let mut sc = c + a;
            let mut sd = d + b;
            sb += a;
            let new_c = sc + sd * right_twiddle;
            sd += new_c;
            sc = new_c;
            *sp_row(0).add(lane) = a;
            *sp_row(1).add(lane) = sb;
            *sp_row(2).add(lane) = sc;
            *sp_row(3).add(lane) = sd;
            let mut va = a;
            let mut vb = b;
            let mut vc = c;
            let mut vd = d;
            let new_a = va + vc * t_outer;
            vc += new_a;
            va = new_a;
            let new_b = vb + vd * t_outer;
            vd += new_b;
            vb = new_b;
            let new_a = va + vb * t_inner_a;
            vb += new_a;
            va = new_a;
            let new_c = vc + vd * t_inner_b;
            vd += new_c;
            vc = new_c;
            *dn_row(0).add(lane) = va;
            *dn_row(1).add(lane) = vb;
            *dn_row(2).add(lane) = vc;
            *dn_row(3).add(lane) = vd;
        }
        lane += 1;
    }
}

/// # Safety
/// The caller guarantees target features, pointer validity, and disjoint rows.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_4layer_row(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    active_lanes: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_4layer_row_impl::<false, false, 0, 0, 0>(
                ptr,
                sixteenth,
                num_ntts,
                active_lanes,
                r,
                twiddles,
                0,
                core::ptr::null(),
            )
        } else {
            butterfly_fused_4layer_row_impl::<true, false, 0, 0, 0>(
                ptr,
                sixteenth,
                num_ntts,
                active_lanes,
                r,
                twiddles,
                0,
                core::ptr::null(),
            )
        }
    }
}

/// [`butterfly_fused_4layer_row`] that also issues one line hint per row of
/// row group `pf_r` at every lane step. `H` selects the hint level
/// (1 = L1, 2 = L2).
///
/// # Safety
/// Same contract as [`butterfly_fused_4layer_row`]; in addition, row group
/// `pf_r` must lie inside the same block.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_4layer_row_pf<const H: u8>(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    active_lanes: usize,
    r: usize,
    twiddles: &[F128; 15],
    pf_r: usize,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_4layer_row_impl::<false, false, H, 0, 0>(
                ptr,
                sixteenth,
                num_ntts,
                active_lanes,
                r,
                twiddles,
                pf_r,
                core::ptr::null(),
            )
        } else {
            butterfly_fused_4layer_row_impl::<true, false, H, 0, 0>(
                ptr,
                sixteenth,
                num_ntts,
                active_lanes,
                r,
                twiddles,
                pf_r,
                core::ptr::null(),
            )
        }
    }
}

/// Shape-monomorphized [`butterfly_fused_4layer_row_pf`] for the recurring
/// deep-pass geometries: `S16` (`sixteenth`) and `NN` (`num_ntts`) become
/// compile-time constants, so the sixteen strided row addresses collapse to
/// one base register plus constant displacements. The generic kernel's
/// register pressure forces those sixteen pointers to the stack and reloads
/// them every lane step (a fifth of its cycle samples on the production
/// profile); the shaped form deletes that traffic without touching the
/// butterfly body — same kernel, same element order, bit-identical stores.
/// `H = 0` runs un-hinted and ignores `pf_r`.
///
/// # Safety
/// Same contract as [`butterfly_fused_4layer_row_pf`], with
/// `sixteenth == S16` and `num_ntts == NN`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_4layer_row_shaped<
    const S16: usize,
    const NN: usize,
    const H: u8,
>(
    ptr: *mut F128,
    active_lanes: usize,
    r: usize,
    twiddles: &[F128; 15],
    pf_r: usize,
) {
    // SAFETY: forwarded caller contract; S16/NN substitute equal runtime
    // values in the same impl body (a distinct monomorphization).
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_4layer_row_impl::<false, false, H, S16, NN>(
                ptr,
                S16,
                NN,
                active_lanes,
                r,
                twiddles,
                pf_r,
                core::ptr::null(),
            )
        } else {
            butterfly_fused_4layer_row_impl::<true, false, H, S16, NN>(
                ptr,
                S16,
                NN,
                active_lanes,
                r,
                twiddles,
                pf_r,
                core::ptr::null(),
            )
        }
    }
}

/// Fixed-register prepared four-layer butterfly with lazy x^64 folds.
/// Rows retain a shared pending high half until they become multiplicands or
/// are stored. Linearity combines those halves into 20 reductions instead of
/// 32. The top three prepared pairs remain in zmm16..21 across the loop;
/// terminal pairs shorten temporary lifetimes so all work fits without spills.
/// Every original row load, store and prefetch hint remains present.
///
/// # Safety
/// Requires AVX-512F, AVX-512BW and VPCLMULQDQ. The caller provides valid,
/// disjoint row storage and initialized prepared pairs for the shaped geometry.
#[cfg(target_feature = "avx512bw")]
#[inline(always)]
unsafe fn butterfly_four_fixed_registers<const S16: usize, const NN: usize, const H: u8, const ORD: u8>(
    ptr: *mut F128,
    prepared: *const PreparedFour,
    vector_bytes: usize,
    prefetch: *const F128,
) {
    // SAFETY: the caller establishes all storage and ISA preconditions.
    // All clobbered vector registers, flags and rax are declared. Memory is
    // both read and written; nostack only asserts that this block uses no
    // stack space. No options claim purity or read-only memory.
    unsafe {
        core::arch::asm!(
            "test rdx, rdx",
            "jz 3f",
            "vbroadcasti32x4 zmm16, [rsi+0]",
            "vbroadcasti32x4 zmm17, [rsi+16]",
            "mov eax, 0x87",
            "vmovq xmm30, rax",
            "vshufi32x4 zmm30, zmm30, zmm30, 0",
            "vbroadcasti32x4 zmm18, [rsi+32]",
            "vbroadcasti32x4 zmm19, [rsi+48]",
            "vbroadcasti32x4 zmm20, [rsi+64]",
            "vbroadcasti32x4 zmm21, [rsi+80]",
            "xor eax, eax",
            ".p2align 5",
            "2:",
            // Load order (`ord`). In the two deep fused-four sweeps the 16 rows of a group are a multiple of
            // 4 KiB apart, so the 16 lines of one lane step fall into ONE set of the 12-way L1D and four of
            // them are always out of L1 when their turn comes. The near hint fetched rows 0..15 in that
            // order, so the set has already dropped rows 0..3. ord == 1 loads rows 12..15 first (their
            // stores come last), then 4..11, and rows 0..3 last: the four L1 misses then evict only rows
            // whose value is already in a register and whose store is the last of the step, instead of
            // rows that are about to be loaded. The 16 loads are independent, so the register contents,
            // and with them every output byte, are the same for either order.
            ".if {ord} == 1",
            "vmovdqu64 zmm12, [rdi+rax+{stride}*12]",
            "vmovdqu64 zmm13, [rdi+rax+{stride}*13]",
            "vmovdqu64 zmm14, [rdi+rax+{stride}*14]",
            "vmovdqu64 zmm15, [rdi+rax+{stride}*15]",
            "vmovdqu64 zmm4, [rdi+rax+{stride}*4]",
            "vmovdqu64 zmm5, [rdi+rax+{stride}*5]",
            "vmovdqu64 zmm6, [rdi+rax+{stride}*6]",
            "vmovdqu64 zmm7, [rdi+rax+{stride}*7]",
            "vmovdqu64 zmm8, [rdi+rax+{stride}*8]",
            "vmovdqu64 zmm9, [rdi+rax+{stride}*9]",
            "vmovdqu64 zmm10, [rdi+rax+{stride}*10]",
            "vmovdqu64 zmm11, [rdi+rax+{stride}*11]",
            "vmovdqu64 zmm0, [rdi+rax+{stride}*0]",
            "vmovdqu64 zmm1, [rdi+rax+{stride}*1]",
            "vmovdqu64 zmm2, [rdi+rax+{stride}*2]",
            "vmovdqu64 zmm3, [rdi+rax+{stride}*3]",
            ".else",
            "vmovdqu64 zmm0, [rdi+rax+{stride}*0]",
            "vmovdqu64 zmm1, [rdi+rax+{stride}*1]",
            "vmovdqu64 zmm2, [rdi+rax+{stride}*2]",
            "vmovdqu64 zmm3, [rdi+rax+{stride}*3]",
            "vmovdqu64 zmm4, [rdi+rax+{stride}*4]",
            "vmovdqu64 zmm5, [rdi+rax+{stride}*5]",
            "vmovdqu64 zmm6, [rdi+rax+{stride}*6]",
            "vmovdqu64 zmm7, [rdi+rax+{stride}*7]",
            "vmovdqu64 zmm8, [rdi+rax+{stride}*8]",
            "vmovdqu64 zmm9, [rdi+rax+{stride}*9]",
            "vmovdqu64 zmm10, [rdi+rax+{stride}*10]",
            "vmovdqu64 zmm11, [rdi+rax+{stride}*11]",
            "vmovdqu64 zmm12, [rdi+rax+{stride}*12]",
            "vmovdqu64 zmm13, [rdi+rax+{stride}*13]",
            "vmovdqu64 zmm14, [rdi+rax+{stride}*14]",
            "vmovdqu64 zmm15, [rdi+rax+{stride}*15]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*0]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*0]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*1]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*1]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*2]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*2]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*3]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*3]",
            ".endif",
            "vpclmulqdq zmm22, zmm12, zmm16, 0",
            "vpclmulqdq zmm23, zmm12, zmm17, 1",
            "vpternlogd zmm4, zmm22, zmm23, 0x96",
            "vpclmulqdq zmm22, zmm13, zmm16, 0",
            "vpclmulqdq zmm23, zmm13, zmm17, 1",
            "vpternlogd zmm5, zmm22, zmm23, 0x96",
            "vpclmulqdq zmm22, zmm14, zmm16, 0",
            "vpclmulqdq zmm23, zmm14, zmm17, 1",
            "vpternlogd zmm6, zmm22, zmm23, 0x96",
            "vpclmulqdq zmm22, zmm15, zmm16, 0",
            "vpclmulqdq zmm23, zmm15, zmm17, 1",
            "vpternlogd zmm7, zmm22, zmm23, 0x96",
            "vpclmulqdq zmm22, zmm12, zmm16, 16",
            "vpclmulqdq zmm23, zmm12, zmm17, 17",
            "vpclmulqdq zmm24, zmm13, zmm16, 16",
            "vpclmulqdq zmm25, zmm13, zmm17, 17",
            "vpclmulqdq zmm26, zmm14, zmm16, 16",
            "vpclmulqdq zmm27, zmm14, zmm17, 17",
            "vpclmulqdq zmm28, zmm15, zmm16, 16",
            "vpclmulqdq zmm29, zmm15, zmm17, 17",
            "vpxorq zmm22, zmm22, zmm23",
            "vpxorq zmm23, zmm24, zmm25",
            "vpxorq zmm24, zmm26, zmm27",
            "vpxorq zmm25, zmm28, zmm29",
            "vpslldq zmm26, zmm22, 8",
            "vpclmulqdq zmm22, zmm22, zmm30, 1",
            "vpternlogd zmm4, zmm22, zmm26, 0x96",
            "vpxorq zmm12, zmm12, zmm4",
            "vpslldq zmm22, zmm23, 8",
            "vpclmulqdq zmm23, zmm23, zmm30, 1",
            "vpternlogd zmm5, zmm23, zmm22, 0x96",
            "vpxorq zmm13, zmm13, zmm5",
            "vpslldq zmm22, zmm24, 8",
            "vpclmulqdq zmm23, zmm24, zmm30, 1",
            "vpternlogd zmm6, zmm23, zmm22, 0x96",
            "vpxorq zmm14, zmm14, zmm6",
            "vpslldq zmm22, zmm25, 8",
            "vpclmulqdq zmm23, zmm25, zmm30, 1",
            "vpternlogd zmm7, zmm23, zmm22, 0x96",
            "vpxorq zmm15, zmm15, zmm7",
            "vpclmulqdq zmm22, zmm8, zmm16, 0",
            "vpclmulqdq zmm23, zmm8, zmm17, 1",
            "vpternlogd zmm0, zmm22, zmm23, 0x96",
            "vpclmulqdq zmm22, zmm9, zmm16, 0",
            "vpclmulqdq zmm23, zmm9, zmm17, 1",
            "vpternlogd zmm1, zmm22, zmm23, 0x96",
            "vpclmulqdq zmm22, zmm10, zmm16, 0",
            "vpclmulqdq zmm23, zmm10, zmm17, 1",
            "vpternlogd zmm2, zmm22, zmm23, 0x96",
            "vpclmulqdq zmm22, zmm11, zmm16, 0",
            "vpclmulqdq zmm23, zmm11, zmm17, 1",
            "vpternlogd zmm3, zmm22, zmm23, 0x96",
            "vpclmulqdq zmm22, zmm8, zmm16, 16",
            "vpclmulqdq zmm23, zmm8, zmm17, 17",
            "vpclmulqdq zmm24, zmm9, zmm16, 16",
            "vpclmulqdq zmm25, zmm9, zmm17, 17",
            "vpclmulqdq zmm26, zmm10, zmm16, 16",
            "vpclmulqdq zmm27, zmm10, zmm17, 17",
            "vpclmulqdq zmm28, zmm11, zmm16, 16",
            "vpclmulqdq zmm29, zmm11, zmm17, 17",
            "vpxorq zmm22, zmm22, zmm23",
            "vpxorq zmm23, zmm24, zmm25",
            "vpxorq zmm24, zmm26, zmm27",
            "vpxorq zmm25, zmm28, zmm29",
            "vpxorq zmm8, zmm8, zmm0",
            "vpxorq zmm9, zmm9, zmm1",
            "vpxorq zmm10, zmm10, zmm2",
            "vpxorq zmm11, zmm11, zmm3",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*4]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*4]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*5]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*5]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*6]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*6]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*7]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*7]",
            ".endif",
            "vpclmulqdq zmm26, zmm14, zmm20, 0",
            "vpclmulqdq zmm27, zmm14, zmm21, 1",
            "vpternlogd zmm10, zmm26, zmm27, 0x96",
            "vpclmulqdq zmm26, zmm15, zmm20, 0",
            "vpclmulqdq zmm27, zmm15, zmm21, 1",
            "vpternlogd zmm11, zmm26, zmm27, 0x96",
            "vpclmulqdq zmm26, zmm14, zmm20, 16",
            "vpclmulqdq zmm27, zmm14, zmm21, 17",
            "vpclmulqdq zmm28, zmm15, zmm20, 16",
            "vpclmulqdq zmm29, zmm15, zmm21, 17",
            "vpternlogd zmm26, zmm27, zmm24, 0x96",
            "vpternlogd zmm28, zmm29, zmm25, 0x96",
            "vpslldq zmm27, zmm26, 8",
            "vpclmulqdq zmm26, zmm26, zmm30, 1",
            "vpternlogd zmm10, zmm26, zmm27, 0x96",
            "vpxorq zmm14, zmm14, zmm10",
            "vpslldq zmm26, zmm28, 8",
            "vpclmulqdq zmm27, zmm28, zmm30, 1",
            "vpternlogd zmm11, zmm27, zmm26, 0x96",
            "vpxorq zmm15, zmm15, zmm11",
            "vpclmulqdq zmm26, zmm6, zmm18, 0",
            "vpclmulqdq zmm27, zmm6, zmm19, 1",
            "vpternlogd zmm2, zmm26, zmm27, 0x96",
            "vpclmulqdq zmm26, zmm7, zmm18, 0",
            "vpclmulqdq zmm27, zmm7, zmm19, 1",
            "vpternlogd zmm3, zmm26, zmm27, 0x96",
            "vpclmulqdq zmm26, zmm6, zmm18, 16",
            "vpclmulqdq zmm27, zmm6, zmm19, 17",
            "vpclmulqdq zmm28, zmm7, zmm18, 16",
            "vpclmulqdq zmm29, zmm7, zmm19, 17",
            "vpternlogd zmm24, zmm26, zmm27, 0x96",
            "vpternlogd zmm25, zmm28, zmm29, 0x96",
            "vpslldq zmm26, zmm24, 8",
            "vpclmulqdq zmm24, zmm24, zmm30, 1",
            "vpternlogd zmm2, zmm24, zmm26, 0x96",
            "vpxorq zmm6, zmm6, zmm2",
            "vpslldq zmm24, zmm25, 8",
            "vpclmulqdq zmm25, zmm25, zmm30, 1",
            "vpternlogd zmm3, zmm25, zmm24, 0x96",
            "vpxorq zmm7, zmm7, zmm3",
            "vpclmulqdq zmm24, zmm12, zmm20, 0",
            "vpclmulqdq zmm25, zmm12, zmm21, 1",
            "vpternlogd zmm8, zmm24, zmm25, 0x96",
            "vpclmulqdq zmm24, zmm13, zmm20, 0",
            "vpclmulqdq zmm25, zmm13, zmm21, 1",
            "vpternlogd zmm9, zmm24, zmm25, 0x96",
            "vpclmulqdq zmm24, zmm12, zmm20, 16",
            "vpclmulqdq zmm25, zmm12, zmm21, 17",
            "vpclmulqdq zmm26, zmm13, zmm20, 16",
            "vpclmulqdq zmm27, zmm13, zmm21, 17",
            "vpternlogd zmm24, zmm25, zmm22, 0x96",
            "vpternlogd zmm26, zmm27, zmm23, 0x96",
            "vpxorq zmm12, zmm12, zmm8",
            "vpxorq zmm13, zmm13, zmm9",
            "vpclmulqdq zmm25, zmm4, zmm18, 0",
            "vpclmulqdq zmm27, zmm4, zmm19, 1",
            "vpternlogd zmm0, zmm25, zmm27, 0x96",
            "vpclmulqdq zmm25, zmm5, zmm18, 0",
            "vpclmulqdq zmm27, zmm5, zmm19, 1",
            "vpternlogd zmm1, zmm25, zmm27, 0x96",
            "vpclmulqdq zmm25, zmm4, zmm18, 16",
            "vpclmulqdq zmm27, zmm4, zmm19, 17",
            "vpclmulqdq zmm28, zmm5, zmm18, 16",
            "vpclmulqdq zmm29, zmm5, zmm19, 17",
            "vpternlogd zmm22, zmm25, zmm27, 0x96",
            "vpternlogd zmm23, zmm28, zmm29, 0x96",
            "vpxorq zmm4, zmm4, zmm0",
            "vpxorq zmm5, zmm5, zmm1",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*8]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*8]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*9]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*9]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*10]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*10]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*11]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*11]",
            ".endif",
            "vbroadcasti32x4 zmm25, [rsi+128]",
            "vbroadcasti32x4 zmm27, [rsi+144]",
            "vpclmulqdq zmm28, zmm7, zmm25, 0",
            "vpclmulqdq zmm29, zmm7, zmm27, 1",
            "vpternlogd zmm5, zmm28, zmm29, 0x96",
            "vpclmulqdq zmm28, zmm6, zmm25, 0",
            "vpclmulqdq zmm29, zmm6, zmm27, 1",
            "vpternlogd zmm4, zmm28, zmm29, 0x96",
            "vpclmulqdq zmm28, zmm7, zmm25, 16",
            "vpclmulqdq zmm29, zmm7, zmm27, 17",
            "vpclmulqdq zmm25, zmm6, zmm25, 16",
            "vpclmulqdq zmm27, zmm6, zmm27, 17",
            "vpternlogd zmm28, zmm29, zmm23, 0x96",
            "vpternlogd zmm25, zmm27, zmm22, 0x96",
            "vpslldq zmm27, zmm28, 8",
            "vpclmulqdq zmm28, zmm28, zmm30, 1",
            "vpternlogd zmm5, zmm28, zmm27, 0x96",
            "vpxorq zmm7, zmm7, zmm5",
            "vpxorq zmm6, zmm6, zmm4",
            "vbroadcasti32x4 zmm27, [rsi+96]",
            "vbroadcasti32x4 zmm28, [rsi+112]",
            "vpclmulqdq zmm29, zmm3, zmm27, 0",
            "vpclmulqdq zmm31, zmm3, zmm28, 1",
            "vpternlogd zmm1, zmm29, zmm31, 0x96",
            "vpclmulqdq zmm29, zmm2, zmm27, 0",
            "vpclmulqdq zmm31, zmm2, zmm28, 1",
            "vpternlogd zmm0, zmm29, zmm31, 0x96",
            "vpclmulqdq zmm29, zmm3, zmm27, 16",
            "vpclmulqdq zmm31, zmm3, zmm28, 17",
            "vpclmulqdq zmm27, zmm2, zmm27, 16",
            "vpclmulqdq zmm28, zmm2, zmm28, 17",
            "vpternlogd zmm23, zmm29, zmm31, 0x96",
            "vpternlogd zmm22, zmm27, zmm28, 0x96",
            "vpslldq zmm27, zmm23, 8",
            "vpclmulqdq zmm23, zmm23, zmm30, 1",
            "vpternlogd zmm1, zmm23, zmm27, 0x96",
            "vpxorq zmm3, zmm3, zmm1",
            "vpxorq zmm2, zmm2, zmm0",
            "vbroadcasti32x4 zmm23, [rsi+192]",
            "vbroadcasti32x4 zmm27, [rsi+208]",
            "vpclmulqdq zmm28, zmm15, zmm23, 0",
            "vpclmulqdq zmm29, zmm15, zmm27, 1",
            "vpternlogd zmm13, zmm28, zmm29, 0x96",
            "vpclmulqdq zmm28, zmm14, zmm23, 0",
            "vpclmulqdq zmm29, zmm14, zmm27, 1",
            "vpternlogd zmm12, zmm28, zmm29, 0x96",
            "vpclmulqdq zmm28, zmm15, zmm23, 16",
            "vpclmulqdq zmm29, zmm15, zmm27, 17",
            "vpclmulqdq zmm23, zmm14, zmm23, 16",
            "vpclmulqdq zmm27, zmm14, zmm27, 17",
            "vpternlogd zmm28, zmm29, zmm26, 0x96",
            "vpternlogd zmm23, zmm27, zmm24, 0x96",
            "vpslldq zmm27, zmm28, 8",
            "vpclmulqdq zmm28, zmm28, zmm30, 1",
            "vpternlogd zmm13, zmm28, zmm27, 0x96",
            "vpxorq zmm15, zmm15, zmm13",
            "vpxorq zmm14, zmm14, zmm12",
            "vbroadcasti32x4 zmm27, [rsi+160]",
            "vbroadcasti32x4 zmm28, [rsi+176]",
            "vpclmulqdq zmm29, zmm11, zmm27, 0",
            "vpclmulqdq zmm31, zmm11, zmm28, 1",
            "vpternlogd zmm9, zmm29, zmm31, 0x96",
            "vpclmulqdq zmm29, zmm10, zmm27, 0",
            "vpclmulqdq zmm31, zmm10, zmm28, 1",
            "vpternlogd zmm8, zmm29, zmm31, 0x96",
            "vpclmulqdq zmm29, zmm11, zmm27, 16",
            "vpclmulqdq zmm31, zmm11, zmm28, 17",
            "vpclmulqdq zmm27, zmm10, zmm27, 16",
            "vpclmulqdq zmm28, zmm10, zmm28, 17",
            "vpternlogd zmm26, zmm29, zmm31, 0x96",
            "vpternlogd zmm24, zmm27, zmm28, 0x96",
            "vpslldq zmm27, zmm26, 8",
            "vpclmulqdq zmm26, zmm26, zmm30, 1",
            "vpternlogd zmm9, zmm26, zmm27, 0x96",
            "vpxorq zmm11, zmm11, zmm9",
            "vpxorq zmm10, zmm10, zmm8",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*12]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*12]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*13]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*13]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*14]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*14]",
            ".endif",
            ".if {hint} == 1",
            "prefetcht0 [rcx+rax+{stride}*15]",
            ".elseif {hint} == 2",
            "prefetcht1 [rcx+rax+{stride}*15]",
            ".endif",
            "vbroadcasti32x4 zmm26, [rsi+256]",
            "vbroadcasti32x4 zmm27, [rsi+272]",
            "vpclmulqdq zmm28, zmm3, zmm26, 0",
            "vpclmulqdq zmm29, zmm3, zmm27, 1",
            "vpternlogd zmm2, zmm28, zmm29, 0x96",
            "vpclmulqdq zmm26, zmm3, zmm26, 16",
            "vpclmulqdq zmm27, zmm3, zmm27, 17",
            "vpternlogd zmm26, zmm27, zmm22, 0x96",
            "vbroadcasti32x4 zmm27, [rsi+224]",
            "vbroadcasti32x4 zmm28, [rsi+240]",
            "vpslldq zmm29, zmm26, 8",
            "vpclmulqdq zmm26, zmm26, zmm30, 1",
            "vpclmulqdq zmm31, zmm1, zmm27, 0",
            "vpternlogd zmm2, zmm26, zmm29, 0x96",
            "vpxorq zmm3, zmm3, zmm2",
            "vmovdqu64 [rdi+rax+{stride}*2], zmm2",
            "vmovdqu64 [rdi+rax+{stride}*3], zmm3",
            "vpclmulqdq zmm26, zmm1, zmm28, 1",
            "vpternlogd zmm0, zmm31, zmm26, 0x96",
            "vpclmulqdq zmm26, zmm1, zmm27, 16",
            "vpclmulqdq zmm27, zmm1, zmm28, 17",
            "vpternlogd zmm26, zmm27, zmm22, 0x96",
            "vpslldq zmm22, zmm26, 8",
            "vpclmulqdq zmm26, zmm26, zmm30, 1",
            "vpternlogd zmm0, zmm26, zmm22, 0x96",
            "vpxorq zmm1, zmm1, zmm0",
            "vmovdqu64 [rdi+rax+{stride}*0], zmm0",
            "vmovdqu64 [rdi+rax+{stride}*1], zmm1",
            "vbroadcasti32x4 zmm22, [rsi+320]",
            "vbroadcasti32x4 zmm26, [rsi+336]",
            "vpclmulqdq zmm27, zmm7, zmm22, 0",
            "vpclmulqdq zmm28, zmm7, zmm26, 1",
            "vpternlogd zmm6, zmm27, zmm28, 0x96",
            "vpclmulqdq zmm22, zmm7, zmm22, 16",
            "vpclmulqdq zmm26, zmm7, zmm26, 17",
            "vpternlogd zmm22, zmm26, zmm25, 0x96",
            "vbroadcasti32x4 zmm26, [rsi+288]",
            "vbroadcasti32x4 zmm27, [rsi+304]",
            "vpslldq zmm28, zmm22, 8",
            "vpclmulqdq zmm22, zmm22, zmm30, 1",
            "vpclmulqdq zmm29, zmm5, zmm26, 0",
            "vpternlogd zmm6, zmm22, zmm28, 0x96",
            "vpxorq zmm7, zmm7, zmm6",
            "vmovdqu64 [rdi+rax+{stride}*6], zmm6",
            "vmovdqu64 [rdi+rax+{stride}*7], zmm7",
            "vpclmulqdq zmm22, zmm5, zmm27, 1",
            "vpternlogd zmm4, zmm29, zmm22, 0x96",
            "vpclmulqdq zmm22, zmm5, zmm26, 16",
            "vpclmulqdq zmm26, zmm5, zmm27, 17",
            "vpternlogd zmm22, zmm26, zmm25, 0x96",
            "vpslldq zmm25, zmm22, 8",
            "vpclmulqdq zmm22, zmm22, zmm30, 1",
            "vpternlogd zmm4, zmm22, zmm25, 0x96",
            "vpxorq zmm5, zmm5, zmm4",
            "vmovdqu64 [rdi+rax+{stride}*4], zmm4",
            "vmovdqu64 [rdi+rax+{stride}*5], zmm5",
            "vbroadcasti32x4 zmm22, [rsi+384]",
            "vbroadcasti32x4 zmm25, [rsi+400]",
            "vpclmulqdq zmm26, zmm11, zmm22, 0",
            "vpclmulqdq zmm27, zmm11, zmm25, 1",
            "vpternlogd zmm10, zmm26, zmm27, 0x96",
            "vpclmulqdq zmm22, zmm11, zmm22, 16",
            "vpclmulqdq zmm25, zmm11, zmm25, 17",
            "vpternlogd zmm22, zmm25, zmm24, 0x96",
            "vbroadcasti32x4 zmm25, [rsi+352]",
            "vbroadcasti32x4 zmm26, [rsi+368]",
            "vpslldq zmm27, zmm22, 8",
            "vpclmulqdq zmm22, zmm22, zmm30, 1",
            "vpclmulqdq zmm28, zmm9, zmm25, 0",
            "vpternlogd zmm10, zmm22, zmm27, 0x96",
            "vpxorq zmm11, zmm11, zmm10",
            "vmovdqu64 [rdi+rax+{stride}*10], zmm10",
            "vmovdqu64 [rdi+rax+{stride}*11], zmm11",
            "vpclmulqdq zmm22, zmm9, zmm26, 1",
            "vpternlogd zmm8, zmm28, zmm22, 0x96",
            "vpclmulqdq zmm22, zmm9, zmm25, 16",
            "vpclmulqdq zmm25, zmm9, zmm26, 17",
            "vpternlogd zmm22, zmm25, zmm24, 0x96",
            "vpslldq zmm24, zmm22, 8",
            "vpclmulqdq zmm22, zmm22, zmm30, 1",
            "vpternlogd zmm8, zmm22, zmm24, 0x96",
            "vpxorq zmm9, zmm9, zmm8",
            "vmovdqu64 [rdi+rax+{stride}*8], zmm8",
            "vmovdqu64 [rdi+rax+{stride}*9], zmm9",
            "vbroadcasti32x4 zmm22, [rsi+448]",
            "vbroadcasti32x4 zmm24, [rsi+464]",
            "vpclmulqdq zmm25, zmm15, zmm22, 0",
            "vpclmulqdq zmm26, zmm15, zmm24, 1",
            "vpternlogd zmm14, zmm25, zmm26, 0x96",
            "vpclmulqdq zmm22, zmm15, zmm22, 16",
            "vpclmulqdq zmm24, zmm15, zmm24, 17",
            "vpternlogd zmm22, zmm24, zmm23, 0x96",
            "vbroadcasti32x4 zmm24, [rsi+416]",
            "vbroadcasti32x4 zmm25, [rsi+432]",
            "vpslldq zmm26, zmm22, 8",
            "vpclmulqdq zmm22, zmm22, zmm30, 1",
            "vpclmulqdq zmm27, zmm13, zmm24, 0",
            "vpternlogd zmm14, zmm22, zmm26, 0x96",
            "vpxorq zmm15, zmm15, zmm14",
            "vmovdqu64 [rdi+rax+{stride}*14], zmm14",
            "vmovdqu64 [rdi+rax+{stride}*15], zmm15",
            "vpclmulqdq zmm22, zmm13, zmm25, 1",
            "vpternlogd zmm12, zmm27, zmm22, 0x96",
            "vpclmulqdq zmm22, zmm13, zmm24, 16",
            "vpclmulqdq zmm24, zmm13, zmm25, 17",
            "vpternlogd zmm22, zmm24, zmm23, 0x96",
            "vpslldq zmm23, zmm22, 8",
            "vpclmulqdq zmm22, zmm22, zmm30, 1",
            "vpternlogd zmm12, zmm22, zmm23, 0x96",
            "vpxorq zmm13, zmm13, zmm12",
            "vmovdqu64 [rdi+rax+{stride}*12], zmm12",
            "vmovdqu64 [rdi+rax+{stride}*13], zmm13",
            "add rax, 64",
            "cmp rax, rdx",
            "jb 2b",
            "3:",
            stride = const S16 * NN * core::mem::size_of::<F128>(),
            hint = const H,
            ord = const ORD,
            in("rdi") ptr,
            in("rsi") prepared,
            in("rdx") vector_bytes,
            in("rcx") prefetch,
            out("rax") _,
            out("zmm0") _,
            out("zmm1") _,
            out("zmm2") _,
            out("zmm3") _,
            out("zmm4") _,
            out("zmm5") _,
            out("zmm6") _,
            out("zmm7") _,
            out("zmm8") _,
            out("zmm9") _,
            out("zmm10") _,
            out("zmm11") _,
            out("zmm12") _,
            out("zmm13") _,
            out("zmm14") _,
            out("zmm15") _,
            out("zmm16") _,
            out("zmm17") _,
            out("zmm18") _,
            out("zmm19") _,
            out("zmm20") _,
            out("zmm21") _,
            out("zmm22") _,
            out("zmm23") _,
            out("zmm24") _,
            out("zmm25") _,
            out("zmm26") _,
            out("zmm27") _,
            out("zmm28") _,
            out("zmm29") _,
            out("zmm30") _,
            out("zmm31") _,
            options(nostack),
        );
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_4layer_row`]. `S16`/`NN` are either 0
/// (use the runtime `sixteenth`/`num_ntts`) or the exact runtime values (the
/// shaped wrappers): distinct constants force a distinct monomorphization, so
/// the shaped forms get compile-time address arithmetic even though the
/// ninety-line body is never inlined into its wrapper.
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_4layer_row_impl<
    const DIET: bool,
    const PREPARED: bool,
    const H: u8,
    const S16: usize,
    const NN: usize,
>(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    active_lanes: usize,
    r: usize,
    twiddles: &[F128; 15],
    pf_r: usize,
    prepared: *const PreparedFour,
) {
    use core::arch::x86_64::*;

    // Shape substitution: a compile-time constant when the wrapper pins one
    // (S16/NN nonzero), the runtime argument otherwise. The wrappers only
    // pin values equal to the runtime shape, so this is the identity.
    let sixteenth = if S16 != 0 { S16 } else { sixteenth };
    let num_ntts = if NN != 0 { NN } else { num_ntts };

    // SAFETY: caller provides target features and pointer geometry.
    unsafe {
        #[cfg(target_feature = "avx512bw")]
        if DIET && PREPARED && S16 != 0 && NN != 0 {
            let lanes = active_lanes & !3;
            // Keep the hint within the current group or its checked successor.
            // A short rolling lead covers the next columns without fetching
            // a whole 1024-byte row ahead on the 64-column geometry.
            let pf_index = if H != 0 && pf_r > r && pf_r < sixteenth {
                let lead = ((*prepared).prefetch_lines * 4).min(num_ntts);
                pf_r * num_ntts - (num_ntts - lead)
            } else { pf_r * num_ntts };
            // Aliasing-aware load order only where the 16 rows share an L1D set, i.e. the row spacing
            // `S16 * NN * 16` bytes is a multiple of 4 KiB (the deep sweeps: 128 KiB and 8 KiB). The
            // contiguous-row shapes keep the original order. `FLOCK_NO_NTT_ALIAS_LOAD_ORDER=1` restores
            // the original order everywhere (exact same-binary A/B).
            if (S16 * NN) % 256 == 0 && super::super::ntt_alias_load_order_enabled() {
                butterfly_four_fixed_registers::<S16, NN, H, 1>(
                    ptr.add(r * num_ntts),
                    prepared,
                    lanes * core::mem::size_of::<F128>(),
                    ptr.add(pf_index),
                );
            } else {
                butterfly_four_fixed_registers::<S16, NN, H, 0>(
                    ptr.add(r * num_ntts),
                    prepared,
                    lanes * core::mem::size_of::<F128>(),
                    ptr.add(pf_index),
                );
            }
            if lanes < active_lanes {
                if lanes + 4 <= num_ntts && !masked_tail_disabled() {
                    butterfly_fused_4layer_masked_group(
                        ptr,
                        sixteenth,
                        num_ntts,
                        r,
                        lanes,
                        active_lanes,
                        prepared,
                    );
                } else {
                    butterfly_fused_4layer_scalar_tail(
                        ptr,
                        sixteenth,
                        num_ntts,
                        r,
                        lanes,
                        active_lanes,
                        twiddles,
                    );
                }
            }
            return;
        }
        // Broadcast (and, under DIET, x^64-companion) every twiddle ONCE per
        // row group: 15 setup CLMULs against 32 butterflies × ⌊lanes/4⌋ lane
        // steps of savings. Under DIET, keeping all 15 `(t, t·x^64)` pairs as
        // 30 live `__m512i` variables across the 16-ZMM `values` array exceeds
        // the 32-ZMM register file and forces LLVM to spill 28 ZMMs (1,792 B)
        // to the stack and reload 28 ZMMs (1,792 B) on every lane step.
        // Instead, keep the 3 top twiddle pairs (`tw_top[0..3]`, reused 8 and 4
        // times per lane step = 6 ZMMs) in ZMM registers and store the
        // remaining 12 `(t, t·x^64)` pairs in compact 32-byte `[__m128i; 2]`
        // slots (`384` bytes total vs `1,792` bytes), so LLVM folds each load
        // into a single 16-byte `vbroadcasti32x4` with zero ZMM stack spills!
        let zero = _mm512_setzero_si512();
        let mut tw_top = [(zero, zero); 3];
        let mut tw_compact = [(_mm_setzero_si128(), _mm_setzero_si128()); 12];
        if PREPARED {
            for i in 0..3 {
                let p = (*prepared).pairs[i];
                tw_top[i] = (
                    _mm512_broadcast_i32x4(_mm_set_epi64x(p[0].hi as i64, p[0].lo as i64)),
                    _mm512_broadcast_i32x4(_mm_set_epi64x(p[1].hi as i64, p[1].lo as i64)),
                );
            }
        } else if DIET {
            for i in 0..3 {
                tw_top[i] = tw_x4::<false, true>(twiddles[i]);
            }
            for i in 0..12 {
                let pair = tw_x4::<false, true>(twiddles[3 + i]);
                tw_compact[i] = (
                    _mm512_castsi512_si128(pair.0),
                    _mm512_castsi512_si128(pair.1),
                );
            }
        } else {
            for i in 0..3 {
                tw_top[i] = tw_x4::<false, false>(twiddles[i]);
            }
            for i in 0..12 {
                let pair = tw_x4::<false, false>(twiddles[3 + i]);
                tw_compact[i] = (
                    _mm512_castsi512_si128(pair.0),
                    _mm512_castsi512_si128(pair.1),
                );
            }
        }
        let tw_c_ptr = if PREPARED {
            (*prepared)
                .pairs
                .as_ptr()
                .add(3)
                .cast::<(__m128i, __m128i)>()
        } else {
            core::hint::black_box(tw_compact.as_ptr())
        };
        let row = |i: usize| ptr.add((i * sixteenth + r) * num_ntts);
        let pf_row = |i: usize| ptr.add((i * sixteenth + pf_r) * num_ntts) as *const i8;
        let lanes = active_lanes & !3;
        let mut lane = 0;
        while lane < lanes {
            // Hint DELIVERY, not hint content: the same sixteen lines of the
            // next row group are still requested exactly once per lane step,
            // four at a time at four points spaced through the body instead
            // of all sixteen back to back at its head. The incumbent burst
            // put sixteen prefetch uops on the load ports immediately in
            // front of the sixteen DEMAND loads of this lane step, which are
            // on the critical path; the three other hot prefetch sites in
            // this prover (`seed_pf_spread`, `zc_r1ab_pf_spread`,
            // `zc_tail_pf_spread`) already ship exactly this delivery and
            // each is worth several percent of its window. Architecturally
            // invisible: a prefetch moves no value.
            macro_rules! pf_quad {
                ($g:expr) => {{
                    if H != 0 {
                        let off = lane * core::mem::size_of::<F128>();
                        for i in (4 * $g)..(4 * $g + 4) {
                            let p = pf_row(i).add(off);
                            if H == 1 {
                                _mm_prefetch::<_MM_HINT_T0>(p);
                            } else {
                                _mm_prefetch::<_MM_HINT_T1>(p);
                            }
                        }
                    }
                }};
            }
            let mut values = [zero; 16];
            for (i, value) in values.iter_mut().enumerate() {
                *value = _mm512_loadu_si512(row(i).add(lane) as *const __m512i);
            }

            macro_rules! butterfly {
                ($u:expr, $v:expr, $twiddle:expr) => {{
                    let new_u =
                        _mm512_xor_si512(values[$u], mul_x4::<false, DIET>($twiddle, values[$v]));
                    values[$v] = _mm512_xor_si512(values[$v], new_u);
                    values[$u] = new_u;
                }};
            }

            pf_quad!(0);
            let outer = tw_top[0];
            for i in 0..8 {
                butterfly!(i, i + 8, outer);
            }
            pf_quad!(1);
            for s in 0..2 {
                let twiddle = tw_top[1 + s];
                for i in 0..4 {
                    butterfly!(8 * s + i, 8 * s + i + 4, twiddle);
                }
            }
            pf_quad!(2);
            for s in 0..4 {
                let p = tw_c_ptr.add(s) as *const __m128i;
                let twiddle = (
                    _mm512_broadcast_i32x4(_mm_loadu_si128(p)),
                    _mm512_broadcast_i32x4(_mm_loadu_si128(p.add(1))),
                );
                for i in 0..2 {
                    butterfly!(4 * s + i, 4 * s + i + 2, twiddle);
                }
            }
            pf_quad!(3);
            for s in 0..8 {
                let p = tw_c_ptr.add(4 + s) as *const __m128i;
                let twiddle = (
                    _mm512_broadcast_i32x4(_mm_loadu_si128(p)),
                    _mm512_broadcast_i32x4(_mm_loadu_si128(p.add(1))),
                );
                // This is the terminal layer: neither result participates in
                // another butterfly. Publish each pair immediately, shortening
                // live ranges instead of retaining sixteen output registers.
                let u = 2 * s;
                let v = u + 1;
                let new_u = _mm512_xor_si512(values[u], mul_x4::<false, DIET>(twiddle, values[v]));
                let new_v = _mm512_xor_si512(values[v], new_u);
                _mm512_storeu_si512(row(u).add(lane) as *mut __m512i, new_u);
                _mm512_storeu_si512(row(v).add(lane) as *mut __m512i, new_v);
            }
            lane += 4;
        }

        if lane < active_lanes {
            butterfly_fused_4layer_scalar_tail(
                ptr,
                sixteenth,
                num_ntts,
                r,
                lane,
                active_lanes,
                twiddles,
            );
        }
    }
}

/// `FLOCK_NO_NTT_MASKED_TAIL=1` restores the portable scalar lane tails of the
/// fused-four and fused-three deep kernels (same-binary A/B).
#[inline]
fn masked_tail_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_NTT_MASKED_TAIL").is_some())
}

/// One four-lane group `lane..lane+4` of a fused-four row group whose live
/// lanes end at `active_lanes` (`lane < active_lanes < lane + 4 <= num_ntts`).
/// Lanes `active_lanes..` are statically zero on these rows (the zero odd
/// tail), so they are loaded as zero (masked load), run through the same
/// network, and never written back (masked store): only lanes below
/// `active_lanes` change, with the same field values the scalar tail writes.
///
/// # Safety
/// `avx512f` + `vpclmulqdq`; the sixteen rows cover `lane + 4` lanes;
/// `prepared` holds `(t, t·x^64 mod p)` for the fifteen twiddles.
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_4layer_masked_group(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    lane: usize,
    active_lanes: usize,
    prepared: *const PreparedFour,
) {
    use core::arch::x86_64::*;
    unsafe {
        let live = active_lanes - lane;
        let mask: __mmask8 = ((1u32 << (2 * live)) - 1) as __mmask8;
        let row = |i: usize| ptr.add((i * sixteenth + r) * num_ntts + lane);
        let tw = |i: usize| -> TwX4 {
            let p = (*prepared).pairs[i];
            (
                _mm512_broadcast_i32x4(_mm_set_epi64x(p[0].hi as i64, p[0].lo as i64)),
                _mm512_broadcast_i32x4(_mm_set_epi64x(p[1].hi as i64, p[1].lo as i64)),
            )
        };
        let mut values = [_mm512_setzero_si512(); 16];
        for (i, value) in values.iter_mut().enumerate() {
            *value = _mm512_maskz_loadu_epi64(mask, row(i) as *const i64);
        }
        macro_rules! butterfly {
            ($u:expr, $v:expr, $t:expr) => {{
                let new_u = _mm512_xor_si512(values[$u], mul_x4::<false, true>($t, values[$v]));
                values[$v] = _mm512_xor_si512(values[$v], new_u);
                values[$u] = new_u;
            }};
        }
        let t0 = tw(0);
        for i in 0..8 {
            butterfly!(i, i + 8, t0);
        }
        for s in 0..2 {
            let t = tw(1 + s);
            for i in 0..4 {
                butterfly!(8 * s + i, 8 * s + i + 4, t);
            }
        }
        for s in 0..4 {
            let t = tw(3 + s);
            for i in 0..2 {
                butterfly!(4 * s + i, 4 * s + i + 2, t);
            }
        }
        for s in 0..8 {
            let t = tw(7 + s);
            butterfly!(2 * s, 2 * s + 1, t);
        }
        for (i, value) in values.iter().enumerate() {
            _mm512_mask_storeu_epi64(row(i) as *mut i64, mask, *value);
        }
    }
}

#[cold]
#[inline(never)]
unsafe fn butterfly_fused_4layer_scalar_tail(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    mut lane: usize,
    active_lanes: usize,
    twiddles: &[F128; 15],
) {
    unsafe {
        let row = |i: usize| ptr.add((i * sixteenth + r) * num_ntts);
        while lane < active_lanes {
            let mut values = [F128::ZERO; 16];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *row(i).add(lane);
            }
            super::portable::butterfly_fused_4layer(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *row(i).add(lane) = *value;
            }
            lane += 1;
        }
    }
}

/// Process one fused-three-layer group of eight CONSECUTIVE rows.
///
/// `ptr` addresses row 0; row `i` starts at `i · num_ntts`. Lanes
/// `0..dense_lanes` run the full 12-butterfly network; lanes
/// `dense_lanes..num_ntts` run the zero-odd-row specialization (see
/// [`super::portable::butterfly_fused_3layer_zero_odd`]).
///
/// # Safety
/// The caller guarantees `avx512f` + `vpclmulqdq`, that the eight rows are
/// valid and disjoint from any concurrently processed group,
/// `dense_lanes <= num_ntts`, and that rows 1, 3, 5 and 7 are zero on lanes
/// `dense_lanes..num_ntts`.
///
/// Kept out of line (like [`butterfly_fused_4layer_row`]) so the deep-pass
/// closure that calls it once per block does not inline two full
/// monomorphizations of a twelve-butterfly network into its own frame.
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_3layer_rows(
    ptr: *mut F128,
    num_ntts: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    // The fused-three sweep is the LAST group of the deep region, so its two
    // inner layers are the domain's two deepest — and every twiddle of the
    // two deepest standard-basis layers has a zero high limb (an exact
    // property of the fixed basis, pinned by
    // `deepest_standard_twiddle_layers_have_zero_high_limb`). Six of the
    // seven twiddles then take the 3-CLMUL `LOW` product instead of the
    // 5-CLMUL split form, which is eight of every twelve butterflies in the
    // group. The predicate is still checked on the values themselves, so a
    // non-final fused-three group simply keeps the general form.
    let low_inner = !low_twiddle_fused3_disabled() && twiddles[1..].iter().all(|t| t.hi == 0);
    // SAFETY: forwarded caller contract; `low_inner` proves the LOW
    // precondition for `twiddles[1..]` by inspection of the values.
    unsafe {
        match (mul_diet_disabled(), low_inner) {
            (true, false) => butterfly_fused_3layer_rows_impl::<false, false, 0, false, false>(
                ptr,
                num_ntts,
                dense_lanes,
                twiddles,
            ),
            (true, true) => butterfly_fused_3layer_rows_impl::<false, true, 0, false, false>(
                ptr,
                num_ntts,
                dense_lanes,
                twiddles,
            ),
            (false, false) => butterfly_fused_3layer_rows_impl::<true, false, 0, false, false>(
                ptr,
                num_ntts,
                dense_lanes,
                twiddles,
            ),
            (false, true) => butterfly_fused_3layer_rows_impl::<true, true, 0, false, false>(
                ptr,
                num_ntts,
                dense_lanes,
                twiddles,
            ),
        }
    }
}

/// Shape-monomorphized [`butterfly_fused_3layer_rows`] (`num_ntts == NN`):
/// the eight consecutive row addresses become base-plus-constant
/// displacements. Same body, same element order, bit-identical stores.
///
/// # Safety
/// Same contract as [`butterfly_fused_3layer_rows`], with `num_ntts == NN`.
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_3layer_rows_shaped<const NN: usize>(
    ptr: *mut F128,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    let short_twiddles = !low_twiddle_fused3_disabled();
    let low_inner = short_twiddles && twiddles[1..].iter().all(|t| t.hi == 0);
    // Select each outer short form from its exact high-limb value, without
    // assuming a particular basis or a fixed fraction of eligible blocks.
    let low_outer = short_twiddles && !low_outer_fused3_disabled() && twiddles[0].hi == 0;
    let high_one_outer = short_twiddles && twiddles[0].hi == 1 && !high_one_fused3_disabled();
    // SAFETY: forwarded caller contract; `low_inner` proves the LOW
    // precondition for `twiddles[1..]`; low_outer/high_one_outer establish
    // their respective outer preconditions. NN equals the runtime stride.
    unsafe {
        // Inspect the value, rather than assuming a particular standard
        // basis or domain size. Every other high limb keeps its prior path.
        if high_one_outer {
            match (mul_diet_disabled(), low_inner) {
                (true, false) => butterfly_fused_3layer_rows_impl::<false, false, NN, false, true>(
                    ptr,
                    NN,
                    dense_lanes,
                    twiddles,
                ),
                (true, true) => butterfly_fused_3layer_rows_impl::<false, true, NN, false, true>(
                    ptr,
                    NN,
                    dense_lanes,
                    twiddles,
                ),
                (false, false) => butterfly_fused_3layer_rows_impl::<true, false, NN, false, true>(
                    ptr,
                    NN,
                    dense_lanes,
                    twiddles,
                ),
                (false, true) => butterfly_fused_3layer_rows_impl::<true, true, NN, false, true>(
                    ptr,
                    NN,
                    dense_lanes,
                    twiddles,
                ),
            }
            return;
        }
        match (mul_diet_disabled(), low_inner) {
            (true, false) => {
                if low_outer {
                    butterfly_fused_3layer_rows_impl::<false, false, NN, true, false>(
                        ptr,
                        NN,
                        dense_lanes,
                        twiddles,
                    )
                } else {
                    butterfly_fused_3layer_rows_impl::<false, false, NN, false, false>(
                        ptr,
                        NN,
                        dense_lanes,
                        twiddles,
                    )
                }
            }
            (true, true) => {
                if low_outer {
                    butterfly_fused_3layer_rows_impl::<false, true, NN, true, false>(
                        ptr,
                        NN,
                        dense_lanes,
                        twiddles,
                    )
                } else {
                    butterfly_fused_3layer_rows_impl::<false, true, NN, false, false>(
                        ptr,
                        NN,
                        dense_lanes,
                        twiddles,
                    )
                }
            }
            (false, false) => {
                if low_outer {
                    butterfly_fused_3layer_rows_impl::<true, false, NN, true, false>(
                        ptr,
                        NN,
                        dense_lanes,
                        twiddles,
                    )
                } else {
                    butterfly_fused_3layer_rows_impl::<true, false, NN, false, false>(
                        ptr,
                        NN,
                        dense_lanes,
                        twiddles,
                    )
                }
            }
            (false, true) => {
                if low_outer {
                    butterfly_fused_3layer_rows_impl::<true, true, NN, true, false>(
                        ptr,
                        NN,
                        dense_lanes,
                        twiddles,
                    )
                } else {
                    butterfly_fused_3layer_rows_impl::<true, true, NN, false, false>(
                        ptr,
                        NN,
                        dense_lanes,
                        twiddles,
                    )
                }
            }
        }
    }
}

/// Restore the general product for outer twiddles whose high limb is zero.
/// Read once per process, outside every butterfly lane loop.
#[inline(never)]
fn low_outer_fused3_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_NTT_LOW_OUTER_FUSED3").is_some())
}

/// Restore the general product for outer twiddles whose high limb is one.
/// Read once per process, outside every butterfly lane loop.
#[inline(never)]
fn high_one_fused3_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_NTT_HIGH_ONE_FUSED3").is_some())
}

/// `FLOCK_NO_NTT_LOW_TWIDDLE_FUSED3=1` restores general products in all three
/// layers. The LOW_OUTER and HIGH_ONE switches disable just their respective
/// outer forms.
#[inline(never)]
fn low_twiddle_fused3_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_NTT_LOW_TWIDDLE_FUSED3").is_some())
}

/// # Safety
/// Same contract as [`butterfly_fused_3layer_rows`]. `NNC` is either 0 (use
/// the runtime `num_ntts`) or the exact runtime value (the shaped wrapper):
/// a distinct constant forces a distinct monomorphization with compile-time
/// row addressing.
/// `HIGH_ONE_OUTER` requires `twiddles[0].hi == 1`; the shaped dispatcher
/// verifies it before entering that specialization.
// Keep specialization bodies separate. Inlining every DIET/LOW/NNC arm
// into one dispatcher gives all calls its union-sized stack frame and makes
// the hot kernel share an instruction footprint with unused alternatives.
// This preserves every arithmetic operation and all dispatch preconditions.
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_3layer_rows_impl<
    const DIET: bool,
    const LOW_INNER: bool,
    const NNC: usize,
    const LOW_OUTER: bool,
    const HIGH_ONE_OUTER: bool,
>(
    ptr: *mut F128,
    num_ntts: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    use core::arch::x86_64::*;

    // Shape substitution (identity: the wrapper only pins the runtime value).
    let num_ntts = if NNC != 0 { NNC } else { num_ntts };
    debug_assert!(!HIGH_ONE_OUTER || twiddles[0].hi == 1);

    // SAFETY: caller provides target features and pointer geometry.
    unsafe {
        // Seven broadcasts (plus their x^64 companions under DIET) hoisted
        // out of the lane loop: 8 rows stay live in registers across all
        // three layers, so a row is loaded once and stored once for 12
        // butterflies instead of the fused-two + single-layer pair's two
        // loads and two stores for the same 12.
        let zero = _mm512_setzero_si512();
        let mut tw = [(zero, zero); 7];
        // The high-one product needs only the original twiddle broadcast.
        tw[0] = if HIGH_ONE_OUTER {
            tw_x4::<false, false>(twiddles[0])
        } else {
            tw_x4::<LOW_OUTER, DIET>(twiddles[0])
        };
        for (slot, value) in tw[1..].iter_mut().zip(twiddles[1..].iter()) {
            *slot = tw_x4::<LOW_INNER, DIET>(*value);
        }
        let row = |i: usize| ptr.add(i * num_ntts);
        let mut lane = 0;

        macro_rules! butterfly {
            ($values:ident, $u:expr, $v:expr, $twiddle:expr, $low:expr) => {{
                butterfly!($values, $u, $v, $twiddle, $low, false);
            }};
            ($values:ident, $u:expr, $v:expr, $twiddle:expr, $low:expr, $high_one:expr) => {{
                let product = if $high_one {
                    mul_x4_high_one($twiddle, $values[$v])
                } else {
                    mul_x4::<$low, DIET>($twiddle, $values[$v])
                };
                let new_u = _mm512_xor_si512($values[$u], product);
                $values[$v] = _mm512_xor_si512($values[$v], new_u);
                $values[$u] = new_u;
            }};
        }
        // Two lane chunks per iteration: 16 live row registers + 14 twiddle
        // registers still fit the 32 zmm file, and doubling the independent
        // butterflies per layer (4 -> 8) covers the CLMUL latency and doubles
        // the outstanding row misses.
        // Dense-loop butterfly with lazy x^64 folds (diet forms only): the
        // product's x^64-position half accumulates in `$m[$u]`, `v ^= u`
        // shares it, and it is folded when `$fold` (the row is a multiplicand
        // in the next layer, or this is the last layer). Eight folds per four
        // lanes instead of twelve; same field elements.
        macro_rules! lazy_butterfly {
            ($p:expr, $m:expr, $u:expr, $v:expr, $twiddle:expr, $low:expr, $high_one:expr, $fold:expr) => {{
                if DIET {
                    let (lo, hi) = if $high_one {
                        mul_x4_high_one_parts($twiddle, $p[$v])
                    } else {
                        mul_x4_parts::<$low>($twiddle, $p[$v])
                    };
                    $m[$u] = _mm512_xor_si512($m[$u], hi);
                    let mut new_u = _mm512_xor_si512($p[$u], lo);
                    if $fold {
                        new_u = _mm512_xor_si512(new_u, fold_x64($m[$u]));
                        $m[$u] = zero;
                    }
                    $p[$v] = _mm512_xor_si512($p[$v], new_u);
                    $m[$v] = _mm512_xor_si512($m[$v], $m[$u]);
                    $p[$u] = new_u;
                } else {
                    let product = if $high_one {
                        mul_x4_high_one($twiddle, $p[$v])
                    } else {
                        mul_x4::<$low, DIET>($twiddle, $p[$v])
                    };
                    let new_u = _mm512_xor_si512($p[$u], product);
                    $p[$v] = _mm512_xor_si512($p[$v], new_u);
                    $p[$u] = new_u;
                }
            }};
        }

        while lane + 8 <= dense_lanes {
            let mut values = [[zero; 8]; 2];
            for (c, chunk) in values.iter_mut().enumerate() {
                for (i, value) in chunk.iter_mut().enumerate() {
                    *value = _mm512_loadu_si512(row(i).add(lane + 4 * c) as *const __m512i);
                }
            }

            let outer = tw[0];
            let mut pending = [[zero; 8]; 2];
            for i in 0..4 {
                for c in 0..2 {
                    lazy_butterfly!(
                        values[c],
                        pending[c],
                        i,
                        i + 4,
                        outer,
                        LOW_OUTER,
                        HIGH_ONE_OUTER,
                        i & 2 != 0
                    );
                }
            }
            for s in 0..2 {
                let twiddle = tw[1 + s];
                for i in 0..2 {
                    for c in 0..2 {
                        lazy_butterfly!(
                            values[c],
                            pending[c],
                            4 * s + i,
                            4 * s + i + 2,
                            twiddle,
                            LOW_INNER,
                            false,
                            i & 1 != 0
                        );
                    }
                }
            }
            for s in 0..4 {
                for c in 0..2 {
                    lazy_butterfly!(
                        values[c],
                        pending[c],
                        2 * s,
                        2 * s + 1,
                        tw[3 + s],
                        LOW_INNER,
                        false,
                        true
                    );
                }
            }

            for (c, chunk) in values.iter().enumerate() {
                for (i, value) in chunk.iter().enumerate() {
                    _mm512_storeu_si512(row(i).add(lane + 4 * c) as *mut __m512i, *value);
                }
            }
            lane += 8;
        }
        while lane + 4 <= dense_lanes {
            let mut values = [zero; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = _mm512_loadu_si512(row(i).add(lane) as *const __m512i);
            }

            let outer = tw[0];
            let mut pending = [zero; 8];
            for i in 0..4 {
                lazy_butterfly!(
                    values,
                    pending,
                    i,
                    i + 4,
                    outer,
                    LOW_OUTER,
                    HIGH_ONE_OUTER,
                    i & 2 != 0
                );
            }
            for s in 0..2 {
                let twiddle = tw[1 + s];
                for i in 0..2 {
                    lazy_butterfly!(
                        values,
                        pending,
                        4 * s + i,
                        4 * s + i + 2,
                        twiddle,
                        LOW_INNER,
                        false,
                        i & 1 != 0
                    );
                }
            }
            for s in 0..4 {
                lazy_butterfly!(
                    values,
                    pending,
                    2 * s,
                    2 * s + 1,
                    tw[3 + s],
                    LOW_INNER,
                    false,
                    true
                );
            }

            for (i, value) in values.iter().enumerate() {
                _mm512_storeu_si512(row(i).add(lane) as *mut __m512i, *value);
            }
            lane += 4;
        }
        // Mixed group straddling `dense_lanes` (ranked: lanes 56..59 with
        // dense_lanes = 57): one vector group instead of eight portable
        // scalar lanes. Same butterflies, same twiddles, same values.
        if NNC != 0 && lane < dense_lanes && lane + 4 <= num_ntts && !masked_tail_disabled() {
            // Rows 1, 3, 5, 7 are zero on lanes `dense_lanes..`: load them
            // masked, so the dense network sees exactly the true inputs.
            let odd_mask: __mmask8 = ((1u32 << (2 * (dense_lanes - lane))) - 1) as __mmask8;
            let mut values = [zero; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = if i & 1 == 0 {
                    _mm512_loadu_si512(row(i).add(lane) as *const __m512i)
                } else {
                    _mm512_maskz_loadu_epi64(odd_mask, row(i).add(lane) as *const i64)
                };
            }

            let outer = tw[0];
            let mut pending = [zero; 8];
            for i in 0..4 {
                lazy_butterfly!(
                    values,
                    pending,
                    i,
                    i + 4,
                    outer,
                    LOW_OUTER,
                    HIGH_ONE_OUTER,
                    i & 2 != 0
                );
            }
            for s in 0..2 {
                let twiddle = tw[1 + s];
                for i in 0..2 {
                    lazy_butterfly!(
                        values,
                        pending,
                        4 * s + i,
                        4 * s + i + 2,
                        twiddle,
                        LOW_INNER,
                        false,
                        i & 1 != 0
                    );
                }
            }
            for s in 0..4 {
                lazy_butterfly!(
                    values,
                    pending,
                    2 * s,
                    2 * s + 1,
                    tw[3 + s],
                    LOW_INNER,
                    false,
                    true
                );
            }

            for (i, value) in values.iter().enumerate() {
                _mm512_storeu_si512(row(i).add(lane) as *mut __m512i, *value);
            }
            lane += 4;
        }
        if (NNC == 0 || dense_lanes & 3 != 0) && lane < dense_lanes {
            // Recompute or fall through to the cold scalar tail only when
            // `dense_lanes` is not a multiple of 4 (never on shaped `NNC = 64`).
            if NNC != 0 {
                butterfly_fused_3layer_dense_and_zero_odd_scalar_tail(
                    ptr,
                    num_ntts,
                    lane,
                    dense_lanes,
                    twiddles,
                );
                return;
            }
            lane = butterfly_fused_3layer_dense_scalar_tail(
                ptr,
                num_ntts,
                lane,
                dense_lanes,
                twiddles,
            );
        }

        // Published zero tail: rows 1, 3, 5, 7 are zero here, so only the
        // four even rows are read and the deepest layer is a copy.
        while lane + 4 <= num_ntts {
            let mut values = [zero; 8];
            for i in 0..4 {
                values[2 * i] = _mm512_loadu_si512(row(2 * i).add(lane) as *const __m512i);
            }
            let outer = tw[0];
            butterfly!(values, 0, 4, outer, LOW_OUTER, HIGH_ONE_OUTER);
            butterfly!(values, 2, 6, outer, LOW_OUTER, HIGH_ONE_OUTER);
            butterfly!(values, 0, 2, tw[1], LOW_INNER);
            butterfly!(values, 4, 6, tw[2], LOW_INNER);
            for i in 0..4 {
                let v = values[2 * i];
                _mm512_storeu_si512(row(2 * i).add(lane) as *mut __m512i, v);
                _mm512_storeu_si512(row(2 * i + 1).add(lane) as *mut __m512i, v);
            }
            lane += 4;
        }
        if NNC == 0 && lane < num_ntts {
            butterfly_fused_3layer_zero_odd_scalar_tail(ptr, num_ntts, lane, twiddles);
        }
    }
}

#[cold]
#[inline(never)]
unsafe fn butterfly_fused_3layer_dense_and_zero_odd_scalar_tail(
    ptr: *mut F128,
    num_ntts: usize,
    lane: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    unsafe {
        let lane =
            butterfly_fused_3layer_dense_scalar_tail(ptr, num_ntts, lane, dense_lanes, twiddles);
        butterfly_fused_3layer_zero_odd_scalar_tail(ptr, num_ntts, lane, twiddles);
    }
}

#[cold]
#[inline(never)]
unsafe fn butterfly_fused_3layer_dense_scalar_tail(
    ptr: *mut F128,
    num_ntts: usize,
    mut lane: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) -> usize {
    unsafe {
        let row = |i: usize| ptr.add(i * num_ntts);
        while lane < dense_lanes {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *row(i).add(lane);
            }
            super::portable::butterfly_fused_3layer(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *row(i).add(lane) = *value;
            }
            lane += 1;
        }
        lane
    }
}

#[cold]
#[inline(never)]
unsafe fn butterfly_fused_3layer_zero_odd_scalar_tail(
    ptr: *mut F128,
    num_ntts: usize,
    mut lane: usize,
    twiddles: &[F128; 7],
) {
    unsafe {
        let row = |i: usize| ptr.add(i * num_ntts);
        while lane < num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *row(i).add(lane);
            }
            super::portable::butterfly_fused_3layer_zero_odd(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *row(i).add(lane) = *value;
            }
            lane += 1;
        }
    }
}

#[cfg(test)]
mod diet_tests {
    use super::*;
    use crate::field::F128;

    fn rng(seed: u64) -> impl FnMut() -> F128 {
        let mut st = seed | 1;
        move || {
            let mut n = || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st
            };
            F128 { lo: n(), hi: n() }
        }
    }

    #[target_feature(enable = "avx512f,vpclmulqdq")]
    unsafe fn check_direct_publish_case<
        const OUTER_LOW: bool,
        const INNER_LOW: bool,
        const DIET: bool,
        const ALIGNED_ZMM: bool,
    >(
        lanes: usize,
        residue: usize,
    ) {
        use core::arch::x86_64::_mm_sfence;

        let seed = 0xD1EC_7000
            ^ ((OUTER_LOW as u64) << 1)
            ^ ((INNER_LOW as u64) << 2)
            ^ ((DIET as u64) << 3)
            ^ ((lanes as u64) << 8)
            ^ ((residue as u64) << 16);
        let mut next = rng(seed);
        let mut src: Vec<F128> = (0..4 * 64).map(|_| next()).collect();
        // Poison the omitted scratch suffix. Production's corresponding values
        // are zero; poison proves the direct kernel has no stale-scratch reload
        // dependency and publishes the contractual zero value itself.
        for row in 0..4 {
            for lane in lanes..64 {
                src[row * 64 + lane] = F128 {
                    lo: 0xA11C_E000 + (row * 64 + lane) as u64,
                    hi: 0x5A11_0000 + (row * 64 + lane) as u64,
                };
            }
        }

        let mut t_outer = next();
        let mut t_inner_a = next();
        let mut t_inner_b = next();
        if OUTER_LOW {
            t_outer.hi = 0;
        } else {
            t_outer.hi |= 1;
        }
        if INNER_LOW {
            t_inner_a.hi = 0;
            t_inner_b.hi = 0;
        } else {
            t_inner_a.hi |= 1;
            t_inner_b.hi |= 1;
        }

        let mut want = src.clone();
        let (a, rest) = want.split_at_mut(64);
        let (b, rest) = rest.split_at_mut(64);
        let (c, d) = rest.split_at_mut(64);
        // SAFETY: four equal active prefixes; target features are carried by
        // this helper and the const low-twiddle preconditions were established.
        unsafe {
            butterfly_fused_2layer_impl::<OUTER_LOW, INNER_LOW, DIET>(
                &mut a[..lanes],
                &mut b[..lanes],
                &mut c[..lanes],
                &mut d[..lanes],
                t_outer,
                t_inner_a,
                t_inner_b,
            );
        }
        for row in 0..4 {
            want[row * 64 + lanes..(row + 1) * 64].fill(F128::ZERO);
        }

        const DST_STRIDE: usize = 80;
        let junk = F128 {
            lo: 0xDEAD_DEAD_DEAD_DEAD,
            hi: 0xBAD0_BAD0_BAD0_BAD0,
        };
        let mut dst = vec![junk; 4 * DST_STRIDE + 4];
        let base = dst.as_mut_ptr();
        let offset = (0..4)
            .find(|&offset| unsafe { base.add(offset) as usize % 64 == residue })
            .expect("four F128 offsets cover every 16-byte residue");
        let dst_a = unsafe { base.add(offset) };
        let dst_b = unsafe { base.add(offset + DST_STRIDE) };
        let dst_c = unsafe { base.add(offset + 2 * DST_STRIDE) };
        let dst_d = unsafe { base.add(offset + 3 * DST_STRIDE) };

        // SAFETY: source rows are 64 elements, destinations are disjoint,
        // 16-byte aligned and cover 64 elements, and the two allocations do not
        // overlap. The task-level production caller owns the same sfence.
        unsafe {
            butterfly_fused_2layer_publish_nt_impl::<OUTER_LOW, INNER_LOW, DIET, ALIGNED_ZMM>(
                src.as_ptr(),
                64,
                dst_a,
                dst_b,
                dst_c,
                dst_d,
                lanes,
                t_outer,
                t_inner_a,
                t_inner_b,
            );
            _mm_sfence();
            for (row, out) in [dst_a, dst_b, dst_c, dst_d].into_iter().enumerate() {
                assert_eq!(
                    core::slice::from_raw_parts(out, 64),
                    &want[row * 64..(row + 1) * 64],
                    "lanes={lanes} residue={residue} row={row} outer_low={OUTER_LOW} inner_low={INNER_LOW} diet={DIET}",
                );
            }
        }
    }

    /// Direct NT publication is byte-identical to the incumbent final fused2
    /// store, including the contractual-zero 60..64 suffix and every 16-byte-aligned
    /// pool-base residue, for all low-twiddle and mul-diet specializations.
    #[test]
    fn direct_publish_matches_in_place_fused2() {
        // SAFETY: this module and test binary exist only when the target has
        // AVX-512F + VPCLMULQDQ.
        unsafe {
            for lanes in [60, 64] {
                for residue in [0, 16, 32, 48] {
                    check_direct_publish_case::<false, false, false, false>(lanes, residue);
                    check_direct_publish_case::<false, false, true, false>(lanes, residue);
                    check_direct_publish_case::<false, true, false, false>(lanes, residue);
                    check_direct_publish_case::<false, true, true, false>(lanes, residue);
                    check_direct_publish_case::<true, false, false, false>(lanes, residue);
                    check_direct_publish_case::<true, false, true, false>(lanes, residue);
                    check_direct_publish_case::<true, true, false, false>(lanes, residue);
                    check_direct_publish_case::<true, true, true, false>(lanes, residue);
                    if residue == 0 {
                        check_direct_publish_case::<false, false, false, true>(lanes, residue);
                        check_direct_publish_case::<false, false, true, true>(lanes, residue);
                        check_direct_publish_case::<false, true, false, true>(lanes, residue);
                        check_direct_publish_case::<false, true, true, true>(lanes, residue);
                        check_direct_publish_case::<true, false, false, true>(lanes, residue);
                        check_direct_publish_case::<true, false, true, true>(lanes, residue);
                        check_direct_publish_case::<true, true, false, true>(lanes, residue);
                        check_direct_publish_case::<true, true, true, true>(lanes, residue);
                    }
                }
            }
        }
    }

    /// The fused-three sweep's low-inner specialization must be
    /// bit-identical to the general product form and to the portable
    /// reference, for every lane path (8-wide, 4-wide, scalar tail) and for
    /// both `DIET` arms.
    #[test]
    fn fused3_low_inner_matches_general() {
        let mut next = rng(0x0BAD_C0DE);
        for (num_ntts, dense_lanes) in [(64usize, 64usize), (67, 67), (64, 48), (67, 50)] {
            let mut base = vec![F128::ZERO; 8 * num_ntts];
            for v in base.iter_mut() {
                *v = next();
            }
            // Beyond `dense_lanes` the published tail is zero on odd rows —
            // the kernel's zero-odd contract for those lanes.
            for lane in dense_lanes..num_ntts {
                for i in (1..8).step_by(2) {
                    base[i * num_ntts + lane] = F128::ZERO;
                }
            }
            // Slot 0 is the group's outer layer (a general twiddle); slots
            // 1..7 are the two deepest layers, whose production twiddles
            // always have a zero high limb.
            let mut twiddles = [F128::ZERO; 7];
            twiddles[0] = next();
            for t in twiddles[1..].iter_mut() {
                let v = next();
                *t = F128 { lo: v.lo, hi: 0 };
            }

            let mut want = base.clone();
            for lane in 0..num_ntts {
                let mut values = [F128::ZERO; 8];
                for (i, value) in values.iter_mut().enumerate() {
                    *value = want[i * num_ntts + lane];
                }
                if lane < dense_lanes {
                    crate::ntt::additive_ntt_f128::kernels::portable::butterfly_fused_3layer(
                        &mut values,
                        &twiddles,
                    );
                } else {
                    crate::ntt::additive_ntt_f128::kernels::portable::butterfly_fused_3layer_zero_odd(&mut values, &twiddles);
                }
                for (i, value) in values.iter().enumerate() {
                    want[i * num_ntts + lane] = *value;
                }
            }

            for (diet, low) in [(false, false), (false, true), (true, false), (true, true)] {
                let mut got = base.clone();
                // SAFETY: this module compiles only with avx512f+vpclmulqdq;
                // the buffer is 8 rows of `num_ntts` lanes, and every
                // `twiddles[1..]` entry has a zero high limb as the LOW arm
                // requires.
                unsafe {
                    match (diet, low) {
                        (false, false) => {
                            butterfly_fused_3layer_rows_impl::<false, false, 0, false, false>(
                                got.as_mut_ptr(),
                                num_ntts,
                                dense_lanes,
                                &twiddles,
                            )
                        }
                        (false, true) => {
                            butterfly_fused_3layer_rows_impl::<false, true, 0, false, false>(
                                got.as_mut_ptr(),
                                num_ntts,
                                dense_lanes,
                                &twiddles,
                            )
                        }
                        (true, false) => {
                            butterfly_fused_3layer_rows_impl::<true, false, 0, false, false>(
                                got.as_mut_ptr(),
                                num_ntts,
                                dense_lanes,
                                &twiddles,
                            )
                        }
                        (true, true) => {
                            butterfly_fused_3layer_rows_impl::<true, true, 0, false, false>(
                                got.as_mut_ptr(),
                                num_ntts,
                                dense_lanes,
                                &twiddles,
                            )
                        }
                    }
                }
                assert_eq!(
                    got, want,
                    "num_ntts={num_ntts} dense={dense_lanes} diet={diet} low={low}"
                );
            }
        }
    }

    /// `ghash_shift64_x4` really is multiplication by `x^64`.
    #[test]
    fn shift64_matches_scalar() {
        use crate::field::gf2_128::x86_64::ghash_shift64_x4;
        use core::arch::x86_64::*;

        let mut next = rng(0x51F7_7EED);
        let x64 = F128 { lo: 0, hi: 1 };
        // SAFETY: this module only compiles with avx512f+vpclmulqdq statically
        // enabled (see the cfg gate on `mod x86_64`).
        unsafe {
            for _ in 0..512 {
                let t = next();
                let tv = _mm512_broadcast_i32x4(_mm_set_epi64x(t.hi as i64, t.lo as i64));
                let mut got = [0u64; 8];
                _mm512_storeu_si512(got.as_mut_ptr() as *mut __m512i, ghash_shift64_x4(tv));
                let want = t * x64;
                for i in 0..4 {
                    assert_eq!((got[2 * i], got[2 * i + 1]), (want.lo, want.hi), "t={t:?}");
                }
            }
        }
    }

    /// The 5-CLMUL split product must equal the incumbent 6-CLMUL product in
    /// every lane — and the scalar reference product — for arbitrary twiddles
    /// and values, including the degenerate zero/one/low twiddles the sparse
    /// seed layers carry.
    #[test]
    fn split_product_matches_incumbent() {
        use crate::field::gf2_128::x86_64::{ghash_mul_x4, ghash_mul_x4_split, ghash_shift64_x4};
        use core::arch::x86_64::*;

        let mut next = rng(0xD1E7_5EED);
        // SAFETY: module is statically gated on avx512f+vpclmulqdq.
        unsafe {
            let mut check = |t: F128, v: [F128; 4]| {
                let tv = _mm512_broadcast_i32x4(_mm_set_epi64x(t.hi as i64, t.lo as i64));
                let vv = _mm512_set_epi64(
                    v[3].hi as i64,
                    v[3].lo as i64,
                    v[2].hi as i64,
                    v[2].lo as i64,
                    v[1].hi as i64,
                    v[1].lo as i64,
                    v[0].hi as i64,
                    v[0].lo as i64,
                );
                let want = ghash_mul_x4(tv, vv);
                let got = ghash_mul_x4_split(vv, tv, ghash_shift64_x4(tv));
                let mut w = [0u64; 8];
                let mut g = [0u64; 8];
                _mm512_storeu_si512(w.as_mut_ptr() as *mut __m512i, want);
                _mm512_storeu_si512(g.as_mut_ptr() as *mut __m512i, got);
                assert_eq!(w, g, "t={t:?} v={v:?}");
                for (i, vi) in v.iter().enumerate() {
                    let p = *vi * t;
                    assert_eq!((g[2 * i], g[2 * i + 1]), (p.lo, p.hi), "lane {i} t={t:?}");
                }
            };
            for _ in 0..512 {
                let t = next();
                check(t, [next(), next(), next(), next()]);
            }
            let low = F128 {
                lo: next().lo,
                hi: 0,
            };
            let high = F128 {
                lo: 0,
                hi: next().hi,
            };
            for t in [
                F128::ZERO,
                F128::ONE,
                F128 { lo: 0, hi: 1 },
                F128 {
                    lo: u64::MAX,
                    hi: u64::MAX,
                },
                low,
                high,
            ] {
                check(
                    t,
                    [
                        F128::ZERO,
                        F128::ONE,
                        next(),
                        F128 {
                            lo: u64::MAX,
                            hi: u64::MAX,
                        },
                    ],
                );
            }
        }
    }

    /// Every AVX-512 butterfly kernel must be bit-identical with the diet
    /// multiply ON and OFF, over random twiddles, random data, and lane counts
    /// that exercise both the vector body and its scalar tail.
    #[test]
    fn kernels_match_incumbent() {
        let mut next = rng(0xFEED_FACE);

        for len in [1usize, 3, 4, 5, 7, 8, 16, 64] {
            // --- row pair, general and LOW twiddle ------------------------
            let top: Vec<F128> = (0..len).map(|_| next()).collect();
            let bot: Vec<F128> = (0..len).map(|_| next()).collect();
            for twiddle in [
                next(),
                F128 {
                    lo: next().lo,
                    hi: 0,
                },
            ] {
                let run_pair = |diet: bool| {
                    let (mut t, mut b) = (top.clone(), bot.clone());
                    // SAFETY: statically gated features; twiddle.hi == 0 is
                    // never asserted here (LOW = false in both arms).
                    unsafe {
                        if diet {
                            butterfly_row_pair_impl::<false, true>(&mut t, &mut b, twiddle);
                        } else {
                            butterfly_row_pair_impl::<false, false>(&mut t, &mut b, twiddle);
                        }
                    }
                    (t, b)
                };
                assert_eq!(run_pair(true), run_pair(false), "row_pair len={len}");
            }
            // The LOW monomorphization must also agree with the diet general
            // one on a zero-high-limb twiddle.
            let low_tw = F128 {
                lo: next().lo,
                hi: 0,
            };
            let run_low = |low: bool| {
                let (mut t, mut b) = (top.clone(), bot.clone());
                // SAFETY: statically gated features; low_tw.hi == 0.
                unsafe {
                    if low {
                        butterfly_row_pair_impl::<true, true>(&mut t, &mut b, low_tw);
                    } else {
                        butterfly_row_pair_impl::<false, true>(&mut t, &mut b, low_tw);
                    }
                }
                (t, b)
            };
            assert_eq!(run_low(true), run_low(false), "row_pair low len={len}");

            // --- fused two-layer, in place --------------------------------
            let (t_outer, t_a, t_b) = (next(), next(), next());
            let rows: Vec<Vec<F128>> = (0..4).map(|_| (0..len).map(|_| next()).collect()).collect();
            let run_fused2 = |diet: bool| {
                let mut r = rows.clone();
                let (a, rest) = r.split_at_mut(1);
                let (b, rest) = rest.split_at_mut(1);
                let (c, d) = rest.split_at_mut(1);
                // SAFETY: statically gated features; equal row lengths.
                unsafe {
                    if diet {
                        butterfly_fused_2layer_impl::<false, false, true>(
                            &mut a[0], &mut b[0], &mut c[0], &mut d[0], t_outer, t_a, t_b,
                        );
                    } else {
                        butterfly_fused_2layer_impl::<false, false, false>(
                            &mut a[0], &mut b[0], &mut c[0], &mut d[0], t_outer, t_a, t_b,
                        );
                    }
                }
                r
            };
            assert_eq!(run_fused2(true), run_fused2(false), "fused2 len={len}");

            // --- fused two-layer, out of place (dense + sparse) ------------
            let src: Vec<F128> = (0..4 * len).map(|_| next()).collect();
            let tw3 = [next(), next(), next()];
            let run_from = |diet: bool| {
                let mut dst = vec![F128::ZERO; 4 * len];
                // SAFETY: 4 rows of `len` lanes each, src/dst disjoint.
                unsafe {
                    if diet {
                        butterfly_fused_2layer_row_from_geo_impl::<false, true, false>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            &tw3,
                        );
                    } else {
                        butterfly_fused_2layer_row_from_geo_impl::<false, false, false>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            &tw3,
                        );
                    }
                }
                dst
            };
            assert_eq!(run_from(true), run_from(false), "row_from len={len}");

            let low_tw3 = [
                F128 {
                    lo: next().lo,
                    hi: 0,
                },
                next(),
                next(),
            ];
            let run_from_outer_low = |outer_low: bool| {
                let mut dst = vec![F128::ZERO; 4 * len];
                // SAFETY: 4 rows of `len` lanes each, src/dst disjoint, and
                // low_tw3[0].hi == 0 in the OUTER_LOW arm.
                unsafe {
                    if outer_low {
                        butterfly_fused_2layer_row_from_geo_impl::<true, true, false>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            &low_tw3,
                        );
                    } else {
                        butterfly_fused_2layer_row_from_geo_impl::<false, true, false>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            &low_tw3,
                        );
                    }
                }
                dst
            };
            assert_eq!(
                run_from_outer_low(true),
                run_from_outer_low(false),
                "row_from outer-low len={len}"
            );

            let right = next();
            let run_sparse = |diet: bool| {
                let mut dst = vec![F128::ZERO; 4 * len];
                // SAFETY: 4 rows of `len` lanes each, src/dst disjoint.
                unsafe {
                    if diet {
                        butterfly_fused_2layer_row_from_sparse_geo_impl::<true, false, false, false>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            right,
                            core::ptr::null(),
                        );
                    } else {
                        butterfly_fused_2layer_row_from_sparse_geo_impl::<false, false, false, false>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            right,
                            core::ptr::null(),
                        );
                    }
                }
                dst
            };
            assert_eq!(run_sparse(true), run_sparse(false), "sparse len={len}");

            let run_sparse_dense_outer_low = |outer_low: bool| {
                let mut sparse = vec![F128::ZERO; 4 * len];
                let mut dense = vec![F128::ZERO; 4 * len];
                // SAFETY: all rows have `len` lanes, destinations are
                // disjoint, and low_tw3[0].hi == 0 in the OUTER_LOW arm.
                unsafe {
                    if outer_low {
                        butterfly_fused_2layer_row_from_sparse_dense_geo_impl::<true, true>(
                            src.as_ptr(),
                            1,
                            0,
                            sparse.as_mut_ptr(),
                            dense.as_mut_ptr(),
                            1,
                            len,
                            len,
                            right,
                            &low_tw3,
                            core::ptr::null(),
                        );
                    } else {
                        butterfly_fused_2layer_row_from_sparse_dense_geo_impl::<false, true>(
                            src.as_ptr(),
                            1,
                            0,
                            sparse.as_mut_ptr(),
                            dense.as_mut_ptr(),
                            1,
                            len,
                            len,
                            right,
                            &low_tw3,
                            core::ptr::null(),
                        );
                    }
                }
                (sparse, dense)
            };
            assert_eq!(
                run_sparse_dense_outer_low(true),
                run_sparse_dense_outer_low(false),
                "sparse+dense outer-low len={len}"
            );

            // --- fused four-layer -----------------------------------------
            let mut tw15 = [F128::ZERO; 15];
            for t in tw15.iter_mut() {
                *t = next();
            }
            let base: Vec<F128> = (0..16 * len).map(|_| next()).collect();
            let run_fused4 = |diet: bool| {
                let mut buf = base.clone();
                // SAFETY: 16 rows of `len` lanes, sixteenth = 1, r = 0.
                unsafe {
                    if diet {
                        butterfly_fused_4layer_row_impl::<true, false, 0, 0, 0>(
                            buf.as_mut_ptr(),
                            1,
                            len,
                            len,
                            0,
                            &tw15,
                            0,
                            core::ptr::null(),
                        );
                    } else {
                        butterfly_fused_4layer_row_impl::<false, false, 0, 0, 0>(
                            buf.as_mut_ptr(),
                            1,
                            len,
                            len,
                            0,
                            &tw15,
                            0,
                            core::ptr::null(),
                        );
                    }
                }
                buf
            };
            assert_eq!(run_fused4(true), run_fused4(false), "fused4 len={len}");
        }
    }
}

/// Immutable split twiddles shared across all rows of a fused-four block.
/// Plain F128 storage permits sharing between Rayon workers without SIMD ABI.
#[repr(C, align(16))]
pub(crate) struct PreparedFour {
    pairs: [[F128; 2]; 15],
    // Kept after pairs: fixed-register assembly uses the unchanged offsets.
    zero_chain: bool,
    prefetch_lines: usize,
}

#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn prepare_four(t: &[F128; 15]) -> Option<PreparedFour> {
    static OFF: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_NTT_TWIDDLE_PREP").is_some());
    if *OFF {
        return None;
    }
    if mul_diet_disabled() {
        return None;
    }
    static ZERO_CHAIN:std::sync::LazyLock<bool>=std::sync::LazyLock::new(||std::env::var_os("FLOCK_NO_NTT_ZERO_CHAIN").is_none());
    static PREFETCH_LINES: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("FLOCK_NTT_PREFETCH_LINES")
            .ok().and_then(|s| s.parse().ok())
            .filter(|n| (1..=64).contains(n)).unwrap_or(4)
    });
    let mut result = PreparedFour {
        pairs: [[F128::ZERO; 2]; 15],
        zero_chain: *ZERO_CHAIN && zero_chain_twiddles(t),
        prefetch_lines: *PREFETCH_LINES,
    };
    for i in (0..12).step_by(4) {
        unsafe {
            use core::arch::x86_64::*;
            let v = _mm512_loadu_si512(t.as_ptr().add(i) as *const __m512i);
            let shifted = _mm512_xor_si512(
                _mm512_maskz_shuffle_epi32::<0x40>(0xcccc, v),
                _mm512_clmulepi64_epi128::<0x01>(v, _mm512_set1_epi64(0x87)),
            );
            let first = _mm512_permutex2var_epi64(v,
                _mm512_setr_epi64(0,1,8,9,2,3,10,11), shifted);
            let last = _mm512_permutex2var_epi64(v,
                _mm512_setr_epi64(4,5,12,13,6,7,14,15), shifted);
            let out = result.pairs.as_mut_ptr().add(i) as *mut __m512i;
            _mm512_storeu_si512(out, first);
            _mm512_storeu_si512(out.add(1), last);
        }
    }
    for i in 12..15 {
        // x^64 multiplication is a field operation, not a value assumption.
        use core::arch::x86_64::*;
        let companion = unsafe {
            let v = _mm_set_epi64x(t[i].hi as i64, t[i].lo as i64);
            let reduction = _mm_clmulepi64_si128::<0x01>(v, _mm_set_epi64x(0, 0x87));
            let folded = _mm_xor_si128(_mm_slli_si128::<8>(v), reduction);
            F128::new(_mm_cvtsi128_si64(folded) as u64,
                      _mm_extract_epi64::<1>(folded) as u64)
        };
        result.pairs[i] = [t[i], companion];
    }
    Some(result)
}

#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn row_prepared<const H: u8>(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    lanes: usize,
    r: usize,
    t: &[F128; 15],
    pf_r: usize,
    p: &PreparedFour,
) {
    unsafe {
        if super::super::ntt_shaped_enabled() {
            if num_ntts == 64 {
                match sixteenth {
                    128 => {
                        return butterfly_fused_4layer_row_impl::<true, true, H, 128, 64>(
                            ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                        );
                    }
                    8 => {
                        return butterfly_fused_4layer_row_impl::<true, true, H, 8, 64>(
                            ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                        );
                    }
                    4 => {
                        return butterfly_fused_4layer_row_impl::<true, true, H, 4, 64>(
                            ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                        );
                    }
                    1 => {
                        #[cfg(target_feature="avx512bw")]
                        if H==0 && p.zero_chain {
                            return zero_chain_contiguous_rows(ptr.add(r*64),lanes,p);
                        }
                        return butterfly_fused_4layer_row_impl::<true, true, H, 1, 64>(
                            ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                        );
                    }
                    _ => {}
                }
            }
        }
        if super::super::ntt_shaped_enabled() && num_ntts == 8 {
            match sixteenth {
                1024 => {
                    return butterfly_fused_4layer_row_impl::<true, true, H, 1024, 8>(
                        ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                    );
                }
                128 => {
                    return butterfly_fused_4layer_row_impl::<true, true, H, 128, 8>(
                        ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                    );
                }
                64 => {
                    return butterfly_fused_4layer_row_impl::<true, true, H, 64, 8>(
                        ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                    );
                }
                16 => {
                    return butterfly_fused_4layer_row_impl::<true, true, H, 16, 8>(
                        ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                    );
                }
                8 => {
                    return butterfly_fused_4layer_row_impl::<true, true, H, 8, 8>(
                        ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                    );
                }
                4 => {
                    return butterfly_fused_4layer_row_impl::<true, true, H, 4, 8>(
                        ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                    );
                }
                2 => {
                    return butterfly_fused_4layer_row_impl::<true, true, H, 2, 8>(
                        ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                    );
                }
                1 => {
                    return butterfly_fused_4layer_row_impl::<true, true, H, 1, 8>(
                        ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
                    );
                }
                _ => {}
            }
        }
        butterfly_fused_4layer_row_impl::<true, true, H, 0, 0>(
            ptr, sixteenth, num_ntts, lanes, r, t, pf_r, p,
        )
    }
}

#[cfg(all(test, target_feature = "avx512f", target_feature = "vpclmulqdq"))]
mod ab_ntt_tail_composition_tests {
    use super::*;
    use std::hint::black_box;

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    // Independent polynomial-basis reference: one bit at a time, with no
    // field multiplication, SIMD intrinsic, or prepared-table operation.
    fn multiply(a: F128, b: F128) -> F128 {
        let mut x = (u128::from(a.hi) << 64) | u128::from(a.lo);
        let mut y = (u128::from(b.hi) << 64) | u128::from(b.lo);
        let mut result = 0u128;
        for _ in 0..128 {
            if y & 1 != 0 {
                result ^= x;
            }
            let carry = x >> 127;
            x <<= 1;
            if carry != 0 {
                x ^= 0x87;
            }
            y >>= 1;
        }
        F128::new(result as u64, (result >> 64) as u64)
    }

    fn xor(a: F128, b: F128) -> F128 {
        F128::new(a.lo ^ b.lo, a.hi ^ b.hi)
    }

    #[test]
    fn ntt_companions_match_bitwise_oracle() {
        let mut seed = 0xc7a4_8be2_51d6_90f3;
        for case in 0..260 {
            let offset = case % 4;
            let mut input = [F128::new(0xceda, 0xbabe); 19];
            for i in 0..15 {
                input[offset + i] = match case {
                    0 => F128::ZERO,
                    1 => F128::new(u64::MAX, u64::MAX),
                    2..=129 => {
                        let bit = (case - 2 + i) % 128;
                        let v = 1u128 << bit;
                        F128::new(v as u64, (v >> 64) as u64)
                    }
                    _ => F128::new(next(&mut seed), next(&mut seed)),
                };
            }
            let original = input;
            let t: &[F128; 15] = input[offset..offset + 15].try_into().unwrap();
            // SAFETY: this module has the production ISA requirements and
            // supplies the exact fifteen-element initialized input.
            let prepared = unsafe { prepare_four(black_box(t)) }.unwrap();
            for i in 0..15 {
                assert_eq!(prepared.pairs[i][0], t[i], "case={case} i={i}");
                assert_eq!(
                    prepared.pairs[i][1],
                    multiply(t[i], F128::new(0, 1)),
                    "case={case} i={i}"
                );
            }
            assert_eq!(input, original);
        }
        println!("3900 production companions match the independent bitwise oracle");
    }


    fn oracle_rows(
        data: &mut [F128],
        offset: usize,
        stride: usize,
        lanes: usize,
        tw: &[F128],
        layers: usize,
    ) {
        for lane in 0..lanes {
            for layer in 0..layers {
                let distance = 1 << (layers - layer - 1);
                for group in 0..(1 << layer) {
                    let t = tw[(1 << layer) - 1 + group];
                    for j in 0..distance {
                        let u = offset + (group * 2 * distance + j) * stride + lane;
                        let v = u + distance * stride;
                        data[u] = xor(data[u], multiply(data[v], t));
                        data[v] = xor(data[v], data[u]);
                    }
                }
            }
        }
    }

    fn three_rows<const LOW_INNER: bool, const LOW_OUTER: bool, const HIGH_ONE: bool>() {
        let mut seed = 0xf31a_2e85_b736_094d;
        let guard = F128::new(0xfeed_cafe, 0xdead_beef);
        for n in [1usize, 3, 4, 7, 8, 12, 60, 64, 67] {
            for dense in [0, 1.min(n), 3.min(n), 4.min(n), n - 1, n] {
                for case in 0..4 {
                    let offset = 4 + case;
                    let mut data = vec![guard; offset + 8 * n + 4];
                    for row in 0..8 {
                        for lane in 0..n {
                            data[offset + row * n + lane] = if row & 1 != 0 && lane >= dense {
                                F128::ZERO
                            } else {
                                F128::new(next(&mut seed), next(&mut seed))
                            };
                        }
                    }
                    let mut tw: [F128; 7] = std::array::from_fn(|_| match case {
                        0 => F128::ZERO,
                        1 => F128::ONE,
                        2 => F128::new(u64::MAX, u64::MAX),
                        _ => F128::new(next(&mut seed), next(&mut seed)),
                    });
                    if LOW_INNER {
                        for t in &mut tw[1..] {
                            t.hi = 0;
                        }
                    }
                    if LOW_OUTER {
                        tw[0].hi = 0;
                    }
                    if HIGH_ONE {
                        tw[0].hi = 1;
                    }
                    let original_tw = tw;
                    let mut expected = data.clone();
                    oracle_rows(&mut expected, offset, n, n, &tw, 3);
                    // SAFETY: eight disjoint initialized rows, dense <= n,
                    // zero odd-row suffix, and const twiddle forms set above.
                    unsafe {
                        butterfly_fused_3layer_rows_impl::<true, LOW_INNER, 0, LOW_OUTER, HIGH_ONE>(
                            black_box(data.as_mut_ptr().add(offset)),
                            n,
                            dense,
                            black_box(&tw),
                        );
                    }
                    assert_eq!(
                        data, expected,
                        "n={n} dense={dense} case={case} li={LOW_INNER} lo={LOW_OUTER} ho={HIGH_ONE}"
                    );
                    assert_eq!(tw, original_tw);
                    if n == 64 {
                        // Also exercise the production shape specialization.
                        let mut shaped = vec![guard; offset + 8 * n + 4];
                        for row in 0..8 {
                            for lane in 0..n {
                                shaped[offset + row * n + lane] = if row & 1 != 0 && lane >= dense {
                                    F128::ZERO
                                } else {
                                    F128::new(next(&mut seed), next(&mut seed))
                                };
                            }
                        }
                        let mut expected = shaped.clone();
                        oracle_rows(&mut expected, offset, n, n, &tw, 3);
                        unsafe {
                            butterfly_fused_3layer_rows_impl::<
                                true,
                                LOW_INNER,
                                64,
                                LOW_OUTER,
                                HIGH_ONE,
                            >(
                                black_box(shaped.as_mut_ptr().add(offset)),
                                n,
                                dense,
                                black_box(&tw),
                            );
                        }
                        assert_eq!(shaped, expected, "shaped dense={dense} case={case}");
                    }
                }
            }
        }
    }

    #[test]
    fn ntt_three_layer_rows_match_bitwise_oracle() {
        three_rows::<false, false, false>();
        three_rows::<true, false, false>();
        three_rows::<false, true, false>();
        three_rows::<true, true, false>();
        three_rows::<false, false, true>();
        three_rows::<true, false, true>();
    }

    fn two_rows<const OUTER_LOW: bool, const INNER_LOW: bool, const ALIGNED: bool>(residue: usize) {
        let mut seed = 0xf21a_2e85_b736_094d;
        let guard = F128::new(0xfeed_cafe, 0xdead_beef);
        for lanes in [60, 64] {
            for case in 0..4 {
                let mut src: Vec<F128> = (0..4 * 67)
                    .map(|_| F128::new(next(&mut seed), next(&mut seed)))
                    .collect();
                let mut tw: [F128; 3] = std::array::from_fn(|_| match case {
                    0 => F128::ZERO,
                    1 => F128::ONE,
                    2 => F128::new(u64::MAX, u64::MAX),
                    _ => F128::new(next(&mut seed), next(&mut seed)),
                });
                if OUTER_LOW {
                    tw[0].hi = 0;
                }
                if INNER_LOW {
                    tw[1].hi = 0;
                    tw[2].hi = 0;
                }
                let original_src = src.clone();
                let original_tw = tw;
                let mut expected = vec![guard; 4 * 80 + 12];
                let mut got = expected.clone();
                let offset = 4
                    + (0..4)
                        .find(|&i| (got.as_ptr() as usize + (4 + i) * 16) % 64 == residue)
                        .unwrap();
                for row in 0..4 {
                    expected[offset + row * 80..offset + row * 80 + lanes]
                        .copy_from_slice(&src[row * 67..row * 67 + lanes]);
                    expected[offset + row * 80 + lanes..offset + row * 80 + 64].fill(F128::ZERO);
                }
                oracle_rows(&mut expected, offset, 80, lanes, &tw, 2);
                // SAFETY: readable source rows, disjoint aligned destinations
                // with space for 64 lanes, and const twiddle forms set above.
                unsafe {
                    let dst = got.as_mut_ptr().add(offset);
                    butterfly_fused_2layer_publish_nt_impl::<OUTER_LOW, INNER_LOW, true, ALIGNED>(
                        black_box(src.as_mut_ptr()),
                        67,
                        dst,
                        dst.add(80),
                        dst.add(160),
                        dst.add(240),
                        lanes,
                        tw[0],
                        tw[1],
                        tw[2],
                    );
                    core::arch::x86_64::_mm_sfence();
                }
                assert_eq!(
                    got, expected,
                    "lanes={lanes} case={case} ol={OUTER_LOW} il={INNER_LOW} aligned={ALIGNED} residue={residue}"
                );
                assert_eq!(src, original_src);
                assert_eq!(tw, original_tw);
            }
        }
    }

    #[test]
    fn ntt_two_layer_publish_rows_match_bitwise_oracle() {
        for residue in [0, 16, 32, 48] {
            two_rows::<false, false, false>(residue);
            two_rows::<false, true, false>(residue);
            two_rows::<true, false, false>(residue);
            two_rows::<true, true, false>(residue);
        }
        two_rows::<false, false, true>(0);
        two_rows::<false, true, true>(0);
        two_rows::<true, false, true>(0);
        two_rows::<true, true, true>(0);
    }


    fn rows<const H: u8>() -> usize {
        let shapes = [
            (128, 64), (8, 64), (4, 64), (1, 64),
            (1024, 8), (128, 8), (64, 8), (16, 8),
            (8, 8), (4, 8), (2, 8), (1, 8),
            (3, 7), (2, 13), (5, 16),
        ];
        let mut seed = 0xc751_8234_daf3_981b;
        let guard = F128::new(0x3ca9_8546_b721_936d, 0xcaa9_d173_985e_4693);
        let mut cases = 0;
        for (s, n) in shapes {
            for (case, lanes) in [0, 1, 3, 4, n - 1, n].into_iter().enumerate() {
                let offset = 4 + case % 4;
                let count = 16 * s * n;
                let r = if case % 2 == 0 { 0 } else { s - 1 };
                let pf_r = (r + 1) % s;
                let mut data = vec![guard; offset + count + 4];
                for row in 0..16 {
                    for lane in 0..n {
                        data[offset + (row * s + r) * n + lane] =
                            F128::new(next(&mut seed), next(&mut seed));
                    }
                }
                let tw: [F128; 15] = std::array::from_fn(|_| match case {
                    0 => F128::ZERO,
                    1 => F128::ONE,
                    2 => F128::new(u64::MAX, u64::MAX),
                    _ => F128::new(next(&mut seed), next(&mut seed)),
                });
                let original_tw = tw;
                let mut expected = data.clone();
                for lane in 0..lanes {
                    for layer in 0..4 {
                        let distance = 8 >> layer;
                        for group in 0..(1 << layer) {
                            let t = tw[(1 << layer) - 1 + group];
                            for j in 0..distance {
                                let u = offset + ((group * 2 * distance + j) * s + r) * n + lane;
                                let v = u + distance * s * n;
                                expected[u] = xor(expected[u], multiply(expected[v], t));
                                expected[v] = xor(expected[v], expected[u]);
                            }
                        }
                    }
                }
                // SAFETY: the initialized block contains sixteen disjoint
                // rows with lanes <= n, valid row and prefetch indices, and
                // the table is produced by the actual production producer.
                unsafe {
                    let prepared = prepare_four(black_box(&tw)).unwrap();
                    row_prepared::<H>(
                        black_box(data.as_mut_ptr().add(offset)), s, n, lanes,
                        r, black_box(&tw), pf_r, black_box(&prepared),
                    );
                }
                assert_eq!(tw, original_tw);
                assert_eq!(data, expected, "H={H} s={s} n={n} lanes={lanes} r={r}");
                cases += 1;
            }
        }
        cases
    }

    #[test]
    fn ntt_tail_rows_match_bitwise_oracle() {
        let cases = rows::<0>() + rows::<1>() + rows::<2>();
        assert_eq!(cases, 270);
        println!("270 production NTT row cases match the independent bitwise oracle and guards");
    }
}

/// Canonical sparse/dense seed transform with compact base/stride addresses.
/// Eight lanes remain live across each bulk iteration; a four-lane suffix
/// and the inherited scalar remainder preserve the generic implementation.
///
/// # Safety
/// Same contract as `butterfly_fused_2layer_row_from_sparse_dense_geo_impl`:
/// source, sparse destination and dense destination are disjoint; each row
/// and optional prefetch row covers the active prefix. When OUTER_LOW is
/// true, the outer twiddle's high limb is zero. Requires AVX-512F/BW/CLMUL.
/// The asm declares every modified register, does not preserve flags or
/// claim readonly memory, and uses no stack inside the block. Strides are
/// in bytes; unaligned vector loads and stores retain F128 alignment support.
#[cfg(target_feature = "avx512bw")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
unsafe fn seed_sparse_dense_compact<const OUTER_LOW: bool, const PREPARED:bool>(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst_sparse: *mut F128,
    dst_dense: *mut F128,
    dst_quarter: usize,
    num_ntts: usize,
    active_lanes: usize,
    right_twiddle: F128,
    dense_tw: &[F128; 3],
    pf_src: *const F128,
    prepared: *const F128,
    source_cols:usize,
) {
    const DIET: bool = true;
    use core::arch::x86_64::*;
    let [t_outer, t_inner_a, t_inner_b] = *dense_tw;
    unsafe {
        debug_assert!(!OUTER_LOW || t_outer.hi == 0);
        let load_pair=|i:usize,low:bool| {let t=_mm512_broadcast_i32x4(_mm_loadu_si128(prepared.add(2*i).cast()));let companion=if low{t}else{_mm512_broadcast_i32x4(_mm_loadu_si128(prepared.add(2*i+1).cast()))};(t,companion)};
        let sparse_b = if PREPARED {load_pair(0,false)}else{tw_x4::<false, DIET>(right_twiddle)};
        let outer = if PREPARED {load_pair(1,OUTER_LOW)}else{tw_x4::<OUTER_LOW, DIET>(t_outer)};
        let inner_a = if PREPARED {load_pair(2,false)}else{tw_x4::<false, DIET>(t_inner_a)};
        let inner_b = if PREPARED {load_pair(3,false)}else{tw_x4::<false, DIET>(t_inner_b)};
        let src_num_ntts=if PREPARED {source_cols}else{num_ntts};
        debug_assert!(!PREPARED||active_lanes.is_multiple_of(4));
        let src_row = |i: usize| src.add((i * src_quarter + src_r) * src_num_ntts);
        let sp_row = |i: usize| dst_sparse.add((i * dst_quarter) * num_ntts);
        let dn_row = |i: usize| dst_dense.add((i * dst_quarter) * num_ntts);
        let lanes = active_lanes & !3;
        if OUTER_LOW {
        core::arch::asm!(
            "test r9, r9",
            "jz 9f",
            "cmp r9, 8",
            "jb 5f",
            "2:",
            "test r10, r10",
            "jz 3f",
            "prefetcht0 [r10 + 0]",
            "prefetcht0 [r10 + 64]",
            "prefetcht0 [r10 + rsi + 0]",
            "prefetcht0 [r10 + rsi + 64]",
            "prefetcht0 [r10 + rsi*2 + 0]",
            "prefetcht0 [r10 + rsi*2 + 64]",
            "prefetcht0 [r10 + r11 + 0]",
            "prefetcht0 [r10 + r11 + 64]",
            "add r10, 128",
            "3:",
            "vmovdqu64 zmm0, [rdi + 0]",
            "vmovdqu64 zmm1, [rdi + rsi + 0]",
            "vmovdqu64 zmm2, [rdi + rsi*2 + 0]",
            "vmovdqu64 zmm3, [rdi + r11 + 0]",
            "vmovdqu64 [rdx + 0], zmm0",
            "vpxorq zmm8, zmm0, zmm1",
            "vmovdqu64 [rdx + r8 + 0], zmm8",
            "vmovdqu64 zmm4, [rdi + 64]",
            "vmovdqu64 zmm5, [rdi + rsi + 64]",
            "vmovdqu64 zmm6, [rdi + rsi*2 + 64]",
            "vmovdqu64 zmm7, [rdi + r11 + 64]",
            "vmovdqu64 [rdx + 64], zmm4",
            "vpxorq zmm8, zmm4, zmm5",
            "vmovdqu64 [rdx + r8 + 64], zmm8",
            "vpxorq zmm8, zmm3, zmm1",
            "vpxorq zmm9, zmm7, zmm5",
            "vpclmulqdq zmm12, zmm8, zmm16, 0x00",
            "vpclmulqdq zmm13, zmm8, zmm17, 0x01",
            "vpxorq zmm12, zmm12, zmm13",
            "vpclmulqdq zmm14, zmm8, zmm16, 0x10",
            "vpclmulqdq zmm15, zmm8, zmm17, 0x11",
            "vpxorq zmm14, zmm14, zmm15",
            "vpclmulqdq zmm10, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm10, zmm12, zmm14, 0x96",
            "vpclmulqdq zmm12, zmm9, zmm16, 0x00",
            "vpclmulqdq zmm13, zmm9, zmm17, 0x01",
            "vpxorq zmm12, zmm12, zmm13",
            "vpclmulqdq zmm14, zmm9, zmm16, 0x10",
            "vpclmulqdq zmm15, zmm9, zmm17, 0x11",
            "vpxorq zmm14, zmm14, zmm15",
            "vpclmulqdq zmm11, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm11, zmm12, zmm14, 0x96",
            "vpternlogq zmm10, zmm2, zmm0, 0x96",
            "vmovdqu64 [rdx + r8*2 + 0], zmm10",
            "vpxorq zmm8, zmm8, zmm10",
            "vmovdqu64 [rdx + rax + 0], zmm8",
            "vpternlogq zmm11, zmm6, zmm4, 0x96",
            "vmovdqu64 [rdx + r8*2 + 64], zmm11",
            "vpxorq zmm9, zmm9, zmm11",
            "vmovdqu64 [rdx + rax + 64], zmm9",
            "vpclmulqdq zmm12, zmm2, zmm18, 0x00",
            "vpxorq zmm0, zmm12, zmm0",
            "vpclmulqdq zmm24, zmm2, zmm18, 0x01",
            "vpclmulqdq zmm12, zmm6, zmm18, 0x00",
            "vpxorq zmm4, zmm12, zmm4",
            "vpclmulqdq zmm25, zmm6, zmm18, 0x01",
            "vpxorq zmm2, zmm2, zmm0",
            "vpxorq zmm6, zmm6, zmm4",
            "vpclmulqdq zmm12, zmm3, zmm18, 0x00",
            "vpxorq zmm12, zmm12, zmm1",
            "vpclmulqdq zmm14, zmm3, zmm18, 0x01",
            "vpclmulqdq zmm1, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm1, zmm12, zmm14, 0x96",
            "vpclmulqdq zmm12, zmm7, zmm18, 0x00",
            "vpxorq zmm12, zmm12, zmm5",
            "vpclmulqdq zmm14, zmm7, zmm18, 0x01",
            "vpclmulqdq zmm5, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm5, zmm12, zmm14, 0x96",
            "vpxorq zmm3, zmm3, zmm1",
            "vpxorq zmm7, zmm7, zmm5",
            "vpclmulqdq zmm12, zmm1, zmm20, 0x00",
            "vpclmulqdq zmm13, zmm1, zmm21, 0x01",
            "vpternlogq zmm12, zmm13, zmm0, 0x96",
            "vpclmulqdq zmm14, zmm1, zmm20, 0x10",
            "vpclmulqdq zmm15, zmm1, zmm21, 0x11",
            "vpternlogq zmm14, zmm15, zmm24, 0x96",
            "vpclmulqdq zmm0, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm0, zmm12, zmm14, 0x96",
            "vpclmulqdq zmm12, zmm5, zmm20, 0x00",
            "vpclmulqdq zmm13, zmm5, zmm21, 0x01",
            "vpternlogq zmm12, zmm13, zmm4, 0x96",
            "vpclmulqdq zmm14, zmm5, zmm20, 0x10",
            "vpclmulqdq zmm15, zmm5, zmm21, 0x11",
            "vpternlogq zmm14, zmm15, zmm25, 0x96",
            "vpclmulqdq zmm4, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm4, zmm12, zmm14, 0x96",
            "vpxorq zmm1, zmm1, zmm0",
            "vpxorq zmm5, zmm5, zmm4",
            "vpclmulqdq zmm12, zmm3, zmm22, 0x00",
            "vpclmulqdq zmm13, zmm3, zmm23, 0x01",
            "vpternlogq zmm12, zmm13, zmm2, 0x96",
            "vpclmulqdq zmm14, zmm3, zmm22, 0x10",
            "vpclmulqdq zmm15, zmm3, zmm23, 0x11",
            "vpternlogq zmm14, zmm15, zmm24, 0x96",
            "vpclmulqdq zmm2, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm2, zmm12, zmm14, 0x96",
            "vpclmulqdq zmm12, zmm7, zmm22, 0x00",
            "vpclmulqdq zmm13, zmm7, zmm23, 0x01",
            "vpternlogq zmm12, zmm13, zmm6, 0x96",
            "vpclmulqdq zmm14, zmm7, zmm22, 0x10",
            "vpclmulqdq zmm15, zmm7, zmm23, 0x11",
            "vpternlogq zmm14, zmm15, zmm25, 0x96",
            "vpclmulqdq zmm6, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm6, zmm12, zmm14, 0x96",
            "vpxorq zmm3, zmm3, zmm2",
            "vpxorq zmm7, zmm7, zmm6",
            "vmovdqu64 [rcx + 0], zmm0",
            "vmovdqu64 [rcx + r8 + 0], zmm1",
            "vmovdqu64 [rcx + r8*2 + 0], zmm2",
            "vmovdqu64 [rcx + rax + 0], zmm3",
            "vmovdqu64 [rcx + 64], zmm4",
            "vmovdqu64 [rcx + r8 + 64], zmm5",
            "vmovdqu64 [rcx + r8*2 + 64], zmm6",
            "vmovdqu64 [rcx + rax + 64], zmm7",
            "add rdi, 128",
            "add rdx, 128",
            "add rcx, 128",
            "sub r9, 8",
            "cmp r9, 8",
            "jae 2b",
            "test r9, r9",
            "jz 9f",
            "5:",
            "test r10, r10",
            "jz 6f",
            "prefetcht0 [r10]",
            "prefetcht0 [r10 + rsi]",
            "prefetcht0 [r10 + rsi*2]",
            "prefetcht0 [r10 + r11]",
            "6:",
            "vmovdqu64 zmm0, [rdi + 0]",
            "vmovdqu64 zmm1, [rdi + rsi + 0]",
            "vmovdqu64 zmm2, [rdi + rsi*2 + 0]",
            "vmovdqu64 zmm3, [rdi + r11 + 0]",
            "vmovdqu64 [rdx + 0], zmm0",
            "vpxorq zmm8, zmm0, zmm1",
            "vmovdqu64 [rdx + r8 + 0], zmm8",
            "vpxorq zmm8, zmm3, zmm1",
            "vpclmulqdq zmm12, zmm8, zmm16, 0x00",
            "vpclmulqdq zmm13, zmm8, zmm17, 0x01",
            "vpxorq zmm12, zmm12, zmm13",
            "vpclmulqdq zmm14, zmm8, zmm16, 0x10",
            "vpclmulqdq zmm15, zmm8, zmm17, 0x11",
            "vpxorq zmm14, zmm14, zmm15",
            "vpclmulqdq zmm10, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm10, zmm12, zmm14, 0x96",
            "vpternlogq zmm10, zmm2, zmm0, 0x96",
            "vmovdqu64 [rdx + r8*2 + 0], zmm10",
            "vpxorq zmm8, zmm8, zmm10",
            "vmovdqu64 [rdx + rax + 0], zmm8",
            "vpclmulqdq zmm12, zmm2, zmm18, 0x00",
            "vpxorq zmm0, zmm12, zmm0",
            "vpclmulqdq zmm24, zmm2, zmm18, 0x01",
            "vpxorq zmm2, zmm2, zmm0",
            "vpclmulqdq zmm12, zmm3, zmm18, 0x00",
            "vpxorq zmm12, zmm12, zmm1",
            "vpclmulqdq zmm14, zmm3, zmm18, 0x01",
            "vpclmulqdq zmm1, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm1, zmm12, zmm14, 0x96",
            "vpxorq zmm3, zmm3, zmm1",
            "vpclmulqdq zmm12, zmm1, zmm20, 0x00",
            "vpclmulqdq zmm13, zmm1, zmm21, 0x01",
            "vpternlogq zmm12, zmm13, zmm0, 0x96",
            "vpclmulqdq zmm14, zmm1, zmm20, 0x10",
            "vpclmulqdq zmm15, zmm1, zmm21, 0x11",
            "vpternlogq zmm14, zmm15, zmm24, 0x96",
            "vpclmulqdq zmm0, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm0, zmm12, zmm14, 0x96",
            "vpxorq zmm1, zmm1, zmm0",
            "vpclmulqdq zmm12, zmm3, zmm22, 0x00",
            "vpclmulqdq zmm13, zmm3, zmm23, 0x01",
            "vpternlogq zmm12, zmm13, zmm2, 0x96",
            "vpclmulqdq zmm14, zmm3, zmm22, 0x10",
            "vpclmulqdq zmm15, zmm3, zmm23, 0x11",
            "vpternlogq zmm14, zmm15, zmm24, 0x96",
            "vpclmulqdq zmm2, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm2, zmm12, zmm14, 0x96",
            "vpxorq zmm3, zmm3, zmm2",
            "vmovdqu64 [rcx + 0], zmm0",
            "vmovdqu64 [rcx + r8 + 0], zmm1",
            "vmovdqu64 [rcx + r8*2 + 0], zmm2",
            "vmovdqu64 [rcx + rax + 0], zmm3",
            "9:",
            inout("rdi") src_row(0) => _,
            in("rsi") src_quarter * src_num_ntts * 16,
            inout("rdx") dst_sparse => _,
            inout("rcx") dst_dense => _,
            in("r8") dst_quarter * num_ntts * 16,
            inout("r9") lanes => _,
            inout("r10") pf_src => _,
            in("r11") src_quarter * src_num_ntts * 48,
            in("rax") dst_quarter * num_ntts * 48,
            in("zmm16") sparse_b.0, in("zmm17") sparse_b.1,
            in("zmm18") outer.0, in("zmm19") outer.1,
            in("zmm20") inner_a.0, in("zmm21") inner_a.1,
            in("zmm22") inner_b.0, in("zmm23") inner_b.1,
            in("zmm31") _mm512_set_epi64(0,0x87,0,0x87,0,0x87,0,0x87),
            out("zmm0") _,
            out("zmm1") _,
            out("zmm2") _,
            out("zmm3") _,
            out("zmm4") _,
            out("zmm5") _,
            out("zmm6") _,
            out("zmm7") _,
            out("zmm8") _,
            out("zmm9") _,
            out("zmm10") _,
            out("zmm11") _,
            out("zmm12") _,
            out("zmm13") _,
            out("zmm14") _,
            out("zmm15") _,
            out("zmm24") _,
            out("zmm25") _,
            options(nostack),
        );
        } else {
        core::arch::asm!(
            "test r9, r9",
            "jz 9f",
            "cmp r9, 8",
            "jb 5f",
            "2:",
            "test r10, r10",
            "jz 3f",
            "prefetcht0 [r10 + 0]",
            "prefetcht0 [r10 + 64]",
            "prefetcht0 [r10 + rsi + 0]",
            "prefetcht0 [r10 + rsi + 64]",
            "prefetcht0 [r10 + rsi*2 + 0]",
            "prefetcht0 [r10 + rsi*2 + 64]",
            "prefetcht0 [r10 + r11 + 0]",
            "prefetcht0 [r10 + r11 + 64]",
            "add r10, 128",
            "3:",
            "vmovdqu64 zmm0, [rdi + 0]",
            "vmovdqu64 zmm1, [rdi + rsi + 0]",
            "vmovdqu64 zmm2, [rdi + rsi*2 + 0]",
            "vmovdqu64 zmm3, [rdi + r11 + 0]",
            "vmovdqu64 [rdx + 0], zmm0",
            "vpxorq zmm8, zmm0, zmm1",
            "vmovdqu64 [rdx + r8 + 0], zmm8",
            "vmovdqu64 zmm4, [rdi + 64]",
            "vmovdqu64 zmm5, [rdi + rsi + 64]",
            "vmovdqu64 zmm6, [rdi + rsi*2 + 64]",
            "vmovdqu64 zmm7, [rdi + r11 + 64]",
            "vmovdqu64 [rdx + 64], zmm4",
            "vpxorq zmm8, zmm4, zmm5",
            "vmovdqu64 [rdx + r8 + 64], zmm8",
            "vpxorq zmm8, zmm3, zmm1",
            "vpxorq zmm9, zmm7, zmm5",
            "vpclmulqdq zmm12, zmm8, zmm16, 0x00",
            "vpclmulqdq zmm13, zmm8, zmm17, 0x01",
            "vpxorq zmm12, zmm12, zmm13",
            "vpclmulqdq zmm14, zmm8, zmm16, 0x10",
            "vpclmulqdq zmm15, zmm8, zmm17, 0x11",
            "vpxorq zmm14, zmm14, zmm15",
            "vpclmulqdq zmm10, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm10, zmm12, zmm14, 0x96",
            "vpclmulqdq zmm12, zmm9, zmm16, 0x00",
            "vpclmulqdq zmm13, zmm9, zmm17, 0x01",
            "vpxorq zmm12, zmm12, zmm13",
            "vpclmulqdq zmm14, zmm9, zmm16, 0x10",
            "vpclmulqdq zmm15, zmm9, zmm17, 0x11",
            "vpxorq zmm14, zmm14, zmm15",
            "vpclmulqdq zmm11, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm11, zmm12, zmm14, 0x96",
            "vpternlogq zmm10, zmm2, zmm0, 0x96",
            "vmovdqu64 [rdx + r8*2 + 0], zmm10",
            "vpxorq zmm8, zmm8, zmm10",
            "vmovdqu64 [rdx + rax + 0], zmm8",
            "vpternlogq zmm11, zmm6, zmm4, 0x96",
            "vmovdqu64 [rdx + r8*2 + 64], zmm11",
            "vpxorq zmm9, zmm9, zmm11",
            "vmovdqu64 [rdx + rax + 64], zmm9",
            "vpclmulqdq zmm12, zmm2, zmm18, 0x00",
            "vpclmulqdq zmm13, zmm2, zmm19, 0x01",
            "vpternlogq zmm0, zmm12, zmm13, 0x96",
            "vpclmulqdq zmm24, zmm2, zmm18, 0x10",
            "vpclmulqdq zmm15, zmm2, zmm19, 0x11",
            "vpxorq zmm24, zmm24, zmm15",
            "vpclmulqdq zmm12, zmm6, zmm18, 0x00",
            "vpclmulqdq zmm13, zmm6, zmm19, 0x01",
            "vpternlogq zmm4, zmm12, zmm13, 0x96",
            "vpclmulqdq zmm25, zmm6, zmm18, 0x10",
            "vpclmulqdq zmm15, zmm6, zmm19, 0x11",
            "vpxorq zmm25, zmm25, zmm15",
            "vpxorq zmm2, zmm2, zmm0",
            "vpxorq zmm6, zmm6, zmm4",
            "vpclmulqdq zmm12, zmm3, zmm18, 0x00",
            "vpclmulqdq zmm13, zmm3, zmm19, 0x01",
            "vpternlogq zmm12, zmm13, zmm1, 0x96",
            "vpclmulqdq zmm14, zmm3, zmm18, 0x10",
            "vpclmulqdq zmm15, zmm3, zmm19, 0x11",
            "vpxorq zmm14, zmm14, zmm15",
            "vpclmulqdq zmm1, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm1, zmm12, zmm14, 0x96",
            "vpclmulqdq zmm12, zmm7, zmm18, 0x00",
            "vpclmulqdq zmm13, zmm7, zmm19, 0x01",
            "vpternlogq zmm12, zmm13, zmm5, 0x96",
            "vpclmulqdq zmm14, zmm7, zmm18, 0x10",
            "vpclmulqdq zmm15, zmm7, zmm19, 0x11",
            "vpxorq zmm14, zmm14, zmm15",
            "vpclmulqdq zmm5, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm5, zmm12, zmm14, 0x96",
            "vpxorq zmm3, zmm3, zmm1",
            "vpxorq zmm7, zmm7, zmm5",
            "vpclmulqdq zmm12, zmm1, zmm20, 0x00",
            "vpclmulqdq zmm13, zmm1, zmm21, 0x01",
            "vpternlogq zmm12, zmm13, zmm0, 0x96",
            "vpclmulqdq zmm14, zmm1, zmm20, 0x10",
            "vpclmulqdq zmm15, zmm1, zmm21, 0x11",
            "vpternlogq zmm14, zmm15, zmm24, 0x96",
            "vpclmulqdq zmm0, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm0, zmm12, zmm14, 0x96",
            "vpclmulqdq zmm12, zmm5, zmm20, 0x00",
            "vpclmulqdq zmm13, zmm5, zmm21, 0x01",
            "vpternlogq zmm12, zmm13, zmm4, 0x96",
            "vpclmulqdq zmm14, zmm5, zmm20, 0x10",
            "vpclmulqdq zmm15, zmm5, zmm21, 0x11",
            "vpternlogq zmm14, zmm15, zmm25, 0x96",
            "vpclmulqdq zmm4, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm4, zmm12, zmm14, 0x96",
            "vpxorq zmm1, zmm1, zmm0",
            "vpxorq zmm5, zmm5, zmm4",
            "vpclmulqdq zmm12, zmm3, zmm22, 0x00",
            "vpclmulqdq zmm13, zmm3, zmm23, 0x01",
            "vpternlogq zmm12, zmm13, zmm2, 0x96",
            "vpclmulqdq zmm14, zmm3, zmm22, 0x10",
            "vpclmulqdq zmm15, zmm3, zmm23, 0x11",
            "vpternlogq zmm14, zmm15, zmm24, 0x96",
            "vpclmulqdq zmm2, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm2, zmm12, zmm14, 0x96",
            "vpclmulqdq zmm12, zmm7, zmm22, 0x00",
            "vpclmulqdq zmm13, zmm7, zmm23, 0x01",
            "vpternlogq zmm12, zmm13, zmm6, 0x96",
            "vpclmulqdq zmm14, zmm7, zmm22, 0x10",
            "vpclmulqdq zmm15, zmm7, zmm23, 0x11",
            "vpternlogq zmm14, zmm15, zmm25, 0x96",
            "vpclmulqdq zmm6, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm6, zmm12, zmm14, 0x96",
            "vpxorq zmm3, zmm3, zmm2",
            "vpxorq zmm7, zmm7, zmm6",
            "vmovdqu64 [rcx + 0], zmm0",
            "vmovdqu64 [rcx + r8 + 0], zmm1",
            "vmovdqu64 [rcx + r8*2 + 0], zmm2",
            "vmovdqu64 [rcx + rax + 0], zmm3",
            "vmovdqu64 [rcx + 64], zmm4",
            "vmovdqu64 [rcx + r8 + 64], zmm5",
            "vmovdqu64 [rcx + r8*2 + 64], zmm6",
            "vmovdqu64 [rcx + rax + 64], zmm7",
            "add rdi, 128",
            "add rdx, 128",
            "add rcx, 128",
            "sub r9, 8",
            "cmp r9, 8",
            "jae 2b",
            "test r9, r9",
            "jz 9f",
            "5:",
            "test r10, r10",
            "jz 6f",
            "prefetcht0 [r10]",
            "prefetcht0 [r10 + rsi]",
            "prefetcht0 [r10 + rsi*2]",
            "prefetcht0 [r10 + r11]",
            "6:",
            "vmovdqu64 zmm0, [rdi + 0]",
            "vmovdqu64 zmm1, [rdi + rsi + 0]",
            "vmovdqu64 zmm2, [rdi + rsi*2 + 0]",
            "vmovdqu64 zmm3, [rdi + r11 + 0]",
            "vmovdqu64 [rdx + 0], zmm0",
            "vpxorq zmm8, zmm0, zmm1",
            "vmovdqu64 [rdx + r8 + 0], zmm8",
            "vpxorq zmm8, zmm3, zmm1",
            "vpclmulqdq zmm12, zmm8, zmm16, 0x00",
            "vpclmulqdq zmm13, zmm8, zmm17, 0x01",
            "vpxorq zmm12, zmm12, zmm13",
            "vpclmulqdq zmm14, zmm8, zmm16, 0x10",
            "vpclmulqdq zmm15, zmm8, zmm17, 0x11",
            "vpxorq zmm14, zmm14, zmm15",
            "vpclmulqdq zmm10, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm10, zmm12, zmm14, 0x96",
            "vpternlogq zmm10, zmm2, zmm0, 0x96",
            "vmovdqu64 [rdx + r8*2 + 0], zmm10",
            "vpxorq zmm8, zmm8, zmm10",
            "vmovdqu64 [rdx + rax + 0], zmm8",
            "vpclmulqdq zmm12, zmm2, zmm18, 0x00",
            "vpclmulqdq zmm13, zmm2, zmm19, 0x01",
            "vpternlogq zmm0, zmm12, zmm13, 0x96",
            "vpclmulqdq zmm24, zmm2, zmm18, 0x10",
            "vpclmulqdq zmm15, zmm2, zmm19, 0x11",
            "vpxorq zmm24, zmm24, zmm15",
            "vpxorq zmm2, zmm2, zmm0",
            "vpclmulqdq zmm12, zmm3, zmm18, 0x00",
            "vpclmulqdq zmm13, zmm3, zmm19, 0x01",
            "vpternlogq zmm12, zmm13, zmm1, 0x96",
            "vpclmulqdq zmm14, zmm3, zmm18, 0x10",
            "vpclmulqdq zmm15, zmm3, zmm19, 0x11",
            "vpxorq zmm14, zmm14, zmm15",
            "vpclmulqdq zmm1, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm1, zmm12, zmm14, 0x96",
            "vpxorq zmm3, zmm3, zmm1",
            "vpclmulqdq zmm12, zmm1, zmm20, 0x00",
            "vpclmulqdq zmm13, zmm1, zmm21, 0x01",
            "vpternlogq zmm12, zmm13, zmm0, 0x96",
            "vpclmulqdq zmm14, zmm1, zmm20, 0x10",
            "vpclmulqdq zmm15, zmm1, zmm21, 0x11",
            "vpternlogq zmm14, zmm15, zmm24, 0x96",
            "vpclmulqdq zmm0, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm0, zmm12, zmm14, 0x96",
            "vpxorq zmm1, zmm1, zmm0",
            "vpclmulqdq zmm12, zmm3, zmm22, 0x00",
            "vpclmulqdq zmm13, zmm3, zmm23, 0x01",
            "vpternlogq zmm12, zmm13, zmm2, 0x96",
            "vpclmulqdq zmm14, zmm3, zmm22, 0x10",
            "vpclmulqdq zmm15, zmm3, zmm23, 0x11",
            "vpternlogq zmm14, zmm15, zmm24, 0x96",
            "vpclmulqdq zmm2, zmm14, zmm31, 0x01",
            "vpslldq zmm14, zmm14, 8",
            "vpternlogq zmm2, zmm12, zmm14, 0x96",
            "vpxorq zmm3, zmm3, zmm2",
            "vmovdqu64 [rcx + 0], zmm0",
            "vmovdqu64 [rcx + r8 + 0], zmm1",
            "vmovdqu64 [rcx + r8*2 + 0], zmm2",
            "vmovdqu64 [rcx + rax + 0], zmm3",
            "9:",
            inout("rdi") src_row(0) => _,
            in("rsi") src_quarter * src_num_ntts * 16,
            inout("rdx") dst_sparse => _,
            inout("rcx") dst_dense => _,
            in("r8") dst_quarter * num_ntts * 16,
            inout("r9") lanes => _,
            inout("r10") pf_src => _,
            in("r11") src_quarter * src_num_ntts * 48,
            in("rax") dst_quarter * num_ntts * 48,
            in("zmm16") sparse_b.0, in("zmm17") sparse_b.1,
            in("zmm18") outer.0, in("zmm19") outer.1,
            in("zmm20") inner_a.0, in("zmm21") inner_a.1,
            in("zmm22") inner_b.0, in("zmm23") inner_b.1,
            in("zmm31") _mm512_set_epi64(0,0x87,0,0x87,0,0x87,0,0x87),
            out("zmm0") _,
            out("zmm1") _,
            out("zmm2") _,
            out("zmm3") _,
            out("zmm4") _,
            out("zmm5") _,
            out("zmm6") _,
            out("zmm7") _,
            out("zmm8") _,
            out("zmm9") _,
            out("zmm10") _,
            out("zmm11") _,
            out("zmm12") _,
            out("zmm13") _,
            out("zmm14") _,
            out("zmm15") _,
            out("zmm24") _,
            out("zmm25") _,
            options(nostack),
        );
        }
        let lane = lanes;
        // Ranked L0 / recursive seed pass `num_ntts ∈ {64, 8}`: this
        // remainder is dead. Keep it out of the 4-wide leaf's I-cache; other
        // geometries still enter the exact scalar remainder when needed.
        if lane < active_lanes {
            sparse_dense_geo_scalar_tail(
                src_row,
                sp_row,
                dn_row,
                right_twiddle,
                t_outer,
                t_inner_a,
                t_inner_b,
                lane,
                active_lanes,
            );
        }
    }
}


#[cfg(all(test, target_feature = "avx512f", target_feature = "vpclmulqdq"))]
mod seed_compact_address_oracle_tests {
    use super::*;
    use std::hint::black_box;
    fn bits(x: F128) -> u128 { u128::from(x.lo) | (u128::from(x.hi) << 64) }
    fn field(x: u128) -> F128 { F128::new(x as u64, (x >> 64) as u64) }
    fn multiply(mut a: u128, mut b: u128) -> u128 {
        let mut z = 0;
        for _ in 0..128 {
            if b & 1 != 0 { z ^= a; }
            b >>= 1;
            let high = a >> 127;
            a <<= 1;
            if high != 0 { a ^= 0x87; }
        }
        z
    }
    #[test]
    fn seed_compact_addresses_match_bitwise_oracle() {
        let mut state = 0x71f8_8cf0_3391_d22au64;
        let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            F128::new(state, state.rotate_left(27) ^ 0x94d0_49bb_1331_11eb) };
        let guard = F128::new(0xa5a5_a5a5_a5a5_a5a5, 0x5a5a_5a5a_5a5a_5a5a);
        let mut cases = 0;
        for low in [false, true] { for n in [4usize,8,12,16,60,64,68,128] {
            for dq in [1usize,5,64] { for case in 0..12 {
                let sq = [1usize,3,7][case%3];
                let sr = if case%2 == 0 { 0 } else { sq-1 };
                let active = [0,4,n-4,n][case%4];
                let start = 4 + case%4;
                let src_len = start + (3*sq+sr+1)*n + 5;
                let dst_len = start + (3*dq+1)*n + 5;
                let mut input: Vec<F128> = (0..src_len).map(|_| next()).collect();
                if case == 0 { input.fill(F128::ZERO); }
                if case == 1 { input.fill(F128::new(u64::MAX,u64::MAX)); }
                let mut right = next();
                let mut tw = [next(),next(),next()];
                if low { tw[0].hi = 0; }
                if case%6 == 0 { right = F128::ZERO; }
                if case%6 == 1 { right = F128::new(1,0); }
                let saved_input = input.clone(); let saved_tw = tw;
                let mut expected_s = vec![guard;dst_len]; let mut expected_d = vec![guard;dst_len];
                for lane in 0..active {
                    let a = bits(input[start+sr*n+lane]); let b = bits(input[start+(sq+sr)*n+lane]);
                    let c = bits(input[start+(2*sq+sr)*n+lane]); let z = bits(input[start+(3*sq+sr)*n+lane]);
                    let sc = c ^ a ^ multiply(z ^ b,bits(right));
                    let sparse = [a,b ^ a,sc,z ^ b ^ sc];
                    let aa = a ^ multiply(c,bits(tw[0])); let bb = b ^ multiply(z,bits(tw[0]));
                    let cc = c ^ aa; let dd = z ^ bb;
                    let oa = aa ^ multiply(bb,bits(tw[1])); let oc = cc ^ multiply(dd,bits(tw[2]));
                    let dense = [oa,bb ^ oa,oc,dd ^ oc];
                    for row in 0..4 {
                        expected_s[start+row*dq*n+lane] = field(sparse[row]);
                        expected_d[start+row*dq*n+lane] = field(dense[row]);
                    }
                }
                let mut sparse = vec![guard;dst_len]; let mut dense = vec![guard;dst_len];
                // SAFETY: all source/destination rows are allocated and disjoint;
                // active is a multiple of four and does not exceed n. The optional
                // prefetch row is within the same source allocation.
                unsafe {
                    let src = input.as_ptr().add(start);
                    let pf = if case%2 == 0 { core::ptr::null() } else { src.add(sr*n) };
                    butterfly_fused_2layer_row_from_sparse_dense_geo(
                        black_box(src),black_box(sq),black_box(sr),black_box(sparse.as_mut_ptr().add(start)),
                        black_box(dense.as_mut_ptr().add(start)),black_box(dq),black_box(n),black_box(active),
                        black_box(right),black_box(&tw),black_box(pf));
                }
                assert_eq!(sparse,expected_s,"sparse low={low} n={n} dq={dq} case={case}");
                assert_eq!(dense,expected_d,"dense low={low} n={n} dq={dq} case={case}");
                assert_eq!(input,saved_input); assert_eq!(tw,saved_tw);
                cases += 1;
            }}
        }}
        assert_eq!(cases,576);
        println!("576 production compact-address seed cases match independent bitwise products, guards, source and twiddle immutability");
    }
}


/// Seed a four-aligned window with independent source/destination pitches.
/// All demanded source fields are initialized; destination ranges are exclusive.
#[cfg(target_feature="avx512bw")]
#[inline]
#[target_feature(enable="avx512f,avx512bw,vpclmulqdq")]
pub(super) unsafe fn seed_window_prepared(src:*const F128,src_quarter:usize,src_r:usize,dst_sparse:*mut F128,dst_dense:*mut F128,source_cols:usize,active:usize,prepared:&[F128;8]){unsafe{
 assert!(matches!(source_cols,8|16|32|64));
 let dense=[prepared[2],prepared[4],prepared[6]];
 if dense[0].hi==0 {seed_sparse_dense_compact::<true,true>(src,src_quarter,src_r,dst_sparse,dst_dense,64,64,active,prepared[0],&dense,core::ptr::null(),prepared.as_ptr(),source_cols);}
 else{seed_sparse_dense_compact::<false,true>(src,src_quarter,src_r,dst_sparse,dst_dense,64,64,active,prepared[0],&dense,core::ptr::null(),prepared.as_ptr(),source_cols);}
}}


#[inline]
fn zero_chain_twiddles(t:&[F128;15])->bool {
    [0usize,1,3,7].into_iter().all(|i|t[i]==F128::ZERO)
}

// The checked zero path has fifteen identity butterflies. Retaining those
// as XORs avoids their field products without changing the four-layer order.
#[cfg(target_feature="avx512bw")]
#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,vpclmulqdq")]
unsafe fn zero_chain_contiguous_rows(ptr:*mut F128,lanes:usize,prepared:*const PreparedFour){unsafe{
    use core::arch::x86_64::*;
    let zero=_mm512_setzero_si512();
    let mut lane=0;
    let wide=lanes&!3;
    while lane<wide {
        let mut v=[zero;16];
        v[0]=_mm512_loadu_si512(ptr.add(0*64+lane).cast());
        v[1]=_mm512_loadu_si512(ptr.add(1*64+lane).cast());
        v[2]=_mm512_loadu_si512(ptr.add(2*64+lane).cast());
        v[3]=_mm512_loadu_si512(ptr.add(3*64+lane).cast());
        v[4]=_mm512_loadu_si512(ptr.add(4*64+lane).cast());
        v[5]=_mm512_loadu_si512(ptr.add(5*64+lane).cast());
        v[6]=_mm512_loadu_si512(ptr.add(6*64+lane).cast());
        v[7]=_mm512_loadu_si512(ptr.add(7*64+lane).cast());
        v[8]=_mm512_loadu_si512(ptr.add(8*64+lane).cast());
        v[9]=_mm512_loadu_si512(ptr.add(9*64+lane).cast());
        v[10]=_mm512_loadu_si512(ptr.add(10*64+lane).cast());
        v[11]=_mm512_loadu_si512(ptr.add(11*64+lane).cast());
        v[12]=_mm512_loadu_si512(ptr.add(12*64+lane).cast());
        v[13]=_mm512_loadu_si512(ptr.add(13*64+lane).cast());
        v[14]=_mm512_loadu_si512(ptr.add(14*64+lane).cast());
        v[15]=_mm512_loadu_si512(ptr.add(15*64+lane).cast());
        macro_rules! bf {($u:literal,$v:literal,$tw:literal)=>{{
            let pair=(*prepared).pairs.as_ptr().add($tw).cast::<__m128i>();
            let t=_mm512_broadcast_i32x4(_mm_loadu_si128(pair));
            let c=_mm512_broadcast_i32x4(_mm_loadu_si128(pair.add(1)));
            let lo=_mm512_xor_si512(_mm512_clmulepi64_epi128::<0>(v[$v],t),_mm512_clmulepi64_epi128::<1>(v[$v],c));
            let hi=_mm512_xor_si512(_mm512_clmulepi64_epi128::<16>(v[$v],t),_mm512_clmulepi64_epi128::<17>(v[$v],c));
            let poly=_mm512_setr_epi64(0x87,0,0x87,0,0x87,0,0x87,0);
            let product=_mm512_ternarylogic_epi64::<0x96>(lo,_mm512_bslli_epi128::<8>(hi),_mm512_clmulepi64_epi128::<1>(hi,poly));
            v[$u]=_mm512_xor_si512(v[$u],product);v[$v]=_mm512_xor_si512(v[$v],v[$u]);
        }}}
        v[8]=_mm512_xor_si512(v[8],v[0]);
        v[9]=_mm512_xor_si512(v[9],v[1]);
        v[10]=_mm512_xor_si512(v[10],v[2]);
        v[11]=_mm512_xor_si512(v[11],v[3]);
        v[12]=_mm512_xor_si512(v[12],v[4]);
        v[13]=_mm512_xor_si512(v[13],v[5]);
        v[14]=_mm512_xor_si512(v[14],v[6]);
        v[15]=_mm512_xor_si512(v[15],v[7]);
        v[4]=_mm512_xor_si512(v[4],v[0]);
        v[5]=_mm512_xor_si512(v[5],v[1]);
        v[6]=_mm512_xor_si512(v[6],v[2]);
        v[7]=_mm512_xor_si512(v[7],v[3]);
        bf!(8,12,2);
        bf!(9,13,2);
        bf!(10,14,2);
        bf!(11,15,2);
        v[2]=_mm512_xor_si512(v[2],v[0]);
        v[3]=_mm512_xor_si512(v[3],v[1]);
        bf!(4,6,4);
        bf!(5,7,4);
        bf!(8,10,5);
        bf!(9,11,5);
        bf!(12,14,6);
        bf!(13,15,6);
        v[1]=_mm512_xor_si512(v[1],v[0]);
        bf!(2,3,8);
        bf!(4,5,9);
        bf!(6,7,10);
        bf!(8,9,11);
        bf!(10,11,12);
        bf!(12,13,13);
        bf!(14,15,14);
        _mm512_storeu_si512(ptr.add(0*64+lane).cast(),v[0]);
        _mm512_storeu_si512(ptr.add(1*64+lane).cast(),v[1]);
        _mm512_storeu_si512(ptr.add(2*64+lane).cast(),v[2]);
        _mm512_storeu_si512(ptr.add(3*64+lane).cast(),v[3]);
        _mm512_storeu_si512(ptr.add(4*64+lane).cast(),v[4]);
        _mm512_storeu_si512(ptr.add(5*64+lane).cast(),v[5]);
        _mm512_storeu_si512(ptr.add(6*64+lane).cast(),v[6]);
        _mm512_storeu_si512(ptr.add(7*64+lane).cast(),v[7]);
        _mm512_storeu_si512(ptr.add(8*64+lane).cast(),v[8]);
        _mm512_storeu_si512(ptr.add(9*64+lane).cast(),v[9]);
        _mm512_storeu_si512(ptr.add(10*64+lane).cast(),v[10]);
        _mm512_storeu_si512(ptr.add(11*64+lane).cast(),v[11]);
        _mm512_storeu_si512(ptr.add(12*64+lane).cast(),v[12]);
        _mm512_storeu_si512(ptr.add(13*64+lane).cast(),v[13]);
        _mm512_storeu_si512(ptr.add(14*64+lane).cast(),v[14]);
        _mm512_storeu_si512(ptr.add(15*64+lane).cast(),v[15]);
        lane+=4;
    }
    if wide<lanes {zero_chain_contiguous_scalar_tail(ptr,wide,lanes,prepared);}
}}

#[cfg(target_feature="avx512bw")]
#[cold]
#[inline(never)]
unsafe fn zero_chain_contiguous_scalar_tail(ptr:*mut F128,first:usize,end:usize,prepared:*const PreparedFour){unsafe{
    let tw=core::array::from_fn(|i|(*prepared).pairs[i][0]);
    butterfly_fused_4layer_scalar_tail(ptr,1,64,0,first,end,&tw);
}}

#[cfg(all(test,target_feature="avx512bw"))]
mod zero_chain_tests {
    use super::*;
    #[test]
    fn zero_chain_contiguous_matches_portable_for_live_prefixes(){unsafe{
        let mut state=0x48da91de18u64;
        let mut next=|| {state^=state<<13;state^=state>>7;state^=state<<17;state};
        for case in 0..4 {
            let mut tw=core::array::from_fn(|_|F128::new(next(),next()));
            for i in [0usize,1,3,7] {tw[i]=F128::ZERO;}
            assert!(zero_chain_twiddles(&tw));
            for i in [0usize,1,3,7] {let mut changed=tw;changed[i]=F128::ONE;assert!(!zero_chain_twiddles(&changed));}
            let mut prepared=prepare_four(&tw).unwrap();prepared.zero_chain=true;
            for lanes in [0usize,1,2,3,4,7,8,15,16,31,32,56,57,60,63,64] {
                let original:Vec<F128>=(0..16*64+8).map(|_|F128::new(next(),next())).collect();
                let mut expected=original.clone();let mut got=original;
                for lane in 0..lanes {
                    let mut values=core::array::from_fn(|row|expected[4+row*64+lane]);
                    super::super::portable::butterfly_fused_4layer(&mut values,&tw);
                    for row in 0..16 {expected[4+row*64+lane]=values[row];}
                }
                zero_chain_contiguous_rows(got.as_mut_ptr().add(4),lanes,&prepared);
                assert!(got==expected,"zero-chain case={case} live={lanes}");
            }
        }
        println!("ZERO_CHAIN_ORACLE_OK 64completebuffercomparisons fourguardpositions inactivefields+outerguards");
    }}
}

/// Fold the four real trailing fields of the S8 odd groups together.
/// The caller guarantees 128 rows of 64 initialized fields and seven zero
/// trailing fields on every odd row. No callback may read the block until
/// this function returns. Returns false without changing data on rollback.
#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq"
))]
pub(crate) unsafe fn rows_raw7_s8(ptr: *mut F128, t: &[F128; 15], hint: u8) -> bool {
    if !super::super::ntt_shaped_enabled() {
        return false;
    }
    let Some(p) = (unsafe { prepare_four(t) }) else {
        return false;
    };
    let ordered = super::super::ntt_alias_load_order_enabled();
    unsafe {
        match (hint, ordered) {
            (0, false) => rows_raw7_s8_impl::<0, 0>(ptr, &p),
            (0, true) => rows_raw7_s8_impl::<0, 1>(ptr, &p),
            (1, false) => rows_raw7_s8_impl::<1, 0>(ptr, &p),
            (1, true) => rows_raw7_s8_impl::<1, 1>(ptr, &p),
            (_, false) => rows_raw7_s8_impl::<2, 0>(ptr, &p),
            (_, true) => rows_raw7_s8_impl::<2, 1>(ptr, &p),
        }
    }
    #[cfg(test)]
    RAW7_PACK_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}
#[cfg(all(
    test,
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq"
))]
pub(crate) static RAW7_PACK_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq"
))]
#[inline]
unsafe fn rows_raw7_s8_impl<const H: u8, const ORD: u8>(ptr: *mut F128, p: &PreparedFour) {
    #[repr(C, align(64))]
    struct Tail([F128; 64]);
    let mut storage = core::mem::MaybeUninit::<Tail>::uninit();
    let tail = storage.as_mut_ptr().cast::<F128>();
    let lead = (p.prefetch_lines * 4).min(64);
    unsafe {
        for r in 0..8 {
            let odd = r & 1 == 1;
            let lanes = if odd { 56 } else { 64 };
            if r == 7 || H == 0 {
                butterfly_four_fixed_registers::<8, 64, 0, ORD>(
                    ptr.add(r * 64),
                    p,
                    lanes * 16,
                    ptr,
                );
            } else {
                butterfly_four_fixed_registers::<8, 64, H, ORD>(
                    ptr.add(r * 64),
                    p,
                    lanes * 16,
                    ptr.add(r * 64 + lead),
                );
            }
            if odd {
                // Four odd r values each initialize one column in all 16
                // compact rows. Every one of the 64 fields is written.
                for i in 0..16 {
                    tail.add(i * 4 + (r >> 1))
                        .write(ptr.add((i * 8 + r) * 64 + 56).read());
                }
            }
        }
        // All compact fields are initialized. Columns have the identical
        // twiddles of their original r groups; the butterfly is lane-local.
        butterfly_four_fixed_registers::<1, 4, 0, 0>(tail, p, 64, tail);
        for i in 0..16 {
            for j in 0..4 {
                ptr.add((i * 8 + 2 * j + 1) * 64 + 56)
                    .write(tail.add(i * 4 + j).read());
            }
        }
    }
}
