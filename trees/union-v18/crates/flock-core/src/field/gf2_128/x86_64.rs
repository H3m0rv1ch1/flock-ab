use super::{F128, F256Unreduced, ghash_reduce};
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn pmull(a: u64, b: u64) -> __m128i {
    let va = _mm_set_epi64x(0, a as i64);
    let vb = _mm_set_epi64x(0, b as i64);
    _mm_clmulepi64_si128::<0x00>(va, vb)
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane0(v: __m128i) -> u64 {
    _mm_extract_epi64::<0>(v) as u64
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane1(v: __m128i) -> u64 {
    _mm_extract_epi64::<1>(v) as u64
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_low_rhs(a: F128, b_lo: u64) -> F128 {
    unsafe {
        let p0 = pmull(a.lo, b_lo);
        let q = pmull(a.hi, b_lo);
        super::ghash_reduce(lane0(p0), lane1(p0) ^ lane0(q), lane1(q), 0)
    }
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_schoolbook(a: F128, b: F128) -> F128 {
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let cr_lo = lane0(cross);
        let cr_hi = lane1(cross);

        ghash_reduce(
            lane0(p_ll),
            lane1(p_ll) ^ cr_lo,
            lane0(p_hh) ^ cr_hi,
            lane1(p_hh),
        )
    }
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_binius(a: F128, b: F128) -> F128 {
    unsafe {
        let t0 = pmull(a.lo, b.lo);
        let t1a = pmull(a.lo, b.hi);
        let t1b = pmull(a.hi, b.lo);
        let t2 = pmull(a.hi, b.hi);
        let mut t1 = _mm_xor_si128(t1a, t1b);

        let t2_shifted = _mm_slli_si128::<8>(t2); // {0, t2.lo}
        t1 = _mm_xor_si128(t1, t2_shifted);
        let t2_red = pmull(lane1(t2), 0x87);
        t1 = _mm_xor_si128(t1, t2_red);

        let t1_shifted = _mm_slli_si128::<8>(t1); // {0, t1.lo}
        let mut t0 = _mm_xor_si128(t0, t1_shifted);
        let t1_red = pmull(lane1(t1), 0x87);
        t0 = _mm_xor_si128(t0, t1_red);

        F128 {
            lo: lane0(t0),
            hi: lane1(t0),
        }
    }
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba_vec(a: F128, b: F128) -> F128 {
    unsafe {
        let x = _mm_set_epi64x(a.hi as i64, a.lo as i64);
        let y = _mm_set_epi64x(b.hi as i64, b.lo as i64);
        let low = _mm_clmulepi64_si128::<0x00>(x, y);
        let high = _mm_clmulepi64_si128::<0x11>(x, y);
        let xs = _mm_xor_si128(x, _mm_shuffle_epi32::<0x4e>(x));
        let ys = _mm_xor_si128(y, _mm_shuffle_epi32::<0x4e>(y));
        let middle = _mm_xor_si128(
            _mm_xor_si128(_mm_clmulepi64_si128::<0x00>(xs, ys), low), high);
        reduce_register_product(low, middle, high)
    }
}

/// Reduce the three Karatsuba limbs without moving intermediates to GPRs.
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn reduce_register_product(low: __m128i, middle: __m128i, high: __m128i) -> F128 {
    unsafe {
        let polynomial = _mm_set_epi64x(0, 0x87);
        let middle = _mm_xor_si128(middle, _mm_xor_si128(
            _mm_slli_si128::<8>(high),
            _mm_clmulepi64_si128::<0x01>(high, polynomial)));
        let result = _mm_xor_si128(low, _mm_xor_si128(
            _mm_slli_si128::<8>(middle),
            _mm_clmulepi64_si128::<0x01>(middle, polynomial)));
        F128 { lo: lane0(result), hi: lane1(result) }
    }
}

/// Squaring has no cross term in characteristic two.
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_square_register(a: F128) -> F128 {
    unsafe {
        let x = _mm_set_epi64x(a.hi as i64, a.lo as i64);
        reduce_register_product(
            _mm_clmulepi64_si128::<0x00>(x, x),
            _mm_setzero_si128(),
            _mm_clmulepi64_si128::<0x11>(x, x))
    }
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba(a: F128, b: F128) -> F128 {
    unsafe {
        let p0 = pmull(a.lo, b.lo);
        let p1 = pmull(a.hi, b.hi);
        let pm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);

        let p0_lo = lane0(p0);
        let p0_hi = lane1(p0);
        let p1_lo = lane0(p1);
        let p1_hi = lane1(p1);
        let pm_lo = lane0(pm);
        let pm_hi = lane1(pm);

        let cross_lo = pm_lo ^ p0_lo ^ p1_lo;
        let cross_hi = pm_hi ^ p0_hi ^ p1_hi;

        ghash_reduce(p0_lo, p0_hi ^ cross_lo, p1_lo ^ cross_hi, p1_hi)
    }
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba_barrett(a: F128, b: F128) -> F128 {
    unsafe {
        let d0 = pmull(a.lo, b.lo);
        let d2 = pmull(a.hi, b.hi);
        let dm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);
        let d1 = _mm_xor_si128(_mm_xor_si128(dm, d0), d2);

        let d0_lo = lane0(d0);
        let d0_hi = lane1(d0);
        let d1_lo = lane0(d1);
        let d1_hi = lane1(d1);
        let d2_lo = lane0(d2);
        let d2_hi = lane1(d2);

        let lo_lo = d0_lo;
        let lo_hi = d0_hi ^ d1_lo;
        let hi_lo = d2_lo ^ d1_hi;
        let hi_hi = d2_hi;

        let r_hi = pmull(hi_hi, 0x87);
        let r_lo = pmull(hi_lo, 0x87);

        let r_lo_lo = lane0(r_lo);
        let r_lo_hi = lane1(r_lo);
        let r_hi_lo = lane0(r_hi);
        let r_hi_hi = lane1(r_hi);

        let ov = r_hi_hi;
        let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);

        F128 {
            lo: lo_lo ^ r_lo_lo ^ corr,
            hi: lo_hi ^ r_lo_hi ^ r_hi_lo,
        }
    }
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_unreduced_x86(a: F128, b: F128) -> F256Unreduced {
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let cr_lo = lane0(cross);
        let cr_hi = lane1(cross);

        F256Unreduced {
            r0: lane0(p_ll),
            r1: lane1(p_ll) ^ cr_lo,
            r2: lane0(p_hh) ^ cr_hi,
            r3: lane1(p_hh),
        }
    }
}


#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn ghash_poly_x4() -> __m512i {
    _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87)
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn gf2_128_reduce_x4(mut t0: __m512i, t1: __m512i) -> __m512i {
    unsafe {
        let poly = ghash_poly_x4();
        t0 = _mm512_xor_si512(t0, _mm512_bslli_epi128::<8>(t1));
        t0 = _mm512_xor_si512(t0, _mm512_clmulepi64_epi128::<0x01>(t1, poly));
        t0
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_reduce_acc_x4(lo: __m512i, mid: __m512i, hi: __m512i) -> __m512i {
    unsafe {
        let t1 = gf2_128_reduce_x4(mid, hi);
        gf2_128_reduce_x4(lo, t1)
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4(x: __m512i, y: __m512i) -> __m512i {
    unsafe {
        let t1a = _mm512_clmulepi64_epi128::<0x01>(x, y);
        let t1b = _mm512_clmulepi64_epi128::<0x10>(x, y);
        let mut t1 = _mm512_xor_si512(t1a, t1b);
        let t2 = _mm512_clmulepi64_epi128::<0x11>(x, y);
        t1 = gf2_128_reduce_x4(t1, t2);
        let t0 = _mm512_clmulepi64_epi128::<0x00>(x, y);
        gf2_128_reduce_x4(t0, t1)
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4_low_lhs(x: __m512i, y: __m512i) -> __m512i {
    unsafe {
        let t1 = _mm512_clmulepi64_epi128::<0x10>(x, y);
        let t0 = _mm512_clmulepi64_epi128::<0x00>(x, y);
        gf2_128_reduce_x4(t0, t1)
    }
}


#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_shift64_x4(t: __m512i) -> __m512i {
    unsafe { gf2_128_reduce_x4(_mm512_setzero_si512(), t) }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4_split(v: __m512i, t: __m512i, t_x64: __m512i) -> __m512i {
    unsafe {
        let lo = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x00>(v, t),
            _mm512_clmulepi64_epi128::<0x01>(v, t_x64),
        );
        let hi = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x10>(v, t),
            _mm512_clmulepi64_epi128::<0x11>(v, t_x64),
        );
        gf2_128_reduce_x4(lo, hi)
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_broadcast_split(t: F128) -> (__m512i, __m512i) {
    unsafe {
        let tv = _mm512_broadcast_i32x4(_mm_set_epi64x(t.hi as i64, t.lo as i64));
        (tv, ghash_shift64_x4(tv))
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4_split_unroll4(
    v0: __m512i,
    v1: __m512i,
    v2: __m512i,
    v3: __m512i,
    t: __m512i,
    t_x64: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    unsafe {
        (
            ghash_mul_x4_split(v0, t, t_x64),
            ghash_mul_x4_split(v1, t, t_x64),
            ghash_mul_x4_split(v2, t, t_x64),
            ghash_mul_x4_split(v3, t, t_x64),
        )
    }
}


#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_loadu(p: *const F128) -> __m512i {
    unsafe { _mm512_loadu_si512(p as *const __m512i) }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,sse4.1")]
pub unsafe fn f128x4_extract(v: __m512i) -> [F128; 4] {
    unsafe {
        let l0 = _mm512_extracti32x4_epi32::<0>(v);
        let l1 = _mm512_extracti32x4_epi32::<1>(v);
        let l2 = _mm512_extracti32x4_epi32::<2>(v);
        let l3 = _mm512_extracti32x4_epi32::<3>(v);
        [
            F128 {
                lo: _mm_extract_epi64::<0>(l0) as u64,
                hi: _mm_extract_epi64::<1>(l0) as u64,
            },
            F128 {
                lo: _mm_extract_epi64::<0>(l1) as u64,
                hi: _mm_extract_epi64::<1>(l1) as u64,
            },
            F128 {
                lo: _mm_extract_epi64::<0>(l2) as u64,
                hi: _mm_extract_epi64::<1>(l2) as u64,
            },
            F128 {
                lo: _mm_extract_epi64::<0>(l3) as u64,
                hi: _mm_extract_epi64::<1>(l3) as u64,
            },
        ]
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_set(a: F128, b: F128, c: F128, d: F128) -> __m512i {
    _mm512_set_epi64(
        d.hi as i64,
        d.lo as i64,
        c.hi as i64,
        c.lo as i64,
        b.hi as i64,
        b.lo as i64,
        a.hi as i64,
        a.lo as i64,
    )
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xor4_lanes(v: __m512i) -> __m128i {
    let l0 = _mm512_extracti32x4_epi32::<0>(v);
    let l1 = _mm512_extracti32x4_epi32::<1>(v);
    let l2 = _mm512_extracti32x4_epi32::<2>(v);
    let l3 = _mm512_extracti32x4_epi32::<3>(v);
    _mm_xor_si128(_mm_xor_si128(l0, l1), _mm_xor_si128(l2, l3))
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[derive(Clone, Copy)]
pub struct WideGhashX4 {
    pub(crate) lo: __m512i,
    pub(crate) hi: __m512i,
    pub(crate) mid: __m512i,
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
impl WideGhashX4 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn zero() -> Self {
        let z = _mm512_setzero_si512();
        Self {
            lo: z,
            hi: z,
            mid: z,
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn xor(self, other: Self) -> Self {
        Self {
            lo: _mm512_xor_si512(self.lo, other.lo),
            hi: _mm512_xor_si512(self.hi, other.hi),
            mid: _mm512_xor_si512(self.mid, other.mid),
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn from_mul(x: __m512i, y: __m512i) -> Self {
        Self {
            lo: _mm512_clmulepi64_epi128::<0x00>(x, y),
            hi: _mm512_clmulepi64_epi128::<0x11>(x, y),
            mid: _mm512_xor_si512(
                _mm512_clmulepi64_epi128::<0x01>(x, y),
                _mm512_clmulepi64_epi128::<0x10>(x, y),
            ),
        }
    }

    pub unsafe fn mul_acc(&mut self, x: __m512i, y: __m512i) {
        self.lo = _mm512_xor_si512(self.lo, _mm512_clmulepi64_epi128::<0x00>(x, y));
        self.hi = _mm512_xor_si512(self.hi, _mm512_clmulepi64_epi128::<0x11>(x, y));
        self.mid = _mm512_ternarylogic_epi64::<0x96>(
            self.mid,
            _mm512_clmulepi64_epi128::<0x01>(x, y),
            _mm512_clmulepi64_epi128::<0x10>(x, y),
        );
    }

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn mul_acc2(&mut self, x0: __m512i, y0: __m512i, x1: __m512i, y1: __m512i) {
        let c00_0 = _mm512_clmulepi64_epi128::<0x00>(x0, y0);
        let c00_1 = _mm512_clmulepi64_epi128::<0x00>(x1, y1);
        let c11_0 = _mm512_clmulepi64_epi128::<0x11>(x0, y0);
        let c11_1 = _mm512_clmulepi64_epi128::<0x11>(x1, y1);
        let c01_0 = _mm512_clmulepi64_epi128::<0x01>(x0, y0);
        let c10_0 = _mm512_clmulepi64_epi128::<0x10>(x0, y0);
        let c01_1 = _mm512_clmulepi64_epi128::<0x01>(x1, y1);
        let c10_1 = _mm512_clmulepi64_epi128::<0x10>(x1, y1);
        self.lo = _mm512_ternarylogic_epi64::<0x96>(self.lo, c00_0, c00_1);
        self.hi = _mm512_ternarylogic_epi64::<0x96>(self.hi, c11_0, c11_1);
        self.mid = _mm512_ternarylogic_epi64::<0x96>(self.mid, c01_0, c10_0);
        self.mid = _mm512_ternarylogic_epi64::<0x96>(self.mid, c01_1, c10_1);
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn mul_acc_one(&mut self, x: __m512i) {
        self.lo = _mm512_xor_si512(self.lo, _mm512_maskz_mov_epi64(0x55, x));
        self.mid = _mm512_xor_si512(self.mid, _mm512_bsrli_epi128::<8>(x));
    }

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn fold4_and_reduce(
        a0: Self,
        a1: Self,
        a2: Self,
        a3: Self,
    ) -> __m512i {
        unsafe {
            let s1_lo = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
            let s1_hi = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
            let s2_lo = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
            let s2_hi = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
            let hfold4 = |v0: __m512i, v1: __m512i, v2: __m512i, v3: __m512i| -> __m512i {
                let t0 = _mm512_xor_si512(
                    _mm512_permutex2var_epi64(v0, s1_lo, v1),
                    _mm512_permutex2var_epi64(v0, s1_hi, v1),
                );
                let t1 = _mm512_xor_si512(
                    _mm512_permutex2var_epi64(v2, s1_lo, v3),
                    _mm512_permutex2var_epi64(v2, s1_hi, v3),
                );
                _mm512_xor_si512(
                    _mm512_permutex2var_epi64(t0, s2_lo, t1),
                    _mm512_permutex2var_epi64(t0, s2_hi, t1),
                )
            };
            let lo = hfold4(a0.lo, a1.lo, a2.lo, a3.lo);
            let mid = hfold4(a0.mid, a1.mid, a2.mid, a3.mid);
            let hi = hfold4(a0.hi, a1.hi, a2.hi, a3.hi);
            ghash_reduce_acc_x4(lo, mid, hi)
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn reduce_lanes(self) -> __m512i {
        unsafe { ghash_reduce_acc_x4(self.lo, self.mid, self.hi) }
    }

    #[inline]
    #[target_feature(enable = "avx512f,sse4.1")]
    pub unsafe fn fold(self) -> F256Unreduced {
        unsafe {
            let lo = xor4_lanes(self.lo);
            let hi = xor4_lanes(self.hi);
            let mid = xor4_lanes(self.mid);
            let lo_lo = _mm_extract_epi64::<0>(lo) as u64;
            let lo_hi = _mm_extract_epi64::<1>(lo) as u64;
            let hi_lo = _mm_extract_epi64::<0>(hi) as u64;
            let hi_hi = _mm_extract_epi64::<1>(hi) as u64;
            let mid_lo = _mm_extract_epi64::<0>(mid) as u64;
            let mid_hi = _mm_extract_epi64::<1>(mid) as u64;
            F256Unreduced {
                r0: lo_lo,
                r1: lo_hi ^ mid_lo,
                r2: hi_lo ^ mid_hi,
                r3: hi_hi,
            }
        }
    }
}
