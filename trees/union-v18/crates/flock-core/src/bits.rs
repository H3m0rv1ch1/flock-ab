//! Small bit-manipulation primitives shared across modules.

/// Hacker's Delight (Sec. 7-3) 8×8 bit-matrix transpose stored in a `u64`.
///
/// The input holds 8 bytes representing 8 rows of 8 bits each; the output holds
/// the transposed matrix (bit `r·8 + c` of input → bit `c·8 + r` of output).
///
/// Shared by the lincheck byte-stripe builder (`flock_prover::r1cs_hashes::common`)
/// and the PCS ring-switch `fold_1b` kernels ([`crate::pcs::ring_switch`]).
#[inline(always)]
pub(crate) fn transpose_8x8_bits(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AAu64;
    x = x ^ t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCCu64;
    x = x ^ t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0u64;
    x = x ^ t ^ (t << 28);
    x
}

/// Bit-transpose 8 little-endian `u64` lanes (the 64-byte block they form) into
/// a 64-byte output stripe.
///
/// The 8 LE u64s viewed as 64 bytes are exactly the input shape of the NEON
/// [`bit_transpose_64bytes`] kernel (input byte `r·8 + c` = byte `c` of lane
/// `r`; output byte `c·8 + t` bit `r` = that byte's bit `t`), so this delegates
/// to it — ~5× fewer ops than the scalar per-column loop. Shared by the
/// lincheck byte-stripe builder (`flock_prover::r1cs_hashes::common`) and the
/// core R1CS matrix-apply ([`crate::r1cs`]).
///
/// [`bit_transpose_64bytes`]: crate::zerocheck::univariate_skip_optimized::bit_transpose_64bytes
///
/// The AVX-512 body needs `VPERMB` (avx512vbmi) and `VGF2P8AFFINEQB` (gfni,
/// in its 512-bit `epi8` form also avx512bw), and it names
/// `core::arch::x86_64`, so it must be behind BOTH an architecture `cfg` and
/// a feature `cfg` — otherwise this file does not compile on aarch64 and
/// faults on an x86-64 CPU without GFNI. Compile-time gating is what the rest
/// of this tree does for GFNI, because the workspace builds with
/// `-C target-cpu=native`.
#[inline(always)]
pub fn transpose_8_u64s_to_64_bytes(lanes: &[u64; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 64);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    // SAFETY: the `cfg` supplies exactly the features the callee enables.
    unsafe {
        transpose_8_u64s_to_64_bytes_gfni(lanes, out)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    )))]
    {
        // SAFETY: [u64; 8] is 64 bytes with no padding; u8 has weaker alignment.
        let input: &[u8; 64] = unsafe { &*(lanes.as_ptr() as *const [u8; 64]) };
        let out64: &mut [u8; 64] = out.try_into().expect("64-byte stripe slice");
        crate::zerocheck::univariate_skip_optimized::bit_transpose_64bytes(input, out64);
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[rustfmt::skip]
#[inline]
#[cfg_attr(
    target_feature = "vpclmulqdq",
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")
)]
#[cfg_attr(
    not(target_feature = "vpclmulqdq"),
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")
)]
unsafe fn transpose_8_u64s_to_64_bytes_gfni(lanes: &[u64; 8], out: &mut [u8]) {
    use core::arch::x86_64::*;
    const I:[u8;64]=[56,48,40,32,24,16,8,0,57,49,41,33,25,17,9,1,58,50,42,34,26,18,10,2,59,51,43,35,27,19,11,3,60,52,44,36,28,20,12,4,61,53,45,37,29,21,13,5,62,54,46,38,30,22,14,6,63,55,47,39,31,23,15,7];
    unsafe {
        let x=_mm512_loadu_si512(lanes.as_ptr() as *const __m512i);
        let i=_mm512_loadu_si512(I.as_ptr() as *const __m512i);
        let id=_mm512_set1_epi64(0x8040201008040201u64 as i64);
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i,_mm512_gf2p8affine_epi64_epi8::<0>(id,_mm512_permutexvar_epi8(i,x)));
    }
}

/// Bit-transpose 8 little-endian `u64` lanes directly into 8 `VGF2P8AFFINEQB`
/// `u64` matrices (where output row `i` sits at byte `7 - i`, matching
/// `u64::from_le_bytes(group).swap_bytes()`).
///
/// Using `0x0102040810204080` (`0x8040201008040201u64.swap_bytes()`) as the
/// `VGF2P8AFFINEQB` selector extracts bit `7 - i` into output byte `i` of each
/// qword, fusing the 8-byte reversal into the affine instruction and writing
/// the 8 `u64` matrices in a single ZMM store with zero stack staging.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[rustfmt::skip]
#[inline]
#[cfg_attr(
    target_feature = "vpclmulqdq",
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")
)]
#[cfg_attr(
    not(target_feature = "vpclmulqdq"),
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")
)]
pub(crate) unsafe fn transpose_8_u64s_to_8_mats_swapped_gfni(lanes: &[u64; 8], out_mats: &mut [u64]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(out_mats.len(), 8);
    const I:[u8;64]=[56,48,40,32,24,16,8,0,57,49,41,33,25,17,9,1,58,50,42,34,26,18,10,2,59,51,43,35,27,19,11,3,60,52,44,36,28,20,12,4,61,53,45,37,29,21,13,5,62,54,46,38,30,22,14,6,63,55,47,39,31,23,15,7];
    const BSWAP:[u8;64]=[7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,23,22,21,20,19,18,17,16,31,30,29,28,27,26,25,24,39,38,37,36,35,34,33,32,47,46,45,44,43,42,41,40,55,54,53,52,51,50,49,48,63,62,61,60,59,58,57,56];
    unsafe {
        let x=_mm512_loadu_si512(lanes.as_ptr() as *const __m512i);
        let i=_mm512_loadu_si512(I.as_ptr() as *const __m512i);
        let bswap=_mm512_loadu_si512(BSWAP.as_ptr() as *const __m512i);
        let id=_mm512_set1_epi64(0x8040201008040201u64 as i64);
        let t=_mm512_gf2p8affine_epi64_epi8::<0>(id,_mm512_permutexvar_epi8(i,x));
        _mm512_storeu_si512(out_mats.as_mut_ptr() as *mut __m512i,_mm512_shuffle_epi8(t,bswap));
    }
}

/// Deinterleave 8 `F128` basis elements into `lo` and `hi` `u64` lanes in ZMM
/// registers and bit-transpose both directly into 16 `VGF2P8AFFINEQB` `u64`
/// matrices (`out_mats[0..8]` for `lo`, `out_mats[8..16]` for `hi`).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[rustfmt::skip]
#[inline]
#[cfg_attr(
    target_feature = "vpclmulqdq",
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")
)]
#[cfg_attr(
    not(target_feature = "vpclmulqdq"),
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")
)]
pub(crate) unsafe fn transpose_8_f128_to_16_mats_swapped_gfni(
    basis: &[crate::field::F128],
    out_mats: &mut [u64],
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(basis.len(), 8);
    debug_assert_eq!(out_mats.len(), 16);
    const I:[u8;64]=[56,48,40,32,24,16,8,0,57,49,41,33,25,17,9,1,58,50,42,34,26,18,10,2,59,51,43,35,27,19,11,3,60,52,44,36,28,20,12,4,61,53,45,37,29,21,13,5,62,54,46,38,30,22,14,6,63,55,47,39,31,23,15,7];
    const BSWAP:[u8;64]=[7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,23,22,21,20,19,18,17,16,31,30,29,28,27,26,25,24,39,38,37,36,35,34,33,32,47,46,45,44,43,42,41,40,55,54,53,52,51,50,49,48,63,62,61,60,59,58,57,56];
    unsafe {
        let p = basis.as_ptr() as *const __m512i;
        let v0 = _mm512_loadu_si512(p);
        let v1 = _mm512_loadu_si512(p.add(1));
        let lo_idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let hi_idx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        let lo = _mm512_permutex2var_epi64(v0, lo_idx, v1);
        let hi = _mm512_permutex2var_epi64(v0, hi_idx, v1);
        let i = _mm512_loadu_si512(I.as_ptr() as *const __m512i);
        let bswap = _mm512_loadu_si512(BSWAP.as_ptr() as *const __m512i);
        let id = _mm512_set1_epi64(0x8040201008040201u64 as i64);
        let dst = out_mats.as_mut_ptr() as *mut __m512i;
        let t_lo = _mm512_gf2p8affine_epi64_epi8::<0>(id, _mm512_permutexvar_epi8(i, lo));
        let t_hi = _mm512_gf2p8affine_epi64_epi8::<0>(id, _mm512_permutexvar_epi8(i, hi));
        _mm512_storeu_si512(dst, _mm512_shuffle_epi8(t_lo, bswap));
        _mm512_storeu_si512(dst.add(1), _mm512_shuffle_epi8(t_hi, bswap));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference for [`transpose_8_u64s_to_64_bytes`] — test oracle only.
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn transpose_8_u64s_to_64_bytes_scalar(lanes: &[u64; 8], out: &mut [u8]) {
        debug_assert_eq!(out.len(), 64);
        for c in 0..8 {
            let shift = c * 8;
            let mut packed: u64 = 0;
            packed |= ((lanes[0] >> shift) & 0xFF) << (0 * 8);
            packed |= ((lanes[1] >> shift) & 0xFF) << (1 * 8);
            packed |= ((lanes[2] >> shift) & 0xFF) << (2 * 8);
            packed |= ((lanes[3] >> shift) & 0xFF) << (3 * 8);
            packed |= ((lanes[4] >> shift) & 0xFF) << (4 * 8);
            packed |= ((lanes[5] >> shift) & 0xFF) << (5 * 8);
            packed |= ((lanes[6] >> shift) & 0xFF) << (6 * 8);
            packed |= ((lanes[7] >> shift) & 0xFF) << (7 * 8);
            let transposed = transpose_8x8_bits(packed);
            out[c * 8..c * 8 + 8].copy_from_slice(&transposed.to_le_bytes());
        }
    }

    /// The NEON-delegating transpose must match the scalar per-column oracle
    /// bit-for-bit on varied inputs.
    #[test]
    fn transpose_8_u64s_matches_scalar() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        for _ in 0..100 {
            let lanes: [u64; 8] = std::array::from_fn(|_| next());
            let mut fast = [0u8; 64];
            let mut oracle = [0u8; 64];
            transpose_8_u64s_to_64_bytes(&lanes, &mut fast);
            transpose_8_u64s_to_64_bytes_scalar(&lanes, &mut oracle);
            assert_eq!(fast, oracle);
        }
        // Edge patterns.
        for lanes in [[0u64; 8], [u64::MAX; 8], std::array::from_fn(|i| 1u64 << i)] {
            let mut fast = [0u8; 64];
            let mut oracle = [0u8; 64];
            transpose_8_u64s_to_64_bytes(&lanes, &mut fast);
            transpose_8_u64s_to_64_bytes_scalar(&lanes, &mut oracle);
            assert_eq!(fast, oracle, "lanes={lanes:?}");
        }
    }

    /// Transposing twice is the identity.
    #[test]
    fn transpose_is_involution() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..256 {
            state = state.wrapping_mul(0x2545_F491_4F6C_DD1D).rotate_left(31);
            assert_eq!(transpose_8x8_bits(transpose_8x8_bits(state)), state);
        }
    }

    /// Cross-check against a naive bit-by-bit transpose of the 8×8 matrix.
    #[test]
    fn matches_naive() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..256 {
            state = state.wrapping_mul(0x2545_F491_4F6C_DD1D).rotate_left(17);
            let got = transpose_8x8_bits(state);
            let mut want = 0u64;
            for r in 0..8 {
                for c in 0..8 {
                    if (state >> (r * 8 + c)) & 1 == 1 {
                        want |= 1u64 << (c * 8 + r);
                    }
                }
            }
            assert_eq!(got, want, "input={state:016x}");
        }
    }
}
