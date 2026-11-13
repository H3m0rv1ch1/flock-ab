use super::super::{F128, build_sum_table};

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[repr(C, align(64))]
struct FusedGatherIndices([u8; 64]);

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
static FUSED_GATHER_LO: FusedGatherIndices =
    FusedGatherIndices(super::super::gather_transpose_vpermt2b_indices(false));

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
static FUSED_GATHER_HI: FusedGatherIndices =
    FusedGatherIndices(super::super::gather_transpose_vpermt2b_indices(true));

/// GFNI twin of [`partial_fold_packed_z_x86_tiled_padded`]: each stripe's
/// 256-entry sum table is F2-linear (`T[0] = 0`, XOR-composed from the eight
/// `eq_outer` values), so it IS sixteen 8×8 bit matrices, and
/// `VGF2P8AFFINEQB` folds 64 output columns per instruction with no table
/// loads and no table build. Accumulation is byte-plane-major (plane `k` of
/// 64 columns per ZMM, sixteen independent chains) and transposes back to
/// F128s once after the parallel reduce — the same XOR terms reassociated,
/// so the result is bit-identical.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
pub fn partial_fold_packed_z_x86_gfni_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 6, "need n_outer ≥ 64 for tile of 8 stripes");
    assert!(k_log >= 6, "GFNI fold needs 64-column blocks");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    // Columns past useful_bits hold zero padding: affine(0) = 0 contributes
    // nothing, so cover only the 64-column blocks that touch useful bits.
    let n_blocks64 = useful_bits.div_ceil(64).min(k / 64);

    let n_tiles = n_stripes / TILE_T;
    let tiles_per_chunk = (n_tiles / 256).max(1);
    let bytes_per_chunk = tiles_per_chunk * TILE_T * k;

    let planes = z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![0u8; k * 16],
            |mut out_planes, (chunk_idx, chunk_bytes)| {
                let tile_start = chunk_idx * tiles_per_chunk;
                let n_tiles_in_chunk = chunk_bytes.len() / (TILE_T * k);
                let mut mats = [0u64; TILE_T * 16];
                for tile_rel in 0..n_tiles_in_chunk {
                    let tile_idx = tile_start + tile_rel;
                    let stripe_base = tile_idx * TILE_T;
                    for t in 0..TILE_T {
                        let eq_off = 8 * (stripe_base + t);
                        fold_mats_from_basis(
                            &eq_outer[eq_off..eq_off + 8],
                            &mut mats[t * 16..(t + 1) * 16],
                        );
                    }
                    // SAFETY: tile_rel < n_tiles_in_chunk keeps the tile in
                    // bounds; the block loop stays within k columns and the
                    // plane buffer is k*16 bytes. tile_rel == 0 seeds from
                    // the fold's zeroed plane buffer (XOR 0 is identity).
                    unsafe {
                        gfni_fold_tile(
                            chunk_bytes.as_ptr().add(tile_rel * TILE_T * k),
                            k,
                            n_blocks64,
                            &mats,
                            out_planes.as_mut_ptr(),
                            tile_rel == 0,
                        );
                    }
                }
                out_planes
            },
        )
        .reduce(
            || vec![0u8; k * 16],
            |mut a, b| {
                // Plane XOR merge — same sums, still plane-major.
                // k_log ≥ 6 ⇒ k*16 is a multiple of 64.
                debug_assert_eq!(a.len(), b.len());
                debug_assert_eq!(a.len() % 64, 0);
                // SAFETY: both plane buffers are k*16 bytes, 64-aligned length.
                unsafe {
                    xor_bytes_avx512(a.as_mut_ptr(), b.as_ptr(), a.len());
                }
                a
            },
        );

    // One transpose back to F128 columns at the very end.
    let mut out = vec![F128::ZERO; k];
    for b in 0..k / 64 {
        let base = b * 1024;
        for col in 0..64 {
            let mut lo = 0u64;
            let mut hi = 0u64;
            for byte in 0..8 {
                lo |= (planes[base + byte * 64 + col] as u64) << (8 * byte);
            }
            for byte in 8..16 {
                hi |= (planes[base + byte * 64 + col] as u64) << (8 * (byte - 8));
            }
            out[b * 64 + col] = F128 { lo, hi };
        }
    }
    out
}

/// Fused gather + bit-transpose of one stripe: eight strided F128 lanes go
/// straight from memory to the 128-byte transposed form in registers — no
/// staging arrays, no store-forwarding round trips. Byte-level transpose by
/// `vpermb` (out byte `c*8 + r` = lane r's byte c), bit-level flip by
/// `VGF2P8AFFINEQB` with the bit-transpose identity — the same recipe as
/// the zerocheck fused C drain. Output byte `b` of half h, bit r = bit
/// `8*b_local` ... semantics identical to `transpose_8_f128s_to_128_bytes`
/// (asserted by the oracle test).
///
/// # Safety
/// Eight readable F128 at `z_ptr + r*stride` for r in 0..8; `out` covers
/// 128 writable bytes; avx512f + avx512vbmi + gfni (module/cfg-gated).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[cfg_attr(
    all(target_feature = "avx512bw", target_feature = "vpclmulqdq"),
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")
)]
#[cfg_attr(
    not(all(target_feature = "avx512bw", target_feature = "vpclmulqdq")),
    target_feature(enable = "avx512f,avx512vbmi,gfni")
)]
#[inline]
pub(crate) unsafe fn gather_transpose_stripe_x86<const FUSE: bool, const RAW: bool>(
    z_ptr: *const F128,
    stride: usize,
    out: *mut u8,
) {
    use core::arch::x86_64::*;
    const BIT_TRANSPOSE_ID: i64 = 0x8040_2010_0804_0201u64 as i64;
    // SAFETY: bounds per the contract; features per the cfg gate.
    unsafe {
        let ld = |r: usize| _mm_loadu_si128(z_ptr.add(r * stride) as *const __m128i);
        // Build two ZMMs of four lanes each (lane r occupies 128-bit slot r%4).
        let z0 = {
            let a = _mm512_castsi128_si512(ld(0));
            let a = _mm512_inserti32x4::<1>(a, ld(1));
            let a = _mm512_inserti32x4::<2>(a, ld(2));
            _mm512_inserti32x4::<3>(a, ld(3))
        };
        let z1 = {
            let a = _mm512_castsi128_si512(ld(4));
            let a = _mm512_inserti32x4::<1>(a, ld(5));
            let a = _mm512_inserti32x4::<2>(a, ld(6));
            _mm512_inserti32x4::<3>(a, ld(7))
        };
        let ident = _mm512_set1_epi64(BIT_TRANSPOSE_ID);
        if FUSE {
            // One VPERMT2B composes the old lo/hi qword split with the 8x8
            // byte transpose. Index bit 6 selects z1 (rows 4..8).
            let f_lo = _mm512_load_si512(FUSED_GATHER_LO.0.as_ptr() as *const __m512i);
            let f_hi = _mm512_load_si512(FUSED_GATHER_HI.0.as_ptr() as *const __m512i);
            let t_lo = if RAW {
                _mm512_permutex2var_epi8(z0, f_lo, z1)
            } else {
                _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutex2var_epi8(z0, f_lo, z1))
            };
            let t_hi = if RAW {
                _mm512_permutex2var_epi8(z0, f_hi, z1)
            } else {
                _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutex2var_epi8(z0, f_hi, z1))
            };
            _mm512_storeu_si512(out as *mut __m512i, t_lo);
            _mm512_storeu_si512(out.add(64) as *mut __m512i, t_hi);
        } else {
            // Incumbent: split into lo/hi qwords, then transpose bytes.
            let lo_idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
            let hi_idx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
            let zlo = _mm512_permutex2var_epi64(z0, lo_idx, z1);
            let zhi = _mm512_permutex2var_epi64(z0, hi_idx, z1);
            #[repr(C, align(64))]
            struct BIdx([u8; 64]);
            static BIDX: BIdx = {
                let mut t = [0u8; 64];
                let mut i = 0;
                while i < 64 {
                    t[i] = ((7 - i % 8) * 8 + i / 8) as u8;
                    i += 1;
                }
                BIdx(t)
            };
            let bidx = _mm512_load_si512(BIDX.0.as_ptr() as *const __m512i);
            let t_lo = if RAW {
                _mm512_permutexvar_epi8(bidx, zlo)
            } else {
                _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(bidx, zlo))
            };
            let t_hi = if RAW {
                _mm512_permutexvar_epi8(bidx, zhi)
            } else {
                _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(bidx, zhi))
            };
            _mm512_storeu_si512(out as *mut __m512i, t_lo);
            _mm512_storeu_si512(out.add(64) as *mut __m512i, t_hi);
        }
    }
}

/// With `RAW`, output the byte-transposed GFNI matrices before bit transpose.
/// This lets root selection commute with the linear bit-transpose operator.
/// Four-column twin of [`gather_transpose_stripe_x86`]: gathers columns
/// `q..q+4` of the eight rows in ONE pass — eight 64-byte row loads plus two
/// 4×4 lane transposes replace thirty-two 16-byte strided loads and
/// twenty-four inserts. The row stride is 2048 bytes at the ranked shape, so
/// a tile's 64 live rows land in only two L1 sets; batching four columns per
/// visit quarters the number of times each row line must survive in those
/// sets. Column `c`'s 128 transposed bytes land at `out + c*out_stride +
/// (the caller's stripe offset)` — identical bytes to four single calls.
///
/// # Safety
/// 4 readable F128 at `z_ptr + r*stride + c` for r in 0..8, c in 0..4; `out
/// + c*out_stride` covers 128 writable bytes for c in 0..4; features per the
/// cfg gate.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[cfg_attr(
    all(target_feature = "avx512bw", target_feature = "vpclmulqdq"),
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")
)]
#[cfg_attr(
    not(all(target_feature = "avx512bw", target_feature = "vpclmulqdq")),
    target_feature(enable = "avx512f,avx512vbmi,gfni")
)]
#[inline]
pub(crate) unsafe fn gather_transpose_stripe4_x86<const FUSE: bool, const RAW: bool>(
    z_ptr: *const F128,
    stride: usize,
    out: *mut u8,
    out_stride: usize,
) {
    use core::arch::x86_64::*;
    const BIT_TRANSPOSE_ID: i64 = 0x8040_2010_0804_0201u64 as i64;
    // SAFETY: bounds per the contract; features per the cfg gate.
    unsafe {
        // Row r's columns q..q+3 in one wide load (64-aligned in production —
        // the pool's recyclable class — but loadu tolerates test vectors).
        let ld = |r: usize| _mm512_loadu_si512(z_ptr.add(r * stride) as *const __m512i);
        let r = [ld(0), ld(1), ld(2), ld(3), ld(4), ld(5), ld(6), ld(7)];
        // 4×4 transpose of 128-bit lanes: cols[c] = lanes {row0..row3 at c}.
        #[inline(always)]
        unsafe fn tr4(a: __m512i, b: __m512i, c: __m512i, d: __m512i) -> [__m512i; 4] {
            // SAFETY: caller carries avx512f.
            unsafe {
                let ab_lo = _mm512_shuffle_i64x2::<0x44>(a, b); // a0 a1 b0 b1
                let ab_hi = _mm512_shuffle_i64x2::<0xEE>(a, b); // a2 a3 b2 b3
                let cd_lo = _mm512_shuffle_i64x2::<0x44>(c, d);
                let cd_hi = _mm512_shuffle_i64x2::<0xEE>(c, d);
                [
                    _mm512_shuffle_i64x2::<0x88>(ab_lo, cd_lo), // a0 b0 c0 d0
                    _mm512_shuffle_i64x2::<0xDD>(ab_lo, cd_lo), // a1 b1 c1 d1
                    _mm512_shuffle_i64x2::<0x88>(ab_hi, cd_hi), // a2 b2 c2 d2
                    _mm512_shuffle_i64x2::<0xDD>(ab_hi, cd_hi), // a3 b3 c3 d3
                ]
            }
        }
        let z0s = tr4(r[0], r[1], r[2], r[3]); // per column: rows 0..3
        let z1s = tr4(r[4], r[5], r[6], r[7]); // per column: rows 4..7
        let ident = _mm512_set1_epi64(BIT_TRANSPOSE_ID);
        if FUSE {
            let f_lo = _mm512_load_si512(FUSED_GATHER_LO.0.as_ptr() as *const __m512i);
            let f_hi = _mm512_load_si512(FUSED_GATHER_HI.0.as_ptr() as *const __m512i);
            for c in 0..4 {
                let t_lo = if RAW {
                    _mm512_permutex2var_epi8(z0s[c], f_lo, z1s[c])
                } else {
                    _mm512_gf2p8affine_epi64_epi8::<0>(
                        ident,
                        _mm512_permutex2var_epi8(z0s[c], f_lo, z1s[c]),
                    )
                };
                let t_hi = if RAW {
                    _mm512_permutex2var_epi8(z0s[c], f_hi, z1s[c])
                } else {
                    _mm512_gf2p8affine_epi64_epi8::<0>(
                        ident,
                        _mm512_permutex2var_epi8(z0s[c], f_hi, z1s[c]),
                    )
                };
                let dst = out.add(c * out_stride);
                _mm512_storeu_si512(dst as *mut __m512i, t_lo);
                _mm512_storeu_si512(dst.add(64) as *mut __m512i, t_hi);
            }
        } else {
            let lo_idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
            let hi_idx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
            #[repr(C, align(64))]
            struct BIdx4([u8; 64]);
            static BIDX4: BIdx4 = {
                let mut t = [0u8; 64];
                let mut i = 0;
                while i < 64 {
                    t[i] = ((7 - i % 8) * 8 + i / 8) as u8;
                    i += 1;
                }
                BIdx4(t)
            };
            let bidx = _mm512_load_si512(BIDX4.0.as_ptr() as *const __m512i);
            for c in 0..4 {
                let zlo = _mm512_permutex2var_epi64(z0s[c], lo_idx, z1s[c]);
                let zhi = _mm512_permutex2var_epi64(z0s[c], hi_idx, z1s[c]);
                let t_lo = if RAW {
                    _mm512_permutexvar_epi8(bidx, zlo)
                } else {
                    _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(bidx, zlo))
                };
                let t_hi = if RAW {
                    _mm512_permutexvar_epi8(bidx, zhi)
                } else {
                    _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(bidx, zhi))
                };
                let dst = out.add(c * out_stride);
                _mm512_storeu_si512(dst as *mut __m512i, t_lo);
                _mm512_storeu_si512(dst.add(64) as *mut __m512i, t_hi);
            }
        }
    }
}

/// `FLOCK_NO_LC_MATS_AOS=1` restores sixteen scalar `.lo`/`.hi` extracts.
/// Default: two `loadu` of the AoS `[F128; 8]` and one even/odd qword
/// deinterleave into the lo/hi lane arrays the bit-transpose already
/// consumes. Same 16 qwords.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
fn lc_mats_aos_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_LC_MATS_AOS").is_none())
}

/// Deinterleave eight AoS F128 into lo-qword and hi-qword lane arrays.
///
/// # Safety
/// `eq8.len() == 8`. `avx512f`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f")]
unsafe fn aos8_lohi(eq8: &[F128]) -> ([u64; 8], [u64; 8]) {
    use core::arch::x86_64::*;
    let mut lo_lanes = [0u64; 8];
    let mut hi_lanes = [0u64; 8];
    // SAFETY: eight contiguous F128 (128 bytes) = two ZMM. F128 is lo||hi
    // qwords, so even qwords are `.lo` and odd qwords are `.hi`.
    unsafe {
        let p = eq8.as_ptr() as *const __m512i;
        let v0 = _mm512_loadu_si512(p);
        let v1 = _mm512_loadu_si512(p.add(1));
        let lo_idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let hi_idx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        _mm512_storeu_si512(
            lo_lanes.as_mut_ptr() as *mut __m512i,
            _mm512_permutex2var_epi64(v0, lo_idx, v1),
        );
        _mm512_storeu_si512(
            hi_lanes.as_mut_ptr() as *mut __m512i,
            _mm512_permutex2var_epi64(v0, hi_idx, v1),
        );
    }
    (lo_lanes, hi_lanes)
}

/// The sixteen `VGF2P8AFFINEQB` matrices of one stripe's sum table, straight
/// from its eight `eq_outer` basis values (encoding: `out.bit[i] =
/// parity(byte[7-i] & in)`; input bit `j` ↔ stripe bit `j`, matching
/// `build_sum_table`'s `T[1 << j] = eq8[j]`).
///
/// Built as two 8×64 bit-transposes (lo / hi limbs) plus per-byte-group
/// `swap_bytes`. The scalar extractor walked 16 × 8 × 8 isolated bits;
/// `transpose_8_u64s_to_64_bytes` is the already-proven ISA kernel for
/// that exact 8-lane → 8-byte-group map, and the GFNI affine qword stores
/// row `i` at byte `7 − i`, which is `u64::swap_bytes` of the little-endian
/// group. Bit-identical to the bit-extract loop: see
/// `fold_mats_from_basis_matches_sum_table`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[inline]
pub(crate) fn fold_mats_from_basis(eq8: &[F128], mats: &mut [u64]) {
    debug_assert_eq!(eq8.len(), 8);
    debug_assert_eq!(mats.len(), 16);

    #[cfg(all(target_feature = "avx512bw", target_feature = "avx512vbmi"))]
    {
        // SAFETY: len==8 and mats.len()==16 asserted; cfg supplies avx512f/bw/vbmi/gfni.
        unsafe {
            crate::bits::transpose_8_f128_to_16_mats_swapped_gfni(eq8, mats);
        }
        return;
    }
    #[cfg(not(all(target_feature = "avx512bw", target_feature = "avx512vbmi")))]
    {
        let (lo_lanes, hi_lanes) = if lc_mats_aos_enabled() {
            // SAFETY: len==8 asserted; cfg supplies avx512f.
            unsafe { aos8_lohi(eq8) }
        } else {
            (
                std::array::from_fn(|j| eq8[j].lo),
                std::array::from_fn(|j| eq8[j].hi),
            )
        };
        let mut lo_bytes = [0u8; 64];
        let mut hi_bytes = [0u8; 64];
        crate::bits::transpose_8_u64s_to_64_bytes(&lo_lanes, &mut lo_bytes);
        crate::bits::transpose_8_u64s_to_64_bytes(&hi_lanes, &mut hi_bytes);

        // `transpose_8_u64s_to_64_bytes` writes group `c` at `bytes[c*8 .. c*8+8]`
        // with `bytes[c*8 + i] bit j = lane[j] bit (8*c + i)` — exactly the
        // extract-loop `row` for `(byte_k = c, i)`. The affine qword wants that
        // row at byte `7 − i` = `from_le_bytes(group).swap_bytes()`.
        for c in 0..8 {
            let lo: [u8; 8] = lo_bytes[c * 8..c * 8 + 8].try_into().unwrap();
            let hi: [u8; 8] = hi_bytes[c * 8..c * 8 + 8].try_into().unwrap();
            mats[c] = u64::from_le_bytes(lo).swap_bytes();
            mats[c + 8] = u64::from_le_bytes(hi).swap_bytes();
        }
    }
}

/// Register-resident single-block GFNI fold: given the eight stripe rows of one
/// 64-column block already in ZMM registers, folds them through `mats` into the
/// sixteen 64-byte planes at `planes` without a store-to-load round trip.
///
/// # Safety
/// - `planes` must point to at least `1024` writable bytes (and initialized if `!seed_zero`).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[cfg_attr(
    all(
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq"
    ),
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")
)]
#[cfg_attr(
    not(all(
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq"
    )),
    target_feature(enable = "avx512f,gfni")
)]
#[inline]
pub(crate) unsafe fn gfni_fold_rows(
    rows: &[core::arch::x86_64::__m512i; 8],
    mats: &[u64; 128],
    planes: *mut u8,
    seed_zero: bool,
) {
    use core::arch::x86_64::*;
    unsafe {
        for byte_k in 0..16 {
            let plane_ptr = planes.add(byte_k * 64) as *mut __m512i;
            let mut acc = if seed_zero {
                _mm512_setzero_si512()
            } else {
                _mm512_loadu_si512(plane_ptr as *const __m512i)
            };
            for t in (0..8).step_by(2) {
                let g0 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[t],
                    _mm512_set1_epi64(mats[t * 16 + byte_k] as i64),
                );
                let g1 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[t + 1],
                    _mm512_set1_epi64(mats[(t + 1) * 16 + byte_k] as i64),
                );
                acc = _mm512_ternarylogic_epi64::<0x96>(acc, g0, g1);
            }
            _mm512_storeu_si512(plane_ptr, acc);
        }
    }
}


/// Fold one selected-word block across 1..=4 adjacent tiles, retaining all
/// sixteen byte-plane accumulators until the last tile has been consumed.
///
/// Exactly 22 ZMM registers are used: accumulators 0..15, transformed current
/// rows 16..17, products 18..19, permutation 20 and transpose identity 21.
/// This outlined leaf's asm never touches the stack or spills a vector. Inspect
/// the optimized wrapper on the measured target before enabling production.
///
/// # Safety
/// - `count` is in 1..=4 and `tile_stride >= 512`.
/// - For each `t < count`, `source.add(t * tile_stride)` covers 512 readable
///   bytes containing eight raw selected-word stripes (low u32 lanes then high).
///   The entire addressed span belongs to one allocation; offset arithmetic
///   must fit usize/isize. No alignment beyond byte alignment is required.
/// - `mats` covers `count * 128` initialized u64s, ordered
///   `[tile][stripe * 16 + plane]`; no non-live matrix slot is read.
/// - `planes` covers 1024 writable bytes, initialized iff `!seed_zero`, and
///   does not alias source, matrices, or permutation storage.
/// - `permutation` covers the 64 initialized bytes of WORD_TO_COLUMNS, and
///   all readable inputs remain immutable for this call.
/// - `prefetch_next_block` enables nonfaulting hints at source + 512 and
///   planes + 1024. Hint addresses may exceed the demand spans above; no
///   out-of-range demand load is introduced. Production disables hints at
///   the final block so they stay inside the caller's source/plane arenas.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
#[inline(never)]
pub(crate) unsafe fn gfni_fold_word_tiles4(
    source: *const u8,
    mats: *const u64,
    planes: *mut u8,
    tile_stride: usize,
    count: usize,
    seed_zero: bool,
    permutation: *const u8,
    prefetch_next_block: bool,
) {
    debug_assert!((1..=4).contains(&count));
    debug_assert!(tile_stride >= 512);
    // SAFETY: the caller upholds the exact byte ranges above. Every modified
    // GPR and ZMM is explicitly an output; flags and memory are also clobbered
    // (no preserves_flags/nomem/readonly). No callee-saved GPR is touched. The
    // ABI may place the seventh argument on the caller's stack; this asm itself
    // neither reads nor writes the stack and contains no call instruction.
    unsafe {
        core::arch::asm!(
            "vmovdqu64 zmm20, [r10]",
            "mov r10, r11",
            "mov rax, 0x8040201008040201",
            "vpbroadcastq zmm21, rax",
            // Read-only, nonfaulting hints for the next private plane bank.
            "test r10, r10",
            "jz 6f",
            "prefetcht0 [rdx + 1024]",
            "prefetcht0 [rdx + 1088]",
            "prefetcht0 [rdx + 1152]",
            "prefetcht0 [rdx + 1216]",
            "prefetcht0 [rdx + 1280]",
            "prefetcht0 [rdx + 1344]",
            "prefetcht0 [rdx + 1408]",
            "prefetcht0 [rdx + 1472]",
            "prefetcht0 [rdx + 1536]",
            "prefetcht0 [rdx + 1600]",
            "prefetcht0 [rdx + 1664]",
            "prefetcht0 [rdx + 1728]",
            "prefetcht0 [rdx + 1792]",
            "prefetcht0 [rdx + 1856]",
            "prefetcht0 [rdx + 1920]",
            "prefetcht0 [rdx + 1984]",
            "6:",
            "test r9, r9",
            "jz 2f",
            "vpxord zmm0, zmm0, zmm0",
            "vpxord zmm1, zmm1, zmm1",
            "vpxord zmm2, zmm2, zmm2",
            "vpxord zmm3, zmm3, zmm3",
            "vpxord zmm4, zmm4, zmm4",
            "vpxord zmm5, zmm5, zmm5",
            "vpxord zmm6, zmm6, zmm6",
            "vpxord zmm7, zmm7, zmm7",
            "vpxord zmm8, zmm8, zmm8",
            "vpxord zmm9, zmm9, zmm9",
            "vpxord zmm10, zmm10, zmm10",
            "vpxord zmm11, zmm11, zmm11",
            "vpxord zmm12, zmm12, zmm12",
            "vpxord zmm13, zmm13, zmm13",
            "vpxord zmm14, zmm14, zmm14",
            "vpxord zmm15, zmm15, zmm15",
            "jmp 3f",
            "2:",
            "vmovdqu64 zmm0, [rdx + 0]",
            "vmovdqu64 zmm1, [rdx + 64]",
            "vmovdqu64 zmm2, [rdx + 128]",
            "vmovdqu64 zmm3, [rdx + 192]",
            "vmovdqu64 zmm4, [rdx + 256]",
            "vmovdqu64 zmm5, [rdx + 320]",
            "vmovdqu64 zmm6, [rdx + 384]",
            "vmovdqu64 zmm7, [rdx + 448]",
            "vmovdqu64 zmm8, [rdx + 512]",
            "vmovdqu64 zmm9, [rdx + 576]",
            "vmovdqu64 zmm10, [rdx + 640]",
            "vmovdqu64 zmm11, [rdx + 704]",
            "vmovdqu64 zmm12, [rdx + 768]",
            "vmovdqu64 zmm13, [rdx + 832]",
            "vmovdqu64 zmm14, [rdx + 896]",
            "vmovdqu64 zmm15, [rdx + 960]",
            "3:",
            // Each tile's next source block is 512 bytes ahead.
            "test r10, r10",
            "jz 7f",
            "prefetcht0 [rdi + 512]",
            "prefetcht0 [rdi + 576]",
            "prefetcht0 [rdi + 640]",
            "prefetcht0 [rdi + 704]",
            "prefetcht0 [rdi + 768]",
            "prefetcht0 [rdi + 832]",
            "prefetcht0 [rdi + 896]",
            "prefetcht0 [rdi + 960]",
            "7:",
            "vpermb zmm16, zmm20, [rdi]",
            "vpermb zmm17, zmm20, [rdi + 64]",
            // Same operand order as affine(identity, permuted_raw_row).
            "vgf2p8affineqb zmm16, zmm21, zmm16, 0",
            "vgf2p8affineqb zmm17, zmm21, zmm17, 0",
            "vgf2p8affineqb zmm18, zmm16, [rsi]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 128]{{1to8}}, 0",
            "vpternlogq zmm0, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 8]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 136]{{1to8}}, 0",
            "vpternlogq zmm1, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 16]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 144]{{1to8}}, 0",
            "vpternlogq zmm2, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 24]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 152]{{1to8}}, 0",
            "vpternlogq zmm3, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 32]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 160]{{1to8}}, 0",
            "vpternlogq zmm4, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 40]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 168]{{1to8}}, 0",
            "vpternlogq zmm5, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 48]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 176]{{1to8}}, 0",
            "vpternlogq zmm6, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 56]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 184]{{1to8}}, 0",
            "vpternlogq zmm7, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 64]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 192]{{1to8}}, 0",
            "vpternlogq zmm8, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 72]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 200]{{1to8}}, 0",
            "vpternlogq zmm9, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 80]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 208]{{1to8}}, 0",
            "vpternlogq zmm10, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 88]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 216]{{1to8}}, 0",
            "vpternlogq zmm11, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 96]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 224]{{1to8}}, 0",
            "vpternlogq zmm12, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 104]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 232]{{1to8}}, 0",
            "vpternlogq zmm13, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 112]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 240]{{1to8}}, 0",
            "vpternlogq zmm14, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 120]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 248]{{1to8}}, 0",
            "vpternlogq zmm15, zmm18, zmm19, 0x96",
            "vpermb zmm16, zmm20, [rdi + 128]",
            "vpermb zmm17, zmm20, [rdi + 192]",
            // Same operand order as affine(identity, permuted_raw_row).
            "vgf2p8affineqb zmm16, zmm21, zmm16, 0",
            "vgf2p8affineqb zmm17, zmm21, zmm17, 0",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 256]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 384]{{1to8}}, 0",
            "vpternlogq zmm0, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 264]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 392]{{1to8}}, 0",
            "vpternlogq zmm1, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 272]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 400]{{1to8}}, 0",
            "vpternlogq zmm2, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 280]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 408]{{1to8}}, 0",
            "vpternlogq zmm3, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 288]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 416]{{1to8}}, 0",
            "vpternlogq zmm4, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 296]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 424]{{1to8}}, 0",
            "vpternlogq zmm5, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 304]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 432]{{1to8}}, 0",
            "vpternlogq zmm6, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 312]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 440]{{1to8}}, 0",
            "vpternlogq zmm7, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 320]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 448]{{1to8}}, 0",
            "vpternlogq zmm8, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 328]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 456]{{1to8}}, 0",
            "vpternlogq zmm9, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 336]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 464]{{1to8}}, 0",
            "vpternlogq zmm10, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 344]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 472]{{1to8}}, 0",
            "vpternlogq zmm11, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 352]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 480]{{1to8}}, 0",
            "vpternlogq zmm12, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 360]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 488]{{1to8}}, 0",
            "vpternlogq zmm13, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 368]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 496]{{1to8}}, 0",
            "vpternlogq zmm14, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 376]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 504]{{1to8}}, 0",
            "vpternlogq zmm15, zmm18, zmm19, 0x96",
            "vpermb zmm16, zmm20, [rdi + 256]",
            "vpermb zmm17, zmm20, [rdi + 320]",
            // Same operand order as affine(identity, permuted_raw_row).
            "vgf2p8affineqb zmm16, zmm21, zmm16, 0",
            "vgf2p8affineqb zmm17, zmm21, zmm17, 0",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 512]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 640]{{1to8}}, 0",
            "vpternlogq zmm0, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 520]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 648]{{1to8}}, 0",
            "vpternlogq zmm1, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 528]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 656]{{1to8}}, 0",
            "vpternlogq zmm2, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 536]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 664]{{1to8}}, 0",
            "vpternlogq zmm3, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 544]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 672]{{1to8}}, 0",
            "vpternlogq zmm4, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 552]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 680]{{1to8}}, 0",
            "vpternlogq zmm5, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 560]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 688]{{1to8}}, 0",
            "vpternlogq zmm6, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 568]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 696]{{1to8}}, 0",
            "vpternlogq zmm7, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 576]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 704]{{1to8}}, 0",
            "vpternlogq zmm8, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 584]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 712]{{1to8}}, 0",
            "vpternlogq zmm9, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 592]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 720]{{1to8}}, 0",
            "vpternlogq zmm10, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 600]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 728]{{1to8}}, 0",
            "vpternlogq zmm11, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 608]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 736]{{1to8}}, 0",
            "vpternlogq zmm12, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 616]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 744]{{1to8}}, 0",
            "vpternlogq zmm13, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 624]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 752]{{1to8}}, 0",
            "vpternlogq zmm14, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 632]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 760]{{1to8}}, 0",
            "vpternlogq zmm15, zmm18, zmm19, 0x96",
            "vpermb zmm16, zmm20, [rdi + 384]",
            "vpermb zmm17, zmm20, [rdi + 448]",
            // Same operand order as affine(identity, permuted_raw_row).
            "vgf2p8affineqb zmm16, zmm21, zmm16, 0",
            "vgf2p8affineqb zmm17, zmm21, zmm17, 0",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 768]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 896]{{1to8}}, 0",
            "vpternlogq zmm0, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 776]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 904]{{1to8}}, 0",
            "vpternlogq zmm1, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 784]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 912]{{1to8}}, 0",
            "vpternlogq zmm2, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 792]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 920]{{1to8}}, 0",
            "vpternlogq zmm3, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 800]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 928]{{1to8}}, 0",
            "vpternlogq zmm4, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 808]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 936]{{1to8}}, 0",
            "vpternlogq zmm5, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 816]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 944]{{1to8}}, 0",
            "vpternlogq zmm6, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 824]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 952]{{1to8}}, 0",
            "vpternlogq zmm7, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 832]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 960]{{1to8}}, 0",
            "vpternlogq zmm8, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 840]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 968]{{1to8}}, 0",
            "vpternlogq zmm9, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 848]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 976]{{1to8}}, 0",
            "vpternlogq zmm10, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 856]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 984]{{1to8}}, 0",
            "vpternlogq zmm11, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 864]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 992]{{1to8}}, 0",
            "vpternlogq zmm12, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 872]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 1000]{{1to8}}, 0",
            "vpternlogq zmm13, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 880]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 1008]{{1to8}}, 0",
            "vpternlogq zmm14, zmm18, zmm19, 0x96",
            "vgf2p8affineqb zmm18, zmm16, [rsi + 888]{{1to8}}, 0",
            "vgf2p8affineqb zmm19, zmm17, [rsi + 1016]{{1to8}}, 0",
            "vpternlogq zmm15, zmm18, zmm19, 0x96",
            "dec r8",
            "jz 5f",
            // Only advance when another live tile remains. The four stripe
            // pairs use fixed offsets from the unchanged tile/matrix bases.
            "add rdi, rcx",
            "add rsi, 1024",
            "jmp 3b",
            "5:",
            "vmovdqu64 [rdx + 0], zmm0",
            "vmovdqu64 [rdx + 64], zmm1",
            "vmovdqu64 [rdx + 128], zmm2",
            "vmovdqu64 [rdx + 192], zmm3",
            "vmovdqu64 [rdx + 256], zmm4",
            "vmovdqu64 [rdx + 320], zmm5",
            "vmovdqu64 [rdx + 384], zmm6",
            "vmovdqu64 [rdx + 448], zmm7",
            "vmovdqu64 [rdx + 512], zmm8",
            "vmovdqu64 [rdx + 576], zmm9",
            "vmovdqu64 [rdx + 640], zmm10",
            "vmovdqu64 [rdx + 704], zmm11",
            "vmovdqu64 [rdx + 768], zmm12",
            "vmovdqu64 [rdx + 832], zmm13",
            "vmovdqu64 [rdx + 896], zmm14",
            "vmovdqu64 [rdx + 960], zmm15",
            inout("rdi") source => _,
            inout("rsi") mats => _,
            in("rdx") planes,
            in("rcx") tile_stride,
            inout("r8") count => _,
            in("r9") usize::from(seed_zero),
            inout("r10") permutation => _,
            inout("r11") usize::from(prefetch_next_block) => _,
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
            options(nostack),
        );
    }
}

/// One tile's GFNI sweep: for every 64-column block, sixteen byte-plane
/// accumulators fold the eight stripes' GFNI products (two per `vpternlogq`).
///
/// `seed_zero` replaces the first-tile `_mm512_loadu_si512` of a known-zero
/// plane buffer with `_mm512_setzero_si512`. XOR identity: `0 ⊕ x = x`, so
/// later tiles still `loadu` the running acc. Bit-identical to loadu-of-zeros.
///
/// # Safety
/// - `tile_bytes_ptr` must point to at least `7 * stripe_stride + n_blocks64 * 64` bytes.
/// - `mats` holds the tile's 8×16 matrices.
/// - `out_planes_ptr` must point to at least `n_blocks64 * 1024` bytes.
/// - With `seed_zero = true`, that output range may be uninitialized: the
///   kernel seeds registers from zero and stores every byte of every block.
///   With `seed_zero = false`, the full output range must already be
///   initialized because every 64-byte plane is loaded before being updated.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[cfg_attr(
    all(
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq"
    ),
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")
)]
#[cfg_attr(
    not(all(
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq"
    )),
    target_feature(enable = "avx512f,gfni")
)]
#[inline]
pub(crate) unsafe fn gfni_fold_tile(
    tile_bytes_ptr: *const u8,
    stripe_stride: usize,
    n_blocks64: usize,
    mats: &[u64; 128],
    out_planes_ptr: *mut u8,
    seed_zero: bool,
) {
    use core::arch::x86_64::*;
    // SAFETY: caller upholds the pointer/length contract above.
    unsafe {
        for block in 0..n_blocks64 {
            let bs = block * 64;
            let rows: [__m512i; 8] = core::array::from_fn(|t| {
                _mm512_loadu_si512(tile_bytes_ptr.add(t * stripe_stride + bs) as *const __m512i)
            });
            gfni_fold_rows(&rows, mats, out_planes_ptr.add(block * 1024), seed_zero);
        }
    }
}

/// `dst[i] ^= src[i]` for `len` bytes. `len` must be a multiple of 64.
///
/// Bit-identical to the scalar byte loop: XOR is bitwise and `_mm512_xor_si512`
/// / `VPXORD` is the 512-bit encoding of the same operation (Intel SDM
/// `PXOR`/`VPXORD`). Used for the cross-worker plane merge in
/// `fold_block_major_gfni` (and available to the stripe GFNI `.reduce`).
///
/// # Safety
/// `dst` and `src` must each cover `len` bytes; `len % 64 == 0`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn xor_bytes_avx512(dst: *mut u8, src: *const u8, len: usize) {
    use core::arch::x86_64::*;
    debug_assert_eq!(len % 64, 0);
    // SAFETY: caller guarantees `len` bytes at both pointers and 64-byte steps.
    unsafe {
        let mut i = 0;
        // Two-ZMM unroll matches the Linux `xor_avx512_2` RAID xor_gen kernel
        // (load pair / vpxor pair / store pair). Odd 64-byte tail is one more.
        while i + 128 <= len {
            let a0 = _mm512_loadu_si512(dst.add(i) as *const __m512i);
            let a1 = _mm512_loadu_si512(dst.add(i + 64) as *const __m512i);
            let b0 = _mm512_loadu_si512(src.add(i) as *const __m512i);
            let b1 = _mm512_loadu_si512(src.add(i + 64) as *const __m512i);
            _mm512_storeu_si512(dst.add(i) as *mut __m512i, _mm512_xor_si512(a0, b0));
            _mm512_storeu_si512(dst.add(i + 64) as *mut __m512i, _mm512_xor_si512(a1, b1));
            i += 128;
        }
        if i < len {
            let a = _mm512_loadu_si512(dst.add(i) as *const __m512i);
            let b = _mm512_loadu_si512(src.add(i) as *const __m512i);
            _mm512_storeu_si512(dst.add(i) as *mut __m512i, _mm512_xor_si512(a, b));
        }
    }
}

/// x86 single-matrix inner kernel — SSE2 mirror of
/// [`process_block_neon_single`]. Sweeps `TILE_T = 8` stripes for one
/// `BLOCK_K = 8` block of i_inner positions, keeping all 8 F128 accumulators in
/// xmm registers so the per-tile output is read/written once (vs once per
/// stripe in the untiled [`partial_fold_packed_z_fast_padded`] path).
///
/// # Safety
/// - `tile_bytes_ptr` must point to at least `TILE_T * k` bytes.
/// - `tables_ptr` must point to at least `TILE_T * 256 * 16` bytes.
/// - `out_ptr` must point to at least 8 F128 (128 bytes) of mutable storage.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn process_block_x86(
    tile_bytes_ptr: *const u8,
    k: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    use core::arch::x86_64::*;
    const TILE_T: usize = 8;
    // SAFETY: caller upholds the pointer/length contract documented above; SSE2
    // is baseline on x86_64.
    unsafe {
        let o = out_ptr as *mut u8;
        let mut a0 = _mm_loadu_si128(o as *const __m128i);
        let mut a1 = _mm_loadu_si128(o.add(16) as *const __m128i);
        let mut a2 = _mm_loadu_si128(o.add(32) as *const __m128i);
        let mut a3 = _mm_loadu_si128(o.add(48) as *const __m128i);
        let mut a4 = _mm_loadu_si128(o.add(64) as *const __m128i);
        let mut a5 = _mm_loadu_si128(o.add(80) as *const __m128i);
        let mut a6 = _mm_loadu_si128(o.add(96) as *const __m128i);
        let mut a7 = _mm_loadu_si128(o.add(112) as *const __m128i);
        for t in 0..TILE_T {
            let stripe_ptr = tile_bytes_ptr.add(t * k + bs);
            let ta = tables_ptr.add(t * 256 * 16);
            let i0 = *stripe_ptr as usize;
            let i1 = *stripe_ptr.add(1) as usize;
            let i2 = *stripe_ptr.add(2) as usize;
            let i3 = *stripe_ptr.add(3) as usize;
            let i4 = *stripe_ptr.add(4) as usize;
            let i5 = *stripe_ptr.add(5) as usize;
            let i6 = *stripe_ptr.add(6) as usize;
            let i7 = *stripe_ptr.add(7) as usize;
            a0 = _mm_xor_si128(a0, _mm_loadu_si128(ta.add(i0 * 16) as *const __m128i));
            a1 = _mm_xor_si128(a1, _mm_loadu_si128(ta.add(i1 * 16) as *const __m128i));
            a2 = _mm_xor_si128(a2, _mm_loadu_si128(ta.add(i2 * 16) as *const __m128i));
            a3 = _mm_xor_si128(a3, _mm_loadu_si128(ta.add(i3 * 16) as *const __m128i));
            a4 = _mm_xor_si128(a4, _mm_loadu_si128(ta.add(i4 * 16) as *const __m128i));
            a5 = _mm_xor_si128(a5, _mm_loadu_si128(ta.add(i5 * 16) as *const __m128i));
            a6 = _mm_xor_si128(a6, _mm_loadu_si128(ta.add(i6 * 16) as *const __m128i));
            a7 = _mm_xor_si128(a7, _mm_loadu_si128(ta.add(i7 * 16) as *const __m128i));
        }
        _mm_storeu_si128(o as *mut __m128i, a0);
        _mm_storeu_si128(o.add(16) as *mut __m128i, a1);
        _mm_storeu_si128(o.add(32) as *mut __m128i, a2);
        _mm_storeu_si128(o.add(48) as *mut __m128i, a3);
        _mm_storeu_si128(o.add(64) as *mut __m128i, a4);
        _mm_storeu_si128(o.add(80) as *mut __m128i, a5);
        _mm_storeu_si128(o.add(96) as *mut __m128i, a6);
        _mm_storeu_si128(o.add(112) as *mut __m128i, a7);
    }
}

/// x86 tiled single-matrix partial fold — `TILE_T × BLOCK_K` register tile
/// mirror of `partial_fold_packed_z_neon_single_padded`. Replaces the untiled
/// full-k accumulator (streamed once per stripe ≈ `n_stripes × k` of acc
/// traffic) with per-tile register accumulators, cutting acc traffic ≈ `TILE_T`×
/// — the dominant cost of this bandwidth-bound fold on many-core hosts.
#[cfg(target_arch = "x86_64")]
pub fn partial_fold_packed_z_x86_tiled_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = 8;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 6, "need n_outer ≥ 64 for tile of 8 stripes");
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_blocks_full = k / BLOCK_K;
    // Boundary block past useful_bits holds 0 padding → table[0] = 0 contributes
    // nothing, so cover only blocks that touch useful bits.
    let n_blocks = useful_bits.div_ceil(BLOCK_K).min(n_blocks_full);

    let n_tiles = n_stripes / TILE_T;
    let tiles_per_chunk = (n_tiles / 256).max(1);
    let bytes_per_chunk = tiles_per_chunk * TILE_T * k;

    z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![F128::ZERO; k],
            |mut out, (chunk_idx, chunk_bytes)| {
                let tile_start = chunk_idx * tiles_per_chunk;
                // TILE_T × 256 F128 = 32 KB tables. L1 resident.
                let mut tables = vec![F128::ZERO; TILE_T * 256];
                let n_tiles_in_chunk = chunk_bytes.len() / (TILE_T * k);
                for tile_rel in 0..n_tiles_in_chunk {
                    let tile_idx = tile_start + tile_rel;
                    let stripe_base = tile_idx * TILE_T;
                    // SAFETY: tile_rel < n_tiles_in_chunk so the offset stays in bounds.
                    let tile_bytes_ptr = unsafe { chunk_bytes.as_ptr().add(tile_rel * TILE_T * k) };
                    for t in 0..TILE_T {
                        let byte_idx = stripe_base + t;
                        let eq_off = 8 * byte_idx;
                        build_sum_table(
                            &eq_outer[eq_off..eq_off + 8],
                            &mut tables[t * 256..(t + 1) * 256],
                        );
                    }
                    let tables_ptr = tables.as_ptr() as *const u8;
                    for block_idx in 0..n_blocks {
                        let bs = block_idx * BLOCK_K;
                        // SAFETY: bs + BLOCK_K ≤ k ≤ out.len(); tile_bytes_ptr
                        // covers TILE_T*k bytes; tables_ptr covers TILE_T*256 F128.
                        unsafe {
                            process_block_x86(
                                tile_bytes_ptr,
                                k,
                                bs,
                                tables_ptr,
                                out.as_mut_ptr().add(bs),
                            );
                        }
                    }
                }
                out
            },
        )
        .reduce(
            || vec![F128::ZERO; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

// ---------------------------------------------------------------------------
// Block-major fold: AVX-512 nibble-table accumulate.
//
// The ranked lincheck folds the block-major packed witness (2^18 blocks ×
// 128 F128) over the outer index. Per (tile of 8 stripes, 128-column chunk)
// the driver transposes each stripe's 8 lanes into 128 index bytes (byte b =
// the 8 blocks' bit b) and then, for every column b and stripe t, adds the
// 8-bit subset-sum `T_t[byte]` into `partial[b]`. Today that inner loop is
// scalar: 8 stripes × 128 columns = 1,024 byte-indexed 16-byte gathers per
// (tile, chunk), ~5·10^8 per ranked prove.
//
// This kernel keeps the same arithmetic but performs it eight columns wide:
// the 256-entry table is never built; instead the subset sum splits exactly
// into a low-nibble table `TL_t[l] = Σ_{r∈l} eq_t[r]` (r < 4) and a
// high-nibble table `TH_t[h] = Σ_{r∈h} eq_t[4+r]`, and `T_t[byte] =
// TL_t[byte & 15] + TH_t[byte >> 4]` (GF(2^128) addition is XOR, so the
// split is exact). Each 16-entry table lives in registers as two zmm of
// qwords (lo halves) plus two zmm (hi halves); `vpermi2q` looks up eight
// columns' entries at once from the eight index nibbles. Accumulation is SoA
// (lo/hi qword vectors) over the eight stripes and interleaved back to the
// AoS `partial` once per eight columns. Same set of XORs as the scalar loop,
// in a different association — exact in a characteristic-2 field.
// ---------------------------------------------------------------------------

/// Nibble sum tables for one stripe (8 outer weights): `[TL lo(16), TL hi(16),
/// TH lo(16), TH hi(16)]` as qwords, i.e. `TL_t[l]` = `(lo[l], hi[l])`.
#[allow(dead_code)] // Kept as the scalar/table oracle for the active GFNI path.
pub(crate) type NibbleTables = [u64; 64];

/// Build the lo/hi-nibble subset-sum tables from eight outer weights.
#[inline]
#[allow(dead_code)] // Kept as the scalar/table oracle for the active GFNI path.
pub(crate) fn build_nibble_tables(eq8: &[F128; 8], out: &mut NibbleTables) {
    let mut tl = [F128::ZERO; 16];
    let mut th = [F128::ZERO; 16];
    for i in 0..4 {
        let (el, eh) = (eq8[i], eq8[4 + i]);
        let len = 1usize << i;
        for j in 0..len {
            tl[len + j] = tl[j] + el;
            th[len + j] = th[j] + eh;
        }
    }
    for i in 0..16 {
        out[i] = tl[i].lo;
        out[16 + i] = tl[i].hi;
        out[32 + i] = th[i].lo;
        out[48 + i] = th[i].hi;
    }
}

/// AVX-512 accumulate for one (full 8-stripe tile, 128-column chunk):
/// `partial[b] += Σ_t T_t[transposed[t*128 + b]]` for `b < chunk_bits`.
///
/// `transposed`: 8 rows × 128 index bytes; `nib`: the 8 stripes' nibble
/// tables; `partial`: at least `chunk_bits` F128 (AoS). Columns
/// `chunk_bits..128` of `transposed` are ignored (masked out at the store).
///
/// # Safety
/// Requires AVX-512F/BW at runtime (guaranteed by the cfg gate that compiles
/// this function in). All loads/stores are bounds-checked by the asserts.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn fold_block_major_chunk_x86_avx512(
    transposed: &[u8],
    nib: &[NibbleTables],
    partial: &mut [F128],
    chunk_bits: usize,
) {
    use core::arch::x86_64::*;
    const STRIPES: usize = 8;
    debug_assert_eq!(transposed.len(), STRIPES * 128);
    debug_assert_eq!(nib.len(), STRIPES);
    debug_assert!(chunk_bits <= 128 && partial.len() >= chunk_bits);
    let n_groups = chunk_bits.div_ceil(8);
    // SoA accumulators for the 128 columns: [lo qwords; 16 groups][hi; 16].
    let mut acc = [_mm512_setzero_si512(); 32];
    unsafe {
        let nib_mask = _mm512_set1_epi64(0xF);
        for t in 0..STRIPES {
            let tp = nib.as_ptr().add(t) as *const u64;
            let tl_lo0 = _mm512_loadu_si512(tp as *const __m512i);
            let tl_lo1 = _mm512_loadu_si512(tp.add(8) as *const __m512i);
            let tl_hi0 = _mm512_loadu_si512(tp.add(16) as *const __m512i);
            let tl_hi1 = _mm512_loadu_si512(tp.add(24) as *const __m512i);
            let th_lo0 = _mm512_loadu_si512(tp.add(32) as *const __m512i);
            let th_lo1 = _mm512_loadu_si512(tp.add(40) as *const __m512i);
            let th_hi0 = _mm512_loadu_si512(tp.add(48) as *const __m512i);
            let th_hi1 = _mm512_loadu_si512(tp.add(56) as *const __m512i);
            let row = transposed.as_ptr().add(t * 128);
            for g in 0..n_groups {
                let idx8 = _mm_loadl_epi64(row.add(g * 8) as *const __m128i);
                let idx = _mm512_cvtepu8_epi64(idx8);
                let n0 = _mm512_and_si512(idx, nib_mask);
                let n1 = _mm512_srli_epi64::<4>(idx);
                let lo = _mm512_xor_si512(
                    _mm512_permutex2var_epi64(tl_lo0, n0, tl_lo1),
                    _mm512_permutex2var_epi64(th_lo0, n1, th_lo1),
                );
                let hi = _mm512_xor_si512(
                    _mm512_permutex2var_epi64(tl_hi0, n0, tl_hi1),
                    _mm512_permutex2var_epi64(th_hi0, n1, th_hi1),
                );
                acc[g] = _mm512_xor_si512(acc[g], lo);
                acc[16 + g] = _mm512_xor_si512(acc[16 + g], hi);
            }
        }
        // Interleave SoA → AoS and XOR into `partial` (F128 = lo || hi LE).
        let idx0 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
        let idx1 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
        let base = partial.as_mut_ptr() as *mut u64;
        for g in 0..n_groups {
            let aos0 = _mm512_permutex2var_epi64(acc[g], idx0, acc[16 + g]);
            let aos1 = _mm512_permutex2var_epi64(acc[g], idx1, acc[16 + g]);
            let cols = (chunk_bits - g * 8).min(8);
            let p = base.add(g * 16);
            if cols == 8 {
                let p0 = p as *mut __m512i;
                let p1 = p.add(8) as *mut __m512i;
                _mm512_storeu_si512(p0, _mm512_xor_si512(_mm512_loadu_si512(p0), aos0));
                _mm512_storeu_si512(p1, _mm512_xor_si512(_mm512_loadu_si512(p1), aos1));
            } else {
                // Tail group: 2 qwords per column; aos0 covers columns 0..4,
                // aos1 columns 4..8 of this group.
                let q = 2 * cols; // qwords to touch
                let m0: __mmask8 = if q >= 8 {
                    0xFF
                } else {
                    ((1u16 << q) - 1) as u8
                };
                let m1: __mmask8 = if q <= 8 {
                    0
                } else {
                    ((1u16 << (q - 8)) - 1) as u8
                };
                let pi = p as *mut i64;
                let v0 = _mm512_maskz_loadu_epi64(m0, pi);
                _mm512_mask_storeu_epi64(pi, m0, _mm512_xor_si512(v0, aos0));
                if m1 != 0 {
                    let v1 = _mm512_maskz_loadu_epi64(m1, pi.add(8));
                    _mm512_mask_storeu_epi64(pi.add(8), m1, _mm512_xor_si512(v1, aos1));
                }
            }
        }
    }
}
