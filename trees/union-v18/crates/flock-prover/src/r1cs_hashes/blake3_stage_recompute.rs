//! Direct16-lane full witness qword stages, local to PCS consumers.
use super::*;
struct StageW16 {
    pending: V16,
    previous: V16,
    first: *mut u8,
    second: *mut u8,
}
impl StageW16 {
    #[inline(always)]
    fn new(first: *mut u8, second: *mut u8, pending: V16) -> Self {
        Self {
            pending,
            previous: dup16(0),
            first,
            second,
        }
    }
    #[inline(always)]
    unsafe fn publish(&self, q: usize, lo: V16, hi: V16) {
        unsafe {
            let a = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            let b = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            _mm512_store_si512(
                self.first.add(q * 64).cast(),
                _mm512_permutex2var_epi32(lo, a, hi),
            );
            _mm512_store_si512(
                self.second.add(q * 64).cast(),
                _mm512_permutex2var_epi32(lo, b, hi),
            );
        }
    }
    #[inline(always)]
    unsafe fn write_word<const WORD: usize>(&mut self, v: V16) {
        unsafe {
            if WORD % 2 == 0 {
                self.previous = v;
            } else {
                self.publish(WORD / 2, self.previous, v);
            }
        }
    }
    #[inline(always)]
    unsafe fn push<const USED: i32, const WIDTH: i32, const BACK: i32, const WORD: usize>(
        &mut self,
        v: V16,
    ) {
        unsafe {
            if USED == 0 {
                if WIDTH == 32 {
                    self.write_word::<WORD>(v);
                    self.pending = dup16(0);
                } else {
                    self.pending = shl16::<1>(v);
                }
            } else {
                self.write_word::<WORD>(_mm512_shldi_epi32::<USED>(v, self.pending));
                self.pending = if WIDTH == 31 { shl16::<1>(v) } else { v };
            }
        }
    }
}
unsafe fn recompute_stages16(prepared: &Prepared16, out: *mut u8, out_second: *mut u8) {
    unsafe {
        let cv_v = &prepared.cv;
        let m = &prepared.message;
        let tlo = prepared.counter_lo;
        let thi = prepared.counter_hi;
        let blen = prepared.block_len;
        let flags = prepared.flags;

        let mut state: [V16; 16] = [
            cv_v[0],
            cv_v[1],
            cv_v[2],
            cv_v[3],
            cv_v[4],
            cv_v[5],
            cv_v[6],
            cv_v[7],
            dup16(BLAKE3_IV[0]),
            dup16(BLAKE3_IV[1]),
            dup16(BLAKE3_IV[2]),
            dup16(BLAKE3_IV[3]),
            tlo,
            thi,
            blen,
            flags,
        ];

        let mut wa = StageW16::new(out, out_second, initial16(shr16::<31>(flags)));
        wa.write_word::<0>(cv_v[0]);
        wa.write_word::<1>(cv_v[1]);
        wa.write_word::<2>(cv_v[2]);
        wa.write_word::<3>(cv_v[3]);
        wa.write_word::<4>(cv_v[4]);
        wa.write_word::<5>(cv_v[5]);
        wa.write_word::<6>(cv_v[6]);
        wa.write_word::<7>(cv_v[7]);
        let one = dup16(1);
        let chain: [V16; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15], tlo, thi, blen, flags,
        ];
        wa.write_word::<16>(or16(one, shl16::<1>(chain[0])));
        wa.write_word::<17>(or16(shr16::<31>(chain[0]), shl16::<1>(chain[1])));
        wa.write_word::<18>(or16(shr16::<31>(chain[1]), shl16::<1>(chain[2])));
        wa.write_word::<19>(or16(shr16::<31>(chain[2]), shl16::<1>(chain[3])));
        wa.write_word::<20>(or16(shr16::<31>(chain[3]), shl16::<1>(chain[4])));
        wa.write_word::<21>(or16(shr16::<31>(chain[4]), shl16::<1>(chain[5])));
        wa.write_word::<22>(or16(shr16::<31>(chain[5]), shl16::<1>(chain[6])));
        wa.write_word::<23>(or16(shr16::<31>(chain[6]), shl16::<1>(chain[7])));
        wa.write_word::<24>(or16(shr16::<31>(chain[7]), shl16::<1>(chain[8])));
        wa.write_word::<25>(or16(shr16::<31>(chain[8]), shl16::<1>(chain[9])));
        wa.write_word::<26>(or16(shr16::<31>(chain[9]), shl16::<1>(chain[10])));
        wa.write_word::<27>(or16(shr16::<31>(chain[10]), shl16::<1>(chain[11])));
        wa.write_word::<28>(or16(shr16::<31>(chain[11]), shl16::<1>(chain[12])));
        wa.write_word::<29>(or16(shr16::<31>(chain[12]), shl16::<1>(chain[13])));
        wa.write_word::<30>(or16(shr16::<31>(chain[13]), shl16::<1>(chain[14])));
        wa.write_word::<31>(or16(shr16::<31>(chain[14]), shl16::<1>(chain[15])));
        wa.write_word::<32>(or16(shr16::<31>(chain[15]), shl16::<1>(chain[16])));
        wa.write_word::<33>(or16(shr16::<31>(chain[16]), shl16::<1>(chain[17])));
        wa.write_word::<34>(or16(shr16::<31>(chain[17]), shl16::<1>(chain[18])));
        wa.write_word::<35>(or16(shr16::<31>(chain[18]), shl16::<1>(chain[19])));

        macro_rules! g {
            ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
             $mx:literal, $my:literal) => {{
                let (t0, p0) = add_root16(state[$la], state[$lb]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, p0);
                let (a1, p1) = add_root16(t0, m[$mx]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, p1);
                let d1 = xor_rotr16::<16, 16>(state[$ld], a1);
                let (c1s, p2) = add_root16(state[$lc], d1);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, p2);
                let b1 = xor_rotr16::<12, 20>(state[$lb], c1s);
                let (t1, p3) = add_root16(a1, b1);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, p3);
                let (a2, p4) = add_root16(t1, m[$my]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, p4);
                let d2 = xor_rotr16::<8, 24>(d1, a2);
                let (c2s, p5) = add_root16(c1s, d2);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, p5);
                let bn = xor_rotr16::<7, 25>(b1, c2s);

                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
                state[$la] = a2;
                state[$lb] = bn;
                state[$lc] = c2s;
                state[$ld] = d2;
            }};
        }
        macro_rules! round {
            ($gb:literal, $m0:literal, $m1:literal, $m2:literal, $m3:literal,
             $m4:literal, $m5:literal, $m6:literal, $m7:literal,
             $m8:literal, $m9:literal, $m10:literal, $m11:literal,
             $m12:literal, $m13:literal, $m14:literal, $m15:literal) => {{
                g!($gb, 0, 4, 8, 12, $m0, $m1);
                g!($gb + 1, 1, 5, 9, 13, $m2, $m3);
                g!($gb + 2, 2, 6, 10, 14, $m4, $m5);
                g!($gb + 3, 3, 7, 11, 15, $m6, $m7);
                g!($gb + 4, 0, 5, 10, 15, $m8, $m9);
                g!($gb + 5, 1, 6, 11, 12, $m10, $m11);
                g!($gb + 6, 2, 7, 8, 13, $m12, $m13);
                g!($gb + 7, 3, 4, 9, 14, $m14, $m15);
            }};
        }
        round!(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        round!(8, 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
        round!(16, 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
        round!(24, 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
        round!(32, 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
        round!(40, 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
        round!(48, 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

        macro_rules! oh {
            ($w:literal) => {{
                pushf8!(
                    wa,
                    OUT_HI_BASE + 32 * $w,
                    32,
                    xor16(state[$w + 8], cv_v[$w])
                );
            }};
        }
        oh!(0);
        oh!(1);
        oh!(2);
        oh!(3);
        oh!(4);
        oh!(5);
        oh!(6);
        oh!(7);
        wa.write_word::<481>(shr16::<15>(wa.pending));
        for q in 241..242 {
            wa.publish(q, dup16(0), dup16(0));
        }
        macro_rules! ol {
            ($w:literal) => {{
                wa.write_word::<{ 8 + $w }>(xor16(state[$w], state[$w + 8]));
            }};
        }
        ol!(0);
        ol!(1);
        ol!(2);
        ol!(3);
        ol!(4);
        ol!(5);
        ol!(6);
        ol!(7);
    }
}

/// Regenerate the two physical octas of a ranked witness pair directly into
/// canonical qword stages, bypassing the root-cache selection and expansion.
///
/// # Safety
/// Each destination is a disjoint, writable, 64-byte aligned array of 242
/// ZMMs. The caller validates the ranked geometry and closed input stream.
/// Every qword, including zero padding, is overwritten before publication.
pub(crate) unsafe fn generate_stages(
    init: u64,
    tile: usize,
    pair: usize,
    first: *mut u8,
    second: *mut u8,
) {
    unsafe {
        assert!(tile < 4096 && pair < 4);
        let physical = tile * 64 + pair * 16;
        let base = physical / 256 + ((physical % 256) / 8) * 2048;
        let input = prepare16(init, base);
        recompute_stages16(&input, first, second);
    }
}
#[test]
fn full_stage_matches_scalar_packed_witness() {
    #[repr(align(64))]
    struct Stages([[u64; 8]; 244]);
    const SENTINEL: u64 = 0xdead_beef_0123_4567;
    for init in [0, 0xdeadbeef, u64::MAX] {
        for (tile, pair) in [(0, 0), (0, 3), (1777, 1), (4095, 3)] {
            let mut stages: [Box<Stages>; 2] =
                core::array::from_fn(|_| Box::new(Stages([[SENTINEL; 8]; 244])));
            let [first, second] = &mut stages;
            unsafe {
                generate_pair_provider(
                    init,
                    tile,
                    pair,
                    first.0.as_mut_ptr().add(1).cast(),
                    second.0.as_mut_ptr().add(1).cast(),
                    core::ptr::null(),
                );
            }
            for stage in &stages {
                assert_eq!(stage.0[0], [SENTINEL; 8], "prefix guard");
                assert_eq!(stage.0[243], [SENTINEL; 8], "suffix guard");
            }
            let physical = tile * 64 + pair * 16;
            let base = physical / 256 + ((physical % 256) / 8) * 2048;
            let mut blocks = Vec::new();
            for octa in 0..2 {
                for lane in 0..8 {
                    blocks.push(crate::seed_pipe::gen_block(
                        init,
                        base + octa * 2048 + compression_lane_offset(lane, true),
                    ));
                }
            }
            let (z, _, _) = super::super::super::generate_witness_with_ab_packed(&blocks, 4);
            let bytes =
                unsafe { core::slice::from_raw_parts(z.as_ptr().cast::<u8>(), z.len() * 16) };
            for octa in 0..2 {
                for q in 0..242 {
                    for lane in 0..8 {
                        let off = (octa * 8 + lane) * 2048 + q * 8;
                        let expected = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                        assert_eq!(
                            stages[octa].0[q + 1][lane],
                            expected,
                            "tile={tile}pair={pair}octa={octa}q={q}lane={lane}"
                        );
                    }
                }
            }
        }
    }
}

/// Preserve the root-plus-dependency expansion for same-binary comparisons.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_DIRECT_Z_STAGE").is_none());
    *ON
}

/// Exact ranked witness provider; a null weight pointer requests full stages.
///
/// # Safety
/// In stage mode both outputs are aligned arrays of242ZMMs. In fold mode,
/// weights is an aligned8192-byte immutable weight frame, first is an aligned
/// writable512-byte fold result, and second is unused. All spans are disjoint.
/// See RootTileCache::set_stage_generator for the complete layout contract.
pub(crate) unsafe fn generate_pair_provider(init:u64,tile:usize,pair:usize,first:*mut u8,second:*mut u8,weights:*const u8){unsafe{
    if weights.is_null(){generate_stages(init,tile,pair,first,second);}
    else{super::fused_fold::generate_pair_fold(init,tile,pair,weights,first);}
}}
