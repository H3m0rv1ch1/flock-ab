//! Direct BLAKE3 witness generation and challenge-dependent PCS field fold.
use super::*;
#[repr(C, align(64))]
struct FoldWeights {
    lohi: [V16; 64],
    sum: [V16; 64],
}
impl FoldWeights {
    #[cfg(test)]
    fn new(w: &[flock_core::field::F128; 64]) -> Box<Self> {
        unsafe {
            Box::new(Self {
                lohi: core::array::from_fn(|i| {
                    _mm512_broadcast_i32x4(_mm_set_epi64x(w[i].hi as i64, w[i].lo as i64))
                }),
                sum: core::array::from_fn(|i| {
                    _mm512_broadcast_i32x4(_mm_set_epi64x(0, (w[i].hi ^ w[i].lo) as i64))
                }),
            })
        }
    }
}
struct Acc {
    le: V16,
    lo: V16,
    he: V16,
    ho: V16,
    me: V16,
    mo: V16,
}
impl Acc {
    #[inline(always)]
    fn zero() -> Self {
        let z = dup16(0);
        Self {
            le: z,
            lo: z,
            he: z,
            ho: z,
            me: z,
            mo: z,
        }
    }
    #[inline(always)]
    unsafe fn add(&mut self, a: V16, b: V16, w: V16, s: V16) {
        unsafe {
            let x = xor16(a, b);
            self.le = xor16(self.le, _mm512_clmulepi64_epi128::<0x00>(a, w));
            self.lo = xor16(self.lo, _mm512_clmulepi64_epi128::<0x01>(a, w));
            self.he = xor16(self.he, _mm512_clmulepi64_epi128::<0x10>(b, w));
            self.ho = xor16(self.ho, _mm512_clmulepi64_epi128::<0x11>(b, w));
            self.me = xor16(self.me, _mm512_clmulepi64_epi128::<0x00>(x, s));
            self.mo = xor16(self.mo, _mm512_clmulepi64_epi128::<0x01>(x, s));
        }
    }
    #[inline(always)]
    unsafe fn save(&self, p: *mut V16) {
        unsafe {
            *p = self.le;
            *p.add(1) = self.lo;
            *p.add(2) = self.he;
            *p.add(3) = self.ho;
            *p.add(4) = self.me;
            *p.add(5) = self.mo;
        }
    }
    #[inline(always)]
    unsafe fn load(p: *const V16) -> Self {
        unsafe {
            Self {
                le: *p,
                lo: *p.add(1),
                he: *p.add(2),
                ho: *p.add(3),
                me: *p.add(4),
                mo: *p.add(5),
            }
        }
    }
    #[inline(always)]
    unsafe fn finish(&self, p: *mut V16) {
        unsafe {
            let me = _mm512_ternarylogic_epi64::<0x96>(self.me, self.le, self.he);
            let mo = _mm512_ternarylogic_epi64::<0x96>(self.mo, self.lo, self.ho);
            *p = flock_core::field::gf2_128::x86_64::ghash_reduce_acc_x4(self.le, me, self.he);
            *p.add(1) =
                flock_core::field::gf2_128::x86_64::ghash_reduce_acc_x4(self.lo, mo, self.ho);
        }
    }
}
struct FoldW16<'a> {
    pending: V16,
    word0: V16,
    word1: V16,
    word2: V16,
    a: Acc,
    b: Acc,
    weights: &'a FoldWeights,
    partial: *mut V16,
}
impl<'a> FoldW16<'a> {
    #[inline(always)]
    fn new(weights: &'a FoldWeights, partial: *mut V16, pending: V16) -> Self {
        let z = dup16(0);
        Self {
            pending,
            word0: z,
            word1: z,
            word2: z,
            a: Acc::zero(),
            b: Acc::zero(),
            weights,
            partial,
        }
    }
    #[inline(always)]
    unsafe fn write_word<const WORD: usize>(&mut self, v: V16) {
        unsafe {
            match WORD % 4 {
                0 => self.word0 = v,
                1 => self.word1 = v,
                2 => self.word2 = v,
                _ => {
                    let ia =
                        _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                    let ib = _mm512_setr_epi32(
                        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
                    );
                    let w = self.weights.lohi[(WORD / 4) % 64];
                    let s = self.weights.sum[(WORD / 4) % 64];
                    self.a.add(
                        _mm512_permutex2var_epi32(self.word0, ia, self.word1),
                        _mm512_permutex2var_epi32(self.word2, ia, v),
                        w,
                        s,
                    );
                    self.b.add(
                        _mm512_permutex2var_epi32(self.word0, ib, self.word1),
                        _mm512_permutex2var_epi32(self.word2, ib, v),
                        w,
                        s,
                    );
                    if WORD == 255 {
                        self.a.save(self.partial);
                        self.b.save(self.partial.add(6));
                        self.a = Acc::zero();
                        self.b = Acc::zero();
                    }
                }
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
    #[inline(always)]
    unsafe fn restore_partial(&mut self) {
        unsafe {
            self.a = Acc::load(self.partial);
            self.b = Acc::load(self.partial.add(6));
        }
    }
    #[inline(always)]
    unsafe fn finish_half(&self, half: usize, out: *mut V16) {
        unsafe {
            self.a.finish(out.add(half * 4));
            self.b.finish(out.add(half * 4 + 2));
        }
    }
}
unsafe fn recompute_folded16(prepared: &Prepared16, weights: &FoldWeights, out: *mut V16) {
    unsafe {
        let mut partial = core::mem::MaybeUninit::<[V16; 12]>::uninit();
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

        let mut wa = FoldW16::new(
            weights,
            partial.as_mut_ptr().cast(),
            initial16(shr16::<31>(flags)),
        );
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
        wa.write_word::<482>(dup16(0));
        wa.write_word::<483>(dup16(0));
        wa.finish_half(1, out);
        wa.restore_partial();
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
        wa.finish_half(0, out);
    }
}

#[cfg(test)]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
unsafe fn baseline_fold(st: &Stage, w: &FoldWeights) -> [[__m512i; 2]; 2] {
    unsafe {
        let mut out = [[_mm512_setzero_si512(); 2]; 2];
        let st = st.0.as_ptr();
        let wl = w.lohi.as_ptr();
        let ws = w.sum.as_ptr();
        for (h, slot) in out.iter_mut().enumerate() {
            // Element pairs per half: 64 and 57 (qword 241 is zero).
            let ne = if h == 0 { 64 } else { (242 - 128) / 2 };
            let base = st.add(128 * h);
            let z = _mm512_setzero_si512();
            let (mut le, mut lo, mut he, mut ho, mut me, mut mo) = (z, z, z, z, z, z);
            let mut e = 0;
            while e + 2 <= ne {
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

unsafe fn generate_folded(init: u64, tile: usize, pair: usize, w: &FoldWeights, out: *mut V16) {
    unsafe {
        let physical = tile * 64 + pair * 16;
        let base = physical / 256 + ((physical % 256) / 8) * 2048;
        let input = prepare16(init, base);
        recompute_folded16(&input, w, out);
    }
}
#[cfg(test)]
#[repr(C, align(64))]
struct Stage([V16; 242]);
#[cfg(test)]
#[repr(C, align(64))]
struct Out([V16; 8]);
#[test]
fn fused_fold_matches_scalar() {
    use flock_core::field::F128;
    let w: [F128; 64] = core::array::from_fn(|i| {
        let x = (i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
        F128::new(x, x.rotate_left(23) ^ 0x12345678)
    });
    let weights = FoldWeights::new(&w);
    #[repr(align(64))]
    struct GuardOut([V16; 10]);
    for init in [0, 0xdeadbeef, u64::MAX] {
        for (tile, pair) in [(0, 0), (0, 3), (1777, 1), (4095, 3)] {
            let sentinel = dup16(0x5a5adead);
            let mut out = Box::new(GuardOut([sentinel; 10]));
            unsafe {
                super::stage_recompute::generate_pair_provider(
                    init,
                    tile,
                    pair,
                    out.0.as_mut_ptr().add(1).cast(),
                    core::ptr::null_mut(),
                    (&*weights as *const FoldWeights).cast(),
                );
                assert_eq!(
                    _mm512_cmpeq_epi64_mask(out.0[0], sentinel),
                    0xff,
                    "prefix guard"
                );
                assert_eq!(
                    _mm512_cmpeq_epi64_mask(out.0[9], sentinel),
                    0xff,
                    "suffix guard"
                );
            }
            let physical = tile * 64 + pair * 16;
            let base = physical / 256 + ((physical % 256) / 8) * 2048;
            let blocks: Vec<_> = (0..16)
                .map(|i| {
                    crate::seed_pipe::gen_block(
                        init,
                        base + (i / 8) * 2048 + compression_lane_offset(i % 8, true),
                    )
                })
                .collect();
            let (z, _, _) = super::super::super::generate_witness_with_ab_packed(&blocks, 4);
            for half in 0..2 {
                for octa in 0..2 {
                    for parity in 0..2 {
                        let mut lanes = [F128::ZERO; 4];
                        unsafe {
                            _mm512_storeu_si512(
                                lanes.as_mut_ptr().cast(),
                                out.0[1 + half * 4 + octa * 2 + parity],
                            );
                        }
                        for l in 0..4 {
                            let c = octa * 8 + 2 * l + parity;
                            let expected = (0..64)
                                .map(|e| z[c * 128 + half * 64 + e] * w[e])
                                .fold(F128::ZERO, |a, b| a + b);
                            assert_eq!(
                                lanes[l], expected,
                                "seed={init}tile={tile}pair={pair}half={half}c={c}"
                            );
                        }
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "native full-batch fused fold timing"]
fn fused_fold_timing() {
    use flock_core::field::F128;
    use rayon::prelude::*;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .unwrap();
    let w: [F128; 64] = core::array::from_fn(|i| {
        let x = (i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
        F128::new(x, x.rotate_left(23) ^ 0x12345678)
    });
    let weights = FoldWeights::new(&w);
    thread_local! {static BUFFERS:std::cell::RefCell<(Box<[Stage;2]>,Box<Out>)>=std::cell::RefCell::new((Box::new([Stage([dup16(0);242]),Stage([dup16(0);242])]),Box::new(Out([dup16(0);8]))));}
    let mut samples = [Vec::new(), Vec::new()];
    for pass in 0..12 {
        for arm in 0..2 {
            let fused = if pass % 2 == 0 { arm == 1 } else { arm == 0 };
            let start = std::time::Instant::now();
            let sink: u64 = pool.install(|| {
                (0..512usize)
                    .into_par_iter()
                    .map(|u| {
                        BUFFERS.with(|cell| {
                            let mut guard = cell.borrow_mut();
                            let (stages, out) = &mut *guard;
                            let mut sink = 0u64;
                            for i in u * 32..(u + 1) * 32 {
                                let tile = i / 4;
                                let pair = i % 4;
                                unsafe {
                                    if fused {
                                        generate_folded(
                                            0xdeadbeef,
                                            tile,
                                            pair,
                                            &weights,
                                            out.0.as_mut_ptr(),
                                        );
                                    } else {
                                        super::stage_recompute::generate_stages(
                                            0xdeadbeef,
                                            tile,
                                            pair,
                                            stages[0].0.as_mut_ptr().cast(),
                                            stages[1].0.as_mut_ptr().cast(),
                                        );
                                        for octa in 0..2 {
                                            let result = baseline_fold(&stages[octa], &weights);
                                            for half in 0..2 {
                                                for p in 0..2 {
                                                    out.0[half * 4 + octa * 2 + p] =
                                                        result[half][p];
                                                }
                                            }
                                        }
                                    }
                                    let words = core::slice::from_raw_parts(
                                        out.0.as_ptr().cast::<u64>(),
                                        64,
                                    );
                                    sink ^= words.iter().copied().fold(0, |a, b| a ^ b);
                                }
                            }
                            sink
                        })
                    })
                    .reduce(|| 0, |a, b| a ^ b)
            });
            let ms = start.elapsed().as_secs_f64() * 1000.;
            println!("FUSED pass={pass} fused={fused} ms={ms:.3} sink={sink}");
            if pass > 1 {
                samples[usize::from(fused)].push(ms);
            }
        }
    }
    for (i, s) in samples.iter_mut().enumerate() {
        s.sort_by(f64::total_cmp);
        println!("FUSED_MEDIAN fused={i} ms={:.3}", (s[4] + s[5]) * 0.5);
    }
}

/// Fold the two physical octas belonging to one exact ranked witness pair.
///
/// # Safety
/// `weights` is an aligned, initialized 8192-byte immutable span: 64 ZMMs
/// containing repeated F128 weights followed by 64 ZMMs containing their
/// low/high XOR in the low qword of every 128-bit lane. `out` is a disjoint,
/// writable, 64-byte aligned span of eight ZMMs. All eight are overwritten in
/// [half][octa][parity] order; each vector holds four canonical F128 values.
/// Lane l represents compression 2*l+parity of that physical octa.
/// The seed and physical tile mapping must match the current closed witness.
pub(crate) unsafe fn generate_pair_fold(
    init: u64,
    tile: usize,
    pair: usize,
    weights: *const u8,
    out: *mut u8,
) {
    unsafe {
        const {
            assert!(core::mem::size_of::<FoldWeights>() == 8192);
            assert!(core::mem::align_of::<FoldWeights>() == 64);
        }
        assert!(tile < 4096 && pair < 4);
        generate_folded(
            init,
            tile,
            pair,
            &*weights.cast::<FoldWeights>(),
            out.cast(),
        );
    }
}
