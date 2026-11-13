//! Experimental roots-only BLAKE3 recomputation, using the inherited exact
//! round schedule and packed wire encoding. Each callback owns a full tile.
use super::*;

struct RootDrain8<'a> {
    roots: root_emit::RootOctaWriter<'a>,
    stage: *mut V8,
}
impl RootDrain8<'_> {
    unsafe fn drain_range_gen<const R: bool>(&mut self, base: usize, ring: usize, words: usize) {
        unsafe {
            self.roots.emit_z_range(base, ring, words, self.stage);
        }
    }
}
struct RootW8<'t, const FLUSH: bool, const RANKED_CLOSED: bool = false> {
    pending: V8,
    stage: *mut V8,
    drain: *mut RootDrain8<'t>,
}

impl<'t, const FLUSH: bool, const RANKED_CLOSED: bool> RootW8<'t, FLUSH, RANKED_CLOSED> {
    #[inline(always)]
    fn at(stage: *mut V8, pending: V8, drain: *mut RootDrain8<'t>) -> Self {
        Self {
            pending,
            stage,
            drain,
        }
    }

    #[inline(always)]
    unsafe fn write_word<const WORD: usize>(&mut self, v: V8) {
        unsafe {
            store_v8(self.stage.add(WORD & (RING_WORDS - 1)) as *mut u32, v);
            if FLUSH && WORD % RING_WORDS == RING_WORDS - 1 {
                // Words 0..15 cannot be published until the final chaining
                // value is known.  The first rolling epoch therefore starts
                // at word 16; later epochs cover their complete 128 words.
                if WORD + 1 == RING_WORDS {
                    (*self.drain).drain_range_gen::<RANKED_CLOSED>(16, 16, RING_WORDS - 16);
                } else {
                    (*self.drain).drain_range_gen::<RANKED_CLOSED>(
                        WORD + 1 - RING_WORDS,
                        0,
                        RING_WORDS,
                    );
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn push<const USED: i32, const WIDTH: i32, const BACK: i32, const WORD: usize>(
        &mut self,
        v: V8,
    ) {
        const {
            assert!(USED >= 0 && USED < 32);
            assert!(WIDTH == 31 || WIDTH == 32);
            assert!(BACK >= 1 && BACK < 32);
            assert!(WORD < U32_PER_BLOCK);
        }
        debug_assert!(USED + WIDTH <= 32 || BACK == 32 - USED);
        unsafe {
            #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
            {
                // Every u>0 push crosses a word boundary because WIDTH is 31
                // or 32.  VPSHLD supplies the old high-aligned pending bits
                // from its second source while shifting the new field left.
                if USED == 0 {
                    if WIDTH == 32 {
                        self.write_word::<WORD>(v);
                        self.pending = dup_u32(0);
                    } else {
                        // 31 pending low bits, moved to bits 1..31.
                        self.pending = shl_v8::<1>(v);
                    }
                } else {
                    let out = _mm256_shldi_epi32::<USED>(v, self.pending);
                    self.write_word::<WORD>(out);
                    // A crossed 31-bit field leaves u-1 low bits; high-aligning
                    // them is exactly v<<1.  A 32-bit field leaves u bits, so
                    // v itself already is the high-aligned representation.
                    self.pending = if WIDTH == 31 { shl_v8::<1>(v) } else { v };
                }
            }
            #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
            {
                if USED == 0 {
                    if WIDTH == 32 {
                        self.write_word::<WORD>(v);
                        self.pending = dup_u32(0);
                    } else {
                        self.pending = v;
                    }
                } else if USED + WIDTH < 32 {
                    self.pending = vsli_v8::<USED>(self.pending, v);
                } else {
                    let out = vsli_v8::<USED>(self.pending, v);
                    self.write_word::<WORD>(out);
                    if USED + WIDTH == 32 {
                        self.pending = dup_u32(0);
                    } else {
                        self.pending = shr_v8::<BACK>(v);
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn finish(&mut self) {
        const {
            assert!(USEFUL_BITS % 32 == 17);
        }
        unsafe {
            // The ranked stream finishes with 17 pending bits.  The VBMI2
            // representation holds those bits in 15..31; the incumbent
            // fallback is already low-aligned.
            #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
            let pending = packer_high_to_low::<15>(self.pending);
            #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
            let pending = self.pending;
            self.write_word::<LAST_WORD>(pending);
        }
    }
}

#[inline(never)]
unsafe fn recompute_octa(prepared: &PreparedInputs, roots: root_emit::RootOctaWriter<'_>) {
    unsafe {
        let cv_v = &prepared.cv;
        let m = &prepared.message;
        let tlo = prepared.counter_lo;
        let thi = prepared.counter_hi;
        let blen = prepared.block_len;
        let flags = prepared.flags;

        let mut state: [V8; 16] = [
            cv_v[0],
            cv_v[1],
            cv_v[2],
            cv_v[3],
            cv_v[4],
            cv_v[5],
            cv_v[6],
            cv_v[7],
            dup_u32(BLAKE3_IV[0]),
            dup_u32(BLAKE3_IV[1]),
            dup_u32(BLAKE3_IV[2]),
            dup_u32(BLAKE3_IV[3]),
            tlo,
            thi,
            blen,
            flags,
        ];

        let mut ast = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let ast = ast.as_mut_ptr().cast::<V8>();
        let mut drain = RootDrain8 { roots, stage: ast };
        let one = dup_u32(1);
        let chain: [V8; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15], tlo, thi, blen, flags,
        ];
        // Words 16..35 are available before the rounds. Retain them in the
        // rolling epochs; the writer publishes each epoch when it completes
        // that epoch's last word.
        for k in 0..20usize {
            let v = if k == 0 {
                or_v8(one, shl_v8::<1>(chain[0]))
            } else {
                or_v8(shr_v8::<31>(chain[k - 1]), shl_v8::<1>(chain[k]))
            };
            let w = 16 + k;
            store_v8(ast.add(w & (RING_WORDS - 1)) as *mut u32, v);
        }
        // The round stream starts at word `PROLOGUE_WORDS + 16`, so a first
        // epoch that ends inside the prologue never reaches the writers'
        // epoch boundary — publish it here instead. Constant-folded away for
        // every ring long enough to reach the round stream.
        if RING_WORDS <= PROLOGUE_WORDS + 16 {
            drain.drain_range_gen::<true>(16, 16, RING_WORDS - 16);
        }

        let pending_bit = packer_initial_bit(shr_v8::<31>(flags));
        let drain_ptr = &mut drain as *mut RootDrain8;
        let mut wa = RootW8::<true, true>::at(ast, pending_bit, drain_ptr);

        macro_rules! g {
            ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
             $mx:literal, $my:literal) => {{
                let (t0, p0) = add_root8(state[$la], state[$lb]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, p0);
                let (a1, p1) = add_root8(t0, m[$mx]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, p1);
                let d1 = xor_rotr8::<16, 16>(state[$ld], a1);
                let (c1s, p2) = add_root8(state[$lc], d1);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, p2);
                let b1 = xor_rotr8::<12, 20>(state[$lb], c1s);
                let (t1, p3) = add_root8(a1, b1);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, p3);
                let (a2, p4) = add_root8(t1, m[$my]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, p4);
                let d2 = xor_rotr8::<8, 24>(d1, a2);
                let (c2s, p5) = add_root8(c1s, d2);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, p5);
                let bn = xor_rotr8::<7, 25>(b1, c2s);
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

        const {
            assert!(OUT_HI_BASE % 32 == 17);
        }
        macro_rules! oh {
            ($w:literal) => {{
                let hv = xor_v8(state[$w + 8], cv_v[$w]);
                pushf8!(wa, OUT_HI_BASE + 32 * $w, 32, hv);
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
        wa.finish();

        const ZF: usize = USEFUL_BITS.div_ceil(32);
        const {
            assert!(U32_PER_BLOCK - ZF == 30);
        }
        drain.drain_range_gen::<true>(U32_PER_BLOCK - RING_WORDS, 0, RING_WORDS);

        // Band 0 is the one intentional deferral: words 0..7 are the input
        // CV, while words 8..15 depend on the final compression state. Build
        // the complete cache line now, then publish it through the exact same
        // elide/NT/window policy as every rolling band.
        for w in 0..8usize {
            let lo = xor_v8(state[w], state[w + 8]);
            store_v8(ast.add(w) as *mut u32, cv_v[w]);
            store_v8(ast.add(8 + w) as *mut u32, lo);
        }
        drain.drain_range_gen::<true>(0, 0, 16);
    }
}

#[cfg(test)]
#[repr(align(64))]
struct Tile([u8; 176 * 512]);
#[cfg(test)]
#[repr(align(64))]
struct Words([u32; 8 * U32_PER_BLOCK]);
#[cfg(test)]
#[repr(align(64))]
struct Ab([u8; 8 * 29 * 64]);
#[cfg(test)]
#[repr(align(64))]
struct Staging([u32; STREAM_STAGE_WORDS]);

#[test]
fn roots_only_matches_production_stripes() {
    let ntt_s =
        flock_core::ntt::AdditiveNttGf8::new(super::super::K_SKIP, flock_core::field::F8::ZERO);
    let ntt_l = flock_core::ntt::AdditiveNttGf8::new(
        super::super::K_SKIP,
        flock_core::field::F8(1u8 << super::super::K_SKIP),
    );
    let inv = flock_core::ntt::InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    let mut ab = Box::new(Ab([0; 8 * 29 * 64]));
    let plan = flock_core::zerocheck::univariate_skip_optimized::prepare_round1_ab_window_plan(
        &inv, &ab.0, true,
    );
    assert!(plan.offsets_eligible(2));
    let preps = prepare_ranked_closed_window_preps(&inv, plan);
    let fold = super::super::Blake3AdjointPlan::build().linear_fold_plan();
    let root_plan = root_emit::RootEmitPlan::new(&fold.roots).unwrap();
    let mut z = Box::new(Words([0; 8 * U32_PER_BLOCK]));
    let mut a = Box::new(Words([0; 8 * U32_PER_BLOCK]));
    let mut b = Box::new(Words([0; 8 * U32_PER_BLOCK]));
    let mut stage = Box::new(Staging([0; STREAM_STAGE_WORDS]));
    let mut original = Box::new(Tile([0xa5; 176 * 512]));
    let mut rebuilt = Box::new(Tile([0xa5; 176 * 512]));
    for init in [0u64, 1, 0xdead_beef_cafe_babe] {
        for base in [0usize, 7, 1024] {
            for cross in [false, true] {
                unsafe {
                    let prepared = prepare_closed_inputs_ordered(init, base, cross);
                    recompute_octa(
                        &prepared,
                        root_emit::RootOctaWriter::new(&root_plan, rebuilt.0.as_mut_ptr()),
                    );
                    for lane in 0..8 {
                        let reference_base = base + compression_lane_offset(lane, cross);
                        build_octa_witness_ab_stream_ranked_closed(
                            init,
                            reference_base,
                            z.0.as_mut_ptr(),
                            a.0.as_mut_ptr(),
                            b.0.as_mut_ptr(),
                            StreamProj { compact_z_tile_words:0, seed_window:None, ab_pend: core::ptr::null_mut(),
                                stage: stage.0.as_mut_ptr(),
                                out: ab.0.as_mut_ptr(),
                                out_stride: 29 * 64,
                                out_bias: 2 * 64,
                                inv_table: &inv,
                                plan,
                                one_rows_elided: true,
                                w30_elided: true,
                                cross_quarter: false,
                                compact_z: core::ptr::null_mut(),
                                elide_z: false,
                            },
                            &preps,
                            [true; 3],
                            Some(root_emit::RootOctaWriter::new(
                                &root_plan,
                                original.0.as_mut_ptr(),
                            )),
                            false,
                        );
                        _mm_sfence();
                        for group in 0..176 {
                            for half in 0..2 {
                                let expected = &original.0
                                    [group * 512 + half * 32..group * 512 + half * 32 + 4];
                                let at = group * 512 + half * 32 + lane * 4;
                                assert_eq!(
                                    &rebuilt.0[at..at + 4],
                                    expected,
                                    "init={init:x} base={base} cross={cross} group={group} lane={lane} half={half}"
                                );
                            }
                            assert!(
                                rebuilt.0[group * 512 + 64..(group + 1) * 512]
                                    .iter()
                                    .all(|&x| x == 0xa5)
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "native ranked-scale throughput diagnostic"]
fn roots_only_timing() {
    use rayon::prelude::*;
    let fold = super::super::Blake3AdjointPlan::build().linear_fold_plan();
    let root_plan = root_emit::RootEmitPlan::new(&fold.roots).unwrap();
    let mut times = Vec::new();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .unwrap();
    for pass in 0..12 {
        let t = std::time::Instant::now();
        let sink: u64 = pool.install(|| {
            (0..4096usize)
                .into_par_iter()
                .map_init(
                    || Box::new(Tile([0; 176 * 512])),
                    |tile, index| {
                        for octa in 0..8 {
                            let physical = index * 64 + octa * 8;
                            let base = (physical / 256) + ((physical % 256) / 8) * 2048;
                            unsafe {
                                let prepared =
                                    prepare_closed_inputs_ordered(0xdead_beef, base, true);
                                recompute_octa(
                                    &prepared,
                                    root_emit::RootOctaWriter::new(
                                        &root_plan,
                                        tile.0.as_mut_ptr().add(octa * 64),
                                    ),
                                );
                            }
                        }
                        u64::from(tile.0[index % tile.0.len()])
                    },
                )
                .sum()
        });
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("ROOTS_ONLY pass={pass} ms={ms:.3} sink={sink}");
        if pass > 1 {
            times.push(ms);
        }
    }
    times.sort_by(f64::total_cmp);
    println!("ROOTS_ONLY median_ms={:.3}", (times[4] + times[5]) * 0.5);
}

pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ROOT_RECOMPUTE").is_none());
    *ON
}

/// # Safety
/// `out` exclusively owns one complete, 64-byte-aligned 176*512-byte tile.
/// The seed is the exact closed-source initial state for the ranked proof.
pub(crate) unsafe fn generate_tile(init: u64, tile: usize, out: *mut u8) {
    assert!(tile < 4096);
    debug_assert_eq!(out as usize % 64, 0);
    static PLAN: std::sync::LazyLock<root_emit::RootEmitPlan> = std::sync::LazyLock::new(|| {
        let fold = super::super::BLAKE3_ADJOINT_PLAN
            .linear_fold
            .get_or_init(|| super::super::BLAKE3_ADJOINT_PLAN.linear_fold_plan());
        root_emit::RootEmitPlan::new(&fold.roots).expect("ranked root plan")
    });
    unsafe {
        for octa in 0..8 {
            let physical = tile * 64 + octa * 8;
            let base = physical / 256 + ((physical % 256) / 8) * 2048;
            let prepared = prepare_closed_inputs_ordered(init, base, true);
            recompute_octa(
                &prepared,
                root_emit::RootOctaWriter::new(&PLAN, out.add(octa * 64)),
            );
        }
    }
}

type V16 = __m512i;
struct Prepared16 {
    cv: [V16; 8],
    message: [V16; 16],
    counter_lo: V16,
    counter_hi: V16,
    block_len: V16,
    flags: V16,
}
#[inline(always)]
fn dup16(x: u32) -> V16 {
    unsafe { _mm512_set1_epi32(x as i32) }
}
#[inline(always)]
fn xor16(x: V16, y: V16) -> V16 {
    unsafe { _mm512_xor_si512(x, y) }
}
#[inline(always)]
fn and16(x: V16, y: V16) -> V16 {
    unsafe { _mm512_and_si512(x, y) }
}
#[inline(always)]
fn or16(x: V16, y: V16) -> V16 {
    unsafe { _mm512_or_si512(x, y) }
}
#[inline(always)]
fn shl16<const N: u32>(x: V16) -> V16 {
    unsafe { _mm512_slli_epi32::<N>(x) }
}
#[inline(always)]
fn shr16<const N: u32>(x: V16) -> V16 {
    unsafe { _mm512_srli_epi32::<N>(x) }
}
#[inline(always)]
fn initial16(x: V16) -> V16 {
    shl16::<31>(x)
}
#[inline(always)]
fn xor_rotr16<const N: i32, const M: i32>(x: V16, y: V16) -> V16 {
    unsafe { _mm512_ror_epi32::<N>(xor16(x, y)) }
}
#[cfg(any(test,not(target_feature="avx512dq")))]
#[inline(always)]
unsafe fn prepare16_legacy(init: u64, base: usize) -> Prepared16 {
    unsafe {
        let a = prepare_closed_inputs_ordered(init, base, true);
        let b = prepare_closed_inputs_ordered(init, base + 2048, true);
        let join = |x: V8, y: V8| _mm512_inserti64x4::<1>(_mm512_castsi256_si512(x), y);
        Prepared16 {
            cv: core::array::from_fn(|i| join(a.cv[i], b.cv[i])),
            message: core::array::from_fn(|i| join(a.message[i], b.message[i])),
            counter_lo: join(a.counter_lo, b.counter_lo),
            counter_hi: join(a.counter_hi, b.counter_hi),
            block_len: join(a.block_len, b.block_len),
            flags: join(a.flags, b.flags),
        }
    }
}

#[cfg(target_feature="avx512dq")]
#[path="blake3_roots_prepare16.rs"]
mod prepare16_fused;

#[inline(always)]
unsafe fn prepare16(init:u64,base:usize)->Prepared16 {
    #[cfg(target_feature="avx512dq")]
    {unsafe {prepare16_fused::prepare(init,base)}}
    #[cfg(not(target_feature="avx512dq"))]
    {unsafe {prepare16_legacy(init,base)}}
}

#[derive(Clone, Copy)]
enum DirectAction {
    Skip,
    Full(usize),
    Start,
    End(usize, u64),
    Tail(usize, u64),
}
const DIRECT_BITS: usize = 11249;
const fn direct_roots() -> [usize; DIRECT_BITS] {
    let mut out = [0; DIRECT_BITS];
    let mut i = 0;
    while i < 256 {
        out[i] = i;
        i += 1;
    }
    let mut j = 0;
    while j < 577 {
        out[i] = 512 + j;
        i += 1;
        j += 1;
    }
    let mut g = 0;
    while g < 56 {
        let mut bit = 0;
        while bit < 186 {
            out[i] = GS_BASE + g * G_STRIDE + bit;
            i += 1;
            bit += 1;
        }
        g += 1;
    }
    assert!(i == DIRECT_BITS);
    out
}
const DIRECT_ROOTS: [usize; DIRECT_BITS] = direct_roots();
const fn direct_actions() -> [DirectAction; 256] {
    let mut actions = [DirectAction::Skip; 256];
    let mut group = 0;
    while group < 176 {
        let start = group * 64;
        let first = DIRECT_ROOTS[start];
        assert!(first % 64 == 0);
        let source = first / 64;
        let count = if DIRECT_BITS - start < 64 {
            DIRECT_BITS - start
        } else {
            64
        };
        if count < 64 {
            actions[source] = DirectAction::Tail(group, (1u64 << count) - 1);
        } else if DIRECT_ROOTS[start + 63] / 64 == source {
            actions[source] = DirectAction::Full(group);
        } else {
            let mut split = 0;
            while split < 64 && DIRECT_ROOTS[start + split] < first + 64 {
                split += 1;
            }
            assert!(split > 0 && split < 64);
            actions[source] = DirectAction::Start;
            actions[source + 1] = DirectAction::End(group, (1u64 << split) - 1);
        }
        group += 1;
    }
    actions
}
const DIRECT_ACTIONS: [DirectAction; 256] = direct_actions();
fn direct_plan_valid() -> bool {
    static VALID: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let fold = super::super::BLAKE3_ADJOINT_PLAN
            .linear_fold
            .get_or_init(|| super::super::BLAKE3_ADJOINT_PLAN.linear_fold_plan());
        fold.roots
            .iter()
            .copied()
            .filter(|wire| !(1089..1153).contains(wire))
            .eq(DIRECT_ROOTS)
    });
    *VALID
}
struct RootW16 {
    pending: V16,
    previous: V16,
    carry_lo: V16,
    carry_hi: V16,
    out: *mut u8,
}
impl RootW16 {
    #[inline(always)]
    fn new(out: *mut u8, pending: V16) -> Self {
        let zero = dup16(0);
        Self {
            pending,
            previous: zero,
            carry_lo: zero,
            carry_hi: zero,
            out,
        }
    }
    #[inline(always)]
    unsafe fn select(mask: u32, a: V16, b: V16) -> V16 {
        unsafe {
            if mask == 0 {
                b
            } else if mask == u32::MAX {
                a
            } else {
                _mm512_ternarylogic_epi32::<0xca>(dup16(mask), a, b)
            }
        }
    }
    #[inline(always)]
    unsafe fn publish(&self, group: usize, lo: V16, hi: V16) {
        unsafe {
            _mm512_store_si512(
                self.out.add(group * 512).cast(),
                _mm512_shuffle_i64x2::<0x44>(lo, hi),
            );
            _mm512_store_si512(
                self.out.add(group * 512 + 64).cast(),
                _mm512_shuffle_i64x2::<0xee>(lo, hi),
            );
        }
    }
    #[inline(always)]
    unsafe fn write_word<const WORD: usize>(&mut self, v: V16) {
        unsafe {
            let action = const { DIRECT_ACTIONS[WORD / 2] };
            if matches!(action, DirectAction::Skip) {
                return;
            }
            if WORD % 2 == 0 {
                self.previous = v;
                return;
            }
            match action {
                DirectAction::Skip => {}
                DirectAction::Full(group) => self.publish(group, self.previous, v),
                DirectAction::Start => {
                    self.carry_lo = self.previous;
                    self.carry_hi = v;
                }
                DirectAction::End(group, mask) => {
                    self.publish(
                        group,
                        Self::select(mask as u32, self.carry_lo, self.previous),
                        Self::select((mask >> 32) as u32, self.carry_hi, v),
                    );
                }
                DirectAction::Tail(group, mask) => {
                    self.publish(
                        group,
                        and16(self.previous, dup16(mask as u32)),
                        and16(v, dup16((mask >> 32) as u32)),
                    );
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
                let word = _mm512_shldi_epi32::<USED>(v, self.pending);
                self.write_word::<WORD>(word);
                self.pending = if WIDTH == 31 { shl16::<1>(v) } else { v };
            }
        }
    }
}

#[inline(never)]
unsafe fn recompute_pair<const SAVE: bool>(
    prepared: &Prepared16,
    out: *mut u8,
    deps: *mut V16,
    out_lo: *mut V16,
    out_hi: *mut V16,
) {
    unsafe {
        let il_a = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        let il_b = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
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

        let mut wa = RootW16::new(out, initial16(shr16::<31>(flags)));
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
                if SAVE {
                    *deps.add($g) = _mm512_permutex2var_epi32(bn, il_a, d2);
                    *deps.add(56 + $g) = _mm512_permutex2var_epi32(bn, il_b, d2);
                }
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

        if SAVE {
            for k in 0..4 {
                let lo0 = xor16(state[2 * k], state[2 * k + 8]);
                let lo1 = xor16(state[2 * k + 1], state[2 * k + 9]);
                let hi0 = xor16(state[2 * k + 8], cv_v[2 * k]);
                let hi1 = xor16(state[2 * k + 9], cv_v[2 * k + 1]);
                *out_lo.add(k) = _mm512_permutex2var_epi32(lo0, il_a, lo1);
                *out_lo.add(4 + k) = _mm512_permutex2var_epi32(lo0, il_b, lo1);
                *out_hi.add(k) = _mm512_permutex2var_epi32(hi0, il_a, hi1);
                *out_hi.add(4 + k) = _mm512_permutex2var_epi32(hi0, il_b, hi1);
            }
        }
    }
}

/// # Safety
/// Same full, aligned tile contract as `generate_tile`.
pub(crate) unsafe fn generate_tile16(init: u64, tile: usize, out: *mut u8) {
    assert!(tile < 4096);
    unsafe {
        for pair in 0..4 {
            let physical = tile * 64 + pair * 16;
            let base = physical / 256 + ((physical % 256) / 8) * 2048;
            let input = prepare16(init, base);
            recompute_pair::<false>(
                &input,
                out.add(pair * 128),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
        }
    }
}

#[test]
fn paired_roots_match_eight_lane_generator() {
    assert!(
        direct_plan_valid(),
        "fixed root order must match the actual circuit"
    );
    let mut a = Box::new(Tile([0xa5; 176 * 512]));
    let mut b = Box::new(Tile([0xa5; 176 * 512]));
    for seed in [0u64, 0x1234_5678_9abc_def0, u64::MAX] {
        for tile in [0usize, 1, 3, 4, 31, 127, 1024, 4095] {
            unsafe {
                generate_tile(seed, tile, a.0.as_mut_ptr());
                generate_tile16(seed, tile, b.0.as_mut_ptr());
            }
            assert_eq!(a.0, b.0, "seed={seed:x} tile={tile}");
        }
    }
}

#[test]
#[ignore = "native ranked-scale paired-kernel timing"]
fn paired_roots_timing() {
    use rayon::prelude::*;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .unwrap();
    let mut samples = [Vec::new(), Vec::new()];
    for pass in 0..14 {
        for mode in if pass % 2 == 0 { [0usize, 1] } else { [1, 0] } {
            let kernel: unsafe fn(u64, usize, *mut u8) = if mode == 0 {
                generate_tile
            } else {
                generate_tile16
            };
            let t = std::time::Instant::now();
            let sink: u64 = pool.install(|| {
                (0..4096usize)
                    .into_par_iter()
                    .map_init(
                        || Box::new(Tile([0; 176 * 512])),
                        |tile, index| {
                            unsafe {
                                kernel(0xdead_beef, index, tile.0.as_mut_ptr());
                            }
                            std::hint::black_box(tile.0.as_ptr());
                            u64::from(tile.0[index % tile.0.len()])
                        },
                    )
                    .sum()
            });
            let ms = t.elapsed().as_secs_f64() * 1e3;
            println!("ROOTS_PAIR pass={pass} mode={mode} ms={ms:.3} sink={sink}");
            if pass > 1 {
                samples[mode].push(ms);
            }
        }
    }
    for (mode, v) in samples.iter_mut().enumerate() {
        v.sort_by(f64::total_cmp);
        println!(
            "ROOTS_PAIR mode={mode} median_ms={:.3}",
            (v[5] + v[6]) * 0.5
        );
    }
}

/// # Safety
/// The root span is one aligned tile. Dependency spans are aligned writable
/// arrays of [2][56], [2][4], [2][4] ZMMs, respectively. All are disjoint.
pub(crate) unsafe fn generate_pair_with_deps(
    init: u64,
    tile: usize,
    pair: usize,
    out: *mut u8,
    deps: *mut u8,
    lo: *mut u8,
    hi: *mut u8,
) {
    assert!(tile < 4096 && pair < 4);
    unsafe {
        let physical = tile * 64 + pair * 16;
        let base = physical / 256 + ((physical % 256) / 8) * 2048;
        let input = prepare16(init, base);
        recompute_pair::<true>(
            &input,
            out.add(pair * 128),
            deps.cast(),
            lo.cast(),
            hi.cast(),
        );
    }
}

pub(crate) fn selected_generator() -> Option<(
    unsafe fn(u64, usize, *mut u8),
    Option<flock_core::lincheck::RootPairGenerator>,
)> {
    if !enabled() || !direct_plan_valid() {
        return None;
    }
    static EIGHT: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_ROOT_RECOMPUTE8").is_some());
    static REUSE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ROOT_PAIR_REUSE").is_none());
    if *EIGHT {
        Some((generate_tile, None))
    } else {
        Some((
            generate_tile16,
            if *REUSE {
                Some(generate_pair_with_deps)
            } else {
                None
            },
        ))
    }
}

#[test]
fn paired_dependencies_match_independent_blake3_pass() {
    for seed in [0u64, 0x1234_5678_9abc_def0, u64::MAX] {
        for tile in [0usize, 3, 4, 127, 4095] {
            super::super::blake3_zregen::imp::check_generated_pair(
                seed,
                tile,
                generate_tile16,
                generate_pair_with_deps,
            );
        }
    }
}

// Packed products are (sum XOR x) AND (sum XOR y). Form the full
// three-input Boolean function directly, retaining the already-computed sum.
#[inline(always)]
fn add_root8(x: V8, y: V8) -> (V8, V8) {
    unsafe {
        let s = _mm256_add_epi32(x, y);
        (s, _mm256_ternarylogic_epi32::<0x42>(x, y, s))
    }
}
#[inline(always)]
fn add_root16(x: V16, y: V16) -> (V16, V16) {
    unsafe {
        let s = _mm512_add_epi32(x, y);
        (s, _mm512_ternarylogic_epi32::<0x42>(x, y, s))
    }
}

#[path = "blake3_stage_recompute.rs"]
pub(crate) mod stage_recompute;

#[path="blake3_fused_fold.rs"]
pub(crate) mod fused_fold;
