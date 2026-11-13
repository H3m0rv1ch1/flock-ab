//! Non-streaming rollback regression cases. Run each environment in a fresh process.
use super::*;
use crate::seed_pipe::BlockSource;

fn compare(source: BlockSource<'_>, n_log: usize, ab_nt: bool, public: bool) {
    let blocks = source.to_vec();
    let (z0, a0, b0, mut ab0) =
        generate_witness_with_ab_packed_and_round1_inner_impl_tuned(
            BlockSource::Slice(&blocks), n_log, false, false, false,
        );
    let (z1, a1, b1, mut ab1) = if public {
        generate_witness_with_ab_packed_and_round1_inner_from(source, n_log)
    } else {
        generate_witness_with_ab_packed_and_round1_inner_impl_tuned(
            source, n_log, false, true, ab_nt,
        )
    };
    assert_eq!(z0, z1, "z n_log={n_log} ab_nt={ab_nt} public={public}");
    assert_eq!(a0, a1, "a n_log={n_log} ab_nt={ab_nt} public={public}");
    assert_eq!(b0, b1, "b n_log={n_log} ab_nt={ab_nt} public={public}");
    assert_eq!(ab0.as_bytes_mut(), ab1.as_bytes_mut(), "AB projection");
}

#[test]
fn fallback_nonstreaming_closed_len1_matches_scalar() {
    let BlockSource::Closed { init, .. } = BlockSource::closed(3, 0) else {
        unreachable!()
    };
    // Exact reported shape: len1 padded to8, SIMD requested, AB NT disabled.
    compare(BlockSource::Closed { init, len: 1 }, 3, false, false);
}

#[test]
fn fallback_sources_and_boundaries_match_scalar() {
    let mut cases = 0;
    for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65] {
        let n_log = len.max(8).next_power_of_two().trailing_zeros() as usize;
        for seed in [0u64, u64::MAX] {
            let BlockSource::Closed { init, .. } = BlockSource::closed(n_log as u32, seed) else {
                unreachable!()
            };
            let closed = BlockSource::Closed { init, len };
            let blocks = closed.to_vec();
            for ab_nt in [false, true] {
                compare(closed, n_log, ab_nt, false);
                compare(BlockSource::Slice(&blocks), n_log, ab_nt, false);
                cases += 2;
            }
            compare(closed, n_log, false, true);
            compare(BlockSource::Slice(&blocks), n_log, false, true);
            let (z0, a0, b0, mut ab0) =
                generate_witness_with_ab_packed_and_round1_inner_impl_tuned(
                    BlockSource::Slice(&blocks), n_log, false, false, false,
                );
            let (z1, a1, b1, mut ab1) =
                generate_witness_with_ab_packed_and_round1_inner(&blocks, n_log);
            assert_eq!(z0, z1);
            assert_eq!(a0, a1);
            assert_eq!(b0, b1);
            assert_eq!(ab0.as_bytes_mut(), ab1.as_bytes_mut());
            cases += 3;
        }
    }
    for n_log in 0..3 {
        compare(BlockSource::closed(n_log as u32, 0), n_log, false, false);
        cases += 1;
    }
    println!("FALLBACK_EQUIVALENCE cases={cases} closed/slice/public padding/boundaries plus scalar-small");
}

#[test]
fn fallback_streaming_elided_destinations_all_alignments() {
    let source = BlockSource::closed(3, 0);
    let old_source = BlockSource::closed(3, u64::MAX);
    let (ze, ae, be, mut abe) =
        generate_witness_with_ab_packed_and_round1_inner_impl_tuned(
            source, 3, false, false, false,
        );
    let (zo, ao, bo, _) = generate_witness_with_ab_packed_and_round1_inner_impl_tuned(
        old_source, 3, false, false, false,
    );
    let ntt_s = flock_core::ntt::AdditiveNttGf8::new(K_SKIP, flock_core::field::F8::ZERO);
    let ntt_l = flock_core::ntt::AdditiveNttGf8::new(K_SKIP, flock_core::field::F8(1 << K_SKIP));
    let inv_table = flock_core::ntt::InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    let padding = ([0u32; 8], [0u32; 16], 0, 0, 0);
    let n = ze.len();
    let mut cases = 0;
    for residue in [0usize, 16, 32, 48] {
        // Vary each destination independently. Every view remains F128-aligned
        // and owns a full8-block witness; only the stronger ZMM alignment varies.
        for varied in 0..3 {
            let mut storage: [Vec<F128>; 3] = std::array::from_fn(|_| vec![F128::ZERO; n + 4]);
            let offsets: [usize; 3] = std::array::from_fn(|i| {
                let want = if i == varied { residue } else { 0 };
                (0..4).find(|&j| (storage[i].as_ptr() as usize + j * 16) % 64 == want).unwrap()
            });
            let [zs, as_, bs] = &mut storage;
            let z = &mut zs[offsets[0]..offsets[0] + n];
            let a = &mut as_[offsets[1]..offsets[1] + n];
            let b = &mut bs[offsets[2]..offsets[2] + n];
            z.copy_from_slice(&zo);
            a.copy_from_slice(&ao);
            b.copy_from_slice(&bo);
            let mut ab = flock_core::zerocheck::univariate_skip_optimized::Round1AbInner::take_uninit(8 * K / 8);
            eprintln!("FALLBACK_ALIGNMENT residue={residue} varied={varied} z={:p} a={:p} b={:p} ab={:p}", z.as_ptr(), a.as_ptr(), b.as_ptr(), ab.as_bytes_mut().as_ptr());
            generate_round1_inner_octa(source, 0, z, a, b, &mut ab, &inv_table, &padding, [true; 3], true);
            assert_eq!(z, ze.as_slice(), "z residue={residue} varied={varied}");
            assert_eq!(a, ae.as_slice(), "a residue={residue} varied={varied}");
            assert_eq!(b, be.as_slice(), "b residue={residue} varied={varied}");
            assert_eq!(ab.as_bytes_mut(), abe.as_bytes_mut());
            cases += 1;
        }
    }
    println!("FALLBACK_ALIGNMENT_EQUIVALENCE cases={cases}");
}

#[test]
fn fallback_streaming_projection_all_alignments() {
    use flock_core::zerocheck::univariate_skip_optimized::prepare_round1_ab_window_plan;
    let source = BlockSource::closed(3, 0);
    let blocks = source.to_vec();
    let (ze, ae, be, mut abe) = generate_witness_with_ab_packed_and_round1_inner_impl_tuned(
        source, 3, false, false, false,
    );
    let ntt_s = flock_core::ntt::AdditiveNttGf8::new(K_SKIP, flock_core::field::F8::ZERO);
    let ntt_l = flock_core::ntt::AdditiveNttGf8::new(K_SKIP, flock_core::field::F8(1 << K_SKIP));
    let inv_table = flock_core::ntt::InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    #[repr(align(64))]
    struct Stage([u32; blake3_witgen8::STREAM_STAGE_WORDS]);
    let n = ze.len();
    let mut cases = 0;
    for residue in [0usize, 16, 32, 48] {
        for varied in 0..4 {
            let mut storage: [Vec<F128>; 4] = std::array::from_fn(|_| vec![F128::ZERO; n + 4]);
            let offsets: [usize; 4] = std::array::from_fn(|i| {
                let want = if i == varied { residue } else { 0 };
                (0..4).find(|&j| (storage[i].as_ptr() as usize + j * 16) % 64 == want).unwrap()
            });
            let [zs, as_, bs, abs] = &mut storage;
            let z = &mut zs[offsets[0]..offsets[0] + n];
            let a = &mut as_[offsets[1]..offsets[1] + n];
            let b = &mut bs[offsets[2]..offsets[2] + n];
            // A valid prior witness supplies exactly the elided constants.
            z.copy_from_slice(&ze);
            a.copy_from_slice(&ae);
            b.copy_from_slice(&be);
            let ab = unsafe { std::slice::from_raw_parts_mut(abs.as_mut_ptr().add(offsets[3]).cast::<u8>(), n * 16) };
            let mut stage = Stage([0; blake3_witgen8::STREAM_STAGE_WORDS]);
            let plan = prepare_round1_ab_window_plan(&inv_table, ab, true);
            let proj = blake3_witgen8::StreamProj {
                compact_z_tile_words: 0, seed_window: None,
                cross_quarter: false, compact_z: core::ptr::null_mut(), elide_z: false,
                ab_pend: core::ptr::null_mut(),
                stage: stage.0.as_mut_ptr(), out: ab.as_mut_ptr(), out_stride: K / 8,
                out_bias: 0, inv_table: &inv_table, plan,
                one_rows_elided: false, w30_elided: false,
            };
            eprintln!("FALLBACK_PROJECTION_ALIGNMENT residue={residue} varied={varied}");
            unsafe {
                blake3_witgen8::build_octa_witness_ab_stream_elide(
                    blake3_witgen8::OctaInputs::Blocks(std::array::from_fn(|j| &blocks[j])),
                    z.as_mut_ptr().cast::<u32>(), a.as_mut_ptr().cast::<u32>(),
                    b.as_mut_ptr().cast::<u32>(), proj, [true; 3],
                );
                core::arch::x86_64::_mm_sfence();
            }
            assert_eq!(z, ze.as_slice());
            assert_eq!(a, ae.as_slice());
            assert_eq!(b, be.as_slice());
            assert_eq!(ab, abe.as_bytes_mut(), "AB residue={residue} varied={varied}");
            cases += 1;
        }
    }
    println!("FALLBACK_PROJECTION_ALIGNMENT_EQUIVALENCE cases={cases}");
}
