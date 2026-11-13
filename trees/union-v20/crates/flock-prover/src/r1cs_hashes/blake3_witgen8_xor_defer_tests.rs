//! Source-level publication oracle. Run on x86, not as a whole-proof substitute.
use super::super::{AbPend, compression_lane_offset, flush_ab_pend};
use super::*;
use flock_core::{field::F8, ntt::AdditiveNttGf8};
use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};

// Cross-quarter addresses are far apart. Reserve the real address range but
// touch only the eight owned compression rows, rather than initializing GiBs.
struct SparseRows {
    p: *mut u8,
    layout: Layout,
    stride: usize,
    cq: bool,
}
impl SparseRows {
    fn new(stride: usize, cq: bool) -> Self {
        let len = (compression_lane_offset(7, cq) + 1) * stride;
        let layout = Layout::from_size_align(len, 64).unwrap();
        let p = unsafe { alloc_zeroed(layout) };
        if p.is_null() {
            handle_alloc_error(layout);
        }
        let mut out = Self {
            p,
            layout,
            stride,
            cq,
        };
        for j in 0..8 {
            out.row_mut(j).fill(0xa5);
        }
        out
    }
    fn row_mut(&mut self, j: usize) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.p
                    .add(compression_lane_offset(j, self.cq) * self.stride),
                self.stride,
            )
        }
    }
    fn assert_rows(&mut self, expected: &[Vec<u8>; 8], label: &str) {
        for j in 0..8 {
            assert_eq!(self.row_mut(j), expected[j], "{label}, lane {j}");
        }
    }
}
impl Drop for SparseRows {
    fn drop(&mut self) {
        unsafe { dealloc(self.p, self.layout) };
    }
}
#[repr(align(32))]
struct Ring([[u32; 8]; 32]);
#[repr(align(64))]
struct Compact([u8; 8 * 16 * 16]);

unsafe fn run_sequence<const COLS: usize>(cq: bool, cached: bool, rw: usize) {
    unsafe {
        let ns = AdditiveNttGf8::new(6, F8::ZERO);
        let nl = AdditiveNttGf8::new(6, F8(64));
        let table = InvNttTableByteSingleGf8::new(&ns, &nl);
        let tables = Tables::new(&table);
        let plan_bytes = Compact([0; 8 * 16 * 16]);
        let plan = flock_core::zerocheck::univariate_skip_optimized::prepare_round1_ab_window_plan(
            &table, &plan_bytes.0, true,
        );
        assert!(*RPPRUNE_ON, "run the deferred oracle with pruning enabled");
        let mut pending = AbPend::new();
        flush_ab_pend(core::ptr::null_mut());
        flush_ab_pend(&mut pending);
        let mut seed = 0xa4f9_31d2_7e83_160bu64;
        // Complete octa, skipped windows/early end, and isolated final window.
        // Reusing one pending buffer across fresh allocations checks ownership.
        let sequences = [(2..30).collect::<Vec<_>>(), vec![2, 4, 28], vec![29]];
        for (octa, sequence) in sequences.iter().enumerate() {
            assert!(!pending.live);
            let stride = if octa == 1 { BYTES_PER_BLOCK } else { 29 * 64 };
            let bias = if octa == 1 { 0 } else { 128 };
            let mut output = SparseRows::new(stride, cq);
            let mut a = SparseRows::new(BYTES_PER_BLOCK, cq);
            let mut b = SparseRows::new(BYTES_PER_BLOCK, cq);
            let mut z = SparseRows::new(BYTES_PER_BLOCK, cq);
            let mut compact = Compact([0xa5; 8 * 16 * 16]);
            let mut expected: [Vec<u8>; 8] = std::array::from_fn(|_| vec![0xa5; stride]);
            let mut wa: [Vec<u8>; 8] = std::array::from_fn(|_| vec![0xa5; BYTES_PER_BLOCK]);
            let mut wb = wa.clone();
            let mut wz = wa.clone();
            let mut old: Option<(usize, [[u8; 64]; 8])> = None;
            for &blk in sequence {
                let mut ar = Ring([[0xdeadbeef; 8]; 32]);
                let mut br = Ring([[0xdeadbeef; 8]; 32]);
                let mut aa = [[0u8; 64]; 8];
                let mut bb = [[0u8; 64]; 8];
                for j in 0..8 {
                    for i in 0..64 {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        aa[j][i] = seed as u8;
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        bb[j][i] = if known(mode(blk, i / 8)) & (1 << (i % 8)) != 0 {
                            255
                        } else {
                            seed as u8
                        };
                    }
                    for w in 0..16 {
                        ar.0[rw + w][j] =
                            u32::from_le_bytes(aa[j][4 * w..4 * w + 4].try_into().unwrap());
                        br.0[rw + w][j] =
                            u32::from_le_bytes(bb[j][4 * w..4 * w + 4].try_into().unwrap());
                    }
                }
                let proj = StreamProj {
                    compact_z_tile_words: COLS * 4,
                    seed_window: None,
                    cross_quarter: cq,
                    compact_z: if COLS == 0 {
                        core::ptr::null_mut()
                    } else {
                        compact.0.as_mut_ptr().cast()
                    },
                    elide_z: COLS != 0,
                    stage: core::ptr::null_mut(),
                    out: output.p,
                    out_stride: stride,
                    out_bias: bias,
                    ab_pend: &mut pending,
                    inv_table: &table,
                    plan,
                    one_rows_elided: true,
                    w30_elided: true,
                };
                let offset = blk * 64;
                let rows = RankedRows::<COLS>::new(
                    z.p.add(offset).cast(),
                    a.p.add(offset).cast(),
                    b.p.add(offset).cast(),
                )
                .with_cross_quarter(cq)
                .with_cached_z(cached)
                .with_compact_z(proj.compact_z, blk * 16, COLS * 4)
                .with_elide_z(COLS != 0);
                if cq {
                    // Actual production dispatch: non-null pending, cross-quarter.
                    project(
                        &proj,
                        blk,
                        rows,
                        ar.0.as_ptr().cast(),
                        br.0.as_ptr().cast(),
                        rw,
                        &tables,
                    );
                } else if cached {
                    // Small-layout check of the same constant deferred kernel.
                    project_policy::<true, false, COLS, true, true>(
                        &proj,
                        blk,
                        rows,
                        ar.0.as_ptr().cast(),
                        br.0.as_ptr().cast(),
                        rw,
                        &tables,
                    );
                } else {
                    project_policy::<false, false, COLS, true, true>(
                        &proj,
                        blk,
                        rows,
                        ar.0.as_ptr().cast(),
                        br.0.as_ptr().cast(),
                        rw,
                        &tables,
                    );
                }
                _mm_sfence();
                if let Some((old_offset, values)) = old.take() {
                    for j in 0..8 {
                        expected[j][old_offset..old_offset + 64].copy_from_slice(&values[j]);
                    }
                }
                let keep = match blk {
                    2 => 0xfc,
                    29 => 0x0f,
                    _ => 0xff,
                };
                let mut values = [[0u8; 64]; 8];
                for j in 0..8 {
                    let mut acc = [F8::ZERO; 64];
                    for k in (0..8).rev() {
                        let mut av = [F8::ZERO; 64];
                        let mut bv = [F8::ZERO; 64];
                        table.apply_scalar(&aa[j][k * 8..k * 8 + 8], &mut av);
                        table.apply_scalar(&bb[j][k * 8..k * 8 + 8], &mut bv);
                        for i in 0..64 {
                            acc[i] *= F8(2);
                            if keep & (1 << k) != 0 {
                                acc[i] += av[i] * bv[i];
                            }
                        }
                    }
                    values[j] = acc.map(|x| x.0);
                    wa[j][offset..offset + 64].copy_from_slice(&aa[j]);
                    wb[j][offset..offset + 64].copy_from_slice(&bb[j]);
                    for i in 0..64 {
                        if COLS == 0 {
                            wz[j][offset + i] = aa[j][i] & bb[j][i];
                        } else {
                            let compact_offset = j * COLS * 16 + (offset & (COLS * 16 - 1));
                            assert_eq!(compact.0[compact_offset + i], aa[j][i] & bb[j][i]);
                        }
                    }
                }
                if blk == 29 {
                    assert!(
                        !pending.live,
                        "row7 drained old lines before final direct stores"
                    );
                    for j in 0..8 {
                        expected[j][offset - bias..offset - bias + 64].copy_from_slice(&values[j]);
                    }
                } else {
                    assert!(pending.live);
                    assert_eq!(
                        pending.v, values,
                        "new pending values match independent scalar oracle"
                    );
                    assert_eq!(pending.base, output.p.add(offset - bias));
                    assert_eq!(pending.stride, stride);
                    assert_eq!(pending.cq, cq);
                    old = Some((offset - bias, values));
                }
                output.assert_rows(&expected, "delayed projection and untouched guard windows");
                a.assert_rows(&wa, "A publication");
                b.assert_rows(&wb, "B publication");
                z.assert_rows(&wz, "Z publication or elision");
            }
            flush_ab_pend(&mut pending);
            _mm_sfence();
            if let Some((offset, values)) = old.take() {
                for j in 0..8 {
                    expected[j][offset..offset + 64].copy_from_slice(&values[j]);
                }
            }
            assert!(!pending.live);
            output.assert_rows(&expected, "octa-end flush");
            // A second flush must not use a stale destination or change bytes.
            pending.v.fill([0x5a; 64]);
            flush_ab_pend(&mut pending);
            _mm_sfence();
            output.assert_rows(&expected, "idempotent flush");
        }
    }
}

#[test]
fn deferred_projection_sequence_matches_scalar_and_publication() {
    for cq in [false, true] {
        for cached in [false, true] {
            for rw in [0, 16] {
                unsafe {
                    run_sequence::<0>(cq, cached, rw);
                    run_sequence::<16>(cq, cached, rw);
                }
            }
        }
    }
    println!(
        "PASS deferred complete/partial/final octas: scalar AB, pending reuse, row7 drain, window29, idempotent/end flush, compact/global Z, both ring halves and CPU lane layouts"
    );
}
