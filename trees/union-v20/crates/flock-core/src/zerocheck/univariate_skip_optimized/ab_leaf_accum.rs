//! Accumulate the 78 sparse Karatsuba leaves across all 32 windows before
//! interpolation. The selected groups contain 24 or 27 leaves; the original
//! six/nine-leaf grouping remains available as a diagnostic control.
//! Each complete leaf vector is interpolated and converted to GHASH byte
//! planes once per bank.
//! The matrix-family guard and basis conversion come from the declared
//! polynomial predecessor. Every window contributes its own weighted leaves.
//! The per-window leaf contributions enter the accumulator linearly (fixed
//! GHASH key powers), so the accumulated bank bytes do not depend on claim order.
//! The complete matrix key is checked; incompatible linear maps use the caller's fallback.
use super::F128;
use core::arch::global_asm;
global_asm!(include_str!("ab_leaf_accum.S"), options(raw));
// Lane ab1: pair-major copies of the merged banks (window stride one row).
global_asm!(include_str!("ab_leaf_accum_pmaj.S"), options(raw));
global_asm!(include_str!("ab_leaf_accum_ns.S"), options(raw));

const SCRATCH_DATA_BYTES: usize = (78 + 31) * 64;
const GROUPS0: [usize; 9] = [9, 9, 9, 9, 6, 9, 9, 9, 9];
const GROUPS2: [usize; 9] = [6, 9, 9, 9, 9, 9, 9, 9, 9];

// Each group's 32 coefficient rows are contiguous for the register loop.
// This is an exact permutation of the already-validated coefficient table.
fn pack_groups(windows: Vec<u64>, groups: &[usize]) -> Vec<u64> {
    assert_eq!(windows.len(), 32 * 78);
    let mut packed = Vec::with_capacity(windows.len());
    let mut start = 0;
    for &count in groups {
        for window in 0..32 {
            packed.extend_from_slice(&windows[window * 78 + start..window * 78 + start + count]);
        }
        start += count;
    }
    assert_eq!(start, 78);
    assert_eq!(packed.len(), windows.len());
    packed
}

pub(super) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R1_AB_LEAF_ACCUM").is_none());
    *ON
}

// The layout is fixed once per process. Keep one coefficient allocation
// pair, in the selected grouping, while retaining an exact diagnostic rollback.
fn grouped_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_AB_MERGED").is_none()
    });
    *ON
}

/// Lane ab1: true when [`try_run`] will take the merged leaf path for these
/// matrices (same key check as the per-worker cache).
pub(super) fn would_run(mats: &[u64]) -> bool {
    grouped_enabled() && mats.len() == 32 * 256 && Cache::new(mats).merged.is_some()
}

pub(super) struct Cache {
    key: Vec<u64>,
    tables: Option<(Vec<u64>, Vec<u64>)>,
    merged: Option<(Vec<u64>, Vec<u64>)>,
    scratch: Vec<u8>,
}

fn decode(scale: F128) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (bit, &(lo, hi)) in DECODE_DUAL.iter().enumerate() {
        let parity = ((scale.lo & lo).count_ones() ^ (scale.hi & hi).count_ones()) & 1;
        out[bit >> 3] |= (parity as u8) << (bit & 7);
    }
    out
}

impl Cache {
    fn new(mats: &[u64]) -> Self {
        let mut cache = Self {
            key: mats.to_vec(),
            tables: None,
            merged: None,
            scratch: Vec::new(),
        };
        let mut m0 = Vec::with_capacity(32 * 78);
        let mut m2 = Vec::with_capacity(32 * 78);
        let convert = super::convert_table();
        for row in mats.chunks_exact(256) {
            // At absolute row0, phi8(1)=1, so the first bit column is scale.
            let mut bytes = [0u8; 16];
            for byte in 0..16 {
                for bit in 0..8 {
                    bytes[byte] |= (((row[byte] >> (8 * (7 - bit))) & 1) as u8) << bit;
                }
            }
            let scale = F128 {
                lo: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                hi: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
            };
            let mut expected = [0u64; 256];
            super::ab_eq_fold_mats_row_transposed(convert, scale, &mut expected);
            if expected.as_slice() != row {
                return cache;
            }
            let coefficients = decode(scale);
            for (forms, dest) in [(&COEFF0, &mut m0), (&COEFF2, &mut m2)] {
                for &form in forms {
                    let mut mask = form;
                    let mut coefficient = 0u8;
                    while mask != 0 {
                        coefficient ^= coefficients[mask.trailing_zeros() as usize];
                        mask &= mask - 1;
                    }
                    dest.push(GF8_MATRICES[coefficient as usize]);
                }
            }
        }
        if grouped_enabled() {
            cache.merged = Some((pack_groups(m0, &[27,24,27]), pack_groups(m2, &[24,27,27])));
        } else {
            cache.tables = Some((pack_groups(m0, &GROUPS0), pack_groups(m2, &GROUPS2)));
        }
        // Keep both leaf banks and the temporary polynomial cache-line aligned.
        cache.scratch = vec![0; SCRATCH_DATA_BYTES + 63];
        cache
    }
}

pub(super) fn try_run(
    cache: &mut Option<Cache>,
    source: &[u8],
    mats: &[u64],
    x_hi: usize,
    mode: usize,
    planes: &mut [u8],
    layout: usize,
) -> bool {
    const BAND_BYTES: usize = 2048 * 29 * 64;
    if mats.len() != 32 * 256 || planes.len() != 128 * 1024 || mode > 2 {
        return false;
    }
    // Probe only: the padded split-half layout spans more than one band.
    let x_hi = if layout == 65 { x_hi % 120 } else { x_hi };
    let end_pad = if layout == 65 { 32 * 15 * 4096 * 2 + 4096 - BAND_BYTES } else { 0 };
    let Some(start) = x_hi.checked_mul(BAND_BYTES) else {
        return false;
    };
    let Some(end) = start.checked_add(BAND_BYTES) else {
        return false;
    };
    let Some(rows) = source.get(start..end + end_pad) else {
        return false;
    };
    if cache.as_ref().is_none_or(|c| c.key.as_slice() != mats) {
        *cache = Some(Cache::new(mats));
    }
    let cache = cache.as_mut().unwrap();
    let merged = grouped_enabled();
    let tables = if merged { &cache.merged } else { &cache.tables };
    let Some((m0, m2)) = tables else {
        return false;
    };
    let raw = cache.scratch.as_mut_ptr();
    let offset = raw.align_offset(64);
    debug_assert!(offset + SCRATCH_DATA_BYTES <= cache.scratch.len());
    // SAFETY: byte alignment padding reserves the complete live span.
    let scratch = unsafe { raw.add(offset) };
    // Each original input span, matrix row, initialized scratch bank and
    // output bank is disjoint and complete. All final output planes are written.
    unsafe {
        flock_ab_poly_band(
            rows.as_ptr(),
            m0.as_ptr(),
            m2.as_ptr(),
            BASIS_MATRICES.as_ptr(),
            scratch,
            planes.as_mut_ptr(),
            mode,
            merged,
            if merged { layout } else { 0 },
        );
    }
    true
}

// System V x86-64 ABI. Each bank function sums six/nine-leaf groups in
// registers over all windows and writes every one of the 78 result slots.
// Its demand loads remain inside the original fourteen live source rows.
// Mode 1/2 prefetch the current bank's live rows into L2, once per window,
// in the first group; later groups reuse those rows. No hints are demand reads.
type Bank = unsafe extern "C" fn(*const u8, *const u64, *mut u8);
unsafe extern "C" {
    fn flock_ab_register_bank0_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_register_bank0_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_register_bank0_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_register_bank2_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_register_bank2_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_register_bank2_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_leaf_interpolate0(leaves: *const u8, polynomial: *mut u8);
    fn flock_ab_leaf_interpolate2(leaves: *const u8, polynomial: *mut u8);
    fn flock_ab_poly_finalize(polynomial: *const u8, basis: *const u64, planes: *mut u8);
    fn flock_ab_merged_bank0_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_merged_bank0_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_merged_bank0_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_merged_bank2_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_merged_bank2_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_merged_bank2_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_pmaj_bank0_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_pmaj_bank0_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_pmaj_bank0_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_pmaj_bank2_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_pmaj_bank2_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_pmaj_bank2_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32_bank0_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32_bank0_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32_bank0_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32_bank2_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32_bank2_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32_bank2_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns16_bank0_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns16_bank0_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns16_bank0_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns16_bank2_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns16_bank2_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns16_bank2_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsh_bank0_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsh_bank0_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsh_bank0_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsh_bank2_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsh_bank2_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsh_bank2_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32p_bank0_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32p_bank0_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32p_bank0_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32p_bank2_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32p_bank2_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_ns32p_bank2_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsla_bank0_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsla_bank0_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsla_bank0_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsla_bank2_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsla_bank2_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nsla_bank2_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nshp_bank0_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nshp_bank0_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nshp_bank0_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nshp_bank2_m0(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nshp_bank2_m1(source: *const u8, matrices: *const u64, leaves: *mut u8);
    fn flock_ab_nshp_bank2_m2(source: *const u8, matrices: *const u64, leaves: *mut u8);
}

#[unsafe(no_mangle)]
#[inline(never)]
unsafe extern "C" fn flock_ab_poly_band(
    rows: *const u8,
    mats0: *const u64,
    mats2: *const u64,
    basis: *const u64,
    scratch: *mut u8,
    out: *mut u8,
    mode: usize,
    merged: bool,
    layout: usize,
) {
    // The public caller checked the complete source, tables and output.
    // All leaf accumulators start from zero in registers. The scratch bank
    // is completely overwritten before interpolation, independent of reuse.
    unsafe {
        let (bank0, bank2): (Bank, Bank) = match (merged, mode) {
            (false, 0) => (flock_ab_register_bank0_m0, flock_ab_register_bank2_m0),
            (false, 1) => (flock_ab_register_bank0_m1, flock_ab_register_bank2_m1),
            (false, 2) => (flock_ab_register_bank0_m2, flock_ab_register_bank2_m2),
            (true, 0) => (flock_ab_merged_bank0_m0, flock_ab_merged_bank2_m0),
            (true, 1) => (flock_ab_merged_bank0_m1, flock_ab_merged_bank2_m1),
            (true, 2) => (flock_ab_merged_bank0_m2, flock_ab_merged_bank2_m2),
            _ => unreachable!(),
        };
        // Lane ab1: pair-major band layout. Row `w*64 + pair` of the band sits
        // at row position `pair*32 + w`, so one pair's 32 window rows are one
        // contiguous 59 KiB span and the band is read as ONE sequential stream.
        let (bank0, bank2, pair_bytes, half): (Bank, Bank, usize, usize) = match layout {
            0 => (bank0, bank2, 1856, 896),
            1 => match mode {
                0 => (flock_ab_pmaj_bank0_m0, flock_ab_pmaj_bank2_m0, 32 * 1856, 896),
                1 => (flock_ab_pmaj_bank0_m1, flock_ab_pmaj_bank2_m1, 32 * 1856, 896),
                _ => (flock_ab_pmaj_bank0_m2, flock_ab_pmaj_bank2_m2, 32 * 1856, 896),
            },
            32 => match mode {
                0 => (flock_ab_ns32_bank0_m0, flock_ab_ns32_bank2_m0, 1792, 896),
                1 => (flock_ab_ns32_bank0_m1, flock_ab_ns32_bank2_m1, 1792, 896),
                _ => (flock_ab_ns32_bank0_m2, flock_ab_ns32_bank2_m2, 1792, 896),
            },
            16 => match mode {
                0 => (flock_ab_ns16_bank0_m0, flock_ab_ns16_bank2_m0, 2 * 1792, 2 * 896),
                1 => (flock_ab_ns16_bank0_m1, flock_ab_ns16_bank2_m1, 2 * 1792, 2 * 896),
                _ => (flock_ab_ns16_bank0_m2, flock_ab_ns16_bank2_m2, 2 * 1792, 2 * 896),
            },
            // split halves: 64 streams of 896 B per pair
            64 => match mode {
                0 => (flock_ab_nsh_bank0_m0, flock_ab_nsh_bank2_m0, 896, 32 * 64 * 896),
                1 => (flock_ab_nsh_bank0_m1, flock_ab_nsh_bank2_m1, 896, 32 * 64 * 896),
                _ => (flock_ab_nsh_bank0_m2, flock_ab_nsh_bank2_m2, 896, 32 * 64 * 896),
            },
            65 => match mode {
                0 => (flock_ab_nshp_bank0_m0, flock_ab_nshp_bank2_m0, 896, 32 * 15 * 4096 + 2048),
                1 => (flock_ab_nshp_bank0_m1, flock_ab_nshp_bank2_m1, 896, 32 * 15 * 4096 + 2048),
                _ => (flock_ab_nshp_bank0_m2, flock_ab_nshp_bank2_m2, 896, 32 * 15 * 4096 + 2048),
            },
            // 32 streams, hole-free 1792 B pairs, stream stride 29 pages
            33 => match mode {
                0 => (flock_ab_ns32p_bank0_m0, flock_ab_ns32p_bank2_m0, 1792, 896),
                1 => (flock_ab_ns32p_bank0_m1, flock_ab_ns32p_bank2_m1, 1792, 896),
                _ => (flock_ab_ns32p_bank0_m2, flock_ab_ns32p_bank2_m2, 1792, 896),
            },
            // incumbent layout, software prefetch one half-row ahead
            2 => match mode {
                0 => (flock_ab_nsla_bank0_m0, flock_ab_nsla_bank2_m0, 1856, 896),
                1 => (flock_ab_nsla_bank0_m1, flock_ab_nsla_bank2_m1, 1856, 896),
                _ => (flock_ab_nsla_bank0_m2, flock_ab_nsla_bank2_m2, 1856, 896),
            },
            _ => unreachable!(),
        };
        let leaves = scratch;
        let polynomial = scratch.add(78 * 64);
        for pair in 0..64 {
            let source = rows.add(pair * pair_bytes);
            bank2(source, mats2, leaves);
            flock_ab_leaf_interpolate2(leaves, polynomial);
            flock_ab_poly_finalize(polynomial.add(128), basis.add(32), out.add(2 * pair * 1024));
            bank0(source.add(half), mats0, leaves);
            flock_ab_leaf_interpolate0(leaves, polynomial);
            flock_ab_poly_finalize(polynomial, basis, out.add((2 * pair + 1) * 1024));
        }
    }
}

// Constant maps are derived from gamma^j*phi8(v), with AES GF256 multiplication.
const DECODE_DUAL: [(u64, u64); 128] = [
    (0xd334d3a6ee310001, 0x5bd11ffec97358fe),
    (0x203af6c192e20000, 0xb8277c9033a9299b),
    (0x917a41cacc340000, 0xd7e54afbd05d48f8),
    (0x96c94948f6f90000, 0x989a93c72726b5aa),
    (0x1ff78f2109260000, 0xc862e77dac9dc337),
    (0x8d9b82dcb7810000, 0xe0c7a2915c06251d),
    (0x40f2e81264850000, 0xc0f87f2f8b65bed6),
    (0x7e29a6475d430000, 0x15af49665c38fee8),
    (0x9eabd7d3eb7a0002, 0x9e06f5b5004d0c93),
    (0x1cdac77bace50000, 0x910ae6fc9a882a95),
    (0xa860083783350000, 0x16ec66c3f123e32e),
    (0x2d2e936d7be30000, 0x27abaf17b4a9c1c5),
    (0xf69f31c51f070000, 0x165b4df7a301e4b5),
    (0xd5f422bc58840000, 0x086e1f12b54dd5b2),
    (0xc618314801530000, 0x18e9d0d5903194b8),
    (0xfaf673b922490000, 0xc133e6540763aa10),
    (0xae2d95efca510004, 0xb1ba0322b868559f),
    (0xd65a68865eac0000, 0xf6c146fd439149aa),
    (0x37bb063ecac00000, 0x1a781704adf0ac42),
    (0xaf4a3d7685cb0000, 0x5052c140d67f0fbf),
    (0xd54012e89f070000, 0x138cd93e2e4cde95),
    (0x3e37400735620000, 0x7706a2f56d7bb01d),
    (0xc867ce372e7d0000, 0x411408af7cbbc427),
    (0x8a6f777c2dc20000, 0x2444a0f3e072aacb),
    (0xf1465e3e27d00008, 0x2d0a50dde59b0d29),
    (0x2193522c9cc80000, 0x1bdba7f221c01cd8),
    (0xde36bb76cb560000, 0x5b321862b81539e7),
    (0x270ed4286c180000, 0xfa7512db07bd1cc5),
    (0x5d9745ff9c920000, 0x624e8fa08b1a355e),
    (0xee028df3d4630000, 0xc6a800062a6c8610),
    (0x5da6f6a08ffe0000, 0xa217cce02e141384),
    (0x2a05a0ad62420000, 0x9dde77ae17b815a9),
    (0xc505423fe78f0010, 0xad4c04170fe4ce9a),
    (0x7cebddb9a2540000, 0x47f2d409dcb4d31c),
    (0xa08cb5fbed190000, 0x8146d8affc711e2a),
    (0x6c9b24ef0c520000, 0x0001730cae98bec1),
    (0xc5072db864670000, 0xd13256274a0c9454),
    (0x0f31c84146f70000, 0xd6811ff29daa54df),
    (0x4843c826632f0000, 0xa7d9faaea6f2566d),
    (0x367f25f774630000, 0x0fafc6c905f712cf),
    (0x417ad07a12c40020, 0x865429918d5cfbae),
    (0xc50c8ea4adff0000, 0x00e03ce9069be246),
    (0xd56c58c16a4b0000, 0xd6db08b24d8f272f),
    (0x165f58921b310000, 0x7a6f9914e2fde209),
    (0x46cd07f72fd90000, 0x31d754cdf1f7b360),
    (0x60f6373946bc0000, 0xae33fe1a9d88fdc6),
    (0xb2011cf3123c0000, 0xbb58a3e7ec062b91),
    (0x355e13ea1daa0000, 0xfda8615893040fbf),
    (0xbb21e7dfb0920040, 0x801e34cfc07be0a0),
    (0x665cb6f3b81a0000, 0x3cca7558a194dc4f),
    (0xb48508eae9a30000, 0xd658d324850e8d6a),
    (0xd8bf9d3de66d0000, 0x4ec800e232bee2b5),
    (0xaeb6b518bfc30000, 0x24c9da1cf080e189),
    (0xd802df988a800000, 0x68f6e749d27ac5eb),
    (0x20e994bd9eb20000, 0x10e866527e3010e5),
    (0xe48de8c368570000, 0xf249e7aea02c3e21),
    (0xdbe2bba244470080, 0x58dcb79eef58cdc7),
    (0xb090b022fde60000, 0x3669016477be4236),
    (0x835684a43e7c0000, 0x6d4856c2e348ff47),
    (0xcbf3d889d9c70000, 0x38009e89eef86803),
    (0x2252fbc380c70000, 0xf92e50fbf5509769),
    (0xd2cdda6033f60000, 0xf3606f9f749618da),
    (0x97e88821af180000, 0x79b2204c50232ab6),
    (0x7bf12b466afa0000, 0x628f7eb8798e1962),
    (0x40d2176a2a120100, 0xf4ee52e1093213fa),
    (0x045a0d96e7f70000, 0x01207af8c864ea32),
    (0x59a86e7b115b0000, 0xd20a35d7e169836e),
    (0x9342f5a6fbd00000, 0xc03eeb380a4d8387),
    (0xb218a1e097010000, 0x0487b9168fafa6a5),
    (0x77b950d32bdf0000, 0x18036d646836c3d9),
    (0x6d2383d3ea240000, 0x69ce9d04438ceffb),
    (0xb0a3b61c8bbc0000, 0x4a995b638f027140),
    (0x9ff997bc69110200, 0x923267820c74e4c1),
    (0xdb3cc977b7ce0000, 0x4f0f6296a35e260a),
    (0xb1eca68864360000, 0xe8d841855098a80c),
    (0xf0024bebf3cd0000, 0xce81b201423ca6e3),
    (0x39345f7093300000, 0x5bc748cc500834ea),
    (0x25a8c3eabe770000, 0x5f46de4af075e12b),
    (0x6da28f9b12510000, 0x563278fbdbffe955),
    (0x8d02c783f49c0000, 0xa838ba4db2a7fa43),
    (0xccfd0a1faef10400, 0xc792ac6ae233b8e6),
    (0x2703d325a3a80000, 0x49fb8fd696e104ed),
    (0x2624d7fed0a40000, 0x12fb0f334f64bd4d),
    (0x2cc7cb50008d0000, 0x0eb09c81e197d60e),
    (0xfff33c3d91e10000, 0x57493309fc164cc9),
    (0x0ba36fc7186b0000, 0x7e75c3ba6b8e7c80),
    (0xa56cb97179e10000, 0xb9cbb891ebc72c42),
    (0xc9315ca107090000, 0x0ba06b65ac3cac79),
    (0xe56fc80646210800, 0xd6463b990a1421f5),
    (0x297cb01a8bfa0000, 0xa457c52e627563c5),
    (0x2a248e19cfe00000, 0xb74469419917ba17),
    (0xf28bdd8abcb60000, 0xafec939a1392f75a),
    (0x6a397d04a7a80000, 0xc948fac3ffcc82eb),
    (0x531173291c0d0000, 0x8c2c2e708bc41456),
    (0x355ae2ec9a920000, 0xd5b41d78393ba66f),
    (0x0118830a12b70000, 0x9af73e83e08b144b),
    (0x1f4e7c9dfabc1000, 0x1d301c5462494cd5),
    (0x4ca4d95d2ac10000, 0x3341481cdafa04bf),
    (0x999c7701dad50000, 0x582987a41fa84575),
    (0xffef46c80c070000, 0x9408335fee887e00),
    (0xcd9c4be348a80000, 0xa600eefb16f23bb1),
    (0xe2c94b0982d00000, 0x7e017d7cb1b46f6a),
    (0xe4840ace66270000, 0xa2711feff4536d81),
    (0x3be5413fb0740000, 0xb9581aeb1bd43f6a),
    (0xee026756ffcb2000, 0x886fa0d8d7c3c516),
    (0xc3ffa689d27d0000, 0x7b3573e40dd88d8d),
    (0x70dc4a26a3bc0000, 0x8feb7f7094f39aef),
    (0xd9fd459a84320000, 0x6cc4e62eab4f28ca),
    (0x13ac67e65a9c0000, 0xaf758c7114d201fc),
    (0xc484a1a6a7a20000, 0x5910575d2b017447),
    (0x78f4a36b04890000, 0x3dbe812df1b610f0),
    (0x52ff06428e470000, 0xc924ba50613104cd),
    (0xbd66e6373de24000, 0xdb8c6733a5c61d62),
    (0x49806bea056d0000, 0x29153561ec142e02),
    (0x57f72a38b5720000, 0x29f5992fc78c037f),
    (0x7e93e1fbdb600000, 0xf9b611e381fdca5e),
    (0x78e1a9034e8a0000, 0x40ef0829f4b8d489),
    (0x8b0938cf9fd50000, 0xe8720c093eadeccf),
    (0xc88100010a190000, 0xe155ed2ec34a9c8c),
    (0x81ddbf0d50230000, 0x8ccf5682dc92d911),
    (0x8e70e18b3dda8000, 0x1b917435462db6f2),
    (0x8cf758f503fc0000, 0x9a488dbe74b59f33),
    (0x2275d6addd650000, 0xb32c90ced31e23e3),
    (0x49680600945b0000, 0x9f1de6ba853fa65d),
    (0x901d7b60c9710000, 0x5c13be4819d494cd),
    (0x48bd20e5661a0000, 0x6bf2a57de82ea47c),
    (0x22fecd770c640000, 0xe1a5c61cf72af6aa),
    (0x1fe6bcf04de20000, 0x3822cdf6cf9a7556),
];
const GF8_MATRICES: [u64; 256] = [
    0x0000000000000000,
    0x0102040810204080,
    0x8081028488102040,
    0x8183068c983060c0,
    0x40c08142c4881020,
    0x41c2854ad4a850a0,
    0xc04183c64c983060,
    0xc14387ce5cb870e0,
    0x2060c0a162c48810,
    0x2162c4a972e4c890,
    0xa0e1c225ead4a850,
    0xa1e3c62dfaf4e8d0,
    0x60a041e3a64c9830,
    0x61a245ebb66cd8b0,
    0xe02143672e5cb870,
    0xe123476f3e7cf8f0,
    0x103060d0b162c488,
    0x113264d8a1428408,
    0x90b162543972e4c8,
    0x91b3665c2952a448,
    0x50f0e19275ead4a8,
    0x51f2e59a65ca9428,
    0xd071e316fdfaf4e8,
    0xd173e71eeddab468,
    0x3050a071d3a64c98,
    0x3152a479c3860c18,
    0xb0d1a2f55bb66cd8,
    0xb1d3a6fd4b962c58,
    0x70902133172e5cb8,
    0x7192253b070e1c38,
    0xf01123b79f3e7cf8,
    0xf11327bf8f1e3c78,
    0x889830e858b162c4,
    0x899a34e048912244,
    0x0819326cd0a14284,
    0x091b3664c0810204,
    0xc858b1aa9c3972e4,
    0xc95ab5a28c193264,
    0x48d9b32e142952a4,
    0x49dbb72604091224,
    0xa8f8f0493a75ead4,
    0xa9faf4412a55aa54,
    0x2879f2cdb265ca94,
    0x297bf6c5a2458a14,
    0xe838710bfefdfaf4,
    0xe93a7503eeddba74,
    0x68b9738f76eddab4,
    0x69bb778766cd9a34,
    0x98a85038e9d3a64c,
    0x99aa5430f9f3e6cc,
    0x182952bc61c3860c,
    0x192b56b471e3c68c,
    0xd868d17a2d5bb66c,
    0xd96ad5723d7bf6ec,
    0x58e9d3fea54b962c,
    0x59ebd7f6b56bd6ac,
    0xb8c890998b172e5c,
    0xb9ca94919b376edc,
    0x3849921d03070e1c,
    0x394b961513274e9c,
    0xf80811db4f9f3e7c,
    0xf90a15d35fbf7efc,
    0x7889135fc78f1e3c,
    0x798b1757d7af5ebc,
    0xc44c98f42c58b162,
    0xc54e9cfc3c78f1e2,
    0x44cd9a70a4489122,
    0x45cf9e78b468d1a2,
    0x848c19b6e8d0a142,
    0x858e1dbef8f0e1c2,
    0x040d1b3260c08102,
    0x050f1f3a70e0c182,
    0xe42c58554e9c3972,
    0xe52e5c5d5ebc79f2,
    0x64ad5ad1c68c1932,
    0x65af5ed9d6ac59b2,
    0xa4ecd9178a142952,
    0xa5eedd1f9a3469d2,
    0x246ddb9302040912,
    0x256fdf9b12244992,
    0xd47cf8249d3a75ea,
    0xd57efc2c8d1a356a,
    0x54fdfaa0152a55aa,
    0x55fffea8050a152a,
    0x94bc796659b265ca,
    0x95be7d6e4992254a,
    0x143d7be2d1a2458a,
    0x153f7feac182050a,
    0xf41c3885fffefdfa,
    0xf51e3c8defdebd7a,
    0x749d3a0177eeddba,
    0x759f3e0967ce9d3a,
    0xb4dcb9c73b76edda,
    0xb5debdcf2b56ad5a,
    0x345dbb43b366cd9a,
    0x355fbf4ba3468d1a,
    0x4cd4a81c74e9d3a6,
    0x4dd6ac1464c99326,
    0xcc55aa98fcf9f3e6,
    0xcd57ae90ecd9b366,
    0x0c14295eb061c386,
    0x0d162d56a0418306,
    0x8c952bda3871e3c6,
    0x8d972fd22851a346,
    0x6cb468bd162d5bb6,
    0x6db66cb5060d1b36,
    0xec356a399e3d7bf6,
    0xed376e318e1d3b76,
    0x2c74e9ffd2a54b96,
    0x2d76edf7c2850b16,
    0xacf5eb7b5ab56bd6,
    0xadf7ef734a952b56,
    0x5ce4c8ccc58b172e,
    0x5de6ccc4d5ab57ae,
    0xdc65ca484d9b376e,
    0xdd67ce405dbb77ee,
    0x1c24498e0103070e,
    0x1d264d861123478e,
    0x9ca54b0a8913274e,
    0x9da74f02993367ce,
    0x7c84086da74f9f3e,
    0x7d860c65b76fdfbe,
    0xfc050ae92f5fbf7e,
    0xfd070ee13f7ffffe,
    0x3c44892f63c78f1e,
    0x3d468d2773e7cf9e,
    0xbcc58babebd7af5e,
    0xbdc78fa3fbf7efde,
    0x62a64cfa962c58b1,
    0x63a448f2860c1831,
    0xe2274e7e1e3c78f1,
    0xe3254a760e1c3871,
    0x2266cdb852a44891,
    0x2364c9b042840811,
    0xa2e7cf3cdab468d1,
    0xa3e5cb34ca942851,
    0x42c68c5bf4e8d0a1,
    0x43c48853e4c89021,
    0xc2478edf7cf8f0e1,
    0xc3458ad76cd8b061,
    0x02060d193060c081,
    0x0304091120408001,
    0x82870f9db870e0c1,
    0x83850b95a850a041,
    0x72962c2a274e9c39,
    0x73942822376edcb9,
    0xf2172eaeaf5ebc79,
    0xf3152aa6bf7efcf9,
    0x3256ad68e3c68c19,
    0x3354a960f3e6cc99,
    0xb2d7afec6bd6ac59,
    0xb3d5abe47bf6ecd9,
    0x52f6ec8b458a1429,
    0x53f4e88355aa54a9,
    0xd277ee0fcd9a3469,
    0xd375ea07ddba74e9,
    0x12366dc981020409,
    0x133469c191224489,
    0x92b76f4d09122449,
    0x93b56b45193264c9,
    0xea3e7c12ce9d3a75,
    0xeb3c781adebd7af5,
    0x6abf7e96468d1a35,
    0x6bbd7a9e56ad5ab5,
    0xaafefd500a152a55,
    0xabfcf9581a356ad5,
    0x2a7fffd482050a15,
    0x2b7dfbdc92254a95,
    0xca5ebcb3ac59b265,
    0xcb5cb8bbbc79f2e5,
    0x4adfbe3724499225,
    0x4bddba3f3469d2a5,
    0x8a9e3df168d1a245,
    0x8b9c39f978f1e2c5,
    0x0a1f3f75e0c18205,
    0x0b1d3b7df0e1c285,
    0xfa0e1cc27ffffefd,
    0xfb0c18ca6fdfbe7d,
    0x7a8f1e46f7efdebd,
    0x7b8d1a4ee7cf9e3d,
    0xbace9d80bb77eedd,
    0xbbcc9988ab57ae5d,
    0x3a4f9f043367ce9d,
    0x3b4d9b0c23478e1d,
    0xda6edc631d3b76ed,
    0xdb6cd86b0d1b366d,
    0x5aefdee7952b56ad,
    0x5beddaef850b162d,
    0x9aae5d21d9b366cd,
    0x9bac5929c993264d,
    0x1a2f5fa551a3468d,
    0x1b2d5bad4183060d,
    0xa6ead40eba74e9d3,
    0xa7e8d006aa54a953,
    0x266bd68a3264c993,
    0x2769d28222448913,
    0xe62a554c7efcf9f3,
    0xe72851446edcb973,
    0x66ab57c8f6ecd9b3,
    0x67a953c0e6cc9933,
    0x868a14afd8b061c3,
    0x878810a7c8902143,
    0x060b162b50a04183,
    0x0709122340800103,
    0xc64a95ed1c3871e3,
    0xc74891e50c183163,
    0x46cb9769942851a3,
    0x47c9936184081123,
    0xb6dab4de0b162d5b,
    0xb7d8b0d61b366ddb,
    0x365bb65a83060d1b,
    0x3759b25293264d9b,
    0xf61a359ccf9e3d7b,
    0xf7183194dfbe7dfb,
    0x769b3718478e1d3b,
    0x7799331057ae5dbb,
    0x96ba747f69d2a54b,
    0x97b8707779f2e5cb,
    0x163b76fbe1c2850b,
    0x173972f3f1e2c58b,
    0xd67af53dad5ab56b,
    0xd778f135bd7af5eb,
    0x56fbf7b9254a952b,
    0x57f9f3b1356ad5ab,
    0x2e72e4e6e2c58b17,
    0x2f70e0eef2e5cb97,
    0xaef3e6626ad5ab57,
    0xaff1e26a7af5ebd7,
    0x6eb265a4264d9b37,
    0x6fb061ac366ddbb7,
    0xee336720ae5dbb77,
    0xef316328be7dfbf7,
    0x0e12244780010307,
    0x0f10204f90214387,
    0x8e9326c308112347,
    0x8f9122cb183163c7,
    0x4ed2a50544891327,
    0x4fd0a10d54a953a7,
    0xce53a781cc993367,
    0xcf51a389dcb973e7,
    0x3e42843653a74f9f,
    0x3f40803e43870f1f,
    0xbec386b2dbb76fdf,
    0xbfc182bacb972f5f,
    0x7e820574972f5fbf,
    0x7f80017c870f1f3f,
    0xfe0307f01f3f7fff,
    0xff0103f80f1f3f7f,
    0x1e2244973163c78f,
    0x1f20409f2143870f,
    0x9ea34613b973e7cf,
    0x9fa1421ba953a74f,
    0x5ee2c5d5f5ebd7af,
    0x5fe0c1dde5cb972f,
    0xde63c7517dfbf7ef,
    0xdf61c3596ddbb76f,
];
const COEFF0: [u16; 78] = [
    1, 2, 3, 4, 8, 12, 5, 10, 15, 16, 32, 48, 64, 128, 192, 80, 160, 240, 17, 34, 51, 68, 136, 204,
    85, 170, 255, 256, 512, 768, 1024, 2048, 3072, 1280, 2560, 3840, 4096, 8192, 12288, 20480,
    40960, 61440, 4352, 8704, 13056, 17408, 34816, 52224, 21760, 43520, 65280, 257, 514, 771, 1028,
    2056, 3084, 1285, 2570, 3855, 4112, 8224, 12336, 16448, 32896, 49344, 20560, 41120, 61680,
    4369, 8738, 13107, 17476, 34952, 52428, 21845, 43690, 65535,
];
const COEFF2: [u16; 78] = [
    4, 8, 12, 5, 10, 15, 16, 32, 48, 64, 128, 192, 80, 160, 240, 17, 34, 51, 68, 136, 204, 85, 170,
    255, 256, 512, 768, 1024, 2048, 3072, 1280, 2560, 3840, 4096, 8192, 12288, 16384, 32768, 49152,
    20480, 40960, 61440, 4352, 8704, 13056, 17408, 34816, 52224, 21760, 43520, 65280, 257, 514,
    771, 1028, 2056, 3084, 1285, 2570, 3855, 4112, 8224, 12336, 16448, 32896, 49344, 20560, 41120,
    61680, 4369, 8738, 13107, 17476, 34952, 52428, 21845, 43690, 65535,
];
const BASIS_MATRICES: [u64; 496] = [
    0x5790dcfea09cc60c,
    0x780840c20ec8188a,
    0xc4d69e92bc6e7cb0,
    0x4c54363eeeeadce8,
    0xd8745036e8b856a8,
    0xf864e86c569a1404,
    0x7e96d05024fc446a,
    0xce127432f406be40,
    0xd86a7e0a9a46127a,
    0xc89630fc08c65470,
    0x3852b468382e1c1a,
    0x88a0d4e0e4f012cc,
    0x9832166494d87e48,
    0xb4c62a0066a21c5c,
    0x8676d05e6cd02e02,
    0xfe907e2ec41c6890,
    0x90c700dcfea09c56,
    0x0c780840c20ec818,
    0x8ac4d69e92bc6e7c,
    0xb04c54363eeeeadc,
    0xe8d8745036e8b856,
    0xa8f864e86c569a14,
    0x047e96d05024fc44,
    0x6ace127432f406be,
    0x40d86a7e0a9a4612,
    0x7ac89630fc08c654,
    0x703852b468382e1c,
    0x1a88a0d4e0e4f012,
    0xcc9832166494d87e,
    0x48b4c62a0066a21c,
    0x5c8676d05e6cd02e,
    0x02fe907e2ec41c68,
    0x68f8af00dcfea0f4,
    0x560c780840c20ec8,
    0x188ac4d69e92bc6e,
    0x7cb04c54363eeeea,
    0xdce8d8745036e8b8,
    0x56a8f864e86c569a,
    0x14047e96d05024fc,
    0x446ace127432f406,
    0xbe40d86a7e0a9a46,
    0x127ac89630fc08c6,
    0x54703852b468382e,
    0x1c1a88a0d4e0e4f0,
    0x12cc9832166494d8,
    0x7e48b4c62a0066a2,
    0x1c5c8676d05e6cd0,
    0x2e02fe907e2ec41c,
    0x1c74e4af00dcfebc,
    0xf4560c780840c20e,
    0xc8188ac4d69e92bc,
    0x6e7cb04c54363eee,
    0xeadce8d8745036e8,
    0xb856a8f864e86c56,
    0x9a14047e96d05024,
    0xfc446ace127432f4,
    0x06be40d86a7e0a9a,
    0x46127ac89630fc08,
    0xc654703852b46838,
    0x2e1c1a88a0d4e0e4,
    0xf012cc9832166494,
    0xd87e48b4c62a0066,
    0xa21c5c8676d05e6c,
    0xd02e02fe907e2ec4,
    0xc4d8b0e4af00dc3a,
    0xbcf4560c780840c2,
    0x0ec8188ac4d69e92,
    0xbc6e7cb04c54363e,
    0xeeeadce8d8745036,
    0xe8b856a8f864e86c,
    0x569a14047e96d050,
    0x24fc446ace127432,
    0xf406be40d86a7e0a,
    0x9a46127ac89630fc,
    0x08c654703852b468,
    0x382e1c1a88a0d4e0,
    0xe4f012cc98321664,
    0x94d87e48b4c62a00,
    0x66a21c5c8676d05e,
    0x6cd02e02fe907e2e,
    0x2eeaf6b0e4af00f2,
    0x3abcf4560c780840,
    0xc20ec8188ac4d69e,
    0x92bc6e7cb04c5436,
    0x3eeeeadce8d87450,
    0x36e8b856a8f864e8,
    0x6c569a14047e96d0,
    0x5024fc446ace1274,
    0x32f406be40d86a7e,
    0x0a9a46127ac89630,
    0xfc08c654703852b4,
    0x68382e1c1a88a0d4,
    0xe0e4f012cc983216,
    0x6494d87e48b4c62a,
    0x0066a21c5c8676d0,
    0x5e6cd02e02fe907e,
    0x7e5094f6b0e4af7e,
    0xf23abcf4560c7808,
    0x40c20ec8188ac4d6,
    0x9e92bc6e7cb04c54,
    0x363eeeeadce8d874,
    0x5036e8b856a8f864,
    0xe86c569a14047e96,
    0xd05024fc446ace12,
    0x7432f406be40d86a,
    0x7e0a9a46127ac896,
    0x30fc08c654703852,
    0xb468382e1c1a88a0,
    0xd4e0e4f012cc9832,
    0x166494d87e48b4c6,
    0x2a0066a21c5c8676,
    0xd05e6cd02e02fe90,
    0x90eec094f6b0e43f,
    0x7ef23abcf4560c78,
    0x0840c20ec8188ac4,
    0xd69e92bc6e7cb04c,
    0x54363eeeeadce8d8,
    0x745036e8b856a8f8,
    0x64e86c569a14047e,
    0x96d05024fc446ace,
    0x127432f406be40d8,
    0x6a7e0a9a46127ac8,
    0x9630fc08c6547038,
    0x52b468382e1c1a88,
    0xa0d4e0e4f012cc98,
    0x32166494d87e48b4,
    0xc62a0066a21c5c86,
    0x76d05e6cd02e02fe,
    0xfe6e10c094f6b01a,
    0x3f7ef23abcf4560c,
    0x780840c20ec8188a,
    0xc4d69e92bc6e7cb0,
    0x4c54363eeeeadce8,
    0xd8745036e8b856a8,
    0xf864e86c569a1404,
    0x7e96d05024fc446a,
    0xce127432f406be40,
    0xd86a7e0a9a46127a,
    0xc89630fc08c65470,
    0x3852b468382e1c1a,
    0x88a0d4e0e4f012cc,
    0x9832166494d87e48,
    0xb4c62a0066a21c5c,
    0x8676d05e6cd02e02,
    0x02fc6c10c094f6b2,
    0x1a3f7ef23abcf456,
    0x0c780840c20ec818,
    0x8ac4d69e92bc6e7c,
    0xb04c54363eeeeadc,
    0xe8d8745036e8b856,
    0xa8f864e86c569a14,
    0x047e96d05024fc44,
    0x6ace127432f406be,
    0x40d86a7e0a9a4612,
    0x7ac89630fc08c654,
    0x703852b468382e1c,
    0x1a88a0d4e0e4f012,
    0xcc9832166494d87e,
    0x48b4c62a0066a21c,
    0x5c8676d05e6cd02e,
    0x2e2cd26c10c094d8,
    0xb21a3f7ef23abcf4,
    0x560c780840c20ec8,
    0x188ac4d69e92bc6e,
    0x7cb04c54363eeeea,
    0xdce8d8745036e8b8,
    0x56a8f864e86c569a,
    0x14047e96d05024fc,
    0x446ace127432f406,
    0xbe40d86a7e0a9a46,
    0x127ac89630fc08c6,
    0x54703852b468382e,
    0x1c1a88a0d4e0e4f0,
    0x12cc9832166494d8,
    0x7e48b4c62a0066a2,
    0x1c5c8676d05e6cd0,
    0xd0fefcd26c10c044,
    0xd8b21a3f7ef23abc,
    0xf4560c780840c20e,
    0xc8188ac4d69e92bc,
    0x6e7cb04c54363eee,
    0xeadce8d8745036e8,
    0xb856a8f864e86c56,
    0x9a14047e96d05024,
    0xfc446ace127432f4,
    0x06be40d86a7e0a9a,
    0x46127ac89630fc08,
    0xc654703852b46838,
    0x2e1c1a88a0d4e0e4,
    0xf012cc9832166494,
    0xd87e48b4c62a0066,
    0xa21c5c8676d05e6c,
    0x6cbc92fcd26c10ac,
    0x44d8b21a3f7ef23a,
    0xbcf4560c780840c2,
    0x0ec8188ac4d69e92,
    0xbc6e7cb04c54363e,
    0xeeeadce8d8745036,
    0xe8b856a8f864e86c,
    0x569a14047e96d050,
    0x24fc446ace127432,
    0xf406be40d86a7e0a,
    0x9a46127ac89630fc,
    0x08c654703852b468,
    0x382e1c1a88a0d4e0,
    0xe4f012cc98321664,
    0x94d87e48b4c62a00,
    0x66a21c5c8676d05e,
    0x5e32e292fcd26c4e,
    0xac44d8b21a3f7ef2,
    0x3abcf4560c780840,
    0xc20ec8188ac4d69e,
    0x92bc6e7cb04c5436,
    0x3eeeeadce8d87450,
    0x36e8b856a8f864e8,
    0x6c569a14047e96d0,
    0x5024fc446ace1274,
    0x32f406be40d86a7e,
    0x0a9a46127ac89630,
    0xfc08c654703852b4,
    0x68382e1c1a88a0d4,
    0xe0e4f012cc983216,
    0x6494d87e48b4c62a,
    0x0066a21c5c8676d0,
    0xd08ee2e292fcd2bc,
    0x4eac44d8b21a3f7e,
    0xf23abcf4560c7808,
    0x40c20ec8188ac4d6,
    0x9e92bc6e7cb04c54,
    0x363eeeeadce8d874,
    0x5036e8b856a8f864,
    0xe86c569a14047e96,
    0xd05024fc446ace12,
    0x7432f406be40d86a,
    0x7e0a9a46127ac896,
    0x30fc08c654703852,
    0xb468382e1c1a88a0,
    0xd4e0e4f012cc9832,
    0x166494d87e48b4c6,
    0x2a0066a21c5c8676,
    0x76a6f8e2e292fca4,
    0xbc4eac44d8b21a3f,
    0x7ef23abcf4560c78,
    0x0840c20ec8188ac4,
    0xd69e92bc6e7cb04c,
    0x54363eeeeadce8d8,
    0x745036e8b856a8f8,
    0x64e86c569a14047e,
    0x96d05024fc446ace,
    0x127432f406be40d8,
    0x6a7e0a9a46127ac8,
    0x9630fc08c6547038,
    0x52b468382e1c1a88,
    0xa0d4e0e4f012cc98,
    0x32166494d87e48b4,
    0xc62a0066a21c5c86,
    0x86f020f8e2e2927a,
    0xa4bc4eac44d8b21a,
    0x3f7ef23abcf4560c,
    0x780840c20ec8188a,
    0xc4d69e92bc6e7cb0,
    0x4c54363eeeeadce8,
    0xd8745036e8b856a8,
    0xf864e86c569a1404,
    0x7e96d05024fc446a,
    0xce127432f406be40,
    0xd86a7e0a9a46127a,
    0xc89630fc08c65470,
    0x3852b468382e1c1a,
    0x88a0d4e0e4f012cc,
    0x9832166494d87e48,
    0xb4c62a0066a21c5c,
    0x5cdaac20f8e2e2ce,
    0x7aa4bc4eac44d8b2,
    0x1a3f7ef23abcf456,
    0x0c780840c20ec818,
    0x8ac4d69e92bc6e7c,
    0xb04c54363eeeeadc,
    0xe8d8745036e8b856,
    0xa8f864e86c569a14,
    0x047e96d05024fc44,
    0x6ace127432f406be,
    0x40d86a7e0a9a4612,
    0x7ac89630fc08c654,
    0x703852b468382e1c,
    0x1a88a0d4e0e4f012,
    0xcc9832166494d87e,
    0x48b4c62a0066a21c,
    0x1c40c6ac20f8e2fe,
    0xce7aa4bc4eac44d8,
    0xb21a3f7ef23abcf4,
    0x560c780840c20ec8,
    0x188ac4d69e92bc6e,
    0x7cb04c54363eeeea,
    0xdce8d8745036e8b8,
    0x56a8f864e86c569a,
    0x14047e96d05024fc,
    0x446ace127432f406,
    0xbe40d86a7e0a9a46,
    0x127ac89630fc08c6,
    0x54703852b468382e,
    0x1c1a88a0d4e0e4f0,
    0x12cc9832166494d8,
    0x7e48b4c62a0066a2,
    0xa2bee2c6ac20f840,
    0xfece7aa4bc4eac44,
    0xd8b21a3f7ef23abc,
    0xf4560c780840c20e,
    0xc8188ac4d69e92bc,
    0x6e7cb04c54363eee,
    0xeadce8d8745036e8,
    0xb856a8f864e86c56,
    0x9a14047e96d05024,
    0xfc446ace127432f4,
    0x06be40d86a7e0a9a,
    0x46127ac89630fc08,
    0xc654703852b46838,
    0x2e1c1a88a0d4e0e4,
    0xf012cc9832166494,
    0xd87e48b4c62a0066,
    0x66c4d8e2c6ac209e,
    0x40fece7aa4bc4eac,
    0x44d8b21a3f7ef23a,
    0xbcf4560c780840c2,
    0x0ec8188ac4d69e92,
    0xbc6e7cb04c54363e,
    0xeeeadce8d8745036,
    0xe8b856a8f864e86c,
    0x569a14047e96d050,
    0x24fc446ace127432,
    0xf406be40d86a7e0a,
    0x9a46127ac89630fc,
    0x08c654703852b468,
    0x382e1c1a88a0d4e0,
    0xe4f012cc98321664,
    0x94d87e48b4c62a00,
    0x0066c4d8e2c6ac20,
    0x9e40fece7aa4bc4e,
    0xac44d8b21a3f7ef2,
    0x3abcf4560c780840,
    0xc20ec8188ac4d69e,
    0x92bc6e7cb04c5436,
    0x3eeeeadce8d87450,
    0x36e8b856a8f864e8,
    0x6c569a14047e96d0,
    0x5024fc446ace1274,
    0x32f406be40d86a7e,
    0x0a9a46127ac89630,
    0xfc08c654703852b4,
    0x68382e1c1a88a0d4,
    0xe0e4f012cc983216,
    0x6494d87e48b4c62a,
    0x2a2a4cc4d8e2c686,
    0x209e40fece7aa4bc,
    0x4eac44d8b21a3f7e,
    0xf23abcf4560c7808,
    0x40c20ec8188ac4d6,
    0x9e92bc6e7cb04c54,
    0x363eeeeadce8d874,
    0x5036e8b856a8f864,
    0xe86c569a14047e96,
    0xd05024fc446ace12,
    0x7432f406be40d86a,
    0x7e0a9a46127ac896,
    0x30fc08c654703852,
    0xb468382e1c1a88a0,
    0xd4e0e4f012cc9832,
    0x166494d87e48b4c6,
    0xc6ecec4cc4d8e200,
    0x86209e40fece7aa4,
    0xbc4eac44d8b21a3f,
    0x7ef23abcf4560c78,
    0x0840c20ec8188ac4,
    0xd69e92bc6e7cb04c,
    0x54363eeeeadce8d8,
    0x745036e8b856a8f8,
    0x64e86c569a14047e,
    0x96d05024fc446ace,
    0x127432f406be40d8,
    0x6a7e0a9a46127ac8,
    0x9630fc08c6547038,
    0x52b468382e1c1a88,
    0xa0d4e0e4f012cc98,
    0x32166494d87e48b4,
    0xb47258ec4cc4d856,
    0x0086209e40fece7a,
    0xa4bc4eac44d8b21a,
    0x3f7ef23abcf4560c,
    0x780840c20ec8188a,
    0xc4d69e92bc6e7cb0,
    0x4c54363eeeeadce8,
    0xd8745036e8b856a8,
    0xf864e86c569a1404,
    0x7e96d05024fc446a,
    0xce127432f406be40,
    0xd86a7e0a9a46127a,
    0xc89630fc08c65470,
    0x3852b468382e1c1a,
    0x88a0d4e0e4f012cc,
    0x9832166494d87e48,
    0x48fc3a58ec4cc490,
    0x560086209e40fece,
    0x7aa4bc4eac44d8b2,
    0x1a3f7ef23abcf456,
    0x0c780840c20ec818,
    0x8ac4d69e92bc6e7c,
    0xb04c54363eeeeadc,
    0xe8d8745036e8b856,
    0xa8f864e86c569a14,
    0x047e96d05024fc44,
    0x6ace127432f406be,
    0x40d86a7e0a9a4612,
    0x7ac89630fc08c654,
    0x703852b468382e1c,
    0x1a88a0d4e0e4f012,
    0xcc9832166494d87e,
    0x7e36823a58ec4cba,
    0x90560086209e40fe,
    0xce7aa4bc4eac44d8,
    0xb21a3f7ef23abcf4,
    0x560c780840c20ec8,
    0x188ac4d69e92bc6e,
    0x7cb04c54363eeeea,
    0xdce8d8745036e8b8,
    0x56a8f864e86c569a,
    0x14047e96d05024fc,
    0x446ace127432f406,
    0xbe40d86a7e0a9a46,
    0x127ac89630fc08c6,
    0x54703852b468382e,
    0x1c1a88a0d4e0e4f0,
    0x12cc9832166494d8,
    0xd8a6ee823a58ec94,
    0xba90560086209e40,
    0xfece7aa4bc4eac44,
    0xd8b21a3f7ef23abc,
    0xf4560c780840c20e,
    0xc8188ac4d69e92bc,
    0x6e7cb04c54363eee,
    0xeadce8d8745036e8,
    0xb856a8f864e86c56,
    0x9a14047e96d05024,
    0xfc446ace127432f4,
    0x06be40d86a7e0a9a,
    0x46127ac89630fc08,
    0xc654703852b46838,
    0x2e1c1a88a0d4e0e4,
    0xf012cc9832166494,
    0x944c32ee823a5878,
    0x94ba90560086209e,
    0x40fece7aa4bc4eac,
    0x44d8b21a3f7ef23a,
    0xbcf4560c780840c2,
    0x0ec8188ac4d69e92,
    0xbc6e7cb04c54363e,
    0xeeeadce8d8745036,
    0xe8b856a8f864e86c,
    0x569a14047e96d050,
    0x24fc446ace127432,
    0xf406be40d86a7e0a,
    0x9a46127ac89630fc,
    0x08c654703852b468,
    0x382e1c1a88a0d4e0,
    0xe4f012cc98321664,
    0x64f02832ee823a3c,
    0x7894ba9056008620,
    0x9e40fece7aa4bc4e,
    0xac44d8b21a3f7ef2,
    0x3abcf4560c780840,
    0xc20ec8188ac4d69e,
    0x92bc6e7cb04c5436,
    0x3eeeeadce8d87450,
    0x36e8b856a8f864e8,
    0x6c569a14047e96d0,
    0x5024fc446ace1274,
    0x32f406be40d86a7e,
    0x0a9a46127ac89630,
    0xfc08c654703852b4,
    0x68382e1c1a88a0d4,
    0xe0e4f012cc983216,
    0x1672e62832ee822c,
    0x3c7894ba90560086,
    0x209e40fece7aa4bc,
    0x4eac44d8b21a3f7e,
    0xf23abcf4560c7808,
    0x40c20ec8188ac4d6,
    0x9e92bc6e7cb04c54,
    0x363eeeeadce8d874,
    0x5036e8b856a8f864,
    0xe86c569a14047e96,
    0xd05024fc446ace12,
    0x7432f406be40d86a,
    0x7e0a9a46127ac896,
    0x30fc08c654703852,
    0xb468382e1c1a88a0,
    0xd4e0e4f012cc9832,
];

#[cfg(test)]
mod tests {
    use super::*;
    fn rng(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn ranked_ab_leaf_stage_matches_original_and_rekeys() {
        let mut seed = 0x32abf0819ad57233u64;
        for bit in 0..128 {
            let x = F128 {
                lo: if bit < 64 { 1u64 << bit } else { 0 },
                hi: if bit >= 64 { 1u64 << (bit - 64) } else { 0 },
            };
            let coefficients = decode(x);
            let mut result = F128::ZERO;
            for (j, &v) in coefficients.iter().enumerate() {
                result += super::super::convert_table()[j * 256 + v as usize];
            }
            assert_eq!(result, x);
        }
        let mut source = vec![0u8; 2 * 2048 * 1856];
        for x in &mut source {
            *x = rng(&mut seed) as u8;
        }
        for block in 0..4096 {
            source[block * 1856 + 1792..(block + 1) * 1856].fill(0xa9);
        }
        let original = source.clone();
        let mut scales: Vec<F128> = (0..32)
            .map(|_| F128 {
                lo: rng(&mut seed),
                hi: rng(&mut seed),
            })
            .collect();
        scales[0] = F128::ZERO;
        scales[1] = F128::ONE;
        let mut mats = super::super::build_ab_eq_fold_mats_gated(
            &scales,
            super::super::convert_table(),
            false,
        );
        let mut cache = None;
        let mut comparisons = 0;
        for hi in 0..2 {
            for mode in 0..3 {
                let mut old = vec![0xceu8; 131072];
                let mut new = vec![0x73u8; 131072];
                unsafe {
                    super::super::process_ranked_ab_w4_band(
                        &source, &mats, hi, true, mode, &mut old,
                    );
                }
                assert!(try_run(&mut cache, &source, &mats, hi, mode, &mut new, 0));
                assert_eq!(new, old, "hi={hi} mode={mode}");
                comparisons += 131072;
            }
        }
        // Reusing the same allocation with changed valid matrices must rebuild.
        scales[0] = F128 {
            lo: rng(&mut seed),
            hi: rng(&mut seed),
        };
        scales[7] = F128 {
            lo: rng(&mut seed),
            hi: rng(&mut seed),
        };
        let changed = super::super::build_ab_eq_fold_mats_gated(
            &scales,
            super::super::convert_table(),
            false,
        );
        mats.copy_from_slice(&changed);
        let mut old = vec![0; 131072];
        let mut new = vec![0; 131072];
        unsafe {
            super::super::process_ranked_ab_w4_band(&source, &mats, 0, true, 1, &mut old);
        }
        assert!(try_run(&mut cache, &source, &mats, 0, 1, &mut new, 0));
        assert_eq!(new, old);
        comparisons += 131072;
        // A map outside the multiplication family must fall back without output writes.
        mats[8191] ^= 1;
        new.fill(0xad);
        assert!(!try_run(&mut cache, &source, &mats, 0, 1, &mut new, 0));
        assert!(new.iter().all(|&v| v == 0xad));
        assert!(!try_run(&mut cache, &source, &mats[..8191], 0, 1, &mut new, 0));
        assert!(!try_run(
            &mut cache,
            &source,
            &changed,
            usize::MAX,
            1,
            &mut new,
            0
        ));
        assert!(!try_run(&mut cache, &source, &changed, 0, 3, &mut new, 0));
        assert_eq!(source, original);
        println!(
            "AB_LEAF_ACCUM_PRODUCTION_VALIDATED banks=896 comparisons={comparisons} basis_bits=128 rekey=true invalid_map_fallback=true"
        );
        assert_eq!(comparisons, 917504);
    }
}
