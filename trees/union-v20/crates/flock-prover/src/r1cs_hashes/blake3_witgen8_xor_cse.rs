//! Exact27-product XOR convolution with shared output recombination.
//! The public coefficient banks and prepared input representation are unchanged.
//! Each v0..v26 is one original subset-convolution product. Later v values
//! share XORs across outputs, including the final superset transform.
use super::*;

#[inline(always)]
pub(super) unsafe fn affine27_depth(p: *const u64, m: *const u64) -> [__m512i; 8] {
    unsafe {
        let x = core::ptr::read_unaligned(p.cast::<[u64; 8]>());
        let x0 = _mm512_set1_epi64(x[0] as i64);
        let x1 = _mm512_set1_epi64(x[1] as i64);
        let x2 = _mm512_set1_epi64(x[2] as i64);
        let x3 = _mm512_set1_epi64(x[3] as i64);
        let x4 = _mm512_set1_epi64(x[4] as i64);
        let x5 = _mm512_set1_epi64(x[5] as i64);
        let x6 = _mm512_set1_epi64(x[6] as i64);
        let x7 = _mm512_set1_epi64(x[7] as i64);
        let v11 = _mm512_gf2p8affine_epi64_epi8::<0>(x1, _mm512_loadu_si512(m.add(112).cast()));
        let v23 = _mm512_gf2p8affine_epi64_epi8::<0>(x5, _mm512_loadu_si512(m.add(80).cast()));
        let v25 = _mm512_gf2p8affine_epi64_epi8::<0>(x6, _mm512_loadu_si512(m.add(72).cast()));
        let v7 = _mm512_gf2p8affine_epi64_epi8::<0>(x0, _mm512_loadu_si512(m.add(120).cast()));
        let v28 = _mm512_ternarylogic_epi64::<0x96>(v7, v23, v25);
        let v17 = _mm512_gf2p8affine_epi64_epi8::<0>(x3, _mm512_loadu_si512(m.add(96).cast()));
        let v21 = _mm512_gf2p8affine_epi64_epi8::<0>(x4, _mm512_loadu_si512(m.add(88).cast()));
        let v26 = _mm512_gf2p8affine_epi64_epi8::<0>(x7, _mm512_loadu_si512(m.add(64).cast()));
        let v27 = _mm512_ternarylogic_epi64::<0x96>(v17, v21, v26);
        let v29 = _mm512_ternarylogic_epi64::<0x96>(v11, v27, v28);
        let v15 = _mm512_gf2p8affine_epi64_epi8::<0>(x2, _mm512_loadu_si512(m.add(104).cast()));
        let v30 = _mm512_xor_si512(v15, v29);
        let v6 = _mm512_gf2p8affine_epi64_epi8::<0>(x0, _mm512_loadu_si512(m.add(112).cast()));
        let v24 = _mm512_gf2p8affine_epi64_epi8::<0>(x6, _mm512_loadu_si512(m.add(64).cast()));
        let v14 = _mm512_gf2p8affine_epi64_epi8::<0>(x2, _mm512_loadu_si512(m.add(96).cast()));
        let v20 = _mm512_gf2p8affine_epi64_epi8::<0>(x4, _mm512_loadu_si512(m.add(80).cast()));
        let v31 = _mm512_ternarylogic_epi64::<0x96>(v14, v20, v24);
        let v37 = _mm512_ternarylogic_epi64::<0x96>(v6, v30, v31);
        let v10 = _mm512_gf2p8affine_epi64_epi8::<0>(x1, _mm512_loadu_si512(m.add(96).cast()));
        let v22 = _mm512_gf2p8affine_epi64_epi8::<0>(x5, _mm512_loadu_si512(m.add(64).cast()));
        let v5 = _mm512_gf2p8affine_epi64_epi8::<0>(x0, _mm512_loadu_si512(m.add(104).cast()));
        let v32 = _mm512_ternarylogic_epi64::<0x96>(v5, v10, v22);
        let v19 = _mm512_gf2p8affine_epi64_epi8::<0>(x4, _mm512_loadu_si512(m.add(72).cast()));
        let v33 = _mm512_ternarylogic_epi64::<0x96>(v19, v30, v32);
        let v18 = _mm512_gf2p8affine_epi64_epi8::<0>(x4, _mm512_loadu_si512(m.add(64).cast()));
        let v4 = _mm512_gf2p8affine_epi64_epi8::<0>(x0, _mm512_loadu_si512(m.add(96).cast()));
        let v35 = _mm512_ternarylogic_epi64::<0x96>(v4, v6, v18);
        let v39 = _mm512_ternarylogic_epi64::<0x96>(v31, v33, v35);
        let v12 = _mm512_gf2p8affine_epi64_epi8::<0>(x2, _mm512_loadu_si512(m.add(64).cast()));
        let v9 = _mm512_gf2p8affine_epi64_epi8::<0>(x1, _mm512_loadu_si512(m.add(80).cast()));
        let v13 = _mm512_gf2p8affine_epi64_epi8::<0>(x2, _mm512_loadu_si512(m.add(72).cast()));
        let v3 = _mm512_gf2p8affine_epi64_epi8::<0>(x0, _mm512_loadu_si512(m.add(88).cast()));
        let v34 = _mm512_ternarylogic_epi64::<0x96>(v3, v9, v13);
        let v2 = _mm512_gf2p8affine_epi64_epi8::<0>(x0, _mm512_loadu_si512(m.add(80).cast()));
        let v38 = _mm512_ternarylogic_epi64::<0x96>(v2, v12, v34);
        let v8 = _mm512_gf2p8affine_epi64_epi8::<0>(x1, _mm512_loadu_si512(m.add(64).cast()));
        let v16 = _mm512_gf2p8affine_epi64_epi8::<0>(x3, _mm512_loadu_si512(m.add(64).cast()));
        let v1 = _mm512_gf2p8affine_epi64_epi8::<0>(x0, _mm512_loadu_si512(m.add(72).cast()));
        let v36 = _mm512_ternarylogic_epi64::<0x96>(v1, v8, v16);
        let v0 = _mm512_gf2p8affine_epi64_epi8::<0>(x0, _mm512_loadu_si512(m.add(64).cast()));
        let v40 = _mm512_ternarylogic_epi64::<0x96>(v0, v36, v38);
        let v41 = _mm512_xor_si512(v40, v39);
        let v43 = _mm512_ternarylogic_epi64::<0x96>(v16, v37, v38);
        let v44 = _mm512_ternarylogic_epi64::<0x96>(v16, v30, v34);
        let v42 = _mm512_ternarylogic_epi64::<0x96>(v33, v34, v36);
        [v41, v42, v43, v44, v39, v33, v37, v30]
    }
}
