//! Rank21 identity inherited from 926f9faf, with a new fixed-input basis.
//! Equal-input elementary transformations B_i ^= B_j; C_j ^= C_i preserve
//! the exact bilinear tensor. Eighteen distinct coefficient masks use the
//! existing nineteen-slot layout; the spare slot is zero. The output XOR
//! circuit uses nineteen binary/ternary instructions. The private register
//! kernel must always be regenerated together with this coefficient layout.
use super::*;
#[inline(always)]
pub(super) unsafe fn prepare_network(m:*const u64)->[[u64;8];19] {unsafe {
let masks=[67, 4, 81, 17, 12, 191, 3, 23, 21, 22, 6, 63, 2, 1, 0, 75, 111, 7, 16];
core::array::from_fn(|k| core::array::from_fn(|c| {let mut v=0;for i in 0..8 {if masks[k]&(1<<i)!=0 {v^=*m.add(64+i*8+c);}} v}))
}}
#[inline(always)]
pub(super) unsafe fn affine21_network(p:*const u64,m:*const u64)->[__m512i;8] {unsafe {
let x=core::ptr::read_unaligned(p.cast::<[u64;8]>());
let x0=_mm512_set1_epi64(x[0] as i64);
let x1=_mm512_set1_epi64(x[1] as i64);
let x2=_mm512_set1_epi64(x[2] as i64);
let x3=_mm512_set1_epi64(x[3] as i64);
let x4=_mm512_set1_epi64(x[4] as i64);
let x5=_mm512_set1_epi64(x[5] as i64);
let x6=_mm512_set1_epi64(x[6] as i64);
let x7=_mm512_set1_epi64(x[7] as i64);
let v19=_mm512_gf2p8affine_epi64_epi8::<0>(_mm512_ternarylogic_epi64::<0x96>(x1,x4,x6),_mm512_loadu_si512(m.add(136).cast()));
let v6=_mm512_gf2p8affine_epi64_epi8::<0>(x4,_mm512_loadu_si512(m.add(48).cast()));
let v15=_mm512_gf2p8affine_epi64_epi8::<0>(_mm512_ternarylogic_epi64::<0x96>(x1,x2,x4),_mm512_loadu_si512(m.add(104).cast()));
let v14=_mm512_gf2p8affine_epi64_epi8::<0>(x0,_mm512_loadu_si512(m.add(40).cast()));
let v17=_mm512_gf2p8affine_epi64_epi8::<0>(x7,_mm512_loadu_si512(m.add(104).cast()));
let v1=_mm512_gf2p8affine_epi64_epi8::<0>(_mm512_ternarylogic_epi64::<0x96>(x3,x5,x6),_mm512_loadu_si512(m.add(8).cast()));
let v8=_mm512_gf2p8affine_epi64_epi8::<0>(_mm512_ternarylogic_epi64::<0x96>(x1,x2,x4),_mm512_loadu_si512(m.add(32).cast()));
let v18=_mm512_gf2p8affine_epi64_epi8::<0>(x1,_mm512_loadu_si512(m.add(24).cast()));
let v0=_mm512_gf2p8affine_epi64_epi8::<0>(_mm512_xor_si512(x0,x2),_mm512_loadu_si512(m.add(0).cast()));
let v7=_mm512_gf2p8affine_epi64_epi8::<0>(_mm512_ternarylogic_epi64::<0x96>(x1,x2,x3),_mm512_loadu_si512(m.add(56).cast()));
let v13=_mm512_gf2p8affine_epi64_epi8::<0>(x5,_mm512_loadu_si512(m.add(104).cast()));
let v10=_mm512_gf2p8affine_epi64_epi8::<0>(x6,_mm512_loadu_si512(m.add(80).cast()));
let v12=_mm512_gf2p8affine_epi64_epi8::<0>(_mm512_xor_si512(x0,x4),_mm512_loadu_si512(m.add(96).cast()));
let v2=_mm512_gf2p8affine_epi64_epi8::<0>(x2,_mm512_loadu_si512(m.add(16).cast()));
let v9=_mm512_gf2p8affine_epi64_epi8::<0>(x3,_mm512_loadu_si512(m.add(72).cast()));
let v11=_mm512_gf2p8affine_epi64_epi8::<0>(x0,_mm512_loadu_si512(m.add(104).cast()));
let v20=_mm512_gf2p8affine_epi64_epi8::<0>(_mm512_xor_si512(x0,x1),_mm512_loadu_si512(m.add(144).cast()));
let v3=_mm512_gf2p8affine_epi64_epi8::<0>(x1,_mm512_loadu_si512(m.add(128).cast()));
let v4=_mm512_gf2p8affine_epi64_epi8::<0>(x0,_mm512_loadu_si512(m.add(64).cast()));
let v16=_mm512_gf2p8affine_epi64_epi8::<0>(x0,_mm512_loadu_si512(m.add(120).cast()));
let v5=_mm512_gf2p8affine_epi64_epi8::<0>(x0,_mm512_loadu_si512(m.add(88).cast()));
let q0=_mm512_ternarylogic_epi64::<0x96>(v17,v8,v3);
let q1=_mm512_ternarylogic_epi64::<0x96>(v14,v1,q0);
let q2=_mm512_ternarylogic_epi64::<0x96>(v13,v5,q1);
let q3=_mm512_ternarylogic_epi64::<0x96>(v15,v10,v12);
let q4=_mm512_ternarylogic_epi64::<0x96>(v6,v4,v16);
let q5=_mm512_ternarylogic_epi64::<0x96>(v19,v20,q4);
let q6=_mm512_ternarylogic_epi64::<0x96>(v7,v2,q1);
let q7=_mm512_ternarylogic_epi64::<0x96>(v0,v9,q1);
let q8=_mm512_ternarylogic_epi64::<0x96>(v2,v9,q2);
let q9=_mm512_ternarylogic_epi64::<0x96>(v6,q3,q8);
let q10=_mm512_ternarylogic_epi64::<0x96>(v12,q5,q6);
let q11=_mm512_ternarylogic_epi64::<0x96>(v18,v10,q4);
let q12=_mm512_xor_si512(q7,q11);
let q13=_mm512_ternarylogic_epi64::<0x96>(v11,q5,q8);
let q14=_mm512_ternarylogic_epi64::<0x96>(v19,v15,q7);
let q15=_mm512_ternarylogic_epi64::<0x96>(v4,q3,q6);
let q16=_mm512_ternarylogic_epi64::<0x96>(q6,q7,q9);
let q17=_mm512_ternarylogic_epi64::<0x96>(v19,q3,q16);
let q18=_mm512_ternarylogic_epi64::<0x96>(v18,v16,q16);
[q17,q9,q10,q12,q13,q18,q14,q15]
}}
