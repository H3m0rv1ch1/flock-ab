//! Whole partial-B row with register-carried accumulators.
//! Exact input-pointer and scratch-register clobbers are declared at both call sites.
//! FIRST initializes all eight output vectors; matrices and byte inputs are read-only.
use super::*;
core::arch::global_asm!(include_str!("blake3_witgen8_xor_partial11.S"),options(raw));
unsafe extern "C"{fn flock_partial_gpr_free11_first();fn flock_partial_gpr_free11_next();}
#[inline(always)]
pub(super)unsafe fn run<const FIRST:bool>(ap:*const u64,bp:*const u64,ma:*const u64,mb:*const u64,acc:[__m512i;8])->[__m512i;8]{unsafe{
let[mut a0,mut a1,mut a2,mut a3,mut a4,mut a5,mut a6,mut a7]=acc;
if FIRST{
core::arch::asm!("call {kernel}",kernel=sym flock_partial_gpr_free11_first,
in("rdi")ap,
inout("rsi")bp=>_,
in("rdx")ma,
in("rcx")mb,
lateout("r10")_,
lateout("r9")_,
lateout("rax")_,
out("zmm24")a0,
out("zmm25")a1,
out("zmm26")a2,
out("zmm27")a3,
out("zmm28")a4,
out("zmm29")a5,
out("zmm30")a6,
out("zmm31")a7,
out("zmm0")_,
out("zmm1")_,
out("zmm2")_,
out("zmm3")_,
out("zmm4")_,
out("zmm5")_,
out("zmm6")_,
out("zmm7")_,
out("zmm8")_,
out("zmm9")_,
out("zmm10")_,
out("zmm11")_,
out("zmm12")_,
out("zmm13")_,
out("zmm14")_,
out("zmm15")_,
out("zmm16")_,
out("zmm17")_,
options(pure,readonly));
}else{
core::arch::asm!("call {kernel}",kernel=sym flock_partial_gpr_free11_next,
in("rdi")ap,
inout("rsi")bp=>_,
in("rdx")ma,
in("rcx")mb,
lateout("r10")_,
lateout("r9")_,
lateout("rax")_,
inout("zmm24")a0,
inout("zmm25")a1,
inout("zmm26")a2,
inout("zmm27")a3,
inout("zmm28")a4,
inout("zmm29")a5,
inout("zmm30")a6,
inout("zmm31")a7,
out("zmm0")_,
out("zmm1")_,
out("zmm2")_,
out("zmm3")_,
out("zmm4")_,
out("zmm5")_,
out("zmm6")_,
out("zmm7")_,
out("zmm8")_,
out("zmm9")_,
out("zmm10")_,
out("zmm11")_,
out("zmm12")_,
out("zmm13")_,
out("zmm14")_,
out("zmm15")_,
out("zmm16")_,
out("zmm17")_,
options(pure,readonly));
}
[a0,a1,a2,a3,a4,a5,a6,a7]
}}
