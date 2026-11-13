.text

.p2align 6
.globl flock_partial_gpr_free7_first
.hidden flock_partial_gpr_free7_first
.type flock_partial_gpr_free7_first,@function
flock_partial_gpr_free7_first:
vpbroadcastq zmm0,QWORD PTR [rdi+0x30]
vpbroadcastq zmm1,QWORD PTR [rdi+0x18]
vpbroadcastq zmm2,QWORD PTR [rdi+0x8]
vpbroadcastq zmm3,QWORD PTR [rdi+0x28]
vpbroadcastq zmm4,QWORD PTR [rdi+0x38]
vpbroadcastq zmm5,QWORD PTR [rdi+0x20]
vmovdqu64 zmm6,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm4,zmm4,zmm6,0x0
vpbroadcastq zmm7,QWORD PTR [rdi+0x10]
vpbroadcastq zmm8,QWORD PTR [rdi]
vmovdqa64 zmm9,zmm2
vpternlogq zmm9,zmm7,zmm5,0x96
vgf2p8affineqb zmm10,zmm9,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm9,zmm9,zmm6,0x0
vgf2p8affineqb zmm11,zmm8,ZMMWORD PTR [rdx+0x6c0],0x0
vgf2p8affineqb zmm12,zmm2,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm12,zmm4,zmm10,0x96
vgf2p8affineqb zmm4,zmm3,zmm6,0x0
vpternlogq zmm3,zmm1,zmm0,0x96
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm6,zmm8,zmm6,0x0
vgf2p8affineqb zmm10,zmm8,ZMMWORD PTR [rdx+0x540],0x0
vpternlogq zmm12,zmm10,zmm3,0x96
vpternlogq zmm11,zmm4,zmm12,0x96
vgf2p8affineqb zmm3,zmm7,ZMMWORD PTR [rdx+0x480],0x0
vpxorq zmm4,zmm5,zmm8
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm10,zmm0,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm0,zmm5,zmm2,0x96
vgf2p8affineqb zmm5,zmm5,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x840],0x0
vpxorq zmm13,zmm7,zmm8
vpternlogq zmm7,zmm2,zmm1,0x96
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x640],0x0
vgf2p8affineqb zmm13,zmm13,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm11,zmm1,zmm3,0x96
vpternlogq zmm13,zmm1,zmm12,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x5c0],0x0
vpternlogq zmm7,zmm3,zmm12,0x96
vgf2p8affineqb zmm1,zmm2,ZMMWORD PTR [rdx+0x4c0],0x0
vpxorq zmm2,zmm2,zmm8
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm3,zmm8,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x7c0],0x0
mov rax,QWORD PTR [rsi+0x38]
not rax
vmovdqa64 zmm12,zmm5
vpternlogq zmm12,zmm3,zmm8,0x96
vpternlogq zmm2,zmm0,zmm12,0x96
vpternlogq zmm12,zmm10,zmm1,0x96
vpternlogq zmm10,zmm9,zmm4,0x96
vpternlogq zmm5,zmm10,zmm11,0x96
vpternlogq zmm6,zmm2,zmm11,0x96
vpternlogq zmm2,zmm4,zmm7,0x96
vpternlogq zmm3,zmm10,zmm7,0x96
vpternlogq zmm9,zmm0,zmm13,0x96
vpternlogq zmm7,zmm13,zmm5,0x96
vpternlogq zmm0,zmm7,zmm10,0x96
vpternlogq zmm1,zmm8,zmm7,0x96
vpxorq zmm12,zmm12,zmm13
vpbroadcastq zmm4,rax
vgf2p8affineqb zmm7,zmm4,ZMMWORD PTR [rcx+0x40],0x1
vgf2p8mulb zmm30,zmm9,zmm7
vgf2p8affineqb zmm7,zmm4,ZMMWORD PTR [rcx+0x180],0x1
vgf2p8mulb zmm25,zmm5,zmm7
vgf2p8affineqb zmm5,zmm4,ZMMWORD PTR [rcx+0xc0],0x1
vgf2p8mulb zmm28,zmm6,zmm5
vgf2p8affineqb zmm5,zmm4,ZMMWORD PTR [rcx+0x80],0x1
vgf2p8mulb zmm29,zmm1,zmm5
vgf2p8affineqb zmm1,zmm4,ZMMWORD PTR [rcx+0x140],0x1
vgf2p8mulb zmm26,zmm2,zmm1
vgf2p8affineqb zmm1,zmm4,ZMMWORD PTR [rcx+0x100],0x1
vgf2p8mulb zmm27,zmm12,zmm1
vgf2p8affineqb zmm1,zmm4,ZMMWORD PTR [rcx+0x1c0],0x1
vgf2p8mulb zmm24,zmm0,zmm1
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rcx],0x1
vgf2p8mulb zmm31,zmm3,zmm4
ret
.size flock_partial_gpr_free7_first,.-flock_partial_gpr_free7_first

.p2align 6
.globl flock_partial_gpr_free7_next
.hidden flock_partial_gpr_free7_next
.type flock_partial_gpr_free7_next,@function
flock_partial_gpr_free7_next:
vpbroadcastq zmm0,QWORD PTR [rdi+0x30]
vpbroadcastq zmm1,QWORD PTR [rdi+0x18]
vpbroadcastq zmm2,QWORD PTR [rdi+0x8]
vpbroadcastq zmm3,QWORD PTR [rdi+0x28]
vpbroadcastq zmm4,QWORD PTR [rdi+0x38]
vpbroadcastq zmm5,QWORD PTR [rdi+0x20]
vmovdqu64 zmm6,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm4,zmm4,zmm6,0x0
vpbroadcastq zmm7,QWORD PTR [rdi+0x10]
vpbroadcastq zmm8,QWORD PTR [rdi]
vmovdqa64 zmm9,zmm2
vpternlogq zmm9,zmm7,zmm5,0x96
vgf2p8affineqb zmm10,zmm9,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm9,zmm9,zmm6,0x0
vgf2p8affineqb zmm11,zmm8,ZMMWORD PTR [rdx+0x6c0],0x0
vgf2p8affineqb zmm12,zmm2,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm12,zmm4,zmm10,0x96
vgf2p8affineqb zmm4,zmm3,zmm6,0x0
vpternlogq zmm3,zmm1,zmm0,0x96
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm6,zmm8,zmm6,0x0
vgf2p8affineqb zmm10,zmm8,ZMMWORD PTR [rdx+0x540],0x0
vpternlogq zmm12,zmm10,zmm3,0x96
vpternlogq zmm11,zmm4,zmm12,0x96
vgf2p8affineqb zmm3,zmm7,ZMMWORD PTR [rdx+0x480],0x0
vpxorq zmm4,zmm5,zmm8
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm10,zmm0,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm0,zmm5,zmm2,0x96
vgf2p8affineqb zmm5,zmm5,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x840],0x0
vpxorq zmm13,zmm7,zmm8
vpternlogq zmm7,zmm2,zmm1,0x96
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x640],0x0
vgf2p8affineqb zmm13,zmm13,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm11,zmm1,zmm3,0x96
vpternlogq zmm13,zmm1,zmm12,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x5c0],0x0
vpternlogq zmm7,zmm3,zmm12,0x96
vgf2p8affineqb zmm1,zmm2,ZMMWORD PTR [rdx+0x4c0],0x0
vpxorq zmm2,zmm2,zmm8
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm3,zmm8,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x7c0],0x0
mov rax,QWORD PTR [rsi+0x38]
not rax
vmovdqa64 zmm12,zmm5
vpternlogq zmm12,zmm3,zmm8,0x96
vpternlogq zmm2,zmm0,zmm12,0x96
vpternlogq zmm12,zmm10,zmm1,0x96
vpternlogq zmm10,zmm9,zmm4,0x96
vpternlogq zmm5,zmm10,zmm11,0x96
vpternlogq zmm6,zmm2,zmm11,0x96
vpternlogq zmm2,zmm4,zmm7,0x96
vpternlogq zmm3,zmm10,zmm7,0x96
vpternlogq zmm9,zmm0,zmm13,0x96
vpternlogq zmm7,zmm13,zmm5,0x96
vpternlogq zmm0,zmm7,zmm10,0x96
vpternlogq zmm1,zmm8,zmm7,0x96
vpxorq zmm12,zmm12,zmm13
vpbroadcastq zmm4,rax
vgf2p8affineqb zmm7,zmm4,ZMMWORD PTR [rcx+0x40],0x1
vgf2p8mulb zmm9,zmm9,zmm7
vpxorq zmm30,zmm30,zmm9
vgf2p8affineqb zmm7,zmm4,ZMMWORD PTR [rcx+0x180],0x1
vgf2p8mulb zmm5,zmm5,zmm7
vpxorq zmm25,zmm25,zmm5
vgf2p8affineqb zmm5,zmm4,ZMMWORD PTR [rcx+0xc0],0x1
vgf2p8mulb zmm6,zmm6,zmm5
vpxorq zmm28,zmm28,zmm6
vgf2p8affineqb zmm5,zmm4,ZMMWORD PTR [rcx+0x80],0x1
vgf2p8mulb zmm1,zmm1,zmm5
vpxorq zmm29,zmm29,zmm1
vgf2p8affineqb zmm1,zmm4,ZMMWORD PTR [rcx+0x140],0x1
vgf2p8mulb zmm2,zmm2,zmm1
vpxorq zmm26,zmm26,zmm2
vgf2p8affineqb zmm1,zmm4,ZMMWORD PTR [rcx+0x100],0x1
vgf2p8mulb zmm12,zmm12,zmm1
vpxorq zmm27,zmm27,zmm12
vgf2p8affineqb zmm1,zmm4,ZMMWORD PTR [rcx+0x1c0],0x1
vgf2p8mulb zmm0,zmm0,zmm1
vpxorq zmm24,zmm24,zmm0
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rcx],0x1
vgf2p8mulb zmm3,zmm3,zmm4
vpxorq zmm31,zmm31,zmm3
ret
.size flock_partial_gpr_free7_next,.-flock_partial_gpr_free7_next

.section .note.GNU-stack,"",@progbits
