.text

.p2align 6
.globl flock_partial_gpr_free6_first
.hidden flock_partial_gpr_free6_first
.type flock_partial_gpr_free6_first,@function
flock_partial_gpr_free6_first:
vpbroadcastq zmm0,QWORD PTR [rdi+0x10]
vpbroadcastq zmm1,QWORD PTR [rdi+0x20]
vpbroadcastq zmm2,QWORD PTR [rdi+0x8]
vpbroadcastq zmm3,QWORD PTR [rdi+0x38]
vpbroadcastq zmm4,QWORD PTR [rdi+0x18]
vmovdqu64 zmm5,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm3,zmm3,zmm5,0x0
vpbroadcastq zmm6,QWORD PTR [rdi+0x28]
vpbroadcastq zmm7,QWORD PTR [rdi+0x30]
vmovdqa64 zmm8,zmm2
vpternlogq zmm8,zmm0,zmm1,0x96
vgf2p8affineqb zmm9,zmm8,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm8,zmm8,zmm5,0x0
vpbroadcastq zmm10,QWORD PTR [rdi]
vmovdqa64 zmm11,zmm4
vpternlogq zmm11,zmm6,zmm7,0x96
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm6,zmm6,zmm5,0x0
vgf2p8affineqb zmm5,zmm10,zmm5,0x0
vgf2p8affineqb zmm12,zmm2,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm12,zmm3,zmm9,0x96
vgf2p8affineqb zmm3,zmm10,ZMMWORD PTR [rdx+0x6c0],0x0
vpxorq zmm9,zmm1,zmm10
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm13,zmm10,ZMMWORD PTR [rdx+0x540],0x0
vpternlogq zmm12,zmm13,zmm11,0x96
vpternlogq zmm3,zmm6,zmm12,0x96
vgf2p8affineqb zmm6,zmm4,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm4,zmm2,zmm0,0x96
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x5c0],0x0
vpxorq zmm11,zmm0,zmm10
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x480],0x0
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm4,zmm0,zmm12,0x96
vpternlogq zmm3,zmm6,zmm0,0x96
vpternlogq zmm11,zmm6,zmm12,0x96
vgf2p8affineqb zmm0,zmm1,ZMMWORD PTR [rdx+0x580],0x0
vpternlogq zmm1,zmm2,zmm7,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x680],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm6,zmm10,ZMMWORD PTR [rdx+0x7c0],0x0
vgf2p8affineqb zmm12,zmm10,ZMMWORD PTR [rdx+0x600],0x0
vpxorq zmm10,zmm2,zmm10
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm13,zmm0
vpternlogq zmm13,zmm12,zmm6,0x96
vpternlogq zmm10,zmm1,zmm13,0x96
vpternlogq zmm13,zmm7,zmm2,0x96
vpternlogq zmm7,zmm8,zmm9,0x96
vpternlogq zmm0,zmm7,zmm3,0x96
vpternlogq zmm5,zmm10,zmm3,0x96
vpternlogq zmm10,zmm9,zmm4,0x96
vpternlogq zmm12,zmm7,zmm4,0x96
vpternlogq zmm4,zmm11,zmm0,0x96
vpternlogq zmm2,zmm6,zmm4,0x96
vpternlogq zmm4,zmm1,zmm7,0x96
vpternlogq zmm8,zmm1,zmm11,0x96
vpxorq zmm13,zmm13,zmm11
mov rax,QWORD PTR [rsi+0x38]
mov r9,rax
xor rax,QWORD PTR [rsi+0x30]
not r9
vpbroadcastq zmm1,r9
vpbroadcastq zmm3,rax
vmovdqu64 zmm6,ZMMWORD PTR [rcx+0x380]
vgf2p8affineqb zmm7,zmm1,zmm6,0x0
vgf2p8affineqb zmm6,zmm3,zmm6,0x0
vgf2p8affineqb zmm9,zmm3,ZMMWORD PTR [rcx+0x3c0],0x1
vpxorq zmm7,zmm7,zmm9
vpxorq zmm6,zmm7,zmm6
vgf2p8mulb zmm24,zmm4,zmm6
vgf2p8mulb zmm25,zmm0,zmm7
vmovdqu64 zmm0,ZMMWORD PTR [rcx+0x300]
vmovdqu64 zmm4,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm9,zmm3,ZMMWORD PTR [rcx+0x340],0x0
vgf2p8affineqb zmm11,zmm1,zmm0,0x0
vpxorq zmm11,zmm11,zmm9
vgf2p8affineqb zmm0,zmm3,zmm0,0x0
vpternlogq zmm0,zmm11,zmm6,0x96
vpxorq zmm11,zmm7,zmm11
vgf2p8mulb zmm26,zmm10,zmm0
vgf2p8mulb zmm27,zmm13,zmm11
vmovdqu64 zmm9,ZMMWORD PTR [rcx+0x280]
vgf2p8affineqb zmm10,zmm1,zmm4,0x0
vgf2p8affineqb zmm1,zmm1,zmm9,0x0
vgf2p8affineqb zmm9,zmm3,zmm9,0x0
vgf2p8affineqb zmm4,zmm3,zmm4,0x0
vgf2p8affineqb zmm13,zmm3,ZMMWORD PTR [rcx+0x2c0],0x0
vpxorq zmm1,zmm1,zmm13
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rcx+0x240],0x0
vpxorq zmm10,zmm10,zmm3
vpxorq zmm4,zmm10,zmm4
vpternlogq zmm10,zmm1,zmm11,0x96
vgf2p8mulb zmm31,zmm12,zmm10
vpxorq zmm9,zmm1,zmm9
vpxorq zmm7,zmm7,zmm1
vpternlogq zmm4,zmm0,zmm9,0x96
vpxorq zmm6,zmm6,zmm9
vgf2p8mulb zmm28,zmm5,zmm6
vgf2p8mulb zmm30,zmm8,zmm4
vgf2p8mulb zmm29,zmm2,zmm7
ret
.size flock_partial_gpr_free6_first,.-flock_partial_gpr_free6_first

.p2align 6
.globl flock_partial_gpr_free6_next
.hidden flock_partial_gpr_free6_next
.type flock_partial_gpr_free6_next,@function
flock_partial_gpr_free6_next:
vpbroadcastq zmm0,QWORD PTR [rdi+0x10]
vpbroadcastq zmm1,QWORD PTR [rdi+0x20]
vpbroadcastq zmm2,QWORD PTR [rdi+0x8]
vpbroadcastq zmm3,QWORD PTR [rdi+0x38]
vpbroadcastq zmm4,QWORD PTR [rdi+0x18]
vmovdqu64 zmm5,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm3,zmm3,zmm5,0x0
vpbroadcastq zmm6,QWORD PTR [rdi+0x28]
vpbroadcastq zmm7,QWORD PTR [rdi+0x30]
vmovdqa64 zmm8,zmm2
vpternlogq zmm8,zmm0,zmm1,0x96
vgf2p8affineqb zmm9,zmm8,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm8,zmm8,zmm5,0x0
vpbroadcastq zmm10,QWORD PTR [rdi]
vmovdqa64 zmm11,zmm4
vpternlogq zmm11,zmm6,zmm7,0x96
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm6,zmm6,zmm5,0x0
vgf2p8affineqb zmm5,zmm10,zmm5,0x0
vgf2p8affineqb zmm12,zmm2,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm12,zmm3,zmm9,0x96
vgf2p8affineqb zmm3,zmm10,ZMMWORD PTR [rdx+0x6c0],0x0
vpxorq zmm9,zmm1,zmm10
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm13,zmm10,ZMMWORD PTR [rdx+0x540],0x0
vpternlogq zmm12,zmm13,zmm11,0x96
vpternlogq zmm3,zmm6,zmm12,0x96
vgf2p8affineqb zmm6,zmm4,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm4,zmm2,zmm0,0x96
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x5c0],0x0
vpxorq zmm11,zmm0,zmm10
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x480],0x0
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm4,zmm0,zmm12,0x96
vpternlogq zmm3,zmm6,zmm0,0x96
vpternlogq zmm11,zmm6,zmm12,0x96
vgf2p8affineqb zmm0,zmm1,ZMMWORD PTR [rdx+0x580],0x0
vpternlogq zmm1,zmm2,zmm7,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x680],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm6,zmm10,ZMMWORD PTR [rdx+0x7c0],0x0
vgf2p8affineqb zmm12,zmm10,ZMMWORD PTR [rdx+0x600],0x0
vpxorq zmm10,zmm2,zmm10
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm13,zmm0
vpternlogq zmm13,zmm12,zmm6,0x96
vpternlogq zmm10,zmm1,zmm13,0x96
vpternlogq zmm13,zmm7,zmm2,0x96
vpternlogq zmm7,zmm8,zmm9,0x96
vpternlogq zmm0,zmm7,zmm3,0x96
vpternlogq zmm5,zmm10,zmm3,0x96
vpternlogq zmm10,zmm9,zmm4,0x96
vpternlogq zmm12,zmm7,zmm4,0x96
vpternlogq zmm4,zmm11,zmm0,0x96
vpternlogq zmm2,zmm6,zmm4,0x96
vpternlogq zmm4,zmm1,zmm7,0x96
vpternlogq zmm8,zmm1,zmm11,0x96
vpxorq zmm13,zmm13,zmm11
mov rax,QWORD PTR [rsi+0x38]
mov r9,rax
xor rax,QWORD PTR [rsi+0x30]
not r9
vpbroadcastq zmm1,r9
vpbroadcastq zmm3,rax
vmovdqu64 zmm6,ZMMWORD PTR [rcx+0x380]
vgf2p8affineqb zmm7,zmm1,zmm6,0x0
vgf2p8affineqb zmm6,zmm3,zmm6,0x0
vgf2p8affineqb zmm9,zmm3,ZMMWORD PTR [rcx+0x3c0],0x1
vpxorq zmm7,zmm7,zmm9
vpxorq zmm6,zmm7,zmm6
vgf2p8mulb zmm4,zmm4,zmm6
vpxorq zmm24,zmm24,zmm4
vgf2p8mulb zmm0,zmm0,zmm7
vpxorq zmm25,zmm25,zmm0
vmovdqu64 zmm0,ZMMWORD PTR [rcx+0x300]
vmovdqu64 zmm4,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm9,zmm3,ZMMWORD PTR [rcx+0x340],0x0
vgf2p8affineqb zmm11,zmm1,zmm0,0x0
vpxorq zmm11,zmm11,zmm9
vgf2p8affineqb zmm0,zmm3,zmm0,0x0
vpternlogq zmm0,zmm11,zmm6,0x96
vpxorq zmm11,zmm7,zmm11
vgf2p8mulb zmm10,zmm10,zmm0
vpxorq zmm26,zmm26,zmm10
vgf2p8mulb zmm13,zmm13,zmm11
vpxorq zmm27,zmm27,zmm13
vmovdqu64 zmm9,ZMMWORD PTR [rcx+0x280]
vgf2p8affineqb zmm10,zmm1,zmm4,0x0
vgf2p8affineqb zmm1,zmm1,zmm9,0x0
vgf2p8affineqb zmm9,zmm3,zmm9,0x0
vgf2p8affineqb zmm4,zmm3,zmm4,0x0
vgf2p8affineqb zmm13,zmm3,ZMMWORD PTR [rcx+0x2c0],0x0
vpxorq zmm1,zmm1,zmm13
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rcx+0x240],0x0
vpxorq zmm10,zmm10,zmm3
vpxorq zmm4,zmm10,zmm4
vpternlogq zmm10,zmm1,zmm11,0x96
vgf2p8mulb zmm12,zmm12,zmm10
vpxorq zmm31,zmm31,zmm12
vpxorq zmm9,zmm1,zmm9
vpxorq zmm7,zmm7,zmm1
vpternlogq zmm4,zmm0,zmm9,0x96
vpxorq zmm6,zmm6,zmm9
vgf2p8mulb zmm5,zmm5,zmm6
vpxorq zmm28,zmm28,zmm5
vgf2p8mulb zmm8,zmm8,zmm4
vpxorq zmm30,zmm30,zmm8
vgf2p8mulb zmm2,zmm2,zmm7
vpxorq zmm29,zmm29,zmm2
ret
.size flock_partial_gpr_free6_next,.-flock_partial_gpr_free6_next

.section .note.GNU-stack,"",@progbits
