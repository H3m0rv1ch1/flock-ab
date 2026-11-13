.text

.p2align 6
.globl flock_partial_gpr_free12_first
.hidden flock_partial_gpr_free12_first
.type flock_partial_gpr_free12_first,@function
flock_partial_gpr_free12_first:
vpbroadcastq zmm0,QWORD PTR [rdi+0x20]
vpbroadcastq zmm1,QWORD PTR [rdi+0x8]
vpbroadcastq zmm2,QWORD PTR [rdi+0x30]
vmovdqu64 zmm3,ZMMWORD PTR [rdx+0x740]
vpbroadcastq zmm4,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm4,zmm4,zmm3,0x0
vpbroadcastq zmm5,QWORD PTR [rdi+0x10]
vpbroadcastq zmm6,QWORD PTR [rdi+0x18]
vpbroadcastq zmm7,QWORD PTR [rdi+0x28]
vpbroadcastq zmm8,QWORD PTR [rdi]
vmovdqa64 zmm9,zmm1
vpternlogq zmm9,zmm5,zmm0,0x96
vgf2p8affineqb zmm10,zmm8,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm11,zmm9,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm9,zmm9,zmm3,0x0
vgf2p8affineqb zmm12,zmm5,ZMMWORD PTR [rdx+0x480],0x0
vgf2p8affineqb zmm13,zmm1,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm13,zmm4,zmm11,0x96
vgf2p8affineqb zmm4,zmm6,ZMMWORD PTR [rdx+0x640],0x0
vpxorq zmm11,zmm5,zmm8
vpternlogq zmm5,zmm1,zmm6,0x96
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm6,zmm7,zmm2,0x96
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rdx+0x440],0x0
vpternlogq zmm13,zmm10,zmm6,0x96
vgf2p8affineqb zmm5,zmm5,ZMMWORD PTR [rdx+0x5c0],0x0
vgf2p8affineqb zmm7,zmm7,zmm3,0x0
vpternlogq zmm11,zmm4,zmm13,0x96
vpternlogq zmm5,zmm12,zmm13,0x96
vgf2p8affineqb zmm3,zmm8,zmm3,0x0
vpxorq zmm6,zmm0,zmm8
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm10,zmm8,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm14,zmm0,ZMMWORD PTR [rdx+0x580],0x0
vpternlogq zmm0,zmm1,zmm2,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x680],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm15,zmm8,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm15,zmm7,zmm13,0x96
vpternlogq zmm15,zmm4,zmm12,0x96
vpxorq zmm4,zmm1,zmm8
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x7c0],0x0
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm7,zmm14
vpternlogq zmm7,zmm10,zmm8,0x96
vpternlogq zmm4,zmm0,zmm7,0x96
vpternlogq zmm7,zmm2,zmm1,0x96
vpternlogq zmm2,zmm9,zmm6,0x96
vpternlogq zmm10,zmm2,zmm5,0x96
vpternlogq zmm6,zmm4,zmm5,0x96
vpternlogq zmm3,zmm4,zmm15,0x96
vpternlogq zmm14,zmm2,zmm15,0x96
vpternlogq zmm5,zmm11,zmm14,0x96
vpternlogq zmm1,zmm8,zmm5,0x96
vpternlogq zmm5,zmm0,zmm2,0x96
vpternlogq zmm9,zmm0,zmm11,0x96
vpxorq zmm7,zmm7,zmm11
mov rax,QWORD PTR [rsi+0x8]
mov r10,rax
xor rax,QWORD PTR [rsi]
mov r9,QWORD PTR [rsi+0x10]
not r9
not r10
xor rax,r9
vpbroadcastq zmm0,r9
vpbroadcastq zmm2,r10
vmovdqu64 zmm4,ZMMWORD PTR [rcx+0x340]
vmovdqu64 zmm8,ZMMWORD PTR [rcx+0x280]
vpbroadcastq zmm11,rax
vmovdqu64 zmm12,ZMMWORD PTR [rcx+0x380]
vgf2p8affineqb zmm13,zmm11,ZMMWORD PTR [rcx+0x3c0],0x1
vgf2p8affineqb zmm15,zmm2,zmm12,0x0
vgf2p8affineqb zmm12,zmm11,zmm12,0x0
vgf2p8affineqb zmm16,zmm11,zmm4,0x0
vgf2p8affineqb zmm4,zmm0,zmm4,0x0
vpternlogq zmm4,zmm15,zmm13,0x96
vgf2p8mulb zmm31,zmm10,zmm4
vmovdqu64 zmm10,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm13,zmm0,zmm10,0x0
vpternlogq zmm12,zmm4,zmm13,0x96
vgf2p8mulb zmm30,zmm9,zmm12
vgf2p8affineqb zmm9,zmm2,zmm10,0x0
vpxorq zmm9,zmm9,zmm16
vgf2p8affineqb zmm10,zmm11,zmm10,0x0
vpternlogq zmm10,zmm12,zmm9,0x96
vpxorq zmm9,zmm4,zmm9
vgf2p8mulb zmm28,zmm3,zmm10
vgf2p8mulb zmm29,zmm1,zmm9
vmovdqu64 zmm1,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm3,zmm2,zmm8,0x0
vgf2p8affineqb zmm8,zmm11,zmm8,0x0
vgf2p8affineqb zmm2,zmm2,zmm1,0x0
vmovdqu64 zmm13,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm15,zmm11,ZMMWORD PTR [rcx+0x2c0],0x0
vgf2p8affineqb zmm16,zmm11,zmm13,0x0
vpxorq zmm2,zmm2,zmm16
vgf2p8affineqb zmm13,zmm0,zmm13,0x0
vpternlogq zmm13,zmm3,zmm15,0x96
vgf2p8affineqb zmm0,zmm0,zmm1,0x0
vgf2p8affineqb zmm11,zmm11,zmm1,0x0
vpternlogq zmm8,zmm13,zmm0,0x96
vpxorq zmm4,zmm4,zmm13
vgf2p8mulb zmm27,zmm7,zmm4
vpternlogq zmm13,zmm9,zmm2,0x96
vpxorq zmm2,zmm2,zmm11
vpternlogq zmm2,zmm10,zmm8,0x96
vgf2p8mulb zmm24,zmm5,zmm2
vpxorq zmm12,zmm12,zmm8
vgf2p8mulb zmm25,zmm14,zmm13
vgf2p8mulb zmm26,zmm6,zmm12
ret
.size flock_partial_gpr_free12_first,.-flock_partial_gpr_free12_first

.p2align 6
.globl flock_partial_gpr_free12_next
.hidden flock_partial_gpr_free12_next
.type flock_partial_gpr_free12_next,@function
flock_partial_gpr_free12_next:
vpbroadcastq zmm0,QWORD PTR [rdi+0x20]
vpbroadcastq zmm1,QWORD PTR [rdi+0x8]
vpbroadcastq zmm2,QWORD PTR [rdi+0x30]
vmovdqu64 zmm3,ZMMWORD PTR [rdx+0x740]
vpbroadcastq zmm4,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm4,zmm4,zmm3,0x0
vpbroadcastq zmm5,QWORD PTR [rdi+0x10]
vpbroadcastq zmm6,QWORD PTR [rdi+0x18]
vpbroadcastq zmm7,QWORD PTR [rdi+0x28]
vpbroadcastq zmm8,QWORD PTR [rdi]
vmovdqa64 zmm9,zmm1
vpternlogq zmm9,zmm5,zmm0,0x96
vgf2p8affineqb zmm10,zmm8,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm11,zmm9,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm9,zmm9,zmm3,0x0
vgf2p8affineqb zmm12,zmm5,ZMMWORD PTR [rdx+0x480],0x0
vgf2p8affineqb zmm13,zmm1,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm13,zmm4,zmm11,0x96
vgf2p8affineqb zmm4,zmm6,ZMMWORD PTR [rdx+0x640],0x0
vpxorq zmm11,zmm5,zmm8
vpternlogq zmm5,zmm1,zmm6,0x96
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm6,zmm7,zmm2,0x96
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rdx+0x440],0x0
vpternlogq zmm13,zmm10,zmm6,0x96
vgf2p8affineqb zmm5,zmm5,ZMMWORD PTR [rdx+0x5c0],0x0
vgf2p8affineqb zmm7,zmm7,zmm3,0x0
vpternlogq zmm11,zmm4,zmm13,0x96
vpternlogq zmm5,zmm12,zmm13,0x96
vgf2p8affineqb zmm3,zmm8,zmm3,0x0
vpxorq zmm6,zmm0,zmm8
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm10,zmm8,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm14,zmm0,ZMMWORD PTR [rdx+0x580],0x0
vpternlogq zmm0,zmm1,zmm2,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x680],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm15,zmm8,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm15,zmm7,zmm13,0x96
vpternlogq zmm15,zmm4,zmm12,0x96
vpxorq zmm4,zmm1,zmm8
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x7c0],0x0
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm7,zmm14
vpternlogq zmm7,zmm10,zmm8,0x96
vpternlogq zmm4,zmm0,zmm7,0x96
vpternlogq zmm7,zmm2,zmm1,0x96
vpternlogq zmm2,zmm9,zmm6,0x96
vpternlogq zmm10,zmm2,zmm5,0x96
vpternlogq zmm6,zmm4,zmm5,0x96
vpternlogq zmm3,zmm4,zmm15,0x96
vpternlogq zmm14,zmm2,zmm15,0x96
vpternlogq zmm5,zmm11,zmm14,0x96
vpternlogq zmm1,zmm8,zmm5,0x96
vpternlogq zmm5,zmm0,zmm2,0x96
vpternlogq zmm9,zmm0,zmm11,0x96
vpxorq zmm7,zmm7,zmm11
mov rax,QWORD PTR [rsi+0x8]
mov r10,rax
xor rax,QWORD PTR [rsi]
mov r9,QWORD PTR [rsi+0x10]
not r9
not r10
xor rax,r9
vpbroadcastq zmm0,r9
vpbroadcastq zmm2,r10
vmovdqu64 zmm4,ZMMWORD PTR [rcx+0x340]
vmovdqu64 zmm8,ZMMWORD PTR [rcx+0x280]
vpbroadcastq zmm11,rax
vmovdqu64 zmm12,ZMMWORD PTR [rcx+0x380]
vgf2p8affineqb zmm13,zmm11,ZMMWORD PTR [rcx+0x3c0],0x1
vgf2p8affineqb zmm15,zmm2,zmm12,0x0
vgf2p8affineqb zmm12,zmm11,zmm12,0x0
vgf2p8affineqb zmm16,zmm11,zmm4,0x0
vgf2p8affineqb zmm4,zmm0,zmm4,0x0
vpternlogq zmm4,zmm15,zmm13,0x96
vgf2p8mulb zmm10,zmm10,zmm4
vpxorq zmm31,zmm31,zmm10
vmovdqu64 zmm10,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm13,zmm0,zmm10,0x0
vpternlogq zmm12,zmm4,zmm13,0x96
vgf2p8mulb zmm9,zmm9,zmm12
vpxorq zmm30,zmm30,zmm9
vgf2p8affineqb zmm9,zmm2,zmm10,0x0
vpxorq zmm9,zmm9,zmm16
vgf2p8affineqb zmm10,zmm11,zmm10,0x0
vpternlogq zmm10,zmm12,zmm9,0x96
vpxorq zmm9,zmm4,zmm9
vgf2p8mulb zmm3,zmm3,zmm10
vpxorq zmm28,zmm28,zmm3
vgf2p8mulb zmm1,zmm1,zmm9
vpxorq zmm29,zmm29,zmm1
vmovdqu64 zmm1,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm3,zmm2,zmm8,0x0
vgf2p8affineqb zmm8,zmm11,zmm8,0x0
vgf2p8affineqb zmm2,zmm2,zmm1,0x0
vmovdqu64 zmm13,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm15,zmm11,ZMMWORD PTR [rcx+0x2c0],0x0
vgf2p8affineqb zmm16,zmm11,zmm13,0x0
vpxorq zmm2,zmm2,zmm16
vgf2p8affineqb zmm13,zmm0,zmm13,0x0
vpternlogq zmm13,zmm3,zmm15,0x96
vgf2p8affineqb zmm0,zmm0,zmm1,0x0
vgf2p8affineqb zmm11,zmm11,zmm1,0x0
vpternlogq zmm8,zmm13,zmm0,0x96
vpxorq zmm4,zmm4,zmm13
vgf2p8mulb zmm7,zmm7,zmm4
vpxorq zmm27,zmm27,zmm7
vpternlogq zmm13,zmm9,zmm2,0x96
vpxorq zmm2,zmm2,zmm11
vpternlogq zmm2,zmm10,zmm8,0x96
vgf2p8mulb zmm5,zmm5,zmm2
vpxorq zmm24,zmm24,zmm5
vpxorq zmm12,zmm12,zmm8
vgf2p8mulb zmm14,zmm14,zmm13
vpxorq zmm25,zmm25,zmm14
vgf2p8mulb zmm6,zmm6,zmm12
vpxorq zmm26,zmm26,zmm6
ret
.size flock_partial_gpr_free12_next,.-flock_partial_gpr_free12_next

.section .note.GNU-stack,"",@progbits
