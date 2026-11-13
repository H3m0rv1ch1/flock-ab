.text

.p2align 6
.globl flock_partial_gpr_free4_first
.hidden flock_partial_gpr_free4_first
.type flock_partial_gpr_free4_first,@function
flock_partial_gpr_free4_first:
mov rax,QWORD PTR [rsi+0x28]
mov r9,QWORD PTR [rsi+0x38]
mov r10,r9
xor r10,rax
xor rax,QWORD PTR [rsi+0x20]
mov rsi,QWORD PTR [rsi+0x30]
xor rsi,r9
xor rax,rsi
not r9
vpbroadcastq zmm0,QWORD PTR [rdi+0x8]
vpbroadcastq zmm1,QWORD PTR [rdi+0x10]
vpbroadcastq zmm2,QWORD PTR [rdi+0x18]
vpbroadcastq zmm3,QWORD PTR [rdi+0x28]
vpbroadcastq zmm4,QWORD PTR [rdi+0x20]
vmovdqa64 zmm5,zmm0
vpternlogq zmm5,zmm1,zmm4,0x96
vpbroadcastq zmm6,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm7,zmm5,ZMMWORD PTR [rdx+0x500],0x0
vpbroadcastq zmm8,QWORD PTR [rdi+0x30]
vmovdqu64 zmm9,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm5,zmm5,zmm9,0x0
vgf2p8affineqb zmm6,zmm6,zmm9,0x0
vgf2p8affineqb zmm10,zmm2,ZMMWORD PTR [rdx+0x640],0x0
vmovdqa64 zmm11,zmm2
vpternlogq zmm11,zmm3,zmm8,0x96
vgf2p8affineqb zmm3,zmm3,zmm9,0x0
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x440],0x0
vpternlogq zmm2,zmm0,zmm1,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x5c0],0x0
vpbroadcastq zmm12,QWORD PTR [rdi]
vgf2p8affineqb zmm9,zmm12,zmm9,0x0
vgf2p8affineqb zmm13,zmm8,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm8,zmm4,zmm0,0x96
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm14,zmm1,ZMMWORD PTR [rdx+0x480],0x0
vpxorq zmm1,zmm1,zmm12
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x400],0x0
vgf2p8affineqb zmm15,zmm12,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm16,zmm0,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm16,zmm6,zmm7,0x96
vpternlogq zmm16,zmm15,zmm11,0x96
vpternlogq zmm2,zmm14,zmm16,0x96
vpternlogq zmm1,zmm10,zmm16,0x96
vpxorq zmm6,zmm4,zmm12
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm7,zmm12,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm7,zmm3,zmm16,0x96
vpternlogq zmm7,zmm10,zmm14,0x96
vgf2p8affineqb zmm3,zmm12,ZMMWORD PTR [rdx+0x600],0x0
vmovdqa64 zmm10,zmm5
vpternlogq zmm10,zmm13,zmm6,0x96
vpternlogq zmm5,zmm8,zmm1,0x96
vgf2p8affineqb zmm11,zmm12,ZMMWORD PTR [rdx+0x7c0],0x0
vpxorq zmm12,zmm0,zmm12
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm14,zmm4
vpternlogq zmm14,zmm3,zmm11,0x96
vpternlogq zmm13,zmm14,zmm0,0x96
vpternlogq zmm12,zmm8,zmm14,0x96
vpternlogq zmm6,zmm12,zmm2,0x96
vpternlogq zmm9,zmm12,zmm7,0x96
vpternlogq zmm4,zmm10,zmm7,0x96
vpternlogq zmm3,zmm10,zmm2,0x96
vpternlogq zmm2,zmm1,zmm4,0x96
vpternlogq zmm0,zmm11,zmm2,0x96
vpternlogq zmm2,zmm8,zmm10,0x96
vpxorq zmm13,zmm13,zmm1
vpbroadcastq zmm1,r10
vpbroadcastq zmm7,rax
vgf2p8affineqb zmm8,zmm7,ZMMWORD PTR [rcx+0x3c0],0x1
vmovdqu64 zmm10,ZMMWORD PTR [rcx+0x380]
vmovdqu64 zmm11,ZMMWORD PTR [rcx+0x340]
vpbroadcastq zmm12,r9
vmovdqu64 zmm14,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm15,zmm1,zmm10,0x0
vpxorq zmm15,zmm15,zmm8
vgf2p8affineqb zmm10,zmm7,zmm10,0x0
vpbroadcastq zmm8,rsi
vgf2p8affineqb zmm16,zmm8,zmm11,0x0
vgf2p8affineqb zmm11,zmm7,zmm11,0x0
vgf2p8affineqb zmm17,zmm12,zmm14,0x0
vpternlogq zmm17,zmm15,zmm16,0x96
vgf2p8mulb zmm27,zmm13,zmm17
vmovdqu64 zmm13,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm15,zmm8,zmm14,0x0
vpternlogq zmm10,zmm17,zmm15,0x96
vgf2p8mulb zmm26,zmm6,zmm10
vgf2p8affineqb zmm6,zmm1,zmm14,0x0
vpxorq zmm6,zmm6,zmm11
vgf2p8affineqb zmm14,zmm7,zmm14,0x0
vpternlogq zmm14,zmm10,zmm6,0x96
vpxorq zmm6,zmm17,zmm6
vgf2p8mulb zmm25,zmm4,zmm6
vgf2p8mulb zmm24,zmm2,zmm14
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x280]
vgf2p8affineqb zmm4,zmm1,zmm2,0x0
vgf2p8affineqb zmm2,zmm7,zmm2,0x0
vmovdqu64 zmm11,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm12,zmm12,zmm11,0x0
vgf2p8affineqb zmm1,zmm1,zmm11,0x0
vgf2p8affineqb zmm15,zmm7,ZMMWORD PTR [rcx+0x2c0],0x0
vpxorq zmm4,zmm4,zmm15
vgf2p8affineqb zmm15,zmm8,zmm13,0x0
vpternlogq zmm12,zmm4,zmm15,0x96
vgf2p8affineqb zmm13,zmm7,zmm13,0x0
vpxorq zmm1,zmm1,zmm13
vgf2p8affineqb zmm8,zmm8,zmm11,0x0
vpternlogq zmm2,zmm12,zmm8,0x96
vgf2p8affineqb zmm7,zmm7,zmm11,0x0
vpternlogq zmm6,zmm12,zmm1,0x96
vpxorq zmm17,zmm17,zmm12
vpxorq zmm1,zmm1,zmm7
vpternlogq zmm1,zmm14,zmm2,0x96
vgf2p8mulb zmm31,zmm3,zmm17
vpxorq zmm10,zmm10,zmm2
vgf2p8mulb zmm28,zmm9,zmm1
vgf2p8mulb zmm30,zmm5,zmm10
vgf2p8mulb zmm29,zmm0,zmm6
ret
.size flock_partial_gpr_free4_first,.-flock_partial_gpr_free4_first

.p2align 6
.globl flock_partial_gpr_free4_next
.hidden flock_partial_gpr_free4_next
.type flock_partial_gpr_free4_next,@function
flock_partial_gpr_free4_next:
mov rax,QWORD PTR [rsi+0x28]
mov r9,QWORD PTR [rsi+0x38]
mov r10,r9
xor r10,rax
xor rax,QWORD PTR [rsi+0x20]
mov rsi,QWORD PTR [rsi+0x30]
xor rsi,r9
xor rax,rsi
not r9
vpbroadcastq zmm0,QWORD PTR [rdi+0x8]
vpbroadcastq zmm1,QWORD PTR [rdi+0x10]
vpbroadcastq zmm2,QWORD PTR [rdi+0x18]
vpbroadcastq zmm3,QWORD PTR [rdi+0x28]
vpbroadcastq zmm4,QWORD PTR [rdi+0x20]
vmovdqa64 zmm5,zmm0
vpternlogq zmm5,zmm1,zmm4,0x96
vpbroadcastq zmm6,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm7,zmm5,ZMMWORD PTR [rdx+0x500],0x0
vpbroadcastq zmm8,QWORD PTR [rdi+0x30]
vmovdqu64 zmm9,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm5,zmm5,zmm9,0x0
vgf2p8affineqb zmm6,zmm6,zmm9,0x0
vgf2p8affineqb zmm10,zmm2,ZMMWORD PTR [rdx+0x640],0x0
vmovdqa64 zmm11,zmm2
vpternlogq zmm11,zmm3,zmm8,0x96
vgf2p8affineqb zmm3,zmm3,zmm9,0x0
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x440],0x0
vpternlogq zmm2,zmm0,zmm1,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x5c0],0x0
vpbroadcastq zmm12,QWORD PTR [rdi]
vgf2p8affineqb zmm9,zmm12,zmm9,0x0
vgf2p8affineqb zmm13,zmm8,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm8,zmm4,zmm0,0x96
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm14,zmm1,ZMMWORD PTR [rdx+0x480],0x0
vpxorq zmm1,zmm1,zmm12
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x400],0x0
vgf2p8affineqb zmm15,zmm12,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm16,zmm0,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm16,zmm6,zmm7,0x96
vpternlogq zmm16,zmm15,zmm11,0x96
vpternlogq zmm2,zmm14,zmm16,0x96
vpternlogq zmm1,zmm10,zmm16,0x96
vpxorq zmm6,zmm4,zmm12
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm7,zmm12,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm7,zmm3,zmm16,0x96
vpternlogq zmm7,zmm10,zmm14,0x96
vgf2p8affineqb zmm3,zmm12,ZMMWORD PTR [rdx+0x600],0x0
vmovdqa64 zmm10,zmm5
vpternlogq zmm10,zmm13,zmm6,0x96
vpternlogq zmm5,zmm8,zmm1,0x96
vgf2p8affineqb zmm11,zmm12,ZMMWORD PTR [rdx+0x7c0],0x0
vpxorq zmm12,zmm0,zmm12
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm14,zmm4
vpternlogq zmm14,zmm3,zmm11,0x96
vpternlogq zmm13,zmm14,zmm0,0x96
vpternlogq zmm12,zmm8,zmm14,0x96
vpternlogq zmm6,zmm12,zmm2,0x96
vpternlogq zmm9,zmm12,zmm7,0x96
vpternlogq zmm4,zmm10,zmm7,0x96
vpternlogq zmm3,zmm10,zmm2,0x96
vpternlogq zmm2,zmm1,zmm4,0x96
vpternlogq zmm0,zmm11,zmm2,0x96
vpternlogq zmm2,zmm8,zmm10,0x96
vpxorq zmm13,zmm13,zmm1
vpbroadcastq zmm1,r10
vpbroadcastq zmm7,rax
vgf2p8affineqb zmm8,zmm7,ZMMWORD PTR [rcx+0x3c0],0x1
vmovdqu64 zmm10,ZMMWORD PTR [rcx+0x380]
vmovdqu64 zmm11,ZMMWORD PTR [rcx+0x340]
vpbroadcastq zmm12,r9
vmovdqu64 zmm14,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm15,zmm1,zmm10,0x0
vpxorq zmm15,zmm15,zmm8
vgf2p8affineqb zmm10,zmm7,zmm10,0x0
vpbroadcastq zmm8,rsi
vgf2p8affineqb zmm16,zmm8,zmm11,0x0
vgf2p8affineqb zmm11,zmm7,zmm11,0x0
vgf2p8affineqb zmm17,zmm12,zmm14,0x0
vpternlogq zmm17,zmm15,zmm16,0x96
vgf2p8mulb zmm13,zmm13,zmm17
vpxorq zmm27,zmm27,zmm13
vmovdqu64 zmm13,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm15,zmm8,zmm14,0x0
vpternlogq zmm10,zmm17,zmm15,0x96
vgf2p8mulb zmm6,zmm6,zmm10
vpxorq zmm26,zmm26,zmm6
vgf2p8affineqb zmm6,zmm1,zmm14,0x0
vpxorq zmm6,zmm6,zmm11
vgf2p8affineqb zmm14,zmm7,zmm14,0x0
vpternlogq zmm14,zmm10,zmm6,0x96
vpxorq zmm6,zmm17,zmm6
vgf2p8mulb zmm4,zmm4,zmm6
vpxorq zmm25,zmm25,zmm4
vgf2p8mulb zmm2,zmm2,zmm14
vpxorq zmm24,zmm24,zmm2
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x280]
vgf2p8affineqb zmm4,zmm1,zmm2,0x0
vgf2p8affineqb zmm2,zmm7,zmm2,0x0
vmovdqu64 zmm11,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm12,zmm12,zmm11,0x0
vgf2p8affineqb zmm1,zmm1,zmm11,0x0
vgf2p8affineqb zmm15,zmm7,ZMMWORD PTR [rcx+0x2c0],0x0
vpxorq zmm4,zmm4,zmm15
vgf2p8affineqb zmm15,zmm8,zmm13,0x0
vpternlogq zmm12,zmm4,zmm15,0x96
vgf2p8affineqb zmm13,zmm7,zmm13,0x0
vpxorq zmm1,zmm1,zmm13
vgf2p8affineqb zmm8,zmm8,zmm11,0x0
vpternlogq zmm2,zmm12,zmm8,0x96
vgf2p8affineqb zmm7,zmm7,zmm11,0x0
vpternlogq zmm6,zmm12,zmm1,0x96
vpxorq zmm17,zmm17,zmm12
vpxorq zmm1,zmm1,zmm7
vpternlogq zmm1,zmm14,zmm2,0x96
vgf2p8mulb zmm3,zmm3,zmm17
vpxorq zmm31,zmm31,zmm3
vpxorq zmm10,zmm10,zmm2
vgf2p8mulb zmm9,zmm9,zmm1
vpxorq zmm28,zmm28,zmm9
vgf2p8mulb zmm5,zmm5,zmm10
vpxorq zmm30,zmm30,zmm5
vgf2p8mulb zmm0,zmm0,zmm6
vpxorq zmm29,zmm29,zmm0
ret
.size flock_partial_gpr_free4_next,.-flock_partial_gpr_free4_next

.section .note.GNU-stack,"",@progbits
