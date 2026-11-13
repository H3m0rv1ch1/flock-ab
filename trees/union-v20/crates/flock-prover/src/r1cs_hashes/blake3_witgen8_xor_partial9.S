.text

.p2align 6
.globl flock_partial_gpr_free9_first
.hidden flock_partial_gpr_free9_first
.type flock_partial_gpr_free9_first,@function
flock_partial_gpr_free9_first:
vpbroadcastq zmm0,QWORD PTR [rdi+0x8]
vpbroadcastq zmm1,QWORD PTR [rdi+0x30]
vpbroadcastq zmm2,QWORD PTR [rdi+0x20]
vpbroadcastq zmm3,QWORD PTR [rdi+0x10]
vpbroadcastq zmm4,QWORD PTR [rdi+0x18]
vmovdqu64 zmm5,ZMMWORD PTR [rdx+0x740]
vpbroadcastq zmm6,QWORD PTR [rdi+0x28]
vpbroadcastq zmm7,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm7,zmm7,zmm5,0x0
vmovdqa64 zmm8,zmm0
vpternlogq zmm8,zmm3,zmm2,0x96
vmovdqa64 zmm9,zmm4
vpternlogq zmm9,zmm6,zmm1,0x96
vgf2p8affineqb zmm6,zmm6,zmm5,0x0
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm10,zmm0,ZMMWORD PTR [rdx+0x800],0x0
vpbroadcastq zmm11,QWORD PTR [rdi]
vgf2p8affineqb zmm12,zmm11,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm13,zmm8,ZMMWORD PTR [rdx+0x500],0x0
vpternlogq zmm10,zmm7,zmm13,0x96
vpternlogq zmm10,zmm12,zmm9,0x96
vgf2p8affineqb zmm8,zmm8,zmm5,0x0
vgf2p8affineqb zmm5,zmm11,zmm5,0x0
vgf2p8affineqb zmm7,zmm4,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm4,zmm0,zmm3,0x96
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x5c0],0x0
vpxorq zmm9,zmm2,zmm11
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm12,zmm3,ZMMWORD PTR [rdx+0x480],0x0
vpxorq zmm3,zmm3,zmm11
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm3,zmm7,zmm10,0x96
vpternlogq zmm4,zmm12,zmm10,0x96
vgf2p8affineqb zmm13,zmm1,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm1,zmm2,zmm0,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm14,zmm11,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm14,zmm6,zmm10,0x96
vpternlogq zmm14,zmm7,zmm12,0x96
vgf2p8affineqb zmm6,zmm11,ZMMWORD PTR [rdx+0x600],0x0
vpxorq zmm7,zmm0,zmm11
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x7c0],0x0
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm10,zmm2
vpternlogq zmm10,zmm6,zmm11,0x96
vpternlogq zmm7,zmm1,zmm10,0x96
vpternlogq zmm10,zmm13,zmm0,0x96
vpternlogq zmm13,zmm8,zmm9,0x96
vpternlogq zmm2,zmm13,zmm14,0x96
vpternlogq zmm5,zmm7,zmm14,0x96
vpternlogq zmm7,zmm9,zmm4,0x96
vpxorq zmm10,zmm10,zmm3
vpternlogq zmm8,zmm1,zmm3,0x96
vpternlogq zmm3,zmm4,zmm2,0x96
vpternlogq zmm0,zmm11,zmm3,0x96
vpternlogq zmm6,zmm13,zmm4,0x96
vpternlogq zmm3,zmm1,zmm13,0x96
mov r9,QWORD PTR [rsi+0x28]
mov r10,QWORD PTR [rsi+0x18]
mov r11,QWORD PTR [rsi+0x8]
mov r8,r10
mov rax,r9
not rax
xor r8,r11
xor r8,rax
xor r9,QWORD PTR [rsi+0x20]
xor r11,QWORD PTR [rsi]
vpbroadcastq zmm1,r8
mov r8,QWORD PTR [rsi+0x10]
xor r8,r10
xor r11,r8
not r10
xor r11,r9
vmovdqu64 zmm4,ZMMWORD PTR [rcx+0x380]
vpbroadcastq zmm9,r8
vpbroadcastq zmm11,r11
vmovdqu64 zmm12,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm13,zmm11,ZMMWORD PTR [rcx+0x3c0],0x1
vgf2p8affineqb zmm14,zmm1,zmm4,0x0
vpxorq zmm14,zmm14,zmm13
vgf2p8affineqb zmm4,zmm11,zmm4,0x0
vmovdqu64 zmm13,ZMMWORD PTR [rcx+0x340]
vmovdqu64 zmm15,ZMMWORD PTR [rcx+0x2c0]
vpbroadcastq zmm16,r10
vgf2p8affineqb zmm17,zmm16,zmm12,0x0
vgf2p8affineqb zmm18,zmm9,zmm13,0x0
vpternlogq zmm17,zmm14,zmm18,0x96
vgf2p8affineqb zmm13,zmm11,zmm13,0x0
vpbroadcastq zmm14,r9
vgf2p8affineqb zmm18,zmm9,zmm12,0x0
vpxorq zmm18,zmm18,zmm4
vpbroadcastq zmm4,rax
vmovdqu64 zmm19,ZMMWORD PTR [rcx+0x280]
vgf2p8affineqb zmm20,zmm1,zmm12,0x0
vpxorq zmm20,zmm20,zmm13
vgf2p8affineqb zmm12,zmm11,zmm12,0x0
vgf2p8affineqb zmm13,zmm4,zmm19,0x0
vgf2p8affineqb zmm21,zmm11,zmm15,0x0
vgf2p8affineqb zmm15,zmm14,zmm15,0x0
vpternlogq zmm13,zmm17,zmm15,0x96
vgf2p8mulb zmm31,zmm6,zmm13
vmovdqu64 zmm6,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm16,zmm16,zmm6,0x0
vgf2p8affineqb zmm4,zmm4,zmm6,0x0
vmovdqu64 zmm15,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm17,zmm1,zmm19,0x0
vpxorq zmm17,zmm17,zmm21
vgf2p8affineqb zmm1,zmm1,zmm6,0x0
vgf2p8affineqb zmm21,zmm14,zmm15,0x0
vpternlogq zmm4,zmm20,zmm21,0x96
vgf2p8affineqb zmm20,zmm9,zmm15,0x0
vpternlogq zmm16,zmm17,zmm20,0x96
vgf2p8affineqb zmm15,zmm11,zmm15,0x0
vpxorq zmm1,zmm1,zmm15
vgf2p8affineqb zmm9,zmm9,zmm6,0x0
vgf2p8affineqb zmm15,zmm14,zmm19,0x0
vpternlogq zmm15,zmm13,zmm18,0x96
vgf2p8affineqb zmm14,zmm14,zmm6,0x0
vpxorq zmm14,zmm14,zmm12
vgf2p8affineqb zmm6,zmm11,zmm6,0x0
vgf2p8affineqb zmm11,zmm11,zmm19,0x0
vpternlogq zmm11,zmm16,zmm9,0x96
vpxorq zmm6,zmm1,zmm6
vpternlogq zmm14,zmm15,zmm4,0x96
vpxorq zmm4,zmm13,zmm4
vpternlogq zmm6,zmm14,zmm11,0x96
vgf2p8mulb zmm24,zmm3,zmm6
vgf2p8mulb zmm28,zmm5,zmm14
vpternlogq zmm1,zmm16,zmm4,0x96
vgf2p8mulb zmm25,zmm2,zmm1
vgf2p8mulb zmm29,zmm0,zmm4
vpxorq zmm13,zmm13,zmm16
vgf2p8mulb zmm27,zmm10,zmm13
vgf2p8mulb zmm30,zmm8,zmm15
vpxorq zmm15,zmm15,zmm11
vgf2p8mulb zmm26,zmm7,zmm15
ret
.size flock_partial_gpr_free9_first,.-flock_partial_gpr_free9_first

.p2align 6
.globl flock_partial_gpr_free9_next
.hidden flock_partial_gpr_free9_next
.type flock_partial_gpr_free9_next,@function
flock_partial_gpr_free9_next:
vpbroadcastq zmm0,QWORD PTR [rdi+0x8]
vpbroadcastq zmm1,QWORD PTR [rdi+0x30]
vpbroadcastq zmm2,QWORD PTR [rdi+0x20]
vpbroadcastq zmm3,QWORD PTR [rdi+0x10]
vpbroadcastq zmm4,QWORD PTR [rdi+0x18]
vmovdqu64 zmm5,ZMMWORD PTR [rdx+0x740]
vpbroadcastq zmm6,QWORD PTR [rdi+0x28]
vpbroadcastq zmm7,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm7,zmm7,zmm5,0x0
vmovdqa64 zmm8,zmm0
vpternlogq zmm8,zmm3,zmm2,0x96
vmovdqa64 zmm9,zmm4
vpternlogq zmm9,zmm6,zmm1,0x96
vgf2p8affineqb zmm6,zmm6,zmm5,0x0
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm10,zmm0,ZMMWORD PTR [rdx+0x800],0x0
vpbroadcastq zmm11,QWORD PTR [rdi]
vgf2p8affineqb zmm12,zmm11,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm13,zmm8,ZMMWORD PTR [rdx+0x500],0x0
vpternlogq zmm10,zmm7,zmm13,0x96
vpternlogq zmm10,zmm12,zmm9,0x96
vgf2p8affineqb zmm8,zmm8,zmm5,0x0
vgf2p8affineqb zmm5,zmm11,zmm5,0x0
vgf2p8affineqb zmm7,zmm4,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm4,zmm0,zmm3,0x96
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x5c0],0x0
vpxorq zmm9,zmm2,zmm11
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm12,zmm3,ZMMWORD PTR [rdx+0x480],0x0
vpxorq zmm3,zmm3,zmm11
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm3,zmm7,zmm10,0x96
vpternlogq zmm4,zmm12,zmm10,0x96
vgf2p8affineqb zmm13,zmm1,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm1,zmm2,zmm0,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm14,zmm11,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm14,zmm6,zmm10,0x96
vpternlogq zmm14,zmm7,zmm12,0x96
vgf2p8affineqb zmm6,zmm11,ZMMWORD PTR [rdx+0x600],0x0
vpxorq zmm7,zmm0,zmm11
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x7c0],0x0
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm10,zmm2
vpternlogq zmm10,zmm6,zmm11,0x96
vpternlogq zmm7,zmm1,zmm10,0x96
vpternlogq zmm10,zmm13,zmm0,0x96
vpternlogq zmm13,zmm8,zmm9,0x96
vpternlogq zmm2,zmm13,zmm14,0x96
vpternlogq zmm5,zmm7,zmm14,0x96
vpternlogq zmm7,zmm9,zmm4,0x96
vpxorq zmm10,zmm10,zmm3
vpternlogq zmm8,zmm1,zmm3,0x96
vpternlogq zmm3,zmm4,zmm2,0x96
vpternlogq zmm0,zmm11,zmm3,0x96
vpternlogq zmm6,zmm13,zmm4,0x96
vpternlogq zmm3,zmm1,zmm13,0x96
mov r9,QWORD PTR [rsi+0x28]
mov r10,QWORD PTR [rsi+0x18]
mov r11,QWORD PTR [rsi+0x8]
mov r8,r10
mov rax,r9
not rax
xor r8,r11
xor r8,rax
xor r9,QWORD PTR [rsi+0x20]
xor r11,QWORD PTR [rsi]
vpbroadcastq zmm1,r8
mov r8,QWORD PTR [rsi+0x10]
xor r8,r10
xor r11,r8
not r10
xor r11,r9
vmovdqu64 zmm4,ZMMWORD PTR [rcx+0x380]
vpbroadcastq zmm9,r8
vpbroadcastq zmm11,r11
vmovdqu64 zmm12,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm13,zmm11,ZMMWORD PTR [rcx+0x3c0],0x1
vgf2p8affineqb zmm14,zmm1,zmm4,0x0
vpxorq zmm14,zmm14,zmm13
vgf2p8affineqb zmm4,zmm11,zmm4,0x0
vmovdqu64 zmm13,ZMMWORD PTR [rcx+0x340]
vmovdqu64 zmm15,ZMMWORD PTR [rcx+0x2c0]
vpbroadcastq zmm16,r10
vgf2p8affineqb zmm17,zmm16,zmm12,0x0
vgf2p8affineqb zmm18,zmm9,zmm13,0x0
vpternlogq zmm17,zmm14,zmm18,0x96
vgf2p8affineqb zmm13,zmm11,zmm13,0x0
vpbroadcastq zmm14,r9
vgf2p8affineqb zmm18,zmm9,zmm12,0x0
vpxorq zmm18,zmm18,zmm4
vpbroadcastq zmm4,rax
vmovdqu64 zmm19,ZMMWORD PTR [rcx+0x280]
vgf2p8affineqb zmm20,zmm1,zmm12,0x0
vpxorq zmm20,zmm20,zmm13
vgf2p8affineqb zmm12,zmm11,zmm12,0x0
vgf2p8affineqb zmm13,zmm4,zmm19,0x0
vgf2p8affineqb zmm21,zmm11,zmm15,0x0
vgf2p8affineqb zmm15,zmm14,zmm15,0x0
vpternlogq zmm13,zmm17,zmm15,0x96
vgf2p8mulb zmm6,zmm6,zmm13
vpxorq zmm31,zmm31,zmm6
vmovdqu64 zmm6,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm16,zmm16,zmm6,0x0
vgf2p8affineqb zmm4,zmm4,zmm6,0x0
vmovdqu64 zmm15,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm17,zmm1,zmm19,0x0
vpxorq zmm17,zmm17,zmm21
vgf2p8affineqb zmm1,zmm1,zmm6,0x0
vgf2p8affineqb zmm21,zmm14,zmm15,0x0
vpternlogq zmm4,zmm20,zmm21,0x96
vgf2p8affineqb zmm20,zmm9,zmm15,0x0
vpternlogq zmm16,zmm17,zmm20,0x96
vgf2p8affineqb zmm15,zmm11,zmm15,0x0
vpxorq zmm1,zmm1,zmm15
vgf2p8affineqb zmm9,zmm9,zmm6,0x0
vgf2p8affineqb zmm15,zmm14,zmm19,0x0
vpternlogq zmm15,zmm13,zmm18,0x96
vgf2p8affineqb zmm14,zmm14,zmm6,0x0
vpxorq zmm14,zmm14,zmm12
vgf2p8affineqb zmm6,zmm11,zmm6,0x0
vgf2p8affineqb zmm11,zmm11,zmm19,0x0
vpternlogq zmm11,zmm16,zmm9,0x96
vpxorq zmm6,zmm1,zmm6
vpternlogq zmm14,zmm15,zmm4,0x96
vpxorq zmm4,zmm13,zmm4
vpternlogq zmm6,zmm14,zmm11,0x96
vgf2p8mulb zmm3,zmm3,zmm6
vpxorq zmm24,zmm24,zmm3
vgf2p8mulb zmm5,zmm5,zmm14
vpxorq zmm28,zmm28,zmm5
vpternlogq zmm1,zmm16,zmm4,0x96
vgf2p8mulb zmm2,zmm2,zmm1
vpxorq zmm25,zmm25,zmm2
vgf2p8mulb zmm0,zmm0,zmm4
vpxorq zmm29,zmm29,zmm0
vpxorq zmm13,zmm13,zmm16
vgf2p8mulb zmm10,zmm10,zmm13
vpxorq zmm27,zmm27,zmm10
vgf2p8mulb zmm8,zmm8,zmm15
vpxorq zmm30,zmm30,zmm8
vpxorq zmm15,zmm15,zmm11
vgf2p8mulb zmm7,zmm7,zmm15
vpxorq zmm26,zmm26,zmm7
ret
.size flock_partial_gpr_free9_next,.-flock_partial_gpr_free9_next

.section .note.GNU-stack,"",@progbits
