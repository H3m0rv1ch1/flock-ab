.text

.p2align 6
.globl flock_partial_gpr_free10_first
.hidden flock_partial_gpr_free10_first
.type flock_partial_gpr_free10_first,@function
flock_partial_gpr_free10_first:
mov rax,QWORD PTR [rsi+0x8]
mov r9,QWORD PTR [rsi+0x18]
mov r10,r9
mov r11,QWORD PTR [rsi+16]
xor r10,rax
xor r11,QWORD PTR [rsi+24]
mov r8,QWORD PTR [rsi+32]
not r8
not r9
vpbroadcastq zmm0,QWORD PTR [rdi+0x10]
vpbroadcastq zmm1,QWORD PTR [rdi+0x8]
vpbroadcastq zmm2,QWORD PTR [rdi+0x38]
vmovdqu64 zmm3,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm2,zmm2,zmm3,0x0
vpbroadcastq zmm4,QWORD PTR [rdi+0x20]
vpbroadcastq zmm5,QWORD PTR [rdi+0x30]
vmovdqa64 zmm6,zmm1
vpternlogq zmm6,zmm0,zmm4,0x96
vpbroadcastq zmm7,QWORD PTR [rdi+0x18]
vpbroadcastq zmm8,QWORD PTR [rdi+0x28]
vgf2p8affineqb zmm9,zmm6,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm6,zmm6,zmm3,0x0
vmovdqa64 zmm10,zmm7
vpternlogq zmm10,zmm8,zmm5,0x96
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm8,zmm8,zmm3,0x0
vgf2p8affineqb zmm11,zmm1,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm11,zmm2,zmm9,0x96
vpbroadcastq zmm2,QWORD PTR [rdi]
vgf2p8affineqb zmm3,zmm2,zmm3,0x0
vgf2p8affineqb zmm9,zmm2,ZMMWORD PTR [rdx+0x540],0x0
vpternlogq zmm11,zmm9,zmm10,0x96
vgf2p8affineqb zmm9,zmm2,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm9,zmm8,zmm11,0x96
vpxorq zmm8,zmm4,zmm2
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x700],0x0
vpbroadcastq zmm10,r10
mov r10,r11
xor r10,QWORD PTR [rsi]
mov rsi,r8
xor rsi,rax
xor rsi,r10
vgf2p8affineqb zmm12,zmm7,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm7,zmm1,zmm0,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x5c0],0x0
vgf2p8affineqb zmm13,zmm0,ZMMWORD PTR [rdx+0x480],0x0
vpternlogq zmm9,zmm12,zmm13,0x96
vpternlogq zmm7,zmm13,zmm11,0x96
vpxorq zmm0,zmm0,zmm2
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm0,zmm12,zmm11,0x96
vgf2p8affineqb zmm11,zmm5,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm5,zmm4,zmm1,0x96
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm5,zmm5,ZMMWORD PTR [rdx+0x840],0x0
vpxorq zmm12,zmm1,zmm2
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x4c0],0x0
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm13,zmm2,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x7c0],0x0
vmovdqa64 zmm14,zmm4
vpternlogq zmm14,zmm13,zmm2,0x96
vpternlogq zmm12,zmm5,zmm14,0x96
vpternlogq zmm14,zmm11,zmm1,0x96
vpternlogq zmm11,zmm6,zmm8,0x96
vpternlogq zmm4,zmm11,zmm9,0x96
vpternlogq zmm3,zmm12,zmm9,0x96
vpternlogq zmm12,zmm8,zmm7,0x96
vpxorq zmm14,zmm14,zmm0
vpternlogq zmm6,zmm5,zmm0,0x96
vpternlogq zmm0,zmm7,zmm4,0x96
vpternlogq zmm13,zmm11,zmm7,0x96
vpternlogq zmm1,zmm2,zmm0,0x96
vpternlogq zmm0,zmm5,zmm11,0x96
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x380]
vmovdqu64 zmm5,ZMMWORD PTR [rcx+0x340]
vmovdqu64 zmm7,ZMMWORD PTR [rcx+0x2c0]
vpbroadcastq zmm8,rsi
vpbroadcastq zmm9,r11
vgf2p8affineqb zmm11,zmm9,zmm5,0x0
vgf2p8affineqb zmm5,zmm8,zmm5,0x0
vpbroadcastq zmm15,r8
vpbroadcastq zmm16,r9
vmovdqu64 zmm17,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm18,zmm10,zmm2,0x0
vgf2p8affineqb zmm2,zmm8,zmm2,0x0
vmovdqu64 zmm19,ZMMWORD PTR [rcx+0x280]
vgf2p8affineqb zmm20,zmm8,ZMMWORD PTR [rcx+0x3c0],0x1
vpternlogq zmm11,zmm18,zmm20,0x96
vgf2p8affineqb zmm18,zmm16,zmm17,0x0
vgf2p8affineqb zmm20,zmm8,zmm7,0x0
vgf2p8affineqb zmm7,zmm15,zmm7,0x0
vpternlogq zmm7,zmm11,zmm18,0x96
vgf2p8mulb zmm31,zmm13,zmm7
vgf2p8affineqb zmm11,zmm10,zmm19,0x0
vpxorq zmm11,zmm11,zmm20
vgf2p8affineqb zmm13,zmm9,zmm17,0x0
vpxorq zmm13,zmm13,zmm2
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm16,zmm16,zmm2,0x0
vgf2p8affineqb zmm18,zmm15,zmm2,0x0
vgf2p8affineqb zmm20,zmm15,zmm19,0x0
vpternlogq zmm20,zmm7,zmm13,0x96
vgf2p8affineqb zmm19,zmm8,zmm19,0x0
vgf2p8mulb zmm30,zmm6,zmm20
vmovdqu64 zmm6,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm15,zmm15,zmm6,0x0
vgf2p8affineqb zmm13,zmm8,zmm6,0x0
vgf2p8affineqb zmm6,zmm9,zmm6,0x0
vpternlogq zmm16,zmm11,zmm6,0x96
vgf2p8affineqb zmm9,zmm9,zmm2,0x0
vpternlogq zmm19,zmm16,zmm9,0x96
vgf2p8affineqb zmm6,zmm10,zmm17,0x0
vpternlogq zmm15,zmm6,zmm5,0x96
vgf2p8affineqb zmm10,zmm10,zmm2,0x0
vpxorq zmm10,zmm10,zmm13
vgf2p8affineqb zmm17,zmm8,zmm17,0x0
vpxorq zmm18,zmm18,zmm17
vgf2p8affineqb zmm8,zmm8,zmm2,0x0
vpternlogq zmm18,zmm20,zmm15,0x96
vpxorq zmm15,zmm7,zmm15
vpxorq zmm20,zmm20,zmm19
vgf2p8mulb zmm26,zmm12,zmm20
vpxorq zmm8,zmm10,zmm8
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8mulb zmm24,zmm0,zmm8
vgf2p8mulb zmm28,zmm3,zmm18
vpternlogq zmm10,zmm16,zmm15,0x96
vpxorq zmm7,zmm7,zmm16
vgf2p8mulb zmm27,zmm14,zmm7
vgf2p8mulb zmm25,zmm4,zmm10
vgf2p8mulb zmm29,zmm1,zmm15
ret
.size flock_partial_gpr_free10_first,.-flock_partial_gpr_free10_first

.p2align 6
.globl flock_partial_gpr_free10_next
.hidden flock_partial_gpr_free10_next
.type flock_partial_gpr_free10_next,@function
flock_partial_gpr_free10_next:
mov rax,QWORD PTR [rsi+0x8]
mov r9,QWORD PTR [rsi+0x18]
mov r10,r9
mov r11,QWORD PTR [rsi+16]
xor r10,rax
xor r11,QWORD PTR [rsi+24]
mov r8,QWORD PTR [rsi+32]
not r8
not r9
vpbroadcastq zmm0,QWORD PTR [rdi+0x10]
vpbroadcastq zmm1,QWORD PTR [rdi+0x8]
vpbroadcastq zmm2,QWORD PTR [rdi+0x38]
vmovdqu64 zmm3,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm2,zmm2,zmm3,0x0
vpbroadcastq zmm4,QWORD PTR [rdi+0x20]
vpbroadcastq zmm5,QWORD PTR [rdi+0x30]
vmovdqa64 zmm6,zmm1
vpternlogq zmm6,zmm0,zmm4,0x96
vpbroadcastq zmm7,QWORD PTR [rdi+0x18]
vpbroadcastq zmm8,QWORD PTR [rdi+0x28]
vgf2p8affineqb zmm9,zmm6,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm6,zmm6,zmm3,0x0
vmovdqa64 zmm10,zmm7
vpternlogq zmm10,zmm8,zmm5,0x96
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm8,zmm8,zmm3,0x0
vgf2p8affineqb zmm11,zmm1,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm11,zmm2,zmm9,0x96
vpbroadcastq zmm2,QWORD PTR [rdi]
vgf2p8affineqb zmm3,zmm2,zmm3,0x0
vgf2p8affineqb zmm9,zmm2,ZMMWORD PTR [rdx+0x540],0x0
vpternlogq zmm11,zmm9,zmm10,0x96
vgf2p8affineqb zmm9,zmm2,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm9,zmm8,zmm11,0x96
vpxorq zmm8,zmm4,zmm2
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x700],0x0
vpbroadcastq zmm10,r10
mov r10,r11
xor r10,QWORD PTR [rsi]
mov rsi,r8
xor rsi,rax
xor rsi,r10
vgf2p8affineqb zmm12,zmm7,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm7,zmm1,zmm0,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x5c0],0x0
vgf2p8affineqb zmm13,zmm0,ZMMWORD PTR [rdx+0x480],0x0
vpternlogq zmm9,zmm12,zmm13,0x96
vpternlogq zmm7,zmm13,zmm11,0x96
vpxorq zmm0,zmm0,zmm2
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm0,zmm12,zmm11,0x96
vgf2p8affineqb zmm11,zmm5,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm5,zmm4,zmm1,0x96
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm5,zmm5,ZMMWORD PTR [rdx+0x840],0x0
vpxorq zmm12,zmm1,zmm2
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x4c0],0x0
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm13,zmm2,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x7c0],0x0
vmovdqa64 zmm14,zmm4
vpternlogq zmm14,zmm13,zmm2,0x96
vpternlogq zmm12,zmm5,zmm14,0x96
vpternlogq zmm14,zmm11,zmm1,0x96
vpternlogq zmm11,zmm6,zmm8,0x96
vpternlogq zmm4,zmm11,zmm9,0x96
vpternlogq zmm3,zmm12,zmm9,0x96
vpternlogq zmm12,zmm8,zmm7,0x96
vpxorq zmm14,zmm14,zmm0
vpternlogq zmm6,zmm5,zmm0,0x96
vpternlogq zmm0,zmm7,zmm4,0x96
vpternlogq zmm13,zmm11,zmm7,0x96
vpternlogq zmm1,zmm2,zmm0,0x96
vpternlogq zmm0,zmm5,zmm11,0x96
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x380]
vmovdqu64 zmm5,ZMMWORD PTR [rcx+0x340]
vmovdqu64 zmm7,ZMMWORD PTR [rcx+0x2c0]
vpbroadcastq zmm8,rsi
vpbroadcastq zmm9,r11
vgf2p8affineqb zmm11,zmm9,zmm5,0x0
vgf2p8affineqb zmm5,zmm8,zmm5,0x0
vpbroadcastq zmm15,r8
vpbroadcastq zmm16,r9
vmovdqu64 zmm17,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm18,zmm10,zmm2,0x0
vgf2p8affineqb zmm2,zmm8,zmm2,0x0
vmovdqu64 zmm19,ZMMWORD PTR [rcx+0x280]
vgf2p8affineqb zmm20,zmm8,ZMMWORD PTR [rcx+0x3c0],0x1
vpternlogq zmm11,zmm18,zmm20,0x96
vgf2p8affineqb zmm18,zmm16,zmm17,0x0
vgf2p8affineqb zmm20,zmm8,zmm7,0x0
vgf2p8affineqb zmm7,zmm15,zmm7,0x0
vpternlogq zmm7,zmm11,zmm18,0x96
vgf2p8mulb zmm13,zmm13,zmm7
vpxorq zmm31,zmm31,zmm13
vgf2p8affineqb zmm11,zmm10,zmm19,0x0
vpxorq zmm11,zmm11,zmm20
vgf2p8affineqb zmm13,zmm9,zmm17,0x0
vpxorq zmm13,zmm13,zmm2
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm16,zmm16,zmm2,0x0
vgf2p8affineqb zmm18,zmm15,zmm2,0x0
vgf2p8affineqb zmm20,zmm15,zmm19,0x0
vpternlogq zmm20,zmm7,zmm13,0x96
vgf2p8affineqb zmm19,zmm8,zmm19,0x0
vgf2p8mulb zmm6,zmm6,zmm20
vpxorq zmm30,zmm30,zmm6
vmovdqu64 zmm6,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm15,zmm15,zmm6,0x0
vgf2p8affineqb zmm13,zmm8,zmm6,0x0
vgf2p8affineqb zmm6,zmm9,zmm6,0x0
vpternlogq zmm16,zmm11,zmm6,0x96
vgf2p8affineqb zmm9,zmm9,zmm2,0x0
vpternlogq zmm19,zmm16,zmm9,0x96
vgf2p8affineqb zmm6,zmm10,zmm17,0x0
vpternlogq zmm15,zmm6,zmm5,0x96
vgf2p8affineqb zmm10,zmm10,zmm2,0x0
vpxorq zmm10,zmm10,zmm13
vgf2p8affineqb zmm17,zmm8,zmm17,0x0
vpxorq zmm18,zmm18,zmm17
vgf2p8affineqb zmm8,zmm8,zmm2,0x0
vpternlogq zmm18,zmm20,zmm15,0x96
vpxorq zmm15,zmm7,zmm15
vpxorq zmm20,zmm20,zmm19
vgf2p8mulb zmm12,zmm12,zmm20
vpxorq zmm26,zmm26,zmm12
vpxorq zmm8,zmm10,zmm8
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8mulb zmm0,zmm0,zmm8
vpxorq zmm24,zmm24,zmm0
vgf2p8mulb zmm3,zmm3,zmm18
vpxorq zmm28,zmm28,zmm3
vpternlogq zmm10,zmm16,zmm15,0x96
vpxorq zmm7,zmm7,zmm16
vgf2p8mulb zmm14,zmm14,zmm7
vpxorq zmm27,zmm27,zmm14
vgf2p8mulb zmm4,zmm4,zmm10
vpxorq zmm25,zmm25,zmm4
vgf2p8mulb zmm1,zmm1,zmm15
vpxorq zmm29,zmm29,zmm1
ret
.size flock_partial_gpr_free10_next,.-flock_partial_gpr_free10_next

.section .note.GNU-stack,"",@progbits
