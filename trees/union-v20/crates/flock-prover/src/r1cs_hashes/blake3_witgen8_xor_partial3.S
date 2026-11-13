.text

.p2align 6
.globl flock_partial_gpr_free3_first
.hidden flock_partial_gpr_free3_first
.type flock_partial_gpr_free3_first,@function
flock_partial_gpr_free3_first:
vpbroadcastq zmm0,QWORD PTR [rdi+0x28]
vpbroadcastq zmm1,QWORD PTR [rdi+0x10]
vpbroadcastq zmm2,QWORD PTR [rdi+0x30]
vpbroadcastq zmm3,QWORD PTR [rdi+0x38]
vpbroadcastq zmm4,QWORD PTR [rdi+0x8]
vpbroadcastq zmm5,QWORD PTR [rdi+0x20]
vmovdqa64 zmm6,zmm4
vpternlogq zmm6,zmm1,zmm5,0x96
vmovdqu64 zmm7,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm3,zmm3,zmm7,0x0
vpbroadcastq zmm8,QWORD PTR [rdi]
vgf2p8affineqb zmm9,zmm8,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm10,zmm4,ZMMWORD PTR [rdx+0x800],0x0
vpbroadcastq zmm11,QWORD PTR [rdi+0x18]
vmovdqa64 zmm12,zmm11
vpternlogq zmm12,zmm0,zmm2,0x96
vgf2p8affineqb zmm0,zmm0,zmm7,0x0
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x440],0x0
vpxorq zmm13,zmm1,zmm8
vgf2p8affineqb zmm13,zmm13,ZMMWORD PTR [rdx+0x400],0x0
vgf2p8affineqb zmm14,zmm8,ZMMWORD PTR [rdx+0x6c0],0x0
vgf2p8affineqb zmm15,zmm6,ZMMWORD PTR [rdx+0x500],0x0
vpternlogq zmm10,zmm3,zmm15,0x96
vpternlogq zmm10,zmm9,zmm12,0x96
vpternlogq zmm14,zmm0,zmm10,0x96
vgf2p8affineqb zmm6,zmm6,zmm7,0x0
vgf2p8affineqb zmm7,zmm8,zmm7,0x0
vgf2p8affineqb zmm0,zmm1,ZMMWORD PTR [rdx+0x480],0x0
vpternlogq zmm1,zmm4,zmm11,0x96
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm14,zmm11,zmm0,0x96
vpternlogq zmm13,zmm11,zmm10,0x96
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x5c0],0x0
vpternlogq zmm1,zmm0,zmm10,0x96
vgf2p8affineqb zmm0,zmm5,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm3,zmm2,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm2,zmm5,zmm4,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x840],0x0
vpxorq zmm5,zmm5,zmm8
vgf2p8affineqb zmm5,zmm5,ZMMWORD PTR [rdx+0x700],0x0
vpxorq zmm9,zmm4,zmm8
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x4c0],0x0
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm10,zmm8,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x7c0],0x0
vmovdqa64 zmm11,zmm0
vpternlogq zmm11,zmm10,zmm8,0x96
vpternlogq zmm9,zmm2,zmm11,0x96
vpternlogq zmm11,zmm3,zmm4,0x96
vpternlogq zmm3,zmm6,zmm5,0x96
vpternlogq zmm0,zmm3,zmm14,0x96
vpternlogq zmm7,zmm9,zmm14,0x96
vpternlogq zmm9,zmm5,zmm1,0x96
vpternlogq zmm10,zmm3,zmm1,0x96
vpternlogq zmm6,zmm2,zmm13,0x96
vpxorq zmm11,zmm11,zmm13
vpternlogq zmm1,zmm13,zmm0,0x96
vpternlogq zmm2,zmm1,zmm3,0x96
vpternlogq zmm4,zmm8,zmm1,0x96
mov rax,QWORD PTR [rsi+0x20]
mov r8,QWORD PTR [rsi+24]
mov r10,rax
mov r9,QWORD PTR [rsi+0x30]
xor r10,r9
mov r11,QWORD PTR [rsi+40]
not r8
xor r11,QWORD PTR [rsi+32]
not rax
vpbroadcastq zmm1,r10
mov r10,r11
xor r10,QWORD PTR [rsi+0x38]
mov rsi,r8
xor rsi,r9
xor rsi,r10
vmovdqu64 zmm3,ZMMWORD PTR [rcx+0x300]
vmovdqu64 zmm5,ZMMWORD PTR [rcx+0x280]
vpbroadcastq zmm8,r11
vmovdqu64 zmm12,ZMMWORD PTR [rcx+0x380]
vpbroadcastq zmm13,r8
vpbroadcastq zmm14,rsi
vgf2p8affineqb zmm15,zmm1,zmm12,0x0
vgf2p8affineqb zmm12,zmm14,zmm12,0x0
vgf2p8affineqb zmm16,zmm1,zmm5,0x0
vmovdqu64 zmm17,ZMMWORD PTR [rcx+0x340]
vgf2p8affineqb zmm18,zmm14,ZMMWORD PTR [rcx+0x3c0],0x1
vgf2p8affineqb zmm19,zmm8,zmm17,0x0
vpternlogq zmm19,zmm15,zmm18,0x96
vgf2p8affineqb zmm17,zmm14,zmm17,0x0
vpbroadcastq zmm15,rax
vmovdqu64 zmm18,ZMMWORD PTR [rcx+0x2c0]
vgf2p8affineqb zmm20,zmm14,zmm18,0x0
vpxorq zmm16,zmm16,zmm20
vgf2p8affineqb zmm18,zmm13,zmm18,0x0
vgf2p8affineqb zmm20,zmm8,zmm3,0x0
vpxorq zmm20,zmm20,zmm12
vgf2p8affineqb zmm12,zmm15,zmm3,0x0
vpternlogq zmm18,zmm19,zmm12,0x96
vgf2p8mulb zmm24,zmm2,zmm18
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm15,zmm15,zmm2,0x0
vmovdqu64 zmm12,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm19,zmm13,zmm5,0x0
vpternlogq zmm19,zmm18,zmm20,0x96
vgf2p8affineqb zmm5,zmm14,zmm5,0x0
vgf2p8mulb zmm25,zmm0,zmm19
vgf2p8affineqb zmm0,zmm14,zmm12,0x0
vgf2p8affineqb zmm20,zmm8,zmm2,0x0
vgf2p8affineqb zmm8,zmm8,zmm12,0x0
vpternlogq zmm15,zmm16,zmm8,0x96
vpternlogq zmm5,zmm15,zmm20,0x96
vgf2p8affineqb zmm12,zmm13,zmm12,0x0
vgf2p8affineqb zmm13,zmm13,zmm2,0x0
vgf2p8affineqb zmm8,zmm14,zmm2,0x0
vgf2p8affineqb zmm14,zmm14,zmm3,0x0
vpxorq zmm13,zmm13,zmm14
vgf2p8affineqb zmm2,zmm1,zmm2,0x0
vgf2p8affineqb zmm1,zmm1,zmm3,0x0
vpternlogq zmm12,zmm1,zmm17,0x96
vpxorq zmm2,zmm2,zmm0
vpternlogq zmm13,zmm19,zmm12,0x96
vpxorq zmm12,zmm18,zmm12
vgf2p8mulb zmm26,zmm9,zmm12
vpternlogq zmm12,zmm15,zmm2,0x96
vpxorq zmm18,zmm18,zmm15
vgf2p8mulb zmm30,zmm6,zmm12
vpxorq zmm2,zmm2,zmm8
vgf2p8mulb zmm28,zmm7,zmm18
vpternlogq zmm2,zmm13,zmm5,0x96
vgf2p8mulb zmm31,zmm10,zmm2
vgf2p8mulb zmm27,zmm11,zmm13
vpxorq zmm19,zmm19,zmm5
vgf2p8mulb zmm29,zmm4,zmm19
ret
.size flock_partial_gpr_free3_first,.-flock_partial_gpr_free3_first

.p2align 6
.globl flock_partial_gpr_free3_next
.hidden flock_partial_gpr_free3_next
.type flock_partial_gpr_free3_next,@function
flock_partial_gpr_free3_next:
vpbroadcastq zmm0,QWORD PTR [rdi+0x28]
vpbroadcastq zmm1,QWORD PTR [rdi+0x10]
vpbroadcastq zmm2,QWORD PTR [rdi+0x30]
vpbroadcastq zmm3,QWORD PTR [rdi+0x38]
vpbroadcastq zmm4,QWORD PTR [rdi+0x8]
vpbroadcastq zmm5,QWORD PTR [rdi+0x20]
vmovdqa64 zmm6,zmm4
vpternlogq zmm6,zmm1,zmm5,0x96
vmovdqu64 zmm7,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm3,zmm3,zmm7,0x0
vpbroadcastq zmm8,QWORD PTR [rdi]
vgf2p8affineqb zmm9,zmm8,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm10,zmm4,ZMMWORD PTR [rdx+0x800],0x0
vpbroadcastq zmm11,QWORD PTR [rdi+0x18]
vmovdqa64 zmm12,zmm11
vpternlogq zmm12,zmm0,zmm2,0x96
vgf2p8affineqb zmm0,zmm0,zmm7,0x0
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x440],0x0
vpxorq zmm13,zmm1,zmm8
vgf2p8affineqb zmm13,zmm13,ZMMWORD PTR [rdx+0x400],0x0
vgf2p8affineqb zmm14,zmm8,ZMMWORD PTR [rdx+0x6c0],0x0
vgf2p8affineqb zmm15,zmm6,ZMMWORD PTR [rdx+0x500],0x0
vpternlogq zmm10,zmm3,zmm15,0x96
vpternlogq zmm10,zmm9,zmm12,0x96
vpternlogq zmm14,zmm0,zmm10,0x96
vgf2p8affineqb zmm6,zmm6,zmm7,0x0
vgf2p8affineqb zmm7,zmm8,zmm7,0x0
vgf2p8affineqb zmm0,zmm1,ZMMWORD PTR [rdx+0x480],0x0
vpternlogq zmm1,zmm4,zmm11,0x96
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm14,zmm11,zmm0,0x96
vpternlogq zmm13,zmm11,zmm10,0x96
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x5c0],0x0
vpternlogq zmm1,zmm0,zmm10,0x96
vgf2p8affineqb zmm0,zmm5,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm3,zmm2,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm2,zmm5,zmm4,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x840],0x0
vpxorq zmm5,zmm5,zmm8
vgf2p8affineqb zmm5,zmm5,ZMMWORD PTR [rdx+0x700],0x0
vpxorq zmm9,zmm4,zmm8
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x4c0],0x0
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm10,zmm8,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x7c0],0x0
vmovdqa64 zmm11,zmm0
vpternlogq zmm11,zmm10,zmm8,0x96
vpternlogq zmm9,zmm2,zmm11,0x96
vpternlogq zmm11,zmm3,zmm4,0x96
vpternlogq zmm3,zmm6,zmm5,0x96
vpternlogq zmm0,zmm3,zmm14,0x96
vpternlogq zmm7,zmm9,zmm14,0x96
vpternlogq zmm9,zmm5,zmm1,0x96
vpternlogq zmm10,zmm3,zmm1,0x96
vpternlogq zmm6,zmm2,zmm13,0x96
vpxorq zmm11,zmm11,zmm13
vpternlogq zmm1,zmm13,zmm0,0x96
vpternlogq zmm2,zmm1,zmm3,0x96
vpternlogq zmm4,zmm8,zmm1,0x96
mov rax,QWORD PTR [rsi+0x20]
mov r8,QWORD PTR [rsi+24]
mov r10,rax
mov r9,QWORD PTR [rsi+0x30]
xor r10,r9
mov r11,QWORD PTR [rsi+40]
not r8
xor r11,QWORD PTR [rsi+32]
not rax
vpbroadcastq zmm1,r10
mov r10,r11
xor r10,QWORD PTR [rsi+0x38]
mov rsi,r8
xor rsi,r9
xor rsi,r10
vmovdqu64 zmm3,ZMMWORD PTR [rcx+0x300]
vmovdqu64 zmm5,ZMMWORD PTR [rcx+0x280]
vpbroadcastq zmm8,r11
vmovdqu64 zmm12,ZMMWORD PTR [rcx+0x380]
vpbroadcastq zmm13,r8
vpbroadcastq zmm14,rsi
vgf2p8affineqb zmm15,zmm1,zmm12,0x0
vgf2p8affineqb zmm12,zmm14,zmm12,0x0
vgf2p8affineqb zmm16,zmm1,zmm5,0x0
vmovdqu64 zmm17,ZMMWORD PTR [rcx+0x340]
vgf2p8affineqb zmm18,zmm14,ZMMWORD PTR [rcx+0x3c0],0x1
vgf2p8affineqb zmm19,zmm8,zmm17,0x0
vpternlogq zmm19,zmm15,zmm18,0x96
vgf2p8affineqb zmm17,zmm14,zmm17,0x0
vpbroadcastq zmm15,rax
vmovdqu64 zmm18,ZMMWORD PTR [rcx+0x2c0]
vgf2p8affineqb zmm20,zmm14,zmm18,0x0
vpxorq zmm16,zmm16,zmm20
vgf2p8affineqb zmm18,zmm13,zmm18,0x0
vgf2p8affineqb zmm20,zmm8,zmm3,0x0
vpxorq zmm20,zmm20,zmm12
vgf2p8affineqb zmm12,zmm15,zmm3,0x0
vpternlogq zmm18,zmm19,zmm12,0x96
vgf2p8mulb zmm2,zmm2,zmm18
vpxorq zmm24,zmm24,zmm2
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm15,zmm15,zmm2,0x0
vmovdqu64 zmm12,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm19,zmm13,zmm5,0x0
vpternlogq zmm19,zmm18,zmm20,0x96
vgf2p8affineqb zmm5,zmm14,zmm5,0x0
vgf2p8mulb zmm0,zmm0,zmm19
vpxorq zmm25,zmm25,zmm0
vgf2p8affineqb zmm0,zmm14,zmm12,0x0
vgf2p8affineqb zmm20,zmm8,zmm2,0x0
vgf2p8affineqb zmm8,zmm8,zmm12,0x0
vpternlogq zmm15,zmm16,zmm8,0x96
vpternlogq zmm5,zmm15,zmm20,0x96
vgf2p8affineqb zmm12,zmm13,zmm12,0x0
vgf2p8affineqb zmm13,zmm13,zmm2,0x0
vgf2p8affineqb zmm8,zmm14,zmm2,0x0
vgf2p8affineqb zmm14,zmm14,zmm3,0x0
vpxorq zmm13,zmm13,zmm14
vgf2p8affineqb zmm2,zmm1,zmm2,0x0
vgf2p8affineqb zmm1,zmm1,zmm3,0x0
vpternlogq zmm12,zmm1,zmm17,0x96
vpxorq zmm2,zmm2,zmm0
vpternlogq zmm13,zmm19,zmm12,0x96
vpxorq zmm12,zmm18,zmm12
vgf2p8mulb zmm9,zmm9,zmm12
vpxorq zmm26,zmm26,zmm9
vpternlogq zmm12,zmm15,zmm2,0x96
vpxorq zmm18,zmm18,zmm15
vgf2p8mulb zmm6,zmm6,zmm12
vpxorq zmm30,zmm30,zmm6
vpxorq zmm2,zmm2,zmm8
vgf2p8mulb zmm7,zmm7,zmm18
vpxorq zmm28,zmm28,zmm7
vpternlogq zmm2,zmm13,zmm5,0x96
vgf2p8mulb zmm10,zmm10,zmm2
vpxorq zmm31,zmm31,zmm10
vgf2p8mulb zmm11,zmm11,zmm13
vpxorq zmm27,zmm27,zmm11
vpxorq zmm19,zmm19,zmm5
vgf2p8mulb zmm4,zmm4,zmm19
vpxorq zmm29,zmm29,zmm4
ret
.size flock_partial_gpr_free3_next,.-flock_partial_gpr_free3_next

.section .note.GNU-stack,"",@progbits
