.text

.p2align 6
.globl flock_partial_gpr_free2_first
.hidden flock_partial_gpr_free2_first
.type flock_partial_gpr_free2_first,@function
flock_partial_gpr_free2_first:
mov r9,QWORD PTR [rsi+0x18]
mov r10,QWORD PTR [rsi+0x28]
mov r11,QWORD PTR [rsi+0x38]
mov rax,r9
not rax
mov r8,r11
xor r8,r10
xor r8,rax
vpbroadcastq zmm0,QWORD PTR [rdi+0x8]
vpbroadcastq zmm1,QWORD PTR [rdi+0x30]
vpbroadcastq zmm2,QWORD PTR [rdi+0x10]
xor r9,QWORD PTR [rsi+0x10]
vpbroadcastq zmm3,QWORD PTR [rdi+0x28]
xor r11,r9
vpbroadcastq zmm4,QWORD PTR [rdi+0x20]
vmovdqu64 zmm5,ZMMWORD PTR [rdx+0x740]
vmovdqa64 zmm6,zmm0
vpternlogq zmm6,zmm2,zmm4,0x96
vpbroadcastq zmm7,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm7,zmm7,zmm5,0x0
vgf2p8affineqb zmm8,zmm3,zmm5,0x0
vpbroadcastq zmm9,QWORD PTR [rdi+0x18]
vpternlogq zmm3,zmm9,zmm1,0x96
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm10,zmm6,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm6,zmm6,zmm5,0x0
vpbroadcastq zmm11,QWORD PTR [rdi]
vgf2p8affineqb zmm5,zmm11,zmm5,0x0
vgf2p8affineqb zmm12,zmm0,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm12,zmm7,zmm10,0x96
vgf2p8affineqb zmm7,zmm11,ZMMWORD PTR [rdx+0x6c0],0x0
vgf2p8affineqb zmm10,zmm11,ZMMWORD PTR [rdx+0x540],0x0
vpternlogq zmm12,zmm10,zmm3,0x96
vpternlogq zmm7,zmm8,zmm12,0x96
vgf2p8affineqb zmm3,zmm1,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm1,zmm4,zmm0,0x96
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm8,zmm9,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm9,zmm0,zmm2,0x96
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x5c0],0x0
vpxorq zmm10,zmm4,zmm11
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm13,zmm2,ZMMWORD PTR [rdx+0x480],0x0
vpxorq zmm2,zmm2,zmm11
vpternlogq zmm7,zmm8,zmm13,0x96
vpternlogq zmm9,zmm13,zmm12,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm2,zmm8,zmm12,0x96
vmovdqa64 zmm8,zmm6
vpternlogq zmm8,zmm3,zmm10,0x96
vpternlogq zmm6,zmm1,zmm2,0x96
vpbroadcastq zmm12,r8
mov r8,QWORD PTR [rsi+0x20]
xor r8,r10
not r10
vgf2p8affineqb zmm13,zmm11,ZMMWORD PTR [rdx+0x600],0x0
vpbroadcastq zmm14,r8
xor r8,QWORD PTR [rsi+0x30]
xor r11,r8
vgf2p8affineqb zmm15,zmm11,ZMMWORD PTR [rdx+0x7c0],0x0
vpxorq zmm11,zmm0,zmm11
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm16,zmm4
vpternlogq zmm16,zmm13,zmm15,0x96
vpternlogq zmm4,zmm8,zmm7,0x96
vpternlogq zmm11,zmm1,zmm16,0x96
vpternlogq zmm16,zmm3,zmm0,0x96
vpternlogq zmm5,zmm11,zmm7,0x96
vpternlogq zmm11,zmm10,zmm9,0x96
vpternlogq zmm13,zmm8,zmm9,0x96
vpternlogq zmm9,zmm2,zmm4,0x96
vpternlogq zmm0,zmm15,zmm9,0x96
vpternlogq zmm9,zmm1,zmm8,0x96
vpxorq zmm16,zmm16,zmm2
vpbroadcastq zmm1,r11
vgf2p8affineqb zmm2,zmm1,ZMMWORD PTR [rcx+0x3c0],0x1
vmovdqu64 zmm3,ZMMWORD PTR [rcx+0x380]
vgf2p8affineqb zmm7,zmm12,zmm3,0x0
vpxorq zmm7,zmm7,zmm2
vgf2p8affineqb zmm3,zmm1,zmm3,0x0
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x340]
vpbroadcastq zmm8,r10
vmovdqu64 zmm10,ZMMWORD PTR [rcx+0x280]
vmovdqu64 zmm15,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm17,zmm8,zmm15,0x0
vgf2p8affineqb zmm18,zmm14,zmm2,0x0
vpternlogq zmm17,zmm7,zmm18,0x96
vgf2p8affineqb zmm2,zmm1,zmm2,0x0
vpbroadcastq zmm7,r9
vpbroadcastq zmm18,rax
vmovdqu64 zmm19,ZMMWORD PTR [rcx+0x2c0]
vgf2p8affineqb zmm20,zmm7,zmm19,0x0
vgf2p8affineqb zmm19,zmm1,zmm19,0x0
vgf2p8affineqb zmm21,zmm12,zmm15,0x0
vpxorq zmm21,zmm21,zmm2
vgf2p8affineqb zmm2,zmm12,zmm10,0x0
vpxorq zmm2,zmm2,zmm19
vgf2p8affineqb zmm19,zmm14,zmm15,0x0
vpxorq zmm19,zmm19,zmm3
vgf2p8affineqb zmm15,zmm1,zmm15,0x0
vgf2p8affineqb zmm3,zmm18,zmm10,0x0
vpternlogq zmm3,zmm17,zmm20,0x96
vgf2p8mulb zmm25,zmm4,zmm3
vmovdqu64 zmm4,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm17,zmm1,zmm4,0x0
vgf2p8affineqb zmm20,zmm7,zmm10,0x0
vpternlogq zmm20,zmm3,zmm19,0x96
vgf2p8affineqb zmm10,zmm1,zmm10,0x0
vmovdqu64 zmm19,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm18,zmm18,zmm19,0x0
vgf2p8affineqb zmm8,zmm8,zmm19,0x0
vgf2p8affineqb zmm1,zmm1,zmm19,0x0
vgf2p8affineqb zmm12,zmm12,zmm19,0x0
vpxorq zmm12,zmm12,zmm17
vpxorq zmm1,zmm12,zmm1
vgf2p8mulb zmm24,zmm9,zmm20
vgf2p8affineqb zmm9,zmm14,zmm19,0x0
vgf2p8affineqb zmm19,zmm7,zmm19,0x0
vpxorq zmm19,zmm19,zmm15
vgf2p8affineqb zmm7,zmm7,zmm4,0x0
vpternlogq zmm18,zmm21,zmm7,0x96
vgf2p8affineqb zmm14,zmm14,zmm4,0x0
vpternlogq zmm8,zmm2,zmm14,0x96
vpternlogq zmm10,zmm8,zmm9,0x96
vpternlogq zmm19,zmm20,zmm18,0x96
vpxorq zmm18,zmm3,zmm18
vpxorq zmm3,zmm3,zmm8
vpternlogq zmm8,zmm18,zmm12,0x96
vgf2p8mulb zmm31,zmm13,zmm8
vgf2p8mulb zmm29,zmm0,zmm3
vgf2p8mulb zmm27,zmm16,zmm18
vpxorq zmm20,zmm20,zmm10
vgf2p8mulb zmm28,zmm5,zmm20
vpternlogq zmm1,zmm19,zmm10,0x96
vgf2p8mulb zmm26,zmm11,zmm19
vgf2p8mulb zmm30,zmm6,zmm1
ret
.size flock_partial_gpr_free2_first,.-flock_partial_gpr_free2_first

.p2align 6
.globl flock_partial_gpr_free2_next
.hidden flock_partial_gpr_free2_next
.type flock_partial_gpr_free2_next,@function
flock_partial_gpr_free2_next:
mov r9,QWORD PTR [rsi+0x18]
mov r10,QWORD PTR [rsi+0x28]
mov r11,QWORD PTR [rsi+0x38]
mov rax,r9
not rax
mov r8,r11
xor r8,r10
xor r8,rax
vpbroadcastq zmm0,QWORD PTR [rdi+0x8]
vpbroadcastq zmm1,QWORD PTR [rdi+0x30]
vpbroadcastq zmm2,QWORD PTR [rdi+0x10]
xor r9,QWORD PTR [rsi+0x10]
vpbroadcastq zmm3,QWORD PTR [rdi+0x28]
xor r11,r9
vpbroadcastq zmm4,QWORD PTR [rdi+0x20]
vmovdqu64 zmm5,ZMMWORD PTR [rdx+0x740]
vmovdqa64 zmm6,zmm0
vpternlogq zmm6,zmm2,zmm4,0x96
vpbroadcastq zmm7,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm7,zmm7,zmm5,0x0
vgf2p8affineqb zmm8,zmm3,zmm5,0x0
vpbroadcastq zmm9,QWORD PTR [rdi+0x18]
vpternlogq zmm3,zmm9,zmm1,0x96
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm10,zmm6,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm6,zmm6,zmm5,0x0
vpbroadcastq zmm11,QWORD PTR [rdi]
vgf2p8affineqb zmm5,zmm11,zmm5,0x0
vgf2p8affineqb zmm12,zmm0,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm12,zmm7,zmm10,0x96
vgf2p8affineqb zmm7,zmm11,ZMMWORD PTR [rdx+0x6c0],0x0
vgf2p8affineqb zmm10,zmm11,ZMMWORD PTR [rdx+0x540],0x0
vpternlogq zmm12,zmm10,zmm3,0x96
vpternlogq zmm7,zmm8,zmm12,0x96
vgf2p8affineqb zmm3,zmm1,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm1,zmm4,zmm0,0x96
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x840],0x0
vgf2p8affineqb zmm8,zmm9,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm9,zmm0,zmm2,0x96
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x5c0],0x0
vpxorq zmm10,zmm4,zmm11
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm13,zmm2,ZMMWORD PTR [rdx+0x480],0x0
vpxorq zmm2,zmm2,zmm11
vpternlogq zmm7,zmm8,zmm13,0x96
vpternlogq zmm9,zmm13,zmm12,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm2,zmm8,zmm12,0x96
vmovdqa64 zmm8,zmm6
vpternlogq zmm8,zmm3,zmm10,0x96
vpternlogq zmm6,zmm1,zmm2,0x96
vpbroadcastq zmm12,r8
mov r8,QWORD PTR [rsi+0x20]
xor r8,r10
not r10
vgf2p8affineqb zmm13,zmm11,ZMMWORD PTR [rdx+0x600],0x0
vpbroadcastq zmm14,r8
xor r8,QWORD PTR [rsi+0x30]
xor r11,r8
vgf2p8affineqb zmm15,zmm11,ZMMWORD PTR [rdx+0x7c0],0x0
vpxorq zmm11,zmm0,zmm11
vgf2p8affineqb zmm11,zmm11,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x4c0],0x0
vmovdqa64 zmm16,zmm4
vpternlogq zmm16,zmm13,zmm15,0x96
vpternlogq zmm4,zmm8,zmm7,0x96
vpternlogq zmm11,zmm1,zmm16,0x96
vpternlogq zmm16,zmm3,zmm0,0x96
vpternlogq zmm5,zmm11,zmm7,0x96
vpternlogq zmm11,zmm10,zmm9,0x96
vpternlogq zmm13,zmm8,zmm9,0x96
vpternlogq zmm9,zmm2,zmm4,0x96
vpternlogq zmm0,zmm15,zmm9,0x96
vpternlogq zmm9,zmm1,zmm8,0x96
vpxorq zmm16,zmm16,zmm2
vpbroadcastq zmm1,r11
vgf2p8affineqb zmm2,zmm1,ZMMWORD PTR [rcx+0x3c0],0x1
vmovdqu64 zmm3,ZMMWORD PTR [rcx+0x380]
vgf2p8affineqb zmm7,zmm12,zmm3,0x0
vpxorq zmm7,zmm7,zmm2
vgf2p8affineqb zmm3,zmm1,zmm3,0x0
vmovdqu64 zmm2,ZMMWORD PTR [rcx+0x340]
vpbroadcastq zmm8,r10
vmovdqu64 zmm10,ZMMWORD PTR [rcx+0x280]
vmovdqu64 zmm15,ZMMWORD PTR [rcx+0x300]
vgf2p8affineqb zmm17,zmm8,zmm15,0x0
vgf2p8affineqb zmm18,zmm14,zmm2,0x0
vpternlogq zmm17,zmm7,zmm18,0x96
vgf2p8affineqb zmm2,zmm1,zmm2,0x0
vpbroadcastq zmm7,r9
vpbroadcastq zmm18,rax
vmovdqu64 zmm19,ZMMWORD PTR [rcx+0x2c0]
vgf2p8affineqb zmm20,zmm7,zmm19,0x0
vgf2p8affineqb zmm19,zmm1,zmm19,0x0
vgf2p8affineqb zmm21,zmm12,zmm15,0x0
vpxorq zmm21,zmm21,zmm2
vgf2p8affineqb zmm2,zmm12,zmm10,0x0
vpxorq zmm2,zmm2,zmm19
vgf2p8affineqb zmm19,zmm14,zmm15,0x0
vpxorq zmm19,zmm19,zmm3
vgf2p8affineqb zmm15,zmm1,zmm15,0x0
vgf2p8affineqb zmm3,zmm18,zmm10,0x0
vpternlogq zmm3,zmm17,zmm20,0x96
vgf2p8mulb zmm4,zmm4,zmm3
vpxorq zmm25,zmm25,zmm4
vmovdqu64 zmm4,ZMMWORD PTR [rcx+0x240]
vgf2p8affineqb zmm17,zmm1,zmm4,0x0
vgf2p8affineqb zmm20,zmm7,zmm10,0x0
vpternlogq zmm20,zmm3,zmm19,0x96
vgf2p8affineqb zmm10,zmm1,zmm10,0x0
vmovdqu64 zmm19,ZMMWORD PTR [rcx+0x200]
vgf2p8affineqb zmm18,zmm18,zmm19,0x0
vgf2p8affineqb zmm8,zmm8,zmm19,0x0
vgf2p8affineqb zmm1,zmm1,zmm19,0x0
vgf2p8affineqb zmm12,zmm12,zmm19,0x0
vpxorq zmm12,zmm12,zmm17
vpxorq zmm1,zmm12,zmm1
vgf2p8mulb zmm9,zmm9,zmm20
vpxorq zmm24,zmm24,zmm9
vgf2p8affineqb zmm9,zmm14,zmm19,0x0
vgf2p8affineqb zmm19,zmm7,zmm19,0x0
vpxorq zmm19,zmm19,zmm15
vgf2p8affineqb zmm7,zmm7,zmm4,0x0
vpternlogq zmm18,zmm21,zmm7,0x96
vgf2p8affineqb zmm14,zmm14,zmm4,0x0
vpternlogq zmm8,zmm2,zmm14,0x96
vpternlogq zmm10,zmm8,zmm9,0x96
vpternlogq zmm19,zmm20,zmm18,0x96
vpxorq zmm18,zmm3,zmm18
vpxorq zmm3,zmm3,zmm8
vpternlogq zmm8,zmm18,zmm12,0x96
vgf2p8mulb zmm13,zmm13,zmm8
vpxorq zmm31,zmm31,zmm13
vgf2p8mulb zmm0,zmm0,zmm3
vpxorq zmm29,zmm29,zmm0
vgf2p8mulb zmm16,zmm16,zmm18
vpxorq zmm27,zmm27,zmm16
vpxorq zmm20,zmm20,zmm10
vgf2p8mulb zmm5,zmm5,zmm20
vpxorq zmm28,zmm28,zmm5
vpternlogq zmm1,zmm19,zmm10,0x96
vgf2p8mulb zmm11,zmm11,zmm19
vpxorq zmm26,zmm26,zmm11
vgf2p8mulb zmm6,zmm6,zmm1
vpxorq zmm30,zmm30,zmm6
ret
.size flock_partial_gpr_free2_next,.-flock_partial_gpr_free2_next

.section .note.GNU-stack,"",@progbits
