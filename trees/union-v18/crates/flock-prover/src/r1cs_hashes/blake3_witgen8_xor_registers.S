.text

.p2align 6
.globl flock_basis20_scheduled_first
.hidden flock_basis20_scheduled_first
.type flock_basis20_scheduled_first,@function
flock_basis20_scheduled_first:
vpbroadcastq zmm0,QWORD PTR [rdi+0x28]
vpbroadcastq zmm1,QWORD PTR [rdi+0x8]
vpbroadcastq zmm2,QWORD PTR [rdi+0x10]
vmovdqu64 zmm3,ZMMWORD PTR [rdx+0x740]
vpbroadcastq zmm4,QWORD PTR [rdi]
vpbroadcastq zmm5,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm5,zmm5,zmm3,0x0
vgf2p8affineqb zmm6,zmm0,zmm3,0x0
vpbroadcastq zmm7,QWORD PTR [rdi+0x20]
vgf2p8affineqb zmm8,zmm4,ZMMWORD PTR [rdx+0x600],0x0
vmovdqa64 zmm9,zmm1
vpternlogq zmm9,zmm2,zmm7,0x96
vpxorq zmm10,zmm7,zmm4
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm11,zmm4,zmm3,0x0
vgf2p8affineqb zmm3,zmm9,zmm3,0x0
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x540],0x0
vpbroadcastq zmm12,QWORD PTR [rdi+0x18]
vpbroadcastq zmm13,QWORD PTR [rdi+0x30]
vpternlogq zmm0,zmm12,zmm13,0x96
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm14,zmm13,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm13,zmm1,zmm7,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm13,zmm13,ZMMWORD PTR [rdx+0x840],0x0
vpxorq zmm15,zmm1,zmm4
vgf2p8affineqb zmm15,zmm15,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm16,zmm4,ZMMWORD PTR [rdx+0x780],0x0
vpternlogq zmm9,zmm16,zmm5,0x96
vgf2p8affineqb zmm5,zmm1,ZMMWORD PTR [rdx+0x4c0],0x0
vpxorq zmm16,zmm2,zmm4
vgf2p8affineqb zmm16,zmm16,ZMMWORD PTR [rdx+0x400],0x0
vmovdqa64 zmm17,zmm1
vpternlogq zmm17,zmm2,zmm12,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x480],0x0
vgf2p8affineqb zmm17,zmm17,ZMMWORD PTR [rdx+0x5c0],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm1,zmm0,zmm9,0x96
vpternlogq zmm8,zmm6,zmm1,0x96
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x640],0x0
vgf2p8affineqb zmm0,zmm4,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm4,zmm11,zmm7,0x96
vpternlogq zmm11,zmm3,zmm13,0x96
vpxorq zmm5,zmm4,zmm5
vpternlogq zmm4,zmm13,zmm15,0x96
vpternlogq zmm3,zmm14,zmm10,0x96
vmovdqa64 zmm6,zmm8
vpternlogq zmm6,zmm2,zmm12,0x96
vpternlogq zmm2,zmm1,zmm17,0x96
vpternlogq zmm12,zmm16,zmm1,0x96
vpternlogq zmm16,zmm17,zmm8,0x96
vpternlogq zmm10,zmm2,zmm4,0x96
vpternlogq zmm13,zmm16,zmm7,0x96
vpternlogq zmm16,zmm5,zmm3,0x96
vpternlogq zmm14,zmm12,zmm5,0x96
vpternlogq zmm11,zmm0,zmm12,0x96
vpternlogq zmm4,zmm0,zmm6,0x96
vpternlogq zmm7,zmm3,zmm6,0x96
vpxorq zmm2,zmm2,zmm3
vpbroadcastq zmm0,QWORD PTR [rsi+0x28]
vpbroadcastq zmm1,QWORD PTR [rsi+0x8]
vpbroadcastq zmm3,QWORD PTR [rsi+0x38]
vmovdqu64 zmm5,ZMMWORD PTR [rcx+0x740]
vgf2p8affineqb zmm3,zmm3,zmm5,0x0
vpbroadcastq zmm6,QWORD PTR [rsi]
vgf2p8affineqb zmm8,zmm6,ZMMWORD PTR [rcx+0x780],0x0
vpbroadcastq zmm9,QWORD PTR [rsi+0x18]
vpbroadcastq zmm12,QWORD PTR [rsi+0x20]
vgf2p8affineqb zmm15,zmm1,ZMMWORD PTR [rcx+0x800],0x0
vgf2p8affineqb zmm17,zmm0,zmm5,0x0
vpbroadcastq zmm18,QWORD PTR [rsi+0x10]
vmovdqa64 zmm19,zmm1
vpternlogq zmm19,zmm18,zmm12,0x96
vgf2p8affineqb zmm20,zmm6,ZMMWORD PTR [rcx+0x600],0x0
vgf2p8affineqb zmm21,zmm19,ZMMWORD PTR [rcx+0x540],0x0
vpternlogq zmm21,zmm8,zmm3,0x96
vgf2p8affineqb zmm19,zmm19,zmm5,0x0
vgf2p8affineqb zmm5,zmm6,zmm5,0x0
vpbroadcastq zmm3,QWORD PTR [rsi+0x30]
vpternlogq zmm0,zmm9,zmm3,0x96
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rcx+0x440],0x0
vpternlogq zmm15,zmm0,zmm21,0x96
vpternlogq zmm20,zmm17,zmm15,0x96
vpxorq zmm0,zmm1,zmm6
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rcx+0x880],0x0
vgf2p8affineqb zmm8,zmm18,ZMMWORD PTR [rcx+0x480],0x0
vpxorq zmm17,zmm18,zmm6
vgf2p8affineqb zmm17,zmm17,ZMMWORD PTR [rcx+0x400],0x0
vpternlogq zmm18,zmm1,zmm9,0x96
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rcx+0x640],0x0
vgf2p8affineqb zmm18,zmm18,ZMMWORD PTR [rcx+0x5c0],0x0
vmovdqa64 zmm21,zmm20
vpternlogq zmm21,zmm8,zmm9,0x96
vpternlogq zmm20,zmm17,zmm18,0x96
vpternlogq zmm17,zmm9,zmm15,0x96
vpternlogq zmm15,zmm8,zmm18,0x96
vgf2p8affineqb zmm8,zmm12,ZMMWORD PTR [rcx+0x580],0x0
vgf2p8affineqb zmm9,zmm1,ZMMWORD PTR [rcx+0x4c0],0x0
vpternlogq zmm1,zmm12,zmm3,0x96
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rcx+0x840],0x0
vpxorq zmm12,zmm12,zmm6
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rcx+0x700],0x0
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rcx+0x680],0x0
vgf2p8affineqb zmm18,zmm6,ZMMWORD PTR [rcx+0x6c0],0x0
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rcx+0x500],0x0
vpternlogq zmm18,zmm5,zmm8,0x96
vpxorq zmm9,zmm18,zmm9
vpternlogq zmm18,zmm1,zmm0,0x96
vpternlogq zmm5,zmm19,zmm1,0x96
vpternlogq zmm19,zmm3,zmm12,0x96
vpternlogq zmm3,zmm17,zmm9,0x96
vpternlogq zmm5,zmm6,zmm17,0x96
vgf2p8mulb zmm30,zmm11,zmm5
vgf2p8mulb zmm27,zmm14,zmm3
vpternlogq zmm9,zmm20,zmm19,0x96
vpternlogq zmm20,zmm8,zmm1,0x96
vgf2p8mulb zmm29,zmm16,zmm9
vgf2p8mulb zmm24,zmm13,zmm20
vpternlogq zmm8,zmm19,zmm21,0x96
vpternlogq zmm6,zmm18,zmm21,0x96
vgf2p8mulb zmm28,zmm4,zmm6
vpternlogq zmm12,zmm15,zmm18,0x96
vpxorq zmm15,zmm15,zmm19
vgf2p8mulb zmm25,zmm7,zmm8
vgf2p8mulb zmm31,zmm2,zmm15
vgf2p8mulb zmm26,zmm10,zmm12
ret
.size flock_basis20_scheduled_first, .-flock_basis20_scheduled_first

.p2align 6
.globl flock_basis20_scheduled_next
.hidden flock_basis20_scheduled_next
.type flock_basis20_scheduled_next,@function
flock_basis20_scheduled_next:
vpbroadcastq zmm0,QWORD PTR [rdi+0x28]
vpbroadcastq zmm1,QWORD PTR [rdi+0x8]
vpbroadcastq zmm2,QWORD PTR [rdi+0x10]
vmovdqu64 zmm3,ZMMWORD PTR [rdx+0x740]
vpbroadcastq zmm4,QWORD PTR [rdi]
vpbroadcastq zmm5,QWORD PTR [rdi+0x38]
vgf2p8affineqb zmm5,zmm5,zmm3,0x0
vgf2p8affineqb zmm6,zmm0,zmm3,0x0
vpbroadcastq zmm7,QWORD PTR [rdi+0x20]
vgf2p8affineqb zmm8,zmm4,ZMMWORD PTR [rdx+0x600],0x0
vmovdqa64 zmm9,zmm1
vpternlogq zmm9,zmm2,zmm7,0x96
vpxorq zmm10,zmm7,zmm4
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm11,zmm4,zmm3,0x0
vgf2p8affineqb zmm3,zmm9,zmm3,0x0
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x540],0x0
vpbroadcastq zmm12,QWORD PTR [rdi+0x18]
vpbroadcastq zmm13,QWORD PTR [rdi+0x30]
vpternlogq zmm0,zmm12,zmm13,0x96
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x440],0x0
vgf2p8affineqb zmm14,zmm13,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm13,zmm1,zmm7,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm13,zmm13,ZMMWORD PTR [rdx+0x840],0x0
vpxorq zmm15,zmm1,zmm4
vgf2p8affineqb zmm15,zmm15,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm16,zmm4,ZMMWORD PTR [rdx+0x780],0x0
vpternlogq zmm9,zmm16,zmm5,0x96
vgf2p8affineqb zmm5,zmm1,ZMMWORD PTR [rdx+0x4c0],0x0
vpxorq zmm16,zmm2,zmm4
vgf2p8affineqb zmm16,zmm16,ZMMWORD PTR [rdx+0x400],0x0
vmovdqa64 zmm17,zmm1
vpternlogq zmm17,zmm2,zmm12,0x96
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x480],0x0
vgf2p8affineqb zmm17,zmm17,ZMMWORD PTR [rdx+0x5c0],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x800],0x0
vpternlogq zmm1,zmm0,zmm9,0x96
vpternlogq zmm8,zmm6,zmm1,0x96
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x640],0x0
vgf2p8affineqb zmm0,zmm4,ZMMWORD PTR [rdx+0x500],0x0
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm4,zmm11,zmm7,0x96
vpternlogq zmm11,zmm3,zmm13,0x96
vpxorq zmm5,zmm4,zmm5
vpternlogq zmm4,zmm13,zmm15,0x96
vpternlogq zmm3,zmm14,zmm10,0x96
vmovdqa64 zmm6,zmm8
vpternlogq zmm6,zmm2,zmm12,0x96
vpternlogq zmm2,zmm1,zmm17,0x96
vpternlogq zmm12,zmm16,zmm1,0x96
vpternlogq zmm16,zmm17,zmm8,0x96
vpternlogq zmm10,zmm2,zmm4,0x96
vpternlogq zmm13,zmm16,zmm7,0x96
vpternlogq zmm16,zmm5,zmm3,0x96
vpternlogq zmm14,zmm12,zmm5,0x96
vpternlogq zmm11,zmm0,zmm12,0x96
vpternlogq zmm4,zmm0,zmm6,0x96
vpternlogq zmm7,zmm3,zmm6,0x96
vpxorq zmm2,zmm2,zmm3
vpbroadcastq zmm0,QWORD PTR [rsi+0x28]
vpbroadcastq zmm1,QWORD PTR [rsi+0x8]
vpbroadcastq zmm3,QWORD PTR [rsi+0x38]
vmovdqu64 zmm5,ZMMWORD PTR [rcx+0x740]
vgf2p8affineqb zmm3,zmm3,zmm5,0x0
vpbroadcastq zmm6,QWORD PTR [rsi]
vgf2p8affineqb zmm8,zmm6,ZMMWORD PTR [rcx+0x780],0x0
vpbroadcastq zmm9,QWORD PTR [rsi+0x18]
vpbroadcastq zmm12,QWORD PTR [rsi+0x20]
vgf2p8affineqb zmm15,zmm1,ZMMWORD PTR [rcx+0x800],0x0
vgf2p8affineqb zmm17,zmm0,zmm5,0x0
vpbroadcastq zmm18,QWORD PTR [rsi+0x10]
vmovdqa64 zmm19,zmm1
vpternlogq zmm19,zmm18,zmm12,0x96
vgf2p8affineqb zmm20,zmm6,ZMMWORD PTR [rcx+0x600],0x0
vgf2p8affineqb zmm21,zmm19,ZMMWORD PTR [rcx+0x540],0x0
vpternlogq zmm21,zmm8,zmm3,0x96
vgf2p8affineqb zmm19,zmm19,zmm5,0x0
vgf2p8affineqb zmm5,zmm6,zmm5,0x0
vpbroadcastq zmm3,QWORD PTR [rsi+0x30]
vpternlogq zmm0,zmm9,zmm3,0x96
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rcx+0x440],0x0
vpternlogq zmm15,zmm0,zmm21,0x96
vpternlogq zmm20,zmm17,zmm15,0x96
vpxorq zmm0,zmm1,zmm6
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rcx+0x880],0x0
vgf2p8affineqb zmm8,zmm18,ZMMWORD PTR [rcx+0x480],0x0
vpxorq zmm17,zmm18,zmm6
vgf2p8affineqb zmm17,zmm17,ZMMWORD PTR [rcx+0x400],0x0
vpternlogq zmm18,zmm1,zmm9,0x96
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rcx+0x640],0x0
vgf2p8affineqb zmm18,zmm18,ZMMWORD PTR [rcx+0x5c0],0x0
vmovdqa64 zmm21,zmm20
vpternlogq zmm21,zmm8,zmm9,0x96
vpternlogq zmm20,zmm17,zmm18,0x96
vpternlogq zmm17,zmm9,zmm15,0x96
vpternlogq zmm15,zmm8,zmm18,0x96
vgf2p8affineqb zmm8,zmm12,ZMMWORD PTR [rcx+0x580],0x0
vgf2p8affineqb zmm9,zmm1,ZMMWORD PTR [rcx+0x4c0],0x0
vpternlogq zmm1,zmm12,zmm3,0x96
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rcx+0x840],0x0
vpxorq zmm12,zmm12,zmm6
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rcx+0x700],0x0
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rcx+0x680],0x0
vgf2p8affineqb zmm18,zmm6,ZMMWORD PTR [rcx+0x6c0],0x0
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rcx+0x500],0x0
vpternlogq zmm18,zmm5,zmm8,0x96
vpxorq zmm9,zmm18,zmm9
vpternlogq zmm18,zmm1,zmm0,0x96
vpternlogq zmm5,zmm19,zmm1,0x96
vpternlogq zmm19,zmm3,zmm12,0x96
vpternlogq zmm3,zmm17,zmm9,0x96
vpternlogq zmm5,zmm6,zmm17,0x96
vgf2p8mulb zmm11,zmm11,zmm5
vpxorq zmm30,zmm30,zmm11
vgf2p8mulb zmm14,zmm14,zmm3
vpxorq zmm27,zmm27,zmm14
vpternlogq zmm9,zmm20,zmm19,0x96
vpternlogq zmm20,zmm8,zmm1,0x96
vgf2p8mulb zmm16,zmm16,zmm9
vpxorq zmm29,zmm29,zmm16
vgf2p8mulb zmm13,zmm13,zmm20
vpxorq zmm24,zmm24,zmm13
vpternlogq zmm8,zmm19,zmm21,0x96
vpternlogq zmm6,zmm18,zmm21,0x96
vgf2p8mulb zmm4,zmm4,zmm6
vpxorq zmm28,zmm28,zmm4
vpternlogq zmm12,zmm15,zmm18,0x96
vpxorq zmm15,zmm15,zmm19
vgf2p8mulb zmm7,zmm7,zmm8
vpxorq zmm25,zmm25,zmm7
vgf2p8mulb zmm2,zmm2,zmm15
vpxorq zmm31,zmm31,zmm2
vgf2p8mulb zmm10,zmm10,zmm12
vpxorq zmm26,zmm26,zmm10
ret
.size flock_basis20_scheduled_next, .-flock_basis20_scheduled_next

.section .note.GNU-stack,"",@progbits
