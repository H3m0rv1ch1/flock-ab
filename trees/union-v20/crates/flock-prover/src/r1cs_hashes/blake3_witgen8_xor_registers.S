.text
.p2align 6
.globl flock_family19_mixedp20_first
.hidden flock_family19_mixedp20_first
.type flock_family19_mixedp20_first,@function
flock_family19_mixedp20_first:
vpbroadcastq zmm0,QWORD PTR [rdi+0x38]
vpbroadcastq zmm1,QWORD PTR [rdi+0x8]
vpbroadcastq zmm2,QWORD PTR [rdi+0x10]
vpbroadcastq zmm3,QWORD PTR [rdi+0x28]
vpbroadcastq zmm4,QWORD PTR [rdi+0x20]
vmovdqa64 zmm5,zmm1
vpternlogq zmm5,zmm2,zmm4,0x96
vmovdqu64 zmm6,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm0,zmm0,zmm6,0x0
vpbroadcastq zmm7,QWORD PTR [rdi+0x18]
vpbroadcastq zmm8,QWORD PTR [rdi+0x30]
vpbroadcastq zmm9,QWORD PTR [rdi]
vgf2p8affineqb zmm10,zmm1,ZMMWORD PTR [rdx+0x800],0x0
vgf2p8affineqb zmm11,zmm9,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm12,zmm5,ZMMWORD PTR [rdx+0x500],0x0
vpternlogq zmm10,zmm0,zmm12,0x96
vgf2p8affineqb zmm5,zmm5,zmm6,0x0
vmovdqa64 zmm0,zmm7
vpternlogq zmm0,zmm3,zmm8,0x96
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x440],0x0
vpternlogq zmm10,zmm11,zmm0,0x96
vgf2p8affineqb zmm3,zmm3,zmm6,0x0
vgf2p8affineqb zmm6,zmm9,zmm6,0x0
vgf2p8affineqb zmm0,zmm9,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm0,zmm3,zmm10,0x96
vpxorq zmm3,zmm4,zmm9
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm11,zmm7,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm7,zmm1,zmm2,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x5c0],0x0
vpxorq zmm12,zmm2,zmm9
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x480],0x0
vpternlogq zmm0,zmm11,zmm2,0x96
vpternlogq zmm7,zmm2,zmm10,0x96
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm10,zmm11,zmm12,0x96
vgf2p8affineqb zmm2,zmm8,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm8,zmm1,zmm4,0x96
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x840],0x0
vmovdqa64 zmm11,zmm5
vpternlogq zmm11,zmm2,zmm3,0x96
vpternlogq zmm5,zmm8,zmm10,0x96
vpxorq zmm12,zmm1,zmm9
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x4c0],0x0
vgf2p8affineqb zmm13,zmm9,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x7c0],0x0
vmovdqa64 zmm14,zmm4
vpternlogq zmm14,zmm13,zmm9,0x96
vpternlogq zmm4,zmm11,zmm0,0x96
vpternlogq zmm2,zmm14,zmm1,0x96
vpternlogq zmm12,zmm8,zmm14,0x96
vpternlogq zmm0,zmm6,zmm12,0x96
vpternlogq zmm3,zmm12,zmm7,0x96
vpternlogq zmm13,zmm11,zmm7,0x96
vpternlogq zmm7,zmm10,zmm4,0x96
vpxorq zmm2,zmm2,zmm10
vpternlogq zmm1,zmm9,zmm7,0x96
vpternlogq zmm7,zmm8,zmm11,0x96
vpbroadcastq zmm6,QWORD PTR [rsi+0x20]
vpbroadcastq zmm8,QWORD PTR [rsi+0x38]
vpbroadcastq zmm9,QWORD PTR [rsi+0x10]
vpbroadcastq zmm10,QWORD PTR [rsi+0x8]
vmovdqa64 zmm11,zmm10
vpternlogq zmm11,zmm9,zmm6,0x96
vpbroadcastq zmm12,QWORD PTR [rsi+0x18]
vpbroadcastq zmm14,QWORD PTR [rsi+0x30]
vgf2p8affineqb zmm15,zmm10,ZMMWORD PTR [rcx+0x800],0x0
vpbroadcastq zmm16,QWORD PTR [rsi+0x28]
vmovdqu64 zmm17,ZMMWORD PTR [rcx+0x740]
vgf2p8affineqb zmm8,zmm8,zmm17,0x0
vmovdqa64 zmm18,zmm12
vpternlogq zmm18,zmm16,zmm14,0x96
vgf2p8affineqb zmm18,zmm18,ZMMWORD PTR [rcx+0x440],0x0
vgf2p8affineqb zmm16,zmm16,zmm17,0x0
vgf2p8affineqb zmm19,zmm11,ZMMWORD PTR [rcx+0x500],0x0
vpternlogq zmm15,zmm8,zmm19,0x96
vgf2p8affineqb zmm11,zmm11,zmm17,0x0
vpbroadcastq zmm8,QWORD PTR [rsi]
vgf2p8affineqb zmm17,zmm8,zmm17,0x0
vgf2p8affineqb zmm19,zmm8,ZMMWORD PTR [rcx+0x540],0x0
vpternlogq zmm15,zmm19,zmm18,0x96
vgf2p8affineqb zmm18,zmm9,ZMMWORD PTR [rcx+0x480],0x0
vgf2p8affineqb zmm19,zmm8,ZMMWORD PTR [rcx+0x6c0],0x0
vpternlogq zmm19,zmm16,zmm15,0x96
vpxorq zmm16,zmm9,zmm8
vpternlogq zmm9,zmm10,zmm12,0x96
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rcx+0x640],0x0
vgf2p8affineqb zmm16,zmm16,ZMMWORD PTR [rcx+0x400],0x0
vpternlogq zmm19,zmm12,zmm18,0x96
vpternlogq zmm12,zmm15,zmm16,0x96
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rcx+0x5c0],0x0
vpternlogq zmm9,zmm18,zmm15,0x96
vpxorq zmm15,zmm6,zmm8
vgf2p8affineqb zmm15,zmm15,ZMMWORD PTR [rcx+0x700],0x0
vgf2p8affineqb zmm16,zmm14,ZMMWORD PTR [rcx+0x680],0x0
vpternlogq zmm14,zmm10,zmm6,0x96
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rcx+0x580],0x0
vgf2p8affineqb zmm14,zmm14,ZMMWORD PTR [rcx+0x840],0x0
vmovdqa64 zmm18,zmm11
vpternlogq zmm18,zmm16,zmm15,0x96
vpternlogq zmm11,zmm14,zmm12,0x96
vgf2p8mulb zmm30,zmm5,zmm11
vgf2p8affineqb zmm5,zmm8,ZMMWORD PTR [rcx+0x600],0x0
vmovdqa64 zmm11,zmm6
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8mulb zmm25,zmm4,zmm11
vpternlogq zmm11,zmm12,zmm9,0x96
vpxorq zmm4,zmm10,zmm8
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rcx+0x7c0],0x0
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rcx+0x880],0x0
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rcx+0x4c0],0x0
vpternlogq zmm6,zmm5,zmm8,0x96
vpternlogq zmm4,zmm14,zmm6,0x96
vpternlogq zmm6,zmm10,zmm16,0x96
vpternlogq zmm19,zmm17,zmm4,0x96
vpternlogq zmm15,zmm4,zmm9,0x96
vpternlogq zmm10,zmm8,zmm11,0x96
vpternlogq zmm11,zmm14,zmm18,0x96
vpternlogq zmm5,zmm18,zmm9,0x96
vgf2p8mulb zmm24,zmm7,zmm11
vpxorq zmm6,zmm6,zmm12
vgf2p8mulb zmm27,zmm2,zmm6
vgf2p8mulb zmm29,zmm1,zmm10
vgf2p8mulb zmm31,zmm13,zmm5
vgf2p8mulb zmm26,zmm3,zmm15
vgf2p8mulb zmm28,zmm0,zmm19
ret
.size flock_family19_mixedp20_first, .-flock_family19_mixedp20_first

.p2align 6
.globl flock_family19_mixedp20_next
.hidden flock_family19_mixedp20_next
.type flock_family19_mixedp20_next,@function
flock_family19_mixedp20_next:
vpbroadcastq zmm0,QWORD PTR [rdi+0x38]
vpbroadcastq zmm1,QWORD PTR [rdi+0x8]
vpbroadcastq zmm2,QWORD PTR [rdi+0x10]
vpbroadcastq zmm3,QWORD PTR [rdi+0x28]
vpbroadcastq zmm4,QWORD PTR [rdi+0x20]
vmovdqa64 zmm5,zmm1
vpternlogq zmm5,zmm2,zmm4,0x96
vmovdqu64 zmm6,ZMMWORD PTR [rdx+0x740]
vgf2p8affineqb zmm0,zmm0,zmm6,0x0
vpbroadcastq zmm7,QWORD PTR [rdi+0x18]
vpbroadcastq zmm8,QWORD PTR [rdi+0x30]
vpbroadcastq zmm9,QWORD PTR [rdi]
vgf2p8affineqb zmm10,zmm1,ZMMWORD PTR [rdx+0x800],0x0
vgf2p8affineqb zmm11,zmm9,ZMMWORD PTR [rdx+0x540],0x0
vgf2p8affineqb zmm12,zmm5,ZMMWORD PTR [rdx+0x500],0x0
vpternlogq zmm10,zmm0,zmm12,0x96
vgf2p8affineqb zmm5,zmm5,zmm6,0x0
vmovdqa64 zmm0,zmm7
vpternlogq zmm0,zmm3,zmm8,0x96
vgf2p8affineqb zmm0,zmm0,ZMMWORD PTR [rdx+0x440],0x0
vpternlogq zmm10,zmm11,zmm0,0x96
vgf2p8affineqb zmm3,zmm3,zmm6,0x0
vgf2p8affineqb zmm6,zmm9,zmm6,0x0
vgf2p8affineqb zmm0,zmm9,ZMMWORD PTR [rdx+0x6c0],0x0
vpternlogq zmm0,zmm3,zmm10,0x96
vpxorq zmm3,zmm4,zmm9
vgf2p8affineqb zmm3,zmm3,ZMMWORD PTR [rdx+0x700],0x0
vgf2p8affineqb zmm11,zmm7,ZMMWORD PTR [rdx+0x640],0x0
vpternlogq zmm7,zmm1,zmm2,0x96
vgf2p8affineqb zmm7,zmm7,ZMMWORD PTR [rdx+0x5c0],0x0
vpxorq zmm12,zmm2,zmm9
vgf2p8affineqb zmm2,zmm2,ZMMWORD PTR [rdx+0x480],0x0
vpternlogq zmm0,zmm11,zmm2,0x96
vpternlogq zmm7,zmm2,zmm10,0x96
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x400],0x0
vpternlogq zmm10,zmm11,zmm12,0x96
vgf2p8affineqb zmm2,zmm8,ZMMWORD PTR [rdx+0x680],0x0
vpternlogq zmm8,zmm1,zmm4,0x96
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rdx+0x580],0x0
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rdx+0x840],0x0
vmovdqa64 zmm11,zmm5
vpternlogq zmm11,zmm2,zmm3,0x96
vpternlogq zmm5,zmm8,zmm10,0x96
vpxorq zmm12,zmm1,zmm9
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rdx+0x880],0x0
vgf2p8affineqb zmm1,zmm1,ZMMWORD PTR [rdx+0x4c0],0x0
vgf2p8affineqb zmm13,zmm9,ZMMWORD PTR [rdx+0x600],0x0
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rdx+0x7c0],0x0
vmovdqa64 zmm14,zmm4
vpternlogq zmm14,zmm13,zmm9,0x96
vpternlogq zmm4,zmm11,zmm0,0x96
vpternlogq zmm2,zmm14,zmm1,0x96
vpternlogq zmm12,zmm8,zmm14,0x96
vpternlogq zmm0,zmm6,zmm12,0x96
vpternlogq zmm3,zmm12,zmm7,0x96
vpternlogq zmm13,zmm11,zmm7,0x96
vpternlogq zmm7,zmm10,zmm4,0x96
vpxorq zmm2,zmm2,zmm10
vpternlogq zmm1,zmm9,zmm7,0x96
vpternlogq zmm7,zmm8,zmm11,0x96
vpbroadcastq zmm6,QWORD PTR [rsi+0x20]
vpbroadcastq zmm8,QWORD PTR [rsi+0x38]
vpbroadcastq zmm9,QWORD PTR [rsi+0x10]
vpbroadcastq zmm10,QWORD PTR [rsi+0x8]
vmovdqa64 zmm11,zmm10
vpternlogq zmm11,zmm9,zmm6,0x96
vpbroadcastq zmm12,QWORD PTR [rsi+0x18]
vpbroadcastq zmm14,QWORD PTR [rsi+0x30]
vgf2p8affineqb zmm15,zmm10,ZMMWORD PTR [rcx+0x800],0x0
vpbroadcastq zmm16,QWORD PTR [rsi+0x28]
vmovdqu64 zmm17,ZMMWORD PTR [rcx+0x740]
vgf2p8affineqb zmm8,zmm8,zmm17,0x0
vmovdqa64 zmm18,zmm12
vpternlogq zmm18,zmm16,zmm14,0x96
vgf2p8affineqb zmm18,zmm18,ZMMWORD PTR [rcx+0x440],0x0
vgf2p8affineqb zmm16,zmm16,zmm17,0x0
vgf2p8affineqb zmm19,zmm11,ZMMWORD PTR [rcx+0x500],0x0
vpternlogq zmm15,zmm8,zmm19,0x96
vgf2p8affineqb zmm11,zmm11,zmm17,0x0
vpbroadcastq zmm8,QWORD PTR [rsi]
vgf2p8affineqb zmm17,zmm8,zmm17,0x0
vgf2p8affineqb zmm19,zmm8,ZMMWORD PTR [rcx+0x540],0x0
vpternlogq zmm15,zmm19,zmm18,0x96
vgf2p8affineqb zmm18,zmm9,ZMMWORD PTR [rcx+0x480],0x0
vgf2p8affineqb zmm19,zmm8,ZMMWORD PTR [rcx+0x6c0],0x0
vpternlogq zmm19,zmm16,zmm15,0x96
vpxorq zmm16,zmm9,zmm8
vpternlogq zmm9,zmm10,zmm12,0x96
vgf2p8affineqb zmm12,zmm12,ZMMWORD PTR [rcx+0x640],0x0
vgf2p8affineqb zmm16,zmm16,ZMMWORD PTR [rcx+0x400],0x0
vpternlogq zmm19,zmm12,zmm18,0x96
vpternlogq zmm12,zmm15,zmm16,0x96
vgf2p8affineqb zmm9,zmm9,ZMMWORD PTR [rcx+0x5c0],0x0
vpternlogq zmm9,zmm18,zmm15,0x96
vpxorq zmm15,zmm6,zmm8
vgf2p8affineqb zmm15,zmm15,ZMMWORD PTR [rcx+0x700],0x0
vgf2p8affineqb zmm16,zmm14,ZMMWORD PTR [rcx+0x680],0x0
vpternlogq zmm14,zmm10,zmm6,0x96
vgf2p8affineqb zmm6,zmm6,ZMMWORD PTR [rcx+0x580],0x0
vgf2p8affineqb zmm14,zmm14,ZMMWORD PTR [rcx+0x840],0x0
vmovdqa64 zmm18,zmm11
vpternlogq zmm18,zmm16,zmm15,0x96
vpternlogq zmm11,zmm14,zmm12,0x96
vgf2p8mulb zmm5,zmm5,zmm11
vpxorq zmm30,zmm30,zmm5
vgf2p8affineqb zmm5,zmm8,ZMMWORD PTR [rcx+0x600],0x0
vmovdqa64 zmm11,zmm6
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8mulb zmm4,zmm4,zmm11
vpxorq zmm25,zmm25,zmm4
vpternlogq zmm11,zmm12,zmm9,0x96
vpxorq zmm4,zmm10,zmm8
vgf2p8affineqb zmm8,zmm8,ZMMWORD PTR [rcx+0x7c0],0x0
vgf2p8affineqb zmm4,zmm4,ZMMWORD PTR [rcx+0x880],0x0
vgf2p8affineqb zmm10,zmm10,ZMMWORD PTR [rcx+0x4c0],0x0
vpternlogq zmm6,zmm5,zmm8,0x96
vpternlogq zmm4,zmm14,zmm6,0x96
vpternlogq zmm6,zmm10,zmm16,0x96
vpternlogq zmm19,zmm17,zmm4,0x96
vpternlogq zmm15,zmm4,zmm9,0x96
vpternlogq zmm10,zmm8,zmm11,0x96
vpternlogq zmm11,zmm14,zmm18,0x96
vpternlogq zmm5,zmm18,zmm9,0x96
vgf2p8mulb zmm7,zmm7,zmm11
vpxorq zmm24,zmm24,zmm7
vpxorq zmm6,zmm6,zmm12
vgf2p8mulb zmm2,zmm2,zmm6
vpxorq zmm27,zmm27,zmm2
vgf2p8mulb zmm1,zmm1,zmm10
vpxorq zmm29,zmm29,zmm1
vgf2p8mulb zmm13,zmm13,zmm5
vpxorq zmm31,zmm31,zmm13
vgf2p8mulb zmm3,zmm3,zmm15
vpxorq zmm26,zmm26,zmm3
vgf2p8mulb zmm0,zmm0,zmm19
vpxorq zmm28,zmm28,zmm0
ret
.size flock_family19_mixedp20_next, .-flock_family19_mixedp20_next

.section .note.GNU-stack,"",@progbits
