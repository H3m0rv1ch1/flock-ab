.text
// Register-resident window sums for small sparse Karatsuba groups.
.p2align 5
.globl flock_ab_register_bank0_m0
.type flock_ab_register_bank0_m0, @function
flock_ab_register_bank0_m0:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 0]
mov ecx, 32
.p2align 5
.Lbank_0_0_0:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_0_0
vmovdqu64 ZMMWORD PTR [rdx + 0], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 64], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 128], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 192], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 256], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 320], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 384], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 448], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 512], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 2304]
mov ecx, 32
.p2align 5
.Lbank_0_0_1:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 320]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_0_1
vmovdqu64 ZMMWORD PTR [rdx + 576], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 640], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 704], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 768], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 832], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 896], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 960], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1024], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 1088], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 4608]
mov ecx, 32
.p2align 5
.Lbank_0_0_2:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 320]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_0_2
vmovdqu64 ZMMWORD PTR [rdx + 1152], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1216], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1280], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1344], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1408], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 1472], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 1536], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1600], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 1664], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 6912]
mov ecx, 32
.p2align 5
.Lbank_0_0_3:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 576]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_0_3
vmovdqu64 ZMMWORD PTR [rdx + 1728], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1792], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1856], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1920], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1984], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 2048], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 2112], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 2176], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 2240], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
mov r8, rdi
lea r9, [rsi + 9216]
mov ecx, 32
.p2align 5
.Lbank_0_0_4:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 832]
vpxorq zmm11, zmm9, zmm10
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
add r8, 118784
add r9, 48
dec ecx
jnz .Lbank_0_0_4
vmovdqu64 ZMMWORD PTR [rdx + 2304], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2368], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2432], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2496], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2560], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 2624], zmm5
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 10752]
mov ecx, 32
.p2align 5
.Lbank_0_0_5:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 512]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 576]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 832]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_0_5
vmovdqu64 ZMMWORD PTR [rdx + 2688], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2752], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2816], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2880], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2944], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3008], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3072], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3136], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3200], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 13056]
mov ecx, 32
.p2align 5
.Lbank_0_0_6:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 576]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_0_6
vmovdqu64 ZMMWORD PTR [rdx + 3264], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3328], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3392], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 3456], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 3520], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3584], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3648], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3712], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3776], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 15360]
mov ecx, 32
.p2align 5
.Lbank_0_0_7:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 256]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 320]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 832]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_0_7
vmovdqu64 ZMMWORD PTR [rdx + 3840], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3904], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3968], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4032], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4096], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4160], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4224], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4288], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4352], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 17664]
mov ecx, 32
.p2align 5
.Lbank_0_0_8:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 256]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 512]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 320]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 576]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 832]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 384]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_0_8
vmovdqu64 ZMMWORD PTR [rdx + 4416], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 4480], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 4544], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4608], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4672], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4736], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4800], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4864], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4928], zmm8
vzeroupper
ret
.size flock_ab_register_bank0_m0, .-flock_ab_register_bank0_m0
.p2align 5
.globl flock_ab_register_bank0_m1
.type flock_ab_register_bank0_m1, @function
flock_ab_register_bank0_m1:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 0]
mov ecx, 32
.p2align 5
.Lbank_0_1_0:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
prefetcht1 [r8 + 0]
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
prefetcht1 [r8 + 64]
prefetcht1 [r8 + 128]
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
prefetcht1 [r8 + 192]
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
prefetcht1 [r8 + 256]
prefetcht1 [r8 + 320]
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
prefetcht1 [r8 + 384]
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
prefetcht1 [r8 + 448]
prefetcht1 [r8 + 512]
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
prefetcht1 [r8 + 576]
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
prefetcht1 [r8 + 640]
prefetcht1 [r8 + 704]
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
prefetcht1 [r8 + 768]
prefetcht1 [r8 + 832]
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_1_0
vmovdqu64 ZMMWORD PTR [rdx + 0], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 64], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 128], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 192], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 256], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 320], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 384], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 448], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 512], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 2304]
mov ecx, 32
.p2align 5
.Lbank_0_1_1:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 320]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_1_1
vmovdqu64 ZMMWORD PTR [rdx + 576], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 640], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 704], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 768], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 832], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 896], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 960], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1024], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 1088], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 4608]
mov ecx, 32
.p2align 5
.Lbank_0_1_2:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 320]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_1_2
vmovdqu64 ZMMWORD PTR [rdx + 1152], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1216], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1280], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1344], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1408], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 1472], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 1536], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1600], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 1664], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 6912]
mov ecx, 32
.p2align 5
.Lbank_0_1_3:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 576]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_1_3
vmovdqu64 ZMMWORD PTR [rdx + 1728], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1792], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1856], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1920], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1984], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 2048], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 2112], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 2176], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 2240], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
mov r8, rdi
lea r9, [rsi + 9216]
mov ecx, 32
.p2align 5
.Lbank_0_1_4:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 832]
vpxorq zmm11, zmm9, zmm10
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
add r8, 118784
add r9, 48
dec ecx
jnz .Lbank_0_1_4
vmovdqu64 ZMMWORD PTR [rdx + 2304], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2368], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2432], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2496], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2560], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 2624], zmm5
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 10752]
mov ecx, 32
.p2align 5
.Lbank_0_1_5:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 512]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 576]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 832]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_1_5
vmovdqu64 ZMMWORD PTR [rdx + 2688], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2752], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2816], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2880], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2944], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3008], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3072], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3136], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3200], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 13056]
mov ecx, 32
.p2align 5
.Lbank_0_1_6:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 576]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_1_6
vmovdqu64 ZMMWORD PTR [rdx + 3264], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3328], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3392], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 3456], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 3520], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3584], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3648], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3712], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3776], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 15360]
mov ecx, 32
.p2align 5
.Lbank_0_1_7:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 256]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 320]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 832]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_1_7
vmovdqu64 ZMMWORD PTR [rdx + 3840], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3904], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3968], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4032], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4096], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4160], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4224], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4288], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4352], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 17664]
mov ecx, 32
.p2align 5
.Lbank_0_1_8:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 256]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 512]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 320]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 576]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 832]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 384]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_1_8
vmovdqu64 ZMMWORD PTR [rdx + 4416], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 4480], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 4544], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4608], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4672], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4736], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4800], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4864], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4928], zmm8
vzeroupper
ret
.size flock_ab_register_bank0_m1, .-flock_ab_register_bank0_m1
.p2align 5
.globl flock_ab_register_bank0_m2
.type flock_ab_register_bank0_m2, @function
flock_ab_register_bank0_m2:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 0]
mov ecx, 32
.p2align 5
.Lbank_0_2_0:
prefetcht1 [r8 + 0]
prefetcht1 [r8 + 64]
prefetcht1 [r8 + 128]
prefetcht1 [r8 + 192]
prefetcht1 [r8 + 256]
prefetcht1 [r8 + 320]
prefetcht1 [r8 + 384]
prefetcht1 [r8 + 448]
prefetcht1 [r8 + 512]
prefetcht1 [r8 + 576]
prefetcht1 [r8 + 640]
prefetcht1 [r8 + 704]
prefetcht1 [r8 + 768]
prefetcht1 [r8 + 832]
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_2_0
vmovdqu64 ZMMWORD PTR [rdx + 0], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 64], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 128], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 192], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 256], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 320], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 384], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 448], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 512], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 2304]
mov ecx, 32
.p2align 5
.Lbank_0_2_1:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 320]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_2_1
vmovdqu64 ZMMWORD PTR [rdx + 576], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 640], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 704], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 768], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 832], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 896], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 960], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1024], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 1088], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 4608]
mov ecx, 32
.p2align 5
.Lbank_0_2_2:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 320]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_2_2
vmovdqu64 ZMMWORD PTR [rdx + 1152], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1216], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1280], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1344], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1408], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 1472], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 1536], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1600], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 1664], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 6912]
mov ecx, 32
.p2align 5
.Lbank_0_2_3:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 576]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_2_3
vmovdqu64 ZMMWORD PTR [rdx + 1728], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1792], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1856], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1920], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1984], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 2048], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 2112], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 2176], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 2240], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
mov r8, rdi
lea r9, [rsi + 9216]
mov ecx, 32
.p2align 5
.Lbank_0_2_4:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 832]
vpxorq zmm11, zmm9, zmm10
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
add r8, 118784
add r9, 48
dec ecx
jnz .Lbank_0_2_4
vmovdqu64 ZMMWORD PTR [rdx + 2304], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2368], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2432], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2496], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2560], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 2624], zmm5
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 10752]
mov ecx, 32
.p2align 5
.Lbank_0_2_5:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 512]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 576]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 832]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_2_5
vmovdqu64 ZMMWORD PTR [rdx + 2688], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2752], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2816], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2880], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2944], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3008], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3072], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3136], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3200], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 13056]
mov ecx, 32
.p2align 5
.Lbank_0_2_6:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 576]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_2_6
vmovdqu64 ZMMWORD PTR [rdx + 3264], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3328], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3392], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 3456], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 3520], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3584], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3648], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3712], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3776], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 15360]
mov ecx, 32
.p2align 5
.Lbank_0_2_7:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 256]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 320]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 832]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_2_7
vmovdqu64 ZMMWORD PTR [rdx + 3840], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3904], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3968], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4032], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4096], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4160], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4224], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4288], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4352], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 17664]
mov ecx, 32
.p2align 5
.Lbank_0_2_8:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 256]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 512]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 320]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 576]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 832]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 128]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 384]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 192]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 448]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 704]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_0_2_8
vmovdqu64 ZMMWORD PTR [rdx + 4416], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 4480], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 4544], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4608], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4672], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4736], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4800], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4864], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4928], zmm8
vzeroupper
ret
.size flock_ab_register_bank0_m2, .-flock_ab_register_bank0_m2
.p2align 5
.globl flock_ab_register_bank2_m0
.type flock_ab_register_bank2_m0, @function
flock_ab_register_bank2_m0:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
mov r8, rdi
lea r9, [rsi + 0]
mov ecx, 32
.p2align 5
.Lbank_2_0_0:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm11, zmm9, zmm10
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
add r8, 118784
add r9, 48
dec ecx
jnz .Lbank_2_0_0
vmovdqu64 ZMMWORD PTR [rdx + 0], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 64], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 128], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 192], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 256], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 320], zmm5
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 1536]
mov ecx, 32
.p2align 5
.Lbank_2_0_1:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_0_1
vmovdqu64 ZMMWORD PTR [rdx + 384], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 448], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 512], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 576], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 640], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 704], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 768], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 832], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 896], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 3840]
mov ecx, 32
.p2align 5
.Lbank_2_0_2:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 0]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 64]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_0_2
vmovdqu64 ZMMWORD PTR [rdx + 960], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1024], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1088], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1152], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1216], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 1280], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 1344], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1408], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 1472], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 6144]
mov ecx, 32
.p2align 5
.Lbank_2_0_3:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 448]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_0_3
vmovdqu64 ZMMWORD PTR [rdx + 1536], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1600], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1664], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1728], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1792], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 1856], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 1920], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1984], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 2048], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 8448]
mov ecx, 32
.p2align 5
.Lbank_2_0_4:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_0_4
vmovdqu64 ZMMWORD PTR [rdx + 2112], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2176], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2240], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2304], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2368], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 2432], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 2496], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 2560], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 2624], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 10752]
mov ecx, 32
.p2align 5
.Lbank_2_0_5:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 384]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 448]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 512]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_0_5
vmovdqu64 ZMMWORD PTR [rdx + 2688], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2752], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2816], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2880], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2944], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3008], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3072], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3136], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3200], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 13056]
mov ecx, 32
.p2align 5
.Lbank_2_0_6:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 448]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 0]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 64]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_0_6
vmovdqu64 ZMMWORD PTR [rdx + 3264], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3328], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3392], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 3456], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 3520], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3584], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3648], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3712], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3776], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 15360]
mov ecx, 32
.p2align 5
.Lbank_2_0_7:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 256]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_0_7
vmovdqu64 ZMMWORD PTR [rdx + 3840], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3904], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3968], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4032], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4096], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4160], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4224], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4288], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4352], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 17664]
mov ecx, 32
.p2align 5
.Lbank_2_0_8:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 384]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 448]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 0]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 256]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 512]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 64]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_0_8
vmovdqu64 ZMMWORD PTR [rdx + 4416], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 4480], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 4544], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4608], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4672], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4736], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4800], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4864], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4928], zmm8
vzeroupper
ret
.size flock_ab_register_bank2_m0, .-flock_ab_register_bank2_m0
.p2align 5
.globl flock_ab_register_bank2_m1
.type flock_ab_register_bank2_m1, @function
flock_ab_register_bank2_m1:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
mov r8, rdi
lea r9, [rsi + 0]
mov ecx, 32
.p2align 5
.Lbank_2_1_0:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm11, zmm9, zmm10
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
prefetcht1 [r8 + 0]
prefetcht1 [r8 + 64]
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
prefetcht1 [r8 + 128]
prefetcht1 [r8 + 192]
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
prefetcht1 [r8 + 256]
prefetcht1 [r8 + 320]
prefetcht1 [r8 + 384]
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
prefetcht1 [r8 + 448]
prefetcht1 [r8 + 512]
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
prefetcht1 [r8 + 576]
prefetcht1 [r8 + 640]
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
prefetcht1 [r8 + 704]
prefetcht1 [r8 + 768]
prefetcht1 [r8 + 832]
add r8, 118784
add r9, 48
dec ecx
jnz .Lbank_2_1_0
vmovdqu64 ZMMWORD PTR [rdx + 0], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 64], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 128], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 192], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 256], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 320], zmm5
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 1536]
mov ecx, 32
.p2align 5
.Lbank_2_1_1:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_1_1
vmovdqu64 ZMMWORD PTR [rdx + 384], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 448], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 512], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 576], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 640], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 704], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 768], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 832], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 896], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 3840]
mov ecx, 32
.p2align 5
.Lbank_2_1_2:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 0]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 64]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_1_2
vmovdqu64 ZMMWORD PTR [rdx + 960], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1024], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1088], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1152], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1216], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 1280], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 1344], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1408], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 1472], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 6144]
mov ecx, 32
.p2align 5
.Lbank_2_1_3:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 448]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_1_3
vmovdqu64 ZMMWORD PTR [rdx + 1536], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1600], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1664], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1728], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1792], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 1856], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 1920], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1984], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 2048], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 8448]
mov ecx, 32
.p2align 5
.Lbank_2_1_4:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_1_4
vmovdqu64 ZMMWORD PTR [rdx + 2112], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2176], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2240], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2304], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2368], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 2432], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 2496], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 2560], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 2624], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 10752]
mov ecx, 32
.p2align 5
.Lbank_2_1_5:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 384]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 448]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 512]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_1_5
vmovdqu64 ZMMWORD PTR [rdx + 2688], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2752], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2816], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2880], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2944], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3008], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3072], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3136], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3200], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 13056]
mov ecx, 32
.p2align 5
.Lbank_2_1_6:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 448]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 0]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 64]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_1_6
vmovdqu64 ZMMWORD PTR [rdx + 3264], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3328], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3392], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 3456], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 3520], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3584], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3648], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3712], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3776], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 15360]
mov ecx, 32
.p2align 5
.Lbank_2_1_7:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 256]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_1_7
vmovdqu64 ZMMWORD PTR [rdx + 3840], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3904], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3968], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4032], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4096], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4160], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4224], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4288], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4352], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 17664]
mov ecx, 32
.p2align 5
.Lbank_2_1_8:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 384]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 448]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 0]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 256]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 512]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 64]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_1_8
vmovdqu64 ZMMWORD PTR [rdx + 4416], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 4480], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 4544], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4608], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4672], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4736], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4800], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4864], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4928], zmm8
vzeroupper
ret
.size flock_ab_register_bank2_m1, .-flock_ab_register_bank2_m1
.p2align 5
.globl flock_ab_register_bank2_m2
.type flock_ab_register_bank2_m2, @function
flock_ab_register_bank2_m2:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
mov r8, rdi
lea r9, [rsi + 0]
mov ecx, 32
.p2align 5
.Lbank_2_2_0:
prefetcht1 [r8 + 0]
prefetcht1 [r8 + 64]
prefetcht1 [r8 + 128]
prefetcht1 [r8 + 192]
prefetcht1 [r8 + 256]
prefetcht1 [r8 + 320]
prefetcht1 [r8 + 384]
prefetcht1 [r8 + 448]
prefetcht1 [r8 + 512]
prefetcht1 [r8 + 576]
prefetcht1 [r8 + 640]
prefetcht1 [r8 + 704]
prefetcht1 [r8 + 768]
prefetcht1 [r8 + 832]
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 0]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 64]
vpxorq zmm11, zmm9, zmm10
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
add r8, 118784
add r9, 48
dec ecx
jnz .Lbank_2_2_0
vmovdqu64 ZMMWORD PTR [rdx + 0], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 64], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 128], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 192], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 256], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 320], zmm5
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 1536]
mov ecx, 32
.p2align 5
.Lbank_2_2_1:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_2_1
vmovdqu64 ZMMWORD PTR [rdx + 384], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 448], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 512], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 576], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 640], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 704], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 768], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 832], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 896], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 3840]
mov ecx, 32
.p2align 5
.Lbank_2_2_2:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 0]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 256]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 64]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_2_2
vmovdqu64 ZMMWORD PTR [rdx + 960], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1024], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1088], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1152], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1216], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 1280], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 1344], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1408], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 1472], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 6144]
mov ecx, 32
.p2align 5
.Lbank_2_2_3:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 448]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_2_3
vmovdqu64 ZMMWORD PTR [rdx + 1536], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 1600], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 1664], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 1728], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 1792], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 1856], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 1920], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 1984], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 2048], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 8448]
mov ecx, 32
.p2align 5
.Lbank_2_2_4:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_2_4
vmovdqu64 ZMMWORD PTR [rdx + 2112], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2176], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2240], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2304], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2368], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 2432], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 2496], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 2560], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 2624], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 10752]
mov ecx, 32
.p2align 5
.Lbank_2_2_5:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 384]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 448]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 512]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_2_5
vmovdqu64 ZMMWORD PTR [rdx + 2688], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 2752], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 2816], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 2880], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 2944], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3008], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3072], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3136], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3200], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 13056]
mov ecx, 32
.p2align 5
.Lbank_2_2_6:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 384]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 448]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 0]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 512]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 64]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_2_6
vmovdqu64 ZMMWORD PTR [rdx + 3264], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3328], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3392], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 3456], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 3520], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 3584], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 3648], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 3712], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 3776], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 15360]
mov ecx, 32
.p2align 5
.Lbank_2_2_7:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 256]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_2_7
vmovdqu64 ZMMWORD PTR [rdx + 3840], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 3904], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 3968], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4032], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4096], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4160], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4224], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4288], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4352], zmm8
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
mov r8, rdi
lea r9, [rsi + 17664]
mov ecx, 32
.p2align 5
.Lbank_2_2_8:
vmovdqu64 zmm9, ZMMWORD PTR [r8 + 128]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 384]
vpxorq zmm9, zmm9, ZMMWORD PTR [r8 + 640]
vmovdqu64 zmm10, ZMMWORD PTR [r8 + 192]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 448]
vpxorq zmm10, zmm10, ZMMWORD PTR [r8 + 704]
vmovdqu64 zmm11, ZMMWORD PTR [r8 + 0]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 256]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 512]
vpxorq zmm11, zmm11, ZMMWORD PTR [r8 + 768]
vmovdqu64 zmm12, ZMMWORD PTR [r8 + 64]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 320]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 576]
vpxorq zmm12, zmm12, ZMMWORD PTR [r8 + 832]
vpxorq zmm13, zmm9, zmm10
vpxorq zmm14, zmm11, zmm12
vpxorq zmm15, zmm9, zmm11
vpxorq zmm16, zmm10, zmm12
vpxorq zmm17, zmm13, zmm14
vgf2p8affineqb zmm18, zmm9, QWORD PTR [r9 + 0]{1to8}, 0
vpxorq zmm0, zmm0, zmm18
vgf2p8affineqb zmm18, zmm10, QWORD PTR [r9 + 8]{1to8}, 0
vpxorq zmm1, zmm1, zmm18
vgf2p8affineqb zmm18, zmm13, QWORD PTR [r9 + 16]{1to8}, 0
vpxorq zmm2, zmm2, zmm18
vgf2p8affineqb zmm18, zmm11, QWORD PTR [r9 + 24]{1to8}, 0
vpxorq zmm3, zmm3, zmm18
vgf2p8affineqb zmm18, zmm12, QWORD PTR [r9 + 32]{1to8}, 0
vpxorq zmm4, zmm4, zmm18
vgf2p8affineqb zmm18, zmm14, QWORD PTR [r9 + 40]{1to8}, 0
vpxorq zmm5, zmm5, zmm18
vgf2p8affineqb zmm18, zmm15, QWORD PTR [r9 + 48]{1to8}, 0
vpxorq zmm6, zmm6, zmm18
vgf2p8affineqb zmm18, zmm16, QWORD PTR [r9 + 56]{1to8}, 0
vpxorq zmm7, zmm7, zmm18
vgf2p8affineqb zmm18, zmm17, QWORD PTR [r9 + 64]{1to8}, 0
vpxorq zmm8, zmm8, zmm18
add r8, 118784
add r9, 72
dec ecx
jnz .Lbank_2_2_8
vmovdqu64 ZMMWORD PTR [rdx + 4416], zmm0
vmovdqu64 ZMMWORD PTR [rdx + 4480], zmm1
vmovdqu64 ZMMWORD PTR [rdx + 4544], zmm2
vmovdqu64 ZMMWORD PTR [rdx + 4608], zmm3
vmovdqu64 ZMMWORD PTR [rdx + 4672], zmm4
vmovdqu64 ZMMWORD PTR [rdx + 4736], zmm5
vmovdqu64 ZMMWORD PTR [rdx + 4800], zmm6
vmovdqu64 ZMMWORD PTR [rdx + 4864], zmm7
vmovdqu64 ZMMWORD PTR [rdx + 4928], zmm8
vzeroupper
ret
.size flock_ab_register_bank2_m2, .-flock_ab_register_bank2_m2
.p2align 5
.globl flock_ab_leaf_interpolate0
.type flock_ab_leaf_interpolate0, @function
flock_ab_leaf_interpolate0:
push rbp
mov rbp, rsp
and rsp, -64
sub rsp, 2432
vmovdqu64 zmm30, ZMMWORD PTR [rdi + 192]
vmovdqu64 zmm28, ZMMWORD PTR [rdi + 64]
vmovdqu64 zmm25, ZMMWORD PTR [rdi + 384]
vpxorq zmm24, zmm30, zmm25
vmovdqu64 zmm23, ZMMWORD PTR [rdi + 0]
vmovdqa64 zmm22, zmm23
vmovdqu64 ZMMWORD PTR [rsi + 0], zmm22
vpxorq zmm23, zmm23, zmm28
vpxorq zmm24, zmm24, zmm23
vmovdqu64 ZMMWORD PTR [rsi + 128], zmm24
vmovdqu64 zmm22, ZMMWORD PTR [rdi + 1792]
vmovdqu64 zmm19, ZMMWORD PTR [rdi + 1856]
vmovdqu64 zmm18, ZMMWORD PTR [rdi + 1728]
vpxorq zmm17, zmm18, zmm22
vmovdqu64 zmm14, ZMMWORD PTR [rdi + 2752]
vmovdqu64 zmm13, ZMMWORD PTR [rdi + 2560]
vmovdqu64 zmm12, ZMMWORD PTR [rdi + 2368]
vmovdqa64 zmm11, zmm12
vpxorq zmm11, zmm11, zmm13
vmovdqu64 ZMMWORD PTR [rsi + 1792], zmm11
vmovdqu64 zmm7, ZMMWORD PTR [rdi + 2816]
vmovdqu64 zmm6, ZMMWORD PTR [rdi + 2688]
vpxorq zmm5, zmm6, zmm14
vmovdqu64 zmm4, ZMMWORD PTR [rdi + 2624]
vmovdqu64 zmm3, ZMMWORD PTR [rdi + 2496]
vmovdqu64 zmm2, ZMMWORD PTR [rdi + 2432]
vmovdqu64 zmm1, ZMMWORD PTR [rdi + 2304]
vpxorq zmm12, zmm1, zmm12
vmovdqu64 ZMMWORD PTR [rsp + 0], zmm18
vmovdqu64 zmm24, ZMMWORD PTR [rdi + 3328]
vmovdqu64 zmm9, ZMMWORD PTR [rdi + 3136]
vmovdqu64 ZMMWORD PTR [rsp + 128], zmm6
vmovdqu64 zmm6, ZMMWORD PTR [rdi + 2944]
vmovdqu64 ZMMWORD PTR [rsp + 192], zmm1
vmovdqu64 zmm1, ZMMWORD PTR [rdi + 2176]
vpxorq zmm8, zmm1, zmm13
vpxorq zmm9, zmm9, zmm8
vpternlogq zmm13, zmm3, zmm4, 0x96
vmovdqu64 ZMMWORD PTR [rsp + 320], zmm8
vmovdqu64 zmm8, ZMMWORD PTR [rdi + 1984]
vmovdqu64 ZMMWORD PTR [rsp + 448], zmm1
vmovdqu64 ZMMWORD PTR [rsp + 576], zmm28
vmovdqu64 zmm28, ZMMWORD PTR [rdi + 3200]
vmovdqu64 zmm21, ZMMWORD PTR [rdi + 3264]
vmovdqu64 ZMMWORD PTR [rsp + 704], zmm21
vmovdqu64 zmm21, ZMMWORD PTR [rdi + 3072]
vmovdqu64 zmm11, ZMMWORD PTR [rdi + 3008]
vmovdqu64 zmm0, ZMMWORD PTR [rdi + 2880]
vmovdqu64 ZMMWORD PTR [rsp + 896], zmm30
vpxorq zmm30, zmm0, zmm6
vpxorq zmm11, zmm11, zmm30
vpternlogq zmm14, zmm30, zmm9, 0x96
vpxorq zmm0, zmm0, zmm5
vmovdqu64 zmm30, ZMMWORD PTR [rdi + 2240]
vpxorq zmm4, zmm30, zmm4
vpxorq zmm9, zmm9, zmm4
vpternlogq zmm28, zmm11, zmm9, 0x96
vmovdqu64 zmm9, ZMMWORD PTR [rdi + 2112]
vpxorq zmm10, zmm9, zmm3
vpxorq zmm21, zmm21, zmm10
vpternlogq zmm3, zmm8, zmm6, 0x96
vpxorq zmm3, zmm3, zmm12
vmovdqu64 ZMMWORD PTR [rsi + 1664], zmm3
vmovdqu64 zmm3, ZMMWORD PTR [rdi + 2048]
vmovdqu64 ZMMWORD PTR [rsp + 1024], zmm4
vpxorq zmm4, zmm19, zmm3
vpternlogq zmm19, zmm24, zmm17, 0x96
vmovdqu64 ZMMWORD PTR [rsp + 1088], zmm19
vpxorq zmm19, zmm2, zmm4
vpternlogq zmm7, zmm19, zmm5, 0x96
vpxorq zmm2, zmm2, zmm12
vpxorq zmm13, zmm13, zmm2
vmovdqu64 ZMMWORD PTR [rsi + 1728], zmm13
vmovdqu64 zmm13, ZMMWORD PTR [rdi + 1920]
vpxorq zmm8, zmm13, zmm8
vpternlogq zmm3, zmm8, zmm11, 0x96
vpxorq zmm3, zmm3, zmm2
vmovdqu64 ZMMWORD PTR [rsi + 1600], zmm3
vpternlogq zmm22, zmm8, zmm12, 0x96
vpxorq zmm22, zmm22, zmm14
vmovdqu64 ZMMWORD PTR [rsi + 1536], zmm22
vpxorq zmm8, zmm17, zmm8
vpxorq zmm12, zmm12, zmm8
vpxorq zmm22, zmm12, zmm21
vpxorq zmm3, zmm7, zmm22
vpxorq zmm3, zmm3, zmm28
vmovdqu64 ZMMWORD PTR [rsi + 1472], zmm3
vmovdqu64 zmm2, ZMMWORD PTR [rdi + 4096]
vmovdqu64 zmm5, ZMMWORD PTR [rdi + 832]
vmovdqu64 ZMMWORD PTR [rsp + 1152], zmm28
vmovdqu64 ZMMWORD PTR [rsp + 1216], zmm17
vmovdqu64 ZMMWORD PTR [rsp + 1280], zmm4
vmovdqu64 zmm4, ZMMWORD PTR [rdi + 4160]
vmovdqu64 zmm1, ZMMWORD PTR [rdi + 4032]
vmovdqu64 ZMMWORD PTR [rsp + 1408], zmm19
vpxorq zmm19, zmm1, zmm2
vpxorq zmm4, zmm4, zmm19
vpternlogq zmm2, zmm5, zmm0, 0x96
vpxorq zmm2, zmm2, zmm22
vmovdqu64 ZMMWORD PTR [rsi + 1408], zmm2
vmovdqu64 zmm2, ZMMWORD PTR [rdi + 896]
vmovdqu64 zmm22, ZMMWORD PTR [rdi + 768]
vpxorq zmm5, zmm22, zmm5
vmovdqu64 ZMMWORD PTR [rsp + 1472], zmm0
vpxorq zmm0, zmm2, zmm5
vpternlogq zmm0, zmm12, zmm4, 0x96
vpxorq zmm0, zmm0, zmm7
vmovdqu64 ZMMWORD PTR [rsi + 1344], zmm0
vmovdqu64 zmm20, ZMMWORD PTR [rdi + 3392]
vmovdqu64 ZMMWORD PTR [rsp + 1536], zmm7
vmovdqu64 zmm7, ZMMWORD PTR [rdi + 128]
vmovdqu64 ZMMWORD PTR [rsp + 1600], zmm20
vmovdqa64 zmm20, zmm7
vpxorq zmm20, zmm20, zmm23
vmovdqu64 ZMMWORD PTR [rsi + 64], zmm20
vmovdqu64 ZMMWORD PTR [rsp + 1664], zmm2
vmovdqu64 ZMMWORD PTR [rsp + 1792], zmm22
vmovdqu64 zmm15, ZMMWORD PTR [rdi + 512]
vmovdqu64 zmm22, ZMMWORD PTR [rdi + 448]
vmovdqu64 ZMMWORD PTR [rsp + 1984], zmm14
vmovdqu64 zmm14, ZMMWORD PTR [rdi + 320]
vpxorq zmm7, zmm7, zmm14
vmovdqa64 zmm17, zmm25
vpternlogq zmm17, zmm15, zmm7, 0x96
vmovdqu64 ZMMWORD PTR [rsp + 2112], zmm15
vmovdqu64 zmm15, ZMMWORD PTR [rdi + 256]
vmovdqu64 ZMMWORD PTR [rsp + 2176], zmm7
vmovdqu64 zmm7, ZMMWORD PTR [rsp + 896]
vpxorq zmm7, zmm7, zmm15
vpxorq zmm23, zmm23, zmm7
vmovdqu64 ZMMWORD PTR [rsp + 896], zmm21
vpxorq zmm21, zmm22, zmm23
vpxorq zmm17, zmm17, zmm21
vmovdqu64 ZMMWORD PTR [rsi + 192], zmm17
vmovdqu64 ZMMWORD PTR [rsp + 2240], zmm21
vmovdqu64 zmm21, ZMMWORD PTR [rdi + 3520]
vmovdqu64 zmm17, ZMMWORD PTR [rdi + 4672]
vmovdqu64 zmm20, ZMMWORD PTR [rdi + 1408]
vpxorq zmm15, zmm15, zmm20
vpternlogq zmm15, zmm6, zmm17, 0x96
vpxorq zmm30, zmm30, zmm21
vpxorq zmm13, zmm20, zmm13
vmovdqu64 zmm6, ZMMWORD PTR [rdi + 3456]
vpxorq zmm24, zmm24, zmm6
vmovdqu64 ZMMWORD PTR [rsp + 832], zmm13
vmovdqu64 zmm13, ZMMWORD PTR [rsp + 576]
vpternlogq zmm13, zmm24, zmm19, 0x96
vmovdqu64 zmm16, ZMMWORD PTR [rdi + 4608]
vpxorq zmm17, zmm16, zmm17
vpxorq zmm1, zmm1, zmm16
vmovdqu64 zmm31, ZMMWORD PTR [rdi + 1344]
vpxorq zmm20, zmm31, zmm20
vmovdqu64 zmm16, ZMMWORD PTR [rdi + 3712]
vmovdqu64 ZMMWORD PTR [rsp + 576], zmm1
vmovdqu64 ZMMWORD PTR [rsp + 2304], zmm30
vmovdqu64 zmm30, ZMMWORD PTR [rdi + 4480]
vmovdqu64 zmm2, ZMMWORD PTR [rdi + 1600]
vmovdqu64 zmm29, ZMMWORD PTR [rdi + 1216]
vmovdqu64 ZMMWORD PTR [rsp + 512], zmm31
vmovdqu64 zmm31, ZMMWORD PTR [rdi + 1024]
vpxorq zmm22, zmm22, zmm31
vpxorq zmm8, zmm8, zmm22
vmovdqu64 ZMMWORD PTR [rsp + 2368], zmm13
vmovdqu64 zmm13, ZMMWORD PTR [rsp + 448]
vpternlogq zmm13, zmm16, zmm8, 0x96
vpxorq zmm2, zmm2, zmm22
vmovdqu64 zmm18, ZMMWORD PTR [rdi + 4864]
vmovdqu64 zmm3, ZMMWORD PTR [rdi + 4288]
vpternlogq zmm16, zmm3, zmm17, 0x96
vpternlogq zmm18, zmm2, zmm16, 0x96
vmovdqu64 zmm22, ZMMWORD PTR [rdi + 3904]
vpxorq zmm22, zmm22, zmm12
vmovdqu64 zmm12, ZMMWORD PTR [rsp + 320]
vpxorq zmm12, zmm22, zmm12
vpternlogq zmm31, zmm3, zmm12, 0x96
vmovdqu64 zmm12, ZMMWORD PTR [rdi + 640]
vmovdqu64 zmm16, ZMMWORD PTR [rdi + 4416]
vmovdqu64 zmm27, ZMMWORD PTR [rdi + 1152]
vmovdqu64 zmm8, ZMMWORD PTR [rsp + 192]
vpxorq zmm8, zmm27, zmm8
vmovdqu64 ZMMWORD PTR [rsp + 192], zmm17
vmovdqu64 zmm17, ZMMWORD PTR [rsp + 128]
vpternlogq zmm8, zmm17, zmm16, 0x96
vmovdqu64 zmm26, ZMMWORD PTR [rdi + 1536]
vmovdqu64 zmm0, ZMMWORD PTR [rdi + 1280]
vmovdqu64 zmm1, ZMMWORD PTR [rdi + 4544]
vpxorq zmm17, zmm12, zmm17
vpternlogq zmm17, zmm5, zmm19, 0x96
vpxorq zmm17, zmm17, zmm31
vmovdqu64 ZMMWORD PTR [rsi + 1280], zmm17
vmovdqu64 ZMMWORD PTR [rsp + 128], zmm31
vmovdqu64 zmm3, ZMMWORD PTR [rdi + 576]
vpxorq zmm12, zmm3, zmm12
vpxorq zmm5, zmm12, zmm5
vmovdqu64 zmm31, ZMMWORD PTR [rdi + 3840]
vpxorq zmm21, zmm21, zmm31
vpxorq zmm7, zmm7, zmm5
vpxorq zmm22, zmm22, zmm21
vpternlogq zmm15, zmm19, zmm22, 0x96
vpternlogq zmm14, zmm6, zmm22, 0x96
vpxorq zmm30, zmm30, zmm22
vpternlogq zmm14, zmm11, zmm4, 0x96
vpxorq zmm10, zmm5, zmm10
vpxorq zmm5, zmm23, zmm5
vpxorq zmm12, zmm12, zmm23
vmovdqu64 zmm23, ZMMWORD PTR [rsp + 704]
vpxorq zmm23, zmm23, zmm5
vpxorq zmm24, zmm23, zmm24
vpternlogq zmm8, zmm21, zmm24, 0x96
vpxorq zmm8, zmm8, zmm13
vmovdqu64 ZMMWORD PTR [rsi + 768], zmm8
vmovdqu64 zmm28, ZMMWORD PTR [rdi + 960]
vpxorq zmm25, zmm25, zmm28
vpxorq zmm26, zmm26, zmm25
vpxorq zmm9, zmm9, zmm25
vmovdqu64 zmm25, ZMMWORD PTR [rsp + 0]
vpternlogq zmm25, zmm23, zmm2, 0x96
vmovdqu64 zmm8, ZMMWORD PTR [rsp + 896]
vpxorq zmm8, zmm26, zmm8
vpxorq zmm16, zmm16, zmm30
vpxorq zmm3, zmm3, zmm27
vmovdqu64 zmm21, ZMMWORD PTR [rsp + 2240]
vpxorq zmm3, zmm3, zmm21
vmovdqu64 ZMMWORD PTR [rsi + 256], zmm3
vpxorq zmm27, zmm27, zmm29
vpxorq zmm29, zmm29, zmm20
vpxorq zmm25, zmm25, zmm29
vmovdqu64 ZMMWORD PTR [rsi + 512], zmm25
vmovdqu64 zmm25, ZMMWORD PTR [rsp + 2368]
vpternlogq zmm25, zmm30, zmm29, 0x96
vmovdqu64 zmm29, ZMMWORD PTR [rsp + 1984]
vpternlogq zmm25, zmm7, zmm29, 0x96
vpxorq zmm25, zmm25, zmm18
vmovdqu64 ZMMWORD PTR [rsi + 1024], zmm25
vpxorq zmm0, zmm0, zmm27
vmovdqu64 zmm25, ZMMWORD PTR [rsp + 512]
vpxorq zmm25, zmm25, zmm27
vmovdqu64 zmm27, ZMMWORD PTR [rsp + 1792]
vpternlogq zmm27, zmm26, zmm25, 0x96
vpxorq zmm27, zmm27, zmm12
vmovdqu64 ZMMWORD PTR [rsi + 384], zmm27
vmovdqu64 zmm30, ZMMWORD PTR [rdi + 3968]
vmovdqu64 zmm27, ZMMWORD PTR [rdi + 704]
vmovdqu64 zmm11, ZMMWORD PTR [rsp + 1664]
vpxorq zmm11, zmm27, zmm11
vmovdqu64 zmm22, ZMMWORD PTR [rsp + 1408]
vpxorq zmm22, zmm11, zmm22
vmovdqu64 zmm6, ZMMWORD PTR [rsp + 2176]
vpternlogq zmm27, zmm6, zmm0, 0x96
vpxorq zmm27, zmm27, zmm12
vmovdqu64 ZMMWORD PTR [rsi + 320], zmm27
vpxorq zmm6, zmm6, zmm11
vpxorq zmm5, zmm5, zmm6
vmovdqu64 zmm11, ZMMWORD PTR [rsp + 1600]
vpxorq zmm11, zmm11, zmm6
vmovdqu64 zmm21, ZMMWORD PTR [rdi + 1088]
vpternlogq zmm31, zmm21, zmm30, 0x96
vmovdqu64 zmm27, ZMMWORD PTR [rsp + 2112]
vpxorq zmm27, zmm27, zmm21
vmovdqu64 zmm6, ZMMWORD PTR [rdi + 1664]
vpxorq zmm6, zmm6, zmm0
vmovdqu64 zmm3, ZMMWORD PTR [rdi + 1472]
vpxorq zmm3, zmm3, zmm20
vpternlogq zmm5, zmm3, zmm26, 0x96
vpternlogq zmm5, zmm2, zmm27, 0x96
vpxorq zmm5, zmm5, zmm6
vmovdqu64 ZMMWORD PTR [rsi + 448], zmm5
vmovdqu64 zmm5, ZMMWORD PTR [rsp + 1088]
vpternlogq zmm5, zmm23, zmm3, 0x96
vpxorq zmm5, zmm5, zmm11
vmovdqu64 ZMMWORD PTR [rsi + 576], zmm5
vmovdqu64 zmm12, ZMMWORD PTR [rdi + 4736]
vpxorq zmm12, zmm12, zmm3
vmovdqu64 zmm21, ZMMWORD PTR [rdi + 3584]
vmovdqu64 zmm3, ZMMWORD PTR [rsp + 2304]
vmovdqu64 zmm5, ZMMWORD PTR [rsp + 1280]
vpternlogq zmm3, zmm21, zmm5, 0x96
vpxorq zmm21, zmm21, zmm30
vmovdqu64 zmm30, ZMMWORD PTR [rsp + 192]
vpternlogq zmm14, zmm21, zmm30, 0x96
vpternlogq zmm14, zmm12, zmm7, 0x96
vpxorq zmm14, zmm14, zmm22
vmovdqu64 ZMMWORD PTR [rsi + 1088], zmm14
vpternlogq zmm3, zmm9, zmm13, 0x96
vpxorq zmm21, zmm21, zmm16
vmovdqu64 zmm13, ZMMWORD PTR [rsp + 1536]
vpternlogq zmm1, zmm13, zmm21, 0x96
vpternlogq zmm0, zmm24, zmm11, 0x96
vpxorq zmm0, zmm0, zmm1
vmovdqu64 ZMMWORD PTR [rsi + 832], zmm0
vmovdqu64 zmm0, ZMMWORD PTR [rdi + 4352]
vpxorq zmm0, zmm0, zmm4
vmovdqu64 zmm4, ZMMWORD PTR [rsp + 1024]
vpternlogq zmm31, zmm4, zmm0, 0x96
vmovdqu64 zmm29, ZMMWORD PTR [rdi + 4928]
vpxorq zmm29, zmm29, zmm6
vpternlogq zmm29, zmm0, zmm12, 0x96
vmovdqu64 zmm4, ZMMWORD PTR [rdi + 3776]
vmovdqu64 zmm12, ZMMWORD PTR [rdi + 4224]
vpternlogq zmm28, zmm12, zmm10, 0x96
vpxorq zmm15, zmm15, zmm28
vmovdqu64 ZMMWORD PTR [rsi + 1152], zmm15
vpternlogq zmm31, zmm22, zmm28, 0x96
vmovdqu64 zmm28, ZMMWORD PTR [rsp + 128]
vpxorq zmm31, zmm31, zmm28
vmovdqu64 ZMMWORD PTR [rsi + 1216], zmm31
vmovdqu64 zmm17, ZMMWORD PTR [rdi + 4800]
vpternlogq zmm12, zmm17, zmm8, 0x96
vmovdqu64 zmm31, ZMMWORD PTR [rdi + 3648]
vpxorq zmm31, zmm31, zmm24
vmovdqu64 zmm24, ZMMWORD PTR [rsp + 576]
vpternlogq zmm24, zmm31, zmm16, 0x96
vmovdqu64 zmm16, ZMMWORD PTR [rsp + 1472]
vpternlogq zmm24, zmm25, zmm16, 0x96
vpxorq zmm24, zmm24, zmm12
vmovdqu64 ZMMWORD PTR [rsi + 896], zmm24
vmovdqu64 zmm24, ZMMWORD PTR [rsp + 832]
vmovdqu64 zmm16, ZMMWORD PTR [rsp + 1216]
vpternlogq zmm24, zmm16, zmm31, 0x96
vpxorq zmm24, zmm24, zmm9
vmovdqu64 ZMMWORD PTR [rsi + 640], zmm24
vpxorq zmm31, zmm31, zmm27
vpternlogq zmm4, zmm11, zmm31, 0x96
vpxorq zmm3, zmm3, zmm4
vmovdqu64 ZMMWORD PTR [rsi + 704], zmm3
vmovdqu64 zmm3, ZMMWORD PTR [rsp + 1152]
vpternlogq zmm29, zmm3, zmm4, 0x96
vpternlogq zmm29, zmm12, zmm1, 0x96
vpxorq zmm29, zmm29, zmm18
vmovdqu64 ZMMWORD PTR [rsi + 960], zmm29
mov rsp, rbp
pop rbp
vzeroupper
ret
.size flock_ab_leaf_interpolate0, .-flock_ab_leaf_interpolate0
.p2align 5
.globl flock_ab_leaf_interpolate2
.type flock_ab_leaf_interpolate2, @function
flock_ab_leaf_interpolate2:
push rbp
mov rbp, rsp
and rsp, -64
sub rsp, 2432
vmovdqu64 zmm30, ZMMWORD PTR [rdi + 192]
vmovdqu64 zmm29, ZMMWORD PTR [rdi + 0]
vmovdqa64 zmm28, zmm29
vpxorq zmm28, zmm28, zmm30
vmovdqu64 ZMMWORD PTR [rsi + 128], zmm28
vmovdqu64 zmm26, ZMMWORD PTR [rdi + 320]
vmovdqu64 zmm25, ZMMWORD PTR [rdi + 256]
vmovdqu64 zmm24, ZMMWORD PTR [rdi + 128]
vmovdqa64 zmm23, zmm24
vpternlogq zmm23, zmm30, zmm26, 0x96
vmovdqu64 zmm22, ZMMWORD PTR [rdi + 64]
vpxorq zmm29, zmm29, zmm22
vpxorq zmm21, zmm25, zmm29
vpxorq zmm23, zmm23, zmm21
vmovdqu64 ZMMWORD PTR [rsi + 192], zmm23
vmovdqu64 zmm20, ZMMWORD PTR [rdi + 384]
vmovdqu64 zmm19, ZMMWORD PTR [rdi + 960]
vpxorq zmm18, zmm20, zmm19
vpxorq zmm18, zmm18, zmm21
vmovdqu64 ZMMWORD PTR [rsi + 256], zmm18
vmovdqu64 zmm17, ZMMWORD PTR [rdi + 512]
vpxorq zmm24, zmm24, zmm17
vmovdqu64 zmm16, ZMMWORD PTR [rdi + 448]
vpxorq zmm20, zmm20, zmm16
vpxorq zmm29, zmm29, zmm20
vpxorq zmm15, zmm29, zmm24
vmovdqu64 zmm14, ZMMWORD PTR [rdi + 1088]
vmovdqu64 zmm13, ZMMWORD PTR [rdi + 1024]
vpxorq zmm12, zmm19, zmm13
vpxorq zmm14, zmm14, zmm12
vpxorq zmm15, zmm15, zmm14
vmovdqu64 ZMMWORD PTR [rsi + 320], zmm15
vmovdqu64 zmm10, ZMMWORD PTR [rdi + 1344]
vmovdqu64 zmm9, ZMMWORD PTR [rdi + 768]
vpxorq zmm30, zmm30, zmm9
vpxorq zmm10, zmm10, zmm30
vmovdqu64 zmm8, ZMMWORD PTR [rdi + 576]
vmovdqa64 zmm7, zmm8
vpternlogq zmm7, zmm29, zmm10, 0x96
vmovdqu64 zmm6, ZMMWORD PTR [rdi + 1152]
vpxorq zmm12, zmm6, zmm12
vpxorq zmm7, zmm7, zmm12
vmovdqu64 ZMMWORD PTR [rsi + 384], zmm7
vmovdqu64 zmm4, ZMMWORD PTR [rdi + 1408]
vmovdqu64 zmm3, ZMMWORD PTR [rdi + 832]
vpxorq zmm25, zmm25, zmm3
vpxorq zmm4, zmm4, zmm25
vmovdqu64 zmm1, ZMMWORD PTR [rdi + 896]
vpxorq zmm26, zmm26, zmm1
vpxorq zmm17, zmm17, zmm1
vmovdqu64 zmm0, ZMMWORD PTR [rdi + 1472]
vpternlogq zmm0, zmm14, zmm26, 0x96
vmovdqu64 ZMMWORD PTR [rsp + 0], zmm17
vmovdqu64 zmm17, ZMMWORD PTR [rdi + 1280]
vmovdqu64 ZMMWORD PTR [rsp + 64], zmm3
vmovdqu64 zmm3, ZMMWORD PTR [rdi + 1216]
vpxorq zmm6, zmm6, zmm3
vpxorq zmm17, zmm17, zmm6
vpxorq zmm13, zmm13, zmm6
vmovdqu64 zmm6, ZMMWORD PTR [rdi + 704]
vmovdqu64 ZMMWORD PTR [rsp + 128], zmm22
vmovdqa64 zmm22, zmm6
vpternlogq zmm22, zmm24, zmm17, 0x96
vmovdqu64 zmm5, ZMMWORD PTR [rdi + 640]
vpxorq zmm8, zmm8, zmm5
vmovdqu64 ZMMWORD PTR [rsp + 256], zmm5
vmovdqa64 zmm5, zmm8
vpternlogq zmm5, zmm29, zmm4, 0x96
vpternlogq zmm22, zmm10, zmm5, 0x96
vpxorq zmm22, zmm22, zmm0
vmovdqu64 ZMMWORD PTR [rsi + 448], zmm22
vmovdqu64 ZMMWORD PTR [rsp + 320], zmm0
vmovdqu64 zmm0, ZMMWORD PTR [rdi + 3264]
vmovdqu64 ZMMWORD PTR [rsp + 384], zmm4
vmovdqu64 zmm4, ZMMWORD PTR [rdi + 1536]
vmovdqa64 zmm1, zmm4
vpternlogq zmm1, zmm0, zmm13, 0x96
vpxorq zmm1, zmm1, zmm5
vmovdqu64 ZMMWORD PTR [rsi + 512], zmm1
vmovdqu64 zmm7, ZMMWORD PTR [rdi + 3392]
vmovdqu64 zmm2, ZMMWORD PTR [rdi + 3328]
vmovdqu64 ZMMWORD PTR [rsp + 640], zmm12
vmovdqu64 zmm12, ZMMWORD PTR [rdi + 1664]
vpxorq zmm6, zmm6, zmm12
vpxorq zmm24, zmm24, zmm6
vpxorq zmm7, zmm7, zmm24
vmovdqu64 ZMMWORD PTR [rsp + 704], zmm12
vmovdqu64 zmm12, ZMMWORD PTR [rdi + 1600]
vpxorq zmm4, zmm4, zmm12
vpxorq zmm8, zmm8, zmm4
vpxorq zmm29, zmm29, zmm8
vpxorq zmm2, zmm2, zmm29
vpxorq zmm0, zmm0, zmm2
vmovdqu64 ZMMWORD PTR [rsp + 768], zmm12
vpxorq zmm12, zmm0, zmm17
vpxorq zmm12, zmm12, zmm7
vmovdqu64 ZMMWORD PTR [rsi + 576], zmm12
vmovdqu64 ZMMWORD PTR [rsp + 832], zmm24
vmovdqu64 zmm24, ZMMWORD PTR [rdi + 3456]
vmovdqu64 ZMMWORD PTR [rsp + 896], zmm6
vmovdqu64 zmm31, ZMMWORD PTR [rdi + 3648]
vmovdqu64 ZMMWORD PTR [rsp + 960], zmm29
vmovdqu64 zmm29, ZMMWORD PTR [rdi + 1920]
vpternlogq zmm30, zmm29, zmm31, 0x96
vmovdqu64 ZMMWORD PTR [rsp + 1024], zmm17
vmovdqu64 zmm17, ZMMWORD PTR [rdi + 1728]
vmovdqu64 ZMMWORD PTR [rsp + 1088], zmm4
vpxorq zmm4, zmm3, zmm17
vpternlogq zmm4, zmm24, zmm0, 0x96
vpxorq zmm4, zmm4, zmm30
vmovdqu64 ZMMWORD PTR [rsi + 640], zmm4
vmovdqu64 ZMMWORD PTR [rsp + 1152], zmm3
vmovdqu64 ZMMWORD PTR [rsp + 1216], zmm20
vmovdqu64 zmm20, ZMMWORD PTR [rdi + 3584]
vmovdqu64 ZMMWORD PTR [rsp + 1280], zmm8
vmovdqu64 zmm8, ZMMWORD PTR [rdi + 3520]
vpxorq zmm24, zmm24, zmm8
vpxorq zmm0, zmm0, zmm24
vmovdqu64 ZMMWORD PTR [rsp + 1344], zmm24
vmovdqu64 zmm27, ZMMWORD PTR [rdi + 3776]
vpxorq zmm27, zmm27, zmm7
vmovdqu64 ZMMWORD PTR [rsp + 1408], zmm8
vmovdqu64 zmm28, ZMMWORD PTR [rdi + 3712]
vmovdqu64 ZMMWORD PTR [rsp + 1472], zmm9
vmovdqu64 zmm9, ZMMWORD PTR [rdi + 2048]
vmovdqu64 zmm24, ZMMWORD PTR [rdi + 1984]
vmovdqu64 zmm8, ZMMWORD PTR [rdi + 1856]
vmovdqa64 zmm3, zmm8
vpternlogq zmm3, zmm9, zmm20, 0x96
vpternlogq zmm3, zmm0, zmm26, 0x96
vpternlogq zmm3, zmm27, zmm30, 0x96
vmovdqu64 zmm30, ZMMWORD PTR [rdi + 1792]
vpxorq zmm17, zmm17, zmm30
vpxorq zmm25, zmm17, zmm25
vpternlogq zmm25, zmm24, zmm28, 0x96
vpxorq zmm3, zmm3, zmm25
vmovdqu64 ZMMWORD PTR [rsi + 704], zmm3
vmovdqu64 zmm26, ZMMWORD PTR [rdi + 4416]
vpxorq zmm26, zmm26, zmm0
vmovdqu64 zmm22, ZMMWORD PTR [rdi + 2688]
vpxorq zmm16, zmm16, zmm22
vmovdqu64 zmm0, ZMMWORD PTR [rdi + 3840]
vmovdqu64 ZMMWORD PTR [rsp + 1728], zmm16
vmovdqu64 zmm16, ZMMWORD PTR [rdi + 2112]
vpternlogq zmm19, zmm16, zmm22, 0x96
vpternlogq zmm19, zmm0, zmm26, 0x96
vpxorq zmm19, zmm19, zmm25
vmovdqu64 ZMMWORD PTR [rsi + 768], zmm19
vmovdqu64 ZMMWORD PTR [rsp + 1792], zmm27
vmovdqu64 zmm27, ZMMWORD PTR [rdi + 4544]
vmovdqu64 zmm5, ZMMWORD PTR [rdi + 2816]
vmovdqu64 ZMMWORD PTR [rsp + 1856], zmm30
vmovdqu64 zmm30, ZMMWORD PTR [rdi + 4480]
vmovdqu64 zmm1, ZMMWORD PTR [rdi + 2752]
vpxorq zmm22, zmm22, zmm1
vpxorq zmm5, zmm5, zmm22
vmovdqu64 ZMMWORD PTR [rsp + 1920], zmm28
vmovdqu64 zmm28, ZMMWORD PTR [rdi + 3968]
vpxorq zmm20, zmm20, zmm28
vpternlogq zmm27, zmm26, zmm20, 0x96
vmovdqu64 ZMMWORD PTR [rsp + 1984], zmm28
vmovdqu64 zmm28, ZMMWORD PTR [rdi + 3904]
vpxorq zmm0, zmm0, zmm28
vmovdqu64 ZMMWORD PTR [rsp + 2048], zmm28
vmovdqu64 zmm28, ZMMWORD PTR [rdi + 2240]
vpxorq zmm8, zmm8, zmm28
vmovdqu64 ZMMWORD PTR [rsp + 2112], zmm28
vpxorq zmm28, zmm30, zmm8
vpternlogq zmm28, zmm14, zmm5, 0x96
vmovdqu64 zmm14, ZMMWORD PTR [rdi + 2176]
vpxorq zmm16, zmm16, zmm14
vpxorq zmm17, zmm17, zmm16
vmovdqu64 ZMMWORD PTR [rsp + 2176], zmm14
vpxorq zmm14, zmm17, zmm0
vpternlogq zmm28, zmm7, zmm14, 0x96
vpxorq zmm28, zmm28, zmm27
vmovdqu64 ZMMWORD PTR [rsi + 832], zmm28
vmovdqu64 zmm7, ZMMWORD PTR [rdi + 4800]
vpxorq zmm7, zmm7, zmm10
vmovdqu64 zmm11, ZMMWORD PTR [rdi + 4608]
vpxorq zmm30, zmm30, zmm11
vpternlogq zmm2, zmm30, zmm13, 0x96
vmovdqu64 zmm23, ZMMWORD PTR [rdi + 4224]
vpternlogq zmm31, zmm23, zmm7, 0x96
vmovdqu64 zmm15, ZMMWORD PTR [rdi + 4032]
vmovdqu64 zmm6, ZMMWORD PTR [rdi + 3072]
vmovdqu64 zmm12, ZMMWORD PTR [rdi + 2880]
vpxorq zmm22, zmm12, zmm22
vmovdqu64 zmm3, ZMMWORD PTR [rdi + 2496]
vpxorq zmm29, zmm29, zmm3
vmovdqu64 zmm7, ZMMWORD PTR [rsp + 1472]
vpternlogq zmm7, zmm23, zmm29, 0x96
vpxorq zmm6, zmm6, zmm29
vpternlogq zmm30, zmm6, zmm31, 0x96
vmovdqu64 zmm31, ZMMWORD PTR [rsp + 1216]
vmovdqu64 zmm29, ZMMWORD PTR [rsp + 1280]
vpternlogq zmm31, zmm29, zmm7, 0x96
vmovdqu64 zmm7, ZMMWORD PTR [rsp + 1088]
vpxorq zmm7, zmm7, zmm6
vpxorq zmm3, zmm3, zmm16
vmovdqu64 zmm16, ZMMWORD PTR [rdi + 2304]
vpxorq zmm6, zmm16, zmm15
vmovdqu64 zmm23, ZMMWORD PTR [rsp + 640]
vpternlogq zmm6, zmm26, zmm23, 0x96
vpternlogq zmm6, zmm22, zmm14, 0x96
vpxorq zmm6, zmm6, zmm30
vmovdqu64 ZMMWORD PTR [rsi + 896], zmm6
vmovdqu64 zmm10, ZMMWORD PTR [rdi + 4928]
vmovdqu64 zmm13, ZMMWORD PTR [rdi + 4736]
vmovdqu64 ZMMWORD PTR [rsp + 1536], zmm2
vmovdqu64 zmm2, ZMMWORD PTR [rsp + 1024]
vpxorq zmm13, zmm13, zmm2
vmovdqu64 zmm21, ZMMWORD PTR [rdi + 4352]
vmovdqu64 zmm23, ZMMWORD PTR [rdi + 4160]
vmovdqu64 zmm2, ZMMWORD PTR [rsp + 960]
vpternlogq zmm11, zmm23, zmm2, 0x96
vpxorq zmm21, zmm23, zmm21
vmovdqu64 zmm29, ZMMWORD PTR [rdi + 3200]
vmovdqu64 zmm14, ZMMWORD PTR [rdi + 3008]
vmovdqu64 zmm25, ZMMWORD PTR [rdi + 2624]
vpxorq zmm9, zmm9, zmm25
vmovdqu64 zmm28, ZMMWORD PTR [rdi + 2432]
vpxorq zmm8, zmm28, zmm8
vmovdqu64 zmm2, ZMMWORD PTR [rsp + 896]
vpxorq zmm2, zmm2, zmm8
vmovdqu64 ZMMWORD PTR [rsp + 896], zmm25
vmovdqu64 ZMMWORD PTR [rsp + 960], zmm28
vmovdqu64 ZMMWORD PTR [rsp + 1024], zmm2
vmovdqu64 zmm28, ZMMWORD PTR [rdi + 4864]
vmovdqu64 zmm25, ZMMWORD PTR [rdi + 4672]
vmovdqu64 zmm18, ZMMWORD PTR [rdi + 4288]
vmovdqu64 zmm26, ZMMWORD PTR [rdi + 4096]
vmovdqu64 zmm2, ZMMWORD PTR [rdi + 3136]
vmovdqu64 zmm4, ZMMWORD PTR [rdi + 2944]
vmovdqu64 ZMMWORD PTR [rsp + 1600], zmm23
vmovdqu64 zmm23, ZMMWORD PTR [rsp + 128]
vmovdqu64 ZMMWORD PTR [rsp + 192], zmm31
vmovdqu64 zmm31, ZMMWORD PTR [rsp + 1152]
vpternlogq zmm23, zmm31, zmm4, 0x96
vpxorq zmm12, zmm12, zmm4
vpternlogq zmm14, zmm8, zmm12, 0x96
vpxorq zmm1, zmm1, zmm12
vmovdqu64 zmm12, ZMMWORD PTR [rsp + 832]
vpternlogq zmm11, zmm12, zmm14, 0x96
vpternlogq zmm11, zmm20, zmm13, 0x96
vmovdqu64 zmm19, ZMMWORD PTR [rdi + 2560]
vpxorq zmm24, zmm24, zmm19
vpxorq zmm2, zmm2, zmm24
vmovdqu64 zmm20, ZMMWORD PTR [rsp + 64]
vpternlogq zmm20, zmm18, zmm24, 0x96
vmovdqu64 zmm24, ZMMWORD PTR [rsp + 1920]
vpternlogq zmm24, zmm18, zmm2, 0x96
vmovdqu64 zmm18, ZMMWORD PTR [rsp + 384]
vpternlogq zmm28, zmm18, zmm24, 0x96
vmovdqu64 zmm6, ZMMWORD PTR [rdi + 2368]
vmovdqa64 zmm24, zmm6
vmovdqu64 ZMMWORD PTR [rsi + 1920], zmm24
vpxorq zmm16, zmm16, zmm6
vmovdqu64 zmm6, ZMMWORD PTR [rsp + 1856]
vpternlogq zmm6, zmm4, zmm16, 0x96
vpxorq zmm6, zmm6, zmm3
vmovdqu64 ZMMWORD PTR [rsi + 1664], zmm6
vpxorq zmm17, zmm16, zmm17
vmovdqa64 zmm6, zmm17
vpxorq zmm6, zmm6, zmm14
vmovdqu64 ZMMWORD PTR [rsi + 1600], zmm6
vpxorq zmm14, zmm14, zmm9
vpternlogq zmm29, zmm5, zmm14, 0x96
vpxorq zmm26, zmm26, zmm17
vmovdqu64 zmm14, ZMMWORD PTR [rsp + 256]
vpternlogq zmm14, zmm26, zmm22, 0x96
vpxorq zmm14, zmm14, zmm7
vmovdqu64 ZMMWORD PTR [rsi + 1408], zmm14
vpxorq zmm15, zmm15, zmm26
vpxorq zmm17, zmm17, zmm2
vmovdqu64 zmm2, ZMMWORD PTR [rsp + 704]
vpternlogq zmm2, zmm7, zmm17, 0x96
vpxorq zmm2, zmm2, zmm29
vmovdqu64 ZMMWORD PTR [rsi + 1472], zmm2
vpxorq zmm0, zmm0, zmm15
vmovdqu64 zmm2, ZMMWORD PTR [rsp + 0]
vmovdqu64 zmm7, ZMMWORD PTR [rsp + 1984]
vpternlogq zmm2, zmm7, zmm0, 0x96
vpxorq zmm25, zmm25, zmm0
vmovdqu64 zmm0, ZMMWORD PTR [rsp + 1792]
vpternlogq zmm10, zmm25, zmm0, 0x96
vpternlogq zmm10, zmm21, zmm13, 0x96
vmovdqu64 zmm13, ZMMWORD PTR [rsp + 320]
vpternlogq zmm10, zmm13, zmm27, 0x96
vpternlogq zmm2, zmm9, zmm21, 0x96
vpternlogq zmm10, zmm29, zmm28, 0x96
vpxorq zmm10, zmm10, zmm30
vmovdqu64 ZMMWORD PTR [rsi + 960], zmm10
vmovdqu64 zmm10, ZMMWORD PTR [rsp + 1408]
vpternlogq zmm23, zmm10, zmm25, 0x96
vmovdqu64 zmm10, ZMMWORD PTR [rsp + 192]
vpxorq zmm23, zmm23, zmm10
vmovdqu64 ZMMWORD PTR [rsi + 1152], zmm23
vmovdqu64 zmm23, ZMMWORD PTR [rsp + 1344]
vpxorq zmm23, zmm23, zmm25
vpxorq zmm11, zmm11, zmm23
vmovdqu64 ZMMWORD PTR [rsi + 1088], zmm11
vmovdqu64 zmm11, ZMMWORD PTR [rsp + 1536]
vpternlogq zmm11, zmm1, zmm23, 0x96
vpxorq zmm11, zmm11, zmm28
vmovdqu64 ZMMWORD PTR [rsi + 1024], zmm11
vmovdqu64 zmm11, ZMMWORD PTR [rsp + 1280]
vpxorq zmm11, zmm11, zmm15
vmovdqu64 zmm15, ZMMWORD PTR [rsp + 1728]
vmovdqu64 zmm28, ZMMWORD PTR [rsp + 2048]
vpternlogq zmm15, zmm28, zmm11, 0x96
vpxorq zmm15, zmm15, zmm20
vmovdqu64 ZMMWORD PTR [rsi + 1280], zmm15
vmovdqu64 zmm15, ZMMWORD PTR [rsp + 1600]
vpternlogq zmm15, zmm5, zmm11, 0x96
vmovdqu64 zmm11, ZMMWORD PTR [rsp + 1024]
vpxorq zmm15, zmm15, zmm11
vmovdqu64 ZMMWORD PTR [rsi + 1344], zmm15
vpternlogq zmm2, zmm11, zmm20, 0x96
vpxorq zmm2, zmm2, zmm10
vmovdqu64 ZMMWORD PTR [rsi + 1216], zmm2
vmovdqu64 zmm2, ZMMWORD PTR [rsp + 768]
vpxorq zmm2, zmm2, zmm1
vpxorq zmm2, zmm2, zmm17
vmovdqu64 ZMMWORD PTR [rsi + 1536], zmm2
vmovdqu64 zmm2, ZMMWORD PTR [rsp + 960]
vpxorq zmm2, zmm2, zmm16
vmovdqa64 zmm17, zmm2
vmovdqu64 ZMMWORD PTR [rsi + 1856], zmm17
vmovdqu64 zmm17, ZMMWORD PTR [rsp + 2112]
vpxorq zmm17, zmm17, zmm19
vmovdqu64 zmm1, ZMMWORD PTR [rsp + 896]
vpternlogq zmm17, zmm1, zmm2, 0x96
vpxorq zmm17, zmm17, zmm3
vmovdqu64 ZMMWORD PTR [rsi + 1728], zmm17
vmovdqu64 zmm17, ZMMWORD PTR [rsp + 2176]
vpxorq zmm17, zmm17, zmm19
vpxorq zmm17, zmm17, zmm16
vmovdqu64 ZMMWORD PTR [rsi + 1792], zmm17
mov rsp, rbp
pop rbp
vzeroupper
ret
.size flock_ab_leaf_interpolate2, .-flock_ab_leaf_interpolate2
.p2align 5
.globl flock_ab_poly_finalize
.type flock_ab_poly_finalize, @function
flock_ab_poly_finalize:
vmovdqu64 zmm16,ZMMWORD PTR [rdi]
vgf2p8affineqb zmm0,zmm16,QWORD PTR [rsi + 0]{1to8},0
vgf2p8affineqb zmm1,zmm16,QWORD PTR [rsi + 8]{1to8},0
vgf2p8affineqb zmm2,zmm16,QWORD PTR [rsi + 16]{1to8},0
vgf2p8affineqb zmm3,zmm16,QWORD PTR [rsi + 24]{1to8},0
vgf2p8affineqb zmm4,zmm16,QWORD PTR [rsi + 32]{1to8},0
vgf2p8affineqb zmm5,zmm16,QWORD PTR [rsi + 40]{1to8},0
vgf2p8affineqb zmm6,zmm16,QWORD PTR [rsi + 48]{1to8},0
vgf2p8affineqb zmm7,zmm16,QWORD PTR [rsi + 56]{1to8},0
vgf2p8affineqb zmm8,zmm16,QWORD PTR [rsi + 64]{1to8},0
vgf2p8affineqb zmm9,zmm16,QWORD PTR [rsi + 72]{1to8},0
vgf2p8affineqb zmm10,zmm16,QWORD PTR [rsi + 80]{1to8},0
vgf2p8affineqb zmm11,zmm16,QWORD PTR [rsi + 88]{1to8},0
vgf2p8affineqb zmm12,zmm16,QWORD PTR [rsi + 96]{1to8},0
vgf2p8affineqb zmm13,zmm16,QWORD PTR [rsi + 104]{1to8},0
vgf2p8affineqb zmm14,zmm16,QWORD PTR [rsi + 112]{1to8},0
vgf2p8affineqb zmm15,zmm16,QWORD PTR [rsi + 120]{1to8},0
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 64]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 128]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 128]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 256]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 136]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 264]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 144]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 272]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 152]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 280]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 160]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 288]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 168]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 296]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 176]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 304]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 184]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 312]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 192]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 320]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 200]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 328]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 208]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 336]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 216]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 344]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 224]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 352]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 232]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 360]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 240]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 368]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 248]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 376]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 192]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 256]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 384]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 512]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 392]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 520]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 400]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 528]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 408]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 536]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 416]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 544]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 424]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 552]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 432]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 560]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 440]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 568]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 448]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 576]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 456]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 584]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 464]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 592]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 472]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 600]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 480]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 608]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 488]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 616]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 496]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 624]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 504]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 632]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 320]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 384]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 640]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 768]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 648]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 776]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 656]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 784]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 664]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 792]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 672]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 800]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 680]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 808]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 688]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 816]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 696]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 824]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 704]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 832]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 712]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 840]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 720]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 848]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 728]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 856]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 736]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 864]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 744]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 872]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 752]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 880]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 760]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 888]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 448]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 512]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 896]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1024]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 904]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1032]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 912]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1040]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 920]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1048]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 928]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1056]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 936]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1064]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 944]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1072]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 952]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1080]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 960]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1088]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 968]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1096]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 976]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1104]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 984]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1112]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 992]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1120]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1000]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1128]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1008]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1136]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1016]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1144]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 576]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 640]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1152]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1280]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1160]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1288]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1168]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1296]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1176]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1304]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1184]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1312]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1192]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1320]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1200]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1328]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1208]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1336]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1216]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1344]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1224]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1352]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1232]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1360]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1240]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1368]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1248]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1376]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1256]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1384]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1264]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1392]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1272]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1400]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 704]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 768]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1408]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1536]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1416]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1544]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1424]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1552]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1432]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1560]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1440]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1568]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1448]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1576]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1456]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1584]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1464]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1592]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1472]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1600]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1480]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1608]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1488]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1616]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1496]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1624]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1504]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1632]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1512]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1640]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1520]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1648]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1528]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1656]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 832]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 896]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1664]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1792]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1672]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1800]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1680]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1808]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1688]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1816]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1696]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1824]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1704]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1832]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1712]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1840]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1720]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1848]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1728]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1856]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1736]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1864]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1744]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1872]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1752]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1880]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1760]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1888]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1768]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1896]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1776]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1904]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1784]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 1912]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 960]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 1024]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1920]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2048]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1928]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2056]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1936]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2064]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1944]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2072]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1952]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2080]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1960]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2088]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1968]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2096]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1976]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2104]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1984]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2112]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 1992]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2120]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2000]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2128]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2008]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2136]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2016]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2144]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2024]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2152]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2032]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2160]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2040]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2168]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 1088]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 1152]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2176]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2304]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2184]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2312]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2192]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2320]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2200]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2328]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2208]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2336]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2216]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2344]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2224]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2352]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2232]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2360]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2240]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2368]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2248]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2376]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2256]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2384]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2264]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2392]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2272]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2400]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2280]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2408]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2288]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2416]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2296]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2424]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 1216]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 1280]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2432]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2560]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2440]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2568]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2448]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2576]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2456]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2584]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2464]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2592]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2472]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2600]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2480]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2608]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2488]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2616]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2496]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2624]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2504]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2632]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2512]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2640]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2520]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2648]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2528]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2656]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2536]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2664]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2544]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2672]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2552]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2680]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 1344]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 1408]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2688]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2816]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2696]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2824]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2704]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2832]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2712]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2840]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2720]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2848]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2728]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2856]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2736]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2864]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2744]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2872]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2752]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2880]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2760]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2888]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2768]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2896]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2776]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2904]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2784]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2912]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2792]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2920]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2800]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2928]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2808]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 2936]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 1472]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 1536]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2944]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3072]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2952]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3080]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2960]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3088]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2968]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3096]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2976]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3104]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2984]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3112]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 2992]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3120]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3000]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3128]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3008]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3136]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3016]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3144]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3024]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3152]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3032]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3160]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3040]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3168]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3048]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3176]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3056]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3184]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3064]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3192]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 1600]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 1664]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3200]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3328]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3208]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3336]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3216]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3344]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3224]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3352]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3232]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3360]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3240]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3368]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3248]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3376]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3256]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3384]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3264]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3392]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3272]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3400]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3280]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3408]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3288]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3416]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3296]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3424]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3304]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3432]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3312]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3440]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3320]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3448]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 zmm16,ZMMWORD PTR [rdi + 1728]
vmovdqu64 zmm17,ZMMWORD PTR [rdi + 1792]
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3456]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3584]{1to8},0
vpternlogq zmm0,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3464]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3592]{1to8},0
vpternlogq zmm1,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3472]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3600]{1to8},0
vpternlogq zmm2,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3480]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3608]{1to8},0
vpternlogq zmm3,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3488]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3616]{1to8},0
vpternlogq zmm4,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3496]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3624]{1to8},0
vpternlogq zmm5,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3504]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3632]{1to8},0
vpternlogq zmm6,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3512]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3640]{1to8},0
vpternlogq zmm7,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3520]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3648]{1to8},0
vpternlogq zmm8,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3528]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3656]{1to8},0
vpternlogq zmm9,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3536]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3664]{1to8},0
vpternlogq zmm10,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3544]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3672]{1to8},0
vpternlogq zmm11,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3552]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3680]{1to8},0
vpternlogq zmm12,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3560]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3688]{1to8},0
vpternlogq zmm13,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3568]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3696]{1to8},0
vpternlogq zmm14,zmm18,zmm19,0x96
vgf2p8affineqb zmm18,zmm16,QWORD PTR [rsi + 3576]{1to8},0
vgf2p8affineqb zmm19,zmm17,QWORD PTR [rsi + 3704]{1to8},0
vpternlogq zmm15,zmm18,zmm19,0x96
vmovdqu64 ZMMWORD PTR [rdx + 0],zmm0
vmovdqu64 ZMMWORD PTR [rdx + 64],zmm1
vmovdqu64 ZMMWORD PTR [rdx + 128],zmm2
vmovdqu64 ZMMWORD PTR [rdx + 192],zmm3
vmovdqu64 ZMMWORD PTR [rdx + 256],zmm4
vmovdqu64 ZMMWORD PTR [rdx + 320],zmm5
vmovdqu64 ZMMWORD PTR [rdx + 384],zmm6
vmovdqu64 ZMMWORD PTR [rdx + 448],zmm7
vmovdqu64 ZMMWORD PTR [rdx + 512],zmm8
vmovdqu64 ZMMWORD PTR [rdx + 576],zmm9
vmovdqu64 ZMMWORD PTR [rdx + 640],zmm10
vmovdqu64 ZMMWORD PTR [rdx + 704],zmm11
vmovdqu64 ZMMWORD PTR [rdx + 768],zmm12
vmovdqu64 ZMMWORD PTR [rdx + 832],zmm13
vmovdqu64 ZMMWORD PTR [rdx + 896],zmm14
vmovdqu64 ZMMWORD PTR [rdx + 960],zmm15
vzeroupper
ret
.size flock_ab_poly_finalize, .-flock_ab_poly_finalize

.section .note.GNU-stack,"",@progbits

.text
.p2align 5
// Grouped bank ABI: ZMM0..26 are leaf sums, ZMM27..30 hold source forms,
// and ZMM31 is transient. All are caller-saved under the existing C ABI.
// Bank zero groups are 27/24/27 leaves; bank two groups are 24/27/27.
// Each group visits every one of the same 32 windows before its stores.
.globl flock_ab_merged_bank0_m0
.type flock_ab_merged_bank0_m0,@function
flock_ab_merged_bank0_m0:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+0]
mov ecx,32
.p2align 5
.Lmerged_0_0_0:
vmovdqu64 zmm27, [r8+0]
vmovdqu64 zmm28, [r8+64]
vmovdqu64 zmm29, [r8+128]
vmovdqu64 zmm30, [r8+192]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+256]
vmovdqu64 zmm28, [r8+320]
vmovdqu64 zmm29, [r8+384]
vmovdqu64 zmm30, [r8+448]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+0]
vpxorq zmm27, zmm27, [r8+256]
vmovdqu64 zmm28, [r8+64]
vpxorq zmm28, zmm28, [r8+320]
vmovdqu64 zmm29, [r8+128]
vpxorq zmm29, zmm29, [r8+384]
vmovdqu64 zmm30, [r8+192]
vpxorq zmm30, zmm30, [r8+448]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_0_0_0
vmovdqu64 [rdx+0],zmm0
vmovdqu64 [rdx+64],zmm1
vmovdqu64 [rdx+128],zmm2
vmovdqu64 [rdx+192],zmm3
vmovdqu64 [rdx+256],zmm4
vmovdqu64 [rdx+320],zmm5
vmovdqu64 [rdx+384],zmm6
vmovdqu64 [rdx+448],zmm7
vmovdqu64 [rdx+512],zmm8
vmovdqu64 [rdx+576],zmm9
vmovdqu64 [rdx+640],zmm10
vmovdqu64 [rdx+704],zmm11
vmovdqu64 [rdx+768],zmm12
vmovdqu64 [rdx+832],zmm13
vmovdqu64 [rdx+896],zmm14
vmovdqu64 [rdx+960],zmm15
vmovdqu64 [rdx+1024],zmm16
vmovdqu64 [rdx+1088],zmm17
vmovdqu64 [rdx+1152],zmm18
vmovdqu64 [rdx+1216],zmm19
vmovdqu64 [rdx+1280],zmm20
vmovdqu64 [rdx+1344],zmm21
vmovdqu64 [rdx+1408],zmm22
vmovdqu64 [rdx+1472],zmm23
vmovdqu64 [rdx+1536],zmm24
vmovdqu64 [rdx+1600],zmm25
vmovdqu64 [rdx+1664],zmm26
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
mov r8,rdi
lea r9,[rsi+6912]
mov ecx,32
.p2align 5
.Lmerged_0_0_1:
vmovdqu64 zmm27, [r8+512]
vmovdqu64 zmm28, [r8+576]
vmovdqu64 zmm29, [r8+640]
vmovdqu64 zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+768]
vmovdqu64 zmm28, [r8+832]
vmovdqu64 zmm29, [r8+768]
vpxorq zmm29, zmm29, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vmovdqu64 zmm27, [r8+512]
vpxorq zmm27, zmm27, [r8+768]
vmovdqu64 zmm28, [r8+576]
vpxorq zmm28, zmm28, [r8+832]
vmovdqu64 zmm29, [r8+640]
vmovdqu64 zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
add r8,118784
add r9,192
dec ecx
jnz .Lmerged_0_0_1
vmovdqu64 [rdx+1728],zmm0
vmovdqu64 [rdx+1792],zmm1
vmovdqu64 [rdx+1856],zmm2
vmovdqu64 [rdx+1920],zmm3
vmovdqu64 [rdx+1984],zmm4
vmovdqu64 [rdx+2048],zmm5
vmovdqu64 [rdx+2112],zmm6
vmovdqu64 [rdx+2176],zmm7
vmovdqu64 [rdx+2240],zmm8
vmovdqu64 [rdx+2304],zmm9
vmovdqu64 [rdx+2368],zmm10
vmovdqu64 [rdx+2432],zmm11
vmovdqu64 [rdx+2496],zmm12
vmovdqu64 [rdx+2560],zmm13
vmovdqu64 [rdx+2624],zmm14
vmovdqu64 [rdx+2688],zmm15
vmovdqu64 [rdx+2752],zmm16
vmovdqu64 [rdx+2816],zmm17
vmovdqu64 [rdx+2880],zmm18
vmovdqu64 [rdx+2944],zmm19
vmovdqu64 [rdx+3008],zmm20
vmovdqu64 [rdx+3072],zmm21
vmovdqu64 [rdx+3136],zmm22
vmovdqu64 [rdx+3200],zmm23
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+13056]
mov ecx,32
.p2align 5
.Lmerged_0_0_2:
vmovdqu64 zmm27, [r8+0]
vpxorq zmm27, zmm27, [r8+512]
vmovdqu64 zmm28, [r8+64]
vpxorq zmm28, zmm28, [r8+576]
vmovdqu64 zmm29, [r8+128]
vpxorq zmm29, zmm29, [r8+640]
vmovdqu64 zmm30, [r8+192]
vpxorq zmm30, zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+256]
vpxorq zmm27, zmm27, [r8+768]
vmovdqu64 zmm28, [r8+320]
vpxorq zmm28, zmm28, [r8+832]
vmovdqu64 zmm29, [r8+384]
vmovdqu64 zmm30, [r8+448]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+0]
vpxorq zmm27, zmm27, [r8+256]
vpxorq zmm27, zmm27, [r8+512]
vpxorq zmm27, zmm27, [r8+768]
vmovdqu64 zmm28, [r8+64]
vpxorq zmm28, zmm28, [r8+320]
vpxorq zmm28, zmm28, [r8+576]
vpxorq zmm28, zmm28, [r8+832]
vmovdqu64 zmm29, [r8+128]
vpxorq zmm29, zmm29, [r8+384]
vpxorq zmm29, zmm29, [r8+640]
vmovdqu64 zmm30, [r8+192]
vpxorq zmm30, zmm30, [r8+448]
vpxorq zmm30, zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_0_0_2
vmovdqu64 [rdx+3264],zmm0
vmovdqu64 [rdx+3328],zmm1
vmovdqu64 [rdx+3392],zmm2
vmovdqu64 [rdx+3456],zmm3
vmovdqu64 [rdx+3520],zmm4
vmovdqu64 [rdx+3584],zmm5
vmovdqu64 [rdx+3648],zmm6
vmovdqu64 [rdx+3712],zmm7
vmovdqu64 [rdx+3776],zmm8
vmovdqu64 [rdx+3840],zmm9
vmovdqu64 [rdx+3904],zmm10
vmovdqu64 [rdx+3968],zmm11
vmovdqu64 [rdx+4032],zmm12
vmovdqu64 [rdx+4096],zmm13
vmovdqu64 [rdx+4160],zmm14
vmovdqu64 [rdx+4224],zmm15
vmovdqu64 [rdx+4288],zmm16
vmovdqu64 [rdx+4352],zmm17
vmovdqu64 [rdx+4416],zmm18
vmovdqu64 [rdx+4480],zmm19
vmovdqu64 [rdx+4544],zmm20
vmovdqu64 [rdx+4608],zmm21
vmovdqu64 [rdx+4672],zmm22
vmovdqu64 [rdx+4736],zmm23
vmovdqu64 [rdx+4800],zmm24
vmovdqu64 [rdx+4864],zmm25
vmovdqu64 [rdx+4928],zmm26
ret
.size flock_ab_merged_bank0_m0,.-flock_ab_merged_bank0_m0
.p2align 5
.globl flock_ab_merged_bank0_m1
.type flock_ab_merged_bank0_m1,@function
flock_ab_merged_bank0_m1:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+0]
mov ecx,32
.p2align 5
.Lmerged_0_1_0:
vmovdqu64 zmm27, [r8+0]
vmovdqu64 zmm28, [r8+64]
vmovdqu64 zmm29, [r8+128]
vmovdqu64 zmm30, [r8+192]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
prefetcht1 [r8 + 0]
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
prefetcht1 [r8 + 64]
prefetcht1 [r8 + 128]
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
prefetcht1 [r8 + 192]
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
prefetcht1 [r8 + 256]
prefetcht1 [r8 + 320]
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
prefetcht1 [r8 + 384]
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
prefetcht1 [r8 + 448]
prefetcht1 [r8 + 512]
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
prefetcht1 [r8 + 576]
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
prefetcht1 [r8 + 640]
prefetcht1 [r8 + 704]
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
prefetcht1 [r8 + 768]
prefetcht1 [r8 + 832]
vmovdqu64 zmm27, [r8+256]
vmovdqu64 zmm28, [r8+320]
vmovdqu64 zmm29, [r8+384]
vmovdqu64 zmm30, [r8+448]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+0]
vpxorq zmm27, zmm27, [r8+256]
vmovdqu64 zmm28, [r8+64]
vpxorq zmm28, zmm28, [r8+320]
vmovdqu64 zmm29, [r8+128]
vpxorq zmm29, zmm29, [r8+384]
vmovdqu64 zmm30, [r8+192]
vpxorq zmm30, zmm30, [r8+448]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_0_1_0
vmovdqu64 [rdx+0],zmm0
vmovdqu64 [rdx+64],zmm1
vmovdqu64 [rdx+128],zmm2
vmovdqu64 [rdx+192],zmm3
vmovdqu64 [rdx+256],zmm4
vmovdqu64 [rdx+320],zmm5
vmovdqu64 [rdx+384],zmm6
vmovdqu64 [rdx+448],zmm7
vmovdqu64 [rdx+512],zmm8
vmovdqu64 [rdx+576],zmm9
vmovdqu64 [rdx+640],zmm10
vmovdqu64 [rdx+704],zmm11
vmovdqu64 [rdx+768],zmm12
vmovdqu64 [rdx+832],zmm13
vmovdqu64 [rdx+896],zmm14
vmovdqu64 [rdx+960],zmm15
vmovdqu64 [rdx+1024],zmm16
vmovdqu64 [rdx+1088],zmm17
vmovdqu64 [rdx+1152],zmm18
vmovdqu64 [rdx+1216],zmm19
vmovdqu64 [rdx+1280],zmm20
vmovdqu64 [rdx+1344],zmm21
vmovdqu64 [rdx+1408],zmm22
vmovdqu64 [rdx+1472],zmm23
vmovdqu64 [rdx+1536],zmm24
vmovdqu64 [rdx+1600],zmm25
vmovdqu64 [rdx+1664],zmm26
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
mov r8,rdi
lea r9,[rsi+6912]
mov ecx,32
.p2align 5
.Lmerged_0_1_1:
vmovdqu64 zmm27, [r8+512]
vmovdqu64 zmm28, [r8+576]
vmovdqu64 zmm29, [r8+640]
vmovdqu64 zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+768]
vmovdqu64 zmm28, [r8+832]
vmovdqu64 zmm29, [r8+768]
vpxorq zmm29, zmm29, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vmovdqu64 zmm27, [r8+512]
vpxorq zmm27, zmm27, [r8+768]
vmovdqu64 zmm28, [r8+576]
vpxorq zmm28, zmm28, [r8+832]
vmovdqu64 zmm29, [r8+640]
vmovdqu64 zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
add r8,118784
add r9,192
dec ecx
jnz .Lmerged_0_1_1
vmovdqu64 [rdx+1728],zmm0
vmovdqu64 [rdx+1792],zmm1
vmovdqu64 [rdx+1856],zmm2
vmovdqu64 [rdx+1920],zmm3
vmovdqu64 [rdx+1984],zmm4
vmovdqu64 [rdx+2048],zmm5
vmovdqu64 [rdx+2112],zmm6
vmovdqu64 [rdx+2176],zmm7
vmovdqu64 [rdx+2240],zmm8
vmovdqu64 [rdx+2304],zmm9
vmovdqu64 [rdx+2368],zmm10
vmovdqu64 [rdx+2432],zmm11
vmovdqu64 [rdx+2496],zmm12
vmovdqu64 [rdx+2560],zmm13
vmovdqu64 [rdx+2624],zmm14
vmovdqu64 [rdx+2688],zmm15
vmovdqu64 [rdx+2752],zmm16
vmovdqu64 [rdx+2816],zmm17
vmovdqu64 [rdx+2880],zmm18
vmovdqu64 [rdx+2944],zmm19
vmovdqu64 [rdx+3008],zmm20
vmovdqu64 [rdx+3072],zmm21
vmovdqu64 [rdx+3136],zmm22
vmovdqu64 [rdx+3200],zmm23
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+13056]
mov ecx,32
.p2align 5
.Lmerged_0_1_2:
vmovdqu64 zmm27, [r8+0]
vpxorq zmm27, zmm27, [r8+512]
vmovdqu64 zmm28, [r8+64]
vpxorq zmm28, zmm28, [r8+576]
vmovdqu64 zmm29, [r8+128]
vpxorq zmm29, zmm29, [r8+640]
vmovdqu64 zmm30, [r8+192]
vpxorq zmm30, zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+256]
vpxorq zmm27, zmm27, [r8+768]
vmovdqu64 zmm28, [r8+320]
vpxorq zmm28, zmm28, [r8+832]
vmovdqu64 zmm29, [r8+384]
vmovdqu64 zmm30, [r8+448]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+0]
vpxorq zmm27, zmm27, [r8+256]
vpxorq zmm27, zmm27, [r8+512]
vpxorq zmm27, zmm27, [r8+768]
vmovdqu64 zmm28, [r8+64]
vpxorq zmm28, zmm28, [r8+320]
vpxorq zmm28, zmm28, [r8+576]
vpxorq zmm28, zmm28, [r8+832]
vmovdqu64 zmm29, [r8+128]
vpxorq zmm29, zmm29, [r8+384]
vpxorq zmm29, zmm29, [r8+640]
vmovdqu64 zmm30, [r8+192]
vpxorq zmm30, zmm30, [r8+448]
vpxorq zmm30, zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_0_1_2
vmovdqu64 [rdx+3264],zmm0
vmovdqu64 [rdx+3328],zmm1
vmovdqu64 [rdx+3392],zmm2
vmovdqu64 [rdx+3456],zmm3
vmovdqu64 [rdx+3520],zmm4
vmovdqu64 [rdx+3584],zmm5
vmovdqu64 [rdx+3648],zmm6
vmovdqu64 [rdx+3712],zmm7
vmovdqu64 [rdx+3776],zmm8
vmovdqu64 [rdx+3840],zmm9
vmovdqu64 [rdx+3904],zmm10
vmovdqu64 [rdx+3968],zmm11
vmovdqu64 [rdx+4032],zmm12
vmovdqu64 [rdx+4096],zmm13
vmovdqu64 [rdx+4160],zmm14
vmovdqu64 [rdx+4224],zmm15
vmovdqu64 [rdx+4288],zmm16
vmovdqu64 [rdx+4352],zmm17
vmovdqu64 [rdx+4416],zmm18
vmovdqu64 [rdx+4480],zmm19
vmovdqu64 [rdx+4544],zmm20
vmovdqu64 [rdx+4608],zmm21
vmovdqu64 [rdx+4672],zmm22
vmovdqu64 [rdx+4736],zmm23
vmovdqu64 [rdx+4800],zmm24
vmovdqu64 [rdx+4864],zmm25
vmovdqu64 [rdx+4928],zmm26
ret
.size flock_ab_merged_bank0_m1,.-flock_ab_merged_bank0_m1
.p2align 5
.globl flock_ab_merged_bank0_m2
.type flock_ab_merged_bank0_m2,@function
flock_ab_merged_bank0_m2:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+0]
mov ecx,32
.p2align 5
.Lmerged_0_2_0:
vmovdqu64 zmm27, [r8+0]
vmovdqu64 zmm28, [r8+64]
vmovdqu64 zmm29, [r8+128]
vmovdqu64 zmm30, [r8+192]
prefetcht1 [r8 + 0]
prefetcht1 [r8 + 64]
prefetcht1 [r8 + 128]
prefetcht1 [r8 + 192]
prefetcht1 [r8 + 256]
prefetcht1 [r8 + 320]
prefetcht1 [r8 + 384]
prefetcht1 [r8 + 448]
prefetcht1 [r8 + 512]
prefetcht1 [r8 + 576]
prefetcht1 [r8 + 640]
prefetcht1 [r8 + 704]
prefetcht1 [r8 + 768]
prefetcht1 [r8 + 832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+256]
vmovdqu64 zmm28, [r8+320]
vmovdqu64 zmm29, [r8+384]
vmovdqu64 zmm30, [r8+448]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+0]
vpxorq zmm27, zmm27, [r8+256]
vmovdqu64 zmm28, [r8+64]
vpxorq zmm28, zmm28, [r8+320]
vmovdqu64 zmm29, [r8+128]
vpxorq zmm29, zmm29, [r8+384]
vmovdqu64 zmm30, [r8+192]
vpxorq zmm30, zmm30, [r8+448]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_0_2_0
vmovdqu64 [rdx+0],zmm0
vmovdqu64 [rdx+64],zmm1
vmovdqu64 [rdx+128],zmm2
vmovdqu64 [rdx+192],zmm3
vmovdqu64 [rdx+256],zmm4
vmovdqu64 [rdx+320],zmm5
vmovdqu64 [rdx+384],zmm6
vmovdqu64 [rdx+448],zmm7
vmovdqu64 [rdx+512],zmm8
vmovdqu64 [rdx+576],zmm9
vmovdqu64 [rdx+640],zmm10
vmovdqu64 [rdx+704],zmm11
vmovdqu64 [rdx+768],zmm12
vmovdqu64 [rdx+832],zmm13
vmovdqu64 [rdx+896],zmm14
vmovdqu64 [rdx+960],zmm15
vmovdqu64 [rdx+1024],zmm16
vmovdqu64 [rdx+1088],zmm17
vmovdqu64 [rdx+1152],zmm18
vmovdqu64 [rdx+1216],zmm19
vmovdqu64 [rdx+1280],zmm20
vmovdqu64 [rdx+1344],zmm21
vmovdqu64 [rdx+1408],zmm22
vmovdqu64 [rdx+1472],zmm23
vmovdqu64 [rdx+1536],zmm24
vmovdqu64 [rdx+1600],zmm25
vmovdqu64 [rdx+1664],zmm26
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
mov r8,rdi
lea r9,[rsi+6912]
mov ecx,32
.p2align 5
.Lmerged_0_2_1:
vmovdqu64 zmm27, [r8+512]
vmovdqu64 zmm28, [r8+576]
vmovdqu64 zmm29, [r8+640]
vmovdqu64 zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+768]
vmovdqu64 zmm28, [r8+832]
vmovdqu64 zmm29, [r8+768]
vpxorq zmm29, zmm29, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vmovdqu64 zmm27, [r8+512]
vpxorq zmm27, zmm27, [r8+768]
vmovdqu64 zmm28, [r8+576]
vpxorq zmm28, zmm28, [r8+832]
vmovdqu64 zmm29, [r8+640]
vmovdqu64 zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
add r8,118784
add r9,192
dec ecx
jnz .Lmerged_0_2_1
vmovdqu64 [rdx+1728],zmm0
vmovdqu64 [rdx+1792],zmm1
vmovdqu64 [rdx+1856],zmm2
vmovdqu64 [rdx+1920],zmm3
vmovdqu64 [rdx+1984],zmm4
vmovdqu64 [rdx+2048],zmm5
vmovdqu64 [rdx+2112],zmm6
vmovdqu64 [rdx+2176],zmm7
vmovdqu64 [rdx+2240],zmm8
vmovdqu64 [rdx+2304],zmm9
vmovdqu64 [rdx+2368],zmm10
vmovdqu64 [rdx+2432],zmm11
vmovdqu64 [rdx+2496],zmm12
vmovdqu64 [rdx+2560],zmm13
vmovdqu64 [rdx+2624],zmm14
vmovdqu64 [rdx+2688],zmm15
vmovdqu64 [rdx+2752],zmm16
vmovdqu64 [rdx+2816],zmm17
vmovdqu64 [rdx+2880],zmm18
vmovdqu64 [rdx+2944],zmm19
vmovdqu64 [rdx+3008],zmm20
vmovdqu64 [rdx+3072],zmm21
vmovdqu64 [rdx+3136],zmm22
vmovdqu64 [rdx+3200],zmm23
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+13056]
mov ecx,32
.p2align 5
.Lmerged_0_2_2:
vmovdqu64 zmm27, [r8+0]
vpxorq zmm27, zmm27, [r8+512]
vmovdqu64 zmm28, [r8+64]
vpxorq zmm28, zmm28, [r8+576]
vmovdqu64 zmm29, [r8+128]
vpxorq zmm29, zmm29, [r8+640]
vmovdqu64 zmm30, [r8+192]
vpxorq zmm30, zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+256]
vpxorq zmm27, zmm27, [r8+768]
vmovdqu64 zmm28, [r8+320]
vpxorq zmm28, zmm28, [r8+832]
vmovdqu64 zmm29, [r8+384]
vmovdqu64 zmm30, [r8+448]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+0]
vpxorq zmm27, zmm27, [r8+256]
vpxorq zmm27, zmm27, [r8+512]
vpxorq zmm27, zmm27, [r8+768]
vmovdqu64 zmm28, [r8+64]
vpxorq zmm28, zmm28, [r8+320]
vpxorq zmm28, zmm28, [r8+576]
vpxorq zmm28, zmm28, [r8+832]
vmovdqu64 zmm29, [r8+128]
vpxorq zmm29, zmm29, [r8+384]
vpxorq zmm29, zmm29, [r8+640]
vmovdqu64 zmm30, [r8+192]
vpxorq zmm30, zmm30, [r8+448]
vpxorq zmm30, zmm30, [r8+704]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_0_2_2
vmovdqu64 [rdx+3264],zmm0
vmovdqu64 [rdx+3328],zmm1
vmovdqu64 [rdx+3392],zmm2
vmovdqu64 [rdx+3456],zmm3
vmovdqu64 [rdx+3520],zmm4
vmovdqu64 [rdx+3584],zmm5
vmovdqu64 [rdx+3648],zmm6
vmovdqu64 [rdx+3712],zmm7
vmovdqu64 [rdx+3776],zmm8
vmovdqu64 [rdx+3840],zmm9
vmovdqu64 [rdx+3904],zmm10
vmovdqu64 [rdx+3968],zmm11
vmovdqu64 [rdx+4032],zmm12
vmovdqu64 [rdx+4096],zmm13
vmovdqu64 [rdx+4160],zmm14
vmovdqu64 [rdx+4224],zmm15
vmovdqu64 [rdx+4288],zmm16
vmovdqu64 [rdx+4352],zmm17
vmovdqu64 [rdx+4416],zmm18
vmovdqu64 [rdx+4480],zmm19
vmovdqu64 [rdx+4544],zmm20
vmovdqu64 [rdx+4608],zmm21
vmovdqu64 [rdx+4672],zmm22
vmovdqu64 [rdx+4736],zmm23
vmovdqu64 [rdx+4800],zmm24
vmovdqu64 [rdx+4864],zmm25
vmovdqu64 [rdx+4928],zmm26
ret
.size flock_ab_merged_bank0_m2,.-flock_ab_merged_bank0_m2
.p2align 5
.globl flock_ab_merged_bank2_m0
.type flock_ab_merged_bank2_m0,@function
flock_ab_merged_bank2_m0:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
mov r8,rdi
lea r9,[rsi+0]
mov ecx,32
.p2align 5
.Lmerged_2_0_0:
vmovdqu64 zmm27, [r8+0]
vmovdqu64 zmm28, [r8+64]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+64]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vmovdqu64 zmm27, [r8+128]
vmovdqu64 zmm28, [r8+192]
vmovdqu64 zmm29, [r8+256]
vmovdqu64 zmm30, [r8+320]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vmovdqu64 zmm27, [r8+128]
vmovdqu64 zmm28, [r8+192]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+256]
vmovdqu64 zmm30, [r8+64]
vpxorq zmm30, zmm30, [r8+320]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
add r8,118784
add r9,192
dec ecx
jnz .Lmerged_2_0_0
vmovdqu64 [rdx+0],zmm0
vmovdqu64 [rdx+64],zmm1
vmovdqu64 [rdx+128],zmm2
vmovdqu64 [rdx+192],zmm3
vmovdqu64 [rdx+256],zmm4
vmovdqu64 [rdx+320],zmm5
vmovdqu64 [rdx+384],zmm6
vmovdqu64 [rdx+448],zmm7
vmovdqu64 [rdx+512],zmm8
vmovdqu64 [rdx+576],zmm9
vmovdqu64 [rdx+640],zmm10
vmovdqu64 [rdx+704],zmm11
vmovdqu64 [rdx+768],zmm12
vmovdqu64 [rdx+832],zmm13
vmovdqu64 [rdx+896],zmm14
vmovdqu64 [rdx+960],zmm15
vmovdqu64 [rdx+1024],zmm16
vmovdqu64 [rdx+1088],zmm17
vmovdqu64 [rdx+1152],zmm18
vmovdqu64 [rdx+1216],zmm19
vmovdqu64 [rdx+1280],zmm20
vmovdqu64 [rdx+1344],zmm21
vmovdqu64 [rdx+1408],zmm22
vmovdqu64 [rdx+1472],zmm23
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+6144]
mov ecx,32
.p2align 5
.Lmerged_2_0_1:
vmovdqu64 zmm27, [r8+384]
vmovdqu64 zmm28, [r8+448]
vmovdqu64 zmm29, [r8+512]
vmovdqu64 zmm30, [r8+576]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+640]
vmovdqu64 zmm28, [r8+704]
vmovdqu64 zmm29, [r8+768]
vmovdqu64 zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+384]
vpxorq zmm27, zmm27, [r8+640]
vmovdqu64 zmm28, [r8+448]
vpxorq zmm28, zmm28, [r8+704]
vmovdqu64 zmm29, [r8+512]
vpxorq zmm29, zmm29, [r8+768]
vmovdqu64 zmm30, [r8+576]
vpxorq zmm30, zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_2_0_1
vmovdqu64 [rdx+1536],zmm0
vmovdqu64 [rdx+1600],zmm1
vmovdqu64 [rdx+1664],zmm2
vmovdqu64 [rdx+1728],zmm3
vmovdqu64 [rdx+1792],zmm4
vmovdqu64 [rdx+1856],zmm5
vmovdqu64 [rdx+1920],zmm6
vmovdqu64 [rdx+1984],zmm7
vmovdqu64 [rdx+2048],zmm8
vmovdqu64 [rdx+2112],zmm9
vmovdqu64 [rdx+2176],zmm10
vmovdqu64 [rdx+2240],zmm11
vmovdqu64 [rdx+2304],zmm12
vmovdqu64 [rdx+2368],zmm13
vmovdqu64 [rdx+2432],zmm14
vmovdqu64 [rdx+2496],zmm15
vmovdqu64 [rdx+2560],zmm16
vmovdqu64 [rdx+2624],zmm17
vmovdqu64 [rdx+2688],zmm18
vmovdqu64 [rdx+2752],zmm19
vmovdqu64 [rdx+2816],zmm20
vmovdqu64 [rdx+2880],zmm21
vmovdqu64 [rdx+2944],zmm22
vmovdqu64 [rdx+3008],zmm23
vmovdqu64 [rdx+3072],zmm24
vmovdqu64 [rdx+3136],zmm25
vmovdqu64 [rdx+3200],zmm26
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+13056]
mov ecx,32
.p2align 5
.Lmerged_2_0_2:
vmovdqu64 zmm27, [r8+384]
vmovdqu64 zmm28, [r8+448]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+512]
vmovdqu64 zmm30, [r8+64]
vpxorq zmm30, zmm30, [r8+576]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+128]
vpxorq zmm27, zmm27, [r8+640]
vmovdqu64 zmm28, [r8+192]
vpxorq zmm28, zmm28, [r8+704]
vmovdqu64 zmm29, [r8+256]
vpxorq zmm29, zmm29, [r8+768]
vmovdqu64 zmm30, [r8+320]
vpxorq zmm30, zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+128]
vpxorq zmm27, zmm27, [r8+384]
vpxorq zmm27, zmm27, [r8+640]
vmovdqu64 zmm28, [r8+192]
vpxorq zmm28, zmm28, [r8+448]
vpxorq zmm28, zmm28, [r8+704]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+256]
vpxorq zmm29, zmm29, [r8+512]
vpxorq zmm29, zmm29, [r8+768]
vmovdqu64 zmm30, [r8+64]
vpxorq zmm30, zmm30, [r8+320]
vpxorq zmm30, zmm30, [r8+576]
vpxorq zmm30, zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_2_0_2
vmovdqu64 [rdx+3264],zmm0
vmovdqu64 [rdx+3328],zmm1
vmovdqu64 [rdx+3392],zmm2
vmovdqu64 [rdx+3456],zmm3
vmovdqu64 [rdx+3520],zmm4
vmovdqu64 [rdx+3584],zmm5
vmovdqu64 [rdx+3648],zmm6
vmovdqu64 [rdx+3712],zmm7
vmovdqu64 [rdx+3776],zmm8
vmovdqu64 [rdx+3840],zmm9
vmovdqu64 [rdx+3904],zmm10
vmovdqu64 [rdx+3968],zmm11
vmovdqu64 [rdx+4032],zmm12
vmovdqu64 [rdx+4096],zmm13
vmovdqu64 [rdx+4160],zmm14
vmovdqu64 [rdx+4224],zmm15
vmovdqu64 [rdx+4288],zmm16
vmovdqu64 [rdx+4352],zmm17
vmovdqu64 [rdx+4416],zmm18
vmovdqu64 [rdx+4480],zmm19
vmovdqu64 [rdx+4544],zmm20
vmovdqu64 [rdx+4608],zmm21
vmovdqu64 [rdx+4672],zmm22
vmovdqu64 [rdx+4736],zmm23
vmovdqu64 [rdx+4800],zmm24
vmovdqu64 [rdx+4864],zmm25
vmovdqu64 [rdx+4928],zmm26
ret
.size flock_ab_merged_bank2_m0,.-flock_ab_merged_bank2_m0
.p2align 5
.globl flock_ab_merged_bank2_m1
.type flock_ab_merged_bank2_m1,@function
flock_ab_merged_bank2_m1:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
mov r8,rdi
lea r9,[rsi+0]
mov ecx,32
.p2align 5
.Lmerged_2_1_0:
vmovdqu64 zmm27, [r8+0]
vmovdqu64 zmm28, [r8+64]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+64]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
prefetcht1 [r8 + 0]
prefetcht1 [r8 + 64]
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
prefetcht1 [r8 + 128]
prefetcht1 [r8 + 192]
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
prefetcht1 [r8 + 256]
prefetcht1 [r8 + 320]
prefetcht1 [r8 + 384]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
prefetcht1 [r8 + 448]
prefetcht1 [r8 + 512]
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
prefetcht1 [r8 + 576]
prefetcht1 [r8 + 640]
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
prefetcht1 [r8 + 704]
prefetcht1 [r8 + 768]
prefetcht1 [r8 + 832]
vmovdqu64 zmm27, [r8+128]
vmovdqu64 zmm28, [r8+192]
vmovdqu64 zmm29, [r8+256]
vmovdqu64 zmm30, [r8+320]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vmovdqu64 zmm27, [r8+128]
vmovdqu64 zmm28, [r8+192]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+256]
vmovdqu64 zmm30, [r8+64]
vpxorq zmm30, zmm30, [r8+320]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
add r8,118784
add r9,192
dec ecx
jnz .Lmerged_2_1_0
vmovdqu64 [rdx+0],zmm0
vmovdqu64 [rdx+64],zmm1
vmovdqu64 [rdx+128],zmm2
vmovdqu64 [rdx+192],zmm3
vmovdqu64 [rdx+256],zmm4
vmovdqu64 [rdx+320],zmm5
vmovdqu64 [rdx+384],zmm6
vmovdqu64 [rdx+448],zmm7
vmovdqu64 [rdx+512],zmm8
vmovdqu64 [rdx+576],zmm9
vmovdqu64 [rdx+640],zmm10
vmovdqu64 [rdx+704],zmm11
vmovdqu64 [rdx+768],zmm12
vmovdqu64 [rdx+832],zmm13
vmovdqu64 [rdx+896],zmm14
vmovdqu64 [rdx+960],zmm15
vmovdqu64 [rdx+1024],zmm16
vmovdqu64 [rdx+1088],zmm17
vmovdqu64 [rdx+1152],zmm18
vmovdqu64 [rdx+1216],zmm19
vmovdqu64 [rdx+1280],zmm20
vmovdqu64 [rdx+1344],zmm21
vmovdqu64 [rdx+1408],zmm22
vmovdqu64 [rdx+1472],zmm23
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+6144]
mov ecx,32
.p2align 5
.Lmerged_2_1_1:
vmovdqu64 zmm27, [r8+384]
vmovdqu64 zmm28, [r8+448]
vmovdqu64 zmm29, [r8+512]
vmovdqu64 zmm30, [r8+576]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+640]
vmovdqu64 zmm28, [r8+704]
vmovdqu64 zmm29, [r8+768]
vmovdqu64 zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+384]
vpxorq zmm27, zmm27, [r8+640]
vmovdqu64 zmm28, [r8+448]
vpxorq zmm28, zmm28, [r8+704]
vmovdqu64 zmm29, [r8+512]
vpxorq zmm29, zmm29, [r8+768]
vmovdqu64 zmm30, [r8+576]
vpxorq zmm30, zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_2_1_1
vmovdqu64 [rdx+1536],zmm0
vmovdqu64 [rdx+1600],zmm1
vmovdqu64 [rdx+1664],zmm2
vmovdqu64 [rdx+1728],zmm3
vmovdqu64 [rdx+1792],zmm4
vmovdqu64 [rdx+1856],zmm5
vmovdqu64 [rdx+1920],zmm6
vmovdqu64 [rdx+1984],zmm7
vmovdqu64 [rdx+2048],zmm8
vmovdqu64 [rdx+2112],zmm9
vmovdqu64 [rdx+2176],zmm10
vmovdqu64 [rdx+2240],zmm11
vmovdqu64 [rdx+2304],zmm12
vmovdqu64 [rdx+2368],zmm13
vmovdqu64 [rdx+2432],zmm14
vmovdqu64 [rdx+2496],zmm15
vmovdqu64 [rdx+2560],zmm16
vmovdqu64 [rdx+2624],zmm17
vmovdqu64 [rdx+2688],zmm18
vmovdqu64 [rdx+2752],zmm19
vmovdqu64 [rdx+2816],zmm20
vmovdqu64 [rdx+2880],zmm21
vmovdqu64 [rdx+2944],zmm22
vmovdqu64 [rdx+3008],zmm23
vmovdqu64 [rdx+3072],zmm24
vmovdqu64 [rdx+3136],zmm25
vmovdqu64 [rdx+3200],zmm26
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+13056]
mov ecx,32
.p2align 5
.Lmerged_2_1_2:
vmovdqu64 zmm27, [r8+384]
vmovdqu64 zmm28, [r8+448]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+512]
vmovdqu64 zmm30, [r8+64]
vpxorq zmm30, zmm30, [r8+576]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+128]
vpxorq zmm27, zmm27, [r8+640]
vmovdqu64 zmm28, [r8+192]
vpxorq zmm28, zmm28, [r8+704]
vmovdqu64 zmm29, [r8+256]
vpxorq zmm29, zmm29, [r8+768]
vmovdqu64 zmm30, [r8+320]
vpxorq zmm30, zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+128]
vpxorq zmm27, zmm27, [r8+384]
vpxorq zmm27, zmm27, [r8+640]
vmovdqu64 zmm28, [r8+192]
vpxorq zmm28, zmm28, [r8+448]
vpxorq zmm28, zmm28, [r8+704]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+256]
vpxorq zmm29, zmm29, [r8+512]
vpxorq zmm29, zmm29, [r8+768]
vmovdqu64 zmm30, [r8+64]
vpxorq zmm30, zmm30, [r8+320]
vpxorq zmm30, zmm30, [r8+576]
vpxorq zmm30, zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_2_1_2
vmovdqu64 [rdx+3264],zmm0
vmovdqu64 [rdx+3328],zmm1
vmovdqu64 [rdx+3392],zmm2
vmovdqu64 [rdx+3456],zmm3
vmovdqu64 [rdx+3520],zmm4
vmovdqu64 [rdx+3584],zmm5
vmovdqu64 [rdx+3648],zmm6
vmovdqu64 [rdx+3712],zmm7
vmovdqu64 [rdx+3776],zmm8
vmovdqu64 [rdx+3840],zmm9
vmovdqu64 [rdx+3904],zmm10
vmovdqu64 [rdx+3968],zmm11
vmovdqu64 [rdx+4032],zmm12
vmovdqu64 [rdx+4096],zmm13
vmovdqu64 [rdx+4160],zmm14
vmovdqu64 [rdx+4224],zmm15
vmovdqu64 [rdx+4288],zmm16
vmovdqu64 [rdx+4352],zmm17
vmovdqu64 [rdx+4416],zmm18
vmovdqu64 [rdx+4480],zmm19
vmovdqu64 [rdx+4544],zmm20
vmovdqu64 [rdx+4608],zmm21
vmovdqu64 [rdx+4672],zmm22
vmovdqu64 [rdx+4736],zmm23
vmovdqu64 [rdx+4800],zmm24
vmovdqu64 [rdx+4864],zmm25
vmovdqu64 [rdx+4928],zmm26
ret
.size flock_ab_merged_bank2_m1,.-flock_ab_merged_bank2_m1
.p2align 5
.globl flock_ab_merged_bank2_m2
.type flock_ab_merged_bank2_m2,@function
flock_ab_merged_bank2_m2:
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
mov r8,rdi
lea r9,[rsi+0]
mov ecx,32
.p2align 5
.Lmerged_2_2_0:
vmovdqu64 zmm27, [r8+0]
vmovdqu64 zmm28, [r8+64]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+64]
prefetcht1 [r8 + 0]
prefetcht1 [r8 + 64]
prefetcht1 [r8 + 128]
prefetcht1 [r8 + 192]
prefetcht1 [r8 + 256]
prefetcht1 [r8 + 320]
prefetcht1 [r8 + 384]
prefetcht1 [r8 + 448]
prefetcht1 [r8 + 512]
prefetcht1 [r8 + 576]
prefetcht1 [r8 + 640]
prefetcht1 [r8 + 704]
prefetcht1 [r8 + 768]
prefetcht1 [r8 + 832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vmovdqu64 zmm27, [r8+128]
vmovdqu64 zmm28, [r8+192]
vmovdqu64 zmm29, [r8+256]
vmovdqu64 zmm30, [r8+320]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vmovdqu64 zmm27, [r8+128]
vmovdqu64 zmm28, [r8+192]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+256]
vmovdqu64 zmm30, [r8+64]
vpxorq zmm30, zmm30, [r8+320]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
add r8,118784
add r9,192
dec ecx
jnz .Lmerged_2_2_0
vmovdqu64 [rdx+0],zmm0
vmovdqu64 [rdx+64],zmm1
vmovdqu64 [rdx+128],zmm2
vmovdqu64 [rdx+192],zmm3
vmovdqu64 [rdx+256],zmm4
vmovdqu64 [rdx+320],zmm5
vmovdqu64 [rdx+384],zmm6
vmovdqu64 [rdx+448],zmm7
vmovdqu64 [rdx+512],zmm8
vmovdqu64 [rdx+576],zmm9
vmovdqu64 [rdx+640],zmm10
vmovdqu64 [rdx+704],zmm11
vmovdqu64 [rdx+768],zmm12
vmovdqu64 [rdx+832],zmm13
vmovdqu64 [rdx+896],zmm14
vmovdqu64 [rdx+960],zmm15
vmovdqu64 [rdx+1024],zmm16
vmovdqu64 [rdx+1088],zmm17
vmovdqu64 [rdx+1152],zmm18
vmovdqu64 [rdx+1216],zmm19
vmovdqu64 [rdx+1280],zmm20
vmovdqu64 [rdx+1344],zmm21
vmovdqu64 [rdx+1408],zmm22
vmovdqu64 [rdx+1472],zmm23
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+6144]
mov ecx,32
.p2align 5
.Lmerged_2_2_1:
vmovdqu64 zmm27, [r8+384]
vmovdqu64 zmm28, [r8+448]
vmovdqu64 zmm29, [r8+512]
vmovdqu64 zmm30, [r8+576]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+640]
vmovdqu64 zmm28, [r8+704]
vmovdqu64 zmm29, [r8+768]
vmovdqu64 zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+384]
vpxorq zmm27, zmm27, [r8+640]
vmovdqu64 zmm28, [r8+448]
vpxorq zmm28, zmm28, [r8+704]
vmovdqu64 zmm29, [r8+512]
vpxorq zmm29, zmm29, [r8+768]
vmovdqu64 zmm30, [r8+576]
vpxorq zmm30, zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_2_2_1
vmovdqu64 [rdx+1536],zmm0
vmovdqu64 [rdx+1600],zmm1
vmovdqu64 [rdx+1664],zmm2
vmovdqu64 [rdx+1728],zmm3
vmovdqu64 [rdx+1792],zmm4
vmovdqu64 [rdx+1856],zmm5
vmovdqu64 [rdx+1920],zmm6
vmovdqu64 [rdx+1984],zmm7
vmovdqu64 [rdx+2048],zmm8
vmovdqu64 [rdx+2112],zmm9
vmovdqu64 [rdx+2176],zmm10
vmovdqu64 [rdx+2240],zmm11
vmovdqu64 [rdx+2304],zmm12
vmovdqu64 [rdx+2368],zmm13
vmovdqu64 [rdx+2432],zmm14
vmovdqu64 [rdx+2496],zmm15
vmovdqu64 [rdx+2560],zmm16
vmovdqu64 [rdx+2624],zmm17
vmovdqu64 [rdx+2688],zmm18
vmovdqu64 [rdx+2752],zmm19
vmovdqu64 [rdx+2816],zmm20
vmovdqu64 [rdx+2880],zmm21
vmovdqu64 [rdx+2944],zmm22
vmovdqu64 [rdx+3008],zmm23
vmovdqu64 [rdx+3072],zmm24
vmovdqu64 [rdx+3136],zmm25
vmovdqu64 [rdx+3200],zmm26
vpxorq zmm0, zmm0, zmm0
vpxorq zmm1, zmm1, zmm1
vpxorq zmm2, zmm2, zmm2
vpxorq zmm3, zmm3, zmm3
vpxorq zmm4, zmm4, zmm4
vpxorq zmm5, zmm5, zmm5
vpxorq zmm6, zmm6, zmm6
vpxorq zmm7, zmm7, zmm7
vpxorq zmm8, zmm8, zmm8
vpxorq zmm9, zmm9, zmm9
vpxorq zmm10, zmm10, zmm10
vpxorq zmm11, zmm11, zmm11
vpxorq zmm12, zmm12, zmm12
vpxorq zmm13, zmm13, zmm13
vpxorq zmm14, zmm14, zmm14
vpxorq zmm15, zmm15, zmm15
vpxorq zmm16, zmm16, zmm16
vpxorq zmm17, zmm17, zmm17
vpxorq zmm18, zmm18, zmm18
vpxorq zmm19, zmm19, zmm19
vpxorq zmm20, zmm20, zmm20
vpxorq zmm21, zmm21, zmm21
vpxorq zmm22, zmm22, zmm22
vpxorq zmm23, zmm23, zmm23
vpxorq zmm24, zmm24, zmm24
vpxorq zmm25, zmm25, zmm25
vpxorq zmm26, zmm26, zmm26
mov r8,rdi
lea r9,[rsi+13056]
mov ecx,32
.p2align 5
.Lmerged_2_2_2:
vmovdqu64 zmm27, [r8+384]
vmovdqu64 zmm28, [r8+448]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+512]
vmovdqu64 zmm30, [r8+64]
vpxorq zmm30, zmm30, [r8+576]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+0]{1to8},0
vpxorq zmm0,zmm0,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+8]{1to8},0
vpxorq zmm1,zmm1,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+16]{1to8},0
vpxorq zmm2,zmm2,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+24]{1to8},0
vpxorq zmm3,zmm3,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+32]{1to8},0
vpxorq zmm4,zmm4,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+40]{1to8},0
vpxorq zmm5,zmm5,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+48]{1to8},0
vpxorq zmm6,zmm6,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+56]{1to8},0
vpxorq zmm7,zmm7,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+64]{1to8},0
vpxorq zmm8,zmm8,zmm31
vmovdqu64 zmm27, [r8+128]
vpxorq zmm27, zmm27, [r8+640]
vmovdqu64 zmm28, [r8+192]
vpxorq zmm28, zmm28, [r8+704]
vmovdqu64 zmm29, [r8+256]
vpxorq zmm29, zmm29, [r8+768]
vmovdqu64 zmm30, [r8+320]
vpxorq zmm30, zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+72]{1to8},0
vpxorq zmm9,zmm9,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+80]{1to8},0
vpxorq zmm10,zmm10,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+88]{1to8},0
vpxorq zmm11,zmm11,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+96]{1to8},0
vpxorq zmm12,zmm12,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+104]{1to8},0
vpxorq zmm13,zmm13,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+112]{1to8},0
vpxorq zmm14,zmm14,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+120]{1to8},0
vpxorq zmm15,zmm15,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+128]{1to8},0
vpxorq zmm16,zmm16,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+136]{1to8},0
vpxorq zmm17,zmm17,zmm31
vmovdqu64 zmm27, [r8+128]
vpxorq zmm27, zmm27, [r8+384]
vpxorq zmm27, zmm27, [r8+640]
vmovdqu64 zmm28, [r8+192]
vpxorq zmm28, zmm28, [r8+448]
vpxorq zmm28, zmm28, [r8+704]
vmovdqu64 zmm29, [r8+0]
vpxorq zmm29, zmm29, [r8+256]
vpxorq zmm29, zmm29, [r8+512]
vpxorq zmm29, zmm29, [r8+768]
vmovdqu64 zmm30, [r8+64]
vpxorq zmm30, zmm30, [r8+320]
vpxorq zmm30, zmm30, [r8+576]
vpxorq zmm30, zmm30, [r8+832]
vgf2p8affineqb zmm31,zmm27,QWORD PTR [r9+144]{1to8},0
vpxorq zmm18,zmm18,zmm31
vgf2p8affineqb zmm31,zmm28,QWORD PTR [r9+152]{1to8},0
vpxorq zmm19,zmm19,zmm31
vpxorq zmm31,zmm27,zmm28
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+160]{1to8},0
vpxorq zmm20,zmm20,zmm31
vgf2p8affineqb zmm31,zmm29,QWORD PTR [r9+168]{1to8},0
vpxorq zmm21,zmm21,zmm31
vgf2p8affineqb zmm31,zmm30,QWORD PTR [r9+176]{1to8},0
vpxorq zmm22,zmm22,zmm31
vpxorq zmm31,zmm29,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+184]{1to8},0
vpxorq zmm23,zmm23,zmm31
vpxorq zmm31,zmm27,zmm29
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+192]{1to8},0
vpxorq zmm24,zmm24,zmm31
vpxorq zmm31,zmm28,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+200]{1to8},0
vpxorq zmm25,zmm25,zmm31
vpxorq zmm31,zmm27,zmm28
vpxorq zmm31,zmm31,zmm29
vpxorq zmm31,zmm31,zmm30
vgf2p8affineqb zmm31,zmm31,QWORD PTR [r9+208]{1to8},0
vpxorq zmm26,zmm26,zmm31
add r8,118784
add r9,216
dec ecx
jnz .Lmerged_2_2_2
vmovdqu64 [rdx+3264],zmm0
vmovdqu64 [rdx+3328],zmm1
vmovdqu64 [rdx+3392],zmm2
vmovdqu64 [rdx+3456],zmm3
vmovdqu64 [rdx+3520],zmm4
vmovdqu64 [rdx+3584],zmm5
vmovdqu64 [rdx+3648],zmm6
vmovdqu64 [rdx+3712],zmm7
vmovdqu64 [rdx+3776],zmm8
vmovdqu64 [rdx+3840],zmm9
vmovdqu64 [rdx+3904],zmm10
vmovdqu64 [rdx+3968],zmm11
vmovdqu64 [rdx+4032],zmm12
vmovdqu64 [rdx+4096],zmm13
vmovdqu64 [rdx+4160],zmm14
vmovdqu64 [rdx+4224],zmm15
vmovdqu64 [rdx+4288],zmm16
vmovdqu64 [rdx+4352],zmm17
vmovdqu64 [rdx+4416],zmm18
vmovdqu64 [rdx+4480],zmm19
vmovdqu64 [rdx+4544],zmm20
vmovdqu64 [rdx+4608],zmm21
vmovdqu64 [rdx+4672],zmm22
vmovdqu64 [rdx+4736],zmm23
vmovdqu64 [rdx+4800],zmm24
vmovdqu64 [rdx+4864],zmm25
vmovdqu64 [rdx+4928],zmm26
ret
.size flock_ab_merged_bank2_m2,.-flock_ab_merged_bank2_m2
