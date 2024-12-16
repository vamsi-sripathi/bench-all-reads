#define NAME        four_vector_ker_asm
#define SIZE        8
#define ALIGN_4 .p2align 4

#define N       %rdi
#define X0      %rsi
#define X1      %rdx
#define X2      %rcx
#define X3      %r8

  .text
  .align 256
  .globl NAME
  .type NAME, @function
NAME:
  mov (N), N

  vxorpd %zmm31, %zmm31, %zmm31
  vxorpd %zmm30, %zmm30, %zmm30
  vxorpd %zmm29, %zmm29, %zmm29

  ALIGN_4
// main loop
.L1:
  vmovupd    (X0),  %zmm10
  vmovupd    (X1),  %zmm11

  vmovddup   %zmm10, %zmm1
  vmulpd     %zmm1, %zmm11, %zmm2
  vpermilpd  $0x55, %zmm11, %zmm11
  vshufpd    $0xFF, %zmm10, %zmm10, %zmm10
  vfmaddsub213pd    %zmm2, %zmm10, %zmm11

  vmovupd 8 * SIZE(X0),  %zmm12
  vmovupd 8 * SIZE(X1),  %zmm13

  vmovddup   %zmm12, %zmm4
  vmulpd     %zmm4, %zmm13, %zmm5
  vpermilpd  $0x55,  %zmm13, %zmm13
  vshufpd    $0xFF,  %zmm12, %zmm12, %zmm12
  vfmaddsub213pd    %zmm5, %zmm12, %zmm13

  vmovupd 16 * SIZE(X0),  %zmm14
  vmovupd 16 * SIZE(X1),  %zmm15

  vmovddup   %zmm14, %zmm7
  vmulpd     %zmm7, %zmm15, %zmm8
  vpermilpd  $0x55,  %zmm15, %zmm15
  vshufpd    $0xFF,  %zmm14, %zmm14, %zmm14
  vfmaddsub213pd    %zmm8, %zmm14, %zmm15

  vaddpd  %zmm11, %zmm13, %zmm11
  vaddpd  %zmm11, %zmm15, %zmm11
  vaddpd  %zmm31,  %zmm11, %zmm31

  vmovupd 24 * SIZE(X1),  %zmm16
  vmulpd     %zmm1, %zmm16, %zmm2
  vpermilpd  $0x55,  %zmm16, %zmm16
  vfmaddsub213pd    %zmm2, %zmm10, %zmm16

  vmovupd 32 * SIZE(X1),  %zmm17
  vmulpd     %zmm4, %zmm17, %zmm5
  vpermilpd  $0x55,  %zmm17, %zmm17
  vfmaddsub213pd    %zmm5, %zmm12, %zmm17

  vmovupd 40 * SIZE(X1),  %zmm18
  vmulpd     %zmm7, %zmm18, %zmm8
  vpermilpd  $0x55,  %zmm18, %zmm18
  vfmaddsub213pd    %zmm8, %zmm14, %zmm18

  vaddpd  %zmm16, %zmm17, %zmm16
  vaddpd  %zmm16, %zmm18, %zmm16
  vaddpd  %zmm30, %zmm16, %zmm30

  vmovupd 48 * SIZE(X1),  %zmm19
  vmulpd     %zmm1, %zmm19, %zmm2
  vpermilpd  $0x55,  %zmm19, %zmm19
  vfmaddsub213pd    %zmm2, %zmm10, %zmm19

  vmovupd 56 * SIZE(X1),  %zmm20
  vmulpd     %zmm4, %zmm20, %zmm5
  vpermilpd  $0x55,  %zmm20, %zmm20
  vfmaddsub213pd    %zmm5, %zmm12, %zmm20

  vmovupd 64 * SIZE(X1),  %zmm21
  vmulpd     %zmm7, %zmm21, %zmm8
  vpermilpd  $0x55,  %zmm21, %zmm21
  vfmaddsub213pd    %zmm8, %zmm14, %zmm21

  vaddpd  %zmm19, %zmm20, %zmm19
  vaddpd  %zmm19, %zmm21, %zmm19
  vaddpd  %zmm29, %zmm19, %zmm29

  vmovupd    (X2),  %zmm22
  vmovupd    (X3),  %zmm23

  vmovddup   %zmm22, %zmm1
  vmulpd     %zmm1, %zmm23, %zmm2
  vpermilpd  $0x55,  %zmm23, %zmm23
  vshufpd    $0xFF,  %zmm22, %zmm22, %zmm22
  vfmaddsub213pd    %zmm2, %zmm22, %zmm23

  vmovupd 8 * SIZE(X2),  %zmm24
  vmovupd 8 * SIZE(X3),  %zmm25

  vmovddup   %zmm24, %zmm4
  vmulpd     %zmm4, %zmm25, %zmm5
  vpermilpd  $0x55,  %zmm25, %zmm25
  vshufpd    $0xFF,  %zmm24, %zmm24, %zmm24
  vfmaddsub213pd    %zmm5, %zmm24, %zmm25

  vmovupd 16 * SIZE(X2),  %zmm26
  vmovupd 16 * SIZE(X3),  %zmm27

  vmovddup   %zmm26, %zmm7
  vmulpd     %zmm7, %zmm27, %zmm8
  vpermilpd  $0x55,  %zmm27, %zmm27
  vshufpd    $0xFF,  %zmm26, %zmm26, %zmm26
  vfmaddsub213pd    %zmm8, %zmm26, %zmm27

  vaddpd  %zmm23, %zmm25, %zmm23
  vaddpd  %zmm23, %zmm27, %zmm23
  vaddpd  %zmm31, %zmm23, %zmm31

  vmovupd 24 * SIZE(X3),  %zmm11
  vmulpd     %zmm1, %zmm11, %zmm2
  vpermilpd  $0x55,  %zmm11, %zmm11
  vfmaddsub213pd    %zmm2, %zmm10, %zmm11

  vmovupd 32 * SIZE(X3),  %zmm13
  vmulpd     %zmm4, %zmm13, %zmm5
  vpermilpd  $0x55,  %zmm13, %zmm13
  vfmaddsub213pd    %zmm5, %zmm12, %zmm13

  vmovupd 40 * SIZE(X3),  %zmm15
  vmulpd     %zmm7, %zmm15, %zmm8
  vpermilpd  $0x55,  %zmm15, %zmm15
  vfmaddsub213pd    %zmm8, %zmm14, %zmm15

  vaddpd  %zmm11, %zmm13, %zmm11
  vaddpd  %zmm11, %zmm15, %zmm11
  vaddpd  %zmm30, %zmm11, %zmm30

  vmovupd 48 * SIZE(X3),  %zmm16
  vmulpd     %zmm1, %zmm16, %zmm2
  vpermilpd  $0x55,  %zmm16, %zmm16
  vfmaddsub213pd    %zmm2, %zmm10, %zmm16

  vmovupd 56 * SIZE(X3),  %zmm17
  vmulpd     %zmm4, %zmm17, %zmm5
  vpermilpd  $0x55,  %zmm17, %zmm17
  vfmaddsub213pd    %zmm5, %zmm12, %zmm17

  vmovupd 64 * SIZE(X3),  %zmm18
  vmulpd     %zmm7, %zmm18, %zmm8
  vpermilpd  $0x55,  %zmm18, %zmm18
  vfmaddsub213pd    %zmm8, %zmm14, %zmm18

  vaddpd  %zmm16, %zmm17, %zmm16
  vaddpd  %zmm16, %zmm18, %zmm16
  vaddpd  %zmm29, %zmm16, %zmm29

  add $24 * SIZE, X0
  add $72 * SIZE, X1
  add $24 * SIZE, X2
  add $72 * SIZE, X3

  dec N
  jg  .L1
  ALIGN_4

.L2:
  vaddpd %zmm31, %zmm30, %zmm31
  vaddpd %zmm31, %zmm29, %zmm0

  vextractf64x4 $0x1, %zmm0, %ymm1
  vaddpd %ymm0, %ymm1, %ymm0

  vpermpd $0x0E, %ymm0, %ymm1
  vaddpd %ymm0, %ymm1, %ymm0

  vpermilpd $0x1, %ymm0, %ymm1
  vaddpd %xmm0, %xmm1, %xmm0

  vzeroupper

  ret
  .size  NAME, .-NAME
