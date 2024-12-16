#define NAME        two_vector_ker_asm
#define SIZE        8
#define ALIGN_4 .p2align 4

#define N       %rdi
#define X       %rsi
#define Y       %rdx

#define BASE_DISP
#undef BASE_IND_X_SCALE_DISP

  .text
  .align 256
  .globl NAME
  .type NAME, @function
NAME:
  mov (N), N
  sar $5,  N

  vxorpd %zmm0, %zmm0, %zmm0
  vxorpd %zmm1, %zmm1, %zmm1
  vxorpd %zmm2, %zmm2, %zmm2
  vxorpd %zmm3, %zmm3, %zmm3

  vxorpd %zmm4, %zmm4, %zmm4
  vxorpd %zmm5, %zmm5, %zmm5
  vxorpd %zmm6, %zmm6, %zmm6
  vxorpd %zmm7, %zmm7, %zmm7

  ALIGN_4
// main loop
.L1:

#ifdef BASE_DISP
  vaddpd           (X), %zmm0, %zmm0
  vaddpd  8  * SIZE(X), %zmm1, %zmm1
  vaddpd  16 * SIZE(X), %zmm2, %zmm2
  vaddpd  24 * SIZE(X), %zmm3, %zmm3

  vaddpd           (Y), %zmm4, %zmm4
  vaddpd  8  * SIZE(Y), %zmm5, %zmm5
  vaddpd  16 * SIZE(Y), %zmm6, %zmm6
  vaddpd  24 * SIZE(Y), %zmm7, %zmm7

  add $32 * SIZE, X
  add $32 * SIZE, Y
#endif

#ifdef BASE_IND_X_SCALE_DISP
  vaddpd             (X, %rdx, 8), %zmm0, %zmm0
  vaddpd    8  * SIZE(X, %rdx, 8), %zmm1, %zmm1
  vaddpd    16 * SIZE(X, %rdx, 8), %zmm2, %zmm2
  vaddpd    24 * SIZE(X, %rdx, 8), %zmm3, %zmm3
  
  add $32, %rdx
#endif

  dec N
  jg  .L1
  ALIGN_4

.L2:
  vaddpd %zmm0, %zmm1, %zmm0
  vaddpd %zmm2, %zmm3, %zmm2
  vaddpd %zmm0, %zmm2, %zmm0

  vextractf64x4 $0x1, %zmm0, %ymm1
  vaddpd %ymm0, %ymm1, %ymm0

  vpermpd $0x0E, %ymm0, %ymm1
  vaddpd %ymm0, %ymm1, %ymm0

  vpermilpd $0x1, %ymm0, %ymm1
  vaddpd %xmm0, %xmm1, %xmm0
  
  vaddpd %zmm4, %zmm5, %zmm4
  vaddpd %zmm6, %zmm7, %zmm6
  vaddpd %zmm4, %zmm6, %zmm4

  vextractf64x4 $0x1, %zmm4, %ymm5
  vaddpd %ymm4, %ymm5, %ymm4

  vpermpd $0x0E, %ymm4, %ymm5
  vaddpd %ymm4, %ymm5, %ymm4

  vpermilpd $0x1, %ymm4, %ymm5
  vaddpd %xmm4, %xmm5, %xmm4

  vaddpd %xmm0, %xmm4, %xmm0
 
  vzeroupper

  ret
  .size  NAME, .-NAME
