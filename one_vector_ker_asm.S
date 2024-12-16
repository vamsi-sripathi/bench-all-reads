#define NAME        one_vector_ker_asm
#define SIZE        8
#define ALIGN_4 .p2align 4

#define N       %rdi
#define X       %rsi

#define BASE_DISP
#undef BASE_IND_X_SCALE_DISP

  .text
  .align 256
  .globl NAME
  .type NAME, @function
NAME:
  mov (N), N
  sar $5,  N

#ifdef BASE_IND_X_SCALE_DISP
  xor %rdx, %rdx
#endif
#ifdef BASE_IND_X_SCALE
  mov  $0,  %rdx
  mov  $8,  %rcx
  mov  $16, %r8
  mov  $24, %r9
#endif
#ifdef BASE_IND
  mov  $0,  %rdx
  mov  $8  * SIZE,  %rcx
  mov  $16 * SIZE, %r8
  mov  $24 * SIZE, %r9
#endif

  vxorpd %zmm0, %zmm0, %zmm0
  vxorpd %zmm1, %zmm1, %zmm1
  vxorpd %zmm2, %zmm2, %zmm2
  vxorpd %zmm3, %zmm3, %zmm3

  ALIGN_4
// main loop
.L1:

#ifdef SPLIT_LOAD_ADD
  vmovupd  (X), %zmm4
  vaddpd   %zmm4, %zmm0, %zmm0

  vmovupd  8 * SIZE(X), %zmm5
  vaddpd  %zmm5, %zmm1, %zmm1

  vmovupd 16 * SIZE(X), %zmm6
  vaddpd  %zmm6, %zmm2, %zmm2

  vmovupd 24 * SIZE(X), %zmm7
  vaddpd  %zmm7, %zmm3, %zmm3

  add $32 * SIZE, X
#endif

#ifdef BASE_DISP
  vaddpd           (X), %zmm0, %zmm0
  vaddpd  8  * SIZE(X), %zmm1, %zmm1
  vaddpd  16 * SIZE(X), %zmm2, %zmm2
  vaddpd  24 * SIZE(X), %zmm3, %zmm3

  add $32 * SIZE, X
#endif

#ifdef BASE_IND_X_SCALE_DISP
  vaddpd             (X, %rdx, 8), %zmm0, %zmm0
  vaddpd    8  * SIZE(X, %rdx, 8), %zmm1, %zmm1
  vaddpd    16 * SIZE(X, %rdx, 8), %zmm2, %zmm2
  vaddpd    24 * SIZE(X, %rdx, 8), %zmm3, %zmm3
  
  add $32, %rdx
#endif

#ifdef BASE_IND_X_SCALE
  vaddpd    (X, %rdx, 8), %zmm0, %zmm0
  vaddpd    (X, %rcx, 8), %zmm1, %zmm1
  vaddpd    (X, %r8,  8), %zmm2, %zmm2
  vaddpd    (X, %r9,  8), %zmm3, %zmm3
  
  add $32 * SIZE, X
#endif

#ifdef BASE_IND
  vaddpd    (X, %rdx), %zmm0, %zmm0
  vaddpd    (X, %rcx), %zmm1, %zmm1
  vaddpd    (X, %r8), %zmm2, %zmm2
  vaddpd    (X, %r9), %zmm3, %zmm3
  
  add $32 * SIZE, X
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
  

  vzeroupper

  ret
  .size  NAME, .-NAME
