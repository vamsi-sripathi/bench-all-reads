#include "stdio.h"
#include "immintrin.h"

#define L2_PF_DIST  (4096)
#define L1_PF_DIST  (512)
/* #define L1_PF_DIST  (1024) */

#define ZMM_MUL(zmm0, zmm1, zmm_t1, zmm_t2) \
  do { \
     zmm_t1  = _mm512_mul_pd(_mm512_movedup_pd(zmm0), zmm1); \
     zmm_t2  = _mm512_permute_pd(zmm1, 0x55); \
     zmm1    = _mm512_fmaddsub_pd(_mm512_shuffle_pd(zmm0, zmm0, 0xFF), zmm_t2, zmm_t1); \
  } while(0);

#define __ZMM_MUL(zmm0, zmm1, zmm_t1, zmm_t2) \
  do { \
     zmm_t1  = _mm512_mul_pd(_mm512_movedup_pd(zmm0), zmm1); \
     zmm_t2  = _mm512_permute_pd(zmm1, 0x55); \
     zmm1  = _mm512_fmaddsub_pd(_mm512_shuffle_pd(zmm1, zmm1, 0xFF), zmm_t2, zmm_t1); \
  } while(0);
   
double one_vector_ker_asm(long long *, double *);
double two_vector_ker_asm(long long *, double *, double *);
double four_vector_ker_asm(long long *, double *, double *, double *, double *);

double one_vector(long long n, long long start, double **p_vectors)
{
  double *p_x = p_vectors[0] + start;
  double tmp = 0.0;

#if defined (USE_ASM)
  int offset;
  long long n_blk;

  offset = ((unsigned long)p_x) % 64;
  n_blk  = (n/32)*32;

  if ((n_blk > 32) && (offset == 0)) {
    tmp += one_vector_ker_asm(&n, p_x);

    int tail = n - n_blk;
    if (tail) {
      p_x += n_blk;
      while (tail) {
        tmp += *p_x++;
        tail--;
      }
    }
  } else {
    printf ("one_vector: falling back to cgc code-path..\n");

    for (long long i=0; i<n; i++) {
      tmp += p_x[i];
    }
  }

  return tmp;

#elif defined (USE_INTRINSICS)

#define UNROLL   (32)
#define PCHASE_4ACC
#define PREFETCH

  __m512d zmm0, zmm1, zmm2, zmm3, zmm_acc;
  __m512d zmm_a0, zmm_a1, zmm_a2, zmm_a3;

  long long i = 0;

  int offset = ((unsigned long)p_x) % 64;
  if (offset) {
    printf ("one_vector: walking to 64B boundary..\n");
    fflush(0);
    int prolog_eles = 8 - (offset/8);
    n -= prolog_eles;
    while (prolog_eles) {
      tmp += *p_x++;
      prolog_eles--;
    }
  }

  long long n_blk = (n/UNROLL)*UNROLL;
  int tail = n - n_blk;

  zmm_acc = _mm512_setzero_pd();
  zmm_a0  = _mm512_setzero_pd();
  zmm_a1  = _mm512_setzero_pd();
  zmm_a2  = _mm512_setzero_pd();
  zmm_a3  = _mm512_setzero_pd();

  while (n_blk) {
#if UNROLL==8
    zmm0    = _mm512_loadu_pd(p_x);
    zmm_acc = _mm512_add_pd(zmm0, zmm_acc);

#ifdef PREFETCH
    _mm_prefetch((char*)(p_x+L2_PF_DIST), _MM_HINT_T1);
    _mm_prefetch((char*)(p_x+L1_PF_DIST), _MM_HINT_T0);
#endif

#elif UNROLL==16
    zmm0    = _mm512_loadu_pd(p_x);
    zmm1    = _mm512_loadu_pd(p_x+8);
    zmm0    = _mm512_add_pd(zmm0, zmm1);
    zmm_acc = _mm512_add_pd(zmm0, zmm_acc);

#ifdef PREFETCH
    _mm_prefetch((char*)(p_x+L2_PF_DIST),   _MM_HINT_T1);
    _mm_prefetch((char*)(p_x+L2_PF_DIST+8), _MM_HINT_T1);

    _mm_prefetch((char*)(p_x+L1_PF_DIST),   _MM_HINT_T0);
    _mm_prefetch((char*)(p_x+L1_PF_DIST+8), _MM_HINT_T0);
#endif

#elif UNROLL==24
    zmm0 = _mm512_loadu_pd(p_x);
    zmm1 = _mm512_loadu_pd(p_x+8);
    zmm2 = _mm512_loadu_pd(p_x+16);
    zmm0 = _mm512_add_pd(zmm0, zmm1);

    zmm_acc = _mm512_add_pd(zmm2, zmm_acc);
    zmm_acc = _mm512_add_pd(zmm0, zmm_acc);

#ifdef PREFETCH
    _mm_prefetch((char*)(p_x+L2_PF_DIST),    _MM_HINT_T1);
    _mm_prefetch((char*)(p_x+L2_PF_DIST+8),  _MM_HINT_T1);
    _mm_prefetch((char*)(p_x+L2_PF_DIST+16), _MM_HINT_T1);

    _mm_prefetch((char*)(p_x+L1_PF_DIST),    _MM_HINT_T0);
    _mm_prefetch((char*)(p_x+L1_PF_DIST+8),  _MM_HINT_T0);
    _mm_prefetch((char*)(p_x+L1_PF_DIST+16), _MM_HINT_T0);
#endif

#elif UNROLL==32

#if defined (PCHASE_1ACC)
    // pointer chasing w/ 1 accumulator
    zmm0 = _mm512_loadu_pd(p_x);
    zmm1 = _mm512_loadu_pd(p_x+8);
    zmm2 = _mm512_loadu_pd(p_x+16);
    zmm3 = _mm512_loadu_pd(p_x+24);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm2 = _mm512_add_pd(zmm2, zmm3);

    zmm0    = _mm512_add_pd(zmm0, zmm2);
    zmm_acc = _mm512_add_pd(zmm0, zmm_acc);
#elif defined (PCHASE_4ACC)
    // pointer chasing w/ 4 accumulators
    zmm0   = _mm512_loadu_pd(p_x);
    zmm_a0 = _mm512_add_pd(zmm0, zmm_a0);

    zmm1   = _mm512_loadu_pd(p_x+8);
    zmm_a1 = _mm512_add_pd(zmm1, zmm_a1);

    zmm2   = _mm512_loadu_pd(p_x+16);
    zmm_a2 = _mm512_add_pd(zmm2, zmm_a2);

    zmm3   = _mm512_loadu_pd(p_x+24);
    zmm_a3 = _mm512_add_pd(zmm3, zmm_a3);
#else
    // relative addressing w/ 4 accumulators
    zmm0   = _mm512_loadu_pd(&p_x[i]);
    zmm_a0 = _mm512_add_pd(zmm0, zmm_a0);

    zmm1   = _mm512_loadu_pd(&p_x[i+8]);
    zmm_a1 = _mm512_add_pd(zmm1, zmm_a1);

    zmm2   = _mm512_loadu_pd(&p_x[i+16]);
    zmm_a2 = _mm512_add_pd(zmm2, zmm_a2);

    zmm3   = _mm512_loadu_pd(&p_x[i+24]);
    zmm_a3 = _mm512_add_pd(zmm3, zmm_a3);
#endif

#ifdef PREFETCH
#if defined (PCHASE_1ACC) || defined (PCHASE_4ACC)
    _mm_prefetch((char*)(p_x+L2_PF_DIST),    _MM_HINT_T1);
    _mm_prefetch((char*)(p_x+L2_PF_DIST+8),  _MM_HINT_T1);
    _mm_prefetch((char*)(p_x+L2_PF_DIST+16), _MM_HINT_T1);
    _mm_prefetch((char*)(p_x+L2_PF_DIST+24), _MM_HINT_T1);

    _mm_prefetch((char*)(p_x+L1_PF_DIST),    _MM_HINT_T0);
    _mm_prefetch((char*)(p_x+L1_PF_DIST+8),  _MM_HINT_T0);
    _mm_prefetch((char*)(p_x+L1_PF_DIST+16), _MM_HINT_T0);
    _mm_prefetch((char*)(p_x+L1_PF_DIST+24), _MM_HINT_T0);
#else
    _mm_prefetch((char*)(&p_x[i+L2_PF_DIST]),    _MM_HINT_T1);
    _mm_prefetch((char*)(&p_x[i+L2_PF_DIST+8]),  _MM_HINT_T1);
    _mm_prefetch((char*)(&p_x[i+L2_PF_DIST+16]), _MM_HINT_T1);
    _mm_prefetch((char*)(&p_x[i+L2_PF_DIST+24]), _MM_HINT_T1);

    _mm_prefetch((char*)(&p_x[i+L1_PF_DIST]),    _MM_HINT_T0);
    _mm_prefetch((char*)(&p_x[i+L1_PF_DIST+8]),  _MM_HINT_T0);
    _mm_prefetch((char*)(&p_x[i+L1_PF_DIST+16]), _MM_HINT_T0);
    _mm_prefetch((char*)(&p_x[i+L1_PF_DIST+24]), _MM_HINT_T0);
#endif
#endif

#if defined (PCHASE_1ACC) || defined (PCHASE_4ACC)
    p_x   += UNROLL;
#else
    i     += UNROLL;
#endif

#else
#error "unsupported unrolling factor"
#endif

#if UNROLL!=32
    p_x   += UNROLL;
#endif

    n_blk -= UNROLL;
  }

#if UNROLL==32
#if !defined (PCHASE_1ACC)
  // only when using 4 accumulators in main-loop
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a1);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm_a3);
    zmm_acc = _mm512_add_pd(zmm_a0, zmm_a2);
#endif

#if !defined (PCHASE_1ACC) && !defined (PCHASE_4ACC)
    // for tail
    p_x = p_x + i;
#endif
#endif

  tmp += _mm512_reduce_add_pd(zmm_acc);

  while (tail) {
    tmp += *p_x++;
    tail--;
  }
  return tmp;

#undef UNROLL
#undef PREFETCH

#else
#if 0
  for (long long i=0; i<n; i++) {
    tmp += p_x[i];
  }
#else
  while (n) {
    tmp += *p_x++;
    n--;
  }
#endif
  return tmp;
#endif

}

double two_vector(long long n, long long start, double **p_vectors)
{
  double *p_x0 = p_vectors[0] + start;
  double *p_x1 = p_vectors[1] + start;

#if defined (USE_ASM)
  double tmp = 0.0;

  int offset;
  long long n_blk;

  offset = ((unsigned long)p_x0) % 64 + ((unsigned long)p_x1) % 64;
  n_blk  = (n/32)*32;

  if ((n_blk > 32) && (offset == 0)) {
    tmp += two_vector_ker_asm(&n, p_x0, p_x1);

    int tail = n - n_blk;
    if (tail) {
      p_x0 += n_blk;
      p_x1 += n_blk;
      while (tail) {
        tmp += *p_x0++ + *p_x1++;
        tail--;
      }
    }
  } else {
    printf ("two_vector: falling back to cgc code-path..\n");

    for (long long i=0; i<n; i++) {
      tmp += p_x0[i] + p_x1[i];
    }
  }

  return tmp;

#elif defined (USE_INTRINSICS)
#define  UNROLL   (32)
#define  PREFETCH
  __m512d zmm0, zmm1, zmm2, zmm3;
  __m512d zmm4, zmm5, zmm6, zmm7;
  __m512d zmm_a0, zmm_a1;

  double tmp = 0.0;

  int x_offset = ((unsigned long)p_x0) % 64;
  int y_offset = ((unsigned long)p_x1) % 64;
  
  if (((x_offset == y_offset) && x_offset) ||
     ((x_offset != 0 && y_offset != 0))) {
    int offset = ((x_offset != 0) ? x_offset : y_offset);
    int prolog_eles = 8 - (offset/8);
    n -= prolog_eles;
    while (prolog_eles) {
      tmp += *p_x0++ + *p_x1++;
      prolog_eles--;
    }
  }

  long long n_blk = (n/UNROLL)*UNROLL;
  int tail = n - n_blk;

  zmm_a0 = _mm512_setzero_pd();
  zmm_a1 = _mm512_setzero_pd();

  while (n_blk) {
    zmm0 = _mm512_loadu_pd(p_x0);
    zmm1 = _mm512_loadu_pd(p_x0+8);
    zmm2 = _mm512_loadu_pd(p_x0+16);
    zmm3 = _mm512_loadu_pd(p_x0+24);

    zmm4 = _mm512_loadu_pd(p_x1);
    zmm5 = _mm512_loadu_pd(p_x1+8);
    zmm6 = _mm512_loadu_pd(p_x1+16);
    zmm7 = _mm512_loadu_pd(p_x1+24);

    zmm0  = _mm512_add_pd(zmm0, zmm1);
    zmm2  = _mm512_add_pd(zmm2, zmm3);
    zmm0  = _mm512_add_pd(zmm0, zmm2);
    zmm_a0 = _mm512_add_pd(zmm0, zmm_a0);

    zmm4  = _mm512_add_pd(zmm4, zmm5);
    zmm6  = _mm512_add_pd(zmm6, zmm7);
    zmm4  = _mm512_add_pd(zmm4, zmm6);
    zmm_a1 = _mm512_add_pd(zmm4, zmm_a1);

#ifdef PREFETCH
    _mm_prefetch((char*)(p_x0+L1_PF_DIST),    _MM_HINT_T0);
    _mm_prefetch((char*)(p_x0+L1_PF_DIST+8),  _MM_HINT_T0);
    _mm_prefetch((char*)(p_x0+L1_PF_DIST+16), _MM_HINT_T0);
    _mm_prefetch((char*)(p_x0+L1_PF_DIST+24), _MM_HINT_T0);

    _mm_prefetch((char*)(p_x1+L1_PF_DIST),    _MM_HINT_T0);
    _mm_prefetch((char*)(p_x1+L1_PF_DIST+8),  _MM_HINT_T0);
    _mm_prefetch((char*)(p_x1+L1_PF_DIST+16), _MM_HINT_T0);
    _mm_prefetch((char*)(p_x1+L1_PF_DIST+24), _MM_HINT_T0);
#endif

    p_x0   += UNROLL;
    p_x1   += UNROLL;
    n_blk  -= UNROLL;
  }

  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a1);

  tmp += _mm512_reduce_add_pd(zmm_a0);

  while (tail) {
    tmp += *p_x0++ + *p_x1++;
    tail--;
  }

  return tmp;
#undef UNROLL
#undef PREFETCH

#else
  double tmp0 = 0.0, tmp1 = 0.0;

  for (long long i=0; i<n; i++) {
    tmp0 += p_x0[i];
    tmp1 += p_x1[i];
  }

  return (tmp0 + tmp1);
#endif
}

double three_vector(long long n, long long start, double **p_vectors)
{
  double *p_v0 = p_vectors[0] + start;
  double *p_v1 = p_vectors[1] + start;
  double *p_v2 = p_vectors[2] + start;

  double tmp0 = 0.0;
  double tmp1 = 0.0;
  double tmp2 = 0.0;

  for (long long i=0; i<n; i++) {
    tmp0 += p_v0[i];
    tmp1 += p_v1[i];
    tmp2 += p_v2[i];
  }
  return tmp0 + tmp1 + tmp2;
}

double four_vector(long long n, long long start, double **p_vectors)
{
  double *p_v0 = p_vectors[0] + start;
  double *p_v1 = p_vectors[1] + start;
  double *p_v2 = p_vectors[2] + start;
  double *p_v3 = p_vectors[3] + start;

  double tmp0 = 0.0;
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;

#if defined (USE_ASM)
  double tmp = 0.0;

  int offset;
  long long n_blk;

  offset = ((unsigned long)p_v0) % 64 + ((unsigned long)p_v1) % 64 +
           ((unsigned long)p_v2) % 64 + ((unsigned long)p_v3) % 64;
  n_blk  = (n/72);

  if ((n_blk > 1) && (offset == 0)) {
    tmp += four_vector_ker_asm(&n_blk, p_v0, p_v1, p_v2, p_v3);

//TODO: tail processing
#if 0
    int tail = n - n_blk;
    if (tail) {
      p_v0 += n_blk;
      p_v1 += n_blk;
      p_v2 += n_blk;
      p_v3 += n_blk;
      while (tail) {
        tmp += *p_v0++ + *p_v1++;
        tail--;
      }
    }
#endif

  } else {
    printf ("four_vector: falling back to cgc code-path..\n");

    for (long long i=0; i<n; i++) {
      tmp += p_v0[i] + p_v1[i];
    }
  }

  return tmp;

#elif defined (USE_INTRINSICS)
#define FOUR_VECTOR_V5

#if defined (FOUR_VECTOR_V1)
  __m512d zmm0, zmm1, zmm2, zmm3,
          zmm4, zmm5, zmm6, zmm7,
          zmm8, zmm9, zmm10, zmm11,
          zmm_t1, zmm_t2,
          zmm_a0, zmm_a1, zmm_a2;

  zmm_a0 = _mm512_setzero_pd();
  zmm_a1 = _mm512_setzero_pd();
  zmm_a2 = _mm512_setzero_pd();


  for (long long ii=0, i=0, j=0; ii<(n/72)*72; ii+=72, i+=72, j+=72) {
    // 1.3x3
    zmm0 = _mm512_loadu_pd(&p_v0[i]);
    zmm1 = _mm512_loadu_pd(&p_v0[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v0[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v0[i]+24);
    zmm4 = _mm512_loadu_pd(&p_v0[i]+32);
    zmm5 = _mm512_loadu_pd(&p_v0[i]+40);
    zmm6 = _mm512_loadu_pd(&p_v0[i]+48);
    zmm7 = _mm512_loadu_pd(&p_v0[i]+56);
    zmm8 = _mm512_loadu_pd(&p_v0[i]+64);

    // 1.3x1
    zmm9  = _mm512_loadu_pd(&p_v1[j]);
    zmm10 = _mm512_loadu_pd(&p_v1[j]+8);
    zmm11 = _mm512_loadu_pd(&p_v1[j]+16);

    // 1.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v1[i]+24);
    zmm10 = _mm512_loadu_pd(&p_v1[i]+32);
    zmm11 = _mm512_loadu_pd(&p_v1[i]+40);
#endif
    
    // 1.mac-2
    ZMM_MUL(zmm9,  zmm3, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm4, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm5, zmm_t1, zmm_t2);
    zmm3 = _mm512_add_pd(zmm3, zmm4);
    zmm3 = _mm512_add_pd(zmm3, zmm5);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm3);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v1[i]+48);
    zmm10 = _mm512_loadu_pd(&p_v1[i]+56);
    zmm11 = _mm512_loadu_pd(&p_v1[i]+64);
#endif

    // 1.mac-3
    ZMM_MUL(zmm9,  zmm6, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm7, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm8, zmm_t1, zmm_t2);
    zmm6 = _mm512_add_pd(zmm6, zmm7);
    zmm6 = _mm512_add_pd(zmm6, zmm8);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm6);

    // 2.3x3
    zmm0 = _mm512_loadu_pd(&p_v2[i]);
    zmm1 = _mm512_loadu_pd(&p_v2[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v2[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v2[i]+24);
    zmm4 = _mm512_loadu_pd(&p_v2[i]+32);
    zmm5 = _mm512_loadu_pd(&p_v2[i]+40);
    zmm6 = _mm512_loadu_pd(&p_v2[i]+48);
    zmm7 = _mm512_loadu_pd(&p_v2[i]+56);
    zmm8 = _mm512_loadu_pd(&p_v2[i]+64);

    // 2.3x1
    zmm9  = _mm512_loadu_pd(&p_v3[j]);
    zmm10 = _mm512_loadu_pd(&p_v3[j]+8);
    zmm11 = _mm512_loadu_pd(&p_v3[j]+16);

    // 2.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v3[i]+24);
    zmm10 = _mm512_loadu_pd(&p_v3[i]+32);
    zmm11 = _mm512_loadu_pd(&p_v3[i]+40);
#endif

    // 2.mac-2
    ZMM_MUL(zmm9,  zmm3, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm4, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm5, zmm_t1, zmm_t2);
    zmm3 = _mm512_add_pd(zmm3, zmm4);
    zmm3 = _mm512_add_pd(zmm3, zmm5);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm3);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v3[i]+48);
    zmm10 = _mm512_loadu_pd(&p_v3[i]+56);
    zmm11 = _mm512_loadu_pd(&p_v3[i]+64);
#endif

    // 2.mac-3
    ZMM_MUL(zmm9,  zmm6, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm7, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm8, zmm_t1, zmm_t2);
    zmm6 = _mm512_add_pd(zmm6, zmm7);
    zmm6 = _mm512_add_pd(zmm6, zmm8);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm6);
  }

  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a1);
  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a2);
  
  return _mm512_reduce_add_pd(zmm_a0);
#elif defined (FOUR_VECTOR_V2)
  __m512d zmm0, zmm1, zmm2, zmm3,
          zmm4, zmm5, zmm6, zmm7,
          zmm8, zmm9, zmm10, zmm11,
          zmm12, zmm_t1, zmm_t2,
          zmm_a0, zmm_a1, zmm_a2;

  zmm_a0 = _mm512_setzero_pd();
  zmm_a1 = _mm512_setzero_pd();
  zmm_a2 = _mm512_setzero_pd();


  for (long long i=0; i<(n/32)*32; i+=32) {
    // 1.3x3
    zmm0 = _mm512_loadu_pd(&p_v0[i]);
    zmm1 = _mm512_loadu_pd(&p_v0[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v0[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v0[i]+24);

    // 1.3x1
    zmm9  = _mm512_loadu_pd(&p_v1[i]);
    zmm10 = _mm512_loadu_pd(&p_v1[i]+8);
    zmm11 = _mm512_loadu_pd(&p_v1[i]+16);
    zmm12 = _mm512_loadu_pd(&p_v1[i]+24);

    // 1.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    ZMM_MUL(zmm12, zmm3, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm0 = _mm512_add_pd(zmm0, zmm3);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

    // 2.3x3
    zmm0 = _mm512_loadu_pd(&p_v2[i]);
    zmm1 = _mm512_loadu_pd(&p_v2[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v2[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v2[i]+24);

    // 2.3x1
    zmm9  = _mm512_loadu_pd(&p_v3[i]);
    zmm10 = _mm512_loadu_pd(&p_v3[i]+8);
    zmm11 = _mm512_loadu_pd(&p_v3[i]+16);
    zmm12 = _mm512_loadu_pd(&p_v3[i]+24);

    // 2.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    ZMM_MUL(zmm12, zmm3, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm0 = _mm512_add_pd(zmm0, zmm3);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm0);
  }

  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a1);
  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a2);
  
  return _mm512_reduce_add_pd(zmm_a0);

#elif defined (FOUR_VECTOR_V3)
  __m512d zmm0, zmm1, zmm2, zmm3,
          zmm4, zmm5, zmm6, zmm7,
          zmm8, zmm9, zmm10, zmm11,
          zmm12, zmm_t1, zmm_t2,
          zmm_a0, zmm_a1, zmm_a2;

  zmm_a0 = _mm512_setzero_pd();
  zmm_a1 = _mm512_setzero_pd();
  zmm_a2 = _mm512_setzero_pd();


  for (long long i=0; i<(n/8)*8; i+=8) {
    // 1.3x3
    zmm0 = _mm512_loadu_pd(&p_v0[i]);

    // 1.3x1
    zmm9  = _mm512_loadu_pd(&p_v1[i]);

    // 1.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

    // 2.3x3
    zmm0 = _mm512_loadu_pd(&p_v2[i]);

    // 2.3x1
    zmm9  = _mm512_loadu_pd(&p_v3[i]);

    // 2.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm0);
  }

  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a1);
  
  return _mm512_reduce_add_pd(zmm_a0);
#elif defined (FOUR_VECTOR_V4)
  __m512d zmm0, zmm1, zmm2, zmm3,
          zmm4, zmm5, zmm6, zmm7,
          zmm8, zmm9, zmm10, zmm11,
          zmm12, zmm_t1, zmm_t2,
          zmm_a0, zmm_a1, zmm_a2;

  zmm_a0 = _mm512_setzero_pd();
  zmm_a1 = _mm512_setzero_pd();
  zmm_a2 = _mm512_setzero_pd();


  for (long long i=0; i<(n/16)*16; i+=16) {
    // 1.3x3
    zmm0 = _mm512_loadu_pd(&p_v0[i]);
    zmm1 = _mm512_loadu_pd(&p_v0[i]+8);

    // 1.3x1
    zmm9  = _mm512_loadu_pd(&p_v1[i]);
    zmm10 = _mm512_loadu_pd(&p_v1[i]+8);

    // 1.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

    // 2.3x3
    zmm0 = _mm512_loadu_pd(&p_v2[i]);
    zmm1 = _mm512_loadu_pd(&p_v2[i]+8);

    // 2.3x1
    zmm9  = _mm512_loadu_pd(&p_v3[i]);
    zmm10 = _mm512_loadu_pd(&p_v3[i]+8);

    // 2.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm0);
  }

  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a1);
  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a2);

  return _mm512_reduce_add_pd(zmm_a0);

#elif defined (FOUR_VECTOR_V5)
  __m512d zmm0, zmm1, zmm2, zmm3,
          zmm4, zmm5, zmm6, zmm7,
          zmm8, zmm9, zmm10, zmm11,
          zmm_t1, zmm_t2,
          zmm_a0, zmm_a1, zmm_a2;

  zmm_a0 = _mm512_setzero_pd();
  zmm_a1 = _mm512_setzero_pd();
  zmm_a2 = _mm512_setzero_pd();


  for (long long i=0, j=0; i<(n/72)*72; i+=72, j+=24) {
    // 1.3x3
    zmm0 = _mm512_loadu_pd(&p_v0[i]);
    zmm1 = _mm512_loadu_pd(&p_v0[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v0[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v0[i]+24);
    zmm4 = _mm512_loadu_pd(&p_v0[i]+32);
    zmm5 = _mm512_loadu_pd(&p_v0[i]+40);
    zmm6 = _mm512_loadu_pd(&p_v0[i]+48);
    zmm7 = _mm512_loadu_pd(&p_v0[i]+56);
    zmm8 = _mm512_loadu_pd(&p_v0[i]+64);

    // 1.3x1
    zmm9  = _mm512_loadu_pd(&p_v1[j]);
    zmm10 = _mm512_loadu_pd(&p_v1[j]+8);
    zmm11 = _mm512_loadu_pd(&p_v1[j]+16);

    // 1.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

    // 1.mac-2
    ZMM_MUL(zmm9,  zmm3, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm4, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm5, zmm_t1, zmm_t2);
    zmm3 = _mm512_add_pd(zmm3, zmm4);
    zmm3 = _mm512_add_pd(zmm3, zmm5);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm3);

    // 1.mac-3
    ZMM_MUL(zmm9,  zmm6, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm7, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm8, zmm_t1, zmm_t2);
    zmm6 = _mm512_add_pd(zmm6, zmm7);
    zmm6 = _mm512_add_pd(zmm6, zmm8);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm6);

    // 2.3x3
    zmm0 = _mm512_loadu_pd(&p_v2[i]);
    zmm1 = _mm512_loadu_pd(&p_v2[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v2[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v2[i]+24);
    zmm4 = _mm512_loadu_pd(&p_v2[i]+32);
    zmm5 = _mm512_loadu_pd(&p_v2[i]+40);
    zmm6 = _mm512_loadu_pd(&p_v2[i]+48);
    zmm7 = _mm512_loadu_pd(&p_v2[i]+56);
    zmm8 = _mm512_loadu_pd(&p_v2[i]+64);

    // 2.3x1
    zmm9  = _mm512_loadu_pd(&p_v3[j]);
    zmm10 = _mm512_loadu_pd(&p_v3[j]+8);
    zmm11 = _mm512_loadu_pd(&p_v3[j]+16);

    // 2.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

    // 2.mac-2
    ZMM_MUL(zmm9,  zmm3, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm4, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm5, zmm_t1, zmm_t2);
    zmm3 = _mm512_add_pd(zmm3, zmm4);
    zmm3 = _mm512_add_pd(zmm3, zmm5);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm3);

    // 2.mac-3
    ZMM_MUL(zmm9,  zmm6, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm7, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm8, zmm_t1, zmm_t2);
    zmm6 = _mm512_add_pd(zmm6, zmm7);
    zmm6 = _mm512_add_pd(zmm6, zmm8);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm6);
  }

  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a1);
  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a2);
  
  return _mm512_reduce_add_pd(zmm_a0);

#else
#error "undefined kernel in four-vector"
#endif

#else
  for (long long i=0; i<n; i++) {
    tmp0 += p_v0[i];
    tmp1 += p_v1[i];
    tmp2 += p_v2[i];
    tmp3 += p_v3[i];
  }
  return tmp0 + tmp1 + tmp2 + tmp3;
#endif

}

double five_vector(long long n, long long start, double **p_vectors)
{
  double *p_v0 = p_vectors[0] + start;
  double *p_v1 = p_vectors[1] + start;
  double *p_v2 = p_vectors[2] + start;
  double *p_v3 = p_vectors[3] + start;
  double *p_v4 = p_vectors[4] + start;

  double tmp0 = 0.0;
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
  double tmp4 = 0.0;

  for (long long i=0; i<n; i++) {
    tmp0 += p_v0[i];
    tmp1 += p_v1[i];
    tmp2 += p_v2[i];
    tmp3 += p_v3[i];
    tmp4 += p_v4[i];
  }
  return tmp0 + tmp1 + tmp2 + tmp3 + tmp4;
}

double six_vector(long long n, long long start, double **p_vectors)
{
  double *p_v0 = p_vectors[0] + start;
  double *p_v1 = p_vectors[1] + start;
  double *p_v2 = p_vectors[2] + start;
  double *p_v3 = p_vectors[3] + start;
  double *p_v4 = p_vectors[4] + start;
  double *p_v5 = p_vectors[5] + start;

  double tmp0 = 0.0;
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
  double tmp4 = 0.0;
  double tmp5 = 0.0;

  for (long long i=0; i<n; i++) {
    tmp0 += p_v0[i];
    tmp1 += p_v1[i];
    tmp2 += p_v2[i];
    tmp3 += p_v3[i];
    tmp4 += p_v4[i];
    tmp5 += p_v5[i];
  }
  return tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5;
}


double seven_vector(long long n, long long start, double **p_vectors)
{
  double *p_v0 = p_vectors[0] + start;
  double *p_v1 = p_vectors[1] + start;
  double *p_v2 = p_vectors[2] + start;
  double *p_v3 = p_vectors[3] + start;
  double *p_v4 = p_vectors[4] + start;
  double *p_v5 = p_vectors[5] + start;
  double *p_v6 = p_vectors[6] + start;

  double tmp0 = 0.0;
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
  double tmp4 = 0.0;
  double tmp5 = 0.0;
  double tmp6 = 0.0;

  for (long long i=0; i<n; i++) {
    tmp0 += p_v0[i];
    tmp1 += p_v1[i];
    tmp2 += p_v2[i];
    tmp3 += p_v3[i];
    tmp4 += p_v4[i];
    tmp5 += p_v5[i];
    tmp6 += p_v6[i];
  }
  return tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6;
}


double eight_vector(long long n, long long start, double **p_vectors)
{
  double *p_v0 = p_vectors[0] + start;
  double *p_v1 = p_vectors[1] + start;
  double *p_v2 = p_vectors[2] + start;
  double *p_v3 = p_vectors[3] + start;
  double *p_v4 = p_vectors[4] + start;
  double *p_v5 = p_vectors[5] + start;
  double *p_v6 = p_vectors[6] + start;
  double *p_v7 = p_vectors[7] + start;

#if 0
  double tmp0 = 0.0;
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
  double tmp4 = 0.0;
  double tmp5 = 0.0;
  double tmp6 = 0.0;
  double tmp7 = 0.0;

  for (long long i=0; i<n; i++) {
    tmp0 += p_v0[i];
    tmp1 += p_v1[i];
    tmp2 += p_v2[i];
    tmp3 += p_v3[i];
    tmp4 += p_v4[i];
    tmp5 += p_v5[i];
    tmp6 += p_v6[i];
    tmp7 += p_v7[i];
  }
  return tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7;
#else
  __m512d zmm0, zmm1, zmm2, zmm3,
          zmm4, zmm5, zmm6, zmm7,
          zmm_t1, zmm_t2,
          zmm_a0, zmm_a1, zmm_a2, zmm_a3;

  zmm_a0 = _mm512_setzero_pd();
  zmm_a1 = _mm512_setzero_pd();
  zmm_a2 = _mm512_setzero_pd();
  zmm_a3 = _mm512_setzero_pd();

  for (long long i=0; i<(n/8)*8; i+=8) {
    zmm0 = _mm512_loadu_pd(&p_v0[i]);
    zmm1 = _mm512_loadu_pd(&p_v1[i]);
    ZMM_MUL(zmm0, zmm1, zmm_t1, zmm_t2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm1);

    zmm2 = _mm512_loadu_pd(&p_v2[i]);
    zmm3 = _mm512_loadu_pd(&p_v3[i]);
    ZMM_MUL(zmm2, zmm3, zmm_t1, zmm_t2);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm3);

    zmm4 = _mm512_loadu_pd(&p_v4[i]);
    zmm5 = _mm512_loadu_pd(&p_v5[i]);
    ZMM_MUL(zmm4, zmm5, zmm_t1, zmm_t2);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm5);

    zmm6 = _mm512_loadu_pd(&p_v6[i]);
    zmm7 = _mm512_loadu_pd(&p_v7[i]);
    ZMM_MUL(zmm6, zmm7, zmm_t1, zmm_t2);
    zmm_a3 = _mm512_add_pd(zmm_a3, zmm7);
  }

  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a1);
  zmm_a2 = _mm512_add_pd(zmm_a2, zmm_a3);
  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a2);
  
  return _mm512_reduce_add_pd(zmm_a0);
#endif
}

double sixteen_vector(long long n, long long start, double **p_vectors)
{
  double *p_v0 = p_vectors[0] + start;
  double *p_v1 = p_vectors[1] + start;
  double *p_v2 = p_vectors[2] + start;
  double *p_v3 = p_vectors[3] + start;
  double *p_v4 = p_vectors[4] + start;
  double *p_v5 = p_vectors[5] + start;
  double *p_v6 = p_vectors[6] + start;
  double *p_v7 = p_vectors[7] + start;
  double *p_v8 = p_vectors[8] + start;
  double *p_v9 = p_vectors[9] + start;
  double *p_v10 = p_vectors[10] + start;
  double *p_v11 = p_vectors[11] + start;
  double *p_v12 = p_vectors[12] + start;
  double *p_v13 = p_vectors[13] + start;
  double *p_v14 = p_vectors[14] + start;
  double *p_v15 = p_vectors[15] + start;

  double tmp0 = 0.0;
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
  double tmp4 = 0.0;
  double tmp5 = 0.0;
  double tmp6 = 0.0;
  double tmp7 = 0.0;
  double tmp8 = 0.0;
  double tmp9 = 0.0;
  double tmp10 = 0.0;
  double tmp11 = 0.0;
  double tmp12 = 0.0;
  double tmp13 = 0.0;
  double tmp14 = 0.0;
  double tmp15 = 0.0;

  for (long long i=0; i<n; i++) {
    tmp0 += p_v0[i];
    tmp1 += p_v1[i];
    tmp2 += p_v2[i];
    tmp3 += p_v3[i];
    tmp4 += p_v4[i];
    tmp5 += p_v5[i];
    tmp6 += p_v6[i];
    tmp7 += p_v7[i];
    tmp8 += p_v8[i];
    tmp9 += p_v9[i];
    tmp10 += p_v10[i];
    tmp11 += p_v11[i];
    tmp12 += p_v12[i];
    tmp13 += p_v13[i];
    tmp14 += p_v14[i];
    tmp15 += p_v15[i];
  }
  return tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 + 
         tmp8 + tmp9 + tmp10 + tmp11 + tmp12 + tmp13 + tmp14 + tmp15;
}

double eight_vector_proxy(long long n, long long start, double **p_vectors)
{
  double *p_v0 = p_vectors[0] + start;
  double *p_v1 = p_vectors[1] + start;
  double *p_v2 = p_vectors[2] + start;
  double *p_v3 = p_vectors[3] + start;
  double *p_v4 = p_vectors[4] + start;
  double *p_v5 = p_vectors[5] + start;
  double *p_v6 = p_vectors[6] + start;
  double *p_v7 = p_vectors[7] + start;

  double tmp0 = 0.0;
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
  double tmp4 = 0.0;
  double tmp5 = 0.0;
  double tmp6 = 0.0;
  double tmp7 = 0.0;

#define V3

#if defined (V0)
  for (long long i=0; i<n; i++) {
    tmp0 += p_v0[i];
    tmp1 += p_v1[i];
    tmp2 += p_v2[i];
    tmp3 += p_v3[i];
    tmp4 += p_v4[i];
    tmp5 += p_v5[i];
    tmp6 += p_v6[i];
    tmp7 += p_v7[i];
  }
  return tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7;

#elif defined (V1)
  for (long long i=0; i<(n/72)*72; i+=72) {
#pragma nounroll
    for (int j=0; j<3; j++) {
#pragma nounroll
      for (int k=0; k<24; k++) {
#if 1
        tmp0 += p_v0[i+(j*24)+k] * p_v1[i+k];
        tmp1 += p_v2[i+(j*24)+k] * p_v3[i+k];
        tmp2 += p_v4[i+(j*24)+k] * p_v5[i+k];
        tmp3 += p_v6[i+(j*24)+k] * p_v7[i+k];
#else
        tmp0 += p_v0[i+(j*24)+k] * p_v1[i+(j*24)+k];
        tmp1 += p_v2[i+(j*24)+k] * p_v3[i+(j*24)+k];
        tmp2 += p_v4[i+(j*24)+k] * p_v5[i+(j*24)+k];
        tmp3 += p_v6[i+(j*24)+k] * p_v7[i+(j*24)+k];

        tmp0 += p_v0[i+(j*24)+k];
        tmp1 += p_v2[i+(j*24)+k];
        tmp2 += p_v4[i+(j*24)+k];
        tmp3 += p_v6[i+(j*24)+k];

        tmp4 += p_v1[i+(j*24)+k];
        tmp5 += p_v3[i+(j*24)+k];
        tmp6 += p_v5[i+(j*24)+k];
        tmp7 += p_v7[i+(j*24)+k];

#endif
      }
    }
  }
  return tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7;
  /* return tmp0 + tmp1 + tmp2 + tmp3; */

#elif defined (V2)

  int blk_size = 8 * 72;
  int t0, t1;

  double tmp_buf0[72*8], tmp_buf2[72*8], tmp_buf4[72*8], tmp_buf6[72*8],
         tmp_buf1[24*8], tmp_buf3[24*8], tmp_buf5[24*8], tmp_buf7[24*8]
          __attribute__((aligned(64)));

  for (long long b=0; b<(n/blk_size)*blk_size; b+=blk_size) { 
    t0 = 0;
    t1 = 0;
    for (long long i=0; i<blk_size; i+=72) {

      for (int j=0; j<3; j++) {
        for (int k=0; k<24; k++) {
          tmp_buf0[t0] = p_v0[b+i+(j*24)+k];
          tmp_buf2[t0] = p_v2[b+i+(j*24)+k];
          tmp_buf4[t0] = p_v4[b+i+(j*24)+k];
          tmp_buf6[t0] = p_v6[b+i+(j*24)+k];
          t0++;
        }
      }

      for (int k=0; k<24; k++) {
        tmp_buf1[t1] = p_v1[b+i+k];
        tmp_buf3[t1] = p_v3[b+i+k];
        tmp_buf5[t1] = p_v5[b+i+k];
        tmp_buf7[t1] = p_v7[b+i+k];
        t1++;
      }
    }

    for (long long i=0; i<blk_size; i+=72) {
      for (int j=0; j<3; j++) {
        for (int k=0; k<24; k++) {
          tmp0 += tmp_buf0[i+(j*24)+k] * tmp_buf1[k];
          tmp1 += tmp_buf2[i+(j*24)+k] * tmp_buf3[k];
          tmp2 += tmp_buf4[i+(j*24)+k] * tmp_buf5[k];
          tmp3 += tmp_buf6[i+(j*24)+k] * tmp_buf7[k];
        }
      }
    }
  }

  return tmp0 + tmp1 + tmp2 + tmp3;

#elif defined (V3)
  __m512d zmm0, zmm1, zmm2, zmm3,
          zmm4, zmm5, zmm6, zmm7,
          zmm8, zmm9, zmm10, zmm11,
          zmm_t1, zmm_t2,
          zmm_a0, zmm_a1, zmm_a2;

  zmm_a0 = _mm512_setzero_pd();
  zmm_a1 = _mm512_setzero_pd();
  zmm_a2 = _mm512_setzero_pd();


  /* long long new_n = n/576; */
  long long new_n = n;


  for (long long ii=0, i=0, j=0; ii<(new_n/72)*72; ii+=72, i+=72, j+=72) {
    // 1.3x3
    zmm0 = _mm512_loadu_pd(&p_v0[i]);
    zmm1 = _mm512_loadu_pd(&p_v0[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v0[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v0[i]+24);
    zmm4 = _mm512_loadu_pd(&p_v0[i]+32);
    zmm5 = _mm512_loadu_pd(&p_v0[i]+40);
    zmm6 = _mm512_loadu_pd(&p_v0[i]+48);
    zmm7 = _mm512_loadu_pd(&p_v0[i]+56);
    zmm8 = _mm512_loadu_pd(&p_v0[i]+64);

    // 1.3x1
    zmm9  = _mm512_loadu_pd(&p_v1[j]);
    zmm10 = _mm512_loadu_pd(&p_v1[j]+8);
    zmm11 = _mm512_loadu_pd(&p_v1[j]+16);

    // 1.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v1[i]+24);
    zmm10 = _mm512_loadu_pd(&p_v1[i]+32);
    zmm11 = _mm512_loadu_pd(&p_v1[i]+40);
#endif
    
    // 1.mac-2
    ZMM_MUL(zmm9,  zmm3, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm4, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm5, zmm_t1, zmm_t2);
    zmm3 = _mm512_add_pd(zmm3, zmm4);
    zmm3 = _mm512_add_pd(zmm3, zmm5);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm3);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v1[i]+48);
    zmm10 = _mm512_loadu_pd(&p_v1[i]+56);
    zmm11 = _mm512_loadu_pd(&p_v1[i]+64);
#endif

    // 1.mac-3
    ZMM_MUL(zmm9,  zmm6, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm7, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm8, zmm_t1, zmm_t2);
    zmm6 = _mm512_add_pd(zmm6, zmm7);
    zmm6 = _mm512_add_pd(zmm6, zmm8);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm6);

    // 2.3x3
    zmm0 = _mm512_loadu_pd(&p_v2[i]);
    zmm1 = _mm512_loadu_pd(&p_v2[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v2[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v2[i]+24);
    zmm4 = _mm512_loadu_pd(&p_v2[i]+32);
    zmm5 = _mm512_loadu_pd(&p_v2[i]+40);
    zmm6 = _mm512_loadu_pd(&p_v2[i]+48);
    zmm7 = _mm512_loadu_pd(&p_v2[i]+56);
    zmm8 = _mm512_loadu_pd(&p_v2[i]+64);

    // 2.3x1
    zmm9  = _mm512_loadu_pd(&p_v3[j]);
    zmm10 = _mm512_loadu_pd(&p_v3[j]+8);
    zmm11 = _mm512_loadu_pd(&p_v3[j]+16);

    // 2.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v3[i]+24);
    zmm10 = _mm512_loadu_pd(&p_v3[i]+32);
    zmm11 = _mm512_loadu_pd(&p_v3[i]+40);
#endif

    // 2.mac-2
    ZMM_MUL(zmm9,  zmm3, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm4, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm5, zmm_t1, zmm_t2);
    zmm3 = _mm512_add_pd(zmm3, zmm4);
    zmm3 = _mm512_add_pd(zmm3, zmm5);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm3);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v3[i]+48);
    zmm10 = _mm512_loadu_pd(&p_v3[i]+56);
    zmm11 = _mm512_loadu_pd(&p_v3[i]+64);
#endif

    // 2.mac-3
    ZMM_MUL(zmm9,  zmm6, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm7, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm8, zmm_t1, zmm_t2);
    zmm6 = _mm512_add_pd(zmm6, zmm7);
    zmm6 = _mm512_add_pd(zmm6, zmm8);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm6);


    // 3.3x3
    zmm0 = _mm512_loadu_pd(&p_v4[i]);
    zmm1 = _mm512_loadu_pd(&p_v4[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v4[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v4[i]+24);
    zmm4 = _mm512_loadu_pd(&p_v4[i]+32);
    zmm5 = _mm512_loadu_pd(&p_v4[i]+40);
    zmm6 = _mm512_loadu_pd(&p_v4[i]+48);
    zmm7 = _mm512_loadu_pd(&p_v4[i]+56);
    zmm8 = _mm512_loadu_pd(&p_v4[i]+64);

    // 3.3x1
    zmm9  = _mm512_loadu_pd(&p_v5[j]);
    zmm10 = _mm512_loadu_pd(&p_v5[j]+8);
    zmm11 = _mm512_loadu_pd(&p_v5[j]+16);

    // 3.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v5[i]+24);
    zmm10 = _mm512_loadu_pd(&p_v5[i]+32);
    zmm11 = _mm512_loadu_pd(&p_v5[i]+40);
#endif

    // 3.mac-2
    ZMM_MUL(zmm9,  zmm3, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm4, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm5, zmm_t1, zmm_t2);
    zmm3 = _mm512_add_pd(zmm3, zmm4);
    zmm3 = _mm512_add_pd(zmm3, zmm5);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm3);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v5[i]+48);
    zmm10 = _mm512_loadu_pd(&p_v5[i]+56);
    zmm11 = _mm512_loadu_pd(&p_v5[i]+64);
#endif

    // 3.mac-3
    ZMM_MUL(zmm9,  zmm6, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm7, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm8, zmm_t1, zmm_t2);
    zmm6 = _mm512_add_pd(zmm6, zmm7);
    zmm6 = _mm512_add_pd(zmm6, zmm8);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm6);

    // 4.3x3
    zmm0 = _mm512_loadu_pd(&p_v6[i]);
    zmm1 = _mm512_loadu_pd(&p_v6[i]+8);
    zmm2 = _mm512_loadu_pd(&p_v6[i]+16);
    zmm3 = _mm512_loadu_pd(&p_v6[i]+24);
    zmm4 = _mm512_loadu_pd(&p_v6[i]+32);
    zmm5 = _mm512_loadu_pd(&p_v6[i]+40);
    zmm6 = _mm512_loadu_pd(&p_v6[i]+48);
    zmm7 = _mm512_loadu_pd(&p_v6[i]+56);
    zmm8 = _mm512_loadu_pd(&p_v6[i]+64);

    // 4.3x1
    zmm9  = _mm512_loadu_pd(&p_v7[j]);
    zmm10 = _mm512_loadu_pd(&p_v7[j]+8);
    zmm11 = _mm512_loadu_pd(&p_v7[j]+16);

    // 4.mac-1
    ZMM_MUL(zmm9,  zmm0, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm1, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm2, zmm_t1, zmm_t2);
    zmm0 = _mm512_add_pd(zmm0, zmm1);
    zmm0 = _mm512_add_pd(zmm0, zmm2);
    zmm_a0 = _mm512_add_pd(zmm_a0, zmm0);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v7[i]+24);
    zmm10 = _mm512_loadu_pd(&p_v7[i]+32);
    zmm11 = _mm512_loadu_pd(&p_v7[i]+40);
#endif

    // 4.mac-2
    ZMM_MUL(zmm9,  zmm3, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm4, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm5, zmm_t1, zmm_t2);
    zmm3 = _mm512_add_pd(zmm3, zmm4);
    zmm3 = _mm512_add_pd(zmm3, zmm5);
    zmm_a1 = _mm512_add_pd(zmm_a1, zmm3);

#if 1
    zmm9  = _mm512_loadu_pd(&p_v7[i]+48);
    zmm10 = _mm512_loadu_pd(&p_v7[i]+56);
    zmm11 = _mm512_loadu_pd(&p_v7[i]+64);
#endif

    // 4.mac-3
    ZMM_MUL(zmm9,  zmm6, zmm_t1, zmm_t2);
    ZMM_MUL(zmm10, zmm7, zmm_t1, zmm_t2);
    ZMM_MUL(zmm11, zmm8, zmm_t1, zmm_t2);
    zmm6 = _mm512_add_pd(zmm6, zmm7);
    zmm6 = _mm512_add_pd(zmm6, zmm8);
    zmm_a2 = _mm512_add_pd(zmm_a2, zmm6);
  }

  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a1);
  zmm_a0 = _mm512_add_pd(zmm_a0, zmm_a2);
  
  return _mm512_reduce_add_pd(zmm_a0);
#endif
}
