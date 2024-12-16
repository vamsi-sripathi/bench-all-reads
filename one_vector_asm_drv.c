#include "stdio.h"
#include "immintrin.h"

#define L2_PF_DIST  (4096)
#define L1_PF_DIST  (512)

double one_vector_ker(long long *, double *);

double one_vector(long long n, long long start, double **p_vectors)
{
  double *p_x = p_vectors[0] + start;
  double tmp = 0.0;

  int offset;
  long long n_blk;

  offset = ((unsigned long)p_x) % 64;
  n_blk  = (n/32)*32;

  if ((n_blk > 32) && (offset == 0)) {
    tmp += one_vector_ker(&n, p_x);

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
}
