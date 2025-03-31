#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "math.h"
#include "omp.h"
#include "float.h"
#include "mkl.h"
#include "immintrin.h"

#define MAX_NUM_VECTORS (20)
#define MAX_THREADS     (256)

#ifndef NTRIALS
#define NTRIALS         (100)
#endif

#ifndef ALIGNMENT
#define ALIGNMENT  (64)
/* #define ALIGNMENT  (262144) */
#endif

/* #define SKIP_VALIDATION */

double one_vector(long long n, long long start, double **p_vectors);
double two_vector(long long n, long long start, double **p_vectors);
double three_vector(long long n, long long start, double **p_vectors);
double four_vector(long long n, long long start, double **p_vectors);
double five_vector(long long n, long long start, double **p_vectors);
double six_vector(long long n, long long start, double **p_vectors);
double seven_vector(long long n, long long start, double **p_vectors);
double eight_vector(long long n, long long start, double **p_vectors);
double eight_vector_proxy(long long n, long long start, double **p_vectors);
double sixteen_vector(long long n, long long start, double **p_vectors);

typedef double (*p_func)(long long, long long, double **);

p_func p_bench_func;

p_func get_bench_func(int num_vectors)
{
  if (num_vectors == 1) {
    return one_vector;
  } else if (num_vectors == 2) {
    return two_vector;
  } else if (num_vectors == 3) {
    return three_vector;
  } else if (num_vectors == 4) {
    return four_vector;
  } else if (num_vectors == 5) {
    return five_vector;
  } else if (num_vectors == 6) {
    return six_vector;
  } else if (num_vectors == 7) {
    return seven_vector;
  } else if (num_vectors == 8) {
    return eight_vector;
  } else if (num_vectors == 16) {
    return sixteen_vector;
  } else {
    printf ("ERROR: %d vectors are not supported\n", num_vectors);
    exit (1);
  }
}

void omp_all_read(int num_vectors, long long n, double **p_vectors, double *p_partial_res)
{
#if 0
#pragma omp parallel default(shared) num_threads(7)
  {
    int nthrs = omp_get_num_threads();
    int ithr = omp_get_thread_num();
    ssize_t chunk, start, rem;

    ssize_t outer_chunk = n/nthrs;
    start = ithr * outer_chunk;
    double *outer_a = p_vectors[0] + start;
    double ithr_tmp[3] = {0};

    #pragma omp parallel default(shared) num_threads(2)
    {
      ssize_t inner_chunk = (56*1024)/8;
      int inner_nthrs = 2;

      ithr = omp_get_thread_num();
      int n_blks = outer_chunk/(inner_nthrs * inner_chunk);

      for (int i=0; i<n_blks; i++) {
       long long ithr_start = start + (i * inner_nthrs * inner_chunk) + (ithr * inner_chunk);
       ithr_tmp[ithr] += p_bench_func(inner_chunk, ithr_start, p_vectors);
      }
    }
    p_partial_res[ithr] = ithr_tmp[0] + ithr_tmp[1] + ithr_tmp[2];
  }
#else
#pragma omp parallel
  {
    int nthrs = omp_get_num_threads();
    int ithr  = omp_get_thread_num();

    long long chunk = ((n/nthrs)/8)*8;
    int rem         = n - (chunk * nthrs);
    int rem_8e = rem/8;
    int tail   = n - (chunk * nthrs) - (rem_8e * 8);
    int start;

    if (ithr < rem_8e) {
      chunk += 8;
      start = ithr * chunk;
    } else {
      start = (ithr * chunk) + (rem_8e * 8);
    }

    if (tail) {
      if (ithr == (nthrs-1)) {
        chunk += tail;
      }
    }

    double tmp = p_bench_func(chunk, start, p_vectors);
    p_partial_res[ithr] = tmp;
  }
#endif
}

int check_results (int num_vectors, long long n, double **p_vectors, double obs_res)
{
  double exp_res = 0.;

  for (int i=0; i<num_vectors; i++) {
    double *p_x = p_vectors[i];
    for (long long i=0; i<n; i++) {
      exp_res += p_x[i];
    }
  }

  if (fabs(exp_res - obs_res) > 0.1) {
    printf ("exp_res = %lf, obs_res = %lf\n", exp_res, obs_res);
    return 1;
  } else {
    return 0;
  }
}

int main (int argc, char **argv)
{
  double *p_vectors[MAX_NUM_VECTORS] = {NULL};
  int num_vectors, num_threads;
  long long n;

  if (argc != 3) {
    printf ("\nUSAGE: %s num_vectors elems_per_vector\n", argv[0]);
    exit (1);
  }

  num_vectors = atoi(argv[1]);
  n           = atoll(argv[2]);

  if (num_vectors > MAX_NUM_VECTORS) {
    printf ("num_vectors = %d is not supported.. exiting..\n",
            num_vectors);
    exit(1);
  }

  double *p_iter_time   = (double *) _mm_malloc(sizeof(double)*NTRIALS, ALIGNMENT);
  double *p_partial_res = (double *) _mm_malloc(sizeof(double)*MAX_THREADS, ALIGNMENT);

  for (int i=0; i<num_vectors; i++) {
    p_vectors[i] = (double *) _mm_malloc(sizeof(double)*n, ALIGNMENT);
    if (p_vectors[i] == NULL) {
      printf ("Memory allocation failed for vector-%d\n", i);
      fflush(0);
      exit(1);
    }
  }

#if 0
#pragma omp parallel default(shared) num_threads(7)
  {
    int nthrs = omp_get_num_threads();
    int ithr = omp_get_thread_num();
    ssize_t chunk, start, rem;

    ssize_t outer_chunk = n/nthrs;
    start = ithr * outer_chunk;
    double *outer_a = p_vectors[0] + start;

    #pragma omp parallel default(shared) num_threads(2)
    {
      ssize_t inner_chunk = (56*1024)/8;
      int inner_nthrs = 2;

      ithr = omp_get_thread_num();
      int n_blks = outer_chunk/(inner_nthrs * inner_chunk);

      for (int i=0; i<n_blks; i++) {
       double *ithr_a = outer_a + (i * inner_nthrs * inner_chunk) + (ithr * inner_chunk);
       for (int j=0; j<inner_chunk; j++) {
         ithr_a[j] = j%500;
       }
      }
    }
    #pragma omp master
      num_threads = nthrs;
  }
#else
#pragma omp parallel
  {
    int nthrs = omp_get_num_threads();
    int ithr  = omp_get_thread_num();

    long long chunk = ((n/nthrs)/8)*8;
    int rem         = n - (chunk * nthrs);
    int rem_8e = rem/8;
    int tail   = n - (chunk * nthrs) - (rem_8e * 8);
    int start;

    if (ithr < rem_8e) {
      chunk += 8;
      start = ithr * chunk;
    } else {
      start = (ithr * chunk) + (rem_8e * 8);
    }

    if (tail) {
      if (ithr == (nthrs-1)) {
        chunk += tail;
      }
    }

    double *p_x0 = p_vectors[0] + start;
    for (long long j=0; j<chunk; j++) {
      p_x0[j] = j%10 + ithr;
    }

    for (int i=1; i<num_vectors; i++) {
      double *p_xx = p_vectors[i] + start;
      for (long long j=0; j<chunk; j++) {
        p_xx[j] = i + p_x0[j];
      }
    }
#pragma omp master
    num_threads = nthrs;
  }
#endif

  double d_time, d_elapsed;
  d_time = dsecnd();
  d_time = dsecnd();
  double num_bytes_processed = num_vectors * n * sizeof(double);
  /* double num_bytes_processed = ((2 * n) + (2 * (n/3))) * sizeof(double); */
  double res = 0.;

  p_bench_func = get_bench_func(num_vectors);

  for (int i=0; i<NTRIALS; i++) {
    d_time = dsecnd();
    omp_all_read(num_vectors, n, p_vectors, p_partial_res);
    d_elapsed = dsecnd() - d_time;
    
    for (int t=0; t<num_threads; t++) {
      res += p_partial_res[t];
    }

    p_iter_time[i] = d_elapsed;
#ifdef VERBOSE
    printf ("Iter-%d: GB/s = %.2f\n", i, (num_bytes_processed/d_elapsed)*1.e-09);
    fflush(0);
#endif
  }

#ifndef SKIP_VALIDATION
  // one validation check for all trials
  if (check_results(num_vectors, n, p_vectors, res/NTRIALS)) {
    printf ("validation failed!\n");
    goto bailout;
  } else {
    printf ("validation passed\n");
  }
#else
  printf ("Skipping validation..\n");
#endif

  double max_bw = FLT_MIN;
  double min_bw = FLT_MAX;
  double total_bw = 0.;

  for (int i=0; i<NTRIALS; i++) {
    double bw = (num_bytes_processed/p_iter_time[i])*1.e-09;
    if (bw > max_bw) {
      max_bw = bw;
    }
    if (bw < min_bw) {
      min_bw = bw;
    }
    total_bw += bw;
  }

  printf ("N = %lld, num_vectors = %d, num_bytes_processed = %.2f GB; %.2f MB, GB/s: min_bw = %.2f, avg_bw = %.2f, max_bw = %.2f\n",
          n, num_vectors, num_bytes_processed*1.e-09, num_bytes_processed*1.e-06, min_bw, total_bw/NTRIALS, max_bw);

bailout:
  for (int i=0; i<num_vectors; i++) {
    _mm_free(p_vectors[i]);
  }
  _mm_free(p_iter_time);
  _mm_free(p_partial_res);

  return 0;
}
