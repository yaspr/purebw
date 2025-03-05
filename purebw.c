#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "kernels.h"
#include "monotonic_timer.h"

#define SAMPLES       5
#define RUNS          5
#define BYTES_PER_GB (1024 * 1024 * 1024LL)
#define SIZE         (1 * BYTES_PER_GB)
#define PAGE_SIZE    (1 << 12)

//AVX requires 32-byte alignment
//PAGE_SIZE buffering for easier address prefetching
char array[SIZE + PAGE_SIZE] __attribute__((aligned(32)));

//Compute the bandwidth in GiB/s.
static inline double compute_bw(size_t bytes, double secs)
{
  double size_bytes = (double)bytes;
  double size_gb = size_bytes / ((double)BYTES_PER_GB);
  
  return size_gb / secs;
}

#ifdef WITH_OPENMP

#define benchmark_omp(f) measure_omp(f, #f)

void measure_omp(void (*kernel)(void *, size_t), char *name)
{
  double min = INFINITY;
  
  for (size_t i = 0; i < SAMPLES; i++)
    {
      double total  = 0.0;
      double after  = 0.0;
      double before = 0.0;
      
      assert(SIZE % omp_get_max_threads() == 0);
      
      size_t chunk_size = SIZE / omp_get_max_threads();
      
#pragma omp parallel
      {
#pragma omp barrier
#pragma omp master
	
	before = monotonic_time();
	
	for (size_t j = 0; j < RUNS; j++) 
	  kernel(&array[chunk_size * omp_get_thread_num()], chunk_size);
	
#pragma omp barrier
#pragma omp master
	
	after = monotonic_time();
      }
      
      total = (after - before);
      
      if (total < min)
	min = total;
    }
  
  printf("%28s_omp: %5.2f GiB/s\n", name, compute_bw(SIZE * RUNS, min));
}

#endif

#define benchmark(f) measure(f, #f)

void measure(void (*kernel)(void *, size_t), char *name)
{
  double min = INFINITY;
  
  for (size_t i = 0; i < SAMPLES; i++)
    {
      double total = 0.0;
      double after = 0.0;
      double before = 0.0;
      
      before = monotonic_time();
      
      for (size_t j = 0; j < RUNS; j++)
	kernel(array, SIZE);
    
      after = monotonic_time();

      total = (after - before);

      if (total < min) 
	min = total;
    }
  
  printf("%32s: %5.2f GiB/s\n", name, compute_bw(SIZE * RUNS, min));
}

int main()
{
  //Zero-Fill-On-Demand
  memset(array, 0xFF, SIZE);
  
  *((uint64_t *)&array[SIZE]) = 0;

  benchmark(memory_rep_lodsq);
  benchmark(memory_load_loop);

#ifdef __SSE4_1__
  benchmark(memory_load_sse);
#endif

#ifdef __AVX__
  benchmark(memory_load_avx);
  benchmark(memory_load_prefetch_avx);
#endif

  benchmark(memory_store_loop);
  benchmark(memory_rep_stosq);

#ifdef __SSE4_1__
  benchmark(memory_store_sse);
  benchmark(memory_store_nontemporal_sse);
#endif

#ifdef __AVX__
  benchmark(memory_store_avx);
  benchmark(memory_store_nontemporal_avx);
#endif

  benchmark(memory_store_memset);
  
#ifdef WITH_OPENMP

  //Zero-Fill-On-Demand
  memset(array, 0xFF, SIZE);
  
  *((uint64_t *)&array[SIZE]) = 0;

  benchmark_omp(memory_rep_lodsq);
  benchmark_omp(memory_load_loop);
  
#ifdef __SSE4_1__
  benchmark_omp(memory_load_sse);
#endif

#ifdef __AVX__
  benchmark_omp(memory_load_avx);
  benchmark_omp(memory_load_prefetch_avx);
#endif

  benchmark_omp(memory_load_loop);
  benchmark_omp(memory_rep_stosq);

#ifdef __SSE4_1__
  benchmark_omp(memory_store_sse);
  benchmark_omp(memory_store_nontemporal_sse);
#endif

#ifdef __AVX__
  benchmark_omp(memory_store_avx);
  benchmark_omp(memory_store_nontemporal_avx);
#endif

  benchmark_omp(memory_store_memset);

#endif

  return 0;
}
