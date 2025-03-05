#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "kernels.h"

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

//Initial kernels
void memory_store_memset(void *ptr, size_t size)
{
  memset(ptr, 0xff, size);
}

void memory_rep_stosq(void *ptr, size_t size)
{
  asm("cld\n"
      "rep stosq"
      : : "D" (ptr), "c" (size >> 3), "a" (0) );
}

void memory_rep_lodsq(void *ptr, size_t size)
{
  asm("cld\n"
      "rep lodsq"
       : : "S" (ptr), "c" (size >> 3) : "%eax");
}

void memory_store_loop(void *ptr, size_t size)
{
  size_t *cptr = (size_t *)ptr;
  
  for (size_t i = 0; i < size / sizeof(size_t); i++)
    cptr[i] = 1;
}

//Here, we 'ignore' the addition
void memory_load_loop(void *ptr, size_t size)
{
  size_t val = 0;
  size_t *cptr = (size_t *) ptr;

  for (size_t i = 0; i < size / sizeof(size_t); i++)
    val += cptr[i];

  //Use val to avoid having the code optimized away by the compiler
  assert(val != 0xdeadbeef);
}

//SSE kernels
#ifdef __SSE4_1__

void memory_store_nontemporal_sse(void *ptr, size_t size)
{
  __m128i *vptr = (__m128i *)ptr;

  __m128i vals = _mm_set1_epi32(1);
  
  for (size_t i = 0; i < size / sizeof(__m128i); i++)
    {
      _mm_stream_si128(&vptr[i], vals);
      vals = _mm_add_epi16(vals, vals);
    }
}

void memory_store_sse(void* ptr, size_t size)
{
  __m128i *vptr = (__m128i *)ptr;

  __m128i vals = _mm_set1_epi32(1);

  for (size_t i = 0; i < size / sizeof(__m128i); i++)
    {
      _mm_store_si128(&vptr[i], vals);
      vals = _mm_add_epi16(vals, vals);
    }
}

void memory_load_sse(void *ptr, size_t size)
{
  __m128i *vptr = (__m128i *) ptr;
  __m128i accum = _mm_set1_epi32(0xDEADBEEF);

  for (size_t i = 0; i < size / sizeof(__m128i); i++)
    accum = _mm_add_epi16(vptr[i], accum);
  
  assert(!_mm_testz_si128(accum, accum));
}

#endif

//AVX kernels
#ifdef __AVX__

void memory_store_nontemporal_avx(void* ptr, size_t size)
{
  __m256i *vptr = (__m256i *)ptr;
  __m256i vals = _mm256_set1_epi32(0xDEADBEEF);

  for (size_t i = 0; i < size / sizeof(__m256); i++)
    _mm256_stream_si256((__m256i *) &vptr[i], vals);
}

void memory_store_avx(void *ptr, size_t size)
{
  __m256i *vptr = (__m256i *)ptr;
  
  __m256i vals = _mm256_set1_epi32(0xDEADBEEF);
  
  for (size_t i = 0; i < size / sizeof(__m256i); i++)
    _mm256_store_si256(&vptr[i], vals);
}

void memory_load_prefetch_avx(void* ptr, size_t size)
{
  __m256 *vptr = (__m256 *)ptr;
  __m256 accum = _mm256_set1_ps((float) 0xDEADBEEF);

  for (size_t i = 0; i < size / sizeof(__m256i); i++)
    {
      //Use PREFETCHNTA as instructed by the Intel Optimization Manual
      //when the algorithm is single pass (Page 7-3).
      //https://www.intel.com/content/dam/doc/manual/64-ia-32-architectures-optimization-manual.pdf.
      //https://lwn.net/Articles/444336/
      
      _mm_prefetch(&vptr[i+2], _MM_HINT_NTA);
      accum = _mm256_add_ps(vptr[i], accum);
    }
  
  assert(!_mm256_testz_ps(accum, accum));
}

void memory_load_avx(void* ptr, size_t size)
{
  __m256 *vptr = (__m256 *)ptr;
  __m256 accum = _mm256_set1_ps((float) 0xDEADBEEF);
  
  for (size_t i = 0; i < size / sizeof(__m256i); i++)
    accum = _mm256_add_ps(vptr[i], accum);
  
  assert(!_mm256_testz_ps(accum, accum));
}
#endif
