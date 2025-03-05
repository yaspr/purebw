#pragma once

//#include <stddef.h>

void memory_store_memset(void *, size_t);
void memory_rep_stosq(void *, size_t);
void memory_rep_lodsq(void *, size_t);
void memory_store_loop(void *, size_t);
void memory_load_loop(void *, size_t);

#ifdef __SSE4_1__

void memory_store_nontemporal_sse(void *, size_t);
void memory_store_sse(void *, size_t);
void memory_load_sse(void *, size_t);

#endif

#ifdef __AVX__

void memory_store_nontemporal_avx(void *, size_t);
void memory_store_avx(void *, size_t);
void memory_load_prefetch_avx(void *, size_t);
void memory_load_avx(void *, size_t);

#endif


