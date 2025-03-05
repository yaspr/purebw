//Timer with fallback to rdtsc
#include <unistd.h>

#define NANOS_PER_SECF 1000000000.0
#define USECS_PER_SEC 1000000

// If available, use the clock_gettime and CLOCK_MONOTONIC_RAW.
#if _POSIX_TIMERS > 0 && defined(_POSIX_MONOTONIC_CLOCK_RAW)

#include <time.h>

double monotonic_time() {

  struct timespec time;
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &time);

  return ((double) time.tv_sec) + ((double) time.tv_nsec / (NANOS_PER_SECF));
}

#else //Fall back to rdtsc

#include <stdint.h>

static uint64_t rdtsc_per_sec = 0;

static inline uint64_t rdtsc() {
  
  uint32_t hi = 0, lo = 0;

  __asm__ volatile(		   
		   "rdtscp\n"
		   "movl %%edx, %0\n"
		   "movl %%eax, %1\n"
		   "cpuid"
		   
		   : "=r" (hi), "=r" (lo)
		   :
		   : "%rax", "%rbx", "%rcx", "%rdx"
		   );
  
  return (((uint64_t)hi) << 32) | (uint64_t)lo;  
}

static void __attribute__((constructor)) init_rdtsc_per_sec()
{
  uint64_t before, after;
  
  before = rdtsc();
  usleep(USECS_PER_SEC);
  after = rdtsc();
  
  rdtsc_per_sec = (after - before);
}

double monotonic_time()
{
  return (double)rdtsc() / (double)rdtsc_per_sec;
}

#endif
