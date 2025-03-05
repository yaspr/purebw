CC=gcc
CFLAGS=-Wall -Wextra -g
OFLAGS=-march=native -Ofast -fopenmp

purebw: purebw.c monotonic_timer.c kernels.c
	$(CC) -DWITH_OPENMP $(CFLAGS) $(OFLAGS) $^ -o $@

.PHONY: run

run: purebw
	./purebw

clean:
	rm -Rf purebw

