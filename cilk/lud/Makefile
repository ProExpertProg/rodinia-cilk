all: lud_cilk # lud_omp_offload

lud_cilk:
	cd cilk; make

# lud_omp_offload:
# 	cd omp; make -f Makefile.offload

clean:
	cd cilk; make clean ; make -f Makefile.offload clean
