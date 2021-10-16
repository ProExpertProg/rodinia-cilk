#makefile
CC = clang
CFLAGS = -O3 -ffast-math -fopencilk $(EXTRA_CFLAGS)

cilk: ex_particle_Cilk_seq.c
	$(CC) $(CFLAGS) ex_particle_Cilk_seq.c -o particle_filter -lm 


# icc_openmp: ex_particle_Cilk_seq.c
# 	icc -O3 -fast -openmp ex_particle_OPENMP_seq.c -o particle_filter


clean:
	rm particle_filter
