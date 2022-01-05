# CC = gcc
# CCFLAGS = -g -fopenmp -O3
CC = clang
CILKFLAG = -fopencilk
CCFLAGS = -g $(CILKFLAG) -O3 $(EXTRA_CFLAGS)
OUTPUT = *.out

3D:
	$(CC) -o 3D $(CCFLAGS) 3D.c -lm $(EXTRA_LDFLAGS) 

clean:
	rm -f 3D $(OUTPUT)
