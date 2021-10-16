# CC = gcc
# CCFLAGS = -g -fopenmp -O3
CC = clang
CCFLAGS = -g -fopencilk -O3 $(EXTRA_CFLAGS)
OUTPUT = *.out

3D:
	$(CC) -o 3D $(CCFLAGS) 3D.c -lm 

clean:
	rm -f 3D $(OUTPUT)
