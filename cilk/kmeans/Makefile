all: OPENMP/kmeans SERIAL/kmeans CILK/kmeans

OPENMP/kmeans:
	make -C kmeans_openmp
	
SERIAL/kmeans:
	make -C kmeans_openmp

CILK/kmeans:
	make -C kmeans_cilk

clean:
	make -C kmeans_serial clean
	make -C kmeans_openmp clean
	make -C kmeans_cilk clean

