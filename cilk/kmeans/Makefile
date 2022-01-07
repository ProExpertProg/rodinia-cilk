all: CILK/kmeans # SERIAL/kmeans

CILK/kmeans:
	cd kmeans_cilk; make

SERIAL/kmeans:
	cd kmeans_serial; make

clean:
	cd kmeans_serial; make clean
	cd kmeans_cilk; make clean
