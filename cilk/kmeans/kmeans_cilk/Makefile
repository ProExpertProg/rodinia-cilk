# C compiler
# CC = gcc
# CC_FLAGS = -g -fopenmp -O2 
CC = clang
CC_FLAGS = -g -fopencilk -O2 $(EXTRA_CFLAGS)
LD_FLAGS = -fopencilk $(EXTRA_LDFLAGS)

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o
	$(CC) $(LD_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o  -o kmeans

kmeans_reducers: cluster.o getopt.o kmeans.o kmeans_clustering_reducers.o
	$(CC) $(LD_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering_reducers.o  -o kmeans_reducers

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c

getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c

kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) kmeans.c -c

kmeans_clustering.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c

kmeans_clustering_reducers.o: kmeans_clustering_reducers.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering_reducers.c -c

clean:
	rm -f *.o *~ kmeans kmeans_reducers
