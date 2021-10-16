# C compiler
CXX = g++
# ICC = icc
CC_FLAGS = -g -fopencilk -O2 $(EXTRA_CFLAGS)

all: bfs # bfs_offload

bfs: bfs.cpp 
	$(CXX) $(CC_FLAGS) bfs.cpp -o bfs 

# bfs_offload: bfs.cpp
# 	$(ICC) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

clean:
	rm -f bfs bfs_offload result.txt
