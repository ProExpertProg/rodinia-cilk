# C compiler
CXX = clang++
ICC = icc
CILKFLAG = -fopencilk
CC_FLAGS = -g $(CILKFLAG) -O2 -fopenmp $(EXTRA_CFLAGS)  # -fopenmp included for pragma omp simd
# OFFLOAD_CC_FLAGS = -offload-option,mic,compiler,"-no-opt-prefetch"

all: hotspot # hotspot_offload 


hotspot: hotspot_cilk.cpp Makefile 
	$(CXX) $(CC_FLAGS) hotspot_cilk.cpp $(EXTRA_LDFLAGS) -o hotspot

# hotspot_offload: hotspot_openmp.cpp Makefile
# 	$(ICC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD hotspot_openmp.cpp -o hotspot_offload

clean:
	rm -f hotspot hotspot_offload
