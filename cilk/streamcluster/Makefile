PREFIX=${PARSECDIR}/pkgs/kernels/streamcluster/inst/${PARSECPLAT}

TARGET_C = sc_cpu

TARGET_O = sc_cilk

ifdef version
  ifeq "$(version)" "parallel"
    CXXFLAGS :=	$(CXXFLAGS) -DENABLE_THREADS -pthread
  endif
endif

CXX = clang++
CXXFLAGS = -O3 -fopencilk $(EXTRA_CFLAGS)

all: cpu cilk

cpu:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) streamcluster_original.cpp -o $(TARGET_C) $(LIBS) -DENABLE_THREADS -pthread

cilk:
	$(CXX) $(CXXFLAGS) -o $(TARGET_O) streamcluster_cilk.cpp $(EXTRA_LDFLAGS)

omp:
	$(CXX) -O3 -fopenmp -o sc_omp streamcluster_omp.cpp


clean:
	rm -f *.o *~ *.txt $(TARTET_C) $(TARGET_O) sc_omp
