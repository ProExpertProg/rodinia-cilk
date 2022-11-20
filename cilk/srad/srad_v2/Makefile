# C compiler
CXX = clang++
CC_FLAGS = -g -fopencilk -O2 $(EXTRA_CFLAGS)
LD_FLAGS = -fopencilk $(EXTRA_LDFLAGS)

srad: 
	$(CXX) $(CC_FLAGS) $(LD_FLAGS) srad.cpp -o srad

clean:
	rm -f srad
