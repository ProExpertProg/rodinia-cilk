# CC = g++
CXX = clang++
SRC = pathfinder.cpp
EXE = pathfinder
# FLAGS = -fopenmp
FLAGS = -fopencilk $(EXTRA_CFLAGS)

release:
	$(CXX) $(SRC) $(FLAGS) -o $(EXE)

debug:
	$(CXX) $(SRC) -g -Wall -o $(EXE)

clean:
	rm -f pathfinder


