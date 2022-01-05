CXX = g++
FLAGS := -O3 -Wno-write-strings

all: hotspotex hotspotver

hotspotex: hotspotex.cpp
	$(CXX) $(FLAGS) -o $@ $<

hotspotver: hotspotver.cpp
	$(CXX) $(FLAGS) -o $@ $<

clean:
	rm hotspotex hotspotver

