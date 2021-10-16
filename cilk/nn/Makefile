CC = clang
LOCAL_CC = $(CC) -g -O3 -Wall
CFLAGS = -lm -fopencilk -Wall


all : nn

clean :
	rm -rf *.o nn

nn : nn_cilk.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) 

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

