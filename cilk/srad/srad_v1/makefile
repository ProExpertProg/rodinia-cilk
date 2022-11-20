# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

CC = clang
CFLAGS = -O3 -fopencilk $(EXTRA_CFLAGS)
LD_FLAGS = -lm -fopencilk $(EXTRA_LDFLAGS)

# link objects(binaries) together
a.out:	main.o
	$(CC)   main.o $(LD_FLAGS) -o srad

# compile main function file into object (binary)
main.o: 	main.c \
				define.c \
				graphics.c
	$(CC)	main.c \
			-c $(CFLAGS)

# delete all object files
clean:
	rm -f *.o srad
