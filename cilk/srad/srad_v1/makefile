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

# link objects(binaries) together
a.out:	main.o
	$(CC)	main.o \
			-lm $(CFLAGS) -o srad

# compile main function file into object (binary)
main.o: 	main.c \
				define.c \
				graphics.c
	$(CC)	main.c \
			-c $(CFLAGS)

# delete all object files
clean:
	rm *.o srad
