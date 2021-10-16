
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

CC = clang
CFLAGS = -O3 -fopencilk $(EXTRA_CFLAGS)

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(CC) main.o ./AVI/avilib.o ./AVI/avimod.o -lm $(CFLAGS) -o heartwall

# compile main function file into object (binary)
main.o: main.c define.c kernel.c
	$(CC) $(OUTPUT) main.c -I./AVI -c $(CFLAGS)

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall
