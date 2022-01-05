
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

CC = clang
CILKFLAG = -fopencilk
CFLAGS = -O3 $(CILKFLAG) $(EXTRA_CFLAGS)

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(CC) main.o ./AVI/avilib.o ./AVI/avimod.o -lm $(CFLAGS) $(EXTRA_LDFLAGS) -o heartwall

# compile main function file into object (binary)
main.o: main.c define.c kernel.c
	$(CC) $(OUTPUT) main.c -I./AVI -c $(CFLAGS)

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall
