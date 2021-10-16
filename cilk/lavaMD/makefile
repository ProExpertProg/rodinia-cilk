# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# C_C = gcc
CC = clang
C_C = $(CC)
# OMP_LIB = -lgomp
# OMP_FLAG = -fopenmp
CILK_LIB = -fopencilk
CILK_FLAG = -fopencilk

# link objects (binaries) together
a.out:		main.o \
			./kernel/kernel_cpu.o \
			./util/num/num.o \
			./util/timer/timer.o
	$(C_C)	main.o \
			./kernel/kernel_cpu.o \
			./util/num/num.o \
			./util/timer/timer.o \
			-lm \
			$(CILK_LIB) $(EXTRA_CFLAGS) \
			-o lavaMD
# compile function files into objects (binaries)
main.o:		main.h \
			main.c \
			./kernel/kernel_cpu.h \
			./kernel/kernel_cpu.c \
			./util/num/num.h \
			./util/num/num.c \
			./util/timer/timer.h \
			./util/timer/timer.c
	$(C_C)	${OUTPUT} main.c \
			-c \
			-o main.o \
			-O3 $(EXTRA_CFLAGS)

./kernel/kernel_cpu.o:	./kernel/kernel_cpu.h \
						./kernel/kernel_cpu.c
	$(C_C)				./kernel/kernel_cpu.c \
						-c \
						-o ./kernel/kernel_cpu.o \
						-O3 \
						$(CILK_FLAG) $(EXTRA_CFLAGS)

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(C_C)			./util/num/num.c \
					-c \
					-o ./util/num/num.o \
					-O3 $(EXTRA_CFLAGS)

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C)				./util/timer/timer.c \
						-c \
						-o ./util/timer/timer.o \
						-O3 $(EXTRA_CFLAGS)

# delete all object and executable files
clean:
	rm	*.o \
		./kernel/*.o \
		./util/num/*.o \
		./util/timer/*.o \
                ./util/device/*.o \
		lavaMD
