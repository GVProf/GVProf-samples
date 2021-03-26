
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

GPU_ARCH ?=

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o 
	nvcc main.o ./AVI/avilib.o ./AVI/avimod.o -I/usr/local/cuda/include -lm -o heartwall -O3 -g -lineinfo

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	nvcc $(GPU_ARCH) $(OUTPUT) $(KERNEL_DIM) main.cu -I./AVI -c -O3 -g -lineinfo

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
