# C compiler
CC = gcc
CC_FLAGS = -g -O3

# CUDA compiler
NVCC = nvcc
GPU_ARCH ?=
NVCC_FLAGS = -lineinfo $(GPU_ARCH) -g -O3

backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o 
	$(NVCC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop -lcuda -lcudart -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c

clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo *.qdrep *.sqlite tmp*
