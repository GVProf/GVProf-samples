LOCAL_CC = gcc -g -O3 -Wall

GPU_ARCH ?=

NVCCFLGAS = $(GPU_ARCH) -g -O3 -lineinfo

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.cu
	$(CC) -cuda nn_cuda.cu $(NVCCFLAGS)
	$(CC) -o nn nn_cuda.cu $(NVCCFLAGS)

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
