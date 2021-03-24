# CUDA toolkit installation path

CC := nvcc

GPU_ARCH ?=

NVCCFLAG := -g -O3 $(GPU_ARCH) -lineinfo

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(NVCCFLAG) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm gaussian
