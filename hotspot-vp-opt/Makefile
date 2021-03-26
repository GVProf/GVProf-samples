# CUDA toolkit installation path

CC := nvcc

GPU_ARCH ?=

NVCC_FLAG := -lineinfo -O3 -g $(GPU_ARCH)

SRC = hotspot.cu

EXE = hotspot

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) ${NVCC_FLAG}

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) ${NVCC_FLAG}

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) ${NVCC_FLAG}

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) ${NVCC_FLAG}

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt *.qdrep *.sqlite tmp*
