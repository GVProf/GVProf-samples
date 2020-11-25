CUDA_DIR ?= /usr/local/cuda/
CUDA_LIB_DIR := $(CUDA_DIR)/lib64

CC := $(CUDA_DIR)/bin/nvcc
CC_FLAG = -g -O3 -lineinfo
INCLUDE := $(CUDA_DIR)/include

SRC = needle.cu

EXE = needle

release: $(SRC)
	$(CC) ${KERNEL_DIM} $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  $(CC_FLAG)

enum: $(SRC)
	$(CC) ${KERNEL_DIM} -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(CC_FLAG)

debug: $(SRC)
	$(CC) ${KERNEL_DIM} -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(CC_FLAG)

debugenum: $(SRC)
	$(CC) ${KERNEL_DIM} -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(CC_FLAG)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
