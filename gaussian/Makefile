# CUDA toolkit installation path
CUDA_DIR ?= /usr/local/cuda

# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
     	CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif


CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include
NVCCFLAG := -g -O3 -arch sm_70 -lineinfo

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(NVCCFLAG) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm gaussian
