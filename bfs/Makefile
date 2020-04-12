# CUDA toolkit installation path
CUDA_DIR = /sw/summit/cuda/10.1.243

# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
     	CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif


CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include
NVCCFlag=-g -O2 -arch sm_70
SRC = bfs.cu

EXE = bfs

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  ${NVCCFlag}

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) ${NVCCFlag}

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) ${NVCCFlag}

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) ${NVCCFlag}

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
