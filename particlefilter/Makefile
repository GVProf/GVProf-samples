#makefile

CUDA_DIR = /scratch/opt/cuda-11.0.3/
CUDA_LIB_DIR := $(CUDA_DIR)/lib64

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

all: naive float

naive: ex_particle_CUDA_naive_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lineinfo -lm -O3 -use_fast_math -arch sm_70 ex_particle_CUDA_naive_seq.cu -o particlefilter_naive
	
float: ex_particle_CUDA_float_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lineinfo -lm -O3 -use_fast_math -arch sm_70 ex_particle_CUDA_float_seq.cu -o particlefilter_float

clean:
	rm particlefilter_naive particlefilter_float
