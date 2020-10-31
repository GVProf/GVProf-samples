CUDA_DIR = /usr/local/cuda

# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
     	CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif

NVCC_FLAG := -g -lineinfo 

COMMON_INC=../common/inc
COMMON_LIB=../common/lib
architecture=sm_70




all: euler3d euler3d_double  pre_euler3d  pre_euler3d_double 

euler3d: euler3d.cu
	nvcc $(KERNEL_DIM) -O2 -Xptxas -v --gpu-architecture=${architecture}  ${NVCC_FLAG} euler3d.cu -o euler3d -I${COMMON_INC}  -L${COMMON_LIB}

euler3d_double: euler3d_double.cu
	nvcc -Xptxas -v -O3 --gpu-architecture=${architecture} ${NVCC_FLAG}  euler3d_double.cu -o euler3d_double -I${COMMON_INC}  -L${COMMON_LIB} 


pre_euler3d: pre_euler3d.cu
	nvcc -Xptxas -v -O3 --gpu-architecture=${architecture} ${NVCC_FLAG}  pre_euler3d.cu -o pre_euler3d -I${COMMON_INC}  -L${COMMON_LIB} 

pre_euler3d_double: pre_euler3d_double.cu
	nvcc -Xptxas -v -O3 --gpu-architecture=${architecture} ${NVCC_FLAG}  pre_euler3d_double.cu -o pre_euler3d_double -I${COMMON_INC}  -L${COMMON_LIB} 


clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
