#makefile

CC := nvcc

GPU_ARCH ?=

all: naive float

naive: ex_particle_CUDA_naive_seq.cu
	$(CC) $(GPU_ARCH) -lcuda -g -lineinfo -lm -O3 -use_fast_math ex_particle_CUDA_naive_seq.cu -o particlefilter_naive
	
float: ex_particle_CUDA_float_seq.cu
	$(CC) $(GPU_ARCH) -lcuda -g -lineinfo -lm -O3 -use_fast_math ex_particle_CUDA_float_seq.cu -o particlefilter_float

clean:
	rm particlefilter_naive particlefilter_float *.qdrep *.sqlite tmp*
