
CC := nvcc
GPU_ARCH ?=
CC_FLAG = $(GPU_ARCH) -g -O3 -lineinfo

SRC = needle.cu

EXE = needle

release: $(SRC)
	$(CC) ${KERNEL_DIM} $(SRC) -o $(EXE) $(CC_FLAG)

enum: $(SRC)
	$(CC) ${KERNEL_DIM} -deviceemu $(SRC) -o $(EXE) $(CC_FLAG)

debug: $(SRC)
	$(CC) ${KERNEL_DIM} -g $(SRC) -o $(EXE) $(CC_FLAG)

debugenum: $(SRC)
	$(CC) ${KERNEL_DIM} -g -deviceemu $(SRC) -o $(EXE) $(CC_FLAG)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
