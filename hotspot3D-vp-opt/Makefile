CC := nvcc

SRC = 3D.cu

EXE = 3D 

OUTPUT = *.out

GPU_ARCH ?=

FLAGS = -g $(GPU_ARCH) -O3 -lineinfo #--ptxas-options=-v

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) $(SRC) -o $(EXE)

enum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -deviceemu $(SRC) -o $(EXE)

debug: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -g $(SRC) -o $(EXE)

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -g -deviceemu $(SRC) -o $(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) *.qdrep *.sqlite tmp*
