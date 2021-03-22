CC := nvcc

GPU_ARCH ?= 
NVCCFlag=-g -O3 $(GPU_ARCH) -lineinfo
SRC = bfs.cu

EXE = bfs

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) ${NVCCFlag}

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) ${NVCCFlag}

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) ${NVCCFlag}

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) ${NVCCFlag}

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt *.qdrep *.sqlite tmp*
