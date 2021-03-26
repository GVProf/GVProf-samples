CC := nvcc

GPU_ARCH ?=

NVCCFLAGS := $(GPU_ARCH) -g -lineinfo -O3

SRC = pathfinder.cu

EXE = pathfinder

release:
	$(CC) $(SRC) -o $(EXE) $(NVCCFLAGS)

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE)  $(NVCCFLAGS)

debug:
	$(CC) -g $(SRC) -o $(EXE) $(NVCCFLAGS)

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) $(NVCCFLAGS)

clean:
	rm -f pathfinder *.qdrep *.sqlite tmp*
