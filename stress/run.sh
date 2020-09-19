#!/bin/bash

export NUM_CONTEXTS=4
export NUM_STREAMS_PER_CONTEXT=4
export HPCTOOLKIT_GPU_TEST_REP=4
export OMP_NUM_THREADS=$(expr "$NUM_CONTEXTS" '*' "$NUM_STREAMS_PER_CONTEXT")

rm -rf hpctoolkit-* *csv log
make
time ./main
hpcrun -e gpu=nvidia -o hpctoolkit-main-measurements ./main
hpcstruct --gpucfg yes hpctoolkit-main-measurements
time hpcrun -o hpctoolkit-main-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,redundancy ./main
hpcstruct ./main
hpcprof -S ./main.hpcstruct hpctoolkit-main-measurements/
