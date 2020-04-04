#!/bin/bash
rm -rf hpctoolkit-* *csv log
hpcrun -e gpu=nvidia ./backprop 65536
hpcstruct --gpucfg yes  hpctoolkit-backprop-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30  -e gpu=nvidia,sanitizer ./backprop 65536 &> log
hpcstruct ./backprop
hpcprof -S ./backprop.hpcstruct hpctoolkit-backprop-measurements/
