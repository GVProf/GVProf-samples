#!/bin/bash
rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./vectorAdd
hpcstruct --gpucfg yes hpctoolkit-vectorAdd-measurements/
hpcrun -d -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30 -e  gpu=nvidia,sanitizer ./vectorAdd 
hpcstruct ./vectorAdd
hpcprof -S ./vectorAdd.hpcstruct hpctoolkit-vectorAdd-measurements/