#!/bin/bash
hpcrun -e gpu=nvidia ./hotspot 512 2 2 ../data/temp_512 ../data/power_512 output.out
hpcstruct --gpucfg yes  hpctoolkit-hotspot-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./hotspot 512 2 2 ../data/temp_512 ../data/power_512 output.out &> log 
hpcstruct ./hotspot
hpcprof -S ./hotspot.hpcstruct hpctoolkit-hotspot-measurements/
