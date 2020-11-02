#!/bin/bash
rm -rf hpctoolkit-* log *.csv
hpcrun -e gpu=nvidia -o hpctoolkit-hotspot-measurements ./hotspot 512 2 2 ../data/temp_512 ../data/power_512 output.out
time ./hotspot 512 2 2 ../data/temp_512 ../data/power_512 output.out
hpcstruct --gpucfg yes hpctoolkit-hotspot-measurements
time hpcrun -o hpctoolkit-hotspot-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./hotspot 512 2 2 ../data/temp_512 ../data/power_512 output.out &> log 
hpcstruct ./hotspot
hpcprof -S ./hotspot.hpcstruct hpctoolkit-hotspot-measurements/
