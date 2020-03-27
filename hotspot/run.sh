#!/bin/bash
hpcrun -e gpu=nvidia ./hotspot 512 2 2 ../data/temp_512 ../data/power_512 output.out
hpcstruct --gpucfg yes  hpctoolkit-hotspot-measurements
hpcrun -e gpu=nvidia,sanitizer ./hotspot 512 2 2 ../data/temp_512 ../data/power_512 output.out &> log 