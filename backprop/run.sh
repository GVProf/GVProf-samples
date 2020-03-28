#!/bin/bash
rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./backprop 65536
hpcstruct --gpucfg yes  hpctoolkit-backprop-measurements
hpcrun -e gpu=nvidia,sanitizer@2 ./backprop 65536 &> log
hpcstruct ./backprop
hpcprof -S ./backprop.hpcstruct hpctoolkit-backprop-measurements/