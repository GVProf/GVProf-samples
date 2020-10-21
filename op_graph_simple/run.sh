#!/bin/bash

M=hpctoolkit-main-measurements
rm -rf $M
echo $M
hpcrun -e gpu=nvidia ./main
hpcstruct --gpucfg yes $M
rm -rf $M/*.hpcrun
hpcrun -e gpu=nvidia,value_flow ./main
hpcstruct ./main
hpcprof -S ./main.hpcstruct $M
