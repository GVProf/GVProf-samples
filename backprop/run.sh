#!/bin/bash
hpcrun -e gpu=nvidia ./backprop 65536
hpcstruct --gpucfg yes  hpctoolkit-backprop-measurements
hpcrun -e gpu=nvidia,sanitizer ./backprop 65536 &> log