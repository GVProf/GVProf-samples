
hpcrun -e gpu=nvidia ./srad 1 0.5 502 458
hpcstruct --gpucfg yes  hpctoolkit-srad-measurements
hpcrun -e gpu=nvidia,sanitizer ./srad 1 0.5 502 458 &> log 