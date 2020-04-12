rm -rf hpctoolkit-* *csv log
hpcrun -e gpu=nvidia ./srad 2048 2048 0 127 0 127 0.5 2
hpcstruct --gpucfg yes  hpctoolkit-srad-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10   -e gpu=nvidia,sanitizer ./srad 2048 2048 0 127 0 127 0.5 2 &>log
hpcstruct ./srad
hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
