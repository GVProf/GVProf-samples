rm -rf hpctoolkit-* *csv log
hpcrun -e gpu=nvidia ./srad 1 0.5 502 458
hpcstruct --gpucfg yes  hpctoolkit-srad-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30 -e gpu=nvidia,sanitizer ./srad 1 0.5 502 458 &>log
hpcstruct ./srad
hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
