rm -rf hpctoolkit-* *csv log
hpcrun -e gpu=nvidia ./dxtc
hpcstruct --gpucfg yes  hpctoolkit-dxtc-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10   -e gpu=nvidia,sanitizer ./dxtc &>log
hpcstruct ./dxtc
hpcprof -S ./dxtc.hpcstruct hpctoolkit-dxtc-measurements/