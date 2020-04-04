rm -rf hpctoolkit-* *csv log
hpcrun -e gpu=nvidia ./dct8x8
hpcstruct --gpucfg yes  hpctoolkit-dct8x8-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30   -e gpu=nvidia,sanitizer ./dct8x8 &>log
hpcstruct ./dct8x8
hpcprof -S ./dct8x8.hpcstruct hpctoolkit-dct8x8-measurements/
