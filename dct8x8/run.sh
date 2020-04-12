rm -rf hpctoolkit-* *csv log
time ./dct8x8
hpcrun -o hpctoolkit-dct8x8-measurements -e gpu=nvidia ./dct8x8
hpcstruct --gpucfg yes hpctoolkit-dct8x8-measurements
time hpcrun -o hpctoolkit-dct8x8-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30 -e gpu=nvidia,sanitizer ./dct8x8 &>log
hpcstruct ./dct8x8
hpcprof -S ./dct8x8.hpcstruct hpctoolkit-dct8x8-measurements/
