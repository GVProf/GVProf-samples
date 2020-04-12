rm -rf hpctoolkit-* *csv log
time ./dct8x8
hpcrun -e gpu=nvidia -o hpctoolkit-dct8x8-measurements ./dct8x8
hpcstruct --gpucfg yes hpctoolkit-dct8x8-measurements
time hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -o hpctoolkit-dct8x8-measurements -e gpu=nvidia,sanitizer@50 ./dct8x8 &>log
hpcstruct ./dct8x8
hpcprof -S ./dct8x8.hpcstruct hpctoolkit-dct8x8-measurements/
