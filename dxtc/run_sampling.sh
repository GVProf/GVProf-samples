rm -rf hpctoolkit-* *csv log
time ./dxtc ../data
hpcrun -o hpctoolkit-dxtc-measurements -e gpu=nvidia ./dxtc ../data
hpcstruct --gpucfg yes hpctoolkit-dxtc-measurements
time hpcrun -o hpctoolkit-dxtc-measurements  -ck HPCRUN_SANITIZER_DEFAULT_TYPE=INT -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=30 -e gpu=nvidia,sanitizer@50 ./dxtc ../data &>log
hpcstruct ./dxtc
hpcprof -S ./dxtc.hpcstruct hpctoolkit-dxtc-measurements/
