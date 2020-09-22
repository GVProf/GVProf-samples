rm -rf hpctoolkit-* *csv log
make clean
make
time ./vectorAdd
hpcrun -e gpu=nvidia -o hpctoolkit-vectorAdd-measurements ./vectorAdd
hpcstruct --gpucfg yes hpctoolkit-vectorAdd-measurements
time hpcrun -o hpctoolkit-vectorAdd-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,redundancy ./vectorAdd &> log
hpcstruct ./vectorAdd
hpcprof -S ./vectorAdd.hpcstruct hpctoolkit-vectorAdd-measurements/
