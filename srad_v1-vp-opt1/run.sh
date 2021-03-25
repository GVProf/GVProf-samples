rm -rf hpctoolkit-* *csv log
time ./srad 1 0.5 502 458
hpcrun -o hpctoolkit-srad-measurements -e gpu=nvidia ./srad 1 0.5 502 458
hpcstruct --gpucfg yes  hpctoolkit-srad-measurements
time hpcrun -o hpctoolkit-srad-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./srad 1 0.5 502 458 &>log
hpcstruct ./srad
hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
