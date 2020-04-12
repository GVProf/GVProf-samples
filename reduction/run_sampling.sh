rm -rf hpctoolkit-* log *csv
time ./reduction --type=double --n=1024
hpcrun -o hpctoolkit-reduction-measurements -e gpu=nvidia ./reduction --type=double --n=1024
hpcstruct --gpucfg yes hpctoolkit-reduction-measurements
time hpcrun -o hpctoolkit-reduction-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer@50 ./reduction --type=double --n=1024 &>log
hpcstruct ./reduction
hpcprof -S ./reduction.hpcstruct hpctoolkit-reduction-measurements/
