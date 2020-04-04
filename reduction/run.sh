rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./reduction --type=double --n=1024
hpcstruct --gpucfg yes  hpctoolkit-reduction-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30   -e gpu=nvidia,sanitizer ./reduction --type=double --n=1024 &>log
hpcstruct ./reduction
hpcprof -S ./reduction.hpcstruct hpctoolkit-reduction-measurements/
