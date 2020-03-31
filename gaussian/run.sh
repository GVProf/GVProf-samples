rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./gaussian -f ../data/matrix4.txt
hpcstruct --gpucfg yes  hpctoolkit-gaussian-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10   -e gpu=nvidia,sanitizer ./gaussian -f ../data/matrix4.txt &>log
hpcstruct ./gaussian
hpcprof -S ./gaussian.hpcstruct hpctoolkit-gaussian-measurements/
