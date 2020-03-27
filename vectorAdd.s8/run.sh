rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./vectorAdd
hpcstruct --gpucfg yes  hpctoolkit-vectorAdd-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10   -e gpu=nvidia,sanitizer ./vectorAdd