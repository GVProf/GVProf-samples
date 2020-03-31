rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./histogram
hpcstruct --gpucfg yes  hpctoolkit-histogram-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./histogram &> log 
hpcstruct ./histogram
hpcprof -S ./histogram.hpcstruct hpctoolkit-histogram-measurements/
