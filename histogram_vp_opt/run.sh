rm -rf hpctoolkit-* log *csv
time ./histogram
hpcrun -o hpctoolkit-histogram-measurements -e gpu=nvidia ./histogram
hpcstruct --gpucfg yes hpctoolkit-histogram-measurements
time hpcrun -o hpctoolkit-histogram-measurements -ck HPCRUN_SANITIZER_DEFAULT_TYPE=INT -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./histogram &> log 
hpcstruct ./histogram
hpcprof -S ./histogram.hpcstruct hpctoolkit-histogram-measurements/
