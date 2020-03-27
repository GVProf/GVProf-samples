
hpcrun -e gpu=nvidia ./histogram
hpcstruct --gpucfg yes  hpctoolkit-histogram-measurements
hpcrun -e gpu=nvidia,sanitizer ./histogram &> log 
hpcstruct ./histogram
hpcprof -S ./histogram.hpcstruct hpctoolkit-histogram-measurements/
