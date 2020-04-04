rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./kmeans -o -i ../data/kdd_cup
hpcstruct --gpucfg yes  hpctoolkit-kmeans-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30 -e gpu=nvidia,sanitizer ./kmeans -o -i ../data/kdd_cup &>log
hpcstruct ./kmeans 
hpcprof -S ./kmeans.hpcstruct hpctoolkit-kmeans-measurements/
