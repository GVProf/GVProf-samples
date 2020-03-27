rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./bfs ../data/graph1MW_6.txt
hpcstruct --gpucfg yes  hpctoolkit-bfs-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10   -e gpu=nvidia,sanitizer ./bfs ../data/graph1MW_6.txt
hpcstruct ./bfs
hpcprof -S ./bfs.hpcstruct hpctoolkit-bfs-measurements/
