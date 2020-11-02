rm -rf hpctoolkit-* *csv log
time ./bfs ../data/graph1MW_6.txt
hpcrun -e gpu=nvidia -o hpctoolkit-bfs-measurements ./bfs ../data/graph1MW_6.txt
hpcstruct --gpucfg yes hpctoolkit-bfs-measurements
time hpcrun -o hpctoolkit-bfs-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,redundancy ./bfs ../data/graph1MW_6.txt &> log
hpcstruct ./bfs
hpcprof -S ./bfs.hpcstruct hpctoolkit-bfs-measurements/
