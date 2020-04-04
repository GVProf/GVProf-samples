rm -rf hpctoolkit-* *csv log
hpcrun -e gpu=nvidia ./pathfinder 100000 100 20
hpcstruct --gpucfg yes  hpctoolkit-pathfinder-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30 -e gpu=nvidia,sanitizer ./pathfinder 100000 100 20 &> log 
hpcstruct ./pathfinder
hpcprof -S ./pathfinder.hpcstruct hpctoolkit-pathfinder-measurements/
