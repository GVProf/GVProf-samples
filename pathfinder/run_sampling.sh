rm -rf hpctoolkit-* *csv log
time  ./pathfinder 100000 100 20
hpcrun -o hpctoolkit-pathfinder-measurements -e gpu=nvidia ./pathfinder 100000 100 20
hpcstruct --gpucfg yes hpctoolkit-pathfinder-measurements
time hpcrun -o hpctoolkit-pathfinder-measurements -ck HPCRUN_SANITIZER_DEFAULT_TYPE=INT -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer@50 ./pathfinder 100000 100 20 &> log 
hpcstruct ./pathfinder
hpcprof -S ./pathfinder.hpcstruct hpctoolkit-pathfinder-measurements/
