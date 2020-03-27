
hpcrun -e gpu=nvidia ./pathfinder 100000 100 20
hpcstruct --gpucfg yes  hpctoolkit-pathfinder-measurements
hpcrun -e gpu=nvidia,sanitizer ./pathfinder 100000 100 20 &> log 
hpcstruct ./pathfinder
hpcprof -S ./pathfinder.hpcstruct hpctoolkit-pathfinder-measurements/
