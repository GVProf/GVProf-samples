rm -rf hpctoolkit-* *csv log
time ./dwt2d 192.bmp -d 192x192 -f -5 -l 3
hpcrun -o hpctoolkit-dwt2d-measurements -e gpu=nvidia ./dwt2d 192.bmp -d 192x192 -f -5 -l 3
hpcstruct --gpucfg yes hpctoolkit-dwt2d-measurements
time hpcrun -o hpctoolkit-dwt2d-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer@50 ./dwt2d 192.bmp -d 192x192 -f -5 -l 3 &>log
hpcstruct ./dwt2d 
hpcprof -S ./dwt2d.hpcstruct hpctoolkit-dwt2d-measurements/
