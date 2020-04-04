rm -rf hpctoolkit-* *csv log
hpcrun -e gpu=nvidia ./dwt2d 192.bmp -d 192x192 -f -5 -l 3
hpcstruct --gpucfg yes  hpctoolkit-dwt2d-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30 -e gpu=nvidia,sanitizer./dwt2d 192.bmp -d 192x192 -f -5 -l 3 &>log
hpcstruct ./dwt2d 
hpcprof -S ./dwt2d.hpcstruct hpctoolkit-dwt2d-measurements/
