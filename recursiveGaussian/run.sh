rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./recursiveGaussian --file=lena_10.ppm
hpcstruct --gpucfg yes  hpctoolkit-recursiveGaussian-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10   -e gpu=nvidia,sanitizer ./recursiveGaussian --file=lena_10.ppm &>log
hpcstruct ./recursiveGaussian 
hpcprof -S ./recursiveGaussian.hpcstruct hpctoolkit-recursiveGaussian-measurements/
