rm -rf hpctoolkit-* *csv log
hpcrun -e gpu=nvidia ./matrixMulCUBLAS
hpcstruct --gpucfg yes -j12 hpctoolkit-matrixMulCUBLAS-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=30 -ck HPCRUN_SANITIZER_PC_VIEWS=30   -e gpu=nvidia,sanitizer ./matrixMulCUBLAS &>log
hpcstruct ./matrixMulCUBLAS
hpcprof -S ./matrixMulCUBLAS.hpcstruct hpctoolkit-matrixMulCUBLAS-measurements/
