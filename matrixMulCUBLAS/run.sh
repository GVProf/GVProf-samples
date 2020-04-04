rm -rf hpctoolkit-* *csv log
hpcrun -e gpu=nvidia ./matrixMulCUBLAS
hpcstruct --gpucfg yes -j8 hpctoolkit-matrixMulCUBLAS-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10   -e gpu=nvidia,sanitizer ./matrixMulCUBLAS &>log
hpcstruct ./matrixMulCUBLAS
hpcprof -S ./matrixMulCUBLAS.hpcstruct hpctoolkit-matrixMulCUBLAS-measurements/
