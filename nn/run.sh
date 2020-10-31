rm -rf hpctoolkit-* *csv log
# time ./nn ../../data/nn/test.avi 20
hpcrun -o hpctoolkit-nn-measurements -e gpu=nvidia ./nn filelist_4 -r 5 -lat 30 -lng 90
hpcstruct --gpucfg yes  hpctoolkit-nn-measurements
time hpcrun -o hpctoolkit-nn-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./nn filelist_4 -r 5 -lat 30 -lng 90 &>log
# hpcstruct ./srad
# hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
