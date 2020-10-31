rm -rf hpctoolkit-* *csv log
# time ./heartwall ../../data/heartwall/test.avi 20
hpcrun -o hpctoolkit-lud_cuda-measurements -e gpu=nvidia ./lud_cuda -s 256 -v
hpcstruct --gpucfg yes  hpctoolkit-lud_cuda-measurements
time hpcrun -o hpctoolkit-lud_cuda-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./lud_cuda -s 256 -v &>log
# hpcstruct ./srad
# hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
