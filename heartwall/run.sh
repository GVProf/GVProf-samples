rm -rf hpctoolkit-* *csv log
# time ./heartwall ../../data/heartwall/test.avi 20
hpcrun -o hpctoolkit-heartwall-measurements -e gpu=nvidia ./heartwall ../../data/heartwall/test.avi 20
hpcstruct --gpucfg yes  hpctoolkit-heartwall-measurements
time hpcrun -o hpctoolkit-heartwall-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./heartwall ../../data/heartwall/test.avi 20 &>log
# hpcstruct ./srad
# hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
