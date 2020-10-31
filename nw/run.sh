rm -rf hpctoolkit-* *csv log
# time ./needle ../../data/needle/test.avi 20
hpcrun -o hpctoolkit-needle-measurements -e gpu=nvidia ./needle 2048 10
hpcstruct --gpucfg yes  hpctoolkit-needle-measurements
time hpcrun -o hpctoolkit-needle-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./needle 2048 10 &>log
# hpcstruct ./srad
# hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
