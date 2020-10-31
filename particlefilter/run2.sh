rm -rf hpctoolkit-* *csv log
# time ./particlefilter_float ../../data/particlefilter_float/test.avi 20
hpcrun -o hpctoolkit-particlefilter_float-measurements -e gpu=nvidia ./particlefilter_float -x 128 -y 128 -z 10 -np 1000
hpcstruct --gpucfg yes  hpctoolkit-particlefilter_float-measurements
time hpcrun -o hpctoolkit-particlefilter_float-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./particlefilter_float -x 128 -y 128 -z 10 -np 1000 &>log
# hpcstruct ./srad
# hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
