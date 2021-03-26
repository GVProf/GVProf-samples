rm -rf hpctoolkit-* *csv log
# time ./sc_gpu ../../data/sc_gpu/test.avi 20
hpcrun -o hpctoolkit-sc_gpu-measurements -e gpu=nvidia ./sc_gpu 10 20 256 65536 65536 1000 none output.txt 1
hpcstruct --gpucfg yes  hpctoolkit-sc_gpu-measurements
time hpcrun -o hpctoolkit-sc_gpu-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./sc_gpu 10 20 256 65536 65536 1000 none output.txt 1 &>log
# hpcstruct ./srad
# hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
