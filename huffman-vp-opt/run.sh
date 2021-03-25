rm -rf hpctoolkit-* *csv log
# time ./pavle ../../data/pavle/test.avi 20
hpcrun -o hpctoolkit-pavle-measurements -e gpu=nvidia ./pavle ../../data/huffman/test1024_H2.206587175259.in 
hpcstruct --gpucfg yes  hpctoolkit-pavle-measurements
time hpcrun -o hpctoolkit-pavle-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./pavle ../../data/huffman/test1024_H2.206587175259.in  &>log
# hpcstruct ./srad
# hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
