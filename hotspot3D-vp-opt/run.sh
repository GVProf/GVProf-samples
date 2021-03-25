rm -rf hpctoolkit-* *csv log
# time ./heartwall ../../data/heartwall/test.avi 20
hpcrun -o hpctoolkit-hotspot3D-measurements -e gpu=nvidia ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out
hpcstruct --gpucfg yes  hpctoolkit-hotspot3D-measurements
time hpcrun -o hpctoolkit-hotspot3D-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out &>log
# hpcstruct ./srad
# hpcprof -S ./srad.hpcstruct hpctoolkit-srad-measurements/
