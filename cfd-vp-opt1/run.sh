rm -rf hpctoolkit-* *csv log
time ./euler3d ../data/fvcorr.domn.097K
hpcrun -o hpctoolkit-euler3d-measurements -e gpu=nvidia ./euler3d ../data/fvcorr.domn.097K
hpcstruct --gpucfg yes hpctoolkit-euler3d-measurements
time hpcrun -o hpctoolkit-euler3d-measurements -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10 -e gpu=nvidia,sanitizer ./euler3d ../data/fvcorr.domn.097K &>log
hpcstruct ./euler3d
hpcprof -S ./euler3d.hpcstruct hpctoolkit-euler3d-measurements/
