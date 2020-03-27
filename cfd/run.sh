rm -rf hpctoolkit-* *csv
hpcrun -e gpu=nvidia ./euler3d ../data/fvcorr.domn.097K
hpcstruct --gpucfg yes  hpctoolkit-euler3d-measurements
hpcrun -ck HPCRUN_SANITIZER_MEM_VIEWS=10 -ck HPCRUN_SANITIZER_PC_VIEWS=10   -e gpu=nvidia,sanitizer ./euler3d ../data/fvcorr.domn.097K
hpcstruct ./euler3d
hpcprof -S ./euler3d.hpcstruct hpctoolkit-euler3d-measurements/
