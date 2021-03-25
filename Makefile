all:
	cd backprop; make;  cd ..;\
	cd backprop-vp-opt; make;  cd ..;\
	cd backprop-vp-opt1; make;  cd ..;\
	cd backprop-vp-opt2; make;  cd ..; \
	cd bfs; make;  cd ..;\
	cd bfs-vp-opt; make;  cd ..;\
	cd bfs-vp-opt1; make;  cd ..;\
	cd bfs-vp-opt2; make;  cd ..; \
	cd b+tree; make;  cd ..; \
	cd cfd; make;  cd ..;\
	cd cfd-vp-opt; make;  cd ..;\
	cd cfd-vp-opt1; make;  cd ..;\
	cd cfd-vp-opt2; make;  cd ..; \
	cd dct8x8; make;  cd ..; \
	cd dwt2d; make;  cd ..; \
	cd dxtc; make;  cd ..; \
	cd gaussian; make;  cd ..; \
	cd heartwall; make;  cd ..;\
	cd histogram; make;  cd ..; \
	cd hotspot; make;  cd ..; \
	cd hotspot-vp-opt; make;  cd ..; \
	cd hotspot3D; make;  cd ..; \
	cd hotspot3D-vp-opt; make;  cd ..;

clean:
	cd backprop; make clean;  cd ..;\
	cd backprop-vp-opt; make clean;  cd ..;\
	cd backprop-vp-opt1; make clean;  cd ..;\
	cd backprop-vp-opt2; make clean;  cd ..; \
	cd bfs; make clean;  cd ..;\
	cd bfs-vp-opt; make clean;  cd ..;\
	cd bfs-vp-opt1; make clean;  cd ..;\
	cd bfs-vp-opt2; make clean;  cd ..; \
	cd b+tree; make clean;  cd ..; \
	cd cfd; make clean;  cd ..;\
	cd cfd-vp-opt; make clean;  cd ..;\
	cd cfd-vp-opt1; make clean;  cd ..;\
	cd cfd-vp-opt2; make clean;  cd ..; \
	cd dct8x8; make clean;  cd ..; \
	cd dwt2d; make clean;  cd ..; \
	cd dxtc; make clean;  cd ..; \
	cd gaussian; make clean;  cd ..; \
	cd heartwall; make clean;  cd ..; \
	cd histogram; make clean;  cd ..; \
	cd hotspot; make clean;  cd ..; \
	cd hotspot-vp-opt; make clean;  cd ..; \
	cd hotspot3D; make clean;  cd ..; \
	cd hotspot3D-vp-opt; make clean;  cd ..;
