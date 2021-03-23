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
	cd cfd-vp-opt2; make;  cd ..;

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
	cd cfd-vp-opt2; make clean;  cd ..;
