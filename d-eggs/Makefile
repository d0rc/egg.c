NVCC = nvcc
CXX = g++
CFLAGS = -O3 -Iinclude
NVCCFLAGS = -O3 -Iinclude -arch=sm_70 -lcublas -rdc=true

all: coordinator worker

coordinator: src/coordinator.cpp
	$(CXX) $(CFLAGS) -o coordinator src/coordinator.cpp -lpthread

worker: src/worker.cu
	$(NVCC) $(NVCCFLAGS) -o worker src/worker.cu

clean:
	rm -f coordinator worker
