NVCC = nvcc
CXX = g++
CFLAGS = -O3 -Iinclude $(EXTRA_FLAGS)
NVCCFLAGS = -O3 -Iinclude -arch=sm_86 -lcublas -rdc=true $(EXTRA_FLAGS)

all: coordinator worker print_arch

print_arch: print_arch.cpp
	$(CXX) $(CFLAGS) -o print_arch print_arch.cpp

coordinator: src/coordinator.cpp
	$(CXX) $(CFLAGS) -o coordinator src/coordinator.cpp -lpthread

worker: src/worker.cu
	$(NVCC) $(NVCCFLAGS) -o worker src/worker.cu

clean:
	rm -f coordinator worker print_arch
