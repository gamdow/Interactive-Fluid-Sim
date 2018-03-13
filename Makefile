CPP_SRC = $(wildcard *.cpp)
CU_SRC = $(wildcard *.cu)
OBJS = ${CU_SRC:.cu=.o} ${CPP_SRC:.cpp=.o}

CC = nvcc
CXX = nvcc
NVCC_INCS = -I/usr/include/SDL2
NVCC_LIBS = -lSDL2 -lGLEW -lGL

go: clean run
	./run

%.o: %.cpp
	$(CXX) $(NVCC_INCS) $(NVCC_LIBS) -o $@ -c $+

%.o: %.cu
	$(CC) $(NVCC_INCS) $(NVCC_LIBS) -o $@ -c $+

run: $(OBJS)
	$(CC) $(NVCC_LIBS) -o $@ $+

clean:
	rm -f run *.o*
