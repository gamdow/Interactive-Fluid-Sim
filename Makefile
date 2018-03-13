CC = nvcc
NVCC_FLAGS = -I/usr/include/SDL2 -lSDL2 -lGLEW -lGLU -lGL

go: clean run
	./run

run: main.o kernels.o
	$(CC) $(NVCC_FLAGS) -o $@ $+

main.o: main.cpp
	$(CC) $(NVCC_FLAGS) -o $@ -c $<

kernels.o: kernels.cu
	$(CC) $(NVCC_FLAGS) -o $@ -c $<

clean:
	rm -f run *.o
