BUILDDIR = build

CPP_SRC = $(wildcard *.cpp)
CU_SRC = $(wildcard *.cu)
OBJS = $(patsubst %.cu,$(BUILDDIR)/%.o,$(CU_SRC)) $(patsubst %.cpp,$(BUILDDIR)/%.o,$(CPP_SRC))

CC = nvcc
CXX = nvcc
NVCC_INCS = -I/usr/include/SDL2
NVCC_LIBS = -lSDL2 -lGLEW -lGL

go: clean $(BUILDDIR)/run
	$(BUILDDIR)/run

$(BUILDDIR)/%.o: %.cpp
	@mkdir -p build
	$(CXX) $(NVCC_INCS) $(NVCC_LIBS) -o $@ -c $+

$(BUILDDIR)/%.o: %.cu
	@mkdir -p build
	$(CC) $(NVCC_INCS) $(NVCC_LIBS) -o $@ -c $+

$(BUILDDIR)/run: $(OBJS)
	$(CC) $(NVCC_LIBS) -o $@ $+

clean:
	rm -rf $(BUILDDIR)
