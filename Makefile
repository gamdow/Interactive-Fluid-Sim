BUILDDIR = build
SOURCEDIR = source

CPP_SRC = $(wildcard $(SOURCEDIR)/*.cpp)
CU_SRC = $(wildcard $(SOURCEDIR)/*.cu)
OBJS = $(patsubst $(SOURCEDIR)/%.cu,$(BUILDDIR)/%.o,$(CU_SRC)) $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(CPP_SRC))

CC = nvcc
CXX = nvcc
NVCC_INCS = -I/usr/include/SDL2 -I/usr/include/opencv2
NVCC_LIBS = -lSDL2 -lSDL2_ttf -lGLEW -lGL -lopencv_core -lopencv_videoio -lopencv_imgproc

go: run
	./run

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp
	@mkdir -p build
	$(CXX) $(NVCC_INCS) $(NVCC_LIBS) -o $@ -c $+

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.cu
	@mkdir -p build
	$(CC) $(NVCC_INCS) $(NVCC_LIBS) -o $@ -c $+

run: $(OBJS)
	$(CC) $(NVCC_LIBS) -o $@ $+

clean:
	rm -f run
	rm -rf $(BUILDDIR)
