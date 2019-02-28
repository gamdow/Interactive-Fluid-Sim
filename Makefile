BUILDDIR := build
SOURCEDIR := source
SUBDIRS := $(sort $(dir $(wildcard $(SOURCEDIR)/*/)))

# CPP_SRC = $(wildcard $(SOURCEDIR)/**/*.cpp)
CPP_SRC = $(wildcard $(addsuffix *.cpp,$(SUBDIRS)))
# CU_SRC = $(wildcard $(SOURCEDIR)/**/*.cu)
CU_SRC = $(wildcard $(addsuffix *.cu,$(SUBDIRS)))
# OBJS = $(patsubst $(SOURCEDIR)/%.cu,$(BUILDDIR)/%.o,$(CU_SRC)) $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(CPP_SRC))
OBJS = $(patsubst $(SOURCEDIR)/%.cu,$(BUILDDIR)/%.o,$(CU_SRC)) $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(CPP_SRC))
# HEADERS = $(wildcard $(SOURCEDIR)/**/*.hpp) $(wildcard $(SOURCEDIR)/*.cuh)
HEADERS = $(wildcard $(addsuffix *.h,$(SUBDIRS)))

CC = nvcc -O2
CXX = nvcc -O2
NVCC_INCS = -I/usr/include/SDL2 -I/usr/include/opencv2
NVCC_LIBS = -lSDL2 -lSDL2_ttf -lGLEW -lGL -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_optflow


go: run
	./run

$(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp $(HEADERS)
	@mkdir -p $(@D)
	$(CXX) $(NVCC_INCS) $(NVCC_LIBS) -o $@ -c $<

$(BUILDDIR)/%.o : $(SOURCEDIR)/%.cu $(HEADERS)
	@mkdir -p $(@D)
	$(CC) $(NVCC_INCS) $(NVCC_LIBS) -o $@ -c $<

run: $(OBJS)
	$(CC) $(NVCC_LIBS) -o $@ $+

clean:
	rm -f run
	rm -rf $(BUILDDIR)
