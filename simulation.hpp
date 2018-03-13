#pragma once

#include <cuda_runtime.h>

#include "memory.hpp"

struct Kernels;

struct Simulation {
  Simulation(Kernels & _kernels);
  virtual ~Simulation();
  void applyBoundary(float _vel);
  void step(float2 _d, float _dt);
  MirroredArray<float2> __velocity;
  MirroredArray<float> __fluidCells;
  float * __divergence;
  float * __pressure;
  float * __buffer;
private:
  Kernels & __kernels;
  int __buffered_size;
};
