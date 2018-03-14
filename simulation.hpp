#pragma once

#include <cuda_runtime.h>

#include "memory.hpp"

struct Kernels;

struct Simulation {
  Simulation(Kernels & _kernels);
  virtual ~Simulation();
  void step(float2 _d, float _dt);
  void applyBoundary(float _vel);
  void applySmoke();
  void reset();
  MirroredArray<float2> __velocity;
  MirroredArray<float> __fluidCells;
  MirroredArray<float> __divergence;
  MirroredArray<float> __pressure;
  MirroredArray<float> __smoke;
private:
  float2 * __f2temp1;
  float2 * __f2temp2;
  float * __f1temp;
  Kernels & __kernels;
  int __buffered_size;
};
