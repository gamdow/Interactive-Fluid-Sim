#pragma once

#include <cuda_runtime.h>

#include "kernels.cuh"
#include "memory.hpp"

struct Simulation : public Kernels {
  Simulation(int2 _dimensions, int _buffer, dim3 _block_size);
  virtual ~Simulation();
  void applyBoundary(float _vel);
  void step(float2 _d, float _dt);
  MirroredArray<float2> __velocity;
  MirroredArray<float> __fluidCells;
  float * __divergence;
  float * __pressure;
  float * __buffer;
private:
  int __buffered_size;
};
