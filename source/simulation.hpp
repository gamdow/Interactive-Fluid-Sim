#pragma once

#include <cuda_runtime.h>

#include "memory.hpp"

struct Kernels;
struct BufferSpec;

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
  MirroredArray<float4> __smoke;
private:
  Kernels & __kernels;
  BufferSpec const & __buffer_spec;
  float2 * __f2temp;
  float * __f1temp;
};
