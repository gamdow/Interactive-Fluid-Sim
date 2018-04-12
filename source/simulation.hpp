#pragma once

#include <cuda_runtime.h>

#include "kernels_wrapper.cuh"
#include "cuda_utility.cuh"

struct SimulationBase {
  SimulationBase(KernelsWrapper const & _kers);
};

struct Simulation : public SimulationBase {
  Simulation(KernelsWrapper & _kers);
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
  KernelsWrapper & __kernels;
  float2 * __f2temp;
  float * __f1temp;
};
