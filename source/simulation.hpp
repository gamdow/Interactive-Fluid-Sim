#pragma once

#include <cuda_runtime.h>

#include "debug.hpp"
#include "data_structs/managed_array.cuh"
#include "kernels/kernels_wrapper.cuh"

enum Mode : int {
  smoke = 0,
  velocity,
  divergence,
  pressure,
  fluid
};

struct Simulation : public Debug<Simulation> {
  Simulation(KernelsWrapper & _kers, int _pressure_steps);
  virtual ~Simulation();
  void step(Mode _mode, float2 _d, float _dt, float _mul);
  void applyBoundary(float _vel);
  void applySmoke();
  void reset();
  DeviceArray<float4> __f4temp;
  MirroredArray<float> __fluidCells;
protected:
  KernelsWrapper & __kernels;
  MirroredArray<float> __divergence;
  MirroredArray<float> __pressure;
  MirroredArray<float2> __velocity;
  MirroredArray<float4> __smoke;
  MirroredArray<float3> __color_map;
private:
  virtual void advectVelocity(float2 _rd, float _dt);
  int const PRESSURE_SOLVER_STEPS;
  DeviceArray<float> __f1temp;
  float __min_rgb;
  float __max_rgb;
};

struct BFECCSimulation : public Simulation {
  BFECCSimulation(KernelsWrapper & _kers, int _pressure_steps);
private:
  virtual void advectVelocity(float2 _rd, float _dt);
  DeviceArray<float2> __f2temp;
};

struct LBFECCSimulation : public Simulation {
  LBFECCSimulation(KernelsWrapper & _kers, int _pressure_steps);
private:
  virtual void advectVelocity(float2 _rd, float _dt);
  DeviceArray<float2> __f2tempA;
  DeviceArray<float2> __f2tempB;
  DeviceArray<float2> __f2tempC;
};
