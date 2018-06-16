#pragma once

#include <cuda_runtime.h>

#include "renderer.hpp"
#include "data/managed_array.cuh"

enum Mode : int {
  smoke = 0,
  velocity,
  divergence,
  pressure,
  fluid
};

struct Interface;
struct KernelsWrapper;

struct Simulation : public Renderable {
  Simulation(Interface & _interface, KernelsWrapper & _kers, int _pressure_steps);
  virtual ~Simulation();
  void step(float2 _d, float _dt);
  void applyBoundary();
  void applySmoke();
  void reset();
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
  virtual void __render(Resolution const & _window_res, float _mag, float2 _off);
  Interface & __interface;
  int const PRESSURE_SOLVER_STEPS;
  float __min_rgb;
  float __max_rgb;
  DeviceArray<float4> __f4temp;
  DeviceArray<float> __f1temp;
  SurfaceRenderQuad __quad;
};

struct BFECCSimulation : public Simulation {
  BFECCSimulation(Interface & _interface, KernelsWrapper & _kers, int _pressure_steps);
private:
  virtual void advectVelocity(float2 _rd, float _dt);
  DeviceArray<float2> __f2temp;
};

struct LBFECCSimulation : public Simulation {
  LBFECCSimulation(Interface & _interface, KernelsWrapper & _kers, int _pressure_steps);
private:
  virtual void advectVelocity(float2 _rd, float _dt);
  DeviceArray<float2> __f2tempA;
  DeviceArray<float2> __f2tempB;
  DeviceArray<float2> __f2tempC;
};
