#pragma once

#include <cuda_runtime.h>

#include "renderer.hpp"
#include "data/managed_array.cuh"
#include "kernels/simulation_wrapper.cuh"

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
  Simulation(Interface & _interface, OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx, KernelsWrapper & _kers, int _pressure_steps);
  virtual ~Simulation() {}
  void step(float2 _d, float _dt);
  void applyBoundary();
  void applySmoke();
  void reset();
  DeviceArray<float> & devicefluidCells() {return __sim_kernels.fluidCells();}
private:
  virtual void __render(Resolution const & _window_res, float _mag, float2 _off);
  Interface & __interface;
  SimulationWrapper __sim_kernels;
  KernelsWrapper & __kernels;
  int const PRESSURE_SOLVER_STEPS;
  float __min_rgb, __max_rgb;
  HostArray<float> __fluidCells;
  HostArray<float2> __velocity;
  HostArray<float4> __smoke;
  DeviceArray<float4> __f4temp;
  MirroredArray<float3> __color_map;
  SurfaceRenderQuad __quad;
};

// struct BFECCSimulation : public Simulation {
//   BFECCSimulation(Interface & _interface, KernelsWrapper & _kers, int _pressure_steps);
// private:
//   virtual void advectVelocity(float2 _rd, float _dt);
//   DeviceArray<float2> __f2temp;
// };
//
// struct LBFECCSimulation : public Simulation {
//   LBFECCSimulation(Interface & _interface, KernelsWrapper & _kers, int _pressure_steps);
// private:
//   virtual void advectVelocity(float2 _rd, float _dt);
//   DeviceArray<float2> __f2tempA;
//   DeviceArray<float2> __f2tempB;
//   DeviceArray<float2> __f2tempC;
// };
