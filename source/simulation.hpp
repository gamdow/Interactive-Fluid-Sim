#pragma once

#include <cuda_runtime.h>

#include "renderer.hpp"
#include "data/managed_array.cuh"
#include "kernels/simulation_wrapper.cuh"
#include "kernels/visualisation_wrapper.cuh"

enum Mode : int {
  smoke = 0,
  velocity,
  divergence,
  pressure,
  fluid
};

struct Interface;
struct KernelsWrapper;
struct Camera;

struct Simulation : public Renderable {
  Simulation(Interface & _interface, OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx, int _pressure_steps);
  virtual ~Simulation() {}
  void step(float2 _d, float _dt);
  void applyBoundary();
  void applySmoke();
  void reset();
  void copyToFluidCells(Camera const & _cam);
private:
  virtual void __render(Resolution const & _window_res, float _mag, float2 _off);
  Interface & __interface;
  LBFECCSimulationWrapper __sim_kernels;
  VisualisationWrapper __vis_kernels;
  int const PRESSURE_SOLVER_STEPS;
  float __min_rgb, __max_rgb;
  HostArray<float> __fluidCells;
  HostArray<float2> __velocity;
  HostArray<float4> __smoke;
  MirroredArray<float3> __color_map;
  SurfaceRenderQuad __quad;
};
