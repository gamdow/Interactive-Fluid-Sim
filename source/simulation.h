#pragma once

#include <cuda_runtime.h>

#include "i_renderer.h"
#include "i_renderable.h"
#include "data/managed_array.h"
#include "kernels/simulation.h"
#include "kernels/visualisation.h"

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

struct Simulation : public IRenderable {
  Simulation(Interface const & _interface, IRenderer & _renderer, OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx, int _pressure_steps);
  virtual ~Simulation() {}
  void step(float _dt);
  void applyBoundary();
  void applySmoke();
  void reset();
  void updateFluidCells(DeviceArray<float> const & _fluid_cells);
private:
  virtual void __render();
  LBFECCSimulationWrapper __simulation;
  VisualisationWrapper __visualisation;
  Interface const & __interface;
  ISurfaceRenderTarget & __renderTarget;
  int const PRESSURE_SOLVER_STEPS;
  float4 __min_rgba, __max_rgba;
  HostArray<float> __fluidCells;
  HostArray<float2> __velocity;
  HostArray<float4> __smoke;
  MirroredArray<float3> __color_map;
};
