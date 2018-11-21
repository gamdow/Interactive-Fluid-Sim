#pragma once

#include <cuda_runtime.h>

#include "data/managed_array.h"
#include "kernels/simulation.h"
#include "kernels/visualisation.h"
#include "interface/enums.h"

struct Interface;
struct KernelsWrapper;
struct Camera;

struct Simulation {
  Simulation(Interface const & _interface, OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx, int _pressure_steps);
  virtual ~Simulation() {}
  Resolution const & visualisation_resolution() const {return __visualisation.buffer_resolution();}
  SurfaceWriter const & visualisation_surface_writer() const {return __visualisation;}
  void step(float _dt);
  void applyBoundary();
  void applySmoke();
  void reset();
  void updateFluidCells(DeviceArray<float> const & _fluid_cells);
private:
  LBFECCSimulationWrapper __simulation;
  VisualisationWrapper __visualisation;
  Interface const & __interface;
  int const PRESSURE_SOLVER_STEPS;
  float4 __min_rgba, __max_rgba;
  HostArray<float> __fluidCells;
  HostArray<float2> __velocity;
  HostArray<float4> __smoke;
  MirroredArray<float3> __color_map;
};
