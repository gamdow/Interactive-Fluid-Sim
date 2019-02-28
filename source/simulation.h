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
  Simulation(OptimalBlockConfig const & _block_config, float2 _dx, int _pressure_steps);
  virtual ~Simulation() {}
  Resolution const & visualisation_resolution() const {return __visualisation.resolution();}
  SurfaceWriter const & visualisation_surface_writer() const {return __visualisation;}
  void step(int _mode, float _dt);
  void applyBoundary(float _velocity_setting, int _flow_rotation);
  void applySmoke(int _flow_rotation);
  void reset();
  void updateFluidCells(DeviceArray<float> const & _fluid_cells);
  void addFlow(DeviceArray<float2> const & _velocity, float _dt);
  int & pressureSolverSteps() {return __pressure_solver_steps;}
private:
  LBFECCSimulationWrapper __simulation;
  VisualisationWrapper __visualisation;
  int __pressure_solver_steps;
  float4 __min_rgba, __max_rgba;
  HostArray<float> __fluidCells;
  HostArray<float2> __velocity;
  HostArray<float4> __smoke;
  MirroredArray<float3> __color_map;
  int __last_mode;
};
