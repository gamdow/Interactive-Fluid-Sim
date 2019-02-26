#pragma once

#include <cuda_runtime.h>

#include "../cuda/utility.h"
#include "../data/resolution.h"
#include "../data/managed_array.h"

struct SimulationWrapper {
  SimulationWrapper(OptimalBlockConfig const & _block_config, float2 _dx);
  virtual ~SimulationWrapper() {}
  Resolution const & resolution() const {return __config.resolution;}
  DeviceArray<float> & fluidCells() {return __fluid_cells;}
  DeviceArray<float> & divergence() {return __divergence;}
  DeviceArray<float> & pressure() {return __pressure;}
  DeviceArray<float2> & velocity() {return __velocity;}
  DeviceArray<float4> & smoke() {return __smoke;}
  virtual void advectVelocity(float _dt);
  void calcDivergence();
  void pressureDecay();
  void pressureSolveStep();
  void subGradient();
  void enforceSlip();
  void advectSmoke(float _dt);
  void applySmoke(int _flow_direction);
  void injectVelocityAndApplyBoundary(int _flow_direction, float _velocity_setting);
protected:
  void advect(float2 * _out, float2 const * _in, float _dt);
  float2 __dx;
  float2 __rdx;
  OptimalBlockConfig const & __config;
  DeviceArray<float> __fluid_cells;
  DeviceArray<float> __divergence;
  DeviceArray<float> __pressure;
  DeviceArray<float2> __velocity;
  DeviceArray<float4> __smoke;
  DeviceArray<float> __f1_temp;
  TextureObject<float> __f1_temp_texture;
  TextureObject<float2> __f2_temp_texture;
  TextureObject<float4> __f4_temp_texture;
};

struct BFECCSimulationWrapper : public SimulationWrapper {
  BFECCSimulationWrapper(OptimalBlockConfig const & _block_config, float2 _dx);
  virtual void advectVelocity(float _dt);
private:
  DeviceArray<float2> __f2_temp;
};

struct LBFECCSimulationWrapper : public SimulationWrapper {
  LBFECCSimulationWrapper(OptimalBlockConfig const & _block_config, float2 _dx);
  virtual void advectVelocity(float _dt);
private:
  DeviceArray<float2> __f2_tempA;
  DeviceArray<float2> __f2_tempB;
  DeviceArray<float2> __f2_tempC;
};
