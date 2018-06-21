#pragma once

#include <cuda_runtime.h>

#include "../data/resolution.cuh"
#include "../cuda/utility.cuh"

struct KernelWrapper {
  KernelWrapper(OptimalBlockConfig const & _block_config, int _buffer_width) {
    __grid_dim = _block_config.grid;
    __block_dim = _block_config.block;
    __buffer_res = Resolution(_block_config.optimal_res, _buffer_width);
  }
  virtual ~KernelWrapper() {}
  dim3 const & grid() const {return __grid_dim;}
  dim3 const & block() const {return __block_dim;}
  Resolution const & buffer_resolution() const {return __buffer_res;}
private:
  dim3 __grid_dim, __block_dim;
  Resolution __buffer_res;
};

struct SimulationWrapper : public KernelWrapper {
  SimulationWrapper(OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx);
  virtual ~SimulationWrapper() {}
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
protected:
  void advect(float2 * _out, float2 const * _in, float _dt);
  float2 __dx;
  float2 __rdx;
  DeviceArray<float> __fluid_cells;
  DeviceArray<float> __divergence;
  DeviceArray<float> __pressure;
  DeviceArray<float2> __velocity;
  DeviceArray<float4> __smoke;
  DeviceArray<float> __f1_temp;
  TextureObject<float2> __f2_temp_texture;
  TextureObject<float4> __f4_temp_texture;
};

struct BFECCSimulationWrapper : public SimulationWrapper {
  virtual void advectVelocity(float _dt);
private:
  DeviceArray<float2> __f2_temp;
};

struct LBFECCSimulationWrapper : public SimulationWrapper {
  virtual void advectVelocity(float _dt);
private:
  DeviceArray<float2> __f2_tempA;
  DeviceArray<float2> __f2_tempB;
  DeviceArray<float2> __f2_tempC;
};
