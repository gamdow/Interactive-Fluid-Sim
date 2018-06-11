#pragma once

#include <cuda_runtime.h>

#include "../debug.hpp"
#include "../cuda/utility.cuh"
#include "../data_structs/resolution.cuh"

// Wrapper for the simulation kernels that ensures the necessary CUDA resources are available and all the dimensions are consistent.
struct KernelsWrapper
  : public Debug<KernelsWrapper>
  , public OptimalBlockConfig
{
  KernelsWrapper(Resolution const & _res, int _buffer_width);
  virtual ~KernelsWrapper();
  void advectVelocity(float2 * io_velocity, float2 _rdx, float _dt);
  void advectVelocity(float2 * o_velocity, float2 const * _velocity, float2 _rdx, float _dt);
  void limitAdvection(float2 * o_e, float2 * _e1, float2 * _e2);
  template<class T> void applyAdvection(T * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx);
  void calcDivergence(float * o_divergence, float2 const * _velocity, float const * _fluid, float2 _rdx);
  void pressureDecay(float * io_pressure, float const * _fluid);
  void pressureSolve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, float2 _dx);
  void subGradient(float2 * io_velocity, float const * _pressure, float const * _fluid, float2 _rdx);
  void enforceSlip(float2 * io_velocity, float const * _fluid);
  void sum(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2);
  void d2rgba(float4 * o_buffer, float const * _buffer, float _multiplier);
  void hsv2rgba(float4 * o_buffer, float2 const * _buffer, float _power);
  void float42rgba(float4 * o_buffer, float4 const * _buffer, float3 const * _map);
  void copyToSurface(cudaSurfaceObject_t o_surface, Resolution const & _surface_res, float4 const * _array);
  void copyToArray(float * o_buffer, uchar3 const * _buffer, Resolution const & _in_res);
  Resolution const & resolution() const {return __buffer_res;}
  Resolution const & getBufferRes() const {return __buffer_res;}
  void minMaxReduce(float4 & o_min, float4 & o_max, float4 const * _array);
  void scale(float4 * o_array, float4 _min, float4 _max);

private:
  template<class T> TextureObject<T> & selectTextureObject() {}
  DeviceArray<float4> __f4reduce;
  MirroredArray<float4> __min;
  MirroredArray<float4> __max;
  Resolution __buffer_res;
  TextureObject<float> __f1Object;
  TextureObject<float2> __f2Object;
  TextureObject<float4> __f4Object;
};
