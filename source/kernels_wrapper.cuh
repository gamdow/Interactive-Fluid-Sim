#pragma once

#include <cuda_runtime.h>

#include "cuda_utility.cuh"
#include "resolution.cuh"

// Wrapper for the simulation kernels that ensures the necessary CUDA resources are available and all the dimensions are consistent.
struct KernelsWrapper : private OptimalBlockConfig {
  KernelsWrapper(Resolution const & _res, int _buffer_width);
  virtual ~KernelsWrapper();
  void advectVelocity(float2 * io_velocity, float2 _rdx, float _dt);
  void advectVelocity(float2 * o_velocity, float2 const * _velocity, float2 _rdx, float _dt);
  template<class T> void applyAdvection(T * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx);
  void calcDivergence(float * o_divergence, float2 const * _velocity, float const * _fluid, float2 _rdx);
  void pressureDecay(float * io_pressure, float const * _fluid);
  void pressureSolve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, float2 _dx);
  void subGradient(float2 * io_velocity, float const * _pressure, float const * _fluid, float2 _rdx);
  void enforceSlip(float2 * io_velocity, float const * _fluid);
  void array2rgba(cudaSurfaceObject_t o_surface, float const * _array, float _mul);
  void array2rgba(cudaSurfaceObject_t o_surface, float2 const * _array, float _mul);
  void array2rgba(cudaSurfaceObject_t o_surface, float4 const * _array, float3 const * _map);
  void sum(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2);
  Resolution const & getBufferRes() const {return __buffer_res;}
private:
  template<class T> TextureObject<T> & selectTextureObject() {}
  Resolution __buffer_res;
  TextureObject<float> __f1Object;
  TextureObject<float2> __f2Object;
  TextureObject<float4> __f4Object;
};
