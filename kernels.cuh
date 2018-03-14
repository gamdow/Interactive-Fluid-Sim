#pragma once

#include <cuda_runtime.h>

template<class T>
struct TextureObject {
  TextureObject();
  void init(int3 _buffer_spec);
  void shutdown();
  T * __buffer;
  size_t __pitch;
  cudaTextureObject_t __object;
};

// Wrapper for the simulation kernels that ensures the necessary CUDA resources are available and all the dimensions are consistent.
struct Kernels {
  Kernels(int2 _dimensions, int _buffer);
  virtual ~Kernels();
  void advectVelocity(float2 * io_velocity, float2 _rdx, float _dt);
  void advectVelocity(float2 * o_velocity, float2 const * _velocity, float2 _rdx, float _dt);
  template<class T> void applyAdvection(T * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx);
  //void applyAdvection(float * io_data, float2 const * _velocity, float _dt, float2 _rdx);
  void calcDivergence(float * o_divergence, float2 const * _velocity, float const * _fluid, float2 _r2dx);
  void pressureDecay(float * io_pressure, float const * _fluid);
  void pressureSolve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, float2 _dx);
  void subGradient(float2 * io_velocity, float const * _pressure, float const * _fluid, float2 _r2dx);
  void enforceSlip(float2 * io_velocity, float const * _fluid);
  void hsv2rgba(cudaSurfaceObject_t o_surface, float2 const * _array, float _mul);
  void v2rgba(cudaSurfaceObject_t o_surface, float const * _array, float _mul);
  void float42rgba(cudaSurfaceObject_t o_surface, float4 const * _array, float3 const * _map);
  void sum(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2);
  int2 __dims;
  int3 __buffer_spec;
  int __buffered_size;
private:
  void reportCapability() const;
  void optimiseBlockSize(int2 _dims, int _buffer);
  template<class T> TextureObject<T> & selectTextureObject() {}
  dim3 __block, __grid;
  TextureObject<float> __f1Object;
  TextureObject<float2> __f2Object;
  TextureObject<float4> __f4Object;
};
