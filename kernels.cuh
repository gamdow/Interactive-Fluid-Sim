#pragma once

#include <cuda_runtime.h>

#include <stdio.h>
#include "configuration.cuh"

struct Kernels {
  Kernels(int2 _dimensions, int _buffer, dim3 _block_size);
  virtual ~Kernels();
  void advectVelocity(float2 * io_velocity, float2 _rdx, float _dt);
  void calcDivergence(float * o_divergence, float2 * _velocity, float * _fluid, float2 _r2dx);
  void pressureDecay(float * io_pressure, float * _fluid);
  void pressureSolve(float * o_pressure, float * _pressure, float * _divergence, float * _fluid, float2 _dx);
  void subGradient(float2 * io_velocity, float * _pressure, float * _fluid, float2 _r2dx);
  void enforceSlip(float2 * io_velocity, float * _fluid);
  void hsv2rgba(cudaSurfaceObject_t o_surface, float2 * _array, float _mul);
  void v2rgba(cudaSurfaceObject_t o_surface, float * _array, float _mul);
  int2 __dims;
  int3 __buffer_spec;
private:
  dim3 __block, __grid;
  float2 * __tex_buffer;
  size_t __tex_pitch;
  cudaTextureObject_t __tex_object;
};
