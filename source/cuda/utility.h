#pragma once

#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "../data/resolution.h"
#include "../data/managed_array.h"
#include "../data/texture_object.h"

// explicit template instantiation so applyAdvection can only be used for element types for which there is a matching TextureObject instance
#define EXPLICT_INSTATIATION(TYPED_MACRO) \
  TYPED_MACRO(float) \
  TYPED_MACRO(float2) \
  TYPED_MACRO(float3) \
  TYPED_MACRO(float4) \
  TYPED_MACRO(unsigned char) \
  TYPED_MACRO(uchar3)

void reportCudaCapability();

struct OptimalBlockConfig {
  OptimalBlockConfig(Resolution _res);
  Resolution optimal_res;
  dim3 grid, block;
};

void print(std::ostream & _out, float4 _v);

void copyToSurface(OptimalBlockConfig const & _block_config, cudaSurfaceObject_t o_surface, Resolution const & _surface_res, uchar3 const * _buffer, Resolution const & _buffer_res);

void copyToSurface(OptimalBlockConfig const & _block_config, cudaSurfaceObject_t o_surface, Resolution const & _surface_res, unsigned char const * _buffer, Resolution const & _buffer_res);

void copyToSurface(OptimalBlockConfig const & _block_config, cudaSurfaceObject_t o_surface, Resolution const & _surface_res, float const * _buffer, Resolution const & _buffer_res);

void copyToSurface(OptimalBlockConfig const & _block_config, cudaSurfaceObject_t o_surface, Resolution const & _surface_res, float4 const * _buffer, Resolution const & _buffer_res);

struct SurfaceWriter {
  void writeToSurface(cudaSurfaceObject_t o_surface, Resolution const & _surface_res) const {writeToSurfaceImpl(o_surface, _surface_res);}
private:
  virtual void writeToSurfaceImpl(cudaSurfaceObject_t o_surface, Resolution const & _surface_res) const = 0;
};
