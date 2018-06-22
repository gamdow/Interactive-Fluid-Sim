#pragma once

#include <cuda_runtime.h>

#include "../data/resolution.cuh"
#include "../cuda/utility.cuh"

struct KernelWrapper {
  KernelWrapper(OptimalBlockConfig const & _block_config, int _buffer_width);
  virtual ~KernelWrapper() {}
  dim3 const & grid() const {return __grid_dim;}
  dim3 const & block() const {return __block_dim;}
  Resolution const & buffer_resolution() const {return __buffer_res;}
  void copyToSurface(cudaSurfaceObject_t o_surface, Resolution const & _surface_res, float4 const * _array) const;
private:
  dim3 __grid_dim, __block_dim;
  Resolution __buffer_res;
};
