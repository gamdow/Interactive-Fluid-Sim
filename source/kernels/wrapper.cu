#include "wrapper.cuh"

#include "general.cuh"

KernelWrapper::KernelWrapper(OptimalBlockConfig const & _block_config, int _buffer_width) {
  __grid_dim = _block_config.grid;
  __block_dim = _block_config.block;
  __buffer_res = Resolution(_block_config.optimal_res, _buffer_width);
}

void KernelWrapper::copyToSurface(cudaSurfaceObject_t o_surface, Resolution const & _surface_res, float4 const * _buffer) const {
  copy_to_surface<<<__grid_dim,__block_dim>>>(o_surface, _surface_res, _buffer, __buffer_res);
}
