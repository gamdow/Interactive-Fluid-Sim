#include "kernels_wrapper.cuh"

#include <iostream>

#include "kernels.cuh"
#include "simulation.cuh"
#include "../debug.hpp"

KernelsWrapper::KernelsWrapper(OptimalBlockConfig const & _block_config, int _buffer_width)
{
  format_out << "Constructing Kernel Device Buffers:" << std::endl;
  OutputIndent indent1;
  __grid_dim = _block_config.grid;
  __block_dim = _block_config.block;
  __buffer_res = Resolution(_block_config.optimal_res, _buffer_width);
  __buffer_res.print("Resolution");
  {
    Allocator alloc;
    __f4reduce.resize(alloc, __grid_dim.x * __grid_dim.y);
    __min.resize(alloc, 1u);
    __max.resize(alloc, 1u);
    __f1Object.init(alloc, __buffer_res);
    __f2Object.init(alloc, __buffer_res);
    __f4Object.init(alloc, __buffer_res);
  }
}

KernelsWrapper::~KernelsWrapper() {
}

void KernelsWrapper::advectVelocity(float2 * io_velocity, float2 _rdx, float _dt) {
  cudaMemcpy2D(__f2Object.getData(), __f2Object.getPitch(), io_velocity, sizeof(float2) * __buffer_res.width, sizeof(float2) * __buffer_res.width, __buffer_res.height, cudaMemcpyDeviceToDevice);
  advect_velocity<<<__grid_dim,__block_dim>>>(io_velocity, __f2Object.getObject(), __buffer_res, _dt, _rdx);
}

void KernelsWrapper::advectVelocity(float2 * o_velocity, float2 const * _velocity, float2 _rdx, float _dt) {
  cudaMemcpy2D(__f2Object.getData(), __f2Object.getPitch(), _velocity, sizeof(float2) * __buffer_res.width, sizeof(float2) * __buffer_res.width, __buffer_res.height, cudaMemcpyDeviceToDevice);
  advect_velocity<<<__grid_dim,__block_dim>>>(o_velocity, __f2Object.getObject(), __buffer_res, _dt, _rdx);
}

void KernelsWrapper::limitAdvection(float2 * o_e, float2 * _e1, float2 * _e2) {
  limit_advection<<<__grid_dim,__block_dim>>>(o_e, _e1, _e2, __buffer_res);
}

// template magic to convert element type (float, float2, etc.) to an instance of the matching TextureObject for the templated applyAdvection
template<> TextureObject<float> & KernelsWrapper::selectTextureObject<float>() {return __f1Object;}
template<> TextureObject<float2> & KernelsWrapper::selectTextureObject<float2>() {return __f2Object;}
template<> TextureObject<float4> & KernelsWrapper::selectTextureObject<float4>() {return __f4Object;}

template<class T>
void KernelsWrapper::applyAdvection(T * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx) {
  auto & object = selectTextureObject<T>();
  cudaMemcpy2D(object.getData(), object.getPitch(), io_data, sizeof(T) * __buffer_res.width, sizeof(T) * __buffer_res.width, __buffer_res.height, cudaMemcpyDeviceToDevice);
  apply_advection<<<__grid_dim,__block_dim>>>(io_data, object.getObject(), _velocity, _fluid, __buffer_res, _dt, _rdx);
}

// explicit template instantiation so applyAdvection can only be used for element types for which there is a matching TextureObject instance
#define APPLY_ADVECTION(TYPE) template void KernelsWrapper::applyAdvection(TYPE * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx);
APPLY_ADVECTION(float)
APPLY_ADVECTION(float2)
APPLY_ADVECTION(float4)
#undef APPLY_ADVECTION

void KernelsWrapper::calcDivergence(float * o_divergence, float2 const * _velocity, float const * _fluid, float2 _rdx) {
  calc_divergence<<<__grid_dim,__block_dim>>>(o_divergence, _velocity, _fluid, __buffer_res, _rdx);
}

void KernelsWrapper::pressureDecay(float * io_pressure, float const * _fluid) {
  pressure_decay<<<__grid_dim,__block_dim>>>(io_pressure, _fluid, __buffer_res);
}

void KernelsWrapper::pressureSolve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, float2 _dx) {
  pressure_solve<<<__grid_dim,__block_dim>>>(o_pressure, _pressure, _divergence, _fluid, __buffer_res, _dx);
}

void KernelsWrapper::subGradient(float2 * io_velocity, float const * _pressure, float const * _fluid, float2 _rdx) {
  sub_gradient<<<__grid_dim,__block_dim>>>(io_velocity, _pressure, _fluid, __buffer_res, _rdx);
}

void KernelsWrapper::enforceSlip(float2 * io_velocity, float const * _fluid) {
  enforce_slip<<<__grid_dim,__block_dim>>>(io_velocity, _fluid, __buffer_res);
}

void KernelsWrapper::d2rgba(float4 * o_buffer, float const * _buffer, float _multiplier) {
  scalar_to_rgba<<<__grid_dim,__block_dim>>>(o_buffer, _buffer, __buffer_res, _multiplier);
}

void KernelsWrapper::hsv2rgba(float4 * o_buffer, float2 const * _buffer, float _power) {
  vector_field_to_rgba<<<__grid_dim,__block_dim>>>(o_buffer, _buffer, __buffer_res, _power);
}

void KernelsWrapper::float42rgba(float4 * o_buffer, float4 const * _buffer, float3 const * _map) {
  map_to_rgba<<<__grid_dim,__block_dim>>>(o_buffer, _buffer, __buffer_res, _map);
}

void KernelsWrapper::copyToSurface(cudaSurfaceObject_t o_surface, Resolution const & _surface_res, float4 const * _buffer) {
  copy_to_surface<<<__grid_dim,__block_dim>>>(o_surface, _surface_res, _buffer, __buffer_res);
}

void KernelsWrapper::copyToArray(float * o_buffer, uchar3 const * _buffer, Resolution const & _in_res) {
  copy_to_array<<<__grid_dim,__block_dim>>>(o_buffer, __buffer_res, _buffer, _in_res);
}

void KernelsWrapper::sum(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2) {
  sum_arrays<<<__grid_dim,__block_dim>>>(o_array, _c1, _array1, _c2, _array2, __buffer_res);
}

void KernelsWrapper::minMaxReduce(float4 & o_min, float4 & o_max, float4 const * _array) {
  min<<<__grid_dim, __block_dim>>>(__f4reduce, _array, __buffer_res);
  min<<<1, __grid_dim>>>(__min.device(), __f4reduce, Resolution(__grid_dim.x, __grid_dim.y, 0));
  __min.copyDeviceToHost();
  o_min = __min[0];
  max<<<__grid_dim, __block_dim>>>(__f4reduce, _array, __buffer_res);
  max<<<1, __grid_dim>>>(__max.device(), __f4reduce, Resolution(__grid_dim.x, __grid_dim.y, 0));
  __max.copyDeviceToHost();
  o_max = __max[0];
}

void KernelsWrapper::scale(float4 * o_array, float4 _min, float4 _max) {
  scaleRGB<<<__grid_dim, __block_dim>>>(o_array, _min, _max, __buffer_res);
}
