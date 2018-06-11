#include "kernels_wrapper.cuh"

#include <iostream>

#include "kernels.cuh"

KernelsWrapper::KernelsWrapper(Resolution const & _res, int _buffer_width)
  : Debug<KernelsWrapper>("Constructing Kernel Device Buffers:")
  , OptimalBlockConfig(_res)
  , __f4reduce(grid.x * grid.y)
  , __min(1)
  , __max(1)
{
  __buffer_res = Resolution(optimal_res, _buffer_width);
  __buffer_res.print("\tResolution");
  __f1Object.init(__buffer_res);
  __f2Object.init(__buffer_res);
  __f4Object.init(__buffer_res);
  std::cout << "\tTotal: " << (__f1Object.__pitch + __f2Object.__pitch + __f4Object.__pitch) * __buffer_res.height << " bytes" << std::endl;
}

KernelsWrapper::~KernelsWrapper() {
  __f1Object.shutdown();
  __f2Object.shutdown();
  __f4Object.shutdown();
}

void KernelsWrapper::advectVelocity(float2 * io_velocity, float2 _rdx, float _dt) {
  cudaMemcpy2D(__f2Object.__buffer, __f2Object.__pitch, io_velocity, sizeof(float2) * __buffer_res.width, sizeof(float2) * __buffer_res.width, __buffer_res.height, cudaMemcpyDeviceToDevice);
  advect_velocity<<<grid,block>>>(io_velocity, __f2Object.__object, __buffer_res, _dt, _rdx);
}

void KernelsWrapper::advectVelocity(float2 * o_velocity, float2 const * _velocity, float2 _rdx, float _dt) {
  cudaMemcpy2D(__f2Object.__buffer, __f2Object.__pitch, _velocity, sizeof(float2) * __buffer_res.width, sizeof(float2) * __buffer_res.width, __buffer_res.height, cudaMemcpyDeviceToDevice);
  advect_velocity<<<grid,block>>>(o_velocity, __f2Object.__object, __buffer_res, _dt, _rdx);
}

void KernelsWrapper::limitAdvection(float2 * o_e, float2 * _e1, float2 * _e2) {
  limit_advection<<<grid,block>>>(o_e, _e1, _e2, __buffer_res);
}

// template magic to convert element type (float, float2, etc.) to an instance of the matching TextureObject for the templated applyAdvection
template<> TextureObject<float> & KernelsWrapper::selectTextureObject<float>() {return __f1Object;}
template<> TextureObject<float2> & KernelsWrapper::selectTextureObject<float2>() {return __f2Object;}
template<> TextureObject<float4> & KernelsWrapper::selectTextureObject<float4>() {return __f4Object;}

template<class T>
void KernelsWrapper::applyAdvection(T * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx) {
  auto & object = selectTextureObject<T>();
  cudaMemcpy2D(object.__buffer, object.__pitch, io_data, sizeof(T) * __buffer_res.width, sizeof(T) * __buffer_res.width, __buffer_res.height, cudaMemcpyDeviceToDevice);
  apply_advection<<<grid,block>>>(io_data, object.__object, _velocity, _fluid, __buffer_res, _dt, _rdx);
}

// explicit template instantiation so applyAdvection can only be used for element types for which there is a matching TextureObject instance
#define APPLY_ADVECTION(TYPE) template void KernelsWrapper::applyAdvection(TYPE * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx);
APPLY_ADVECTION(float)
APPLY_ADVECTION(float2)
APPLY_ADVECTION(float4)
#undef APPLY_ADVECTION

void KernelsWrapper::calcDivergence(float * o_divergence, float2 const * _velocity, float const * _fluid, float2 _rdx) {
  calc_divergence<<<grid,block>>>(o_divergence, _velocity, _fluid, __buffer_res, _rdx);
}

void KernelsWrapper::pressureDecay(float * io_pressure, float const * _fluid) {
  pressure_decay<<<grid,block>>>(io_pressure, _fluid, __buffer_res);
}

void KernelsWrapper::pressureSolve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, float2 _dx) {
  pressure_solve<<<grid,block>>>(o_pressure, _pressure, _divergence, _fluid, __buffer_res, _dx);
}

void KernelsWrapper::subGradient(float2 * io_velocity, float const * _pressure, float const * _fluid, float2 _rdx) {
  sub_gradient<<<grid,block>>>(io_velocity, _pressure, _fluid, __buffer_res, _rdx);
}

void KernelsWrapper::enforceSlip(float2 * io_velocity, float const * _fluid) {
  enforce_slip<<<grid,block>>>(io_velocity, _fluid, __buffer_res);
}

void KernelsWrapper::d2rgba(float4 * o_buffer, float const * _buffer, float _multiplier) {
  d_to_rgba<<<grid,block>>>(o_buffer, _buffer, __buffer_res, _multiplier);
}

void KernelsWrapper::hsv2rgba(float4 * o_buffer, float2 const * _buffer, float _power) {
  hsv_to_rgba<<<grid,block>>>(o_buffer, _buffer, __buffer_res, _power);
}

void KernelsWrapper::float42rgba(float4 * o_buffer, float4 const * _buffer, float3 const * _map) {
  float4_to_rgba<<<grid,block>>>(o_buffer, _buffer, __buffer_res, _map);
}

void KernelsWrapper::copyToSurface(cudaSurfaceObject_t o_surface, Resolution const & _surface_res, float4 const * _buffer) {
  copy_to_surface<<<grid,block>>>(o_surface, _surface_res, _buffer, __buffer_res);
}

void KernelsWrapper::copyToArray(float * o_buffer, uchar3 const * _buffer, Resolution const & _in_res) {
  copy_to_array<<<grid,block>>>(o_buffer, __buffer_res, _buffer, _in_res);
}

void KernelsWrapper::sum(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2) {
  sum_arrays<<<grid,block>>>(o_array, _c1, _array1, _c2, _array2, __buffer_res);
}

void KernelsWrapper::minMaxReduce(float4 & o_min, float4 & o_max, float4 const * _array) {
  min<<<grid, block>>>(__f4reduce, _array, __buffer_res);
  min<<<1, grid>>>(__min.device(), __f4reduce, Resolution(grid.x, grid.y, 0));
  __min.copyDeviceToHost();
  o_min = __min[0];
  max<<<grid, block>>>(__f4reduce, _array, __buffer_res);
  max<<<1, grid>>>(__max.device(), __f4reduce, Resolution(grid.x, grid.y, 0));
  __max.copyDeviceToHost();
  o_max = __max[0];
}

void KernelsWrapper::scale(float4 * o_array, float4 _min, float4 _max) {
  scaleRGB<<<grid, block>>>(o_array, _min, _max, __buffer_res);
}
