#include "kernels_wrapper.cuh"

// #include <iostream> // for host code
// #include <stdio.h> // for kernel code
// #include "helper_math.h"
// #include "helper_cuda.h"

#include "capability.cuh"
#include "kernels.cuh"

Kernels::Kernels(Capability const & _cap)
  : __capability(_cap)
{
  __f1Object.init(_cap.buffer_spec);
  __f2Object.init(_cap.buffer_spec);
  __f4Object.init(_cap.buffer_spec);
}

Kernels::~Kernels() {
  __f1Object.shutdown();
  __f2Object.shutdown();
  __f4Object.shutdown();
}

BufferSpec const & Kernels::getBufferSpec() const {
  return __capability.buffer_spec;
}

void Kernels::advectVelocity(float2 * io_velocity, float2 _rdx, float _dt) {
  cudaMemcpy2D(__f2Object.__buffer, __f2Object.__pitch, io_velocity, sizeof(float2) * __capability.buffer_spec.width, sizeof(float2) * __capability.buffer_spec.width, __capability.buffer_spec.height, cudaMemcpyDeviceToDevice);
  advect_velocity<<<__capability.grid,__capability.block>>>(io_velocity, __f2Object.__object, __capability.buffer_spec, _dt, _rdx);
}

void Kernels::advectVelocity(float2 * o_velocity, float2 const * _velocity, float2 _rdx, float _dt) {
  cudaMemcpy2D(__f2Object.__buffer, __f2Object.__pitch, _velocity, sizeof(float2) * __capability.buffer_spec.width, sizeof(float2) * __capability.buffer_spec.width, __capability.buffer_spec.height, cudaMemcpyDeviceToDevice);
  advect_velocity<<<__capability.grid,__capability.block>>>(o_velocity, __f2Object.__object, __capability.buffer_spec, _dt, _rdx);
}

// template magic to convert element type (float, float2, etc.) to an instance of the matching TextureObject for the templated applyAdvection
template<> TextureObject<float> & Kernels::selectTextureObject<float>() {return __f1Object;}
template<> TextureObject<float2> & Kernels::selectTextureObject<float2>() {return __f2Object;}
template<> TextureObject<float4> & Kernels::selectTextureObject<float4>() {return __f4Object;}

template<class T>
void Kernels::applyAdvection(T * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx) {
  auto & object = selectTextureObject<T>();
  cudaMemcpy2D(object.__buffer, object.__pitch, io_data, sizeof(T) * __capability.buffer_spec.width, sizeof(T) * __capability.buffer_spec.width, __capability.buffer_spec.height, cudaMemcpyDeviceToDevice);
  apply_advection<<<__capability.grid,__capability.block>>>(io_data, object.__object, _velocity, _fluid, __capability.buffer_spec, _dt, _rdx);
}

// explicit template instantiation so applyAdvection can only be used for element types for which there is a matching TextureObject instance
template void Kernels::applyAdvection<float>(float * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx);
template void Kernels::applyAdvection<float2>(float2 * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx);
template void Kernels::applyAdvection<float4>(float4 * io_data, float2 const * _velocity, float const * _fluid, float _dt, float2 _rdx);

void Kernels::calcDivergence(float * o_divergence, float2 const * _velocity, float const * _fluid, float2 _rdx) {
  calc_divergence<<<__capability.grid,__capability.block>>>(o_divergence, _velocity, _fluid, __capability.buffer_spec, _rdx);
}

void Kernels::pressureDecay(float * io_pressure, float const * _fluid) {
  pressure_decay<<<__capability.grid,__capability.block>>>(io_pressure, _fluid, __capability.buffer_spec);
}

void Kernels::pressureSolve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, float2 _dx) {
  pressure_solve<<<__capability.grid,__capability.block>>>(o_pressure, _pressure, _divergence, _fluid, __capability.buffer_spec, _dx);
}

void Kernels::subGradient(float2 * io_velocity, float const * _pressure, float const * _fluid, float2 _rdx) {
  sub_gradient<<<__capability.grid,__capability.block>>>(io_velocity, _pressure, _fluid, __capability.buffer_spec, _rdx);
}

void Kernels::enforceSlip(float2 * io_velocity, float const * _fluid) {
  enforce_slip<<<__capability.grid,__capability.block>>>(io_velocity, _fluid, __capability.buffer_spec);
}

void Kernels::hsv2rgba(cudaSurfaceObject_t o_surface, float2 const * _array, float _power) {
  hsv_to_rgba<<<__capability.grid,__capability.block>>>(o_surface, _array, _power, __capability.buffer_spec);
}

void Kernels::v2rgba(cudaSurfaceObject_t o_surface, float const * _array, float _multiplier) {
  d_to_rgba<<<__capability.grid,__capability.block>>>(o_surface, _array, _multiplier, __capability.buffer_spec);
}

void Kernels::float42rgba(cudaSurfaceObject_t o_surface, float4 const * _array, float3 const * _map) {
  float4_to_rgba<<<__capability.grid,__capability.block>>>(o_surface, _array, _map, __capability.buffer_spec);
}

void Kernels::sum(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2) {
  sum_arrays<<<__capability.grid,__capability.block>>>(o_array, _c1, _array1, _c2, _array2, __capability.buffer_spec);
}
