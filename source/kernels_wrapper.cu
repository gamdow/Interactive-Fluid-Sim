#include "kernels_wrapper.cuh"

#include <iostream>

#include "kernels.cuh"

KernelsWrapper::KernelsWrapper(Resolution const & _res, int _buffer_width)
  : OptimalBlockConfig(_res)
{
  std::cout << std::endl << "Constructing Kernel Device Buffers:" << std::endl;
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

void KernelsWrapper::array2rgba(cudaSurfaceObject_t o_surface, float const * _array, float _multiplier) {
  d_to_rgba<<<grid,block>>>(o_surface, _array, _multiplier, __buffer_res);
}

void KernelsWrapper::array2rgba(cudaSurfaceObject_t o_surface, float2 const * _array, float _power) {
  hsv_to_rgba<<<grid,block>>>(o_surface, _array, _power, __buffer_res);
}

void KernelsWrapper::array2rgba(cudaSurfaceObject_t o_surface, float4 const * _array, float3 const * _map) {
  float4_to_rgba<<<grid,block>>>(o_surface, _array, _map, __buffer_res);
}

void KernelsWrapper::sum(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2) {
  sum_arrays<<<grid,block>>>(o_array, _c1, _array1, _c2, _array2, __buffer_res);
}
