#ifndef _KERNELS_CUH_
#define _KERNELS_CUH_

#include <cuda_runtime.h>

#include "helper_math.h"
#include "helper_cuda.h"

#include "buffer_spec.cuh"

__global__ void advect_velocity(float2 * o_velocity, cudaTextureObject_t _velocityObj, BufferSpec _buffer_spec, float _dt, float2 _rdx);

template<class T>
__global__ void apply_advection(T * o_data, cudaTextureObject_t _dataObj, float2 const * _velocity, float const * _fluid, BufferSpec _buffer_spec, float _dt, float2 _rdx) {
  if(_fluid[_buffer_spec.idx()] > 0.0f) {
    float2 pos = make_float2(_buffer_spec.x() + 0.5f, _buffer_spec.y() + 0.5f) - _dt * _rdx * _velocity[_buffer_spec.idx()];
    o_data[_buffer_spec.idx()] = tex2D<T>(_dataObj, pos.x, pos.y);
  } else {
    o_data[_buffer_spec.idx()] *= 0.9f;
  }
}

__global__ void calc_divergence(float * o_divergence, float2 const * _velocity, float const * _fluid, BufferSpec _buffer_spec, float2 _rdx);
__global__ void pressure_decay(float * io_pressure, float const * _fluid, BufferSpec _buffer_spec);
__global__ void pressure_solve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, BufferSpec _buffer_spec, float2 _dx);
__global__ void sub_gradient(float2 * io_velocity, float const * _pressure, float const * _fluid, BufferSpec _buffer_spec, float2 _rdx);
__global__ void enforce_slip(float2 * io_velocity, float const * _fluid, BufferSpec _buffer_spec);
// Render 2D field (i.e. velocity) by treating as HSV (hue=direction, saturation=1, value=magnitude) and converting to RGBA
__global__ void hsv_to_rgba(cudaSurfaceObject_t o_surface, float2 const * _array, float _power, BufferSpec _buffer_spec);
__global__ void d_to_rgba(cudaSurfaceObject_t o_surface, float const * _array, float _multiplier, BufferSpec _buffer_spec);
// Render 4D field by operating on it with a 4x3 matrix, where the rows are RGB values (a colour for each dimension).
__global__ void float4_to_rgba(cudaSurfaceObject_t o_surface, float4 const * _array, float3 const * _map, BufferSpec _buffer_spec);
__global__ void sum_arrays(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2, BufferSpec _buffer_spec);

#endif
