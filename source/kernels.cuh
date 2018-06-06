#pragma once

#include <cuda_runtime.h>

#include "resolution.cuh"

__global__ void advect_velocity(float2 * o_velocity, cudaTextureObject_t _velocityObj, Resolution _buffer_res, float _dt, float2 _rdx);
__global__ void limit_advection(float2 * o_e, float2 * _e1, float2 * _e2, Resolution _buffer_res);
template<class T> __global__ void apply_advection(T * o_data, cudaTextureObject_t _dataObj, float2 const * _velocity, float const * _fluid, Resolution _buffer_res, float _dt, float2 _rdx);
__global__ void calc_divergence(float * o_divergence, float2 const * _velocity, float const * _fluid, Resolution _buffer_res, float2 _rdx);
__global__ void pressure_decay(float * io_pressure, float const * _fluid, Resolution _buffer_res);
__global__ void pressure_solve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, Resolution _buffer_res, float2 _dx);
__global__ void sub_gradient(float2 * io_velocity, float const * _pressure, float const * _fluid, Resolution _buffer_res, float2 _rdx);
__global__ void enforce_slip(float2 * io_velocity, float const * _fluid, Resolution _buffer_res);
__global__ void d_to_rgba(cudaSurfaceObject_t o_surface, Resolution _surface_res, float const * _buffer, Resolution _buffer_res, float _multiplier);
// Render 2D field (i.e. velocity) by treating as HSV (hue=direction, saturation=1, value=magnitude) and converting to RGBA
__global__ void hsv_to_rgba(cudaSurfaceObject_t o_surface, Resolution _surface_res, float2 const * _buffer, Resolution _buffer_res, float _power);
// Render 4D field by operating on it with a 4x3 matrix, where the rows are RGB values (a colour for each dimension).
__global__ void float4_to_rgba(cudaSurfaceObject_t o_surface, Resolution _surface_res, float4 const * _buffer, Resolution _buffer_res, float3 const * _map);
__global__ void sum_arrays(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2, Resolution _buffer_res);
