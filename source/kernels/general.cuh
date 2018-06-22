#pragma once

#include <cuda_runtime.h>

#include "../data/resolution.cuh"

__global__ void sum_arrays(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2, Resolution _buffer_res);

__global__ void copy_to_surface(cudaSurfaceObject_t o_surface, Resolution _surface_res, float4 const * _buffer, Resolution _buffer_res);

__global__ void copy_to_array(float * o_buffer, Resolution _out_res, uchar3 const * _buffer, Resolution _in_res);
