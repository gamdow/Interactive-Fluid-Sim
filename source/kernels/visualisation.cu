#include "visualisation.cuh"

// #include <iostream> // for host code
#include <stdio.h> // for kernel code

#include "../cuda/helper_math.h"
#include "../cuda/helper_cuda.h"
#include "../cuda/utility.cuh"

float const PI = 3.14159265359f;

__global__ void scalar_to_rgba(float4 * o_buffer, float const * _buffer, Resolution _buffer_res, float _multiplier) {
  int const idx = _buffer_res.idx();
  float pos = (_buffer[idx] + abs(_buffer[idx])) / 2.0f;
  float neg = -(_buffer[idx] - abs(_buffer[idx])) / 2.0f;
  o_buffer[idx] = make_float4(neg * _multiplier, pos * _multiplier, 0.0, 1.0f);
}

// Render 2D field (i.e. velocity) by treating as HSV (hue=direction, saturation=1, value=magnitude) and converting to RGBA
__global__ void vector_field_to_rgba(float4 * o_buffer, float2 const * _buffer, Resolution _buffer_res, float _power) {
  int const idx = _buffer_res.idx();
  float v = __powf(_buffer[idx].x * _buffer[idx].x + _buffer[idx].y * _buffer[idx].y, _power);
  float h = 6.0f * (atan2f(-_buffer[idx].x, -_buffer[idx].y) / (2 * PI) + 0.5);
  float hi = floorf(h);
  float f = h - hi;
  float q = v * (1 - f);
  float t = v * f;
  float3 rgb;
  switch((int)hi) {
    default: rgb = make_float3(v, t, 0.0f); break;
    case 1: rgb = make_float3(q, v, 0.0f); break;
    case 2: rgb = make_float3(0.0f, v, t); break;
    case 3: rgb = make_float3(0.0f, q, v); break;
    case 4: rgb = make_float3(t, 0.0f, v); break;
    case 5: rgb = make_float3(v, 0.0f, q); break;
  }
  o_buffer[idx] = make_float4(rgb, fmin(rgb.x + rgb.y + rgb.z, 1.f));
}

// Render 4D field by operating on it with a 4x3 matrix, where the rows are RGB values (a colour for each dimension).
__global__ void map_to_rgba(float4 * o_buffer, float4 const * _buffer, Resolution _buffer_res, float3 const * _map) {
  int const idx = _buffer_res.idx();
  float4 rgb = make_float4(
    _buffer[idx].x * _map[0].x + _buffer[idx].y * _map[1].x + _buffer[idx].z * _map[2].x + _buffer[idx].w * _map[3].x,
    _buffer[idx].x * _map[0].y + _buffer[idx].y * _map[1].y + _buffer[idx].z * _map[2].y + _buffer[idx].w * _map[3].y,
    _buffer[idx].x * _map[0].z + _buffer[idx].y * _map[1].z + _buffer[idx].z * _map[2].z + _buffer[idx].w * _map[3].z,
    0.75f * (_buffer[idx].x + _buffer[idx].y + _buffer[idx].z + _buffer[idx].w)
  );
  rgb.w = fmin(rgb.x + rgb.y + rgb.z, 1.f);
  o_buffer[idx] = rgb;
}

__global__ void min(float4 * o, float4 const * i, Resolution _buffer_res) {
  __shared__ float4 min_per_block;
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    min_per_block = i[_buffer_res.idx()];
  }
  __syncthreads();
  min_per_block = fminf(min_per_block, i[_buffer_res.idx()]);
  __syncthreads();
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    o[gridDim.x * blockIdx.y + blockIdx.x] = min_per_block;
  }
}

__global__ void max(float4 * o, float4 const * i, Resolution _buffer_res) {
  __shared__ float4 min_per_block;
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    min_per_block = i[_buffer_res.idx()];
  }
  __syncthreads();
  min_per_block = fmaxf(min_per_block, i[_buffer_res.idx()]);
  __syncthreads();
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    o[gridDim.x * blockIdx.y + blockIdx.x] = min_per_block;
  }
}

__global__ void scaleRGB(float4 * io, float4 _min, float4 _max, Resolution _buffer_res) {
  int const idx = _buffer_res.idx();
  io[idx] = (io[idx] - _min) / (_max - _min);
}
