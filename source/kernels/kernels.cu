#include "kernels.cuh"

// #include <iostream> // for host code
#include <stdio.h> // for kernel code

#include "../cuda/helper_math.h"
#include "../cuda/helper_cuda.h"
#include "../cuda/utility.cuh"

float const PI = 3.14159265359f;

template<typename T>
__device__
inline T lerp(T a, T b, T l) {
  //return (1. - l) * a + l * b;
  return fma(l, b, fma(-l, a, a));
}

__global__ void advect_velocity(float2 * o_velocity, cudaTextureObject_t _velocityObj, Resolution _buffer_res, float _dt, float2 _rdx) {
  float s = (float)_buffer_res.x() + 0.5f;
  float t = (float)_buffer_res.y() + 0.5f;
  float2 pos = make_float2(s, t) - _dt * _rdx * tex2D<float2>(_velocityObj, s, t);
  o_velocity[_buffer_res.idx()] = tex2D<float2>(_velocityObj, pos.x, pos.y);
}

template<typename T>
__global__ void apply_advection(T * o_data, cudaTextureObject_t _dataObj, float2 const * _velocity, float const * _fluid, Resolution _buffer_res, float _dt, float2 _rdx) {
  if(_fluid[_buffer_res.idx()] > 0.0f) {
    float2 pos = make_float2(_buffer_res.x() + 0.5f, _buffer_res.y() + 0.5f) - _dt * _rdx * _velocity[_buffer_res.idx()];
    o_data[_buffer_res.idx()] = tex2D<T>(_dataObj, pos.x, pos.y);
  } else {
    o_data[_buffer_res.idx()] *= 0.9f;
  }
}

#define TEMPLATE(T) template __global__ void apply_advection(T * o_data, cudaTextureObject_t _dataObj, float2 const * _velocity, float const * _fluid, Resolution _buffer_res, float _dt, float2 _rdx);
TEMPLATE(float)
TEMPLATE(float2)
TEMPLATE(float4)
#undef TEMPLATE

__global__ void calc_divergence(float * o_divergence, float2 const * _velocity, float const * _fluid, Resolution _buffer_res, float2 _rdx) {
  int4 const stencil = _buffer_res.stencil();
  o_divergence[_buffer_res.idx()] = (_velocity[stencil.x].x - _velocity[stencil.y].x) * (_rdx.x / 2.0f) + (_velocity[stencil.z].y - _velocity[stencil.w].y) * (_rdx.y / 2.0f);
}

__global__ void pressure_decay(float * io_pressure, float const * _fluid, Resolution _buffer_res) {
  int const idx = _buffer_res.idx();
  io_pressure[idx] *= _fluid[idx] * 0.1f + 0.9f;
}

__global__ void pressure_solve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, Resolution _buffer_res, float2 _dx) {
  int const idx = _buffer_res.idx();
  int4 const stencil = _buffer_res.stencil();
  float pR = lerp(_pressure[idx], _pressure[stencil.x], _fluid[stencil.x]);
  float pL = lerp(_pressure[idx], _pressure[stencil.y], _fluid[stencil.y]);
  float pU = lerp(_pressure[idx], _pressure[stencil.z], _fluid[stencil.z]);
  float pD = lerp(_pressure[idx], _pressure[stencil.w], _fluid[stencil.w]);
  o_pressure[idx] = (1.0f / 4.0f) * (pR + pL + pU + pD
    - _divergence[idx] * _dx.x * _dx.y);
}

__global__ void sub_gradient(float2 * io_velocity, float const * _pressure, float const * _fluid, Resolution _buffer_res, float2 _rdx) {
  int const idx = _buffer_res.idx();
  int4 const stencil = _buffer_res.stencil();
  float pR = lerp(_pressure[idx], _pressure[stencil.x], _fluid[stencil.x]);
  float pL = lerp(_pressure[idx], _pressure[stencil.y], _fluid[stencil.y]);
  float pU = lerp(_pressure[idx], _pressure[stencil.z], _fluid[stencil.z]);
  float pD = lerp(_pressure[idx], _pressure[stencil.w], _fluid[stencil.w]);
  io_velocity[idx] -= _fluid[idx] * (_rdx / 2.0f) * make_float2(pR - pL, pU - pD);
}

__global__ void enforce_slip(float2 * io_velocity, float const * _fluid, Resolution _buffer_res) {
  int const idx = _buffer_res.idx();
  int4 const stencil = _buffer_res.stencil();
  if(_fluid[idx] > 0.0f) {
    float xvel = _fluid[stencil.x] * _fluid[stencil.y] == 0.0f
      ? ((1.f - _fluid[stencil.x]) * io_velocity[stencil.x].x + (1.f - _fluid[stencil.y]) * io_velocity[stencil.y].x) / (2.f - _fluid[stencil.x] - _fluid[stencil.y])
      : io_velocity[idx].x;
    float yvel = _fluid[stencil.z] * _fluid[stencil.w] == 0.0f
      ? ((1.f - _fluid[stencil.z]) * io_velocity[stencil.z].y + (1.f - _fluid[stencil.w]) * io_velocity[stencil.w].y) / (2.f - _fluid[stencil.z] - _fluid[stencil.w])
      : io_velocity[idx].y;
    io_velocity[idx] = make_float2(xvel, yvel);
  } else {
    io_velocity[idx] = make_float2(0.0f, 0.0f);
  }
}

__global__ void d_to_rgba(float4 * o_buffer, float const * _buffer, Resolution _buffer_res, float _multiplier) {
  int const idx = _buffer_res.idx();
  float pos = (_buffer[idx] + abs(_buffer[idx])) / 2.0f;
  float neg = -(_buffer[idx] - abs(_buffer[idx])) / 2.0f;
  o_buffer[idx] = make_float4(neg * _multiplier, pos * _multiplier, 0.0, 1.0f);
}

// __global__ void d_to_rgba(cudaSurfaceObject_t o_surface, Resolution _surface_res, float const * _buffer, Resolution _buffer_res, float _multiplier) {
//   int const idx = _buffer_res.idx();
//   float pos = (_buffer[idx] + abs(_buffer[idx])) / 2.0f;
//   float neg = -(_buffer[idx] - abs(_buffer[idx])) / 2.0f;
//   float4 rgb = make_float4(neg * _multiplier, pos * _multiplier, 0.0, 1.0f);
//   rgb.w = fmin(rgb.x + rgb.y + rgb.z, 1.f);
//   int buffer_diff = _surface_res.buffer - _buffer_res.buffer;
//   surf2Dwrite(rgb, o_surface, (_buffer_res.x() + buffer_diff) * sizeof(float4), _buffer_res.y() + buffer_diff);
// }

// Render 2D field (i.e. velocity) by treating as HSV (hue=direction, saturation=1, value=magnitude) and converting to RGBA
__global__ void hsv_to_rgba(float4 * o_buffer, float2 const * _buffer, Resolution _buffer_res, float _power) {
  int const idx = _buffer_res.idx();
  float h = 6.0f * (atan2f(-_buffer[idx].x, -_buffer[idx].y) / (2 * PI) + 0.5);
  float v = __powf(_buffer[idx].x * _buffer[idx].x + _buffer[idx].y * _buffer[idx].y, _power);
  float hi = floorf(h);
  float f = h - hi;
  float q = v * (1 - f);
  float t = v * f;
  float4 rgb = {.0f, .0f, .0f, 1.0f};
  if(hi == 0.0f || hi == 6.0f) {
    rgb.x = v;
    rgb.y = t;
	} else if(hi == 1.0f) {
    rgb.x = q;
    rgb.y = v;
	} else if(hi == 2.0f) {
		rgb.y = v;
    rgb.z = t;
	} else if(hi == 3.0f) {
		rgb.y = q;
    rgb.z = v;
	} else if(hi == 4.0f) {
    rgb.x = t;
    rgb.z = v;
	} else {
    rgb.x = v;
    rgb.z = q;
  }
  rgb.w = fmin(rgb.x + rgb.y + rgb.z, 1.f);
  o_buffer[idx] = rgb;
  // int buffer_diff = _surface_res.buffer - _buffer_res.buffer;
  // surf2Dwrite(rgb, o_surface, (_buffer_res.x() + buffer_diff) * sizeof(float4), _buffer_res.y() + buffer_diff);
}

// Render 4D field by operating on it with a 4x3 matrix, where the rows are RGB values (a colour for each dimension).
__global__ void float4_to_rgba(float4 * o_buffer, float4 const * _buffer, Resolution _buffer_res, float3 const * _map) {
  int const idx = _buffer_res.idx();
  float4 rgb = make_float4(
    _buffer[idx].x * _map[0].x + _buffer[idx].y * _map[1].x + _buffer[idx].z * _map[2].x + _buffer[idx].w * _map[3].x,
    _buffer[idx].x * _map[0].y + _buffer[idx].y * _map[1].y + _buffer[idx].z * _map[2].y + _buffer[idx].w * _map[3].y,
    _buffer[idx].x * _map[0].z + _buffer[idx].y * _map[1].z + _buffer[idx].z * _map[2].z + _buffer[idx].w * _map[3].z,
    0.75f * (_buffer[idx].x + _buffer[idx].y + _buffer[idx].z + _buffer[idx].w)
  );
  rgb.w = fmin(rgb.x + rgb.y + rgb.z, 1.f);
  o_buffer[idx] = rgb;
  // int buffer_diff = _surface_res.buffer - _buffer_res.buffer;
  // surf2Dwrite(rgb, o_surface, (_buffer_res.x() + buffer_diff) * sizeof(float4), _buffer_res.y() + buffer_diff);
}

__global__ void copy_to_surface(cudaSurfaceObject_t o_surface, Resolution _surface_res, float4 const * _buffer, Resolution _buffer_res) {
  int buffer_diff = _surface_res.buffer - _buffer_res.buffer;
  surf2Dwrite(_buffer[_buffer_res.idx()], o_surface, (_buffer_res.x() + buffer_diff) * sizeof(float4), _buffer_res.y() + buffer_diff);
}

__global__ void sum_arrays(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2, Resolution _buffer_res) {
  int const idx = _buffer_res.idx();
  o_array[idx] = _c1 * _array1[idx] + _c2 * _array2[idx];
}

__device__ inline float minmod(float a, float b) {
  return a * b > 0
    ? (a > 0
      ? (a < b ? a : b)
      : (a > b ? a : b))
    : 0;
}

__device__ inline float2 minmod2(float2 a, float2 b) {
  return make_float2(minmod(a.x, b.x), minmod(a.y, b.y));
}

__device__ inline float2 limit_select(float2 * _e1, float2 * _e2, int i, int j) {
  return make_float2(_e2[j].x * _e2[j].x > _e1[j].x * _e1[j].x ? _e1[j].x : _e1[i].x, _e2[j].y * _e2[j].y > _e1[j].y * _e1[j].y ? _e1[j].y : _e1[i].y);
}

__global__ void limit_advection(float2 * o_e, float2 * _e1, float2 * _e2, Resolution _buffer_res) {
  int4 const stencil = _buffer_res.stencil();
  o_e[_buffer_res.idx()] = minmod2(
    minmod2(limit_select(_e1, _e2, _buffer_res.idx(), stencil.x), limit_select(_e1, _e2, _buffer_res.idx(), stencil.y)),
    minmod2(limit_select(_e1, _e2, _buffer_res.idx(), stencil.z), limit_select(_e1, _e2, _buffer_res.idx(), stencil.w)));
}

// // Ax = b
// __global__ void jacobi_solve(float * _b, float * _validCells, Resolution _buffer_res, float alpha, float beta, float * _x, float * o_x) {
//   int const idx = _buffer_res.idx();
//   int4 const stencil = _buffer_res.stencil();
//   float xL = _validCells[stencil.y] > 0 ? _x[stencil.y] : _x[idx];
//   float xR = _validCells[stencil.x] > 0 ? _x[stencil.x] : _x[idx];
//   float xB = _validCells[stencil.w] > 0 ? _x[stencil.w] : _x[idx];
//   float xT = _validCells[stencil.z] > 0 ? _x[stencil.z] : _x[idx];
//   o_x[idx] = beta * (xL + xR + xB + xT + alpha * _b[idx]);
// }
