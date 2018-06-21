#include "kernels.cuh"

// #include <iostream> // for host code
#include <stdio.h> // for kernel code

#include "../cuda/helper_math.h"
#include "../cuda/helper_cuda.h"
#include "../cuda/utility.cuh"

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
  return make_float2(_e2[j].x * _e2[j].x > _e1[j].x * _e1[j].x ? _e1[i].x : _e1[j].x, _e2[j].y * _e2[j].y > _e1[j].y * _e1[j].y ? _e1[i].y : _e1[j].y);
}

__global__ void limit_advection(float2 * o_e, float2 * _e1, float2 * _e2, Resolution _buffer_res) {
  int4 const stencil = _buffer_res.stencil();
  o_e[_buffer_res.idx()] = minmod2(
    minmod2(limit_select(_e1, _e2, _buffer_res.idx(), stencil.x), limit_select(_e1, _e2, _buffer_res.idx(), stencil.y)),
    minmod2(limit_select(_e1, _e2, _buffer_res.idx(), stencil.z), limit_select(_e1, _e2, _buffer_res.idx(), stencil.w)));
}
