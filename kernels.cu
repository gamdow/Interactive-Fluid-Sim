#include "kernels.cuh"

#include "helper_math.h"
#include "utility.hpp"
#include "configuration.cuh"

#include <stdio.h>

float const PI = 3.14159265359f;

Kernels::Kernels(int2 _dims, int _buffer, dim3 _block_size)
  : __dims(_dims)
  , __buffer_spec(make_int3(_dims.x + 2 * _buffer, _dims.y + 2 * _buffer, _buffer))
  , __block(_block_size)
  , __grid(_dims.x / _block_size.x, _dims.y / _block_size.y)
{
  cudaMallocPitch(&__tex_buffer, &__tex_pitch, sizeof(float2) * __buffer_spec.x, __buffer_spec.y);

  cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = __tex_buffer;
  resDesc.res.pitch2D.pitchInBytes = __tex_pitch;
  resDesc.res.pitch2D.width = __buffer_spec.x;
  resDesc.res.pitch2D.height = __buffer_spec.y;
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float2>();

  cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  cudaCreateTextureObject(&__tex_object, &resDesc, &texDesc, nullptr);
}

Kernels::~Kernels() {
  cudaDestroyTextureObject(__tex_object);
  cudaFree(__tex_buffer);
}

struct Index {
  __device__ Index(int3 _buffer_spec)
    : x(blockIdx.x * blockDim.x + threadIdx.x + _buffer_spec.z)
    , y(blockIdx.y * blockDim.y + threadIdx.y + _buffer_spec.z)
    , idx(_buffer_spec.x * y + x)
  {}
  int x, y, idx;
};

struct Stencil :public Index {
  __device__ Stencil(int3 _buffer_spec)
    : Index(_buffer_spec)
    , stencil(idx + make_int4(1, -1, _buffer_spec.x, -_buffer_spec.x))
  {}
  int4 stencil;
};

__global__ void advect_velocity(float2 * o_velocity, cudaTextureObject_t _velocityObj, int3 _buffer_spec, float _dt, float2 _rdx) {
  Index ih(_buffer_spec);
  float s = (float)ih.x + 0.5f;
  float t = (float)ih.y + 0.5f;
  float2 pos = make_float2(s, t) - _dt * _rdx * tex2D<float2>(_velocityObj, s, t);
  o_velocity[ih.idx] = tex2D<float2>(_velocityObj, pos.x, pos.y);
}

__global__ void calc_divergence(float * o_divergence, float2 * _velocity, float * _fluid, int3 _buffer_spec, float2 _r2dx) {
  Stencil ih(_buffer_spec);
  o_divergence[ih.idx] = (_velocity[ih.stencil.x].x * _fluid[ih.stencil.x] - _velocity[ih.stencil.y].x * _fluid[ih.stencil.y]) * _r2dx.x
    + (_velocity[ih.stencil.z].y * _fluid[ih.stencil.z] - _velocity[ih.stencil.w].y * _fluid[ih.stencil.w]) * _r2dx.y;
}

__global__ void pressure_decay(float * io_pressure, float * _fluid, int3 _buffer_spec) {
  Index ih(_buffer_spec);
  io_pressure[ih.idx] *= _fluid[ih.idx] * 0.1f + 0.9f;
}

__global__ void pressure_solve(float * o_pressure, float * _pressure, float * _divergence, float * _fluid, int3 _buffer_spec, float2 _dx) {
  Stencil ih(_buffer_spec);
  o_pressure[ih.idx] = (1.0f / 4.0f) * (
    (4.0f - _fluid[ih.stencil.x] - _fluid[ih.stencil.y] - _fluid[ih.stencil.z] - _fluid[ih.stencil.w]) * _pressure[ih.idx]
    + _fluid[ih.stencil.x] * _pressure[ih.stencil.x]
    + _fluid[ih.stencil.y] * _pressure[ih.stencil.y]
    + _fluid[ih.stencil.z] * _pressure[ih.stencil.z]
    + _fluid[ih.stencil.w] * _pressure[ih.stencil.w]
    - _divergence[ih.idx] * _dx.x * _dx.y);
}

__global__ void sub_gradient(float2 * io_velocity, float * _pressure, float * _fluid, int3 _buffer_spec, float2 _r2dx) {
  Stencil ih(_buffer_spec);
  io_velocity[ih.idx] -= _fluid[ih.idx] * _r2dx * make_float2( _pressure[ih.stencil.x] - _pressure[ih.stencil.y], _pressure[ih.stencil.z] - _pressure[ih.stencil.w]);
}

__global__ void enforce_slip(float2 * io_velocity, float * _fluid, int3 _buffer_spec) {
  Stencil ih(_buffer_spec);
  if(_fluid[ih.idx] > 0.0f) {
    float xvel = _fluid[ih.stencil.x] == 0.0f ? io_velocity[ih.stencil.x].x :
      _fluid[ih.stencil.y] == 0.0f ? io_velocity[ih.stencil.y].x : io_velocity[ih.idx].x;
    float yvel = _fluid[ih.stencil.z] == 0.0f ? io_velocity[ih.stencil.z].y :
    _fluid[ih.stencil.w] == 0.0f ? io_velocity[ih.stencil.w].y : io_velocity[ih.idx].y;
    io_velocity[ih.idx] = make_float2(xvel, yvel);
  } else {
    io_velocity[ih.idx] = make_float2(0.0f, 0.0f);
  }
}

__global__ void hsv_to_rgba(cudaSurfaceObject_t o_surface, float2 * _array, float _power, int3 _buffer_spec) {
  Index ih(_buffer_spec);
  float h = 6.0f * (atan2f(-_array[ih.idx].x, -_array[ih.idx].y) / (2 * PI) + 0.5);
  float v = powf(_array[ih.idx].x * _array[ih.idx].x + _array[ih.idx].y * _array[ih.idx].y, _power);
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
  surf2Dwrite(rgb, o_surface, (ih.x - _buffer_spec.z) * sizeof(float4), (ih.y - _buffer_spec.z));
}

__global__ void d_to_rgba(cudaSurfaceObject_t o_surface, float * _array, float _multiplier, int3 _buffer_spec) {
  Index ih(_buffer_spec);
  float pos = (_array[ih.idx] + abs(_array[ih.idx])) / 2.0f;
  float neg = -(_array[ih.idx] - abs(_array[ih.idx])) / 2.0f;
  float4 rgb = make_float4(neg * _multiplier, pos * _multiplier, 0.0, 1.0f);
  surf2Dwrite(rgb, o_surface, (ih.x - _buffer_spec.z) * sizeof(float4), (ih.y - _buffer_spec.z));
}

void Kernels::advectVelocity(float2 * io_velocity, float2 _rdx, float _dt) {
  cudaMemcpy2D(__tex_buffer, __tex_pitch, io_velocity, sizeof(float2) * __buffer_spec.x, sizeof(float2) * __buffer_spec.x, __buffer_spec.y, cudaMemcpyHostToDevice);
  advect_velocity<<<__grid,__block>>>(io_velocity, __tex_object, __buffer_spec, _dt, _rdx);
}

void Kernels::calcDivergence(float * o_divergence, float2 * _velocity, float * _fluid, float2 _r2dx) {
  calc_divergence<<<__grid,__block>>>(o_divergence, _velocity, _fluid, __buffer_spec, _r2dx);
}

void Kernels::pressureDecay(float * io_pressure, float * _fluid) {
  pressure_decay<<<__grid,__block>>>(io_pressure, _fluid, __buffer_spec);
}

void Kernels::pressureSolve(float * o_pressure, float * _pressure, float * _divergence, float * _fluid, float2 _dx) {
  pressure_solve<<<__grid,__block>>>(o_pressure, _pressure, _divergence, _fluid, __buffer_spec, _dx);
}

void Kernels::subGradient(float2 * io_velocity, float * _pressure, float * _fluid, float2 _r2dx) {
  sub_gradient<<<__grid,__block>>>(io_velocity, _pressure, _fluid, __buffer_spec, _r2dx);
}

void Kernels::enforceSlip(float2 * io_velocity, float * _fluid) {
  enforce_slip<<<__grid,__block>>>(io_velocity, _fluid, __buffer_spec);
}

void Kernels::hsv2rgba(cudaSurfaceObject_t o_surface, float2 * _array, float _power) {
  hsv_to_rgba<<<__grid,__block>>>(o_surface, _array, _power, __buffer_spec);
}

void Kernels::v2rgba(cudaSurfaceObject_t o_surface, float * _array, float _multiplier) {
  d_to_rgba<<<__grid,__block>>>(o_surface, _array, _multiplier, __buffer_spec);
}

// Ax = b
__global__ void jacobi_solve(float * _b, float * _validCells, int3 _buffer_spec, float alpha, float beta, float * _x, float * o_x) {
  Stencil ih(_buffer_spec);
  float xL = _validCells[ih.stencil.y] > 0 ? _x[ih.stencil.y] : _x[ih.idx];
  float xR = _validCells[ih.stencil.x] > 0 ? _x[ih.stencil.x] : _x[ih.idx];
  float xB = _validCells[ih.stencil.w] > 0 ? _x[ih.stencil.w] : _x[ih.idx];
  float xT = _validCells[ih.stencil.z] > 0 ? _x[ih.stencil.z] : _x[ih.idx];
  o_x[ih.idx] = beta * (xL + xR + xB + xT + alpha * _b[ih.idx]);
}
