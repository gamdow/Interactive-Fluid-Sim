#include "kernels.cuh"

#include "helper_math.h"
#include "utility.hpp"
#include "configuration.cuh"

#include <stdio.h>

float const PI = 3.14159265359f;

static cudaArray * velArray = NULL;
texture<float2, 2> velTex;

void kernels_init(int2 _dims) {
  velTex.filterMode = cudaFilterModeLinear;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();
  cudaMallocArray(&velArray, &channelDesc, _dims.x + 2, _dims.y + 2);
  cudaBindTextureToArray(velTex, velArray, channelDesc);
}

void kernels_shutdown() {
  cudaUnbindTexture(velTex);
  cudaFreeArray(velArray);
}

void copy_to_vel_texture(float2 * _array, int2 _dims) {
  cudaMemcpyToArray(velArray, 0, 0, _array, (_dims.x + 2) * (_dims.y + 2) * sizeof(float2), cudaMemcpyDeviceToDevice);
}

// template<int _NX, int _NY, int _BUFFER>
// struct IndexHelper {
//   __device__ __inline__ int x() const {return blockIdx.x * blockDim.x + threadIdx.x + _BUFFER;}
//   __device__ __inline__ int y() const {return blockIdx.y * blockDim.y + threadIdx.y + _BUFFER;}
//   __device__ __inline__ int i() const {return (_NX + 2 * _BUFFER) * y() + x());}
//   __device__ __inline__ int4 stencil() const {return i() + make_int4(1, -1, (_NX + 2 * _BUFFER), -(_NX + 2 * _BUFFER));}
// };

struct IndexHelper {
  __device__ IndexHelper(int2 _res, int _buffer)
    : x(blockIdx.x * blockDim.x + threadIdx.x + _buffer)
    , y(blockIdx.y * blockDim.y + threadIdx.y + _buffer)
    , idx((_res.x + 2 * _buffer) * y + x)
    , stencil(idx + make_int4(1, -1, (_res.x + 2 * _buffer), -(_res.x + 2 * _buffer)))
    {}
  int x, y, idx;
  int4 stencil;
};



__global__ void advect_velocity(int2 _res, float _dt, float2 _rdx, float2 * o_velocity) {
  IndexHelper ih(_res, BUFFER);
  float s = (float)ih.x + 0.5f;
  float t = (float)ih.y + 0.5f;
  float2 pos = make_float2(s, t) - _dt * _rdx * tex2D(velTex, s, t);
  o_velocity[ih.idx] = tex2D(velTex, pos.x, pos.y);
}

__global__ void calc_divergence(float2 * _vel, float * _fluid, int2 _res, float2 _r2dx, float * o_div) {
  IndexHelper ih(_res, BUFFER);
  auto stencil = ih.stencil;
  o_div[ih.idx] = (_vel[stencil.x].x * _fluid[stencil.x] - _vel[stencil.y].x * _fluid[stencil.y]) * _r2dx.x
    + (_vel[stencil.z].y * _fluid[stencil.z] - _vel[stencil.w].y * _fluid[stencil.w]) * _r2dx.y;
}

__global__ void pressure_decay(float * _fluid, int2 _res, float * io_pres) {

  IndexHelper ih(_res, BUFFER);
  io_pres[ih.idx] *= _fluid[ih.idx] * 0.1f + 0.9f;
}

// Ax = b
__global__ void jacobi_solve(float * _b, float * _validCells, int2 _res, float alpha, float beta, float * _x, float * o_x) {

  IndexHelper ih(_res, BUFFER);
  auto stencil = ih.stencil;

  float xL = _validCells[stencil.y] > 0 ? _x[stencil.y] : _x[ih.idx];
  float xR = _validCells[stencil.x] > 0 ? _x[stencil.x] : _x[ih.idx];
  float xB = _validCells[stencil.w] > 0 ? _x[stencil.w] : _x[ih.idx];
  float xT = _validCells[stencil.z] > 0 ? _x[stencil.z] : _x[ih.idx];

  o_x[ih.idx] = beta * (xL + xR + xB + xT + alpha * _b[ih.idx]);
}

__global__ void pressure_solve(float * _div, float * _fluid, int2 _res, float2 _dx, float * i_pres, float * o_pres) {
IndexHelper ih(_res, BUFFER);
  auto stencil = ih.stencil;
  o_pres[ih.idx] = (1.0f / 4.0f) * (
    (4.0f - _fluid[stencil.x] - _fluid[stencil.y] - _fluid[stencil.z] - _fluid[stencil.w]) * i_pres[ih.idx]
    + _fluid[stencil.x] * i_pres[stencil.x]
    + _fluid[stencil.y] * i_pres[stencil.y]
    + _fluid[stencil.z] * i_pres[stencil.z]
    + _fluid[stencil.w] * i_pres[stencil.w]
    - _div[ih.idx] * _dx.x * _dx.y);
}

__global__ void sub_gradient(float * _pressure, float * _fluid, int2 _res, float2 _r2dx, float2 * io_velocity) {
IndexHelper ih(_res, BUFFER);
  auto stencil = ih.stencil;
  io_velocity[ih.idx] -= _fluid[ih.idx] * _r2dx * make_float2( _pressure[stencil.x] - _pressure[stencil.y], _pressure[stencil.z] - _pressure[stencil.w]);
}

__global__ void enforce_slip(float * _fluid, int2 _res, float2 * io_velocity) {

  IndexHelper ih(_res, BUFFER);
  auto stencil = ih.stencil;
  if(_fluid[ih.idx] > 0.0f) {
    float xvel = _fluid[stencil.x] == 0.0f ? io_velocity[stencil.x].x :
      _fluid[stencil.y] == 0.0f ? io_velocity[stencil.y].x : io_velocity[ih.idx].x;
    float yvel = _fluid[stencil.z] == 0.0f ? io_velocity[stencil.z].y :
    _fluid[stencil.w] == 0.0f ? io_velocity[stencil.w].y : io_velocity[ih.idx].y;
    io_velocity[ih.idx] = make_float2(xvel, yvel);
  } else {
    io_velocity[ih.idx] = make_float2(0.0f, 0.0f);
  }
}

__global__ void hsv_to_rgba(float2 * _array, float _mul, int2 _res, cudaSurfaceObject_t o_surface) {

    IndexHelper ih(_res, BUFFER);

  float h = 6.0f * (atan2f(-_array[ih.idx].x, -_array[ih.idx].y) / (2 * PI) + 0.5);
  float v = powf(_array[ih.idx].x * _array[ih.idx].x + _array[ih.idx].y * _array[ih.idx].y, _mul);
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

  surf2Dwrite(rgb, o_surface, (ih.x - BUFFER) * sizeof(float4), (ih.y - BUFFER));
}

__global__ void d_to_rgba(float * _array, float _mul, int2 _res, cudaSurfaceObject_t o_surface) {

    IndexHelper ih(_res, BUFFER);
  float pos = (_array[ih.idx] + abs(_array[ih.idx])) / 2.0f;
  float neg = -(_array[ih.idx] - abs(_array[ih.idx])) / 2.0f;
  float4 rgb = make_float4(neg * _mul, pos * _mul, 0.0, 1.0f);
  surf2Dwrite(rgb, o_surface, (ih.x - BUFFER) * sizeof(float4), (ih.y - BUFFER));
}
