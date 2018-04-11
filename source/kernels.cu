#include "kernels.cuh"

#include "helper_math.h"
#include "helper_cuda.h"

// #include <iostream> // for host code
// #include <stdio.h> // for kernel code

float const PI = 3.14159265359f;


__global__ void advect_velocity(float2 * o_velocity, cudaTextureObject_t _velocityObj, BufferSpec _buffer_spec, float _dt, float2 _rdx) {
  float s = (float)_buffer_spec.x() + 0.5f;
  float t = (float)_buffer_spec.y() + 0.5f;
  float2 pos = make_float2(s, t) - _dt * _rdx * tex2D<float2>(_velocityObj, s, t);
  o_velocity[_buffer_spec.idx()] = tex2D<float2>(_velocityObj, pos.x, pos.y);
}

__global__ void calc_divergence(float * o_divergence, float2 const * _velocity, float const * _fluid, BufferSpec _buffer_spec, float2 _rdx) {
  int4 const stencil = _buffer_spec.stencil();
  o_divergence[_buffer_spec.idx()] = (_velocity[stencil.x].x * _fluid[stencil.x] - _velocity[stencil.y].x * _fluid[stencil.y]) * (_rdx.x / 2.0f)
    + (_velocity[stencil.z].y * _fluid[stencil.z] - _velocity[stencil.w].y * _fluid[stencil.w]) * (_rdx.y / 2.0f);
}

// __global__ void calc_divergence_B(float * o_divergence, float2 const * _velocity, float const * _fluid, BufferSpec _buffer_spec, float2 _rdx) {
//   Stencil ih(_buffer_spec);
//   o_divergence[_buffer_spec.idx()] = _fluid[_buffer_spec.idx()] * (
//     (_fluid[stencil.x] * (_velocity[stencil.x].x - _velocity[_buffer_spec.idx()].x) + _fluid[stencil.y] * (_velocity[_buffer_spec.idx()].x - _velocity[stencil.y].x)) * _rdx.x
//     + (_fluid[stencil.z] * (_velocity[stencil.z].y - _velocity[_buffer_spec.idx()].y) + _fluid[stencil.w] * (_velocity[_buffer_spec.idx()].y - _velocity[stencil.w].y )) * _rdx.y
//   );
// }
//
// __global__ void calc_divergence_C(float * o_divergence, float2 const * _velocity, float const * _fluid, BufferSpec _buffer_spec, float2 _rdx) {
//   Stencil ih(_buffer_spec);
//   o_divergence[_buffer_spec.idx()] = 1.0f * (
//     ((_velocity[stencil.x].x - _velocity[_buffer_spec.idx()].x) + (_velocity[_buffer_spec.idx()].x - _velocity[stencil.y].x)) * _rdx.x
//     + ((_velocity[stencil.z].y - _velocity[_buffer_spec.idx()].y) + (_velocity[_buffer_spec.idx()].y - _velocity[stencil.w].y )) * _rdx.y
//   );
// }
//
// __global__ void calc_divergence_D(float * o_divergence, float2 const * _velocity, float const * _fluid, BufferSpec _buffer_spec, float2 _rdx) {
//   Stencil2 ih(_buffer_spec);
//   o_divergence[_buffer_spec.idx()] = (8.f * (_velocity[stencil.x].x - _velocity[stencil.y].x) - (_velocity[stencil2.x].x - _velocity[stencil2.y].x)) * _rdx.x / 12.0f
//     + (8.f * (_velocity[stencil.z].y - _velocity[stencil.w].y) - (_velocity[stencil2.z].y - _velocity[stencil2.w].y)) * _rdx.y / 12.0f;
// }

__global__ void pressure_decay(float * io_pressure, float const * _fluid, BufferSpec _buffer_spec) {
  int const idx = _buffer_spec.idx();
  io_pressure[idx] *= _fluid[idx] * 0.1f + 0.9f;
}

__global__ void pressure_solve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, BufferSpec _buffer_spec, float2 _dx) {
  int const idx = _buffer_spec.idx();
  int4 const stencil = _buffer_spec.stencil();
  o_pressure[idx] = (1.0f / 4.0f) * (
    (4.0f - _fluid[stencil.x] - _fluid[stencil.y] - _fluid[stencil.z] - _fluid[stencil.w]) * _pressure[idx]
    + _fluid[stencil.x] * _pressure[stencil.x]
    + _fluid[stencil.y] * _pressure[stencil.y]
    + _fluid[stencil.z] * _pressure[stencil.z]
    + _fluid[stencil.w] * _pressure[stencil.w]
    - _divergence[idx] * _dx.x * _dx.y);
}

__global__ void sub_gradient(float2 * io_velocity, float const * _pressure, float const * _fluid, BufferSpec _buffer_spec, float2 _rdx) {
  int const idx = _buffer_spec.idx();
  int4 const stencil = _buffer_spec.stencil();
  io_velocity[idx] -= _fluid[idx] * (_rdx / 2.0f) * make_float2( _pressure[stencil.x] - _pressure[stencil.y], _pressure[stencil.z] - _pressure[stencil.w]);
}

__global__ void enforce_slip(float2 * io_velocity, float const * _fluid, BufferSpec _buffer_spec) {
  int const idx = _buffer_spec.idx();
  int4 const stencil = _buffer_spec.stencil();
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

// Render 2D field (i.e. velocity) by treating as HSV (hue=direction, saturation=1, value=magnitude) and converting to RGBA
__global__ void hsv_to_rgba(cudaSurfaceObject_t o_surface, float2 const * _array, float _power, BufferSpec _buffer_spec) {
  int const idx = _buffer_spec.idx();
  float h = 6.0f * (atan2f(-_array[idx].x, -_array[idx].y) / (2 * PI) + 0.5);
  float v = __powf(_array[idx].x * _array[idx].x + _array[idx].y * _array[idx].y, _power);
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
  surf2Dwrite(rgb, o_surface, (_buffer_spec.x() - _buffer_spec.buffer) * sizeof(float4), (_buffer_spec.y() - _buffer_spec.buffer));
}

__global__ void d_to_rgba(cudaSurfaceObject_t o_surface, float const * _array, float _multiplier, BufferSpec _buffer_spec) {
  int const idx = _buffer_spec.idx();
  float pos = (_array[idx] + abs(_array[idx])) / 2.0f;
  float neg = -(_array[idx] - abs(_array[idx])) / 2.0f;
  float4 rgb = make_float4(neg * _multiplier, pos * _multiplier, 0.0, 1.0f);
  surf2Dwrite(rgb, o_surface, (_buffer_spec.x() - _buffer_spec.buffer) * sizeof(float4), (_buffer_spec.y() - _buffer_spec.buffer));
}

// Render 4D field by operating on it with a 4x3 matrix, where the rows are RGB values (a colour for each dimension).
__global__ void float4_to_rgba(cudaSurfaceObject_t o_surface, float4 const * _array, float3 const * _map, BufferSpec _buffer_spec) {
  int const idx = _buffer_spec.idx();
  float4 rgb = make_float4(
    _array[idx].x * _map[0].x + _array[idx].y * _map[1].x + _array[idx].z * _map[2].x + _array[idx].w * _map[3].x,
    _array[idx].x * _map[0].y + _array[idx].y * _map[1].y + _array[idx].z * _map[2].y + _array[idx].w * _map[3].y,
    _array[idx].x * _map[0].z + _array[idx].y * _map[1].z + _array[idx].z * _map[2].z + _array[idx].w * _map[3].z,
    0.5f * (_array[idx].x + _array[idx].y + _array[idx].z + _array[idx].w)
  );
  surf2Dwrite(rgb, o_surface, (_buffer_spec.x() - _buffer_spec.buffer) * sizeof(float4), (_buffer_spec.y() - _buffer_spec.buffer));
}

__global__ void sum_arrays(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2, BufferSpec _buffer_spec) {
  int const idx = _buffer_spec.idx();
  o_array[idx] = _c1 * _array1[idx] + _c2 * _array2[idx];
}

// Ax = b
__global__ void jacobi_solve(float * _b, float * _validCells, BufferSpec _buffer_spec, float alpha, float beta, float * _x, float * o_x) {
  int const idx = _buffer_spec.idx();
  int4 const stencil = _buffer_spec.stencil();
  float xL = _validCells[stencil.y] > 0 ? _x[stencil.y] : _x[idx];
  float xR = _validCells[stencil.x] > 0 ? _x[stencil.x] : _x[idx];
  float xB = _validCells[stencil.w] > 0 ? _x[stencil.w] : _x[idx];
  float xT = _validCells[stencil.z] > 0 ? _x[stencil.z] : _x[idx];
  o_x[idx] = beta * (xL + xR + xB + xT + alpha * _b[idx]);
}
