#include "kernels.cuh"

#include "helper_math.h"

#include <stdio.h>

float const PI = 3.14159265359f;

int const PRESSURE_SOLVER_STEPS = 1000;

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

#define IDX(_dims) \
  int x = blockIdx.x * blockDim.x + threadIdx.x; \
  int y = blockIdx.y * blockDim.y + threadIdx.y; \
  int idx = _dims.x * y + x;

#define BUFFERED_IDX(_dims) \
  IDX(_dims) \
  x += 1; \
  y += 1; \
  idx = (_dims.x + 2) * y + x;

#define STENCIL(_idx, _dims) \
  int4 stencil = _idx + make_int4(1, -1, (_dims.x + 2), -(_dims.x + 2));

#define BLOCK(_dims) \
  dim3 grid, block; \
  block.x = BLOCK_SIZE.x; \
  block.y = BLOCK_SIZE.y; \
  grid.x = _dims.x / BLOCK_SIZE.x; \
  grid.y = _dims.y / BLOCK_SIZE.y;

__global__ void advect_velocity(int2 N, float dt, float2 rd, float2 * velocity) {
  BUFFERED_IDX(N);
  float s = (float)x + 0.5;
  float t = (float)y + 0.5;
  float2 pos = make_float2(s, t) - dt * rd * tex2D(velTex, s, t);
  velocity[idx] = tex2D(velTex, pos.x, pos.y);
}

__global__ void calc_divergence(float2 * _vel, float * _fluid, int2 _dims, float2 _spaceI2, float * o_div) {
  BUFFERED_IDX(_dims);
  STENCIL(idx, _dims);
  o_div[idx] = (_vel[stencil.x].x * _fluid[stencil.x] - _vel[stencil.y].x * _fluid[stencil.y]) * _spaceI2.x
    + (_vel[stencil.z].y * _fluid[stencil.z] - _vel[stencil.w].y * _fluid[stencil.w]) * _spaceI2.y;
}

__global__ void pressure_decay(float * _fluid, int2 _dims, float * io_pres) {
  BUFFERED_IDX(_dims);
  io_pres[idx] *= _fluid[idx] * 0.1f + 0.9f;
}

// Ax = b
__global__ void jacobi_solve(float * _b, float * _validCells, int2 _dims, float alpha, float beta, float * _x, float * o_x) {
  BUFFERED_IDX(_dims);
  STENCIL(idx, _dims);

  float xL = _validCells[stencil.y] > 0 ? _x[stencil.y] : _x[idx];
  float xR = _validCells[stencil.x] > 0 ? _x[stencil.x] : _x[idx];
  float xB = _validCells[stencil.w] > 0 ? _x[stencil.w] : _x[idx];
  float xT = _validCells[stencil.z] > 0 ? _x[stencil.z] : _x[idx];

  o_x[idx] = beta * (xL + xR + xB + xT + alpha * _b[idx]);
}

__global__ void pressure_solve(float * _div, float * _fluid, int2 _dims, float2 _space, float * i_pres, float * o_pres) {
  BUFFERED_IDX(_dims);
  STENCIL(idx, _dims);

  o_pres[idx] = (1.0f / 4.0f) * (
    (4.0f - _fluid[stencil.x] - _fluid[stencil.y] - _fluid[stencil.z] - _fluid[stencil.w]) * i_pres[idx]
    + _fluid[stencil.x] * i_pres[stencil.x]
    + _fluid[stencil.y] * i_pres[stencil.y]
    + _fluid[stencil.z] * i_pres[stencil.z]
    + _fluid[stencil.w] * i_pres[stencil.w]
    - _div[idx] * _space.x * _space.y);
}

__global__ void sub_gradient(float * _pressure, float * _fluid, int2 _dims, float2 _spaceI2, float2 * io_velocity) {
  BUFFERED_IDX(_dims);
  STENCIL(idx, _dims);
  io_velocity[idx] -= _fluid[idx] * _spaceI2 * make_float2( _pressure[stencil.x] - _pressure[stencil.y], _pressure[stencil.z] - _pressure[stencil.w]);
}

__global__ void enforce_slip(float * _fluid, int2 _dims, float2 * io_velocity) {
  BUFFERED_IDX(_dims);
  STENCIL(idx, _dims);
  if(_fluid[idx] > 0.0f) {
    float xvel = _fluid[stencil.x] == 0.0f ? io_velocity[stencil.x].x :
      _fluid[stencil.y] == 0.0f ? io_velocity[stencil.y].x : io_velocity[idx].x;
    float yvel = _fluid[stencil.z] == 0.0f ? io_velocity[stencil.z].y :
    _fluid[stencil.w] == 0.0f ? io_velocity[stencil.w].y : io_velocity[idx].y;
    io_velocity[idx] = make_float2(xvel, yvel);
  } else {
    io_velocity[idx] = make_float2(0.0f, 0.0f);
  }
}

void simulation_step(int2 _dims, float2 _d, float _dt, float * i_fluidCells, float2 * io_velocity, float * o_divergence, float * o_pressure, float * o_buffer) {
  float2 rd = make_float2(1.0f / _d.x, 1.0f / _d.y);
  float2 r2d = make_float2(1.0f / (2.0f * _d.x), 1.0f / (2.0f * _d.y));
  int bufferedSize = (_dims.x + 2) * (_dims.y + 2);

  // copy the input velocity to the velocity texture (for auto interpolation) via the mapped velArray
  cudaMemcpyToArray(velArray, 0, 0, io_velocity, bufferedSize * sizeof(float2), cudaMemcpyDeviceToDevice);

  BLOCK(_dims);
  advect_velocity<<<grid, block>>>(_dims, _dt, rd, io_velocity);
  calc_divergence<<<grid, block>>>(io_velocity, i_fluidCells, _dims, r2d, o_divergence);
  pressure_decay<<<grid, block>>>(i_fluidCells, _dims, o_pressure);
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
    pressure_solve<<<grid, block>>>(o_divergence, i_fluidCells, _dims, _d, o_pressure, o_buffer);
    float * temp = o_pressure;
    o_pressure = o_buffer;
    o_buffer = temp;
  }
  sub_gradient<<<grid, block>>>(o_pressure, i_fluidCells, _dims, r2d, io_velocity);
  enforce_slip<<<grid, block>>>(i_fluidCells, _dims, io_velocity);
}

__global__ void hsv_to_rgba(float2 * _array, int2 _dims, cudaSurfaceObject_t o_surface) {
  BUFFERED_IDX(_dims);

  float h = 6.0f * (atan2f(-_array[idx].x, -_array[idx].y) / (2 * PI) + 0.5);
  float v = powf(_array[idx].x * _array[idx].x + _array[idx].y * _array[idx].y, 0.25f);
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

  surf2Dwrite(rgb, o_surface, (x - 1) * sizeof(float4), (y - 1));
}

void copy_to_surface(float2 * _array, int2 _dims, cudaGraphicsResource_t & _viewCudaResource) {
  cudaGraphicsMapResources(1, &_viewCudaResource); {
    cudaArray_t writeArray;
    cudaGraphicsSubResourceGetMappedArray(&writeArray, _viewCudaResource, 0, 0);
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    cudaCreateSurfaceObject(&writeSurface, &wdsc);
    BLOCK(_dims);
    hsv_to_rgba<<<grid, block>>>(_array, _dims, writeSurface);
    cudaDestroySurfaceObject(writeSurface);
  } cudaGraphicsUnmapResources(1, &_viewCudaResource);

  cudaStreamSynchronize(0);
}

__global__ void d_to_rgba(float * _array, float _mul, int2 _dims, cudaSurfaceObject_t o_surface) {
  BUFFERED_IDX(_dims);
  float pos = (_array[idx] + abs(_array[idx])) / 2.0f;
  float neg = -(_array[idx] - abs(_array[idx])) / 2.0f;
  float4 rgb = make_float4(neg * _mul, pos * _mul, 0.0, 1.0f);
  surf2Dwrite(rgb, o_surface, (x - 1) * sizeof(float4), (y - 1));
}

void copy_to_surface(float * _array, float _mul, int2 _dims, cudaGraphicsResource_t & _viewCudaResource) {
  cudaGraphicsMapResources(1, &_viewCudaResource); {
    cudaArray_t writeArray;
    cudaGraphicsSubResourceGetMappedArray(&writeArray, _viewCudaResource, 0, 0);
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    cudaCreateSurfaceObject(&writeSurface, &wdsc);
    BLOCK(_dims);
    d_to_rgba<<<grid, block>>>(_array, _mul, _dims, writeSurface);
    cudaDestroySurfaceObject(writeSurface);
  } cudaGraphicsUnmapResources(1, &_viewCudaResource);

  cudaStreamSynchronize(0);
}
