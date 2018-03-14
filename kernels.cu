#include "kernels.cuh"

#include "helper_math.h"
#include "helper_cuda.h"

#include <iostream> // for host code
#include <stdio.h> // for kernel code

float const PI = 3.14159265359f;

template<class T>
TextureObject<T>::TextureObject()
  : __buffer(nullptr)
  , __pitch(0u)
  , __object(0u)
{}

// Initialise the Texture Object required by advect's interpolated sampling.
template<class T>
void TextureObject<T>::init(int3 _buffer_spec) {
  checkCudaErrors(cudaMallocPitch(&__buffer, &__pitch, sizeof(T) * _buffer_spec.x, _buffer_spec.y));
  cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = __buffer;
  resDesc.res.pitch2D.pitchInBytes = __pitch;
  resDesc.res.pitch2D.width = _buffer_spec.x;
  resDesc.res.pitch2D.height = _buffer_spec.y;
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
  cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  checkCudaErrors(cudaCreateTextureObject(&__object, &resDesc, &texDesc, nullptr));

  std::cout << "f1Object " << __buffer << " " << __pitch << " " << __object << std::endl;
}

template<class T>
void TextureObject<T>::shutdown() {
  checkCudaErrors(cudaDestroyTextureObject(__object));
  checkCudaErrors(cudaFree(__buffer));
}

__global__ void advect_velocity(float2 * o_velocity, cudaTextureObject_t _velocityObj, int3 _buffer_spec, float _dt, float2 _rdx);
__global__ void calc_divergence(float * o_divergence, float2 const * _velocity, float const * _fluid, int3 _buffer_spec, float2 _r2dx);
__global__ void pressure_decay(float * io_pressure, float const * _fluid, int3 _buffer_spec);
__global__ void pressure_solve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, int3 _buffer_spec, float2 _dx);
__global__ void sub_gradient(float2 * io_velocity, float const * _pressure, float const * _fluid, int3 _buffer_spec, float2 _r2dx);
__global__ void enforce_slip(float2 * io_velocity, float const * _fluid, int3 _buffer_spec);
__global__ void hsv_to_rgba(cudaSurfaceObject_t o_surface, float2 const * _array, float _power, int3 _buffer_spec);
__global__ void d_to_rgba(cudaSurfaceObject_t o_surface, float const * _array, float _multiplier, int3 _buffer_spec);

Kernels::Kernels(int2 _dims, int _buffer) {
  std::cout << std::endl;
  reportCapability();
  std::cout << std::endl;
  optimiseBlockSize(_dims, _buffer);
  std::cout << std::endl;
  __f1Object.init(__buffer_spec);
  __f2Object.init(__buffer_spec);
}

Kernels::~Kernels() {
  __f1Object.shutdown();
  __f2Object.shutdown();
}

void Kernels::reportCapability() const {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  for(int dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cout << "CUDA Device: " << dev << ":" << deviceProp.name << std::endl;
    std::cout << "\tCapability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "\tRuntime/Driver: " << runtimeVersion << "/" << driverVersion << std::endl;
  }
}

// Use CUDA's occupancy to determine the optimal blocksize and adjust the screen (and therefore array) resolution to be an integer multiple (then there's no need for bounds checking in the kernels).
void Kernels::optimiseBlockSize(int2 _dims, int _buffer) {
  std::cout << "Desired Resolution: " << _dims.x << " x " << _dims.y << std::endl;
  int N = _dims.x * _dims.y;
  int blockSize, minGridSize;   cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pressure_solve, 0, N);
  __block = dim3(32u, blockSize / 32u);
  std::cout << "Optimal Block: " << __block.x << " x " << __block.y << std::endl;
  __grid = dim3(_dims.x / __block.x, _dims.y / __block.y);
  __dims = make_int2(__grid.x * __block.x, __grid.y * __block.y);
  std::cout << "Adjusted Resolution: " << __dims.x << " x " << __dims.y << std::endl;
  __buffer_spec = make_int3(__dims.x + 2 * _buffer, __dims.y + 2 * _buffer, _buffer);
  __buffered_size = __buffer_spec.x * __buffer_spec.y;
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

__global__ void apply_advection(float * o_data, cudaTextureObject_t _dataObj, float2 const * _velocity, int3 _buffer_spec, float _dt, float2 _rdx) {
  Index ih(_buffer_spec);
  float2 pos = make_float2(ih.x + 0.5f, ih.y + 0.5f) - _dt * _rdx * _velocity[ih.idx];
  o_data[ih.idx] = tex2D<float>(_dataObj, pos.x, pos.y);
}

__global__ void calc_divergence(float * o_divergence, float2 const * _velocity, float const * _fluid, int3 _buffer_spec, float2 _r2dx) {
  Stencil ih(_buffer_spec);
  o_divergence[ih.idx] = (_velocity[ih.stencil.x].x * _fluid[ih.stencil.x] - _velocity[ih.stencil.y].x * _fluid[ih.stencil.y]) * _r2dx.x
    + (_velocity[ih.stencil.z].y * _fluid[ih.stencil.z] - _velocity[ih.stencil.w].y * _fluid[ih.stencil.w]) * _r2dx.y;
}

__global__ void pressure_decay(float * io_pressure, float const * _fluid, int3 _buffer_spec) {
  Index ih(_buffer_spec);
  io_pressure[ih.idx] *= _fluid[ih.idx] * 0.1f + 0.9f;
}

__global__ void pressure_solve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, int3 _buffer_spec, float2 _dx) {
  Stencil ih(_buffer_spec);
  o_pressure[ih.idx] = (1.0f / 4.0f) * (
    (4.0f - _fluid[ih.stencil.x] - _fluid[ih.stencil.y] - _fluid[ih.stencil.z] - _fluid[ih.stencil.w]) * _pressure[ih.idx]
    + _fluid[ih.stencil.x] * _pressure[ih.stencil.x]
    + _fluid[ih.stencil.y] * _pressure[ih.stencil.y]
    + _fluid[ih.stencil.z] * _pressure[ih.stencil.z]
    + _fluid[ih.stencil.w] * _pressure[ih.stencil.w]
    - _divergence[ih.idx] * _dx.x * _dx.y);
}

__global__ void sub_gradient(float2 * io_velocity, float const * _pressure, float const * _fluid, int3 _buffer_spec, float2 _r2dx) {
  Stencil ih(_buffer_spec);
  io_velocity[ih.idx] -= _fluid[ih.idx] * _r2dx * make_float2( _pressure[ih.stencil.x] - _pressure[ih.stencil.y], _pressure[ih.stencil.z] - _pressure[ih.stencil.w]);
}

__global__ void enforce_slip(float2 * io_velocity, float const * _fluid, int3 _buffer_spec) {
  Stencil ih(_buffer_spec);
  if(_fluid[ih.idx] > 0.0f) {
    float xvel = _fluid[ih.stencil.x] * _fluid[ih.stencil.y] == 0.0f
      ? ((1.f - _fluid[ih.stencil.x]) * io_velocity[ih.stencil.x].x + (1.f - _fluid[ih.stencil.y]) * io_velocity[ih.stencil.y].x) / (2.f - _fluid[ih.stencil.x] - _fluid[ih.stencil.y])
      : io_velocity[ih.idx].x;
    float yvel = _fluid[ih.stencil.z] * _fluid[ih.stencil.w] == 0.0f
      ? ((1.f - _fluid[ih.stencil.z]) * io_velocity[ih.stencil.z].y + (1.f - _fluid[ih.stencil.w]) * io_velocity[ih.stencil.w].y) / (2.f - _fluid[ih.stencil.z] - _fluid[ih.stencil.w])
      : io_velocity[ih.idx].y;
    io_velocity[ih.idx] = make_float2(xvel, yvel);
  } else {
    io_velocity[ih.idx] = make_float2(0.0f, 0.0f);
  }
}

__global__ void hsv_to_rgba(cudaSurfaceObject_t o_surface, float2 const * _array, float _power, int3 _buffer_spec) {
  Index ih(_buffer_spec);
  float h = 6.0f * (atan2f(-_array[ih.idx].x, -_array[ih.idx].y) / (2 * PI) + 0.5);
  float v = __powf(_array[ih.idx].x * _array[ih.idx].x + _array[ih.idx].y * _array[ih.idx].y, _power);
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

__global__ void d_to_rgba(cudaSurfaceObject_t o_surface, float const * _array, float _multiplier, int3 _buffer_spec) {
  Index ih(_buffer_spec);
  float pos = (_array[ih.idx] + abs(_array[ih.idx])) / 2.0f;
  float neg = -(_array[ih.idx] - abs(_array[ih.idx])) / 2.0f;
  float4 rgb = make_float4(neg * _multiplier, pos * _multiplier, 0.0, 1.0f);
  surf2Dwrite(rgb, o_surface, (ih.x - _buffer_spec.z) * sizeof(float4), (ih.y - _buffer_spec.z));
}

__global__ void sum_arrays(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2, int3 _buffer_spec) {
  Index ih(_buffer_spec);
  o_array[ih.idx] = _c1 * _array1[ih.idx] + _c2 * _array2[ih.idx];
}

void Kernels::advectVelocity(float2 * io_velocity, float2 _rdx, float _dt) {
  cudaMemcpy2D(__f2Object.__buffer, __f2Object.__pitch, io_velocity, sizeof(float2) * __buffer_spec.x, sizeof(float2) * __buffer_spec.x, __buffer_spec.y, cudaMemcpyDeviceToDevice);
  advect_velocity<<<__grid,__block>>>(io_velocity, __f2Object.__object, __buffer_spec, _dt, _rdx);
}

void Kernels::advectVelocity(float2 * o_velocity, float2 const * _velocity, float2 _rdx, float _dt) {
  cudaMemcpy2D(__f2Object.__buffer, __f2Object.__pitch, _velocity, sizeof(float2) * __buffer_spec.x, sizeof(float2) * __buffer_spec.x, __buffer_spec.y, cudaMemcpyDeviceToDevice);
  advect_velocity<<<__grid,__block>>>(o_velocity, __f2Object.__object, __buffer_spec, _dt, _rdx);
}

void Kernels::applyAdvection(float * io_data, float2 const * _velocity, float _dt, float2 _rdx) {
  cudaMemcpy2D(__f1Object.__buffer, __f1Object.__pitch, io_data, sizeof(float) * __buffer_spec.x, sizeof(float) * __buffer_spec.x, __buffer_spec.y, cudaMemcpyDeviceToDevice);
  apply_advection<<<__grid,__block>>>(io_data, __f1Object.__object, _velocity, __buffer_spec, _dt, _rdx);
}

void Kernels::calcDivergence(float * o_divergence, float2 const * _velocity, float const * _fluid, float2 _r2dx) {
  calc_divergence<<<__grid,__block>>>(o_divergence, _velocity, _fluid, __buffer_spec, _r2dx);
}

void Kernels::pressureDecay(float * io_pressure, float const * _fluid) {
  pressure_decay<<<__grid,__block>>>(io_pressure, _fluid, __buffer_spec);
}

void Kernels::pressureSolve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, float2 _dx) {
  pressure_solve<<<__grid,__block>>>(o_pressure, _pressure, _divergence, _fluid, __buffer_spec, _dx);
}

void Kernels::subGradient(float2 * io_velocity, float const * _pressure, float const * _fluid, float2 _r2dx) {
  sub_gradient<<<__grid,__block>>>(io_velocity, _pressure, _fluid, __buffer_spec, _r2dx);
}

void Kernels::enforceSlip(float2 * io_velocity, float const * _fluid) {
  enforce_slip<<<__grid,__block>>>(io_velocity, _fluid, __buffer_spec);
}

void Kernels::hsv2rgba(cudaSurfaceObject_t o_surface, float2 const * _array, float _power) {
  hsv_to_rgba<<<__grid,__block>>>(o_surface, _array, _power, __buffer_spec);
}

void Kernels::v2rgba(cudaSurfaceObject_t o_surface, float const * _array, float _multiplier) {
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

void Kernels::sum(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2) {
  sum_arrays<<<__grid,__block>>>(o_array, _c1, _array1, _c2, _array2, __buffer_spec);
}
