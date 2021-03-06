#include "visualisation.h"
#include "shared.h"

//float const PI = 3.14159265359f;

void lerp(float & _from, float _to) {
  _from = _to * 0.02f + _from * 0.98f;
}

__global__ void scalar_to_rgba(float4 * o_buffer, float const * _buffer, Resolution _buffer_res, float _multiplier);
// Render 2D field (i.e. velocity) by treating as HSV (hue=direction, saturation=1, value=magnitude) and converting to RGBA
__global__ void vector_field_to_rgba(float4 * o_buffer, float2 const * _buffer, Resolution _buffer_res, float _power);
// Render 4D field by operating on it with a 4x3 matrix, where the rows are RGB values (a colour for each dimension).
__global__ void map_to_rgba(float4 * o_buffer, float4 const * _buffer, Resolution _buffer_res, float3 const * _map);
__global__ void min(float4 * o, float4 const * i, Resolution _buffer_res);
__global__ void max(float4 * o, float4 const * i, Resolution _buffer_res);
__global__ void scaleRGB(float4 * io, float4 _min, float4 _max, Resolution _buffer_res);
__global__ void copy_to_surface(cudaSurfaceObject_t o_surface, Resolution _surface_res, float4 const * _buffer, Resolution _buffer_res);

VisualisationWrapper::VisualisationWrapper(OptimalBlockConfig const & _block_config)
  : __config(_block_config)
{
  format_out << "Constructing Visualisation Kernel Buffers:" << std::endl;
  OutputIndent indent;
  Allocator alloc;
  __min.resize(alloc, 1u);
  __max.resize(alloc, 1u);
  __f4reduce.resize(alloc, __config.inner_grid.x * __config.inner_grid.y);
  __rgba.resize(alloc, __config.resolution.total_size());
}

void VisualisationWrapper::visualise(float const * _buffer) {
  scalar_to_rgba<<<__config.inner_grid,__config.block>>>(__rgba, _buffer, __config.resolution, 1.0f);
}

void VisualisationWrapper::visualise(float2 const * _buffer) {
  vector_field_to_rgba<<<__config.inner_grid,__config.block>>>(__rgba, _buffer, __config.resolution, 0.8f);
}

void VisualisationWrapper::visualise(float4 const * _buffer, float3 const * _map) {
  map_to_rgba<<<__config.inner_grid,__config.block>>>(__rgba, _buffer, __config.resolution, _map);
}

void VisualisationWrapper::minMaxReduce(float4 & o_min, float4 & o_max, float4 const * _array) {
  min<<<__config.inner_grid, __config.block>>>(__f4reduce, _array, __config.resolution);
  min<<<1, __config.inner_grid>>>(__min.device(), __f4reduce, Resolution(__config.inner_grid.x, __config.inner_grid.y, 0));
  __min.copyDeviceToHost();
  o_min = __min[0];
  max<<<__config.inner_grid, __config.block>>>(__f4reduce, _array, __config.resolution);
  max<<<1, __config.inner_grid>>>(__max.device(), __f4reduce, Resolution(__config.inner_grid.x, __config.inner_grid.y, 0));
  __max.copyDeviceToHost();
  o_max = __max[0];
}

void VisualisationWrapper::scale(float4 * o_array, float4 _min, float4 _max) {
  scaleRGB<<<__config.inner_grid, __config.block>>>(o_array, _min, _max, __config.resolution);
}

void VisualisationWrapper::adjustBrightness(float4 & io_min, float4 & io_max, bool _instant) {
  float4 min4, max4;
  minMaxReduce(min4, max4, __rgba);
  if(_instant) {
    io_min.x = fmaxf(fminf(fminf(min4.x, min4.y), min4.z), 0.0f);
    io_max.x = fminf(fmaxf(fmaxf(max4.x, max4.y), max4.z), 1.0f);
    io_min.w = fmaxf(min4.w, 0.0f);
    io_max.w = fminf(max4.w, 1.0f);
  } else {
    lerp(io_min.x, fmaxf(fminf(fminf(min4.x, min4.y), min4.z), 0.0f));
    lerp(io_max.x, fminf(fmaxf(fmaxf(max4.x, max4.y), max4.z), 1.0f));
    lerp(io_min.w, fmaxf(min4.w, 0.0f));
    lerp(io_max.w, fminf(max4.w, 1.0f));
  }
  min4 = make_float4(make_float3(io_min.x), io_min.w);
  max4 = make_float4(make_float3(io_max.x), io_max.w);
  scale(__rgba, min4, max4);
}

void VisualisationWrapper::writeToSurfaceImpl(cudaSurfaceObject_t o_surface, Resolution const & _surface_res) const {
  copy_to_surface<<<__config.inner_grid, __config.block>>>(o_surface, _surface_res, __rgba, __config.resolution);
}

__global__ void scalar_to_rgba(float4 * o_buffer, float const * _buffer, Resolution _buffer_res, float _multiplier) {
  int const idx = _buffer_res.idx();
  float pos = (_buffer[idx] + abs(_buffer[idx])) / 2.0f;
  float neg = -(_buffer[idx] - abs(_buffer[idx])) / 2.0f;
  o_buffer[idx] = make_float4(neg * _multiplier, pos * _multiplier, 0.0, abs(_buffer[idx]));
}

// Render 2D field (i.e. velocity) by treating as HSV (hue=direction, saturation=1, value=magnitude) and converting to RGBA
__global__ void vector_field_to_rgba(float4 * o_buffer, float2 const * _buffer, Resolution _buffer_res, float _power) {
  int const idx = _buffer_res.idx();
  o_buffer[idx] = float2_to_hsl(_buffer[idx], _power);
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
  __shared__ float4 per_block;
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    per_block = i[_buffer_res.idx()];
  }
  __syncthreads();
  per_block = fminf(per_block, i[_buffer_res.idx()]);
  __syncthreads();
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    o[gridDim.x * blockIdx.y + blockIdx.x] = per_block;
  }
}

__global__ void max(float4 * o, float4 const * i, Resolution _buffer_res) {
  __shared__ float4 per_block;
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    per_block = i[_buffer_res.idx()];
  }
  __syncthreads();
  per_block = fmaxf(per_block, i[_buffer_res.idx()]);
  __syncthreads();
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    o[gridDim.x * blockIdx.y + blockIdx.x] = per_block;
  }
}

__global__ void scaleRGB(float4 * io, float4 _min, float4 _max, Resolution _buffer_res) {
  int const idx = _buffer_res.idx();
  io[idx] = (io[idx] - _min) / (_max - _min);
}

__global__ void copy_to_surface(cudaSurfaceObject_t o_surface, Resolution _surface_res, float4 const * _buffer, Resolution _buffer_res) {
  surf2Dwrite(_buffer[_buffer_res.idx()], o_surface, (_buffer_res.x()) * sizeof(float4), _buffer_res.y());
}
