#include "camera_filter.h"
#include "shared.h"

#include <iostream>
#include <stdio.h>

#include "../interface/enums.h"

__global__ void bg_hsl_copy(float * o_bg, Resolution _out_res, uchar3 const * _input, Resolution _in_res, bool _mirror, int _mode);
__global__ void hsl_filter(float * o_fluid_output, float2 * o_velocity_output, float4 * o_render, float * _bg, Resolution _out_res, uchar3 const * _image_input, float2 const * _flow_input, Resolution _in_res, bool _mirror, int _mode, bool _bg_subtract, float _value, float _range);

CameraFilter::CameraFilter(OptimalBlockConfig const & _block_config)
  : __config(_block_config)
  , __last_mode(-1)
{
  format_out << "Constructing Camera Filter Kernel Buffers:" << std::endl;
  OutputIndent indent;
  Allocator alloc;
  __fluid_output.resize(alloc, __config.resolution.total_size());
  __velocity_output.resize(alloc, __config.resolution.total_size());
  __render.resize(alloc, __config.resolution.total_size());
  __bg.resize(alloc, __config.resolution.total_size());
}

void CameraFilter::update(ArrayStructConst<uchar3> const & _camera_data, ArrayStructConst<float2> const & _flow_data, bool _mirror, int _mode, bool _bg_subtract, float _value, float _range) {
  if(__image_input.getSize() != _camera_data.resolution.total_size()) {
    __image_input.resize(Allocator(), _camera_data.resolution.total_size());
    __flow_input.resize(Allocator(), _camera_data.resolution.total_size());
    __bg.resize(Allocator(), _camera_data.resolution.total_size());
  }
  checkCudaErrors(cudaMemcpy(__image_input, _camera_data.data, __image_input.getSizeBytes(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(__flow_input, _flow_data.data, __flow_input.getSizeBytes(), cudaMemcpyHostToDevice));

  int combined_mode = static_cast<int>(_bg_subtract) * FilterMode::NUM + _mode;
  if(__last_mode != combined_mode) {
    __last_mode = combined_mode;
    bg_hsl_copy<<<__config.inner_grid,__config.block>>>(__bg, __config.resolution, __image_input, _camera_data.resolution, _mirror, _mode);
  }

  switch(_mode) {
    case FilterMode::HUE:
    case FilterMode::SATURATION:
    case FilterMode::LIGHTNESS:
      hsl_filter<<<__config.inner_grid,__config.block>>>(__fluid_output, __velocity_output, __render, __bg, __config.resolution, __image_input, __flow_input, _camera_data.resolution, _mirror, _mode, _bg_subtract, _value, _range);
      break;
  }
}

__device__ float3 rgbToHsv(float3 _rgb) {
  float3 hsv = make_float3(0.0f);
  float min = fminf(fminf(_rgb.x, _rgb.y), _rgb.z);
  float max = fmaxf(fmaxf(_rgb.x, _rgb.y), _rgb.z);

  hsv.z = max;

  float delta = max - min;
  if(delta == 0.0f) {
    return hsv;
  } else {
    hsv.y = delta / max;
  }

  if(_rgb.x == max) {
    hsv.x = (_rgb.y - _rgb.z) / delta;
  } else if(_rgb.y == max) {
    hsv.x = 2.0f + (_rgb.z - _rgb.x) / delta;
  } else {
    hsv.x = 4.0f + (_rgb.x - _rgb.y) / delta;
  }
  hsv.x /= 6.0f;
  hsv.x = hsv.x < 0 ? hsv.x + 1 : hsv.x;

  return hsv;
}

__device__ float3 rgbToHsl(float3 _rgb) {
  float3 hsl = make_float3(0.f);
  float min = fminf(fminf(_rgb.x, _rgb.y), _rgb.z);
  float max = fmaxf(fmaxf(_rgb.x, _rgb.y), _rgb.z);

  hsl.z = (max + min) / 2.f;

  float delta = max - min;
  if(delta == 0.f) {
    return hsl;
  } else {
    hsl.y = delta / (1.f - fabsf(max + min - 1.f));
  }

  if(_rgb.x == max) {
    hsl.x = (_rgb.y - _rgb.z) / delta;
  } else if(_rgb.y == max) {
    hsl.x = 2.f + (_rgb.z - _rgb.x) / delta;
  } else {
    hsl.x = 4.f + (_rgb.x - _rgb.y) / delta;
  }
  hsl.x /= 6.f;
  hsl.x = hsl.x < 0 ? hsl.x + 1 : hsl.x;

  return hsl;
}

__device__ int2 map_x_y(Resolution const & _out_res, Resolution const & _in_res, bool _mirror) {
  int2 center = make_int2((_in_res.width - _out_res.width) / 2,   (_in_res.height - _out_res.height) / 2);
  return center + make_int2(_mirror ? _out_res.width - _out_res.x() : _out_res.x(), _out_res.y());
}

__global__ void bg_hsl_copy(float * o_bg, Resolution _out_res, uchar3 const * _input, Resolution _in_res, bool _mirror, int _mode) {
  int2 in_pos = map_x_y(_out_res, _in_res, _mirror);
  float value = -1.0f;
  if(in_pos.x >= 0 && in_pos.x < _in_res.width && in_pos.y >= 0 && in_pos.y < _in_res.height) {
    uchar3 const & rgb = _input[in_pos.x + in_pos.y * _in_res.width];
    float3 hsl = rgbToHsl(make_float3(rgb.x, rgb.y, rgb.z) / 255.0f);
    switch(_mode) {
      case FilterMode::HUE: value = hsl.x; break;
      case FilterMode::SATURATION: value = hsl.y; break;
      case FilterMode::LIGHTNESS: value = hsl.z; break;
      default:
        break;
    }
  }
  o_bg[_out_res.idx()] = value;
}

__global__ void hsl_filter(float * o_fluid_output, float2 * o_velocity_output, float4 * o_render, float * _bg, Resolution _out_res, uchar3 const * _image_input, float2 const * _flow_input, Resolution _in_res, bool _mirror, int _mode, bool _bg_subtract, float _value, float _range) {
  int2 in_pos = map_x_y(_out_res, _in_res, _mirror);
  bool is_fluid = true;
  float2 velocity = make_float2(0.0f, 0.0f);
  if(in_pos.x >= 0 && in_pos.x < _in_res.width && in_pos.y >= 0 && in_pos.y < _in_res.height) {
    uchar3 const & rgb = _image_input[in_pos.x + in_pos.y * _in_res.width];
    float3 hsl = rgbToHsl(make_float3(rgb.x, rgb.y, rgb.z) / 255.0f);
    float value = _bg_subtract ? _bg[_out_res.idx()] : _value;
    switch(_mode) {
      case FilterMode::HUE: is_fluid = fabsf(hsl.x - value) <= (_range / 2.f) || fabsf(hsl.x + 1.f - value) <= (_range / 2.f) || fabsf(hsl.x - 1.f - value) <= (_range / 2.f); break;
      case FilterMode::SATURATION: is_fluid = fabsf(hsl.y - value) <= _range; break;
      case FilterMode::LIGHTNESS: is_fluid = fabsf(hsl.z - value) <= _range; break;
      default:
        break;
    }
    velocity = _flow_input[in_pos.x + in_pos.y * _in_res.width] / 10;
    o_render[_out_res.idx()] = float2_to_hsl(velocity, 1.0f) + (is_fluid ? 0.75f : 0.f);
    o_render[_out_res.idx()].w = 1.f;
  } else {
    o_render[_out_res.idx()] = make_float4(0.0f);
  }
  o_fluid_output[_out_res.idx()] = is_fluid ? 1.f : 0.f;
  o_velocity_output[_out_res.idx()] = velocity;
}
