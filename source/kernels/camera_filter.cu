#include "camera_filter.h"

#include "../interface/enums.h"

__global__ void hsl_filter(float * o_output, uchar * o_render, Resolution _out_res, uchar3 const * _input, Resolution _in_res, int _mode, float _value, float _range);
__global__ void bg_subtract(float * o_output, uchar * o_render, Resolution _out_res, uchar3 const * _bg, uchar3 const * _input, Resolution _in_res, float _range);

CameraFilter::CameraFilter(OptimalBlockConfig const & _block_config, int _buffer_width)
  : KernelWrapper(_block_config, _buffer_width)
  , __last_mode(-1)
{
  format_out << "Constructing Camera Filter Kernel Buffers:" << std::endl;
  OutputIndent indent;
  Allocator alloc;
  __output.resize(alloc, buffer_resolution().size);
  __render.resize(alloc, buffer_resolution().size);
}

void CameraFilter::update(ArrayStructConst<uchar3> _camera_data, int _mode, float _value, float _range) {
  if(__input.getSize() != _camera_data.resolution.size) {
    __input.resize(Allocator(), _camera_data.resolution.size);
    __bg.resize(Allocator(), _camera_data.resolution.size);
  }
  checkCudaErrors(cudaMemcpy(__input, _camera_data.data, __input.getSizeBytes(), cudaMemcpyHostToDevice));

  switch(_mode) {
    case FilterMode::HUE:
    case FilterMode::SATURATION:
    case FilterMode::LIGHTNESS:
      hsl_filter<<<grid(),block()>>>(__output, __render, buffer_resolution(), __input, _camera_data.resolution, _mode, _value, _range);
      break;
    case FilterMode::BG_SUBTRACT_TRAINED:
      if(__last_mode != _mode) {
        __bg = __input;
      }
      bg_subtract<<<grid(),block()>>>(__output, __render, buffer_resolution(), __bg, __input, _camera_data.resolution, _range);
      break;
  }
  __last_mode = _mode;
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

__device__ void lerpy(float & _from, float _to) {
  _from = _to * 0.5f + _from * 0.5f;
}

__device__ float hysterises(float _old, float _new) {
  return _old > 0.5f
    ? (_new > 0.1f ? 1.f : 0.f)
    : (_new > 0.9f ? 1.f : 0.f);
}

__global__ void hsl_filter(float * o_output, uchar * o_render, Resolution _out_res, uchar3 const * _input, Resolution _in_res, int _mode, float _value, float _range) {
  int x = (_in_res.width - _out_res.width) / 2 + _out_res.x();
  int y = (_in_res.height - _out_res.height) / 2 + _out_res.y();
  bool is_fluid = true;
  if(x >= 0 && x < _in_res.width && y >= 0 && y < _in_res.height) {
    uchar3 const & rgb = _input[x + y * _in_res.width];
    float3 hsl = rgbToHsl(make_float3(rgb.x, rgb.y, rgb.z) / 255.0f);
    switch(_mode) {
      case FilterMode::HUE: is_fluid = fabsf(hsl.x - _value) <= (_range / 2.f) || fabsf(hsl.x + 1.f - _value) <= (_range / 2.f) || fabsf(hsl.x - 1.f - _value) <= (_range / 2.f); break;
      case FilterMode::SATURATION: is_fluid = fabsf(hsl.y - _value) <= _range; break;
      case FilterMode::LIGHTNESS: is_fluid = fabsf(hsl.z - _value) <= _range; break;
      default:
        break;
    }
  }
  o_output[_out_res.idx()] = is_fluid ? 1.f : 0.f;
  o_render[_out_res.idx()] = is_fluid ? 255 : 0;
}

__global__ void bg_subtract(float * o_output, uchar * o_render,  Resolution _out_res, uchar3 const * _bg, uchar3 const * _input, Resolution _in_res, float _range) {
  int x = (_in_res.width - _out_res.width) / 2 + _out_res.x();
  int y = (_in_res.height - _out_res.height) / 2 + _out_res.y();
  bool is_fluid = true;
  if(x >= 0 && x < _in_res.width && y >= 0 && y < _in_res.height) {
    uchar3 const & rgb = _input[x + y * _in_res.width];
    uchar3 const & bg_rgb = _bg[x + y * _in_res.width];
    float3 diff = (make_float3(rgb.x, rgb.y, rgb.z) - make_float3(bg_rgb.x, bg_rgb.y, bg_rgb.z)) / 255.f;
    is_fluid = fabsf(diff.x) + fabsf(diff.y) + fabsf(diff.z) < 3.f * _range;
  }
  o_output[_out_res.idx()] = is_fluid ? 1.f : 0.f;
  o_render[_out_res.idx()] = is_fluid ? 255 : 0;
}

// __global__ void bg_subtract(float * o_output, Resolution _out_res, uchar3 const * _bg, uchar3 const * _input, Resolution _in_res, float _range) {
//   int x = (_in_res.width - _out_res.width) / 2 + _out_res.x();
//   int y = (_in_res.height - _out_res.height) / 2 + _out_res.y();
//   float new_val = 1.0f;
//   if(x >= 0 && x < _in_res.width && y >= 0 && y < _in_res.height) {
//     uchar3 const & rgb = _input[x + y * _in_res.width];
//     uchar3 const & bg_rgb = _bg[x + y * _in_res.width];
//     float3 diff = (make_float3(rgb.x, rgb.y, rgb.z) - make_float3(bg_rgb.x, bg_rgb.y, bg_rgb.z)) / 255.f;
//     new_val = fabsf(diff.x) + fabsf(diff.y) + fabsf(diff.z) < 3.f * _range ? 1.f : 0.f;
//   }
//   o_output[_out_res.idx()] = new_val > 0.5 ? 1.f : 0.f;
// }
