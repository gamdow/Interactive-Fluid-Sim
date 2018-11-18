#include "camera_filter.h"


__global__ void filter(float * o_buffer, Resolution _out_res, uchar3 const * _buffer, Resolution _in_res, float _threshold);

CameraFilter::CameraFilter(OptimalBlockConfig const & _block_config, int _buffer_width)
  : KernelWrapper(_block_config, _buffer_width)
{
  format_out << "Constructing Camera Filter Device Buffers:" << std::endl;
  OutputIndent indent;
  {
    Allocator alloc;
    __output.resize(alloc, buffer_resolution().size);
  }
}

void CameraFilter::update(ArrayStructConst<uchar3> _camera_data, float _threshold) {
  if(__input.getSize() != _camera_data.resolution.size) {
    __input.resize(Allocator(), _camera_data.resolution.size);
  }
  checkCudaErrors(cudaMemcpy(__input, _camera_data.data, __input.getSizeBytes(), cudaMemcpyHostToDevice));

  filter<<<grid(),block()>>>(__output, buffer_resolution(), __input, _camera_data.resolution, _threshold);
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

  return hsv;
}

__global__ void filter(float * o_buffer, Resolution _out_res, uchar3 const * _buffer, Resolution _in_res, float _threshold) {
  int x = (_in_res.width - _out_res.width) / 2 + _out_res.x();
  int y = (_in_res.height - _out_res.height) / 2 + _out_res.y();
  if(x >= 0 && x < _in_res.width && y >= 0 && y < _in_res.height) {
    uchar3 const & rgb = _buffer[x + y * _in_res.width];
    float3 hsv = rgbToHsv(make_float3(rgb.x, rgb.y, rgb.z) / 255.0f);
    o_buffer[_out_res.idx()] = hsv.x < _threshold ? 1.0f : 0.0f;
  } else {
    o_buffer[_out_res.idx()] = 1.0f;
  }
}
