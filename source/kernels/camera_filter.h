#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "wrapper.h"
#include "../data/managed_array.h"
#include "../data/resolution.h"

struct CameraFilter : public KernelWrapper
{
  CameraFilter(OptimalBlockConfig const & _block_config, int _buffer_width);
  virtual ~CameraFilter() {}
  void update(ArrayStructConst<uchar3> _camera_data, bool _mirror, int _mode, bool _bg_subtract, float _value, float _range);
  DeviceArray<float4> const & render() const {return __render;}
  DeviceArray<float> const & output() const {return __output;}
private:
  DeviceArray<uchar3> __input;
  DeviceArray<float> __bg;
  DeviceArray<float4> __render;
  DeviceArray<float> __output;
  int __last_mode;
};
