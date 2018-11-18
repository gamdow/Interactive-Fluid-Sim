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
  void update(ArrayStructConst<uchar3> _camera_data, float _threshold);
  DeviceArray<float> const & output() const {return __output;}
private:
  DeviceArray<uchar3> __input;
  DeviceArray<float> __output;
};
