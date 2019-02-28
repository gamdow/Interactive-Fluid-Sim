#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "../cuda/utility.h"
#include "../data/resolution.h"
#include "../data/managed_array.h"

struct OptimalBlockConfig;

struct CameraFilter {
  CameraFilter(OptimalBlockConfig const & _block_config);
  virtual ~CameraFilter() {}
  Resolution const & resolution() const {return __config.resolution;}
  void update(ArrayStructConst<uchar3> const & _camera_data, ArrayStructConst<float2> const & _flow_data, bool _mirror, int _mode, bool _bg_subtract, float _value, float _range);
  DeviceArray<float4> const & render() const {return __render;}
  DeviceArray<float> const & fluidOutput() const {return __fluid_output;}
  DeviceArray<float2> const & velocityOutput() const {return __velocity_output;}
private:
  OptimalBlockConfig const & __config;
  DeviceArray<uchar3> __image_input;
  DeviceArray<float2> __flow_input;
  DeviceArray<float> __bg;
  DeviceArray<float4> __render;
  DeviceArray<float> __fluid_output;
  DeviceArray<float2> __velocity_output;
  int __last_mode;
};
