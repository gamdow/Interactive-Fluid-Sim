#pragma once

#include <cuda_runtime.h>

#include "wrapper.cuh"

#include "../data/resolution.cuh"
#include "../cuda/utility.cuh"

// Wrapper for the simulation kernels that ensures the necessary CUDA resources are available and all the dimensions are consistent.
struct VisualisationWrapper : public KernelWrapper
{
  VisualisationWrapper(OptimalBlockConfig const & _block_config, int _buffer_width);
  virtual ~VisualisationWrapper() {}
  DeviceArray<float4> const & rgba() const {return __rgba;}
  void copyToRGBA(float const * _buffer);
  void copyToRGBA(float2 const * _buffer);
  void copyToRGBA(float4 const * _buffer, float3 const * _map);
  void adjustBrightness(float & io_min, float & io_max);
private:
  void minMaxReduce(float4 & o_min, float4 & o_max, float4 const * _array);
  void scale(float4 * o_array, float4 _min, float4 _max);
  MirroredArray<float4> __min, __max;
  DeviceArray<float4> __f4reduce;
  DeviceArray<float4> __rgba;
};
