#pragma once

#include <cuda_runtime.h>

#include "wrapper.h"

#include "../data/resolution.h"
#include "../cuda/utility.h"

// Wrapper for the simulation kernels that ensures the necessary CUDA resources are available and all the dimensions are consistent.
struct VisualisationWrapper : public KernelWrapper, public SurfaceWriter
{
  VisualisationWrapper(OptimalBlockConfig const & _block_config, int _buffer_width);
  virtual ~VisualisationWrapper() {}
  DeviceArray<float4> const & rgba() const {return __rgba;}
  void visualise(float const * _buffer);
  void visualise(float2 const * _buffer);
  void visualise(float4 const * _buffer, float3 const * _map);
  void adjustBrightness(float4 & io_min, float4 & io_max);
private:
  // From SurfaceWriter
  virtual void writeToSurfaceImpl(cudaSurfaceObject_t o_surface, Resolution const & _surface_res) const;
  void minMaxReduce(float4 & o_min, float4 & o_max, float4 const * _array);
  void scale(float4 * o_array, float4 _min, float4 _max);
  MirroredArray<float4> __min, __max;
  DeviceArray<float4> __f4reduce;
  DeviceArray<float4> __rgba;
};
