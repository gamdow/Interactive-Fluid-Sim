#pragma once

#include <cuda_runtime.h>

#include "../cuda/utility.h"
#include "../data/resolution.h"
#include "../data/managed_array.h"

// Wrapper for the simulation kernels that ensures the necessary CUDA resources are available and all the dimensions are consistent.
struct VisualisationWrapper : public SurfaceWriter
{
  VisualisationWrapper(OptimalBlockConfig const & _block_config);
  virtual ~VisualisationWrapper() {}
  Resolution const & resolution() const {return __config.resolution;}
  DeviceArray<float4> const & rgba() const {return __rgba;}
  void visualise(float const * _buffer);
  void visualise(float2 const * _buffer);
  void visualise(float4 const * _buffer, float3 const * _map);
  void adjustBrightness(float4 & io_min, float4 & io_max, bool _instant=false);
private:
  // From SurfaceWriter
  virtual void writeToSurfaceImpl(cudaSurfaceObject_t o_surface, Resolution const & _surface_res) const;
  void minMaxReduce(float4 & o_min, float4 & o_max, float4 const * _array);
  void scale(float4 * o_array, float4 _min, float4 _max);
  OptimalBlockConfig const & __config;
  MirroredArray<float4> __min, __max;
  DeviceArray<float4> __f4reduce;
  DeviceArray<float4> __rgba;
};
