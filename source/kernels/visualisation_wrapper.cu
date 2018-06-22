#include "visualisation_wrapper.cuh"

#include <iostream>

#include "visualisation.cuh"
#include "../debug.hpp"

void lerp(float & _from, float _to) {
  _from = _to * 0.05f + _from * 0.95f;
}

VisualisationWrapper::VisualisationWrapper(OptimalBlockConfig const & _block_config, int _buffer_width)
  : KernelWrapper(_block_config, _buffer_width)
{
  format_out << "Constructing Kernel Device Buffers:" << std::endl;
  OutputIndent indent;
  {
    Allocator alloc;
    __min.resize(alloc, 1u);
    __max.resize(alloc, 1u);
    __f4reduce.resize(alloc, grid().x * grid().y);
    __rgba.resize(alloc, buffer_resolution().size);
  }
}

void VisualisationWrapper::copyToRGBA(float const * _buffer) {
  scalar_to_rgba<<<grid(),block()>>>(__rgba, _buffer, buffer_resolution(), 1.0f);
}

void VisualisationWrapper::copyToRGBA(float2 const * _buffer) {
  vector_field_to_rgba<<<grid(),block()>>>(__rgba, _buffer, buffer_resolution(), 1.0f);
}

void VisualisationWrapper::copyToRGBA(float4 const * _buffer, float3 const * _map) {
  map_to_rgba<<<grid(),block()>>>(__rgba, _buffer, buffer_resolution(), _map);
}

void VisualisationWrapper::minMaxReduce(float4 & o_min, float4 & o_max, float4 const * _array) {
  min<<<grid(), block()>>>(__f4reduce, _array, buffer_resolution());
  min<<<1, grid()>>>(__min.device(), __f4reduce, Resolution(grid().x, grid().y, 0));
  __min.copyDeviceToHost();
  o_min = __min[0];
  max<<<grid(), block()>>>(__f4reduce, _array, buffer_resolution());
  max<<<1, grid()>>>(__max.device(), __f4reduce, Resolution(grid().x, grid().y, 0));
  __max.copyDeviceToHost();
  o_max = __max[0];
}

void VisualisationWrapper::scale(float4 * o_array, float4 _min, float4 _max) {
  scaleRGB<<<grid(), block()>>>(o_array, _min, _max, buffer_resolution());
}

void VisualisationWrapper::adjustBrightness(float & io_min, float & io_max) {
  float4 min4, max4;
  minMaxReduce(min4, max4, __rgba);
  lerp(io_min, fmaxf(fminf(fminf(min4.x, min4.y), min4.z), 0.0f));
  lerp(io_max, fminf(fmaxf(fmaxf(max4.x, max4.y), max4.z), 1.0f));
  min4 = make_float4(make_float3(io_min), 0.0f);
  max4 = make_float4(make_float3(io_max), 1.0f);
  scale(__rgba, min4, max4);
}
