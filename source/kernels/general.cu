#include "general.cuh"

// #include <iostream> // for host code
#include <stdio.h> // for kernel code

#include "../cuda/helper_math.h"
// #include "../cuda/helper_cuda.h"
// #include "../cuda/utility.cuh"

__global__ void sum_arrays(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2, Resolution _buffer_res) {
  int const idx = _buffer_res.idx();
  o_array[idx] = _c1 * _array1[idx] + _c2 * _array2[idx];
}

__global__ void copy_to_surface(cudaSurfaceObject_t o_surface, Resolution _surface_res, float4 const * _buffer, Resolution _buffer_res) {
  int buffer_diff = _surface_res.buffer - _buffer_res.buffer;
  surf2Dwrite(_buffer[_buffer_res.idx()], o_surface, (_buffer_res.x() + buffer_diff) * sizeof(float4), _buffer_res.y() + buffer_diff);
}

__global__ void copy_to_array(float * o_buffer, Resolution _out_res, uchar3 const * _buffer, Resolution _in_res) {
  int x = (_out_res.width - _in_res.width) / 2 + _out_res.x();
  int y = (_out_res.height - _in_res.height) / 2 + _out_res.y();
  if(x >= 0 && x < _in_res.width && y >= 0 && y < _in_res.height) {
    uchar3 const & in = _buffer[x + y * _in_res.width];
    o_buffer[_out_res.idx()] = (float)(in.x + in.y + in.z) / (255.0f * 3.0f) > 0.5f ? 0.0f : 1.0f;
  }
}
