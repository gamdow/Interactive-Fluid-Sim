#pragma once

#include <cuda_runtime.h>

#include "helper_math.h"

struct Resolution {
  Resolution();
  template<typename T> Resolution(T _width, T _height, int _buffer = 0);
  Resolution(Resolution const & _in);
  Resolution(Resolution const & _in, int _buffer);

  void print(char const * _name) const;

#ifdef __CUDACC__
  __device__ int i() const {return blockIdx.x * blockDim.x + threadIdx.x;}
  __device__ int j() const {return blockIdx.y * blockDim.y + threadIdx.y;}
  __device__ int x() const {return blockIdx.x * blockDim.x + threadIdx.x + buffer;}
  __device__ int y() const {return blockIdx.y * blockDim.y + threadIdx.y + buffer;}
  __device__ int idx() const {return width * y() + x();}
  __device__ int4 stencil() const {return idx() + make_int4(1, -1, width, -width);}
#endif

  int width, height, buffer, size;
};

template<typename T>
Resolution::Resolution(T _width, T _height, int _buffer)
  : width(_width + 2 * _buffer)
  , height(_height + 2 * _buffer)
  , buffer(_buffer)
{
  size = width * height;
}
