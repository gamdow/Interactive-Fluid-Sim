#pragma once

#include <iostream>
#include <cuda_runtime.h>

#include "../cuda/helper_math.h"

struct BufferedDimension {
  BufferedDimension(int _inner, int _buffer=0)
    : inner(_inner)
    , buffer(_buffer)
  {}
  friend std::ostream & operator<<(std::ostream & stream, BufferedDimension const & dim);
  __host__ __device__ operator int() const {return total();}
  __host__ __device__ int begin() const {return 0;}
  __host__ __device__ int end() const {return total();}
  __host__ __device__ int begin_inner() const {return buffer;}
  __host__ __device__ int end_inner() const {return total() - buffer;}
  int inner, buffer;
private:
  __host__ __device__ int total() const {return inner + 2 * buffer;}
};

struct Resolution {
  Resolution();
  template<typename T> Resolution(T _width, T _height, int _width_buffer=0, int _height_buffer=0);
  Resolution(Resolution const & _in, int _width_buffer, int _height_buffer);
  void print(char const * _name) const;
#ifdef __CUDACC__
  __device__ int i() const {return blockIdx.x * blockDim.x + threadIdx.x;}
  __device__ int j() const {return blockIdx.y * blockDim.y + threadIdx.y;}
  __device__ int buffer_idx() const {return width * j() + i();}
  __device__ int x() const {return blockIdx.x * blockDim.x + threadIdx.x + width.buffer;}
  __device__ int y() const {return blockIdx.y * blockDim.y + threadIdx.y + height.buffer;}
  __device__ int idx() const {return width * y() + x();}
  __device__ int4 stencil() const {return idx() + make_int4(1, -1, width, -width);}
#endif
  int inner_size() const {return width.inner * height.inner;}
  int total_size() const {return width * height;}
  BufferedDimension width, height;
};

template<typename T>
Resolution::Resolution(T _width, T _height, int _width_buffer, int _height_buffer)
  : width(_width, _width_buffer)
  , height(_height, _height_buffer)
{
}

template<typename T>
struct ArrayStructConst {
  ArrayStructConst(T const * _array, Resolution const & _res)
    : data(_array)
    , resolution(_res)
  {}
  T const * const data;
  Resolution const & resolution;
};
