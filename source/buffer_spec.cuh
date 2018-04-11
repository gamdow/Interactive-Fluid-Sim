#pragma once

#include <cuda_runtime.h>
#include "helper_math.h"

struct BufferSpec {
  BufferSpec();
  BufferSpec(int2 _dims, int _buffer);
#ifdef __CUDACC__
  __forceinline__ __device__ int x() const {return blockIdx.x * blockDim.x + threadIdx.x + buffer;}
  __forceinline__ __device__ int y() const {return blockIdx.y * blockDim.y + threadIdx.y + buffer;}
  __forceinline__ __device__ int idx() const {return width * y() + x();}
  __forceinline__ __device__ int4 stencil() const {return idx() + make_int4(1, -1, width, -width);}
#endif
  int width, height, buffer, size;
};
