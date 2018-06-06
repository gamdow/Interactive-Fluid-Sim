#pragma once

#include <cuda_runtime.h>

#include "resolution.cuh"

template<class T>
struct TextureObject {
  TextureObject();
  void init(Resolution const & _spec);
  void shutdown();
  T * __buffer;
  size_t __pitch;
  cudaTextureObject_t __object;
};
