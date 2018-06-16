#pragma once

#include <cuda_runtime.h>

#include "resolution.cuh"
#include "managed_array.cuh"

template<class T>
struct TextureObject {
  TextureObject();
  void init(Allocator & _alloc, Resolution const & _spec);
  void shutdown();
  T * getData() const {return __array.getData();}
  size_t getPitch() const {return __array.getPitch();}
  cudaTextureObject_t getObject() const {return __object;}
private:
  DevicePitchedArray<T> __array;
  cudaTextureObject_t __object;
};
