#pragma once

#include <cuda_runtime.h>

#include "resolution.h"
#include "managed_array.h"

template<class T>
struct TextureObject {
  TextureObject();
  ~TextureObject();
  void init(Allocator & _alloc, Resolution const & _spec);
  T * getData() const {return __array.getData();}
  size_t getPitch() const {return __array.getPitch();}
  cudaTextureObject_t getObject() const {return __object;}
  void copyFrom(T const * _array, Resolution const & _res);
private:
  DevicePitchedArray<T> __array;
  cudaTextureObject_t __object;
};
