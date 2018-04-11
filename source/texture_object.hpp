#pragma once

#include <cuda_runtime.h>

struct BufferSpec;

template<class T>
struct TextureObject {
  TextureObject();
  void init(BufferSpec const & _spec);
  void shutdown();
  T * __buffer;
  size_t __pitch;
  cudaTextureObject_t __object;
};

template struct TextureObject<float>;
template struct TextureObject<float2>;
template struct TextureObject<float4>;
