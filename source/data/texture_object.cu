#include "texture_object.cuh"

#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>

#include "../cuda/helper_cuda.h"

template<class T>
TextureObject<T>::TextureObject()
  : __object(0u)
{}

template<class T>
void TextureObject<T>::init(Allocator & _alloc, Resolution const & _res) {
  __array.resize(_alloc, _res.width, _res.height);
  cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = __array.getData();
  resDesc.res.pitch2D.pitchInBytes = __array.getPitch();
  resDesc.res.pitch2D.width = _res.width;
  resDesc.res.pitch2D.height = _res.height;
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
  cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  checkCudaErrors(cudaCreateTextureObject(&__object, &resDesc, &texDesc, nullptr));
}

template<class T>
TextureObject<T>::~TextureObject() {
  checkCudaErrors(cudaDestroyTextureObject(__object));
}

template<class T>
void TextureObject<T>::copyFrom(T const * _array, Resolution const & _res) {
  cudaMemcpy2D(__array.getData(), __array.getPitch(), _array, sizeof(T) * _res.width, sizeof(T) * _res.width, _res.height, cudaMemcpyDeviceToDevice);
}

#define EXPLICT_INSTATIATION(TYPED_MACRO) \
  TYPED_MACRO(float) \
  TYPED_MACRO(float2) \
  TYPED_MACRO(float3) \
  TYPED_MACRO(float4)

#define TEMPLATE(T) template class TextureObject<T>;
  EXPLICT_INSTATIATION(TEMPLATE)
#undef TEMPLATE
