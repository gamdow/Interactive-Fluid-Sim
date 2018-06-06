#include "texture_object.cuh"

#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>

#include "../cuda/helper_cuda.h"

template<class T>
TextureObject<T>::TextureObject()
  : __buffer(nullptr)
  , __pitch(0u)
  , __object(0u)
{}

template<class T>
void TextureObject<T>::init(Resolution const & _res) {
  std::cout << "\tcudaMallocPitch(" << typeid(T).name() << "): ";
  checkCudaErrors(cudaMallocPitch(&__buffer, &__pitch, sizeof(T) * _res.width, _res.height));
  std::cout << _res.height * __pitch << " bytes (" << __buffer << ")" << std::endl;
  cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = __buffer;
  resDesc.res.pitch2D.pitchInBytes = __pitch;
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
void TextureObject<T>::shutdown() {
  checkCudaErrors(cudaDestroyTextureObject(__object));
  checkCudaErrors(cudaFree(__buffer));
}

#define EXPLICT_INSTATIATION(TYPED_MACRO) \
  TYPED_MACRO(float) \
  TYPED_MACRO(float2) \
  TYPED_MACRO(float3) \
  TYPED_MACRO(float4)

#define TEMPLATE(T) template class TextureObject<T>;
  EXPLICT_INSTATIATION(TEMPLATE)
#undef TEMPLATE
