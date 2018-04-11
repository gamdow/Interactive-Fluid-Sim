#include "texture_object.hpp"

#include "helper_cuda.h"
#include "buffer_spec.cuh"

template<class T>
TextureObject<T>::TextureObject()
  : __buffer(nullptr)
  , __pitch(0u)
  , __object(0u)
{}

// Initialise the Texture Object required by advect's interpolated sampling.
template<class T>
void TextureObject<T>::init(BufferSpec const & _spec) {
  checkCudaErrors(cudaMallocPitch(&__buffer, &__pitch, sizeof(T) * _spec.width, _spec.height));
  cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = __buffer;
  resDesc.res.pitch2D.pitchInBytes = __pitch;
  resDesc.res.pitch2D.width = _spec.width;
  resDesc.res.pitch2D.height = _spec.height;
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
