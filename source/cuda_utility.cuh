#pragma once

#include <iostream>
#include <cstring>
#include <typeinfo>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "resolution.cuh"

// explicit template instantiation so applyAdvection can only be used for element types for which there is a matching TextureObject instance
#define EXPLICT_INSTATIATION(TYPED_MACRO) \
  TYPED_MACRO(float) \
  TYPED_MACRO(float2) \
  TYPED_MACRO(float3) \
  TYPED_MACRO(float4)

void reportCudaCapability();

template<typename T> void reportCudaMalloc(T * & _ptr, size_t _size);

struct OptimalBlockConfig {
  OptimalBlockConfig(Resolution _res);
  Resolution optimal_res;
  dim3 block, grid;
};

// Class of managing data mirrored in host and device memory
template<class T>
struct MirroredArray {
  MirroredArray();
  MirroredArray(int _size);
  ~MirroredArray();
  T & operator[](int i) {return host[i];}
  void copyHostToDevice();
  void copyDeviceToHost();
  void reset();
  size_t getTotalBytes() const {return size * sizeof(T);}
  int size;
  T * host;
  T * device;
};

template<class T>
struct TextureObject {
  TextureObject();
  void init(Resolution const & _spec);
  void shutdown();
  T * __buffer;
  size_t __pitch;
  cudaTextureObject_t __object;
};

template<typename T>
void reportCudaMalloc(T * & _ptr, size_t _size) {
  size_t num_bytes = _size * sizeof(T);
  std::cout << "\tcudaMalloc("<< typeid(T).name() << "): " << num_bytes << " bytes";
  checkCudaErrors(cudaMalloc((void **) & _ptr, num_bytes));
  std::cout << " (" << _ptr << ")" << std::endl;
}

template<class T>
MirroredArray<T>::MirroredArray()
  : size(0)
  , host(nullptr)
  , device(nullptr)
{}

template<class T>
MirroredArray<T>::MirroredArray(int _size)
  : size(_size)
  , host(nullptr)
  , device(nullptr)
{
  host = new T [size];
  reportCudaMalloc(device, size);
  reset();
}

template<class T>
MirroredArray<T>::~MirroredArray() {
  delete [] host;
  checkCudaErrors(cudaFree(device));
}

template<class T>
void MirroredArray<T>::copyHostToDevice() {checkCudaErrors(cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice));}

template<class T>
void MirroredArray<T>::copyDeviceToHost() {checkCudaErrors(cudaMemcpy(host, device, size * sizeof(T), cudaMemcpyDeviceToHost));}

template<class T>
void MirroredArray<T>::reset() {
  std::memset(host, 0, size * sizeof(T));
  checkCudaErrors(cudaMemset(device, 0, size * sizeof(T)));
}

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
