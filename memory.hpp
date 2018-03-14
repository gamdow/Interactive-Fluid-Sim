#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <iostream>

// Class of managing data mirrored in host and device memory
template<class T>
struct MirroredArray {
  MirroredArray()
    : size(0)
    , host(nullptr)
    , device(nullptr)
  {}
  MirroredArray(int _size)
    : size(_size)
    , host(nullptr)
    , device(nullptr)
  {
    host = new T [size];
    checkCudaErrors(cudaMalloc((void **) & device, size * sizeof(T)));
    reset();
  }
  ~MirroredArray() {
    delete [] host;
    checkCudaErrors(cudaFree(device));
  }
  inline T & operator[](int i) {return host[i];}
  inline void copyHostToDevice() {checkCudaErrors(cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice));}
  inline void copyDeviceToHost() {checkCudaErrors(cudaMemcpy(host, device, size * sizeof(T), cudaMemcpyDeviceToHost));}
  void reset() {
    std::memset(host, 0, size * sizeof(T));
    checkCudaErrors(cudaMemset(device, 0, size * sizeof(T)));
  }
  int size;
  T * host;
  T * device;
};
