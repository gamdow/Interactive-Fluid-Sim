#pragma once

#include <cstring>
#include <cuda_runtime.h>

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
    std::memset(host, 0, size * sizeof(T));
    cudaMalloc((void **) & device, size * sizeof(T));
    cudaMemset(device, 0, size * sizeof(T));
  }
  ~MirroredArray() {
    delete [] host;
    cudaFree(device);
  }
  T & operator[](int i) {
    return host[i];
  }
  void copyHostToDevice() {
    cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice);
  }
  int size;
  T * host;
  T * device;
};
