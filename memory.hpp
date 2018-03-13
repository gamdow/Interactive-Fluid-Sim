#pragma once

// Class of managing data mirrored in host and device memory
template<class T>
struct MirroredArray {
  MirroredArray(int _size)
      : size(_size)
  {
    host = new T [size];
    memset(host, 0, size * sizeof(T));
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
