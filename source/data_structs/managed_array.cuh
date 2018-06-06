#pragma once

#include <cuda_runtime.h>

template<class T>
struct ManagedArray {
  size_t getSizeBytes() const {return size * sizeof(T);}
  operator T * &() {return data;}
  void swap(T * & io);
protected:
  ManagedArray() : size(0u), data(nullptr) {}
  ManagedArray(size_t _size) : size(_size), data(nullptr) {}
  virtual ~ManagedArray() {}
  size_t const size;
  T * data;
private:
  ManagedArray(ManagedArray const &) = delete;
  ManagedArray & operator=(ManagedArray const &) = delete;
};

template<class T>
struct HostArray : public ManagedArray<T> {
  HostArray(size_t _size);
  ~HostArray();
  void reset();
// private:
//   HostArray(HostArray const &) = delete;
//   HostArray & operator=(HostArray const &) = delete;
};

template<class T>
struct DeviceArray : public ManagedArray<T> {
  DeviceArray(size_t _size);
  ~DeviceArray();
  void reset();
// private:
//   DeviceArray(DeviceArray const &) = delete;
//   DeviceArray & operator=(DeviceArray const &) = delete;
};

// Class of managing data mirrored in host and device memory
template<class T>
struct MirroredArray {
  MirroredArray(size_t _size);
  T & operator[](int i) {return static_cast<T*>(__host)[i];}
  void copyHostToDevice();
  void copyDeviceToHost();
  void reset();
  size_t getSizeBytes() const {return __host.getSizeBytes();}
  T * & host() {return __host;}
  T * & device() {return __device;}
private:
  HostArray<T> __host;
  DeviceArray<T> __device;
};
