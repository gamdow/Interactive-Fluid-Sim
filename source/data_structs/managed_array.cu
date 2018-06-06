#include "managed_array.cuh"

#include <iostream>
#include <cstring>
#include <typeinfo>

#include "../cuda/helper_cuda.h"
#include "../cuda/utility.cuh"

template<class T>
void ManagedArray<T>::swap(T * & io) {
  T * temp = data;
  data = io;
  io = temp;
}

template<class T> HostArray<T>::HostArray(size_t _size) : ManagedArray<T>(_size) {this->data = new T [this->size]();}
template<class T> HostArray<T>::~HostArray() {delete [] this->data;}
template<class T> void HostArray<T>::reset() {std::memset(this->data, 0, this->getSizeBytes());}

template<class T> DeviceArray<T>::DeviceArray(size_t _size) : ManagedArray<T>(_size) {reportCudaMalloc(this->data, this->size); reset();}
template<class T> DeviceArray<T>::~DeviceArray() {checkCudaErrors(cudaFree(this->data));}
template<class T> void DeviceArray<T>::reset() {checkCudaErrors(cudaMemset(this->data, 0, this->getSizeBytes()));}

template<class T> MirroredArray<T>::MirroredArray(size_t _size) : __host(_size), __device(_size) {}
template<class T> void MirroredArray<T>::copyHostToDevice() {checkCudaErrors(cudaMemcpy(__device, __host, __host.getSizeBytes(), cudaMemcpyHostToDevice));}
template<class T> void MirroredArray<T>::copyDeviceToHost() {checkCudaErrors(cudaMemcpy(__host, __device, __device.getSizeBytes(), cudaMemcpyDeviceToHost));}
template<class T> void MirroredArray<T>::reset() {__host.reset(); __device.reset();}

#define ARRAYS(T) \
  template class ManagedArray<T>; \
  template class HostArray<T>; \
  template class DeviceArray<T>; \
  template class MirroredArray<T>;
EXPLICT_INSTATIATION(ARRAYS)
#undef ARRAYS
