#include "managed_array.h"

#include <iostream>
#include <cstring>
#include <typeinfo>

#include "../debug.h"
#include "../cuda/helper_cuda.h"
#include "../cuda/utility.h"

void Allocator::report() const {
  format_out << "Total Host: " << __host_allocation_bytes_total << " bytes, Total Device: " << __device_allocation_bytes_total << " bytes" << std::endl;
}

template<typename T> bool Allocator::hostAllocate(T * & o_ptr, size_t _size) const {
  o_ptr = new T [_size]();
  return reportAllocate(__host_allocation_bytes_total, "new", typeid(T).name(), o_ptr, _size * sizeof(T));
}

template<typename T> bool Allocator::deviceAllocate(T * & o_ptr, size_t _size) const {
  size_t num_bytes = _size * sizeof(T);
  checkCudaErrors(cudaMalloc(&o_ptr, num_bytes));
  return reportAllocate(__device_allocation_bytes_total, "cudaMalloc", typeid(T).name(), o_ptr, num_bytes);
}

template<typename T> bool Allocator::devicePitchAllocate(T * & o_ptr, size_t & o_pitch, size_t _width, size_t _height) const {
  checkCudaErrors(cudaMallocPitch(&o_ptr, &o_pitch, sizeof(T) * _width, _height));
  return reportAllocate(__device_allocation_bytes_total, "cudaMallocPitch", typeid(T).name(), o_ptr, (_width + o_pitch) * _height);
}

bool Allocator::reportAllocate(size_t & io_total, char const * _function_name, char const * _type_name, void const * _ptr, size_t _bytes) const {
  ++__allocation_count;
  if(_ptr != nullptr) {
    format_out << _function_name << "<" << _type_name << ">: " << _bytes << " bytes (" << _ptr << ")" << std::endl;
    io_total += _bytes;
    return true;
  } else {
    format_out << _function_name << "<" << _type_name << ">: " << _bytes << " bytes FAILED!" << std::endl;
    return false;
  }
}

template<class T> void ManagedArray<T>::reset() {
  clear(__data, getSizeBytes());
}

template<class T> void Managed1DArray<T>::resize(Allocator const & _alloc, size_t _size) {
  deallocate(this->__data);
  allocate(this->__data, _alloc, _size);
  this->__size = _size;
}

template<class T> HostArray<T> & HostArray<T>::operator=(DeviceArray<T> const & _in) {
  checkCudaErrors(cudaMemcpy(*this, _in, this->getSizeBytes(), cudaMemcpyDeviceToHost));
  return *this;
}

template<class T> bool HostArray<T>::allocate(T * & o_ptr, Allocator const & _alloc, size_t _size) {
  return _alloc.hostAllocate(o_ptr, _size);
}

template<class T> void HostArray<T>::deallocate(T * _ptr) {delete [] _ptr;}

template<class T> void HostArray<T>::clear(T * _ptr, size_t _bytes) {std::memset(_ptr, 0, _bytes);}

template<class T> DeviceArray<T> & DeviceArray<T>::operator=(DeviceArray<T> const & _in) {
  checkCudaErrors(cudaMemcpy(*this, _in, this->getSizeBytes(), cudaMemcpyDeviceToDevice));
  return *this;
}

template<class T> DeviceArray<T> & DeviceArray<T>::operator=(HostArray<T> const & _in) {
  checkCudaErrors(cudaMemcpy(*this, _in, this->getSizeBytes(), cudaMemcpyHostToDevice));
  return *this;
}

template<class T> bool DeviceArray<T>::allocate(T * & o_ptr, Allocator const & _alloc, size_t _size) {
  return _alloc.deviceAllocate(o_ptr, _size);
}

template<class T> void DeviceArray<T>::deallocate(T * _ptr) {checkCudaErrors(cudaFree(_ptr));}

template<class T> void DeviceArray<T>::clear(T * _ptr, size_t _bytes) {checkCudaErrors(cudaMemset(_ptr, 0, _bytes));}

template<class T> MirroredArray<T>::MirroredArray(Allocator const & _alloc, size_t _size) : __host(_alloc, _size), __device(_alloc, _size) {}

template<class T> void MirroredArray<T>::copyHostToDevice() {assert(__device.getSizeBytes() >= __host.getSizeBytes()); checkCudaErrors(cudaMemcpy(__device, __host, __host.getSizeBytes(), cudaMemcpyHostToDevice));}

template<class T> void MirroredArray<T>::copyDeviceToHost() {assert(__host.getSizeBytes() >= __device.getSizeBytes()); checkCudaErrors(cudaMemcpy(__host, __device, __device.getSizeBytes(), cudaMemcpyDeviceToHost));}

template<class T> void MirroredArray<T>::resize(Allocator const & _alloc, size_t _size) {__host.resize(_alloc, _size); __device.resize(_alloc, _size);}

template<class T> void MirroredArray<T>::reset() {__host.reset(); __device.reset();}

template<class T> void DevicePitchedArray<T>::resize(Allocator const & _alloc, size_t _width, size_t _height) {
  deallocate(this->__data);
  allocate(this->__data, __pitch, _alloc, _width, _height);
  this->__size = (_width + __pitch) * _height;
}

template<class T> bool DevicePitchedArray<T>::allocate(T * & o_ptr, size_t & o_pitch, Allocator const & _alloc, size_t _width, size_t _height) {
  return _alloc.devicePitchAllocate(o_ptr, o_pitch, _width, _height);
}

template<class T> void DevicePitchedArray<T>::deallocate(T * _ptr) {checkCudaErrors(cudaFree(_ptr));}

template<class T> void DevicePitchedArray<T>::clear(T * _ptr, size_t _bytes) {checkCudaErrors(cudaMemset(_ptr, 0, _bytes));}

#define ARRAYS(T) \
  template class ManagedArray<T>; \
  template class Managed1DArray<T>; \
  template class HostArray<T>; \
  template class DeviceArray<T>; \
  template class MirroredArray<T>; \
  template class DevicePitchedArray<T>;
EXPLICT_INSTATIATION(ARRAYS)
#undef ARRAYS
