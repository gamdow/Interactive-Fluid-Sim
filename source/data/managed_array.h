#pragma once

#include <cuda_runtime.h>

#include "../debug.h"

struct Allocator {
  Allocator()
    : __host_allocation_bytes_total(0u)
    , __device_allocation_bytes_total(0u)
    , __allocation_count(0u)
  {
    format_out.__indent += 1;
  }
  ~Allocator() {
    format_out.__indent -= 1;
    if(__allocation_count > 1) report();
  }
  void report() const;
  template<typename T> bool hostAllocate(T * & o_ptr, size_t _size) const;
  template<typename T> bool deviceAllocate(T * & o_ptr, size_t _size) const;
  template<typename T> bool devicePitchAllocate(T * & o_ptr, size_t & o_pitch, size_t _width, size_t _height) const;
private:
  bool reportAllocate(size_t & io_total, char const * _function_name, char const * _type_name, void const * _ptr, size_t _bytes) const;
  mutable size_t __host_allocation_bytes_total;
  mutable size_t __device_allocation_bytes_total;
  mutable size_t __allocation_count;
};

template<class T>
struct ManagedArray {
  inline size_t getSizeBytes() const {return __size * sizeof(T);}
  inline size_t getSize() const {return __size;}
  operator T * &() {return __data;}
  operator T const * () const {return __data;}
  T * getData() const {return __data;}
  void reset();
  friend void swap(ManagedArray & _l, ManagedArray & _r) {
      using std::swap;
      swap(_l.__data, _r.__data);
      swap(_l.__size, _r.__size);
  }
protected:
  ManagedArray() : __size(0u), __data(nullptr) {}
  virtual ~ManagedArray() {}
  T * __data;
  size_t __size;
private:
  virtual void clear(T * _ptr, size_t _bytes) = 0;
  ManagedArray(ManagedArray const &) = delete;
  ManagedArray & operator=(ManagedArray const &) = delete;
};

template<class T>
struct Managed1DArray : public ManagedArray<T> {
  void resize(Allocator const & _alloc, size_t _size);
private:
  virtual void deallocate(T * _ptr) = 0;
  virtual bool allocate(T * & o_ptr, Allocator const & _alloc, size_t _size) = 0;
};

template<class T> struct DeviceArray;

template<class T>
struct HostArray : public Managed1DArray<T> {
  HostArray() {}
  HostArray(Allocator const & _alloc, size_t _size) {this->resize(_alloc, _size);}
  HostArray<T> & operator=(DeviceArray<T> const & _in);
private:
  virtual void deallocate(T * _ptr);
  virtual bool allocate(T * & o_ptr, Allocator const & _alloc, size_t _size);
  virtual void clear(T * _ptr, size_t _bytes);
};

template<class T>
struct DeviceArray : public Managed1DArray<T> {
  DeviceArray() {}
  DeviceArray(Allocator const & _alloc, size_t _size) {this->resize(_alloc, _size);}
  DeviceArray<T> & operator=(DeviceArray<T> const & _in);
  DeviceArray<T> & operator=(HostArray<T> const & _in);
private:
  virtual void deallocate(T * _ptr);
  virtual bool allocate(T * & o_ptr, Allocator const & _alloc, size_t _size);
  virtual void clear(T * _ptr, size_t _bytes);
};

// Class of managing data mirrored in host and device memory
template<class T>
struct MirroredArray {
  MirroredArray() {}
  MirroredArray(Allocator const & _alloc, size_t _size);
  T & operator[](int i) {return static_cast<T*>(__host)[i];}
  HostArray<T> & host() {return __host;}
  DeviceArray<T> & device() {return __device;}
  size_t getSizeBytes() const {return __host.getSizeBytes();}
  void resize(Allocator const & _alloc, size_t _size);
  void reset();
  void copyHostToDevice();
  void copyDeviceToHost();
private:
  HostArray<T> __host;
  DeviceArray<T> __device;
};

template<class T>
struct DevicePitchedArray : public ManagedArray<T> {
  DevicePitchedArray() : __pitch(0u)  {}
  DevicePitchedArray(Allocator const & _alloc, size_t _width, size_t _height) : __pitch(0u) {this->resize(_alloc, _width, _height);}
  void resize(Allocator const & _alloc, size_t _width, size_t _height);
  size_t getPitch() const {return __pitch;}
private:
  virtual void deallocate(T * _ptr);
  virtual bool allocate(T * & o_ptr, size_t & o_pitch, Allocator const & _alloc, size_t _width, size_t _height);
  virtual void clear(T * _ptr, size_t _bytes);
  size_t __pitch;
};
