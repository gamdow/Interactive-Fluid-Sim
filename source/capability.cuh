#pragma once

#include <cuda_runtime.h>

#include "buffer_spec.cuh"

struct Capability {
  Capability(int2 _dims, int _buffer);
  BufferSpec buffer_spec;
  int2 original_dims;
  int2 adjusted_dims;
  dim3 block, grid;
  static void printRes(char const * _name, int _x, int _y);
private:
  void reportCapability() const;
};
