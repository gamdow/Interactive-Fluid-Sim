#pragma once

#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "../data_structs/resolution.cuh"
#include "../data_structs/managed_array.cuh"
#include "../data_structs/texture_object.cuh"

// explicit template instantiation so applyAdvection can only be used for element types for which there is a matching TextureObject instance
#define EXPLICT_INSTATIATION(TYPED_MACRO) \
  TYPED_MACRO(float) \
  TYPED_MACRO(float2) \
  TYPED_MACRO(float3) \
  TYPED_MACRO(float4)

void reportCudaCapability();

template<typename T>
void reportCudaMalloc(T * & _ptr, size_t _size) {
  size_t num_bytes = _size * sizeof(T);
  std::cout << "\tcudaMalloc("<< typeid(T).name() << "): " << num_bytes << " bytes";
  checkCudaErrors(cudaMalloc((void **) & _ptr, num_bytes));
  std::cout << " (" << _ptr << ")" << std::endl;
}

struct OptimalBlockConfig {
  OptimalBlockConfig(Resolution _res);
  Resolution optimal_res;
  dim3 block, grid;
};
