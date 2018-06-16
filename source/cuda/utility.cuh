#pragma once

#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "../data/resolution.cuh"
#include "../data/managed_array.cuh"
#include "../data/texture_object.cuh"

// explicit template instantiation so applyAdvection can only be used for element types for which there is a matching TextureObject instance
#define EXPLICT_INSTATIATION(TYPED_MACRO) \
  TYPED_MACRO(float) \
  TYPED_MACRO(float2) \
  TYPED_MACRO(float3) \
  TYPED_MACRO(float4) \
  TYPED_MACRO(uchar3)

void reportCudaCapability();

struct OptimalBlockConfig {
  OptimalBlockConfig(Resolution _res);
  Resolution optimal_res;
  dim3 grid, block;
};

void print(std::ostream & _out, float4 _v);
