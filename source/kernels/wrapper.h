#pragma once

#include <cuda_runtime.h>

#include "../data/resolution.h"
#include "../cuda/utility.h"

struct KernelWrapper {
  KernelWrapper(OptimalBlockConfig const & _block_config, int _buffer_width);
  virtual ~KernelWrapper() {}
  Resolution const & buffer_resolution() const {return __buffer_res;}
protected:
  dim3 const & grid() const {return __grid_dim;}
  dim3 const & block() const {return __block_dim;}
private:
  dim3 __grid_dim, __block_dim;
  Resolution __buffer_res;
};
