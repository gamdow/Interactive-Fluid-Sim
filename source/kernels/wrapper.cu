#include "wrapper.h"


KernelWrapper::KernelWrapper(OptimalBlockConfig const & _block_config, int _buffer_width) {
  __grid_dim = _block_config.grid;
  __block_dim = _block_config.block;
  __buffer_res = Resolution(_block_config.optimal_res, _buffer_width);
}
