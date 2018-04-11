#include "capability.cuh"

#include <iostream>

#include "kernels.cuh"

void Capability::printRes(char const * _name, int _x, int _y) {
  std::cout << _name << ": " << _x << " x " << _y << std::endl;
}

Capability::Capability(int2 _dims, int _buffer)
  : original_dims(_dims)
{
  std::cout << std::endl;
  reportCapability();
  std::cout << std::endl;
  // Use CUDA's occupancy to determine the optimal blocksize and adjust the screen (and therefore array) resolution to be an integer multiple (then there's no need for bounds checking in the kernels).
  int const N = _dims.x * _dims.y;
  int blockSize, minGridSize; cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pressure_solve, 0, N);
  block = dim3(32u, blockSize / 32u);
  grid = dim3(_dims.x / block.x, _dims.y / block.y);
  adjusted_dims = make_int2(grid.x * block.x, grid.y * block.y);
  buffer_spec = BufferSpec(adjusted_dims, _buffer);
  printRes("Desired Resolution", _dims.x, _dims.y);
  printRes("Optimal Block", block.x, block.y);
  printRes("Adjusted Resolution", adjusted_dims.x, adjusted_dims.y);
  std::cout << std::endl;
}

void Capability::reportCapability() const {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  for(int dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int driverVersion; cudaDriverGetVersion(&driverVersion);
    int runtimeVersion; cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Device: " << dev << ": " << deviceProp.name << std::endl;
    std::cout << "\tCapability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "\tRuntime/Driver: " << runtimeVersion << "/" << driverVersion << std::endl;
  }
}
