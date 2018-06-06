#include "utility.cuh"

#include "../kernels/kernels.cuh"

void reportCudaCapability() {
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

OptimalBlockConfig::OptimalBlockConfig(Resolution _res)
{
  std::cout << "Optimising Blocksize:" << std::endl;
  // Use CUDA's occupancy to determine the optimal blocksize and adjust the screen (and therefore array) resolution to be an integer multiple (then there's no need for bounds checking in the kernels).
  _res.print("\tDesired Resolution");
  int blockSize, minGridSize; cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pressure_solve, 0, _res.size);
  block = dim3(32u, blockSize / 32u);
  Resolution(block.x, block.y).print("\tOptimal Block");
  grid = dim3(_res.width / block.x, _res.height / block.y);
  optimal_res = Resolution(grid.x * block.x, grid.y * block.y);
  optimal_res.print("\tAdjusted Resolution");
}
