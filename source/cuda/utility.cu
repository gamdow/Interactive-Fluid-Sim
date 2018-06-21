#include "utility.cuh"

#include "../debug.hpp"
#include "../kernels/simulation.cuh"

void reportCudaCapability() {
  format_out << "CUDA Capability: " << std::endl;
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  for(int dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int driverVersion; cudaDriverGetVersion(&driverVersion);
    int runtimeVersion; cudaRuntimeGetVersion(&runtimeVersion);
    OutputIndent indent1;
    format_out << "Device: " << dev << ": " << deviceProp.name << std::endl;
    OutputIndent indent2;
    format_out << "Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    format_out << "Runtime/Driver: " << runtimeVersion << "/" << driverVersion << std::endl;
  }
}

OptimalBlockConfig::OptimalBlockConfig(Resolution _res)
{
  format_out << "Optimising Blocksize:" << std::endl;
  // Use CUDA's occupancy to determine the optimal blocksize and adjust the screen (and therefore array) resolution to be an integer multiple (then there's no need for bounds checking in the kernels).
  int blockSize, minGridSize; cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pressure_solve, 0, _res.size);
  block = dim3(32u, blockSize / 32u);
  OutputIndent indent1;
  Resolution(block.x, block.y).print("Optimal Block");
  grid = dim3(_res.width / block.x, _res.height / block.y);
  optimal_res = Resolution(grid.x * block.x, grid.y * block.y);
  _res.print("Desired Resolution");
  optimal_res.print("Adjusted Resolution");
}

void print(std::ostream & _out, float4 _v) {
  _out << "(" << _v.x << "," << _v.y << "," << _v.z << "," << _v.w << ")";
}
