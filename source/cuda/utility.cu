#include "utility.h"

#include "../debug.h"

__global__ void pressure_solve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, Resolution _buffer_res, float2 _dx);

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

__global__ void copy_to_surface(cudaSurfaceObject_t o_surface, Resolution _surface_res, uchar3 const * _buffer, Resolution _buffer_res) {
  // surf2Dwrite<uchar3>(_buffer[_buffer_res.idx()], o_surface, (int)(_buffer_res.x() * sizeof(unsigned char)), _buffer_res.y(), cudaBoundaryModeTrap);
  #ifdef __CUDA_ARCH__
    __nv_tex_surf_handler("__surf2Dwrite_v2", (typename __nv_surf_trait<uchar3>::cast_type)&_buffer[_buffer_res.idx()], (int)sizeof(uchar3), o_surface, (int)(_buffer_res.x() * sizeof(uchar3)), _buffer_res.y(),  cudaBoundaryModeTrap);
  #endif /* __CUDA_ARCH__ */
}

void copyToSurface(OptimalBlockConfig const & _block_config, cudaSurfaceObject_t o_surface, Resolution const & _surface_res, uchar3 const * _buffer, Resolution const & _buffer_res) {
  copy_to_surface<<<_block_config.grid, _block_config.block>>>(o_surface, _surface_res, _buffer, _buffer_res);
}

__global__ void copy_to_surface(cudaSurfaceObject_t o_surface, Resolution _surface_res, float const * _buffer, Resolution _buffer_res) {
  surf2Dwrite<float>(_buffer[_buffer_res.idx()], o_surface, (_buffer_res.x()) * sizeof(float), _buffer_res.y());
}

void copyToSurface(OptimalBlockConfig const & _block_config, cudaSurfaceObject_t o_surface, Resolution const & _surface_res, float const * _buffer, Resolution const & _buffer_res) {
  copy_to_surface<<<_block_config.grid, _block_config.block>>>(o_surface, _surface_res, _buffer, _buffer_res);
}

__global__ void copy_to_surface(cudaSurfaceObject_t o_surface, Resolution _surface_res, unsigned char const * _buffer, Resolution _buffer_res) {
  surf2Dwrite<unsigned char>(_buffer[_buffer_res.idx()], o_surface, (_buffer_res.x()) * sizeof(unsigned char), _buffer_res.y());
}

void copyToSurface(OptimalBlockConfig const & _block_config, cudaSurfaceObject_t o_surface, Resolution const & _surface_res, unsigned char const * _buffer, Resolution const & _buffer_res) {
  copy_to_surface<<<_block_config.grid, _block_config.block>>>(o_surface, _surface_res, _buffer, _buffer_res);
}

__global__ void copy_to_surface2(cudaSurfaceObject_t o_surface, Resolution _surface_res, float4 const * _buffer, Resolution _buffer_res) {
  surf2Dwrite<float4>(_buffer[_buffer_res.idx()], o_surface, (_buffer_res.x()) * sizeof(float4), _buffer_res.y());
}

void copyToSurface(OptimalBlockConfig const & _block_config, cudaSurfaceObject_t o_surface, Resolution const & _surface_res, float4 const * _buffer, Resolution const & _buffer_res) {
  copy_to_surface2<<<_block_config.grid, _block_config.block>>>(o_surface, _surface_res, _buffer, _buffer_res);
}

void print(std::ostream & _out, float4 _v) {
  _out << "(" << _v.x << "," << _v.y << "," << _v.z << "," << _v.w << ")";
}
