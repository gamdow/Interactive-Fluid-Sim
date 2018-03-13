#include "simulation.cuh"

#include "helper_math.h"
#include "kernels.cuh"
#include "configuration.cuh"

int const PRESSURE_SOLVER_STEPS = 100;

struct BlockHelper {
  BlockHelper(int2 _dims)
    : block(BLOCK_SIZE)
    , grid(_dims.x / block.x, _dims.y / block.y)
  {
  }
  dim3 block, grid;
};

void simulation_step(int2 _dims, float2 _d, float _dt, float * i_fluidCells, float2 * io_velocity, float * o_divergence, float * o_pressure, float * o_buffer) {
  float2 rd = make_float2(1.0f / _d.x, 1.0f / _d.y);
  float2 r2d = make_float2(1.0f / (2.0f * _d.x), 1.0f / (2.0f * _d.y));

  // copy the input velocity to the velocity texture (for auto interpolation) via the mapped velArray
  copy_to_vel_texture(io_velocity, _dims);

  BlockHelper bh(_dims);
  advect_velocity<<<bh.grid, bh.block>>>(_dims, _dt, rd, io_velocity);
  calc_divergence<<<bh.grid, bh.block>>>(io_velocity, i_fluidCells, _dims, r2d, o_divergence);
  pressure_decay<<<bh.grid, bh.block>>>(i_fluidCells, _dims, o_pressure);
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
   pressure_solve<<<bh.grid, bh.block>>>(o_divergence, i_fluidCells, _dims, _d, o_pressure, o_buffer);
   float * temp = o_pressure;
   o_pressure = o_buffer;
   o_buffer = temp;
  }
  sub_gradient<<<bh.grid, bh.block>>>(o_pressure, i_fluidCells, _dims, r2d, io_velocity);
  enforce_slip<<<bh.grid, bh.block>>>(i_fluidCells, _dims, io_velocity);
}

void copy_to_surface(float2 * _array, float _mul, int2 _dims, cudaGraphicsResource_t & _viewCudaResource) {
  cudaGraphicsMapResources(1, &_viewCudaResource); {
    cudaArray_t writeArray;
    cudaGraphicsSubResourceGetMappedArray(&writeArray, _viewCudaResource, 0, 0);
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    cudaCreateSurfaceObject(&writeSurface, &wdsc);
    BlockHelper bh(_dims);
    hsv_to_rgba<<<bh.grid, bh.block>>>(_array, _mul, _dims, writeSurface);
    cudaDestroySurfaceObject(writeSurface);
  } cudaGraphicsUnmapResources(1, &_viewCudaResource);

  cudaStreamSynchronize(0);
}

void copy_to_surface(float * _array, float _mul, int2 _dims, cudaGraphicsResource_t & _viewCudaResource) {
  cudaGraphicsMapResources(1, &_viewCudaResource); {
    cudaArray_t writeArray;
    cudaGraphicsSubResourceGetMappedArray(&writeArray, _viewCudaResource, 0, 0);
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    cudaCreateSurfaceObject(&writeSurface, &wdsc);
    BlockHelper bh(_dims);
    d_to_rgba<<<bh.grid, bh.block>>>(_array, _mul, _dims, writeSurface);
    cudaDestroySurfaceObject(writeSurface);
  } cudaGraphicsUnmapResources(1, &_viewCudaResource);

  cudaStreamSynchronize(0);
}
