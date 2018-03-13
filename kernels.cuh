#pragma once

#include <cuda_runtime.h>

int2 const BLOCK_SIZE = {32, 18};

void kernels_init(int2 _dims);
void kernels_shutdown();

void simulation_step(int2 _dims, float2 _d, float _dt, float * i_fluidCells, float2 * io_velocity, float * o_divergence, float * o_pressure, float * o_buffer);
void copy_to_surface(float2 * _array, int2 _dims, cudaGraphicsResource_t & _viewCudaResource);
void copy_to_surface(float * _array, float _mul, int2 _dims, cudaGraphicsResource_t & _viewCudaResource);
