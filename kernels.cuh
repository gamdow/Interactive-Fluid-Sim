#pragma once

#include <cuda_runtime.h>


void kernels_init(int2 _dims);
void kernels_shutdown();

void copy_to_vel_texture(float2 * _array, int2 _dims);

__global__ void advect_velocity(int2 _res, float _dt, float2 _rdx, float2 * o_velocity);
__global__ void calc_divergence(float2 * _vel, float * _fluid, int2 _dims, float2 _spaceI2, float * o_div);
__global__ void pressure_decay(float * _fluid, int2 _dims, float * io_pres);
__global__ void jacobi_solve(float * _b, float * _validCells, int2 _dims, float alpha, float beta, float * _x, float * o_x);
__global__ void pressure_solve(float * _div, float * _fluid, int2 _dims, float2 _space, float * i_pres, float * o_pres);
__global__ void sub_gradient(float * _pressure, float * _fluid, int2 _dims, float2 _spaceI2, float2 * io_velocity);
__global__ void enforce_slip(float * _fluid, int2 _dims, float2 * io_velocity);
__global__ void hsv_to_rgba(float2 * _array, float _mul, int2 _dims, cudaSurfaceObject_t o_surface);
__global__ void d_to_rgba(float * _array, float _mul, int2 _dims, cudaSurfaceObject_t o_surface);
