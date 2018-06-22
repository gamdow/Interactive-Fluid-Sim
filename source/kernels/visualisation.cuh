#pragma once

#include <cuda_runtime.h>

#include "../data/resolution.cuh"

__global__ void scalar_to_rgba(float4 * o_buffer, float const * _buffer, Resolution _buffer_res, float _multiplier);

// Render 2D field (i.e. velocity) by treating as HSV (hue=direction, saturation=1, value=magnitude) and converting to RGBA
__global__ void vector_field_to_rgba(float4 * o_buffer, float2 const * _buffer, Resolution _buffer_res, float _power);

// Render 4D field by operating on it with a 4x3 matrix, where the rows are RGB values (a colour for each dimension).
__global__ void map_to_rgba(float4 * o_buffer, float4 const * _buffer, Resolution _buffer_res, float3 const * _map);

__global__ void min(float4 * o, float4 const * i, Resolution _buffer_res);

__global__ void max(float4 * o, float4 const * i, Resolution _buffer_res);

__global__ void scaleRGB(float4 * io, float4 _min, float4 _max, Resolution _buffer_res);
