#pragma once

#include "../cuda/helper_math.h"

float const PI = 3.14159265359f;

inline __device__ float4 float2_to_hsl(float2 _in, float _power) {
  float v = __powf(_in.x * _in.x + _in.y * _in.y, _power);
  float h = 6.0f * (atan2f(-_in.x, -_in.y) / (2 * PI) + 0.5);
  float hi = floorf(h);
  float f = h - hi;
  float q = v * (1 - f);
  float t = v * f;
  float3 rgb;
  switch((int)hi) {
    default: rgb = make_float3(v, t, 0.0f); break;
    case 1: rgb = make_float3(q, v, 0.0f); break;
    case 2: rgb = make_float3(0.0f, v, t); break;
    case 3: rgb = make_float3(0.0f, q, v); break;
    case 4: rgb = make_float3(t, 0.0f, v); break;
    case 5: rgb = make_float3(v, 0.0f, q); break;
  }
  return make_float4(rgb, fmin(rgb.x + rgb.y + rgb.z, 1.f));
}
