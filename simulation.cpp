#include "simulation.hpp"

#include "helper_math.h"
#include "kernels.cuh"
#include "configuration.cuh"

int const PRESSURE_SOLVER_STEPS = 100;

Simulation::Simulation(int2 _dimensions, int _buffer, dim3 _block_size)
  : Kernels(_dimensions, _buffer, _block_size)
  , __buffered_size(__buffer_spec.x * __buffer_spec.y)
  , __velocity(__buffered_size)
  , __fluidCells(__buffered_size)
{
  size_t buffer_bytes = __buffered_size * sizeof(float);
  cudaMalloc((void **) & __divergence, buffer_bytes); cudaMemset(__divergence, 0, buffer_bytes);
  cudaMalloc((void **) & __pressure, buffer_bytes); cudaMemset(__pressure, 0, buffer_bytes);
  cudaMalloc((void **) & __buffer, buffer_bytes); cudaMemset(__buffer, 0, buffer_bytes);

  for(int i = 0; i < __buffer_spec.x; ++i) {
    for(int j = 0; j < __buffer_spec.y; ++j) {
      __fluidCells[i + j * __buffer_spec.x] = 1.0f;
    }
  }

  // for(int k = 0; k < 5; ++k) {
  //   for(int l = 0; l < 3; ++l) {
  //     float2 center = make_float2(k * 80 + 100 + 50 * ((l + 1) % 2), l * 70 + 120);
  //     for(int i = 0; i < BUFFERED_DIMS.x; ++i) {
  //       for(int j = 0; j < BUFFERED_DIMS.y; ++j) {
  //         if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000) {
  //           fluidCells[i + j * BUFFERED_DIMS.x] = 0.0f;
  //         }
  //       }
  //     }
  //   }
  // }

  for(int k = 0; k < 5; ++k) {
    int odd = k % 2;
    for(int l = 0; l < 3 + odd; ++l) {
      float2 center = make_float2(k * 80 + 200, l * 80 + 100 - 40 * odd);
      for(int i = 0; i < __buffer_spec.x; ++i) {
        for(int j = 0; j < __buffer_spec.y; ++j) {
          if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000) {
            __fluidCells[i + j * __buffer_spec.x] = 0.0f;
          }
        }
      }
    }
  }

}

Simulation::~Simulation() {
  cudaFree(__buffer);
  cudaFree(__pressure);
  cudaFree(__divergence);
}

void Simulation::step(float2 _d, float _dt) {
  float2 rd = make_float2(1.0f / _d.x, 1.0f / _d.y);
  float2 r2d = make_float2(1.0f / (2.0f * _d.x), 1.0f / (2.0f * _d.y));

  advectVelocity(__velocity.device, rd, _dt);
  calcDivergence(__divergence, __velocity.device, __fluidCells.device, r2d);
  pressureDecay(__pressure, __fluidCells.device);
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
    pressureSolve(__buffer, __pressure, __divergence, __fluidCells.device, _d);
    float * temp = __pressure;
    __pressure = __buffer;
    __buffer = temp;
  }
  subGradient(__velocity.device, __pressure, __fluidCells.device, r2d);
  enforceSlip(__velocity.device, __fluidCells.device);
}

void Simulation::applyBoundary(float _vel) {
  for(int j = 0; j < __buffer_spec.y; ++j) {
    for(int i = 0; i < 50; ++i) {
      __velocity[i + j * __buffer_spec.x] = make_float2(_vel, 0.0f);
    }
    for(int i = __buffer_spec.x - 50; i < __buffer_spec.x; ++i) {
      __velocity[i + j * __buffer_spec.x] = make_float2(_vel, 0.0f);
    }
  }

  for(int i = 0; i < __buffer_spec.x; ++i) {
    for(int j = 0; j < 1; ++j) {
      __velocity[i + j * __buffer_spec.x] = make_float2(0.0f, 0.0f);
      __fluidCells[i + j * __buffer_spec.x] = 0.0f;
    }
    for(int j = __buffer_spec.y - 1; j < __buffer_spec.y; ++j) {
      __velocity[i + j * __buffer_spec.x] = make_float2(0.0f, 0.0f);
      __fluidCells[i + j * __buffer_spec.x] = 0.0f;
    }
  }

  __velocity.copyHostToDevice();
  __fluidCells.copyHostToDevice();
}
