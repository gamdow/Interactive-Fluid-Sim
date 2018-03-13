#include "simulation.hpp"

#include "helper_math.h"
#include "kernels.cuh"

int const PRESSURE_SOLVER_STEPS = 100;

Simulation::Simulation(Kernels & _kernels)
  : __kernels(_kernels)
  , __velocity(_kernels.__buffered_size)
  , __fluidCells(_kernels.__buffered_size)
{
  size_t buffer_bytes = _kernels.__buffered_size * sizeof(float);
  checkCudaErrors(cudaMalloc((void **) & __divergence, buffer_bytes)); checkCudaErrors(cudaMemset(__divergence, 0, buffer_bytes));
  checkCudaErrors(cudaMalloc((void **) & __pressure, buffer_bytes)); checkCudaErrors(cudaMemset(__pressure, 0, buffer_bytes));
  checkCudaErrors(cudaMalloc((void **) & __buffer, buffer_bytes)); checkCudaErrors(cudaMemset(__buffer, 0, buffer_bytes));

  for(int i = 0; i < _kernels.__buffer_spec.x; ++i) {
    for(int j = 0; j < _kernels.__buffer_spec.y; ++j) {
      __fluidCells[i + j * _kernels.__buffer_spec.x] = 1.0f;
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
      for(int i = 0; i < _kernels.__buffer_spec.x; ++i) {
        for(int j = 0; j < _kernels.__buffer_spec.y; ++j) {
          if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000) {
            __fluidCells[i + j * _kernels.__buffer_spec.x] = 0.0f;
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

  __kernels.advectVelocity(__velocity.device, rd, _dt);
  __kernels.calcDivergence(__divergence, __velocity.device, __fluidCells.device, r2d);
  __kernels.pressureDecay(__pressure, __fluidCells.device);
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
    __kernels.pressureSolve(__buffer, __pressure, __divergence, __fluidCells.device, _d);
    float * temp = __pressure;
    __pressure = __buffer;
    __buffer = temp;
  }
  __kernels.subGradient(__velocity.device, __pressure, __fluidCells.device, r2d);
  __kernels.enforceSlip(__velocity.device, __fluidCells.device);
}

void Simulation::applyBoundary(float _vel) {
  for(int j = 0; j < __kernels.__buffer_spec.y; ++j) {
    for(int i = 0; i < 50; ++i) {
      __velocity[i + j * __kernels.__buffer_spec.x] = make_float2(_vel, 0.0f);
    }
    for(int i = __kernels.__buffer_spec.x - 50; i < __kernels.__buffer_spec.x; ++i) {
      __velocity[i + j * __kernels.__buffer_spec.x] = make_float2(_vel, 0.0f);
    }
  }

  for(int i = 0; i < __kernels.__buffer_spec.x; ++i) {
    for(int j = 0; j < 1; ++j) {
      __velocity[i + j * __kernels.__buffer_spec.x] = make_float2(0.0f, 0.0f);
      __fluidCells[i + j * __kernels.__buffer_spec.x] = 0.0f;
    }
    for(int j = __kernels.__buffer_spec.y - 1; j < __kernels.__buffer_spec.y; ++j) {
      __velocity[i + j * __kernels.__buffer_spec.x] = make_float2(0.0f, 0.0f);
      __fluidCells[i + j * __kernels.__buffer_spec.x] = 0.0f;
    }
  }

  __velocity.copyHostToDevice();
  __fluidCells.copyHostToDevice();
}
