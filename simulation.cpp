#include "simulation.hpp"

#include "helper_math.h"
#include "kernels.cuh"

int const PRESSURE_SOLVER_STEPS = 100;

Simulation::Simulation(Kernels & _kernels)
  : __kernels(_kernels)
  , __velocity(_kernels.__buffered_size)
  , __fluidCells(_kernels.__buffered_size)
  , __divergence(_kernels.__buffered_size)
  , __pressure(_kernels.__buffered_size)
  , __smoke(_kernels.__buffered_size)
{
  checkCudaErrors(cudaMalloc((void **) & __f2temp1, _kernels.__buffered_size * sizeof(float2)));
  checkCudaErrors(cudaMalloc((void **) & __f2temp2, _kernels.__buffered_size * sizeof(float2)));
  checkCudaErrors(cudaMalloc((void **) & __f1temp, _kernels.__buffered_size * sizeof(float)));
  reset();
}

Simulation::~Simulation() {
  cudaFree(__f1temp);
  cudaFree(__f2temp1);
  cudaFree(__f2temp2);
}

void Simulation::step(float2 _d, float _dt) {
  float2 rd = make_float2(1.0f / _d.x, 1.0f / _d.y);
  float2 r2d = make_float2(1.0f / (2.0f * _d.x), 1.0f / (2.0f * _d.y));

  if(true) {
    // Backwards-Forwards Error Compensation & Correction (BFECC)
    __kernels.advectVelocity(__f2temp1, __velocity.device, rd, _dt);
    __kernels.advectVelocity(__f2temp1, rd, -_dt);
    __kernels.sum(__velocity.device, 1.5f, __velocity.device, -.5f, __f2temp1);
    __kernels.advectVelocity(__velocity.device, rd, _dt);
  } else {
    __kernels.advectVelocity(__velocity.device, rd, _dt);
  }
  __kernels.calcDivergence(__divergence.device, __velocity.device, __fluidCells.device, r2d);
  __kernels.pressureDecay(__pressure.device, __fluidCells.device);
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
    __kernels.pressureSolve(__f1temp, __pressure.device, __divergence.device, __fluidCells.device, _d);
    float * temp = __pressure.device;
    __pressure.device = __f1temp;
    __f1temp = temp;
  }
  __kernels.subGradient(__velocity.device, __pressure.device, __fluidCells.device, r2d);
  __kernels.enforceSlip(__velocity.device, __fluidCells.device);
  __kernels.applyAdvection(__smoke.device, __velocity.device, _dt, rd);
}

void Simulation::applyBoundary(float _vel) {
  reset();
  // __velocity.copyDeviceToHost();
  // __fluidCells.copyDeviceToHost();
  // __divergence.copyDeviceToHost();

  for(int j = __kernels.__buffer_spec.z; j < __kernels.__buffer_spec.y - __kernels.__buffer_spec.z; ++j) {
    for(int i = 0; i < __kernels.__buffer_spec.z; ++i) {
      __velocity[i + j * __kernels.__buffer_spec.x] = make_float2(_vel, 0.f);
      __divergence[i + j * __kernels.__buffer_spec.x] = _vel * 10.f;
      __pressure[i + j * __kernels.__buffer_spec.x] = _vel / 50.f;
    }
    for(int i = __kernels.__buffer_spec.x - __kernels.__buffer_spec.z; i < __kernels.__buffer_spec.x; ++i) {
      __velocity[i + j * __kernels.__buffer_spec.x] = make_float2(_vel, 0.f);
      __divergence[i + j * __kernels.__buffer_spec.x] = -_vel * 10.f;
      __pressure[i + j * __kernels.__buffer_spec.x] = -_vel / 50.f;
    }
  }

  for(int i = 0; i < __kernels.__buffer_spec.x; ++i) {
    for(int j = 0; j < __kernels.__buffer_spec.z; ++j) {
      __fluidCells[i + j * __kernels.__buffer_spec.x] = 0.0f;
    }
    for(int j = __kernels.__buffer_spec.y - __kernels.__buffer_spec.z; j < __kernels.__buffer_spec.y; ++j) {
      __fluidCells[i + j * __kernels.__buffer_spec.x] = 0.0f;
      __velocity[i + j * __kernels.__buffer_spec.x] = make_float2(_vel, 0.f);
    }
  }

  // __velocity.copyHostToDevice();
  __fluidCells.copyHostToDevice();
  __pressure.copyHostToDevice();
}

void Simulation::applySmoke() {
  __smoke.copyDeviceToHost();

  for(int j = __kernels.__buffer_spec.z; j < __kernels.__buffer_spec.y - __kernels.__buffer_spec.z; ++j) {
    for(int i = 0; i < __kernels.__buffer_spec.z + 20; ++i) {
      __smoke[i + j * __kernels.__buffer_spec.x] = 1.0f;
    }
  }

  __smoke.copyHostToDevice();
}


void Simulation::reset() {
  __velocity.reset();
  __fluidCells.reset();
  __divergence.reset();
  __pressure.reset();
  __smoke.reset();
  checkCudaErrors(cudaMemset(__f2temp1, 0, __kernels.__buffered_size * sizeof(float2)));
  checkCudaErrors(cudaMemset(__f2temp2, 0, __kernels.__buffered_size * sizeof(float2)));
  checkCudaErrors(cudaMemset(__f1temp, 0, __kernels.__buffered_size * sizeof(float)));

  for(int i = 0; i < __kernels.__buffer_spec.x; ++i) {
    for(int j = __kernels.__buffer_spec.z; j < __kernels.__buffer_spec.y - __kernels.__buffer_spec.z; ++j) {
      __fluidCells[i + j * __kernels.__buffer_spec.x] = 1.0f;
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
      float2 center = make_float2(k * 80 - 2.5f * 80.f + __kernels.__buffer_spec.x / 2, l * 80 - 2.5f * 80.f + 100 - 40 * odd + __kernels.__buffer_spec.y / 2);
      for(int i = 0; i < __kernels.__buffer_spec.x; ++i) {
        for(int j = 0; j < __kernels.__buffer_spec.y; ++j) {
          if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000) {
            __fluidCells[i + j * __kernels.__buffer_spec.x] = 0.0f;
          }
        }
      }
    }
  }
}
