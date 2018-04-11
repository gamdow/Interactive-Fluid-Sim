#include "simulation.hpp"

#include "helper_math.h"
#include "kernels_wrapper.cuh"
#include "buffer_spec.cuh"

int const PRESSURE_SOLVER_STEPS = 50;

Simulation::Simulation(Kernels & _kernels)
  : __kernels(_kernels)
  , __buffer_spec(_kernels.getBufferSpec())
  , __velocity(_kernels.getBufferSpec().size)
  , __fluidCells(_kernels.getBufferSpec().size)
  , __divergence(_kernels.getBufferSpec().size)
  , __pressure(_kernels.getBufferSpec().size)
  , __smoke(_kernels.getBufferSpec().size)
{
  checkCudaErrors(cudaMalloc((void **) & __f2temp, __buffer_spec.size * sizeof(float2)));
  checkCudaErrors(cudaMalloc((void **) & __f1temp, __buffer_spec.size * sizeof(float)));
  reset();
}

Simulation::~Simulation() {
  cudaFree(__f1temp);
  cudaFree(__f2temp);
}

void Simulation::step(float2 _d, float _dt) {
  float2 rd = make_float2(1.0f / _d.x, 1.0f / _d.y);
  float2 r2d = make_float2(1.0f / (2.0f * _d.x), 1.0f / (2.0f * _d.y));

  if(true) {
    // Backwards-Forwards Error Compensation & Correction (BFECC)
    __kernels.advectVelocity(__f2temp, __velocity.device, rd, _dt);
    __kernels.advectVelocity(__f2temp, rd, -_dt);
    __kernels.sum(__velocity.device, 1.5f, __velocity.device, -.5f, __f2temp);
    __kernels.advectVelocity(__velocity.device, rd, _dt);
  } else {
    __kernels.advectVelocity(__velocity.device, rd, _dt);
  }
  __kernels.calcDivergence(__divergence.device, __velocity.device, __fluidCells.device, rd);
  __kernels.pressureDecay(__pressure.device, __fluidCells.device);
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
    __kernels.pressureSolve(__f1temp, __pressure.device, __divergence.device, __fluidCells.device, _d);
    float * temp = __pressure.device;
    __pressure.device = __f1temp;
    __f1temp = temp;
  }
  __kernels.subGradient(__velocity.device, __pressure.device, __fluidCells.device, r2d);
  __kernels.enforceSlip(__velocity.device, __fluidCells.device);
  __kernels.applyAdvection(__smoke.device, __velocity.device, __fluidCells.device, _dt, rd);
}

void Simulation::applyBoundary(float _vel) {
  __velocity.copyDeviceToHost();

  for(int j = __buffer_spec.buffer; j < __buffer_spec.height - __buffer_spec.buffer; ++j) {
    for(int i = 0; i < __buffer_spec.buffer * 2; ++i) {
      __velocity[i + j * __buffer_spec.width] = make_float2(_vel, 0.f);
    }
    for(int i = __buffer_spec.width - __buffer_spec.buffer * 2; i < __buffer_spec.width; ++i) {
      __velocity[i + j * __buffer_spec.width] = make_float2(_vel, 0.f);
    }
  }

  __velocity.copyHostToDevice();
}

void Simulation::applySmoke() {
  __smoke.copyDeviceToHost();

  for(int j = __buffer_spec.buffer; j < __buffer_spec.height - __buffer_spec.buffer; ++j) {
    for(int i = 0; i < __buffer_spec.buffer * 2; ++i) {
      int z = (j / 20) % 4;
      __smoke[i + j * __buffer_spec.width] = make_float4(
        z == 0 ? 1.0f : 0.f,
        z == 1 ? 1.0f : 0.f,
        z == 2 ? 1.0f : 0.f,
        z == 3 ? 1.0f : 0.f
      ) * powf(cosf((j - __buffer_spec.buffer) * 3.14159f * (2.0f / 40)),2);
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
  checkCudaErrors(cudaMemset(__f2temp, 0, __buffer_spec.size * sizeof(float2)));
  checkCudaErrors(cudaMemset(__f1temp, 0, __buffer_spec.size * sizeof(float)));

  for(int i = 0; i < __buffer_spec.width; ++i) {
    for(int j = __buffer_spec.buffer; j < __buffer_spec.height - __buffer_spec.buffer; ++j) {
      __fluidCells[i + j * __buffer_spec.width] = 1.0f;
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
      float2 center = make_float2(k * 80 - 2.5f * 80.f + __buffer_spec.width / 2, l * 80 - 2.5f * 80.f + 100 - 40 * odd + __buffer_spec.height / 2);
      for(int i = 0; i < __buffer_spec.width; ++i) {
        for(int j = 0; j < __buffer_spec.height; ++j) {
          if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000) {
            __fluidCells[i + j * __buffer_spec.width] = 0.0f;
          }
        }
      }
    }
  }

  for(int i = 0; i < __buffer_spec.width; ++i) {
    for(int j = 0; j < __buffer_spec.buffer; ++j) {
      __fluidCells[i + j * __buffer_spec.width] = 0.0f;
    }
    for(int j = __buffer_spec.height - __buffer_spec.buffer; j < __buffer_spec.height; ++j) {
      __fluidCells[i + j * __buffer_spec.width] = 0.0f;
    }
  }

  __fluidCells.copyHostToDevice();
}
