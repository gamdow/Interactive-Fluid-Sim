#include "simulation.hpp"

#include <iostream>
#include <typeinfo>

#include "helper_math.h"
#include "helper_cuda.h"
#include "kernels_wrapper.cuh"

SimulationBase::SimulationBase(KernelsWrapper const & _kers) {
  std::cout << "Constructing Simluation Device Buffers:" << std::endl;
  _kers.getBufferRes().print("\tResolution");
}

Simulation::Simulation(KernelsWrapper & _kers, int _pressure_steps)
  : SimulationBase(_kers)
  , PRESSURE_SOLVER_STEPS(_pressure_steps)
  , __kernels(_kers)
  , __velocity(_kers.getBufferRes().size)
  , __fluidCells(_kers.getBufferRes().size)
  , __divergence(_kers.getBufferRes().size)
  , __pressure(_kers.getBufferRes().size)
  , __smoke(_kers.getBufferRes().size)
{
  reportCudaMalloc(__f1temp, __kernels.getBufferRes().size);
  reportCudaMalloc(__f2tempA, __kernels.getBufferRes().size);
  reportCudaMalloc(__f2tempB, __kernels.getBufferRes().size);
  reportCudaMalloc(__f2tempC, __kernels.getBufferRes().size);
  reset();

  std::cout << "\tTotal: " << __velocity.getTotalBytes() + __fluidCells.getTotalBytes() + __divergence.getTotalBytes() + __pressure.getTotalBytes() + __smoke.getTotalBytes() + __kernels.getBufferRes().size * (sizeof(float) + sizeof(float2)) << " bytes" << std::endl;
}

Simulation::~Simulation() {
  cudaFree(__f1temp);
  cudaFree(__f2tempA);
  cudaFree(__f2tempB);
  cudaFree(__f2tempC);
}

void Simulation::step(float2 _d, float _dt) {
  float2 rd = make_float2(1.0f / _d.x, 1.0f / _d.y);
  float2 r2d = make_float2(1.0f / (2.0f * _d.x), 1.0f / (2.0f * _d.y));

  switch(0) {
    case 0: {
    } break;
    case 1: {
      // Backwards-Forwards Error Compensation & Correction (BFECC). Only stable for low V
      __kernels.advectVelocity(__f2tempA, __velocity.device, rd, _dt);
      __kernels.advectVelocity(__f2tempA, rd, -_dt);
      __kernels.sum(__velocity.device, 1.5f, __velocity.device, -.5f, __f2tempA);
    } break;
    case 2: {
      // Backwards-Forwards Error Compensation & Correction (BFECC) with limiting.
      __kernels.advectVelocity(__f2tempA, __velocity.device, rd, _dt);
      __kernels.advectVelocity(__f2tempA, rd, -_dt);
      __kernels.sum(__f2tempB, .5f, __velocity.device, -.5f, __f2tempA);
      __kernels.sum(__f2tempA, 1.0f, __velocity.device, 1.0f, __f2tempB);
      __kernels.advectVelocity(__f2tempA, rd, _dt);
      __kernels.advectVelocity(__f2tempA, rd, -_dt);
      __kernels.sum(__f2tempA, 1.0f, __f2tempA, 1.0f, __f2tempB);
      __kernels.sum(__f2tempA, 1.0f, __velocity.device, -1.0f, __f2tempA);
      __kernels.limitAdvection(__f2tempC, __f2tempB, __f2tempA);
      __kernels.sum(__velocity.device, 1.0f, __velocity.device, 1.0f, __f2tempC);
    } break;
  }
  __kernels.advectVelocity(__velocity.device, rd, _dt);

  // if(false) {
  //   // Backwards-Forwards Error Compensation & Correction (BFECC). Only stable for low V
  //   // __kernels.advectVelocity(__f2temp, __velocity.device, rd, _dt);
  //   // __kernels.advectVelocity(__f2temp, rd, -_dt);
  //   // __kernels.sum(__velocity.device, 1.5f, __velocity.device, -.5f, __f2temp);
  //   // __kernels.advectVelocity(__velocity.device, rd, _dt);
  // } else {
  //   __kernels.advectVelocity(__velocity.device, rd, _dt);
  // }
  __kernels.calcDivergence(__divergence.device, __velocity.device, __fluidCells.device, rd);
  __kernels.pressureDecay(__pressure.device, __fluidCells.device);
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
    __kernels.pressureSolve(__f1temp, __pressure.device, __divergence.device, __fluidCells.device, _d);
    float * temp = __pressure.device;
    __pressure.device = __f1temp;
    __f1temp = temp;
  }
  __kernels.subGradient(__velocity.device, __pressure.device, __fluidCells.device, r2d);
  //__kernels.enforceSlip(__velocity.device, __fluidCells.device);
  __kernels.applyAdvection(__smoke.device, __velocity.device, __fluidCells.device, _dt, rd);
}

void Simulation::applyBoundary(float _vel) {
  __velocity.copyDeviceToHost();
  Resolution const & buffer_res = __kernels.getBufferRes();
  for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
    for(int i = 0; i < buffer_res.buffer * 2; ++i) {
      __velocity[i + j * buffer_res.width] = make_float2(_vel, 0.f);
    }
    for(int i = buffer_res.width - buffer_res.buffer * 2; i < buffer_res.width; ++i) {
      __velocity[i + j * buffer_res.width] = make_float2(_vel, 0.f);
    }
  }
  __velocity.copyHostToDevice();
}

void Simulation::applySmoke() {
  __smoke.copyDeviceToHost();
  Resolution const & buffer_res = __kernels.getBufferRes();
  for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
    for(int i = 0; i < buffer_res.buffer * 2; ++i) {
      int z = (j / 20) % 4;
      __smoke[i + j * buffer_res.width] = make_float4(
        z == 0 ? 1.0f : 0.f,
        z == 1 ? 1.0f : 0.f,
        z == 2 ? 1.0f : 0.f,
        z == 3 ? 1.0f : 0.f
      ) * powf(cosf((j - buffer_res.buffer) * 3.14159f * (2.0f / 40)),2);
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
  Resolution const & buffer_res = __kernels.getBufferRes();
  checkCudaErrors(cudaMemset(__f1temp, 0, buffer_res.size * sizeof(float)));
  checkCudaErrors(cudaMemset(__f2tempA, 0, buffer_res.size * sizeof(float2)));
  checkCudaErrors(cudaMemset(__f2tempB, 0, buffer_res.size * sizeof(float2)));
  checkCudaErrors(cudaMemset(__f2tempC, 0, buffer_res.size * sizeof(float2)));
  for(int i = 0; i < buffer_res.width; ++i) {
    for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
      __fluidCells[i + j * buffer_res.width] = 1.0f;
    }
  }

  for(int i = 0; i < buffer_res.width; ++i) {
    for(int j = 0; j < buffer_res.height; ++j) {
      if(i >= buffer_res.buffer + 20 && i < buffer_res.buffer + 40 && j >= buffer_res.buffer + 10 && j < buffer_res.buffer + 20) {
        __fluidCells[i + j * buffer_res.width] = 0.0f;
      }
    }
  }

  // for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
  //   for(int i = 0; i < buffer_res.buffer; ++i) {
  //     __fluidCells[i + j * buffer_res.width] = 0.0f;
  //   }
  //   for(int i = buffer_res.width - buffer_res.buffer; i < buffer_res.width; ++i) {
  //     __fluidCells[i + j * buffer_res.width] = 0.0f;
  //   }
  // }

  // for(int k = 0; k < 5; ++k) {
  //   int odd = k % 2;
  //   for(int l = 0; l < 3 + odd; ++l) {
  //     float2 center = make_float2(k * 80 - 2.5f * 80.f + buffer_res.width / 2, l * 80 - 2.5f * 80.f + 100 - 40 * odd + buffer_res.height / 2);
  //     for(int i = 0; i < buffer_res.width; ++i) {
  //       for(int j = 0; j < buffer_res.height; ++j) {
  //         if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000) {
  //           __fluidCells[i + j * buffer_res.width] = 0.0f;
  //         }
  //       }
  //     }
  //   }
  // }
  for(int i = 0; i < buffer_res.width; ++i) {
    for(int j = 0; j < buffer_res.buffer; ++j) {
      __fluidCells[i + j * buffer_res.width] = 0.0f;
    }
    for(int j = buffer_res.height - buffer_res.buffer; j < buffer_res.height; ++j) {
      __fluidCells[i + j * buffer_res.width] = 0.0f;
    }
  }
  __fluidCells.copyHostToDevice();
}
