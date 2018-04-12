#include "simulation.hpp"

#include <iostream>
#include <typeinfo>

#include "helper_math.h"
#include "helper_cuda.h"
#include "kernels_wrapper.cuh"

int const PRESSURE_SOLVER_STEPS = 50;

SimulationBase::SimulationBase(KernelsWrapper const & _kers) {
  std::cout << "Constructing Simluation Device Buffers:" << std::endl;
  _kers.getBufferRes().print("\tResolution");
}

Simulation::Simulation(KernelsWrapper & _kers)
  : SimulationBase(_kers)
  , __kernels(_kers)
  , __velocity(_kers.getBufferRes().size)
  , __fluidCells(_kers.getBufferRes().size)
  , __divergence(_kers.getBufferRes().size)
  , __pressure(_kers.getBufferRes().size)
  , __smoke(_kers.getBufferRes().size)
{
  reportCudaMalloc(__f1temp, __kernels.getBufferRes().size);
  reportCudaMalloc(__f2temp, __kernels.getBufferRes().size);
  reset();

  std::cout << "\tTotal: " << __velocity.getTotalBytes() + __fluidCells.getTotalBytes() + __divergence.getTotalBytes() + __pressure.getTotalBytes() + __smoke.getTotalBytes() + __kernels.getBufferRes().size * (sizeof(float) + sizeof(float2)) << " bytes" << std::endl;
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
  checkCudaErrors(cudaMemset(__f2temp, 0, buffer_res.size * sizeof(float2)));
  for(int i = 0; i < buffer_res.width; ++i) {
    for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
      __fluidCells[i + j * buffer_res.width] = 1.0f;
    }
  }
  for(int k = 0; k < 5; ++k) {
    int odd = k % 2;
    for(int l = 0; l < 3 + odd; ++l) {
      float2 center = make_float2(k * 80 - 2.5f * 80.f + buffer_res.width / 2, l * 80 - 2.5f * 80.f + 100 - 40 * odd + buffer_res.height / 2);
      for(int i = 0; i < buffer_res.width; ++i) {
        for(int j = 0; j < buffer_res.height; ++j) {
          if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000) {
            __fluidCells[i + j * buffer_res.width] = 0.0f;
          }
        }
      }
    }
  }
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
