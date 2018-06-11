#include "simulation.hpp"

#include <iostream>
#include <typeinfo>

#include "cuda/helper_math.h"
#include "cuda/helper_cuda.h"
#include "kernels/kernels_wrapper.cuh"

void lerp(float & _from, float _to) {
  _from = _to * 0.05f + _from * 0.95f;
}

Simulation::Simulation(KernelsWrapper & _kers, int _pressure_steps)
  : Debug<Simulation>("Constructing Simluation Device Buffers:")
  , PRESSURE_SOLVER_STEPS(_pressure_steps)
  , __f4temp(_kers.getBufferRes().size)
  , __kernels(_kers)
  , __velocity(_kers.getBufferRes().size)
  , __fluidCells(_kers.getBufferRes().size)
  , __divergence(_kers.getBufferRes().size)
  , __pressure(_kers.getBufferRes().size)
  , __smoke(_kers.getBufferRes().size)
  , __color_map(4)
  , __f1temp(_kers.getBufferRes().size)
  , __min_rgb(0.0f)
  , __max_rgb(1.0f)
{
  reset();

  __color_map[0] = make_float3(1.0f, 0.5f, 0.5f) * 0.5f; // red
  __color_map[1] = make_float3(0.8f, 0.2f, 1.0f) * 0.5f;
  __color_map[2] = make_float3(0.3f, 1.0f, 0.6f) * 0.5f;
  __color_map[3] = make_float3(0.5f, 0.5f, 0.5f) * 0.5f;
  __color_map.copyHostToDevice();

  // _kers.getBufferRes().print("\tResolution");
  //
  // std::cout << "\tTotal: " << __velocity.getSizeBytes() + __fluidCells.getSizeBytes() + __divergence.getSizeBytes() + __pressure.getSizeBytes() + __smoke.getSizeBytes() + __kernels.getBufferRes().size * (sizeof(float) + sizeof(float2)) << " bytes" << std::endl;
}

Simulation::~Simulation() {
}

void Simulation::advectVelocity(float2 _rd, float _dt) {
  __kernels.advectVelocity(__velocity.device(), _rd, _dt);
}

void Simulation::step(Mode _mode, float2 _d, float _dt, float _mul) {

  switch(_mode) {
    case Mode::smoke:
      __kernels.float42rgba(__f4temp, __smoke.device(), __color_map.device()); break;
    case Mode::velocity:
      __kernels.hsv2rgba(__f4temp, __velocity.device(), 1.0f); break;
    case Mode::divergence:
      __kernels.d2rgba(__f4temp, __divergence.device(), 1.0f); break;
    case Mode::pressure:
      __kernels.d2rgba(__f4temp, __pressure.device(), 1.0f); break;
    case Mode::fluid:
      __kernels.d2rgba(__f4temp, __fluidCells.device(), 1.0f); break;
  }

  // Some smoothed auto brigtness adjustment for the visualisations
  float4 min4, max4;
  __kernels.minMaxReduce(min4, max4, __f4temp);
  lerp(__min_rgb, fmaxf(fminf(fminf(min4.x, min4.y), min4.z), 0.0f));
  lerp(__max_rgb, fminf(fmaxf(fmaxf(max4.x, max4.y), max4.z), 1.0f));
  min4 = make_float4(make_float3(__min_rgb), 0.0f);
  max4 = make_float4(make_float3(__max_rgb), 1.0f);
  __kernels.scale(__f4temp, min4, max4);

  float2 rd = make_float2(1.0f / _d.x, 1.0f / _d.y);
  float2 r2d = make_float2(1.0f / (2.0f * _d.x), 1.0f / (2.0f * _d.y));
  advectVelocity(rd, _dt);
  __kernels.calcDivergence(__divergence.device(), __velocity.device(), __fluidCells.device(), rd);
  __kernels.pressureDecay(__pressure.device(), __fluidCells.device());
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
    __kernels.pressureSolve(__f1temp, __pressure.device(), __divergence.device(), __fluidCells.device(), _d);
    __f1temp.swap(__pressure.device());
  }
  __kernels.subGradient(__velocity.device(), __pressure.device(), __fluidCells.device(), r2d);
  __kernels.enforceSlip(__velocity.device(), __fluidCells.device());
  __kernels.applyAdvection(__smoke.device(), __velocity.device(), __fluidCells.device(), _dt, rd);
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
  for(int i = 0; i < buffer_res.width; ++i) {
    for(int j = 0; j < buffer_res.buffer; ++j) {
      __fluidCells[i + j * buffer_res.width] = 0.0f;
    }
    for(int j = buffer_res.height - buffer_res.buffer; j < buffer_res.height; ++j) {
      __fluidCells[i + j * buffer_res.width] = 0.0f;
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
  __f1temp.reset();
  // __f2tempA.reset();
  // __f2tempB.reset();
  // __f2tempC.reset();

  Resolution const & buffer_res = __kernels.getBufferRes();
  for(int i = 0; i < buffer_res.width; ++i) {
    for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
      __fluidCells[i + j * buffer_res.width] = 1.0f;
    }
  }

  // for(int i = 0; i < buffer_res.width; ++i) {
  //   for(int j = 0; j < buffer_res.height; ++j) {
  //     if(i >= buffer_res.buffer + 20 && i < buffer_res.buffer + 40 && j >= buffer_res.buffer + 10 && j < buffer_res.buffer + 20) {
  //       __fluidCells[i + j * buffer_res.width] = 0.0f;
  //     }
  //   }
  // }

  // for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
  //   for(int i = 0; i < buffer_res.buffer; ++i) {
  //     __fluidCells[i + j * buffer_res.width] = 0.0f;
  //   }
  //   for(int i = buffer_res.width - buffer_res.buffer; i < buffer_res.width; ++i) {
  //     __fluidCells[i + j * buffer_res.width] = 0.0f;
  //   }
  // }

  // for(int k = 2; k < 3; ++k) {
  //   int odd = k % 2;
  //   for(int l = 1; l < 2 + odd; ++l) {
  //     float2 center = make_float2(k * 80 - 2.5f * 80.f + buffer_res.width / 2, l * 80 - 2.5f * 80.f + 100 - 40 * odd + buffer_res.height / 2);
  //     for(int i = 0; i < buffer_res.width; ++i) {
  //       for(int j = 0; j < buffer_res.height; ++j) {
  //         if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 5000) {
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

void BFECCSimulation::advectVelocity(float2 _rd, float _dt) {
  // Backwards-Forwards Error Compensation & Correction (BFECC). Only stable for low V
  __kernels.advectVelocity(__f2temp, __velocity.device(), _rd, _dt);
  __kernels.advectVelocity(__f2temp, _rd, -_dt);
  __kernels.sum(__velocity.device(), 1.5f, __velocity.device(), -.5f, __f2temp);
  __kernels.advectVelocity(__velocity.device(), _rd, _dt);
}

void LBFECCSimulation::advectVelocity(float2 _rd, float _dt) {
  // Backwards-Forwards Error Compensation & Correction (BFECC) with limiting.
  __kernels.advectVelocity(__f2tempA, __velocity.device(), _rd, _dt);
  __kernels.advectVelocity(__f2tempA, _rd, -_dt);
  __kernels.sum(__f2tempB, .5f, __velocity.device(), -.5f, __f2tempA);
  __kernels.sum(__f2tempA, 1.0f, __velocity.device(), 1.0f, __f2tempB);
  __kernels.advectVelocity(__f2tempA, _rd, _dt);
  __kernels.advectVelocity(__f2tempA, _rd, -_dt);
  __kernels.sum(__f2tempA, 1.0f, __f2tempA, 1.0f, __f2tempB);
  __kernels.sum(__f2tempA, 1.0f, __velocity.device(), -1.0f, __f2tempA);
  __kernels.limitAdvection(__f2tempC, __f2tempB, __f2tempA);
  __kernels.sum(__velocity.device(), 1.0f, __velocity.device(), 1.0f, __f2tempC);
  __kernels.advectVelocity(__velocity.device(), _rd, _dt);
}

BFECCSimulation::BFECCSimulation(KernelsWrapper & _kers, int _pressure_steps)
  : Simulation(_kers, _pressure_steps)
  , __f2temp(_kers.getBufferRes().size)
{
}

LBFECCSimulation::LBFECCSimulation(KernelsWrapper & _kers, int _pressure_steps)
  : Simulation(_kers, _pressure_steps)
  , __f2tempA(_kers.getBufferRes().size)
  , __f2tempB(_kers.getBufferRes().size)
  , __f2tempC(_kers.getBufferRes().size)
{
}
