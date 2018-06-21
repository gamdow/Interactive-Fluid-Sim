#include "simulation.hpp"

#include <iostream>
#include <cuda_gl_interop.h>
//#include <typeinfo>

#include "debug.hpp"
#include "cuda/helper_math.h"
#include "cuda/helper_cuda.h"
#include "interface.hpp"
#include "kernels/kernels_wrapper.cuh"
#include "kernels/simulation_wrapper.cuh"

void lerp(float & _from, float _to) {
  _from = _to * 0.05f + _from * 0.95f;
}

Simulation::Simulation(Interface & _interface, OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx, KernelsWrapper & _kers, int _pressure_steps)
  : __interface(_interface)
  , __sim_kernels(_block_config, _buffer_width, _dx)
  , __kernels(_kers)
  , PRESSURE_SOLVER_STEPS(_pressure_steps)
  , __min_rgb(0.0f)
  , __max_rgb(1.0f)
  , __quad(_kers, _kers.resolution(), GL_RGBA32F, GL_RGBA, GL_FLOAT)
{
  format_out << "Constructing Simluation Device Buffers:" << std::endl;
  OutputIndent indent1;
  _kers.getBufferRes().print("Resolution");
  {
    Allocator alloc;
    __fluidCells.resize(alloc, _kers.getBufferRes().size);
    __velocity.resize(alloc, _kers.getBufferRes().size);
    __smoke.resize(alloc, _kers.getBufferRes().size);
    __f4temp.resize(alloc, _kers.getBufferRes().size);
    __color_map.resize(alloc, 4);
  }

  reset();

  __color_map[0] = make_float3(1.0f, 0.5f, 0.5f) * 0.5f; // red
  __color_map[1] = make_float3(0.8f, 0.2f, 1.0f) * 0.5f;
  __color_map[2] = make_float3(0.3f, 1.0f, 0.6f) * 0.5f;
  __color_map[3] = make_float3(0.5f, 0.5f, 0.5f) * 0.5f;
  __color_map.copyHostToDevice();
}

void Simulation::step(float2 _d, float _dt) {
  switch(__interface.mode()) {
    case Mode::smoke: __kernels.float42rgba(__f4temp, __sim_kernels.smoke(), __color_map.device()); break;
    case Mode::velocity: __kernels.hsv2rgba(__f4temp, __sim_kernels.velocity(), 1.0f); break;
    case Mode::divergence: __kernels.d2rgba(__f4temp, __sim_kernels.divergence(), 1.0f); break;
    case Mode::pressure: __kernels.d2rgba(__f4temp, __sim_kernels.pressure(), 1.0f); break;
    case Mode::fluid: __kernels.d2rgba(__f4temp, __sim_kernels.fluidCells(), 1.0f); break;
  }

  // Some smoothed auto brigtness adjustment for the visualisations
  float4 min4, max4;
  __kernels.minMaxReduce(min4, max4, __f4temp);
  lerp(__min_rgb, fmaxf(fminf(fminf(min4.x, min4.y), min4.z), 0.0f));
  lerp(__max_rgb, fminf(fmaxf(fmaxf(max4.x, max4.y), max4.z), 1.0f));
  min4 = make_float4(make_float3(__min_rgb), 0.0f);
  max4 = make_float4(make_float3(__max_rgb), 1.0f);
  __kernels.scale(__f4temp, min4, max4);

  __sim_kernels.advectVelocity(_dt);
  __sim_kernels.calcDivergence();
  __sim_kernels.pressureDecay();
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
    __sim_kernels.pressureSolveStep();
  }
  __sim_kernels.subGradient();
  __sim_kernels.enforceSlip();
  __sim_kernels.advectSmoke(_dt);
}

void Simulation::applyBoundary() {
  __velocity = __sim_kernels.velocity();
  float velocity = __interface.velocity();
  Resolution const & buffer_res = __kernels.getBufferRes();
  for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
    for(int i = 0; i < buffer_res.buffer * 2; ++i) {
      __velocity[i + j * buffer_res.width] = make_float2(velocity, 0.f);
    }
    for(int i = buffer_res.width - buffer_res.buffer * 2; i < buffer_res.width; ++i) {
      __velocity[i + j * buffer_res.width] = make_float2(velocity, 0.f);
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
  __sim_kernels.velocity() = __velocity;
}

void Simulation::applySmoke() {
  __smoke = __sim_kernels.smoke();
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
  __sim_kernels.smoke() = __smoke;
}

void Simulation::reset() {
  __velocity.reset();
  __fluidCells.reset();
  __smoke.reset();

  Resolution const & buffer_res = __kernels.getBufferRes();
  for(int i = 0; i < buffer_res.width; ++i) {
    for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
      __fluidCells[i + j * buffer_res.width] = 1.0f;
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
  __sim_kernels.fluidCells() = __fluidCells;
  __sim_kernels.velocity() = __velocity;
  __sim_kernels.smoke() = __smoke;
}

void Simulation::__render(Resolution const & _window_res, float _mag, float2 _off) {
  __quad.copyToSurface(__f4temp);
  __quad.render(__kernels.resolution(), _window_res, _mag, _off);
}

// BFECCSimulation::BFECCSimulation(Interface & _interface, KernelsWrapper & _kers, int _pressure_steps)
//   : Simulation(_interface, _kers, _pressure_steps)
// {
//   OutputIndent indent1;
//   {
//     Allocator alloc;
//     __f2temp.resize(alloc, _kers.getBufferRes().size);
//   }
// }
//
// void BFECCSimulation::advectVelocity(float2 _rd, float _dt) {
//   // Backwards-Forwards Error Compensation & Correction (BFECC). Only stable for low V
//   __kernels.advectVelocity(__f2temp, __velocity.device(), _rd, _dt);
//   __kernels.advectVelocity(__f2temp, _rd, -_dt);
//   __kernels.sum(__velocity.device(), 1.5f, __velocity.device(), -.5f, __f2temp);
//   __kernels.advectVelocity(__velocity.device(), _rd, _dt);
// }
//
// LBFECCSimulation::LBFECCSimulation(Interface & _interface, KernelsWrapper & _kers, int _pressure_steps)
//   : Simulation(_interface, _kers, _pressure_steps)
// {
//   OutputIndent indent1;
//   {
//     Allocator alloc;
//     __f2tempA.resize(alloc, _kers.getBufferRes().size);
//     __f2tempB.resize(alloc, _kers.getBufferRes().size);
//     __f2tempC.resize(alloc, _kers.getBufferRes().size);
//   }
// }
//
// void LBFECCSimulation::advectVelocity(float2 _rd, float _dt) {
//   // Backwards-Forwards Error Compensation & Correction (BFECC) with limiting.
//   __kernels.advectVelocity(__f2tempA, __velocity.device(), _rd, _dt);
//   __kernels.advectVelocity(__f2tempA, _rd, -_dt);
//   __kernels.sum(__f2tempB, .5f, __velocity.device(), -.5f, __f2tempA);
//   __kernels.sum(__f2tempA, 1.0f, __velocity.device(), 1.0f, __f2tempB);
//   __kernels.advectVelocity(__f2tempA, _rd, _dt);
//   __kernels.advectVelocity(__f2tempA, _rd, -_dt);
//   __kernels.sum(__f2tempA, 1.0f, __f2tempA, 1.0f, __f2tempB);
//   __kernels.sum(__f2tempA, 1.0f, __velocity.device(), -1.0f, __f2tempA);
//   __kernels.limitAdvection(__f2tempC, __f2tempB, __f2tempA);
//   __kernels.sum(__velocity.device(), 1.0f, __velocity.device(), 1.0f, __f2tempC);
//   __kernels.advectVelocity(__velocity.device(), _rd, _dt);
// }
