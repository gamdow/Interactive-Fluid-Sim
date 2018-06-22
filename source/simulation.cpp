#include "simulation.hpp"

#include <iostream>
#include <cuda_gl_interop.h>
//#include <typeinfo>

#include "debug.hpp"
#include "cuda/helper_math.h"
#include "cuda/helper_cuda.h"
#include "interface.hpp"
#include "camera.hpp"

Simulation::Simulation(Interface & _interface, OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx, int _pressure_steps)
  : __interface(_interface)
  , __sim_kernels(_block_config, _buffer_width, _dx)
  , __vis_kernels(_block_config, _buffer_width)
  , PRESSURE_SOLVER_STEPS(_pressure_steps)
  , __min_rgb(0.0f)
  , __max_rgb(1.0f)
  , __quad(__vis_kernels.buffer_resolution(), GL_RGBA32F, GL_RGBA, GL_FLOAT)
{
  format_out << "Constructing Simluation Device Buffers:" << std::endl;
  OutputIndent indent1;
  __sim_kernels.buffer_resolution().print("Resolution");
  {
    Allocator alloc;
    __fluidCells.resize(alloc, __sim_kernels.buffer_resolution().size);
    __velocity.resize(alloc, __sim_kernels.buffer_resolution().size);
    __smoke.resize(alloc, __sim_kernels.buffer_resolution().size);
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
    case Mode::smoke: __vis_kernels.copyToRGBA(__sim_kernels.smoke(), __color_map.device()); break;
    case Mode::velocity: __vis_kernels.copyToRGBA(__sim_kernels.velocity()); break;
    case Mode::divergence: __vis_kernels.copyToRGBA(__sim_kernels.divergence()); break;
    case Mode::pressure: __vis_kernels.copyToRGBA(__sim_kernels.pressure()); break;
    case Mode::fluid: __vis_kernels.copyToRGBA(__sim_kernels.fluidCells()); break;
  }
  __vis_kernels.adjustBrightness(__min_rgb, __max_rgb);

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
  Resolution const & buffer_res = __sim_kernels.buffer_resolution();
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
  Resolution const & buffer_res = __sim_kernels.buffer_resolution();
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

  Resolution const & buffer_res = __sim_kernels.buffer_resolution();
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

void Simulation::copyToFluidCells(Camera const & _cam) {
  __sim_kernels.copyToFluidCells(_cam.deviceArray(), _cam.resolution());
}

void Simulation::__render(Resolution const & _window_res, float _mag, float2 _off) {
  __quad.copyToSurface(__vis_kernels, __vis_kernels.rgba());
  __quad.render(__vis_kernels.buffer_resolution(), _window_res, _mag, _off);
}
