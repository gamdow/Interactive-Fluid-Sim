#include "simulation.h"

#include <iostream>
#include <cuda_gl_interop.h>

#include "debug.h"
#include "cuda/helper_math.h"
#include "cuda/helper_cuda.h"
#include "interface.h"
#include "camera.h"

Simulation::Simulation(Interface const & _interface, OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx, int _pressure_steps)
  : __simulation(_block_config, _buffer_width, _dx)
  , __visualisation(_block_config, _buffer_width)
  , __interface(_interface)
  , PRESSURE_SOLVER_STEPS(_pressure_steps)
  , __min_rgba(make_float4(0.0f))
  , __max_rgba(make_float4(1.0f))
{
  format_out << "Constructing Simluation Buffers:" << std::endl;
  OutputIndent indent1;
  __simulation.buffer_resolution().print("Resolution");
  {
    Allocator alloc;
    __fluidCells.resize(alloc, __simulation.buffer_resolution().size);
    __velocity.resize(alloc, __simulation.buffer_resolution().size);
    __smoke.resize(alloc, __simulation.buffer_resolution().size);
    __color_map.resize(alloc, 4);
  }

  reset();

  __color_map[0] = make_float3(1.0f, 0.35f, 0.35f);// * 0.5f; // red
  __color_map[1] = make_float3(0.85f, 0.30f, 0.63f);// * 0.5f;
  __color_map[2] = make_float3(0.3f, 0.85f, 0.3f);// * 0.5f;
  __color_map[3] = make_float3(0.77f, 0.96f, 0.34f);// * 0.5f;
  __color_map.copyHostToDevice();
}

void Simulation::step(float _dt) {
  switch(__interface.mode()) {
    case Mode::smoke: __visualisation.visualise(__simulation.smoke(), __color_map.device()); break;
    case Mode::velocity: __visualisation.visualise(__simulation.velocity()); break;
    case Mode::divergence: __visualisation.visualise(__simulation.divergence()); break;
    case Mode::pressure: __visualisation.visualise(__simulation.pressure()); break;
    case Mode::fluid: __visualisation.visualise(__simulation.fluidCells()); break;
  }
  __visualisation.adjustBrightness(__min_rgba, __max_rgba);
  __simulation.advectVelocity(_dt);
  __simulation.calcDivergence();
  __simulation.pressureDecay();
  for(int i = 0; i < PRESSURE_SOLVER_STEPS; i++) {
    __simulation.pressureSolveStep();
  }
  __simulation.subGradient();
  __simulation.enforceSlip();
  __simulation.advectSmoke(_dt);
}

void Simulation::applyBoundary() {
  __velocity = __simulation.velocity();
  float velocity = __interface.velocity();
  Resolution const & buffer_res = __simulation.buffer_resolution();
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
  __simulation.velocity() = __velocity;
}

void Simulation::applySmoke() {
  __smoke = __simulation.smoke();
  Resolution const & buffer_res = __simulation.buffer_resolution();
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
  __simulation.smoke() = __smoke;
}

void Simulation::reset() {
  __velocity.reset();
  __fluidCells.reset();
  __smoke.reset();

  Resolution const & buffer_res = __simulation.buffer_resolution();
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
  __simulation.fluidCells() = __fluidCells;
  __simulation.velocity() = __velocity;
  __simulation.smoke() = __smoke;
}

void Simulation::updateFluidCells(DeviceArray<float> const & _fluid_cells) {
  __simulation.fluidCells() = _fluid_cells;
}
