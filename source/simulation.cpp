#include "simulation.h"

#include <iostream>
#include <cuda_gl_interop.h>

#include "debug.h"
#include "cuda/helper_math.h"
#include "cuda/helper_cuda.h"
#include "interface.h"
#include "camera.h"

Simulation::Simulation(OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx, int _pressure_steps)
  : __simulation(_block_config, _buffer_width, _dx)
  , __visualisation(_block_config, _buffer_width)
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

void Simulation::step(int _mode, float _dt) {
  switch(_mode) {
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

void Simulation::applyBoundary(float _velocity_setting, int _flow_rotation) {
  __velocity = __simulation.velocity();
  Resolution const & buffer_res = __simulation.buffer_resolution();
  switch (_flow_rotation) {
    case FlowDirection::LEFT_TO_RIGHT:
    case FlowDirection::RIGHT_TO_LEFT:
      for(int i = 0; i < buffer_res.width; ++i) {
        for(int j = 0; j < buffer_res.buffer; ++j) {
          __fluidCells[i + j * buffer_res.width] = 0.0f;
          __velocity[i + j * buffer_res.width] = make_float2(0.f, 0.f);
        }
        for(int j = buffer_res.height - buffer_res.buffer; j < buffer_res.height; ++j) {
          __fluidCells[i + j * buffer_res.width] = 0.0f;
          __velocity[i + j * buffer_res.width] = make_float2(0.f, 0.f);
        }
      }
      break;
    case FlowDirection::TOP_TO_BOTTOM:
    case FlowDirection::BOTTOM_TO_TOP:
      for(int j = 0; j < buffer_res.height; ++j) {
        for(int i = 0; i < buffer_res.buffer; ++i) {
          __fluidCells[i + j * buffer_res.width] = 0.0f;
          __velocity[i + j * buffer_res.width] = make_float2(0.f, 0.f);
        }
        for(int i = buffer_res.width - buffer_res.buffer; i < buffer_res.width; ++i) {
          __fluidCells[i + j * buffer_res.width] = 0.0f;
          __velocity[i + j * buffer_res.width] = make_float2(0.f, 0.f);
        }
      }
      break;
  }
  switch (_flow_rotation) {
    case FlowDirection::LEFT_TO_RIGHT:
    case FlowDirection::RIGHT_TO_LEFT:
    {
      float vel = _flow_rotation == FlowDirection::LEFT_TO_RIGHT ? _velocity_setting : -_velocity_setting;
      for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
        for(int i = 0; i < buffer_res.buffer * 2; ++i) {
          __velocity[i + j * buffer_res.width] = make_float2(vel, 0.f);
        }
        for(int i = buffer_res.width - buffer_res.buffer * 2; i < buffer_res.width; ++i) {
          __velocity[i + j * buffer_res.width] = make_float2(vel, 0.f);
        }
      }
    } break;
    case FlowDirection::TOP_TO_BOTTOM:
    case FlowDirection::BOTTOM_TO_TOP:
    {
      float vel = _flow_rotation == FlowDirection::TOP_TO_BOTTOM ? _velocity_setting : -_velocity_setting;
      for(int i = buffer_res.buffer; i < buffer_res.width - buffer_res.buffer; ++i) {
        for(int j = 0; j < buffer_res.buffer * 2; ++j) {
          __velocity[i + j * buffer_res.width] = make_float2(0.f, vel);
        }
        for(int j = buffer_res.height - buffer_res.buffer * 2; j < buffer_res.height; ++j) {
          __velocity[i + j * buffer_res.width] = make_float2(0.f, vel);
        }
      }
    } break;
  }
  __simulation.velocity() = __velocity;
}

void Simulation::applySmoke(int _flow_rotation) {
  __smoke = __simulation.smoke();
  Resolution const & buffer_res = __simulation.buffer_resolution();
  int width = 20;
  switch (_flow_rotation) {
    case FlowDirection::LEFT_TO_RIGHT:
      for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
        for(int i = 0; i < buffer_res.buffer * 2; ++i) {
          int z = (j / width) % 4;
          __smoke[i + j * buffer_res.width] = make_float4(
            z == 0 ? 1.0f : 0.f,
            z == 1 ? 1.0f : 0.f,
            z == 2 ? 1.0f : 0.f,
            z == 3 ? 1.0f : 0.f
          ) * powf(cosf((j - buffer_res.buffer) * 3.14159f * (1.0f / width)), 2) * 1.5f;
        }
      }
      break;
    case FlowDirection::RIGHT_TO_LEFT:
      for(int j = buffer_res.buffer; j < buffer_res.height - buffer_res.buffer; ++j) {
        for(int i = buffer_res.width - buffer_res.buffer * 2; i < buffer_res.width; ++i) {
          int z = (j / width) % 4;
          __smoke[i + j * buffer_res.width] = make_float4(
            z == 0 ? 1.0f : 0.f,
            z == 1 ? 1.0f : 0.f,
            z == 2 ? 1.0f : 0.f,
            z == 3 ? 1.0f : 0.f
          ) * powf(cosf((j - buffer_res.buffer) * 3.14159f * (1.0f / width)), 2) * 1.5f;
        }
      }
      break;
    case FlowDirection::TOP_TO_BOTTOM:
      for(int i = buffer_res.buffer; i < buffer_res.width - buffer_res.buffer; ++i) {
        for(int j = 0; j < buffer_res.buffer * 2; ++j) {
          int z = (i / width) % 4;
          __smoke[i + j * buffer_res.width] = make_float4(
            z == 0 ? 1.0f : 0.f,
            z == 1 ? 1.0f : 0.f,
            z == 2 ? 1.0f : 0.f,
            z == 3 ? 1.0f : 0.f
          ) * powf(cosf((i - buffer_res.buffer) * 3.14159f * (1.0f / width)), 2) * 1.5f;
        }
      }
      break;
    case FlowDirection::BOTTOM_TO_TOP:
      for(int i = buffer_res.buffer; i < buffer_res.width - buffer_res.buffer; ++i) {
        for(int j = buffer_res.height - buffer_res.buffer * 2; j < buffer_res.height; ++j) {
          int z = (i / width) % 4;
          __smoke[i + j * buffer_res.width] = make_float4(
            z == 0 ? 1.0f : 0.f,
            z == 1 ? 1.0f : 0.f,
            z == 2 ? 1.0f : 0.f,
            z == 3 ? 1.0f : 0.f
          ) * powf(cosf((i - buffer_res.buffer) * 3.14159f * (1.0f / width)), 2) * 1.5f;
        }
      }
      break;
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

  __simulation.fluidCells() = __fluidCells;
  __simulation.velocity() = __velocity;
  __simulation.smoke() = __smoke;
}

void Simulation::updateFluidCells(DeviceArray<float> const & _fluid_cells) {
  __simulation.fluidCells() = _fluid_cells;
}
