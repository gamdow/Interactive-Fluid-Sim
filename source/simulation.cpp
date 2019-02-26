#include "simulation.h"

#include <iostream>
#include <cuda_gl_interop.h>

#include "debug.h"
#include "cuda/helper_math.h"
#include "cuda/helper_cuda.h"
#include "interface.h"
#include "camera.h"

Simulation::Simulation(OptimalBlockConfig const & _block_config, float2 _dx, int _pressure_steps)
  : __simulation(_block_config, _dx)
  , __visualisation(_block_config)
  , __pressure_solver_steps(_pressure_steps)
  , __min_rgba(make_float4(0.0f))
  , __max_rgba(make_float4(1.0f))
  , __last_mode(-1)
{
  format_out << "Constructing Simluation Buffers:" << std::endl;
  OutputIndent indent1;
  __simulation.resolution().print("Resolution");
  {
    Allocator alloc;
    __fluidCells.resize(alloc, __simulation.resolution().total_size());
    __velocity.resize(alloc, __simulation.resolution().total_size());
    __smoke.resize(alloc, __simulation.resolution().total_size());
    __color_map.resize(alloc, 4);
  }

  reset();

  // ocean colours
  __color_map[0] = make_float3(0.09f, 0.51f, 0.51f);
  __color_map[1] = make_float3(0.141f, 0.298f, 0.565f);
  __color_map[2] = make_float3(0.09f, 0.51f, 0.51f);
  __color_map[3] = make_float3(0.114f, 0.643f, 0.231f);

  // candy colours
  // __color_map[0] = make_float3(1.0f, 0.35f, 0.35f);// * 0.5f; // red
  // __color_map[1] = make_float3(0.85f, 0.30f, 0.63f);// * 0.5f;
  // __color_map[2] = make_float3(0.3f, 0.85f, 0.3f);// * 0.5f;
  // __color_map[3] = make_float3(0.77f, 0.96f, 0.34f);// * 0.5f;
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

  __visualisation.adjustBrightness(__min_rgba, __max_rgba, __last_mode != _mode);
  __simulation.advectVelocity(_dt);
  __simulation.calcDivergence();
  __simulation.pressureDecay();
  for(int i = 0; i < __pressure_solver_steps; i++) {
    __simulation.pressureSolveStep();
  }
  __simulation.subGradient();
  __simulation.enforceSlip();
  __simulation.advectSmoke(_dt);

  __last_mode = _mode;
}

void Simulation::applyBoundary(float _velocity_setting, int _flow_rotation) {
  __simulation.injectVelocityAndApplyBoundary(_flow_rotation, _velocity_setting);
}

void Simulation::applySmoke(int _flow_rotation) {
  __simulation.applySmoke(_flow_rotation);
}

void Simulation::reset() {
  __velocity.reset();
  __fluidCells.reset();
  __smoke.reset();

  Resolution const & buffer_res = __simulation.resolution();
  for(int i = 0; i < buffer_res.width; ++i) {
    for(int j = buffer_res.height.begin_inner(); j < buffer_res.height.end_inner(); ++j) {
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
