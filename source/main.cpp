// #include <cuda_gl_interop.h>
#include <cuda_runtime.h>
// #include <SDL2/SDL_opengl.h>
// #include <SDL2/SDL.h>
// #include <SDL2/SDL_ttf.h>

// #include <algorithm>
// #include <iostream>
// #include <vector>
// #include <iomanip>
// #include <cmath>

// #include "helper_math.h"
#include "capability.cuh"
#include "kernels_wrapper.cuh"
#include "camera.hpp"
#include "render.hpp"
#include "simulation.hpp"
#include "memory.hpp"
#include "interface.hpp"


void quit(int _code, char const * _message);

int const VIDCAM_INDEX = 0;
int2 const RESOLUTION = make_int2(1280, 720);
int const BUFFER = 10u;
float2 const LENGTH = {1.6f, 0.9f};
float const FRAME_RATE = 60.0f;
int const SIM_STEPS_PER_FRAME = 5;

enum Mode : int {
  smoke = 0,
  velocity,
  divergence,
  pressure,
  fluid
};

int main(int argc, char * argv[]) {

  Capability capability(RESOLUTION, BUFFER);
  Camera camera(capability, VIDCAM_INDEX);
  Kernels kernels(capability);
  Simulation sim(kernels);
  Renderer renderer(kernels);

  MirroredArray<uchar3> camera_frame(kernels.getBufferSpec().size);

  std::vector<OptionBase*> options;
  RangeOption<float> vel_multiplier("Velocity Multiplier", 1.0f, 0.1f, 10.0f, 101, SDLK_r, SDLK_f);
  options.push_back(&vel_multiplier);
  RangeOption<float> vis_multiplier("Visualisation Multiplier", 1.0f, 0.1f, 10.0f, 101, SDLK_e, SDLK_d);
  options.push_back(&vis_multiplier);
  RangeOption<float> magnification("Magnification", 1.0f, 1.0f, 4.0f, 101, SDLK_w, SDLK_s);
  options.push_back(&magnification);
  RangeOption<float> offset_x("Offset X-Axis", 0.0f, -1.0f, 1.0f, 100, SDLK_LEFT, SDLK_RIGHT);
  options.push_back(&offset_x);
  RangeOption<float> offset_y("Offset Y-Axis", 0.0f, -1.0f, 1.0f, 100, SDLK_DOWN, SDLK_UP);
  options.push_back(&offset_y);
  CycleOption<int> mode("Visualisation Mode", SDLK_1);
  mode.insert("Smoke", Mode::smoke);
  mode.insert("Velocity Field", Mode::velocity);
  mode.insert("Divergence", Mode::divergence);
  mode.insert("Pressure", Mode::pressure);
  mode.insert("Fluid", Mode::fluid);
  options.push_back(&mode);

  MirroredArray<float3> color_map(4);
  color_map[0] = make_float3(1.0f, 0.5f, 0.5f) * 0.5f; // red
  color_map[1] = make_float3(0.8f, 0.2f, 1.0f) * 0.5f;
  color_map[2] = make_float3(0.3f, 1.0f, 0.6f) * 0.5f;
  color_map[3] = make_float3(0.5f, 0.5f, 0.5f) * 0.5f;
  color_map.copyHostToDevice();

  FPS fps(FRAME_RATE);
  bool stop = false;
  SDL_Event event;
  while(true) {

    for(auto i = options.begin(); i != options.end(); ++i) {
      (*i)->clearChangedFlag();
    }

    while(SDL_PollEvent(&event)) {
      for(auto i = options.begin(); i != options.end(); ++i) {
        (*i)->Update(event);
      }
      switch(event.type) {
        case SDL_KEYDOWN:
          switch(event.key.keysym.sym) {
            case SDLK_a:
              sim.reset();
              break;
            case SDLK_q:
              stop = true;
              break;
            defaut:
              break;
          }
          break;
        case SDL_QUIT:
          stop = true;
        default:
          break;
      }
    }

    if(stop) break;

    switch(mode) {
      case Mode::smoke: renderer.copyToSurface(sim.__smoke.device, color_map.device); break;
      case Mode::velocity: renderer.copyToSurface(sim.__velocity.device, 0.25f * vis_multiplier); break;
      case Mode::divergence: renderer.copyToSurface(sim.__divergence.device, 0.1f * vis_multiplier); break;
      case Mode::pressure: renderer.copyToSurface(sim.__pressure.device, 50.0f * vis_multiplier); break;
      case Mode::fluid: renderer.copyToSurface(sim.__fluidCells.device, vis_multiplier); break;
    }

    std::stringstream os_text;
    os_text.setf(std::ios::fixed, std:: ios::floatfield);
    os_text.precision(2);
    fps.printCurrent(os_text);
    os_text << std::endl;
    mode.printCurrent(os_text);
    renderer.setText(os_text.str().c_str());

    float2 offset = make_float2(offset_x, offset_y) * (magnification - 1.0f);
    renderer.render(magnification, offset);

    sim.applyBoundary(vel_multiplier);
    sim.applySmoke();
    for(int i = 0; i < SIM_STEPS_PER_FRAME; ++i) {
      float2 const dx = make_float2(LENGTH.x / capability.adjusted_dims.x, LENGTH.y / capability.adjusted_dims.y);
      sim.step(dx, 1.0f / (SIM_STEPS_PER_FRAME * FRAME_RATE));
    }

    fps.update();
  }

  quit(0, "");
}

void quit(int _code, char const * _message) {
  std::cout << _message << std::endl;
  exit(_code);
}
