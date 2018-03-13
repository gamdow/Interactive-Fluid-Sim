#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <iomanip>

#include "helper_math.h"
#include "interface.hpp"
#include "kernels.cuh"
#include "memory.hpp"
#include "render.hpp"
#include "simulation.hpp"

void quit(int _code, char const * _message);

static int2 const RESOLUTION = make_int2(1280, 720);
static int const BUFFER = 1u;
float2 const LENGTH = {1.6f, 0.9f};
float const FRAME_RATE = 60.0f;

enum Mode : int {
  velocity = 0,
  divergence,
  pressure
};

int main(int argc, char * argv[]) {
  Kernels kernels(RESOLUTION, BUFFER);
  Simulation sim(kernels);
  Renderer renderer(kernels);

  float init_velocity = 1.0f;
  sim.applyBoundary(init_velocity);

  SDL_Event event;

  std::vector<OptionBase*> options;

  RangeOption<float> vis_multiplier("Visualisation Multiplier", 1.0f, 0.1f, 10.0f, 101, SDLK_e, SDLK_d);
  options.push_back(&vis_multiplier);

  RangeOption<float> magnification("Magnification", 1.0f, 1.0f, 4.0f, 101, SDLK_q, SDLK_a);
  options.push_back(&magnification);

  RangeOption<float> offset_x("Offset X-Axis", 0.0f, -1.0f, 1.0f, 100, SDLK_LEFT, SDLK_RIGHT);
  options.push_back(&offset_x);
  RangeOption<float> offset_y("Offset Y-Axis", 0.0f, -1.0f, 1.0f, 100, SDLK_UP, SDLK_DOWN);
  options.push_back(&offset_y);

  CycleOption<int> mode("Visualisation Mode", SDLK_1);
  mode.insert("Velocity Field", Mode::velocity);
  mode.insert("Divergence", Mode::divergence);
  mode.insert("Pressure", Mode::pressure);
  options.push_back(&mode);

  float velocity_multiplier = 1.0f;
  float fps_max = 0.0f;
  float fps_act = 0.0f;
  Uint32 time = SDL_GetTicks();
  bool stop = false;
  while(true) {
    while(SDL_PollEvent(&event)) {
      for(auto i = options.begin(); i != options.end(); ++i) {
        (*i)->Update(event);
      }
      switch(event.type) {
        case SDL_QUIT:
          stop = true;
        default:
          break;
      }
    }

    if(stop) break;

    switch(mode) {
      case Mode::velocity: renderer.copyToSurface(sim.__velocity.device, 0.25f * vis_multiplier); break;
      case Mode::divergence: renderer.copyToSurface(sim.__divergence, 0.1f * vis_multiplier); break;
      case Mode::pressure: renderer.copyToSurface(sim.__pressure, 50.0f * vis_multiplier); break;
    }

    std::stringstream os_text;
    os_text.setf(std::ios::fixed, std:: ios::floatfield);
    os_text.precision(2);

    os_text << "fps: " << floorf(fps_act * 10.f) / 10.f << " (" << floorf(fps_max * 10.f) / 10.f << ")";

    renderer.setText(os_text.str().c_str());
    float2 offset = make_float2(offset_x, offset_y) * (magnification - 1.0f);
    renderer.render(magnification, offset);


    float2 const dx = make_float2(LENGTH.x / kernels.__dims.x, LENGTH.y / kernels.__dims.y);
    sim.step(dx, 1.0f / FRAME_RATE);

    fps_max = 0.99f * fps_max + 0.01f * (1000.f / (SDL_GetTicks() - time));
    SDL_Delay(std::max(1000.0f / FRAME_RATE - (SDL_GetTicks() - time), 0.0f));
    fps_act = 0.99f * fps_act + 0.01f * (1000.f / (SDL_GetTicks() - time));
    time = SDL_GetTicks();
  }


  quit(0, "");
}

void quit(int _code, char const * _message) {
  SDL_Quit();
  std::cout << _message << std::endl;
  exit(_code);
}
