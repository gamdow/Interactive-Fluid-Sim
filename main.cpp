#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "configuration.cuh"
#include "helper_math.h"
#include "interface.hpp"
#include "kernels.cuh"
#include "memory.hpp"
#include "render.hpp"
#include "simulation.hpp"

void quit(int _code, char const * _message);

enum Mode : int {
  velocity = 0,
  divergence,
  pressure
};

int main(int argc, char * argv[]) {
  Kernels kernels(RESOLUTION, BUFFER, BLOCK_SIZE);
  Renderer renderer(kernels);
  Simulation sim(kernels);

  float init_velocity = 1.0f;
  sim.applyBoundary(init_velocity);

  Uint32 time = SDL_GetTicks();
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

    float2 offset = make_float2(offset_x, offset_y) * (magnification - 1.0f);
    renderer.render(magnification, offset);

    float2 const dx = make_float2(LENGTH.x / RESOLUTION.x, LENGTH.y / RESOLUTION.y);
    sim.step(dx, 1.0f / FRAME_RATE);

    int elasped = SDL_GetTicks() - time;
    SDL_Delay(std::max(1000 / FRAME_RATE - elasped, 0.0f));
    time = SDL_GetTicks();
  }


  quit(0, "");
}

void quit(int _code, char const * _message) {
  SDL_Quit();
  std::cout << _message;
  exit(_code);
}
