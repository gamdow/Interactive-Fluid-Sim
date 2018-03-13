#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "helper_math.h"
#include "kernels.cuh"
#include "simulation.cuh"
#include "memory.hpp"
#include "interface.hpp"
#include "render.hpp"
#include "configuration.cuh"

void quit(int _code, char const * _message);
void advanceSim(cudaGraphicsResource_t & _viewCudaResource);
void applyBoundary(int2 _dims, float _vel, MirroredArray<float2> & io_velocity, MirroredArray<float> & io_fluidCells);

enum Mode : int {
  velocity = 0,
  divergence,
  pressure
};

int main(int argc, char * argv[]) {
  Renderer renderer(RESOLUTION);

  int2 const BUFFERED_DIMS = {RESOLUTION.x + 2, RESOLUTION.y + 2};
  int const BUFFERED_SIZE = BUFFERED_DIMS.x * BUFFERED_DIMS.y;

  MirroredArray<float2> velocity(BUFFERED_SIZE);
  MirroredArray<float> fluidCells(BUFFERED_SIZE);

  float * divergence = NULL; cudaMalloc((void **) & divergence, BUFFERED_SIZE * sizeof(float)); cudaMemset(divergence, 0, BUFFERED_SIZE * sizeof(float));
  float * pressure = NULL; cudaMalloc((void **) & pressure, BUFFERED_SIZE * sizeof(float)); cudaMemset(pressure, 0, BUFFERED_SIZE * sizeof(float));
  float * buffer = NULL; cudaMalloc((void **) & buffer, BUFFERED_SIZE * sizeof(float)); cudaMemset(buffer, 0, BUFFERED_SIZE * sizeof(float));

  for(int i = 0; i < BUFFERED_DIMS.x; ++i) {
    for(int j = 0; j < BUFFERED_DIMS.y; ++j) {
      fluidCells[i + j * BUFFERED_DIMS.x] = 1.0f;
    }
  }

  // for(int k = 0; k < 5; ++k) {
  //   for(int l = 0; l < 3; ++l) {
  //     float2 center = make_float2(k * 80 + 100 + 50 * ((l + 1) % 2), l * 70 + 120);
  //     for(int i = 0; i < BUFFERED_DIMS.x; ++i) {
  //       for(int j = 0; j < BUFFERED_DIMS.y; ++j) {
  //         if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000) {
  //           fluidCells[i + j * BUFFERED_DIMS.x] = 0.0f;
  //         }
  //       }
  //     }
  //   }
  // }

  for(int k = 0; k < 5; ++k) {
    int odd = k % 2;
    for(int l = 0; l < 3 + odd; ++l) {
      float2 center = make_float2(k * 80 + 200, l * 80 + 100 - 40 * odd);
      for(int i = 0; i < BUFFERED_DIMS.x; ++i) {
        for(int j = 0; j < BUFFERED_DIMS.y; ++j) {
          if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000) {
            fluidCells[i + j * BUFFERED_DIMS.x] = 0.0f;
          }
        }
      }
    }
  }

  float init_velocity = 1.0f;
  applyBoundary(BUFFERED_DIMS, init_velocity, velocity, fluidCells);

  kernels_init(RESOLUTION);
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

    float2 offset = make_float2(offset_x, offset_y) * (magnification - 1.0f);
    switch(mode) {
      case Mode::velocity: renderer.render(velocity.device, 0.25f * vis_multiplier, magnification, offset); break;
      case Mode::divergence: renderer.render(divergence, 0.1f * vis_multiplier, magnification, offset); break;
      case Mode::pressure: renderer.render(pressure, 50.0f * vis_multiplier, magnification, offset); break;
    }

    float2 const dx = make_float2(LENGTH.x / RESOLUTION.x, LENGTH.y / RESOLUTION.y);

    int2 rest = {RESOLUTION.x, RESOLUTION.y};
    simulation_step(rest, dx, 1.0f / FRAME_RATE, fluidCells.device, velocity.device, divergence, pressure, buffer);

    int elasped = SDL_GetTicks() - time;
    SDL_Delay(std::max(1000 / FRAME_RATE - elasped, 0.0f));
    time = SDL_GetTicks();
  }
  kernels_shutdown();

  cudaFree(pressure);
  cudaFree(buffer);
  cudaFree(divergence);

  quit(0, "");
}

void quit(int _code, char const * _message) {
  SDL_Quit();
  std::cout << _message;
  exit(_code);
}

void applyBoundary(int2 _dims, float _vel, MirroredArray<float2> & io_velocity, MirroredArray<float> & io_fluidCells) {

  for(int j = 0; j < _dims.y; ++j) {
    for(int i = 0; i < 50; ++i) {
      io_velocity[i + j * _dims.x] = make_float2(_vel, 0.0f);
    }
    for(int i = _dims.x - 50; i < _dims.x; ++i) {
      io_velocity[i + j * _dims.x] = make_float2(_vel, 0.0f);
    }
  }

  for(int i = 0; i < _dims.x; ++i) {
    for(int j = 0; j < 1; ++j) {
      io_velocity[i + j * _dims.x] = make_float2(0.0f, 0.0f);
      io_fluidCells[i + j * _dims.x] = 0.0f;
    }
    for(int j = _dims.y - 1; j < _dims.y; ++j) {
      io_velocity[i + j * _dims.x] = make_float2(0.0f, 0.0f);
      io_fluidCells[i + j * _dims.x] = 0.0f;
    }
  }

  io_velocity.copyHostToDevice();
  io_fluidCells.copyHostToDevice();
}
