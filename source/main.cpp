#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>

#include "data/resolution.cuh"
#include "data/managed_array.cuh"
#include "kernels/kernels_wrapper.cuh"
#include "opengl.hpp"
#include "camera.hpp"
#include "renderer.hpp"
#include "simulation.hpp"
#include "interface.hpp"

void quit(int _code, char const * _message);

int const VIDCAM_INDEX = 0;
Resolution const RESOLUTION = Resolution(640, 360);
int const BUFFER = 10u;
float2 const LENGTH = {1.6f, 0.9f};
float const FRAME_RATE = 60.0f;
int const SIM_STEPS_PER_FRAME = 5;
int const PRESSURE_SOLVER_STEPS = 200;

int main(int argc, char * argv[]) {
  OpenGL opengl(RESOLUTION);
  Interface interface(opengl, FRAME_RATE);
  reportCudaCapability();
  OptimalBlockConfig blockConfig(RESOLUTION);
  float2 const DX = make_float2(LENGTH.x / blockConfig.optimal_res.width, LENGTH.y / blockConfig.optimal_res.height);
  KernelsWrapper kernels(blockConfig, BUFFER);
  Simulation sim(interface, blockConfig, BUFFER, DX, kernels, PRESSURE_SOLVER_STEPS);
  Camera * camera = nullptr;
  try {
    camera = new CVCamera(VIDCAM_INDEX, RESOLUTION, FRAME_RATE);
  } catch (std::exception const & e) {
    std::cout << e.what() << std::endl << std::endl;
    delete camera;
    camera = new NullCamera(RESOLUTION);
  }
  Renderer renderer(opengl, *camera, sim, interface);
  std::cout << "dx = " << DX.x << "," << DX.y << " Max Velocity = " << LENGTH.y / 2.0f * SIM_STEPS_PER_FRAME / FRAME_RATE;

  bool stop = false;
  SDL_Event event;
  while(true) {
    interface.resetFlags();

    while(SDL_PollEvent(&event)) {
      interface.updateInputs(event);
      switch(event.type) {
        case SDL_KEYDOWN:
          switch(event.key.keysym.sym) {
            case SDLK_a: sim.reset(); break;
            case SDLK_q: stop = true; break;
            defaut: break;
          }
          break;
        case SDL_QUIT: stop = true; // fall-through
        default: break;
      }
    }

    if(stop) break;

    renderer.render();

    camera->updateDeviceArray();
    kernels.copyToArray(sim.devicefluidCells(), camera->deviceArray(), const_cast<Camera const *>(camera)->resolution());

    sim.applyBoundary();
    sim.applySmoke();
    for(int i = 0; i < SIM_STEPS_PER_FRAME; ++i) {
      float2 const dx = make_float2(LENGTH.x / RESOLUTION.width, LENGTH.y / RESOLUTION.height);
      sim.step(DX, 1.0f / (SIM_STEPS_PER_FRAME * FRAME_RATE));
    }

    interface.updateAndDelay();
  }

  delete camera;
  quit(0, "");
}

void quit(int _code, char const * _message) {
  std::cout << _message << std::endl;
  exit(_code);
}
