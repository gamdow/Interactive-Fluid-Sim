#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>

#include "data/resolution.h"
#include "data/managed_array.h"
#include "opengl.h"
#include "camera.h"
#include "kernels/camera_filter.h"
#include "simulation.h"
#include "interface.h"
#include "renderer.h"

void quit(int _code, char const * _message);

int const VIDCAM_INDEX = 0;
Resolution const RESOLUTION = Resolution(640, 360);
int const BUFFER = 10u;
float2 const LENGTH = {1.6f, 0.9f};
float const FRAME_RATE = 30.0f;
int const SIM_STEPS_PER_FRAME = 5;
int const PRESSURE_SOLVER_STEPS = 200;
float TIME_DELTA = 1.0f / (SIM_STEPS_PER_FRAME * FRAME_RATE);

int main(int argc, char * argv[]) {
  OpenGL opengl(RESOLUTION);
  Interface interface(FRAME_RATE);
  Renderer renderer(interface, opengl);
  InterfaceRenderer interface_render(interface, renderer);

  reportCudaCapability();
  OptimalBlockConfig blockConfig(RESOLUTION);
  float2 const DX = make_float2(LENGTH.x / blockConfig.optimal_res.width, LENGTH.y / blockConfig.optimal_res.height);

  Simulation simulation(interface, renderer, blockConfig, BUFFER, DX, PRESSURE_SOLVER_STEPS);

  Camera * camera = nullptr;
  try {
    camera = new CVCamera(renderer, VIDCAM_INDEX, RESOLUTION, FRAME_RATE);
  } catch (std::exception const & e) {
    std::cout << e.what() << std::endl << std::endl;
    delete camera;
    camera = new NullCamera(renderer, RESOLUTION);
  }
  CameraFilter camera_filter(blockConfig, BUFFER);

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
            case SDLK_a: simulation.reset(); break;
            case SDLK_q: stop = true; break;
            defaut: break;
          }
          break;
        case SDL_QUIT: stop = true; // fall-through
        default: break;
      }
    }

    if(stop) break;

    camera->render();
    simulation.render();
    interface_render.render();
    renderer.swapBuffers();

    camera_filter.update(camera->frameData(), interface.filterThreshold());

    simulation.updateFluidCells(camera_filter.output());
    simulation.applyBoundary();
    simulation.applySmoke();
    for(int i = 0; i < SIM_STEPS_PER_FRAME; ++i) {
      simulation.step(TIME_DELTA);
    }

    interface.updateAndLimitFps();
  }

  delete camera;
  quit(0, "");
}

void quit(int _code, char const * _message) {
  std::cout << _message << std::endl;
  exit(_code);
}
