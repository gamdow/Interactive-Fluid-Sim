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
#include "data/render_quad.h"

void quit(int _code, char const * _message);

int const VIDCAM_INDEX = 0;
Resolution const RESOLUTION = Resolution(640, 360);
int const BUFFER = 10u;
float2 const LENGTH = {1.6f, 0.9f};
float const FRAME_RATE = 60.0f;
int const SIM_STEPS_PER_FRAME = 3;
int const PRESSURE_SOLVER_STEPS = 200;
float TIME_DELTA = 1.0f / (SIM_STEPS_PER_FRAME * FRAME_RATE);

int main(int argc, char * argv[]) {

  std::cout << std::endl;
  OpenGL opengl(RESOLUTION);
  reportCudaCapability();
  OptimalBlockConfig blockConfig(RESOLUTION);
  Interface interface(FRAME_RATE);
  Renderer renderer(interface, opengl);

  std::cout << std::endl;
  float2 const DX = make_float2(LENGTH.x / blockConfig.optimal_res.width, LENGTH.y / blockConfig.optimal_res.height);
  Simulation simulation(interface, blockConfig, BUFFER, DX, PRESSURE_SOLVER_STEPS);

  std::cout << std::endl;
  Camera * camera = nullptr;
  try {
    camera = new CVCamera(VIDCAM_INDEX, RESOLUTION, FRAME_RATE);
  } catch (std::exception const & e) {
    std::cout << e.what() << std::endl << std::endl;
    delete camera;
    camera = new NullCamera(RESOLUTION);
  }
  CameraFilter camera_filter(blockConfig, BUFFER);

  std::cout << std::endl;
  TextRenderQuad interface_render(renderer);
  TextureRenderQuad camera_render(renderer, GL_RGB, GL_BGR, GL_UNSIGNED_BYTE);
  SurfaceRenderQuad camera_filter_render(renderer, GL_R8, GL_RED,  GL_UNSIGNED_BYTE, camera_filter.buffer_resolution()); {
    RenderQuad::QuadArray verts;
    verts[0] = make_float2(0.3f, -0.3f);
    verts[1] = make_float2(0.9f, -0.3f);
    verts[2] = make_float2(0.9f, -0.9f);
    verts[3] = make_float2(0.3f, -0.9f);
    camera_filter_render.setVerts(verts);
  }
  SurfaceRenderQuad simulation_render(renderer, GL_RGBA32F, GL_RGBA, GL_FLOAT, simulation.visualisation_resolution());

  std::cout << std::endl << "dx = " << DX.x << "," << DX.y << " Max Velocity = " << LENGTH.y / 2.0f * SIM_STEPS_PER_FRAME / FRAME_RATE;

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
            case SDLK_ESCAPE: stop = true; break;
            defaut: break;
          }
          break;
        case SDL_QUIT: stop = true; // fall-through
        default: break;
      }
    }

    if(stop) break;

    ArrayStructConst<uchar3> frame_data = camera->frameData();

    camera_filter.update(frame_data, interface.filterMode(), interface.filterValue(), interface.filterRange());

    simulation.updateFluidCells(camera_filter.output());
    simulation.applyBoundary();
    simulation.applySmoke();
    for(int i = 0; i < SIM_STEPS_PER_FRAME; ++i) {
      simulation.step(TIME_DELTA);
    }

    camera_render.bindTexture(frame_data.resolution.width, frame_data.resolution.height, frame_data.data);
    camera_render.render();
    simulation_render.setSurfaceData(simulation.visualisation_surface_writer());
    simulation_render.render();
    if(interface.filterChangedRecently()) {
      camera_filter_render.setSurfaceData(blockConfig, camera_filter.render(), camera_filter.buffer_resolution());
      camera_filter_render.render();      
    }
    interface_render.setText(interface.screenText().c_str());
    interface_render.render();
    renderer.swapBuffers();

    interface.updateAndLimitFps();
  }

  delete camera;
  quit(0, "");
}

void quit(int _code, char const * _message) {
  std::cout << _message << std::endl;
  exit(_code);
}
