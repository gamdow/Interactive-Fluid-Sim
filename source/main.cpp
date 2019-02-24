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
#include "interface/enums.h"
#include "renderer.h"
#include "data/render_quad.h"


int const VIDCAM_INDEX = 0;
Resolution const RESOLUTION = Resolution(800, 450);//Resolution(640, 360);//
bool const FULLSCREEN = true;
int const BUFFER = 10u;
float2 const LENGTH = {1.6f, 0.9f};
float const FRAME_RATE = 30.0f;
int const SIM_STEPS_PER_FRAME = 3;
int const PRESSURE_SOLVER_STEPS = 200;

void defaultInterface(Interface & interface) {
  interface.velocity() = 1.0f;
  interface.filterValue() = 1.0f;
  interface.filterRange() = 0.75f;
  interface.filterMode() = FilterMode::HUE;
  interface.bgSubtract() = false;
  interface.mode() = Mode::smoke;
  interface.debugMode() = false;
  interface.mirrorCam() = false;
  interface.flowRotate() = FlowDirection::LEFT_TO_RIGHT;
}

void quit(int _code, char const * _message);

int main(int argc, char * argv[]) {

  std::cout << std::endl;
  OpenGL opengl(RESOLUTION, FULLSCREEN);
  reportCudaCapability();
  OptimalBlockConfig blockConfig(RESOLUTION);
  Interface interface(FRAME_RATE);
  defaultInterface(interface);
  Renderer renderer(interface, opengl);

  std::cout << std::endl;
  float const TIME_DELTA = 1.0f / (SIM_STEPS_PER_FRAME * FRAME_RATE);
  float2 const DX = make_float2(LENGTH.x / blockConfig.optimal_res.width, LENGTH.y / blockConfig.optimal_res.height);
  float const MAX_VELOCITY = DX.y / TIME_DELTA;
  Simulation simulation(blockConfig, BUFFER, DX, PRESSURE_SOLVER_STEPS);

  std::cout << std::endl;
  Camera * camera = nullptr;
  try {
    camera = new CVCamera(VIDCAM_INDEX, RESOLUTION, FRAME_RATE);
  } catch (std::exception const & e) {
    std::cout << e.what() << std::endl << std::endl;
    delete camera;
    camera = new NullCamera(RESOLUTION);
    interface.filterMode() = FilterMode::LIGHTNESS;
    interface.filterValue() = 0.0f;
  }
  CameraFilter camera_filter(blockConfig, BUFFER);

  std::cout << std::endl;
  TextRenderQuad interface_render(renderer);
  TextureRenderQuad camera_render(renderer, GL_RGB, GL_BGR, GL_UNSIGNED_BYTE);
  SurfaceRenderQuad pip_render(renderer, GL_RGBA32F, GL_RGBA,  GL_FLOAT, simulation.visualisation_resolution()); {
    RenderQuad::QuadArray verts;
    verts[0] = make_float2(0.3f, -0.3f);
    verts[1] = make_float2(0.9f, -0.3f);
    verts[2] = make_float2(0.9f, -0.9f);
    verts[3] = make_float2(0.3f, -0.9f);
    pip_render.setVerts(verts);
  }
  SurfaceRenderQuad simulation_render(renderer, GL_RGBA32F, GL_RGBA, GL_FLOAT, simulation.visualisation_resolution());

  std::cout << std::endl << "dx = " << DX.x << "," << DX.y << " Max Velocity = " << MAX_VELOCITY;

  bool stop = false;
  SDL_Event event;
  float pressure_solve_steps_float = static_cast<float>(PRESSURE_SOLVER_STEPS);
  while(true) {
    interface.resetFlags();

    while(SDL_PollEvent(&event)) {
      interface.updateInputs(event);
      switch(event.type) {
        case SDL_KEYDOWN:
          switch(event.key.keysym.sym) {
            case SDLK_a: simulation.reset(); break;
            case SDLK_ESCAPE: stop = true; break;
            case SDLK_RETURN: if(event.key.keysym.mod & KMOD_ALT) opengl.toggleFullscreen(); break;
            defaut: break;
          }
          break;
        case SDL_QUIT: stop = true; // fall-through
        default: break;
      }
    }

    if(stop) break;

    ArrayStructConst<uchar3> frame_data = camera->frameData();

    camera_render.flipUVs(interface.mirrorCam(), false);
    camera_filter.update(frame_data, interface.mirrorCam(), interface.filterMode(), interface.bgSubtract(), interface.filterValue(), interface.filterRange());

    simulation.updateFluidCells(camera_filter.output());
    simulation.applyBoundary(interface.velocity() * MAX_VELOCITY, interface.flowRotate());
    simulation.applySmoke(interface.flowRotate());
    for(int i = 0; i < SIM_STEPS_PER_FRAME; ++i) {
      simulation.step(interface.mode(), TIME_DELTA);
    }

    camera_render.bindTexture(frame_data.resolution.width, frame_data.resolution.height, frame_data.data);
    camera_render.render();
    simulation_render.setSurfaceData(simulation.visualisation_surface_writer());
    simulation_render.render();
    if(interface.debugMode() || interface.filterChangedRecently()) {
      pip_render.setSurfaceData(blockConfig, camera_filter.render(), camera_filter.buffer_resolution());
      pip_render.render();
    }
    interface_render.setText(interface.screenText().c_str());
    interface_render.render();
    renderer.swapBuffers();

    interface.updateAndLimitFps();

    // try to tune the number of pressure solver steps to match the desired fps
    pressure_solve_steps_float += 0.1f * interface.fpsDelta();
    simulation.pressureSolverSteps() = static_cast<int>(pressure_solve_steps_float + 0.5f);
  }

  delete camera;
  quit(0, "");
}

void quit(int _code, char const * _message) {
  std::cout << _message << std::endl;
  exit(_code);
}
