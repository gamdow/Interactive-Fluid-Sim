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

bool const FULLSCREEN = true;
int const VIDCAM_INDEX = 0; //normally 0 if only one webcam
int const BUFFER = 10u; // needed for advection which can sample outside renderable area
float2 const LENGTH = {1.6f, 0.9f}; // length dimensions of the simulation

bool const OPTICAL_FLOW = false; // very (10x) slow
Resolution const RESOLUTION = Resolution(1280, 720); // higher -> slower
float const FRAME_RATE = 30.0f; // target fps, will tune PSS to reach
int const SIM_STEPS_PER_FRAME = 5; // higher -> higher max fluid velocity
int const PRESSURE_SOLVER_STEPS = 100; // autotuned, good guess minimise oscillation at start, but higher -> more accurate

// bool const OPTICAL_FLOW = true;
// Resolution const RESOLUTION = Resolution(640, 360);
// float const FRAME_RATE = 15.0f;
// int const SIM_STEPS_PER_FRAME = 1;
// int const PRESSURE_SOLVER_STEPS = 200;

float const SOLVER_FPS_TUNER_DAMPING = 0.1f;
float const SOLVER_FPS_TUNER_DECAY = 0.999f;

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
  if(!reportCudaCapability()){
    throw std::runtime_error("No CUDA capable device");
  }
  OptimalBlockConfig blockConfig(Resolution(RESOLUTION, BUFFER, BUFFER));
  Interface interface(FRAME_RATE);
  defaultInterface(interface);
  Renderer renderer(interface, opengl);

  std::cout << std::endl;
  float const TIME_DELTA = 1.0f / (SIM_STEPS_PER_FRAME * FRAME_RATE);
  float2 const DX = make_float2(LENGTH.x / blockConfig.resolution.width, LENGTH.y / blockConfig.resolution.height);
  float const MAX_VELOCITY = DX.y / TIME_DELTA;
  Simulation simulation(blockConfig, DX, PRESSURE_SOLVER_STEPS);

  std::cout << std::endl;
  Camera * camera = nullptr;
  try {
    camera = new CVCamera(OPTICAL_FLOW, VIDCAM_INDEX, RESOLUTION, FRAME_RATE);
  } catch (std::exception const & e) {
    {
      OutputIndent indent;
      format_out << e.what() << std::endl;
    }
    delete camera;
    camera = new NullCamera(RESOLUTION);
    interface.filterMode() = FilterMode::LIGHTNESS;
    interface.filterValue() = 0.0f;
  }
  CameraFilter camera_filter(blockConfig);
  camera_filter.update(camera->frameData(), camera->flowData(), interface.mirrorCam(), interface.filterMode(), interface.bgSubtract(), interface.filterValue(), interface.filterRange());

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

  float tuner_damping = SOLVER_FPS_TUNER_DAMPING;
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
            case SDLK_BACKSPACE: simulation.reset(); break;
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

    // std::cout << "camera dt: " << camera->time_delta() << " " << 1 / camera->time_delta() << std::endl;
    camera_filter.update(camera->frameData(), camera->flowData(), interface.mirrorCam(), interface.filterMode(), interface.bgSubtract(), interface.filterValue(), interface.filterRange());

    simulation.updateFluidCells(camera_filter.fluidOutput());
    simulation.addFlow(camera_filter.velocityOutput(), 1.0f / FRAME_RATE);
    simulation.applyBoundary(interface.velocity() * MAX_VELOCITY, interface.flowRotate());
    simulation.applySmoke(interface.flowRotate());
    for(int i = 0; i < SIM_STEPS_PER_FRAME; ++i) {
      simulation.step(interface.mode(), TIME_DELTA);
    }

    ArrayStructConst<uchar3> frame_data = camera->frameData();
    camera_render.flipUVs(interface.mirrorCam(), false);
    camera_render.bindTexture(frame_data.resolution.width, frame_data.resolution.height, frame_data.data);
    camera_render.render();
    simulation_render.setSurfaceData(simulation.visualisation_surface_writer());
    simulation_render.render();
    if(interface.debugMode() || interface.filterChangedRecently()) {
      pip_render.setSurfaceData(blockConfig, camera_filter.render(), camera_filter.resolution());
      pip_render.render();
    }
    interface_render.setText(interface.screenText().c_str());
    interface_render.render();
    renderer.swapBuffers();

    interface.updateAndLimitFps();

    // try to tune the number of pressure solver steps to match the desired fps
    pressure_solve_steps_float += tuner_damping * interface.fpsDelta();
    tuner_damping *= SOLVER_FPS_TUNER_DECAY;
    simulation.pressureSolverSteps() = static_cast<int>(pressure_solve_steps_float + 0.5f);
  }

  delete camera;
  quit(0, "");
}

void quit(int _code, char const * _message) {
  std::cout << _message << std::endl;
  exit(_code);
}
