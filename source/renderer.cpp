#include "renderer.hpp"

#include <iostream>
#include <cuda_gl_interop.h>

#include "helper_cuda.h"

OpenGLInitialiser::OpenGLInitialiser(Resolution _res)
  : __window(nullptr)
  , __context(nullptr)
  , __font(nullptr)
{
  if(SDL_Init(SDL_INIT_VIDEO) < 0) {
    throwFailure(SDL_GetError());
  }

  _res.print("Creating SDL Window: Resolution");
  __window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _res.width, _res.height, SDL_WINDOW_OPENGL);
  if(__window == nullptr) {
    throwFailure(SDL_GetError());
  }

  // 3.1 Needed for immediate mode (glBegin/End) rendering
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  __context = SDL_GL_CreateContext(__window);
  if(__context == nullptr) {
    throwFailure(SDL_GetError());
  }

  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetSwapInterval(0); // V-Sync off for max speed

  std::cout << std::endl << "OpenGL: " << glGetString(GL_VERSION) << std::endl;

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_BLEND); // Need blending for text overlay
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if(TTF_Init() < 0) {
    std::cout << TTF_GetError() << std::endl;
    return;
  }

  __font = TTF_OpenFont("FreeSans.ttf", 24);
  if (__font == nullptr) {
    throwFailure(SDL_GetError());
  }
}

OpenGLInitialiser::~OpenGLInitialiser() {
  TTF_Quit();
  TTF_CloseFont(__font);
  SDL_GL_DeleteContext(__context);
  SDL_DestroyWindow(__window);
  SDL_Quit();
}

void OpenGLInitialiser::ReportFailure() const {std::cout << SDL_GetError() << std::endl;}

Renderer::Renderer(Resolution _res, Camera & _cam, KernelsWrapper & _kers)
  : OpenGLInitialiser(_res)
  , __windowRes(_res)
  , __kernels(_kers)
  , __background(_cam.resolution, GL_RGB, GL_BGR, GL_UNSIGNED_BYTE)
  , __visualisation(_kers, _kers.optimal_res, GL_RGBA32F, GL_RGBA, GL_FLOAT)
  , __text()
{
}

Renderer::~Renderer() {
}

void Renderer::render(float _mag, float2 _off) {
  __background.render(__windowRes, _mag, _off);
  __visualisation.render(__windowRes, _mag, _off);
  __text.render(__windowRes, _mag, _off);
  swapWindow();
}
