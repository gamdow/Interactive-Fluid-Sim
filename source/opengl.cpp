#include "opengl.h"

#include <iostream>

#include "debug.h"

OpenGL::OpenGL(Resolution _res)
  : __resolution(_res)
  , __window(nullptr)
  , __context(nullptr)
  , __font(nullptr)
{
  if(SDL_Init(SDL_INIT_VIDEO) < 0) {
    throwFailure(SDL_GetError());
  }

  format_out << "Creating SDL Window:" << std::endl;
  OutputIndent indent1;
  __resolution.print("Resolution");
  __window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, __resolution.width, __resolution.height, SDL_WINDOW_OPENGL);
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

  format_out << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_BLEND); // Need blending for text overlay
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if(TTF_Init() < 0) {
    std::cout << TTF_GetError() << std::endl;
    return;
  }

  __font = TTF_OpenFont("FreeSans.ttf", 20);
  if (__font == nullptr) {
    throwFailure(SDL_GetError());
  }
}

OpenGL::~OpenGL() {
  TTF_Quit();
  TTF_CloseFont(__font);
  SDL_GL_DeleteContext(__context);
  SDL_DestroyWindow(__window);
  SDL_Quit();
}

void OpenGL::ReportFailure() const {std::cout << SDL_GetError() << std::endl;}
