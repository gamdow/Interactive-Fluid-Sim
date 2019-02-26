#include "opengl.h"

#include <iostream>

#include "debug.h"

OpenGL::OpenGL(Resolution _render_res, bool _fullscreen)
  : __render_resolution(_render_res)
  , __window(nullptr)
  , __context(nullptr)
  , __font(nullptr)
{
  if(SDL_Init(SDL_INIT_VIDEO) < 0) {
    throwFailure(SDL_GetError());
  }

  SDL_DisplayMode display_mode;
  SDL_GetDesktopDisplayMode(0, &display_mode);

  format_out << "Creating SDL Window:" << std::endl;
  OutputIndent indent1;
  Resolution window_res = _fullscreen ? Resolution(display_mode.w, display_mode.h) : __render_resolution;
  window_res.print("Window Resolution");
  __window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_res.width.inner, window_res.height.inner, SDL_WINDOW_OPENGL);
  if(__window == nullptr) {
    throwFailure(SDL_GetError());
  }
  SDL_SetWindowFullscreen(__window, _fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);

  // 3.1 Needed for immediate mode (glBegin/End) rendering
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetSwapInterval(0); // V-Sync off for max speed

  __context = SDL_GL_CreateContext(__window);
  if(__context == nullptr) {
    throwFailure(SDL_GetError());
  }

  format_out << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_BLEND); // Need blending for text overlay
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if(TTF_Init() < 0) {
    format_out << TTF_GetError() << std::endl;
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

void OpenGL::toggleFullscreen() {
  // Can't get this code working correctly, so I'm disabling it. Currently only resizes the window without resizing the openGL render viewport. Suspect it needs updating directly, or reinitialising.
  return;

  bool fullscreen = SDL_GetWindowFlags(__window) & SDL_WINDOW_FULLSCREEN_DESKTOP;
  if(fullscreen) {
    SDL_SetWindowFullscreen(__window, 0);
    SDL_SetWindowSize(__window, __render_resolution.width.inner, __render_resolution.height.inner);
    SDL_SetWindowPosition(__window, SDL_WINDOWPOS_CENTERED,  SDL_WINDOWPOS_CENTERED);
  } else {
    SDL_DisplayMode display_mode;
    SDL_GetDesktopDisplayMode(0, &display_mode);
    SDL_SetWindowSize(__window, display_mode.w, display_mode.h);
    SDL_SetWindowFullscreen(__window, SDL_WINDOW_FULLSCREEN_DESKTOP);
  }

  format_out << " " << std::endl << "Fullscreen Toggle" << std::endl;
  SDL_DisplayMode display_mode;
  SDL_GetDesktopDisplayMode(0, &display_mode);
  format_out << "Screen DisplayMode:" << std::endl;
  {
    OutputIndent indent1;
    format_out << "format:" << display_mode.format << std::endl;
    format_out << "w:" << display_mode.w << std::endl;
    format_out << "h:" << display_mode.h << std::endl;
    format_out << "refresh rate:" << display_mode.refresh_rate << std::endl;
  }
  SDL_GetWindowDisplayMode(__window, &display_mode);
  format_out << "Window DisplayMode:" << std::endl;
  {
    OutputIndent indent1;
    format_out << "format:" << display_mode.format << std::endl;
    format_out << "w:" << display_mode.w << std::endl;
    format_out << "h:" << display_mode.h << std::endl;
    format_out << "refresh rate:" << display_mode.refresh_rate << std::endl;
  }
  int w, h;
  SDL_GetWindowSize(__window, &w, &h);
  format_out << "WindowSize:" << std::endl;
  {
    OutputIndent indent1;
    format_out << "w:" << w << std::endl;
    format_out << "h:" << h << std::endl;
  }
  SDL_GL_GetDrawableSize(__window, &w, &h);
  format_out << "DrawableSize:" << std::endl;
  {
    OutputIndent indent1;
    format_out << "w:" << w << std::endl;
    format_out << "h:" << h << std::endl;
  }
}
