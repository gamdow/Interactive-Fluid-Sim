#pragma once

#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "component.h"
#include "data/resolution.h"

struct OpenGL : public Component {
  OpenGL(Resolution _res, bool _fullscreen);
  virtual ~OpenGL();
  Resolution const & render_resolution() const {return __render_resolution;}
  void swapWindow() {SDL_GL_SwapWindow(__window);}
  void toggleFullscreen();
  TTF_Font * font() const {return __font;}
private:
  void ReportFailure() const;
  Resolution __render_resolution;
  SDL_Window * __window;
  SDL_GLContext __context;
  TTF_Font * __font;
};
