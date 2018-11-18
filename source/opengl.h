#pragma once

#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "component.h"
#include "data/resolution.h"

struct OpenGL : public Component {
  OpenGL(Resolution _res);
  virtual ~OpenGL();
  Resolution const & resolution() const {return __resolution;}
  void swapWindow() {SDL_GL_SwapWindow(__window);}
  TTF_Font * font() const {return __font;}
private:
  void ReportFailure() const;
  Resolution __resolution;
  SDL_Window * __window;
  SDL_GLContext __context;
  TTF_Font * __font;
};
