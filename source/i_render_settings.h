#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_ttf.h>

struct Resolution;

struct IRenderSettings {
  Resolution const & resolution() const {return __resolution();}
  float magnification() const {return __magnification();}
  float2 offset() const {return __offset();}
  TTF_Font * font() const {return __font();}
private:
  virtual Resolution const & __resolution() const = 0;
  virtual float __magnification() const = 0;
  virtual float2 __offset() const = 0;
  virtual TTF_Font * __font() const = 0;
};
