#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <opencv2/opencv.hpp>

#include "render_quad.hpp"
#include "component.hpp"
#include "resolution.cuh"
#include "camera.hpp"
#include "kernels_wrapper.cuh"

struct OpenGLInitialiser: public Component {
  OpenGLInitialiser(Resolution _res);
  virtual ~OpenGLInitialiser();
  void swapWindow() {SDL_GL_SwapWindow(__window);}
  TTF_Font * getFont() {return __font;}
private:
  void ReportFailure() const;
  SDL_Window * __window;
  SDL_GLContext __context;
  TTF_Font * __font;
};

struct Renderer : public OpenGLInitialiser {
  Renderer(Resolution _res, Camera & _cam, KernelsWrapper & _kers);
  ~Renderer();
  void setText(char const * _val) {__text.setText(getFont(), _val);}
  void render(float _mag, float2 _off);
  template<typename ARRAY, typename MULTIPLIER> void copyToSurface(ARRAY * _array, MULTIPLIER _mul);
  RenderQuad & getBackground() {return __background;}
  SurfaceRenderQuad & getVisualisation() {return __visualisation;}
private:
  Resolution __windowRes;
  KernelsWrapper & __kernels;
  RenderQuad __background;
  SurfaceRenderQuad __visualisation;
  TextRenderQuad __text;
};
