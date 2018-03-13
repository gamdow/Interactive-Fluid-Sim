#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>

struct Renderer {
  Renderer(int2 _res);
  ~Renderer();
  template<class T> void render(T * _array, float _mul, float _mag, float2 _off);
private:
  void renderTexture(float _mag, float2 _off);
  void ReportFailure() const;
  int2 __res;
  SDL_Window * __window;
  SDL_GLContext __context;
  GLuint __renderTex;
  cudaGraphicsResource_t __renderTexSurface;
};


template<class T>
void Renderer::render(T * _array, float _mul, float _mag, float2 _off) {
  int2 res = {__res.x, __res.y};
  copy_to_surface(_array, _mul, res, __renderTexSurface);
  renderTexture(_mag, _off);
}
