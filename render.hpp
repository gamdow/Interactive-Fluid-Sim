#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>

struct Kernels;

struct Renderer {
  Renderer(Kernels & _kernels);
  ~Renderer();
  void render(float _mag, float2 _off);
  void copyToSurface(float * _array, float _mul);
  void copyToSurface(float2 * _array, float _mul);
private:
  void ReportFailure() const;
  Kernels & __kernels;
  int2 __res;
  SDL_Window * __window;
  SDL_GLContext __context;
  GLuint __renderTex;
  cudaGraphicsResource_t __renderTexSurface;
};
