#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>

#include "kernels.cuh"

struct Renderer : public Kernels {
  Renderer(int2 _dimensions, int _buffer, dim3 _block_size);
  ~Renderer();
  void render(float _mag, float2 _off);
  void copyToSurface(float * _array, float _mul);
  void copyToSurface(float2 * _array, float _mul);
private:
  void ReportFailure() const;
  int2 __res;
  SDL_Window * __window;
  SDL_GLContext __context;
  GLuint __renderTex;
  cudaGraphicsResource_t __renderTexSurface;
};
