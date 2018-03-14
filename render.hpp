#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <iostream>

struct Kernels;

struct Renderer {
  Renderer(Kernels & _kernels);
  ~Renderer();
  void setText(char const * _val);
  void render(float _mag, float2 _off);
  void copyToSurface(float * _array, float _mul);
  void copyToSurface(float2 * _array, float _mul);
  void copyToSurface(float4 * _array, float3 const _map[4]);
private:
  void ReportFailure() const;
  Kernels & __kernels;
  int2 __res;
  SDL_Window * __window;
  SDL_GLContext __context;
  GLuint __visTexture;
  cudaGraphicsResource_t __visSurface;
  TTF_Font * __font;
  GLuint __textTexture;
  SDL_Surface * __textSurface;
};
