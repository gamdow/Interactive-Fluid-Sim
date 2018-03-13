#include "render.hpp"

#include <iostream>

#include <cuda_gl_interop.h>

#include "helper_math.h"
#include "kernels.cuh"

Renderer::Renderer(int2 _dimensions, int _buffer, dim3 _block_size)
  : Kernels(_dimensions, _buffer, _block_size)
  , __window(nullptr)
  , __context(nullptr) {

  if(SDL_Init(SDL_INIT_VIDEO) < 0) {
    ReportFailure();
    return;
  }

  __window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _dimensions.x, _dimensions.y, SDL_WINDOW_OPENGL);
  if(__window == nullptr) {
    ReportFailure();
    return;
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetSwapInterval(1); // V-Sync

  __context = SDL_GL_CreateContext(__window);
  if(__context == nullptr) {
    ReportFailure();
    return;
  }

  std::cout << glGetString(GL_VERSION) << std::endl;

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_2D);

  // Generate Render Texture
  glGenTextures(1, &__renderTex);
  glBindTexture(GL_TEXTURE_2D, __renderTex); {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _dimensions.x, _dimensions.y, 0, GL_RGBA, GL_FLOAT, nullptr);
  } glBindTexture(GL_TEXTURE_2D, 0);

  // Register texture as surface reference (can't write to texture directly)
  cudaGraphicsGLRegisterImage(&__renderTexSurface, __renderTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

Renderer::~Renderer() {
  SDL_GL_DeleteContext(__context);
  SDL_DestroyWindow(__window);
  SDL_Quit();
}

void Renderer::render(float _mag, float2 _off) {
  glBindTexture(GL_TEXTURE_2D, __renderTex); {
      glBegin(GL_QUADS); {
        auto vf = [_mag, _off](float u, float v) {
          float2 vertex = (make_float2(u, v) * 2.0f - make_float2(1.0f)) * _mag + _off;
          glTexCoord2f(u, v);
          glVertex2f(vertex.x, vertex.y);
        };
        vf(0.0f, 0.0f);
        vf(1.0f, 0.0f);
        vf(1.0f, 1.0f);
        vf(0.0f, 1.0f);
      }
      glEnd();
  } glBindTexture(GL_TEXTURE_2D, 0);

  glFinish();

  SDL_GL_SwapWindow(__window);
}

void Renderer::ReportFailure() const {std::cout << SDL_GetError() << std::endl;}

void Renderer::copyToSurface(float2 * _array, float _mul) {
  cudaGraphicsMapResources(1, &__renderTexSurface); {
    cudaArray_t writeArray;
    cudaGraphicsSubResourceGetMappedArray(&writeArray, __renderTexSurface, 0, 0);
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    cudaCreateSurfaceObject(&writeSurface, &wdsc);
    hsv2rgba(writeSurface, _array, _mul);
    cudaDestroySurfaceObject(writeSurface);
  } cudaGraphicsUnmapResources(1, &__renderTexSurface);

  cudaStreamSynchronize(0);
}

void Renderer::copyToSurface(float * _array, float _mul) {
  cudaGraphicsMapResources(1, &__renderTexSurface); {
    cudaArray_t writeArray;
    cudaGraphicsSubResourceGetMappedArray(&writeArray, __renderTexSurface, 0, 0);
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    cudaCreateSurfaceObject(&writeSurface, &wdsc);
    v2rgba(writeSurface, _array, _mul);
    cudaDestroySurfaceObject(writeSurface);
  } cudaGraphicsUnmapResources(1, &__renderTexSurface);

  cudaStreamSynchronize(0);
}
