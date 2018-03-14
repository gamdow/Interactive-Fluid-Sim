#include "render.hpp"

#include <iostream>

#include <cuda_gl_interop.h>

#include "helper_math.h"
#include "helper_cuda.h"
#include "kernels.cuh"

Renderer::Renderer(Kernels & _kernels)
  : __kernels(_kernels)
  , __window(nullptr)
  , __context(nullptr)
  , __visTexture(0u)
  , __visSurface(nullptr)
  , __textTexture(0u)
  , __textSurface(nullptr) {

  if(SDL_Init(SDL_INIT_VIDEO) < 0) {
    ReportFailure();
    return;
  }

  __window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _kernels.__dims.x, _kernels.__dims.y, SDL_WINDOW_OPENGL);
  if(__window == nullptr) {
    ReportFailure();
    return;
  }

  // 3.1 Needed for immediate mode (glBegin/End) rendering
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  __context = SDL_GL_CreateContext(__window);
  if(__context == nullptr) {
    ReportFailure();
    return;
  }

  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetSwapInterval(0); // V-Sync off for max speed

  std::cout << "OpenGL: " << glGetString(GL_VERSION) << std::endl;

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_BLEND); // Need blending for text overlay
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Generate visualisation render texture
  glGenTextures(1, &__visTexture);
  glBindTexture(GL_TEXTURE_2D, __visTexture); {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _kernels.__dims.x, _kernels.__dims.y, 0, GL_RGBA, GL_FLOAT, nullptr);
  } glBindTexture(GL_TEXTURE_2D, 0);

  // Register texture as surface reference (can't write to texture directly)
  checkCudaErrors(cudaGraphicsGLRegisterImage(&__visSurface, __visTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

  if(TTF_Init() < 0) {
    std::cout << TTF_GetError() << std::endl;
    return;
  }

  __font = TTF_OpenFont("FreeSans.ttf", 24);
  if (__font == nullptr) {
    std::cout << "Missing Font" << std::endl;
    return;
  }

  glGenTextures(1, &__textTexture);
  setText("");
}

Renderer::~Renderer() {
  TTF_Quit();
  TTF_CloseFont(__font);
  SDL_FreeSurface(__textSurface);
  SDL_GL_DeleteContext(__context);
  SDL_DestroyWindow(__window);
  SDL_Quit();
}

void Renderer::setText(char const * _val) {
  if(*_val == 0) {
    // empty string -> 0 x 0 texture -> seg fault
    _val = " ";
  }
  SDL_FreeSurface(__textSurface);
  SDL_Color color = {255, 255, 255, 0}; // Red
  __textSurface = TTF_RenderText_Blended_Wrapped(__font, _val, color, 640);
  glBindTexture(GL_TEXTURE_2D, __textTexture); {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, __textSurface->w, __textSurface->h, 0, GL_BGRA, GL_UNSIGNED_BYTE, __textSurface->pixels);
  } glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::render(float _mag, float2 _off) {
  glBindTexture(GL_TEXTURE_2D, __visTexture); {
      glBegin(GL_QUADS); {
        auto vf = [_mag, _off](float u, float v) {
          float2 vertex = (make_float2(u, 1.0f - v) * 2.0f - make_float2(1.0f)) * _mag + _off;
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

  glBindTexture(GL_TEXTURE_2D, __textTexture); {
    float x_prop = static_cast<float>(__textSurface->w) / __kernels.__dims.x;
    float y_prop = static_cast<float>(__textSurface->h) / __kernels.__dims.y;
    glBegin(GL_QUADS); {
      glTexCoord2f(0.0f, 0.0f); glVertex2f(-.9f, .9f);
      glTexCoord2f(1.0f, 0.0f); glVertex2f(-.9f + 1.8f * x_prop, .9f);
      glTexCoord2f(1.0f, 1.0f); glVertex2f(-.9f + 1.8f * x_prop, .9f - 1.8f * y_prop);
      glTexCoord2f(0.0f, 1.0f); glVertex2f(-.9f, .9f - 1.8f * y_prop);
    }
    glEnd();
  } glBindTexture(GL_TEXTURE_2D, 0);

  SDL_GL_SwapWindow(__window);
}

void Renderer::ReportFailure() const {std::cout << SDL_GetError() << std::endl;}

void Renderer::copyToSurface(float * _array, float _mul) {
  checkCudaErrors(cudaGraphicsMapResources(1, &__visSurface)); {
    cudaArray_t writeArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&writeArray, __visSurface, 0, 0));
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    checkCudaErrors(cudaCreateSurfaceObject(&writeSurface, &wdsc));
    __kernels.v2rgba(writeSurface, _array, _mul);
    checkCudaErrors(cudaDestroySurfaceObject(writeSurface));
  } checkCudaErrors(cudaGraphicsUnmapResources(1, &__visSurface));

  //checkCudaErrors(cudaStreamSynchronize(0));
}

void Renderer::copyToSurface(float2 * _array, float _mul) {
  checkCudaErrors(cudaGraphicsMapResources(1, &__visSurface)); {
    cudaArray_t writeArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&writeArray, __visSurface, 0, 0));
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    checkCudaErrors(cudaCreateSurfaceObject(&writeSurface, &wdsc));
    __kernels.hsv2rgba(writeSurface, _array, _mul);
    checkCudaErrors(cudaDestroySurfaceObject(writeSurface));
  } checkCudaErrors(cudaGraphicsUnmapResources(1, &__visSurface));

  //checkCudaErrors(cudaStreamSynchronize(0));
}

void Renderer::copyToSurface(float4 * _array, float3 const _map[4]) {
  checkCudaErrors(cudaGraphicsMapResources(1, &__visSurface)); {
    cudaArray_t writeArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&writeArray, __visSurface, 0, 0));
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    checkCudaErrors(cudaCreateSurfaceObject(&writeSurface, &wdsc));
    __kernels.float42rgba(writeSurface, _array, _map);
    checkCudaErrors(cudaDestroySurfaceObject(writeSurface));
  } checkCudaErrors(cudaGraphicsUnmapResources(1, &__visSurface));

  //checkCudaErrors(cudaStreamSynchronize(0));
}
