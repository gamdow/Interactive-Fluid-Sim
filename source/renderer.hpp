#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "resolution.cuh"
#include "camera.hpp"
#include "kernels_wrapper.cuh"

template<typename T>
struct RenderObject {
  RenderObject(KernelsWrapper & _kers, Resolution const & _res, GLint _internal, GLenum _format, GLenum _type);
  void render(Resolution _window_res, float _mag, float2 _off);
  template<typename ARRAY, typename MULTIPLIER> void copyToSurface(ARRAY * _array, MULTIPLIER _mul);
private:
  KernelsWrapper & __kernels;
  Resolution __resolution;
  GLuint __id;
  T __surface;
};

template<typename T> template<typename ARRAY, typename MULTIPLIER>
void RenderObject<T>::copyToSurface(ARRAY * _array, MULTIPLIER _mul) {
  cudaGraphicsMapResources(1, &__surface); {
    cudaArray_t writeArray;
    cudaGraphicsSubResourceGetMappedArray(&writeArray, __surface, 0, 0);
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = writeArray;
    cudaSurfaceObject_t writeSurface;
    cudaCreateSurfaceObject(&writeSurface, &wdsc);
    __kernels.array2rgba(writeSurface, _array, _mul);
    cudaDestroySurfaceObject(writeSurface);
  } cudaGraphicsUnmapResources(1, &__surface);
}

struct OpenGLInitialiser {
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
  void setText(char const * _val);
  void render(float _mag, float2 _off);
  template<typename ARRAY, typename MULTIPLIER> void copyToSurface(ARRAY * _array, MULTIPLIER _mul);
  RenderObject<cudaGraphicsResource_t> & getVisualisation() {return __visualisation;}
private:
  Resolution __windowRes;
  KernelsWrapper & __kernels;
  RenderObject<cudaGraphicsResource_t> __background;
  RenderObject<cudaGraphicsResource_t> __visualisation;
  GLuint __textTexture;
  SDL_Surface * __textSurface;
};
