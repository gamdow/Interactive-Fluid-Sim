#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <opencv2/opencv.hpp>

#include "../kernels/kernels_wrapper.cuh"
#include "resolution.cuh"

struct RenderQuad {
  RenderQuad(Resolution const & _res, GLint _internal, GLenum _format, GLenum _type);
  void render(Resolution _window_res, float _mag, float2 _off);
  void bindTexture(GLsizei width, GLsizei height, GLvoid const * data);
  void bindTexture(cv::Mat const & _mat);
protected:
  GLuint id() const {return __id;}
  Resolution resolution() const {return __resolution;}
  static const int num_verts = 4;
  float4 verts[num_verts];
private:
  virtual void updateQuad(Resolution _window_res, float _mag, float2 _off);
  GLuint __id;
  Resolution __resolution;
  GLint __internal;
  GLenum __format;
  GLenum __type;
};

struct SurfaceRenderQuad : public RenderQuad {
  SurfaceRenderQuad(KernelsWrapper & _kers, Resolution const & _res, GLint _internal, GLenum _format, GLenum _type);
  void copyToSurface(float4 * _array);
  // template<typename ARRAY, typename MULTIPLIER> void copyToSurface(ARRAY * _array, MULTIPLIER _mul);
private:
  cudaSurfaceObject_t createSurfaceObject();
  void destroySurfaceObject(cudaSurfaceObject_t _writeSurface);
  KernelsWrapper & __kernels;
  cudaGraphicsResource_t __surface;
};

// template<typename ARRAY, typename MULTIPLIER>
// void SurfaceRenderQuad::copyToSurface(ARRAY * _array, MULTIPLIER _mul) {
//   cudaSurfaceObject_t writeSurface = createSurfaceObject();
//   __kernels.array2rgba(writeSurface, resolution(), _array, _mul);
//   destroySurfaceObject(writeSurface);
// }

struct TextRenderQuad : public RenderQuad {
  TextRenderQuad();
  void setText(TTF_Font * _font, char const * _val);
private:
  virtual void updateQuad(Resolution _window_res, float _mag, float2 _off);
  SDL_Surface * __surface;
};
