#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <opencv2/opencv.hpp>

#include "../kernels/wrapper.cuh"
#include "resolution.cuh"

struct RenderQuad {
  RenderQuad(GLint _internal, GLenum _format, GLenum _type);
  void render(Resolution _window_res, float _mag, float2 _off);
  void render(Resolution const & _quad_res, Resolution const & _window_res, float _mag, float2 _off);
  void bindTexture(GLsizei width, GLsizei height, GLvoid const * data);
  void bindTexture(cv::Mat const & _mat);
  void setResolution(Resolution const & _res) {__resolution = _res;}
protected:
  GLuint id() const {return __id;}
  Resolution resolution() const {return __resolution;}
  static const int num_verts = 4;
  float4 verts[num_verts];
private:
  virtual void updateQuad(Resolution _window_res, float _mag, float2 _off);
  virtual void updateQuad(Resolution const & _quad_res, Resolution const & _window_res, float _mag, float2 _off);
  GLuint __id;
  Resolution __resolution;
  GLint __internal;
  GLenum __format;
  GLenum __type;
};

struct SurfaceRenderQuad : public RenderQuad {
  SurfaceRenderQuad(Resolution const & _res, GLint _internal, GLenum _format, GLenum _type);
  void copyToSurface(KernelWrapper const & _kernel, float4 const * _array);
private:
  cudaSurfaceObject_t createSurfaceObject();
  void destroySurfaceObject(cudaSurfaceObject_t _writeSurface);
  cudaGraphicsResource_t __surface;
};

struct TextRenderQuad : public RenderQuad {
  TextRenderQuad();
  void setText(TTF_Font * _font, char const * _val);
private:
  virtual void updateQuad(Resolution _window_res, float _mag, float2 _off);
  SDL_Surface * __surface;
};
