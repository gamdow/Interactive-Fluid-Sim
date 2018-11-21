#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>

#include "../cuda/utility.h"
#include "resolution.h"

struct IRenderSettings;
struct SDL_Surface;
struct SurfaceWriter;

struct RenderQuad {
  static const int NUM_VERTS = 4;
  typedef float2 QuadArray[NUM_VERTS];
  RenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type);
  virtual ~RenderQuad() {}
  void render();
  void bindTexture(GLsizei _width, GLsizei _height, GLvoid const * _data);
  void setVerts(QuadArray const & _verts);
protected:
  GLuint id() const {return __id;}
  Resolution resolution() const {return __resolution;}
  IRenderSettings const & renderSettings() const {return __settings;}
  float2 scale() const;
  QuadArray & verts() {return __verts;}
private:
  virtual float2 scaleVerts(float2 _vert, float2 _uv, float _mag, float2 _scale, float2 _offset) const;
  IRenderSettings const & __settings;
  GLuint __id;
  GLint __internal;
  GLenum __format;
  GLenum __type;
  Resolution __resolution;
  QuadArray __verts;
  QuadArray __uvs;
};

struct TextureRenderQuad: public RenderQuad {
  TextureRenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type) : RenderQuad(_render_settings, _internal, _format, _type) {}
};

struct SurfaceRenderQuad : public RenderQuad {
  SurfaceRenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type, Resolution const & _res);
  virtual ~SurfaceRenderQuad();
  void setSurfaceData(SurfaceWriter const & _writer);
  // void setSurfaceData(OptimalBlockConfig const & _block_config, float const * _buffer, Resolution const & _res);
  // void setSurfaceData(OptimalBlockConfig const & _block_config, unsigned char const * _buffer, Resolution const & _res);
  template<typename T>
  void setSurfaceData(OptimalBlockConfig const & _block_config, DeviceArray<T> const & _buffer, Resolution const & _res) {
    copyToSurface(_block_config, __surface, resolution(), _buffer.getData(), _res);
  }
private:
  cudaGraphicsResource_t __resource;
  cudaSurfaceObject_t __surface;
};

struct TextRenderQuad : public RenderQuad {
  TextRenderQuad(IRenderSettings const & _render_settings);
  void setText(char const * _val);
private:
  static float const SAFE_SCALE;
  virtual float2 scaleVerts(float2 _vert, float2 _uv, float _mag, float2 _scale, float2 _offset) const;
  SDL_Surface * __surface;
};
