#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>

#include "resolution.h"
#include "../i_renderable.h"

struct IRenderSettings;
struct SDL_Surface;
struct SurfaceWriter;

struct RenderQuad {
  RenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type);
  virtual ~RenderQuad() {}
protected:
  GLuint id() const {return __id;}
  Resolution resolution() const {return __resolution;}
  IRenderSettings const & renderSettings() const {return __settings;}
  float2 scale() const;
  void bindTexture(GLsizei _width, GLsizei _height, GLvoid const * _data);
  void updateVerts();
  void renderVerts();
  static const int num_verts = 4;
  float4 verts[num_verts];
private:
  IRenderSettings const & __settings;
  GLuint __id;
  GLint __internal;
  GLenum __format;
  GLenum __type;
  Resolution __resolution;
};

struct TextureRenderQuad: public RenderQuad, public ITextureRenderTarget {
  TextureRenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type) : RenderQuad(_render_settings, _internal, _format, _type) {}
private:
  virtual void __render() {updateVerts(); renderVerts();}
  virtual void __bindTexture(GLsizei _width, GLsizei _height, GLvoid const * _data) {RenderQuad::bindTexture(_width, _height, _data);}
};

struct SurfaceRenderQuad : public RenderQuad, public ISurfaceRenderTarget {
  SurfaceRenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type, Resolution const & _res);
  virtual ~SurfaceRenderQuad();
private:
  virtual void __render() {updateVerts(); renderVerts();}
  virtual void __setSurfaceData(SurfaceWriter const & _writer);
  cudaGraphicsResource_t __resource;
  cudaSurfaceObject_t __surface;
};

struct TextRenderQuad : public RenderQuad, public ITextRenderTarget {
  TextRenderQuad(IRenderSettings const & _render_settings) : RenderQuad(_render_settings, GL_RGBA, GL_BGRA, GL_UNSIGNED_BYTE), __surface(nullptr) {}
private:
  virtual void __render();
  virtual void __setText(char const * _val);
  SDL_Surface * __surface;
};
