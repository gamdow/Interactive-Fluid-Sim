#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>

#include "resolution.h"

struct IRenderSettings;
struct SDL_Surface;
struct SurfaceWriter;

struct RenderQuad {
  RenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type);
  virtual ~RenderQuad() {}
  void render() {__render();}
  void bindTexture(GLsizei _width, GLsizei _height, GLvoid const * _data);
protected:
  GLuint id() const {return __id;}
  Resolution resolution() const {return __resolution;}
  IRenderSettings const & renderSettings() const {return __settings;}
  float2 scale() const;
  void updateVerts();
  void renderVerts();
  static const int num_verts = 4;
  float4 verts[num_verts];
private:
  virtual void __render() {updateVerts(); renderVerts();}
  IRenderSettings const & __settings;
  GLuint __id;
  GLint __internal;
  GLenum __format;
  GLenum __type;
  Resolution __resolution;
};

struct TextureRenderQuad: public RenderQuad {
  TextureRenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type) : RenderQuad(_render_settings, _internal, _format, _type) {}
};

struct SurfaceRenderQuad : public RenderQuad {
  SurfaceRenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type, Resolution const & _res);
  virtual ~SurfaceRenderQuad();
  void setSurfaceData(SurfaceWriter const & _writer);
private:
  cudaGraphicsResource_t __resource;
  cudaSurfaceObject_t __surface;
};

struct TextRenderQuad : public RenderQuad {
  TextRenderQuad(IRenderSettings const & _render_settings) : RenderQuad(_render_settings, GL_RGBA, GL_BGRA, GL_UNSIGNED_BYTE), __surface(nullptr) {}
  void setText(char const * _val);
private:
  virtual void __render();
  SDL_Surface * __surface;
};
