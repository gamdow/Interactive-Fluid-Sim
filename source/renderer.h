#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>

#include "i_renderer.h"
#include "i_renderable.h"
#include "i_render_settings.h"

struct OpenGL;
struct Interface;
struct Resolution;
struct RenderQuad;

struct Renderer : public IRenderer, public IRenderSettings {
  Renderer(Interface const & _interface, OpenGL & _opengl);
  virtual ~Renderer();
  void swapBuffers();
private:
  // From IRenderer
  virtual ITextureRenderTarget & __newTextureRenderTarget(GLint _internal, GLenum _format, GLenum _type);
  virtual ISurfaceRenderTarget & __newSurfaceRenderTarget(GLint _internal, GLenum _format, GLenum _type, Resolution const & _res);
  virtual ITextRenderTarget & __newTextRenderTarget();
  // From IRenderSettings
  virtual Resolution const & __resolution() const;
  virtual float __magnification() const;
  virtual float2 __offset() const;
  virtual TTF_Font * __font() const;
  //
  Interface const & __interface;
  OpenGL & __opengl;
  std::vector<RenderQuad*> __quads;
};
