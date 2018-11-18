#pragma once

#include <SDL2/SDL_opengl.h>

#include "i_renderable.h"

struct Resolution;

struct IRenderer {
  ITextureRenderTarget & newTextureRenderTarget(GLint _internal, GLenum _format, GLenum _type) {return __newTextureRenderTarget(_internal, _format, _type);}
  ISurfaceRenderTarget & newSurfaceRenderTarget(GLint _internal, GLenum _format, GLenum _type, Resolution const & _res) {return __newSurfaceRenderTarget(_internal, _format, _type, _res);}
  ITextRenderTarget & newTextRenderTarget() {return __newTextRenderTarget();}
private:
  virtual ITextureRenderTarget & __newTextureRenderTarget(GLint _internal, GLenum _format, GLenum _type) = 0;
  virtual ISurfaceRenderTarget & __newSurfaceRenderTarget(GLint _internal, GLenum _format, GLenum _type, Resolution const & _res) = 0;
  virtual ITextRenderTarget & __newTextRenderTarget() = 0;
};
