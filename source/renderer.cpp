#include "renderer.h"

#include "interface.h"
#include "opengl.h"
#include "data/resolution.h"
#include "data/render_quad.h"

Renderer::Renderer(Interface const & _interface, OpenGL & _opengl)
  : __interface(_interface)
  , __opengl(_opengl)
{
}

Renderer::~Renderer() {
  for(auto i = __quads.begin(); i != __quads.end(); ++i) {
    delete *i;
  }
}

void Renderer::swapBuffers() {
  __opengl.swapWindow();
}

ITextureRenderTarget & Renderer::__newTextureRenderTarget(GLint _internal, GLenum _format, GLenum _type) {
  auto target = new TextureRenderQuad(*this, _internal, _format, _type);
  __quads.push_back(target);
  return *target;
}

ISurfaceRenderTarget & Renderer::__newSurfaceRenderTarget(GLint _internal, GLenum _format, GLenum _type, Resolution const & _res) {
  auto target = new SurfaceRenderQuad(*this, _internal, _format, _type, _res);
  __quads.push_back(target);
  return *target;
}

ITextRenderTarget & Renderer::__newTextRenderTarget() {
  auto target = new TextRenderQuad(*this);
  __quads.push_back(target);
  return *target;
}

Resolution const & Renderer::__resolution() const {
  return __opengl.resolution();
}

float Renderer::__magnification() const {
  return __interface.magnification();
}

float2 Renderer::__offset() const {
  return __interface.offset();
}

TTF_Font * Renderer::__font() const {
  return __opengl.font();
}
