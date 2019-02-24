#include "renderer.h"

#include "interface.h"
#include "opengl.h"
#include "data/resolution.h"

Renderer::Renderer(Interface const & _interface, OpenGL & _opengl)
  : __interface(_interface)
  , __opengl(_opengl)
{
}

void Renderer::swapBuffers() {
  __opengl.swapWindow();
}

Resolution const & Renderer::__resolution() const {
  return __opengl.render_resolution();
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
