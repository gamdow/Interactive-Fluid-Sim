#include "renderer.hpp"

#include "interface.hpp"
#include "opengl.hpp"
#include "data/resolution.cuh"

Renderer::Renderer(OpenGL & _opengl, Renderable & _camera, Renderable & _simulation, Interface & _interface)
  : __opengl(_opengl)
  , __camera(_camera)
  , __simulation(_simulation)
  , __interface(_interface)
{
}

void Renderer::render() {
  float magnification = __interface.magnification();
  float2 offset = __interface.offset();
  __camera.render(__opengl.resolution(), magnification, offset);
  __simulation.render(__opengl.resolution(), magnification, offset);
  __interface.render(__opengl.resolution(), magnification, offset);
  __opengl.swapWindow();
}
