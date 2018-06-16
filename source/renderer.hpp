#pragma once

#include "data/render_quad.hpp"

struct Resolution;

struct Renderable {
  void render(Resolution const & _window_res, float _mag, float2 _off) {__render(_window_res, _mag, _off);}
private:
  virtual void __render(Resolution const & _window_res, float _mag, float2 _off) = 0;
};

struct OpenGL;
struct Interface;

struct Renderer {
  Renderer(OpenGL & _opengl, Renderable & _camera, Renderable & _simulation, Interface & _interface);
  void render();
private:
  OpenGL & __opengl;
  Renderable & __camera;
  Renderable & __simulation;
  Interface & __interface;
  TextRenderQuad __text;
};
