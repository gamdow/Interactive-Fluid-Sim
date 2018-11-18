#pragma once

#include <cuda_runtime.h>
#include <SDL2/SDL_opengl.h>

struct IRenderable {
  void render() {__render();}
private:
  virtual void __render() = 0;
};

struct ITextureRenderTarget : public IRenderable {
  void bindTexture(GLsizei _width, GLsizei _height, GLvoid const * _data) {__bindTexture(_width, _height, _data);}
private:
  virtual void __bindTexture(GLsizei _width, GLsizei _height, GLvoid const * _data) = 0;
};

struct SurfaceWriter;

struct ISurfaceRenderTarget : public IRenderable {
  void setSurfaceData(SurfaceWriter const & _writer) {__setSurfaceData(_writer);}
private:
  virtual void __setSurfaceData(SurfaceWriter const & _writer) = 0;
};

struct ITextRenderTarget : public IRenderable {
  void setText(char const * _val) {__setText(_val);}
private:
  virtual void __setText(char const * _val) = 0;
};
