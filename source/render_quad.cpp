#include "renderer.hpp"

#include <iostream>
#include <cuda_gl_interop.h>

#include "helper_cuda.h"

RenderQuad::RenderQuad(Resolution const & _res, GLint _internal, GLenum _format, GLenum _type)
  : __id(0u)
  , __resolution(_res)
  , __internal(_internal)
  , __format(_format)
  , __type(_type)
{
  for(int i = 0; i < 1000000; ++i) {
    int j = i;
    ++j;
  }
  verts[0] = make_float4(-1.f, 1.f, 0.f, 0.f);
  verts[1] = make_float4(1.f, 1.f, 1.f, 0.f);
  verts[2] = make_float4(1.f, -1.f, 1.f, 1.f);
  verts[3] = make_float4(-1.f, -1.f, 0.f, 1.f);
  glGenTextures(1, &__id);
  std::cout << "\tglGenTextures: " << __id << std::endl;
}

void RenderQuad::render(Resolution _window_res, float _mag, float2 _off) {
  updateQuad(_window_res, _mag, _off);
  glBindTexture(GL_TEXTURE_2D, __id); {
    glBegin(GL_QUADS); {
      for(int i = 0; i < 4; ++i) {
        float4 const & vert = verts[i];
        glTexCoord2f(vert.z, vert.w); glVertex2f(vert.x, vert.y);
      }
    } glEnd();
  } glBindTexture(GL_TEXTURE_2D, 0);
}

void RenderQuad::bindTexture(GLsizei width, GLsizei height, GLvoid const * data) {
  glBindTexture(GL_TEXTURE_2D, __id); {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, __internal, width, height, 0, __format, __type, data);
  } glBindTexture(GL_TEXTURE_2D, 0);
}

void RenderQuad::bindTexture(cv::Mat const & _mat) {
  bindTexture(_mat.cols, _mat.rows, _mat.data);
}

void RenderQuad::updateQuad(Resolution _window_res, float _mag, float2 _off) {
  for(int i = 0; i < num_verts; ++i) {
    verts[i].x = (verts[i].z * 2.f - 1.f) * _mag * static_cast<float>(__resolution.width) / static_cast<float>(_window_res.width) - _off.x;
    verts[i].y = (1.f - verts[i].w * 2.f) * _mag * static_cast<float>(__resolution.height) / static_cast<float>(_window_res.height) + _off.y;
  }
}

SurfaceRenderQuad::SurfaceRenderQuad(KernelsWrapper & _kers, Resolution const & _res, GLint _internal, GLenum _format, GLenum _type)
  : RenderQuad(_res, _internal, _format, _type)
  , __kernels(_kers)
  , __surface(nullptr)
{
  _res.print("\t\tResolution");
  bindTexture(resolution().width, resolution().height, nullptr);
  checkCudaErrors(cudaGraphicsGLRegisterImage(&__surface, id(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
  std::cout << "\t\tcudaGraphicsGLRegisterImage: " << __surface << std::endl;
}

cudaSurfaceObject_t SurfaceRenderQuad::createSurfaceObject() {
  cudaGraphicsMapResources(1, &__surface);
  cudaArray_t writeArray;
  cudaGraphicsSubResourceGetMappedArray(&writeArray, __surface, 0, 0);
  cudaResourceDesc wdsc;
  wdsc.resType = cudaResourceTypeArray;
  wdsc.res.array.array = writeArray;
  cudaSurfaceObject_t writeSurface;
  cudaCreateSurfaceObject(&writeSurface, &wdsc);
  return writeSurface;
}

void SurfaceRenderQuad::destroySurfaceObject(cudaSurfaceObject_t _writeSurface) {
  cudaDestroySurfaceObject(_writeSurface);
  cudaGraphicsUnmapResources(1, &__surface);
}

TextRenderQuad::TextRenderQuad()
  : RenderQuad(Resolution(), GL_RGBA, GL_BGRA, GL_UNSIGNED_BYTE)
  , __surface(nullptr)
{
}

void TextRenderQuad::setText(TTF_Font * _font, char const * _val) {
  if(*_val == 0) {
    // empty string -> 0 x 0 texture -> seg fault
    _val = " ";
  }
  SDL_FreeSurface(__surface);
  SDL_Color color = {255, 255, 255, 0};
  __surface = TTF_RenderText_Blended_Wrapped(_font, _val, color, 640);
  bindTexture(__surface->w, __surface->h, __surface->pixels);
}

void TextRenderQuad::updateQuad(Resolution _window_res, float _mag, float2 _off) {
  for(int i = 0; i < num_verts; ++i) {
    verts[i].x = -.9f + 1.8f * verts[i].z * static_cast<float>(__surface->w) / static_cast<float>(_window_res.width);
    verts[i].y = .9f - 1.8f * verts[i].w * static_cast<float>(__surface->h) / static_cast<float>(_window_res.height);
  }
}
