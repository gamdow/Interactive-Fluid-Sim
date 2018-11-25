#include "render_quad.h"

#include <iostream>
#include <cuda_gl_interop.h>
#include <SDL2/SDL_ttf.h>

#include "../debug.h"
#include "../cuda/helper_cuda.h"
#include "../i_render_settings.h"

#include <iostream>

RenderQuad::RenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type)
  : __settings(_render_settings)
  , __id(0u)
  , __internal(_internal)
  , __format(_format)
  , __type(_type)
{
  __verts[0] = make_float2(-1.f, 1.f);
  __verts[1] = make_float2(1.f, 1.f);
  __verts[2] = make_float2(1.f, -1.f);
  __verts[3] = make_float2(-1.f, -1.f);
  flipUVs(false, false);
  glGenTextures(1, &__id);
  assert(__id != 0);
  OutputIndent indent;
  format_out << "glGenTextures: " << __id << std::endl;
}

float2 RenderQuad::scale() const {
   return make_float2(static_cast<float>(__resolution.width) / static_cast<float>(__settings.resolution().width), static_cast<float>(__resolution.height) / static_cast<float>(__settings.resolution().height));
 }

void RenderQuad::bindTexture(GLsizei _width, GLsizei _height, GLvoid const * _data) {
  __resolution = Resolution(_width, _height);
  glBindTexture(GL_TEXTURE_2D, __id); {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, __internal, _width, _height, 0, __format, __type, _data);
  } glBindTexture(GL_TEXTURE_2D, 0);
}

void RenderQuad::setVerts(QuadArray const & _verts) {
  QuadArray & v = verts();
  for(int i = 0; i < NUM_VERTS; ++i) {
      v[i] = _verts[i];
  }
}

void RenderQuad::flipUVs(bool _horizontal, bool _vertical) {
  int a, b, c, d;
  if(_horizontal) {
    if(_vertical) {
      a = 2; b = 3; c = 0; d = 1;
    } else {
      a = 1; b = 0; c = 3; d = 2;
    }
  } else {
    if(_vertical) {
      a = 3; b = 2; c = 1; d = 0;
    } else {
      a = 0; b = 1; c = 2; d = 3;
    }
  }
  __uvs[a] = make_float2(0.f, 0.f);
  __uvs[b] = make_float2(1.f, 0.f);
  __uvs[c] = make_float2(1.f, 1.f);
  __uvs[d] = make_float2(0.f, 1.f);
}

float2 RenderQuad::scaleVerts(float2 _vert, float2 _uv, float _mag, float2 _scale, float2 _offset) const {
  return _vert * _mag * _scale - _offset;
}

void RenderQuad::render() {
  auto mag = __settings.magnification();
  auto off = __settings.offset();
  auto scl = scale();
  glBindTexture(GL_TEXTURE_2D, __id); {
    glBegin(GL_QUADS); {
      for(int i = 0; i < NUM_VERTS; ++i) {
        float2 vert = scaleVerts(__verts[i], __uvs[i], mag, scl, off);
        glTexCoord2f(__uvs[i].x, __uvs[i].y); glVertex2f(vert.x, vert.y);
      }
    } glEnd();
  } glBindTexture(GL_TEXTURE_2D, 0);
}

SurfaceRenderQuad::SurfaceRenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type, Resolution const & _res)
  : RenderQuad(_render_settings, _internal, _format, _type)
  , __resource(nullptr)
  , __surface(0u)
{
  OutputIndent indent;
  _res.print("Resolution");
  bindTexture(_res.width, _res.height, nullptr);
  checkCudaErrors(cudaGraphicsGLRegisterImage(&__resource, id(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
  format_out << "cudaGraphicsGLRegisterImage: " << __resource << std::endl;
  cudaGraphicsMapResources(1, &__resource);
  cudaArray_t writeArray;
  cudaGraphicsSubResourceGetMappedArray(&writeArray, __resource, 0, 0);
  cudaResourceDesc wdsc;
  wdsc.resType = cudaResourceTypeArray;
  wdsc.res.array.array = writeArray;
  cudaCreateSurfaceObject(&__surface, &wdsc);
}

SurfaceRenderQuad::~SurfaceRenderQuad() {
  cudaDestroySurfaceObject(__surface);
  cudaGraphicsUnmapResources(1, &__resource);
}

void SurfaceRenderQuad::setSurfaceData(SurfaceWriter const & _writer) {
  _writer.writeToSurface(__surface, resolution());
}

// void SurfaceRenderQuad::setSurfaceData(OptimalBlockConfig const & _block_config, float const * _buffer, Resolution const & _res) {
//   copyToSurface(_block_config, __surface, resolution(), _buffer, _res);
// }
//
// void SurfaceRenderQuad::setSurfaceData(OptimalBlockConfig const & _block_config, unsigned char const * _buffer, Resolution const & _res) {
//   copyToSurface(_block_config, __surface, resolution(), _buffer, _res);
// }

float const TextRenderQuad::SAFE_SCALE = 0.9f;

TextRenderQuad::TextRenderQuad(IRenderSettings const & _render_settings)
  : RenderQuad(_render_settings, GL_RGBA, GL_BGRA, GL_UNSIGNED_BYTE)
  , __surface(nullptr)
{
  QuadArray & v = verts();
  for(int i = 0; i < NUM_VERTS; ++i) {
      v[i] = make_float2(-1.0f, 1.0f) * SAFE_SCALE;
  }
}

void TextRenderQuad::setText(char const * _val) {
  if(*_val == 0) {
    // empty string -> 0 x 0 texture -> seg fault
    _val = " ";
  }
  SDL_FreeSurface(__surface);
  SDL_Color color = {255, 255, 128, 0};
  __surface = TTF_RenderText_Blended_Wrapped(renderSettings().font(), _val, color, 640);
  bindTexture(__surface->w, __surface->h, __surface->pixels);
}


float2 TextRenderQuad::scaleVerts(float2 _vert, float2 _uv, float _mag, float2 _scale, float2 _offset) const {
  return _vert + _uv * _scale * make_float2(2.f, -2.f) * SAFE_SCALE;
}
