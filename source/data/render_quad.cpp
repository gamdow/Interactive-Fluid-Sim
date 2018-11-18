#include "render_quad.h"

#include <iostream>
#include <cuda_gl_interop.h>
#include <SDL2/SDL_ttf.h>

#include "../debug.h"
#include "../cuda/helper_cuda.h"
#include "../cuda/utility.h"
#include "../i_render_settings.h"

#include <iostream>

RenderQuad::RenderQuad(IRenderSettings const & _render_settings, GLint _internal, GLenum _format, GLenum _type)
  : __settings(_render_settings)
  , __id(0u)
  , __internal(_internal)
  , __format(_format)
  , __type(_type)
{
  verts[0] = make_float4(-1.f, 1.f, 0.f, 0.f);
  verts[1] = make_float4(1.f, 1.f, 1.f, 0.f);
  verts[2] = make_float4(1.f, -1.f, 1.f, 1.f);
  verts[3] = make_float4(-1.f, -1.f, 0.f, 1.f);
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

void RenderQuad::updateVerts() {
  auto m = __settings.magnification();
  auto o = __settings.offset();
  auto s = scale();
  for(int i = 0; i < num_verts; ++i) {
    verts[i].x = (verts[i].z * 2.f - 1.f) * m * s.x - o.x;
    verts[i].y = (1.f - verts[i].w * 2.f) * m * s.y - o.y;
  }
}

void RenderQuad::renderVerts() {
  glBindTexture(GL_TEXTURE_2D, __id); {
    glBegin(GL_QUADS); {
      for(int i = 0; i < 4; ++i) {
        float4 const & vert = verts[i];
        glTexCoord2f(vert.z, vert.w); glVertex2f(vert.x, vert.y);
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

void TextRenderQuad::setText(char const * _val) {
  if(*_val == 0) {
    // empty string -> 0 x 0 texture -> seg fault
    _val = " ";
  }
  SDL_FreeSurface(__surface);
  SDL_Color color = {255, 255, 255, 0};
  __surface = TTF_RenderText_Blended_Wrapped(renderSettings().font(), _val, color, 640);
  bindTexture(__surface->w, __surface->h, __surface->pixels);
}

void TextRenderQuad::__render() {
  auto s = scale();
  for(int i = 0; i < num_verts; ++i) {
    verts[i].x = -.9f + 1.8f * verts[i].z * s.x;
    verts[i].y = .9f - 1.8f * verts[i].w * s.y;
  }
  renderVerts();
}
