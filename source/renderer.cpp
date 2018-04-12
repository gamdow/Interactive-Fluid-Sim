#include "renderer.hpp"

#include <iostream>
#include <cuda_gl_interop.h>

#include "helper_cuda.h"

template<typename T>
RenderObject<T>::RenderObject(KernelsWrapper & _kers, Resolution const & _res, GLint _internal, GLenum _format, GLenum _type)
  : __kernels(_kers)
  , __resolution(_res)
  , __id(0u)
  , __surface(nullptr)
{
  _res.print("\tResolution");
  glGenTextures(1, &__id);
  glBindTexture(GL_TEXTURE_2D, __id); {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexImage2D(GL_TEXTURE_2D, 0, _internal, __resolution.width, __resolution.height, 0, _format, _type, nullptr);
  } glBindTexture(GL_TEXTURE_2D, 0);
  std::cout << "\t\tglGenTextures: " << __id << std::endl;
  checkCudaErrors(cudaGraphicsGLRegisterImage(&__surface, __id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
  std::cout << "\t\tcudaGraphicsGLRegisterImage: " << __surface << std::endl;
}

template<typename T>
void RenderObject<T>::render(Resolution _window_res, float _mag, float2 _off) {
  float width_prop = static_cast<float>(__resolution.width) / _window_res.width;
  float height_prop = static_cast<float>(__resolution.height) / _window_res.height;
  glBindTexture(GL_TEXTURE_2D, __id); {
      glBegin(GL_QUADS); {
        auto vf = [_mag, _off](float u, float v) {
          float2 vertex = (make_float2(u, 1.0f - v) * 2.0f - make_float2(1.0f)) * _mag + _off;
          glTexCoord2f(u, v);
          glVertex2f(vertex.x, vertex.y);
        };
        vf(0.0f, 0.0f);
        vf(1.0f, 0.0f);
        vf(1.0f, 1.0f);
        vf(0.0f, 1.0f);
      }
      glEnd();
  } glBindTexture(GL_TEXTURE_2D, 0);
}

OpenGLInitialiser::OpenGLInitialiser(Resolution _res)
  : __window(nullptr)
  , __context(nullptr)
  , __font(nullptr)
{
  if(SDL_Init(SDL_INIT_VIDEO) < 0) {
    ReportFailure();
    return;
  }

  _res.print("Creating SDL Window: Resolution");
  __window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _res.width, _res.height, SDL_WINDOW_OPENGL);
  if(__window == nullptr) {
    ReportFailure();
    return;
  }

  // 3.1 Needed for immediate mode (glBegin/End) rendering
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  __context = SDL_GL_CreateContext(__window);
  if(__context == nullptr) {
    ReportFailure();
    return;
  }

  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetSwapInterval(0); // V-Sync off for max speed

  std::cout << std::endl << "OpenGL: " << glGetString(GL_VERSION) << std::endl;

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_BLEND); // Need blending for text overlay
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if(TTF_Init() < 0) {
    std::cout << TTF_GetError() << std::endl;
    return;
  }

  __font = TTF_OpenFont("FreeSans.ttf", 24);
  if (__font == nullptr) {
    std::cout << "Missing Font" << std::endl;
    return;
  }
}

OpenGLInitialiser::~OpenGLInitialiser() {
  TTF_Quit();
  TTF_CloseFont(__font);
  SDL_GL_DeleteContext(__context);
  SDL_DestroyWindow(__window);
  SDL_Quit();
}

void OpenGLInitialiser::ReportFailure() const {std::cout << SDL_GetError() << std::endl;}

Renderer::Renderer(Resolution _res, Camera & _cam, KernelsWrapper & _kers)
  : OpenGLInitialiser(_res)
  , __windowRes(_res)
  , __kernels(_kers)
  , __background(_kers, _cam.resolution, GL_RGBA32F, GL_RGBA, GL_FLOAT)
  , __visualisation(_kers, _kers.getBufferRes(), GL_RGBA32F, GL_RGBA, GL_FLOAT)
  , __textTexture(0u)
  , __textSurface(nullptr)
{
  glGenTextures(1, &__textTexture);
  setText("");
}

Renderer::~Renderer() {
  SDL_FreeSurface(__textSurface);
}

void Renderer::setText(char const * _val) {
  if(*_val == 0) {
    // empty string -> 0 x 0 texture -> seg fault
    _val = " ";
  }
  SDL_FreeSurface(__textSurface);
  SDL_Color color = {255, 255, 255, 0}; // Red
  __textSurface = TTF_RenderText_Blended_Wrapped(getFont(), _val, color, 640);
  glBindTexture(GL_TEXTURE_2D, __textTexture); {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, __textSurface->w, __textSurface->h, 0, GL_BGRA, GL_UNSIGNED_BYTE, __textSurface->pixels);
  } glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::render(float _mag, float2 _off) {
  __visualisation.render(__windowRes, _mag, _off);

  glBindTexture(GL_TEXTURE_2D, __textTexture); {
    float x_prop = static_cast<float>(__textSurface->w) / __kernels.getBufferRes().width;
    float y_prop = static_cast<float>(__textSurface->h) / __kernels.getBufferRes().height;
    glBegin(GL_QUADS); {
      glTexCoord2f(0.0f, 0.0f); glVertex2f(-.9f, .9f);
      glTexCoord2f(1.0f, 0.0f); glVertex2f(-.9f + 1.8f * x_prop, .9f);
      glTexCoord2f(1.0f, 1.0f); glVertex2f(-.9f + 1.8f * x_prop, .9f - 1.8f * y_prop);
      glTexCoord2f(0.0f, 1.0f); glVertex2f(-.9f, .9f - 1.8f * y_prop);
    }
    glEnd();
  } glBindTexture(GL_TEXTURE_2D, 0);

  swapWindow();
}
