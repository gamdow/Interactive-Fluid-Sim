#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <algorithm>

#include "kernels.cuh"

float2 const SIZE = {1.6f, 0.9f};
int2 const DIMS = {BLOCK_SIZE.x * 20, BLOCK_SIZE.y * 20};
float2 const D_X = {SIZE.x / DIMS.x, SIZE.y / DIMS.y};
int const FRAME_RATE = 60;

void quit(int _code, char const * _message);
void init(int2 _dims, SDL_Window * & o_window, SDL_GLContext & o_glContext);
void genRenderTexture(int2 _dims,  GLuint & o_texture);
void render(GLuint _viewGLTexture, SDL_Window * & _window, float _mul);
void advanceSim(cudaGraphicsResource_t & _viewCudaResource);

template<class T>
struct MirroredArray {
  MirroredArray(int _size)
      : size(_size)
  {
    host = new T [size];
    memset(host, 0, size * sizeof(T));
    cudaMalloc((void **) & device, size * sizeof(T));
    cudaMemset(device, 0, size * sizeof(T));
  }
  ~MirroredArray() {
    delete [] host;
    cudaFree(device);
  }
  T & operator[](int i) {
    return host[i];
  }
  void copyHostToDevice() {
    cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice);
  }
  int size;
  T * host;
  T * device;
};

void applyBoundary(int2 _dims, float _vel, MirroredArray<float2> & io_velocity, MirroredArray<float> & io_fluidCells) {

  for(int j = 0; j < _dims.y; ++j) {
    for(int i = 0; i < 50; ++i) {
      io_velocity[i + j * _dims.x] = make_float2(_vel, 0.0f);
    }
    for(int i = _dims.x - 50; i < _dims.x; ++i) {
      io_velocity[i + j * _dims.x] = make_float2(_vel, 0.0f);
    }
  }

  for(int i = 0; i < _dims.x; ++i) {
    for(int j = 0; j < 1; ++j) {
      io_velocity[i + j * _dims.x] = make_float2(0.0f, 0.0f);
      io_fluidCells[i + j * _dims.x] = 0.0f;
    }
    for(int j = _dims.y - 1; j < _dims.y; ++j) {
      io_velocity[i + j * _dims.x] = make_float2(0.0f, 0.0f);
      io_fluidCells[i + j * _dims.x] = 0.0f;
    }
  }

  io_velocity.copyHostToDevice();
  io_fluidCells.copyHostToDevice();
}

int main(int argc, char * argv[]) {

  SDL_Window * window = nullptr;
  SDL_GLContext glContext = nullptr;
  init(DIMS, window, glContext);
  GLuint viewGLTexture; genRenderTexture(DIMS, viewGLTexture);
  cudaGraphicsResource_t viewCudaResource;
  cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

  int2 const BUFFERED_DIMS = {DIMS.x + 2, DIMS.y + 2};
  int const BUFFERED_SIZE = BUFFERED_DIMS.x * BUFFERED_DIMS.y;

  MirroredArray<float2> velocity(BUFFERED_SIZE);
  MirroredArray<float> fluidCells(BUFFERED_SIZE);

  float * divergence = NULL; cudaMalloc((void **) & divergence, BUFFERED_SIZE * sizeof(float)); cudaMemset(divergence, 0, BUFFERED_SIZE * sizeof(float));
  float * pressure = NULL; cudaMalloc((void **) & pressure, BUFFERED_SIZE * sizeof(float)); cudaMemset(pressure, 0, BUFFERED_SIZE * sizeof(float));
  float * buffer = NULL; cudaMalloc((void **) & buffer, BUFFERED_SIZE * sizeof(float)); cudaMemset(buffer, 0, BUFFERED_SIZE * sizeof(float));

  float2 center = make_float2(BUFFERED_DIMS.x / 2.0f, BUFFERED_DIMS.y / 2.0f);
  for(int i = 0; i < BUFFERED_DIMS.x; ++i) {
    for(int j = 0; j < BUFFERED_DIMS.y; ++j) {
      fluidCells[i + j * BUFFERED_DIMS.x] = (center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 1000 ? 0.0f : 1.0f;
    }
  }

  float init_velocity = 1.0f;
  applyBoundary(BUFFERED_DIMS, init_velocity, velocity, fluidCells);

  kernels_init(DIMS);
  Uint32 time = SDL_GetTicks();
  SDL_Event event;

  float vis_multiplier = 1.0f;
  float size_multiplier = 1.0f;
  float velocity_multiplier = 1.0f;

  bool stop = false;
  while(true) {
    while(SDL_PollEvent(&event)) {
      switch(event.type) {
        case SDL_KEYUP:
          switch(event.key.keysym.sym) {
            case SDLK_q:
              size_multiplier *= 2.0f;
              std::cout << "size_multiplier: " << size_multiplier << std::endl;
              break;
            case SDLK_a:
              size_multiplier /= 2.0f;
              std::cout << "size_multiplier: " << size_multiplier << std::endl;
              break;
            case SDLK_e:
              vis_multiplier *= 2.0f;
              std::cout << "vis_multiplier: " << vis_multiplier << std::endl;
              break;
            case SDLK_d:
              vis_multiplier /= 2.0f;
              std::cout << "vis_multiplier: " << vis_multiplier << std::endl;
              break;
            case SDLK_w:
              velocity_multiplier *= 2.0f;
              std::cout << "velocity_multiplier: " << velocity_multiplier << std::endl;
              applyBoundary(BUFFERED_DIMS, init_velocity * velocity_multiplier, velocity, fluidCells);
              cudaMemset(pressure, 0, BUFFERED_SIZE * sizeof(float));
              break;
            case SDLK_s:
              velocity_multiplier /= 2.0f;
              std::cout << "velocity_multiplier: " << velocity_multiplier << std::endl;
              applyBoundary(BUFFERED_DIMS, init_velocity * velocity_multiplier, velocity, fluidCells);
              cudaMemset(pressure, 0, BUFFERED_SIZE * sizeof(float));
              break;
            default:
              stop = true;
          }
          break;
        case SDL_QUIT:
          stop = true;
        default:
          break;
      }
    }

    if(stop) break;

    copy_to_surface(velocity.device, DIMS, viewCudaResource);
    render(viewGLTexture, window, size_multiplier);

    // std::cout << "Before" << std::endl;
    // for(int i = 0; i < BUFFERED_SIZE; ++i) {
    //   std::cout << i << " " << cpu_velocity[i].x << " " << cpu_velocity[i].y << std::endl;
    // }

    simulation_step(DIMS, D_X, 1.0f / FRAME_RATE, fluidCells.device, velocity.device, divergence, pressure, buffer);
    //cudaMemcpy(cpu_velocity, velocity, BUFFERED_SIZE * sizeof(float2), cudaMemcpyDeviceToHost);

    // std::cout << "After" << std::endl;
    // for(int i = 0; i < BUFFERED_SIZE; ++i) {
    //   std::cout << i << " " << cpu_velocity[i].x << " " << cpu_velocity[i].y << std::endl;
    // }
    int elasped = SDL_GetTicks() - time;
    SDL_Delay(std::max(1000 / FRAME_RATE - elasped, 0));
    time = SDL_GetTicks();
  }
  kernels_shutdown();

  cudaFree(pressure);
  cudaFree(buffer);
  cudaFree(divergence);

  SDL_GL_DeleteContext(glContext);
  SDL_DestroyWindow(window);
  quit(0, "");
}

void init(int2 _dims, SDL_Window * & o_window, SDL_GLContext & o_glContext) {
  if(SDL_Init(SDL_INIT_VIDEO) < 0) {
    quit(1, SDL_GetError());
  }

  o_window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _dims.x, _dims.y, SDL_WINDOW_OPENGL);
  if(o_window == nullptr) {
    quit(1, SDL_GetError());
  }

  // **TODO** require OpenGL 4.5, but setting it causes the texture not to render. Bad news for portability.
  //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetSwapInterval(1); // V-Sync

  o_glContext = SDL_GL_CreateContext(o_window);
  if(o_glContext == nullptr) {
    quit(1, "Unable to create OpenGL context.");
  }

  std::cout << glGetString(GL_VERSION) << std::endl;

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);

  glEnable(GL_TEXTURE_2D);
}

void genRenderTexture(int2 _dims, GLuint & o_texture) {
  glGenTextures(1, &o_texture);
  glBindTexture(GL_TEXTURE_2D, o_texture); {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _dims.x, _dims.y, 0, GL_RGBA, GL_FLOAT, nullptr);
  } glBindTexture(GL_TEXTURE_2D, 0);
}

void render(GLuint _viewGLTexture, SDL_Window * & _window, float _mul) {
  glBindTexture(GL_TEXTURE_2D, _viewGLTexture); {
      glBegin(GL_QUADS);
      {

          glTexCoord2f(0.0f, 0.0f); glVertex2f(-_mul, -_mul);
          glTexCoord2f(1.0f, 0.0f); glVertex2f(+_mul, -_mul);
          glTexCoord2f(1.0f, 1.0f); glVertex2f(+_mul, +_mul);
          glTexCoord2f(0.0f, 1.0f); glVertex2f(-_mul, +_mul);
      }
      glEnd();
  } glBindTexture(GL_TEXTURE_2D, 0);

  glFinish();

  SDL_GL_SwapWindow(_window);
}

void quit(int _code, char const * _message) {
  SDL_Quit();
  std::cout << _message;
  exit(_code);
}
