#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "interface/fps.h"
#include "interface/option.h"
#include "i_renderer.h"
#include "i_renderable.h"

struct Interface {
  Interface(float _fps);
  void resetFlags();
  void updateInputs(SDL_Event const & event);
  void updateAndLimitFps() {__fps.updateAndLimitWithDelay();}
  FPS const & fps() const {return __fps;}
  ScaleOption const & velocity() const {return __vel_multiplier;}
  ScaleOption const & magnification() const {return __magnification;}
  float2 offset() const;
  ScaleOption const & filterThreshold() const {return __filter_threshold;}
  ModeOption const & mode() const {return __mode;}
private:
  FPS __fps;
  ScaleOption __vel_multiplier;
  ScaleOption __magnification;
  ScaleOption __offset_x;
  ScaleOption __offset_y;
  ScaleOption __filter_threshold;
  ModeOption __mode;
  std::vector<OptionBase*> __options;
};

struct InterfaceRenderer : public IRenderable {
  InterfaceRenderer(Interface const & _interface, IRenderer & _renderer);
private:
  virtual void __render();
  Interface const & __interface;
  ITextRenderTarget & __renderTarget;
};
