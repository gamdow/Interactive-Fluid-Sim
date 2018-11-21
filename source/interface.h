#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "interface/enums.h"
#include "interface/fps.h"
#include "interface/option.h"

struct Interface {
  Interface(float _fps);
  void resetFlags();
  void updateInputs(SDL_Event const & event);
  void updateChanges();
  void updateAndLimitFps() {__fps.updateAndLimitWithDelay();}
  FPS const & fps() const {return __fps;}
  ScaleOption const & velocity() const {return __vel_multiplier;}
  ScaleOption const & magnification() const {return __magnification;}
  float2 offset() const;
  float filterValue() const {return __filter_value;}
  float filterRange() const {return __filter_range;}
  int filterMode() const {return __filter_mode;}
  ModeOption const & mode() const {return __mode;}
  std::string screenText() const;
  bool modeChangedRecently() const {return SDL_GetTicks() < __mode_show_until;}
  bool filterChangedRecently() const {return SDL_GetTicks() < __filter_show_until;}
private:
  
  FPS __fps;
  ScaleOption __vel_multiplier, __magnification, __offset_x, __offset_y, __filter_value, __filter_range;
  ModeOption __filter_mode, __mode;
  std::vector<OptionBase*> __options;
  static Uint32 const DEBUG_SHOW_DURATION = 3000; // ms
  Uint32 __mode_show_until, __filter_show_until;
};
