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
  //FPS const & fps() const {return __fps;}
  float velocity() const {return __vel_multiplier;}
  float magnification() const {return __magnification;}
  float2 offset() const;
  float filterValue() const {return __filter_value;}
  float filterRange() const {return __filter_range;}
  int filterMode() const {return __filter_mode;}
  bool bgSubtract() const {return __bg_subtract;}
  int mode() const {return __mode;}
  std::string screenText() const;
  bool debugMode() const {return __debug_mode;}
  bool modeChangedRecently() const {return hasChangedRecently(__mode_show_until);}
  bool filterChangedRecently() const {return hasChangedRecently(__filter_show_until);}
private:
  bool hasChangedRecently(Uint32 _timeout) const {return SDL_GetTicks() < _timeout;}
  FPS __fps;
  ScaleOption __vel_multiplier, __magnification, __offset_x, __offset_y, __filter_value, __filter_range;
  ModeOption __filter_mode, __mode;
  BoolOption __debug_mode, __bg_subtract;
  std::vector<OptionBase*> __options;
  static Uint32 const DEBUG_SHOW_DURATION = 3000; // ms
  Uint32 __mode_show_until, __filter_show_until;
};
