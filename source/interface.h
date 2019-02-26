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
  float fpsDelta() const {return __fps.fpsDelta();}
  std::string screenText() const;
  ScaleOption & velocity() {return __vel_multiplier;}
  ScaleOption & magnification() {return __magnification;}
  float magnification() const {return __magnification;}
  float2 offset() const;
  ScaleOption & filterValue() {return __filter_value;}
  ScaleOption & filterRange() {return __filter_range;}
  ModeOption & filterMode() {return __filter_mode;}
  BoolOption & bgSubtract() {return __bg_subtract;}
  ModeOption & mode() {return __mode;}
  BoolOption & debugMode() {return __debug_mode;}
  bool debugMode() const {return __debug_mode;}
  BoolOption & mirrorCam() {return __mirror_cam;}
  ModeOption & flowRotate() {return __flow_rotate;}
  bool modeChangedRecently() const {return hasChangedRecently(__mode_show_until);}
  bool filterChangedRecently() const {return hasChangedRecently(__filter_show_until);}
private:
  bool hasChangedRecently(Uint32 _timeout) const {return SDL_GetTicks() < _timeout;}
  FPS __fps;
  ScaleOption __vel_multiplier, __magnification, __offset_x, __offset_y, __filter_value, __filter_range;
  ModeOption __filter_mode, __mode, __flow_rotate;
  BoolOption __debug_mode, __bg_subtract, __mirror_cam;
  std::vector<OptionBase*> __options;
  static Uint32 const FPS_SHOW_DURATION = 10000; // ms
  static Uint32 const DEBUG_SHOW_DURATION = 3000; // ms
  Uint32 __fps_show_until, __mode_show_until, __filter_show_until;
};
