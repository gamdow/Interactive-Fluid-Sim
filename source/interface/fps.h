#pragma once

#include <ostream>
//#include <vector>
#include <SDL2/SDL.h>

struct FPS {
  FPS(float _fps);
  void reportCurrent(std::ostream & os) const;
  void updateAndLimitWithDelay();
private:
  static float ticksToFps(float);
  static float fpsToTicks(float);
  float round(float) const;
  float clamp(float) const;
  float lerp(float _from, float _to) const;
  float const __FPS_DISP_LERP;
  float const __FPS_LSF;
  float __desired, __max, __actual;
  Uint32 __lastTicks;
  float __remainder;
};
