#include "fps.h"

struct FormatScope {
  FormatScope(std::ostream & os)
    : __os(os)
    , __temp(nullptr)
  {
    __temp.copyfmt(os);
  }
  ~FormatScope() {
    __os.copyfmt(__temp);
  }
private:
  std::ostream & __os;
  std::ios __temp;
};

FPS::FPS(float _fps)
  : __FPS_DISP_LERP(0.01f)
  , __FPS_LSF(0.1f)
  , __desired(_fps)
  , __max(_fps)
  , __actual(_fps)
  , __lastTicks(SDL_GetTicks())
  , __remainder(0.f)
{}

void FPS::reportCurrent(std::ostream & os) const {
  FormatScope temp(os);
  os.setf(std::ios::fixed, std:: ios::floatfield);
  os.precision(1);
  os << "fps: " << round(__actual) << " (" << round(__max) << ")";
}

void FPS::updateAndLimitWithDelay() {
  // Lerp max possible FPS before delay
  auto tickDiff = SDL_GetTicks() - __lastTicks;
  __max = clamp(lerp(__max, ticksToFps(tickDiff)));
  // If FPS > desired apply delay > 0 else delay == 0
  auto delay = std::max(fpsToTicks(__desired) - static_cast<float>(tickDiff) + __remainder, 0.0f);
  auto floor_delay = std::floor(delay);
  __remainder = delay - floor_delay;
  SDL_Delay(floor_delay);
  // lerp acutal FPS with delay
  auto nowTicks = SDL_GetTicks();
  __actual = clamp(lerp(__actual, ticksToFps(nowTicks - __lastTicks)));
  // start timing update
  __lastTicks = nowTicks;
}

float FPS::ticksToFps(float _val) {return 1000.f / std::max(_val, FLT_EPSILON);}

float FPS::fpsToTicks(float _val) {return ticksToFps(_val);}

float FPS::round(float _in) const {return floorf(_in / __FPS_LSF + 0.5f) * __FPS_LSF;}

float FPS::clamp(float _val) const {
  if(!std::isfinite(_val)) {
    return __desired;
  } else {
    return std::min(std::max(_val, 0.0f), 240.0f);
  }
}

float FPS::lerp(float _from, float _to) const {
  return (1.f - __FPS_DISP_LERP) * _from + __FPS_DISP_LERP * _to;
}
