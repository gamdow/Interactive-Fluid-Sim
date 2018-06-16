#include "interface.hpp"

#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "simulation.hpp"
#include "opengl.hpp"

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
  : __desired(_fps)
  , __max(_fps)
  , __actual(_fps)
  , __lastTicks(SDL_GetTicks())
  , __remainder(0.f)
{}

void FPS::reportCurrent(std::ostream & os) const {
  FormatScope temp(os);
  os.setf(std::ios::fixed, std:: ios::floatfield);
  os.precision(1);
  os << "fps: " << floorf(__actual * 10.f) / 10.f << " (" << floorf(__max * 10.f) / 10.f << ")";
}

void FPS::updateAndDelay() {
  validate(__max, __desired);
  validate(__actual, __desired);
  auto tickDiff = SDL_GetTicks() - __lastTicks;
  lerp(__max, 1000.f / std::max(tickDiff, 1u), 0.01f);
  auto delay = std::max(1000.f / __desired - tickDiff + __remainder, 0.0f);
  auto floor_delay = std::floor(delay);
  __remainder = delay - floor_delay;
  SDL_Delay(floor_delay);
  auto nowTicks = SDL_GetTicks();
  lerp(__actual, 1000.f / std::max(nowTicks - __lastTicks, 1u), 0.01f);
  __lastTicks = nowTicks;
}

void FPS::validate(float & _val, float _default) {
  if(!std::isfinite(_val)) {
    _val = _default;
  } else {
    _val = std::min(std::max(_val, 0.0f), 1000.0f);
  }
}

void FPS::lerp(float & _val, float _new, float _lerp) {
  _val = (1.f - _lerp) * _val + _lerp * _new;
}

OptionBase::OptionBase(char const * _name)
  : __name(_name)
  , __changed(false)
{}

void OptionBase::Update(SDL_Event const & event) {
  if(updateImpl(event)) {
    std::cout << __name << ": ";
    reportCurrentImpl(std::cout);
    std::cout << std::endl;
    __changed = true;
  }
}

template<class T>
CycleOption<T>::CycleOption(char const * _name, SDL_Keycode _cycle)
  : OptionBase(_name)
  , __cur(-1)
  , __cycle(_cycle)
{
}

template<class T>
void CycleOption<T>::insert(char const * _name, T _val) {
  __namedVals.push_back(std::pair<std::string, T>(_name, _val));
  if(__cur == -1) __cur = 0;
}

template<class T>
bool CycleOption<T>::updateImpl(SDL_Event const & event) {
  if(event.type == SDL_KEYUP && event.key.keysym.sym == __cycle) {
    __cur = (__cur + 1) % __namedVals.size();
    return true;
  }
  return false;
}

template<class T>
void CycleOption<T>::reportCurrentImpl(std::ostream & os) const {
  os << "(" << indexToVal(__cur) << ") " << indexToName(__cur);
}

template<class T>
RangeOption<T>::RangeOption(char const * _name, T _ini, T _min, T _max, int _num_steps, SDL_Keycode _up, SDL_Keycode _down)
  : OptionBase(_name)
  , __min(_min)
  , __nSteps(_num_steps)
  , __step((_max - _min) / _num_steps)
  , __up(_up)
  , __down(_down)
{
  __cur = valToIndex(_ini);
}

template<class T>
int RangeOption<T>::valToIndex(T _v) const {
  auto f = std::max(0.0f, std::floor((_v - __min) / __step + 0.5f));
  return std::min(__nSteps, static_cast<int>(f));
}

template<class T>
bool RangeOption<T>::updateImpl(SDL_Event const & event) {
  int last = __cur;
  if(event.type == SDL_KEYDOWN) {
    SDL_Keycode const key = event.key.keysym.sym;
    if(key == __up) {
      if(__cur < __nSteps) ++__cur;
    } else if(key == __down) {
      if(__cur > 0) --__cur;
    }
  }
  return last != __cur;
}

template<class T>
void RangeOption<T>::reportCurrentImpl(std::ostream & os) const {
  os << indexToVal(__cur);
}

Interface::Interface(OpenGL & _opengl, float _fps)
  : __opengl(_opengl)
  , __fps(_fps)
  , __vel_multiplier("Velocity Multiplier", 1.0f, 0.1f, 10.0f, 101, SDLK_r, SDLK_f)
  , __magnification("Magnification", 1.0f, 1.0f, 4.0f, 31, SDLK_w, SDLK_s)
  , __offset_x("Offset X-Axis", 0.0f, -1.0f, 1.0f, 21, SDLK_RIGHT, SDLK_LEFT)
  , __offset_y("Offset Y-Axis", 0.0f, -1.0f, 1.0f, 21, SDLK_DOWN, SDLK_UP)
  , __mode("Visualisation Mode", SDLK_1)
{
  __options.push_back(&__vel_multiplier);
  __options.push_back(&__magnification);
  __options.push_back(&__offset_x);
  __options.push_back(&__offset_y);
  __mode.insert("Smoke", Mode::smoke);
  __mode.insert("Velocity Field", Mode::velocity);
  __mode.insert("Divergence", Mode::divergence);
  __mode.insert("Pressure", Mode::pressure);
  __mode.insert("Fluid", Mode::fluid);
  __options.push_back(&__mode);
}

void Interface::resetFlags() {
  for(auto i = __options.begin(); i != __options.end(); ++i) {
    (*i)->clearChangedFlag();
  }
}

void Interface::updateInputs(SDL_Event const & event) {
  for(auto i = __options.begin(); i != __options.end(); ++i) {
    (*i)->Update(event);
  }
}

void Interface::__render(Resolution const & _window_res, float _mag, float2 _off) {
  std::stringstream os_text;
  os_text.setf(std::ios::fixed, std:: ios::floatfield);
  os_text.precision(2);
  __fps.reportCurrent(os_text);
  os_text << std::endl;
  __mode.reportCurrent(os_text);
  __quad.setText(__opengl.getFont(), os_text.str().c_str());
  __quad.render(_window_res, _mag, _off);
}
