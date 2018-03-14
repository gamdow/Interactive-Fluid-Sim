#include "interface.hpp"

#include <algorithm>
#include <cmath>


FPS::FPS(float _frame_rate)
  : __frame_rate(_frame_rate)
  , __fps_max(_frame_rate)
  , __fps_act(_frame_rate)
  , __time(SDL_GetTicks())
{}

void FPS::printCurrent(std::ostream & os) const {
  os << "fps: " << floorf(__fps_act * 10.f) / 10.f << " (" << floorf(__fps_max * 10.f) / 10.f << ")";
}

void FPS::update() {
  validate(__fps_max);
  validate(__fps_act);
  __fps_max = 0.99f * __fps_max + 0.01f * (1000.f / std::max(SDL_GetTicks() - __time, 1u));
  SDL_Delay(std::max(1000.0f / __frame_rate - (SDL_GetTicks() - __time), 0.0f));
  __fps_act = 0.99f * __fps_act + 0.01f * (1000.f / std::max(SDL_GetTicks() - __time, 1u));
  __time = SDL_GetTicks();
}

void FPS::validate(float & _val) {
  if(!std::isfinite(_val)) {
    _val = __frame_rate;
  } else {
    _val = std::min(std::max(_val, 0.0f), 1000.0f);
  }
}

OptionBase::OptionBase(char const * _name)
  : __name(_name)
  , __changed(false)
{}

void OptionBase::Update(SDL_Event const & event) {
  if(updateImpl(event)) {
    std::cout << __name << ": ";
    printCurrentImpl(std::cout);
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
void CycleOption<T>::printCurrentImpl(std::ostream & os) const {
  os << indexToName(__cur) << "(" << indexToVal(__cur) << ")";
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
void RangeOption<T>::printCurrentImpl(std::ostream & os) const {
  os << indexToVal(__cur);
}

template class CycleOption<int>;
template class RangeOption<float>;
