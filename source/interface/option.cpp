#include "option.h"

#include <iostream>

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
  , __step((_max - _min) / (_num_steps - 1))
  , __nSteps((_num_steps - 1))
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
