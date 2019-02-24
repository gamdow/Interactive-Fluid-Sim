#include "option.h"

#include <iostream>
#include <cassert>

OptionBase::OptionBase(char const * _name)
  : __name(_name)
  , __changed(false)
{}

void OptionBase::Update(SDL_Event const & event) {
  if(updateImpl(event)) {
    reportCurrent(std::cout);
    std::cout << std::endl;
    __changed = true;
  }
}

void OptionBase::reportCurrent(std::ostream & os) const {
  os << __name << ": ";
  __reportCurrent(os);
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
CycleOption<T> & CycleOption<T>::operator=(T const & _val) {
  for(auto it = __namedVals.begin(); it != __namedVals.end(); ++it) {
    if(_val == it->second) {
      __cur = std::distance(__namedVals.begin(), it);
      return *this;
    }
  }
  assert(false);
  return *this;
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
void CycleOption<T>::__reportCurrent(std::ostream & os) const {
  os << "(" << indexToVal(__cur) << ") " << indexToName(__cur);
}

BoolOption::BoolOption(char const * _name, SDL_Keycode _cycle)
  : CycleOption(_name, _cycle)
{
  insert("Off", false);
  insert("On", true);
}

BoolOption & BoolOption::operator=(bool const & _val) {
  CycleOption<bool>::operator=(_val);
  return *this;
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
RangeOption<T> & RangeOption<T>::operator=(T const & _val) {
  __cur = valToIndex(_val);
  return *this;
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
void RangeOption<T>::__reportCurrent(std::ostream & os) const {
  os << indexToVal(__cur);
}
