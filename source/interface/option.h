#pragma once

#include <ostream>
#include <vector>
#include <SDL2/SDL.h>

struct OptionBase {
  OptionBase(char const * _name);
  virtual ~OptionBase() {}
  void Update(SDL_Event const & event);
  inline void clearChangedFlag() {__changed = false;}
  inline bool hasChanged() const {return __changed;}
  char const * name() const {return __name;}
  void reportCurrent(std::ostream & os) const;
private:
  virtual bool updateImpl(SDL_Event const & event) = 0;
  virtual void __reportCurrent(std::ostream & os) const = 0;
  char const * __name;
  bool __changed;
};

// Cycles forward through named values
template<class T>
struct CycleOption : public OptionBase {
  CycleOption(char const * _name, SDL_Keycode _cycle);
  void insert(char const * _name, T _val);
  operator T() const {return indexToVal(__cur);}
private:
  std::string indexToName(int _i) const {return __namedVals[__cur].first;}
  T indexToVal(int _i) const {return __namedVals[__cur].second;}
  virtual bool updateImpl(SDL_Event const & event);
  virtual void __reportCurrent(std::ostream & os) const;
  typedef std::vector< std::pair<std::string, T> > Map;
  Map __namedVals;
  int __cur;
  SDL_Keycode __cycle;
};

struct BoolOption : public CycleOption<bool> {
  BoolOption(char const * _name, SDL_Keycode _cycle);
};

// steps forward and backward over range
template<class T>
struct RangeOption : public OptionBase {
  RangeOption(char const * _name, T _ini, T _min, T _max, int _num_steps, SDL_Keycode _up, SDL_Keycode _down);
  operator T() const {return indexToVal(__cur);}
private:
  int valToIndex(T _v) const;
  T indexToVal(int _i) const {return _i * __step + __min;}
  virtual bool updateImpl(SDL_Event const & event);
  virtual void __reportCurrent(std::ostream & os) const;
  T __min, __step;
  int __cur, __nSteps;
  SDL_Keycode __up;
  SDL_Keycode __down;
};

template class RangeOption<float>; typedef RangeOption<float> ScaleOption;
template class CycleOption<int>; typedef CycleOption<int> ModeOption;
