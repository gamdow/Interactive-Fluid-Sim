#include "interface.h"

#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "cuda/helper_math.h"

#include "simulation.h"

Interface::Interface(float _fps)
  : __fps(_fps)
  , __vel_multiplier("Velocity Multiplier", 1.0f, 0.1f, 10.0f, 101, SDLK_r, SDLK_f)
  , __magnification("Magnification", 1.0f, 1.0f, 4.0f, 31, SDLK_w, SDLK_s)
  , __offset_x("Offset X-Axis", 0.0f, -1.0f, 1.0f, 21, SDLK_RIGHT, SDLK_LEFT)
  , __offset_y("Offset Y-Axis", 0.0f, -1.0f, 1.0f, 21, SDLK_DOWN, SDLK_UP)
  , __filter_value("Filter Value", 1.0f, 0.0f, 1.0f, 101, SDLK_u, SDLK_j)
  , __filter_range("Filter Range", 0.75f, 0.0f, 1.0f, 101, SDLK_i, SDLK_k)
  , __filter_mode("Filter Mode", SDLK_m)
  , __mode("Visualisation Mode", SDLK_1)
  , __mode_show_until(SDL_GetTicks())
  , __filter_show_until(SDL_GetTicks())
{
  __options.push_back(&__vel_multiplier);
  __options.push_back(&__magnification);
  __options.push_back(&__offset_x);
  __options.push_back(&__offset_y);
  __options.push_back(&__filter_value);
  __options.push_back(&__filter_range);
    __filter_mode.insert("Hue", FilterMode::HUE);
    __filter_mode.insert("Saturation", FilterMode::SATURATION);
    __filter_mode.insert("Lightness", FilterMode::LIGHTNESS);
    __filter_mode.insert("BG Sub (Trained)", FilterMode::BG_SUBTRACT_TRAINED);
  __options.push_back(&__filter_mode);
    __mode.insert("Smoke", Mode::smoke);
    __mode.insert("Velocity Field", Mode::velocity);
    __mode.insert("Divergence", Mode::divergence);
    __mode.insert("Pressure", Mode::pressure);
    __mode.insert("Fluid", Mode::fluid);
  __options.push_back(&__mode);
}

void Interface::resetFlags() {
  updateChanges();
  for(auto i = __options.begin(); i != __options.end(); ++i) {
    (*i)->clearChangedFlag();
  }
}

void Interface::updateInputs(SDL_Event const & event) {
  for(auto i = __options.begin(); i != __options.end(); ++i) {
    (*i)->Update(event);
  }
}

void Interface::updateChanges() {
  Uint32 current_ticks = SDL_GetTicks();
  if(__mode.hasChanged()) {
    __mode_show_until = current_ticks + DEBUG_SHOW_DURATION;
  }
  if(__filter_mode.hasChanged() || __filter_value.hasChanged() || __filter_range.hasChanged()) {
    __filter_show_until = current_ticks + DEBUG_SHOW_DURATION;
  }
}

float2 Interface::offset() const {
  return make_float2(__offset_x, __offset_y) * (__magnification - 1.0f);
}

std::string Interface::screenText() const {
  std::stringstream os_text;
  os_text.setf(std::ios::fixed, std:: ios::floatfield);
  os_text.precision(2);
  __fps.reportCurrent(os_text);
  if(modeChangedRecently()) {
    os_text << std::endl;
    __mode.reportCurrent(os_text);
  }
  if(filterChangedRecently()) {
    os_text << std::endl;
    __filter_mode.reportCurrent(os_text);
  }
  return os_text.str();
}
