#include "interface.h"

#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "cuda/helper_math.h"

#include "simulation.h"

Interface::Interface(float _fps)
  : __fps(_fps)
  , __vel_multiplier("Velocity Multiplier (-/=)", 1.0f, 0.1f, 2.0f, 191, SDLK_EQUALS, SDLK_MINUS)
  , __magnification("Magnification (pgup/pgdn)", 1.0f, 1.0f, 4.0f, 31, SDLK_PAGEUP, SDLK_PAGEDOWN)
  , __offset_x("Offset X-Axis (left/right)", 0.0f, -1.0f, 1.0f, 21, SDLK_RIGHT, SDLK_LEFT)
  , __offset_y("Offset Y-Axis (up/down)", 0.0f, -1.0f, 1.0f, 21, SDLK_UP, SDLK_DOWN)
  , __filter_value("Filter Value (\'/#)", 1.0f, 0.0f, 1.0f, 101, SDLK_HASH, SDLK_QUOTE)
  , __filter_range("Filter Range ([/])", 0.75f, 0.0f, 1.0f, 101, SDLK_RIGHTBRACKET, SDLK_LEFTBRACKET)
  , __filter_mode("(F)ilter Mode", SDLK_f)
  , __mirror_cam("(M)irror Camera", SDLK_m)
  , __mode("(V)isualisation Mode", SDLK_v)
  , __debug_mode("(D)ebug Mode", SDLK_d)
  , __bg_subtract("Filter (B)ackground Subtract", SDLK_b)
  , __flow_rotate("(R)otate Flow", SDLK_r)
  , __mode_show_until(SDL_GetTicks())
  , __filter_show_until(SDL_GetTicks())

{
  __options.push_back(&__debug_mode);
  __options.push_back(&__mode); {
    __mode.insert("Smoke", Mode::smoke);
    __mode.insert("Velocity Field", Mode::velocity);
    __mode.insert("Divergence", Mode::divergence);
    __mode.insert("Pressure", Mode::pressure);
    //__mode.insert("Fluid", Mode::fluid);
  }
  __options.push_back(&__flow_rotate); {
    __flow_rotate.insert("Left to Right", FlowDirection::LEFT_TO_RIGHT);
    __flow_rotate.insert("Top to Bottom", FlowDirection::TOP_TO_BOTTOM);
    __flow_rotate.insert("Right to Left", FlowDirection::RIGHT_TO_LEFT);
    __flow_rotate.insert("Bottom to Top", FlowDirection::BOTTOM_TO_TOP);
  }
  __options.push_back(&__mirror_cam);
  __options.push_back(&__filter_mode); {
    __filter_mode.insert("Hue", FilterMode::HUE);
    __filter_mode.insert("Saturation", FilterMode::SATURATION);
    __filter_mode.insert("Lightness", FilterMode::LIGHTNESS);
  }
  __options.push_back(&__bg_subtract);
  __options.push_back(&__filter_range);
  __options.push_back(&__filter_value);
  __options.push_back(&__vel_multiplier);
  __options.push_back(&__magnification);
  __options.push_back(&__offset_x);
  __options.push_back(&__offset_y);
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
  if(__filter_mode.hasChanged() || __bg_subtract.hasChanged() || __filter_value.hasChanged() || __filter_range.hasChanged()) {
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
  os_text << "  ";
  __debug_mode.reportCurrent(os_text);
  if(debugMode()) {
    for(auto i = __options.begin(); i != __options.end(); ++i) {
      if(*i != &__debug_mode) {
        os_text << std::endl;
        (*i)->reportCurrent(os_text);
      }
    }
  } else {
    if(modeChangedRecently()) {
      os_text << std::endl;
      __mode.reportCurrent(os_text);
    }
    if(filterChangedRecently()) {
      os_text << std::endl;
      __filter_mode.reportCurrent(os_text);
      os_text << std::endl;
      __bg_subtract.reportCurrent(os_text);
    }
  }
  return os_text.str();
}
