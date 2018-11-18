#include "interface.h"

#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "cuda/helper_math.h"

#include "simulation.h"
#include "interface.h"

Interface::Interface(float _fps)
  : __fps(_fps)
  , __vel_multiplier("Velocity Multiplier", 1.0f, 0.1f, 10.0f, 101, SDLK_r, SDLK_f)
  , __magnification("Magnification", 1.0f, 1.0f, 4.0f, 31, SDLK_w, SDLK_s)
  , __offset_x("Offset X-Axis", 0.0f, -1.0f, 1.0f, 21, SDLK_RIGHT, SDLK_LEFT)
  , __offset_y("Offset Y-Axis", 0.0f, -1.0f, 1.0f, 21, SDLK_DOWN, SDLK_UP)
  , __filter_threshold("Filter Threshold", 0.5f, 0.0f, 1.0f, 21, SDLK_t, SDLK_g)
  , __mode("Visualisation Mode", SDLK_1)
{
  __options.push_back(&__vel_multiplier);
  __options.push_back(&__magnification);
  __options.push_back(&__offset_x);
  __options.push_back(&__offset_y);
  __options.push_back(&__filter_threshold);
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

float2 Interface::offset() const {
  return make_float2(__offset_x, __offset_y) * (__magnification - 1.0f);
}

InterfaceRenderer::InterfaceRenderer(Interface const & _interface, IRenderer & _renderer)
  : __interface(_interface)
  , __renderTarget(_renderer.newTextRenderTarget())
{
}

void InterfaceRenderer::__render() {
  std::stringstream os_text;
  os_text.setf(std::ios::fixed, std:: ios::floatfield);
  os_text.precision(2);
  __interface.fps().reportCurrent(os_text);
  os_text << std::endl;
  __interface.mode().reportCurrent(os_text);
  __renderTarget.setText(os_text.str().c_str());
  __renderTarget.render();
}
