#pragma once

enum Mode : int {
  smoke = 0,
  velocity,
  divergence,
  pressure,
  fluid
};

enum FilterMode : int {
  HUE = 0,
  SATURATION,
  LIGHTNESS,
  NUM
};

enum FlowDirection : int {
  LEFT_TO_RIGHT = 0,
  TOP_TO_BOTTOM,
  RIGHT_TO_LEFT,
  BOTTOM_TO_TOP
};
