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
