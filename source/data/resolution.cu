#include "resolution.cuh"

#include <iostream>

#include "../debug.hpp"

Resolution::Resolution()
  : width(0)
  , height(0)
  , size(0)
  , buffer(0)
{
}

Resolution::Resolution(Resolution const & _in)
  : width(_in.width)
  , height(_in.height)
  , size(_in.size)
  , buffer(_in.buffer)
{
  size = width * height;
}

Resolution::Resolution(Resolution const & _in, int _buffer)
  : width(_in.width + 2 * (_buffer - _in.buffer))
  , height(_in.height + 2 * (_buffer - _in.buffer))
  , buffer(_buffer)
{
  size = width * height;
}

void Resolution::print(char const * _name) const {
  format_out << _name << ": " << width << " x " << height << std::endl;
}
