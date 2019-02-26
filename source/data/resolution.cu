#include "resolution.h"


#include "../debug.h"


std::ostream & operator<<(std::ostream & stream, BufferedDimension const & dim) {
  stream << dim.inner;
  if(dim.buffer != 0) {
    stream << " (+ " << dim.buffer << ")";
  }
  return stream;
}

Resolution::Resolution()
  : width(0, 0)
  , height(0, 0)
{
}

Resolution::Resolution(Resolution const & _in, int _width_buffer, int _height_buffer)
  : width(_in.width.inner, _width_buffer)
  , height(_in.height.inner, _height_buffer)
{
}

void Resolution::print(char const * _name) const {
  format_out << _name << ": " << width << " x " << height << std::endl;
}
