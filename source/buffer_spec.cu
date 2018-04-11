#include "buffer_spec.cuh"

BufferSpec::BufferSpec()
  : width(0)
  , height(0)
  , buffer(0)
  , size(0)
{
}

BufferSpec::BufferSpec(int2 _dims, int _buffer)
  : width(_dims.x + 2 * _buffer)
  , height(_dims.y + 2 * _buffer)
  , buffer(_buffer)
  , size(width * height)
{
}
