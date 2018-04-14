#pragma once

#include <string>
#include <stdexcept>

struct Component {
  static void throwFailure(std::string const & _error) {
    throw std::runtime_error(_error);
  }
};
