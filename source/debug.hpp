#pragma once

#include <iostream>

template<typename T>
struct Debug {
  Debug(char const * _string) {
    std::cout << _string << std::endl;
  }  
};
