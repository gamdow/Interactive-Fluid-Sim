#pragma once

#include <cassert>
#include <iostream>

template<typename T>
struct Debug {
  Debug(char const * _string) {
    std::cout << _string << std::endl;
  }
};

struct OutputFormatter {
  OutputFormatter() : __indent(0) {}
  int __indent;
};

template<typename T>
std::ostream & operator<<(OutputFormatter & output, T const & i) {
   std::cout << std::string(output.__indent * 2, ' ') << i;
   return std::cout;
}

extern OutputFormatter format_out;

struct OutputIndent {
  OutputIndent() {++format_out.__indent;}
  ~OutputIndent() {--format_out.__indent;}
};
