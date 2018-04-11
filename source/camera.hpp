#include <cuda_runtime.h>

#include <iostream>
#include <opencv2/opencv.hpp>

struct Capability;

struct Camera {
  Camera(Capability const & _cap, int _index);

  void foo() {
    bool success = __capture.read(__input_frame);
  }
private:
  Capability const & __capability;
  cv::VideoCapture __capture;
  cv::Mat __input_frame;
};
