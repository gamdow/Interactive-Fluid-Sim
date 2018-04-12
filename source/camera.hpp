#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "resolution.cuh"

struct Camera {
  Camera(Resolution _res, int _index);

  void foo() {
    bool success = __capture.read(__input_frame);
  }
  Resolution resolution;
private:
  cv::VideoCapture __capture;
  cv::Mat __input_frame;
};
