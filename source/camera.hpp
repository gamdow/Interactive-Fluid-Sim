#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <thread>

#include "component.hpp"
#include "resolution.cuh"

struct Camera : public Component {
  Camera(Resolution _res, int _index, float _frame_rate);
  cv::Mat const & data() {return __input_frame;}
  Resolution resolution;
private:
  cv::VideoCapture __capture;
  cv::Mat __input_frame;
  std::thread __capThread;
};
