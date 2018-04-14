#include "camera.hpp"

#include <iostream>
#include <string>
#include <thread>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

void capture(cv::VideoCapture & _capture, cv::Mat & _mat) {
  while(true) {
    _capture.read(_mat);
  }
}

Camera::Camera(Resolution _res, int _index, float _frame_rate)
  : __capture(_index)
{
  if(__capture.isOpened() == false) {
    std::stringstream error; error << "OpenCV: Cannot open video camera (id:" << _index << ").";
    throwFailure(error.str());
  }
  std::cout << "OpenCV: Video camera found (id:" << _index << ")" << std::endl;
  __capture.set(cv::CAP_PROP_FRAME_WIDTH, _res.width);
  __capture.set(cv::CAP_PROP_FRAME_HEIGHT, _res.height);
  __capture.set(cv::CAP_PROP_FPS, _frame_rate);
  resolution = Resolution(__capture.get(cv::CAP_PROP_FRAME_WIDTH), __capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  resolution.print("\tResolution");
  if(!__capture.read(__input_frame)) {
    std::stringstream error; error << "OpenCV: Could not capture frame.";
    throwFailure(error.str());
  }
  static char const * depth_names[] = {"8U", "8S", "16U", "16S", "32S", "32F", "64F"};
  std::cout << "\tDepth: " << depth_names[__input_frame.depth()] << std::endl;
  std::cout << "\tChannels: " << __input_frame.channels() << std::endl;
  for(int i = 0; i < __input_frame.dims; ++i) {
    std::cout << "\tStep: " << i << ": " << __input_frame.step[i] << std::endl;
  }

  __capThread = std::thread(capture, std::ref(__capture), std::ref(__input_frame));
}
