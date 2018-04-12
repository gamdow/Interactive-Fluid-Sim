#include "camera.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

Camera::Camera(Resolution _res, int _index)
  : __capture(_index)
{
  if(__capture.isOpened() == false) {
    std::cout << "OpenCV: Cannot open the video camera " << _index << "." << std::endl;
    return;
  }
  std::cout << "OpenCV: Video camera found (id:" << _index << ")" << std::endl;
  __capture.set(cv::CAP_PROP_FRAME_WIDTH, _res.width);
  __capture.set(cv::CAP_PROP_FRAME_HEIGHT, _res.height);
  resolution = Resolution(__capture.get(cv::CAP_PROP_FRAME_WIDTH), __capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  resolution.print("\tResolution");
  if(!__capture.read(__input_frame)) {
    std::cout << "OpenCV: Could not capture frame." << std::endl;
    return;
  }
  static char const * depth_names[] = {"8U", "8S", "16U", "16S", "32S", "32F", "64F"};
  std::cout << "\tDepth: " << depth_names[__input_frame.depth()] << std::endl;
  std::cout << "\tChannels: " << __input_frame.channels() << std::endl;
  for(int i = 0; i < __input_frame.dims; ++i) {
    std::cout << "\tStep: " << i << ": " << __input_frame.step[i] << std::endl;
  }
}
