#include "camera.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "capability.cuh"

Camera::Camera(Capability const & _cap, int _index)
  : __capability(_cap)
  , __capture(_index)
{
  if(__capture.isOpened() == false)
  {
    std::cout << "OpenCV: Cannot open the video camera " << _index << "." << std::endl;
    return;
  }
  std::cout << "OpenCV: Video camera " << _index << " found." << std::endl;
  __capture.set(cv::CAP_PROP_FRAME_WIDTH, _cap.original_dims.x);
  __capture.set(cv::CAP_PROP_FRAME_HEIGHT, _cap.original_dims.y);
  double dWidth = __capture.get(cv::CAP_PROP_FRAME_WIDTH);
  double dHeight = __capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  std::cout << "\tResolution: " << dWidth << " x " << dHeight << std::endl;
  if(!__capture.read(__input_frame)) {
    std::cout << "OpenCV: Could not capture frame." << std::endl;
    return;
  }
  std::cout << "\tDepth: " << __input_frame.depth() << std::endl;
  std::cout << "\tChannels: " << __input_frame.channels() << std::endl;
  for(int i = 0; i < __input_frame.dims; ++i) {
    std::cout << "\tStep: " << __input_frame.step[i] << std::endl;
  }
}
