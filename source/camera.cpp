#include "camera.hpp"

#include <string>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

void capture(cv::VideoCapture & _camera, cv::Mat & _mat) {
  while(true) {
    _camera.read(_mat);
  }
}

CVCamera::CVCamera(int _index, Resolution _res, float _fps)
  : Debug<CVCamera>("OpenCV Camera:")
  , __camera(_index)
{
  if(!__camera.isOpened()) {
    std::stringstream error; error << "\tCannot open video camera (id:" << _index << ")";
    throwFailure(error.str());
  }
  std::cout << "\tVideo camera found (id:" << _index << ")" << std::endl;
  setCameraProps(_res, _fps);
  if(!__camera.read(__frame)) {
    std::stringstream error; error << "\tCould not capture frame";
    throwFailure(error.str());
  }
  reportFrameInfo(std::cout);
  __capture = std::thread(capture, std::ref(__camera), std::ref(__frame));
}

Resolution CVCamera::__resolution() const {
  return Resolution(__camera.get(cv::CAP_PROP_FRAME_WIDTH), __camera.get(cv::CAP_PROP_FRAME_HEIGHT));
}

void CVCamera::setCameraProps(Resolution _res, float _fps) {
  __camera.set(cv::CAP_PROP_FRAME_WIDTH, _res.width);
  __camera.set(cv::CAP_PROP_FRAME_HEIGHT, _res.height);
  __camera.set(cv::CAP_PROP_FPS, _fps);
  resolution().print("\tResolution");
}

void CVCamera::reportFrameInfo(std::ostream & _out) {
  static char const * depth_names[] = {"8U", "8S", "16U", "16S", "32S", "32F", "64F"};
  _out << "\tDepth: " << depth_names[__frame.depth()] << std::endl;
  _out << "\tChannels: " << __frame.channels() << std::endl;
}

NullCamera::NullCamera(Resolution _res)
  : Debug<NullCamera>("Null Camera:")
  , __res(_res)
  , __frame(cv::Mat::zeros(_res.width, _res.height, CV_8UC3))
{
}
